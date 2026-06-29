"""Precompute the synthetic OSSE measurements y = F(truth) — the "synthetic L1B".

In a real retrieval the radiances come FROM the instrument at a fixed dimension; only
the OSSE manufactures them by running the forward at the truth's native in-situ
resolution (a per-profile shape, 14..111 nodes -> 62 distinct XLA compiles). That is
the single worst obstacle to compile caching AND a pure OSSE artifact. So we compute
y = F(truth) ONCE here (exact, native resolution, the converged per-band NFourier),
write it to a cache, and every downstream worker (IC, retrieval) just LOADS it — losing
the per-profile-shape compile axis entirely. The cache embeds the observing-system
signature; osse_config.load_radiance asserts it matches before use.

Usage:
  generate_osse_radiances.py <index> <out_dir>      # one profile (HPC array task)
  generate_osse_radiances.py consolidate <out_dir> <out.npz>   # merge sidecars

Env:  VOCALS_DATA, OPTICS_CACHE, PYDISORT_RICCATI_JAX_X64=1.
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1] / "src"))
sys.path.insert(0, str(_here))
import runtime_setup               # noqa: E402
runtime_setup.setup()             # affinity pin (+ optional compile cache) BEFORE jax
import vocals_io as vio            # noqa: E402
import retrieval_oe as roe         # noqa: E402
import osse_config as oc           # noqa: E402

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_10band_nleg1536_re20.npz'))
TAU_BOT_OK = (0.3, 100.0)
FIELDS = ("y", "tau", "re", "r_base", "tau_bot", "lwc", "altitude", "flight")


def generate_one(index, out_dir):
    import jax.numpy as jnp
    if jnp.result_type(float) != jnp.float64:
        print("WARNING: float32 — set PYDISORT_RICCATI_JAX_X64=1", flush=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profiles = vio.load_all_profiles(DATA)
    truth = profiles[index]
    flight = getattr(truth, 'flight', '?')
    sig_payload, sig_hash = oc.signature()

    rec = dict(index=index, flight=flight, signature_hash=sig_hash)
    try:
        if not (TAU_BOT_OK[0] <= float(truth.tau_bot) <= TAU_BOT_OK[1]) \
                or len(np.asarray(truth.tau)) < 5:
            raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f})")
        opt = oc.load_optics(OPTICS_CACHE)
        # anchors are irrelevant to y (osse_observation overrides tau_bot/r_base from
        # the truth state); use the truth's own as harmless valid values. views='full'
        # → the 32-view superset, so the ONE cache serves both the IC (all 32) and the
        # retrieval (its 24-subset, via osse_config.select_retrieval_views).
        # jac_mode='rev' (default adjoint): radiance generation is a pure forward — no
        # derivatives needed — and the adjoint choice does not affect the primal y.
        # tol flows from osse_config.SOLVER_TOL (env SOLVER_TOL); it is an accuracy TAG on the
        # cache (sidecar 'tol' below), NOT part of the signature — see osse_config.SOLVER_TOL.
        # MODE_MAP='vmap' routes the 240-way bands×modes GPU path (re-gen / canary).
        fwd = oc.build_forward(opt, tau_bot=float(truth.tau_bot), r_base=float(truth.r_base),
                               views='full', state_space='log', jac_mode='rev',
                               mode_map=os.environ.get('MODE_MAP', 'scan'))
        t0 = time.time()
        # EXACT synthetic measurement: y = F(truth) at native in-situ resolution, full
        # per-band modes (K_list defaults to the per-band NFourier ceiling).
        y = roe.osse_observation(fwd, truth.tau, truth.r_e)
        dt = time.time() - t0
        sidecar = dict(
            index=int(index), flight=str(flight), signature_hash=sig_hash,
            signature_json=json.dumps(sig_payload), tol=float(oc.SOLVER_TOL),
            y=np.asarray(y, float),
            tau=np.asarray(truth.tau, float), re=np.asarray(truth.r_e, float),
            r_base=float(truth.r_base), tau_bot=float(truth.tau_bot),
            lwc=np.asarray(truth.lwc, float), altitude=np.asarray(truth.altitude, float))
        np.savez(out_dir / f"osse_{int(index)}.npz", **sidecar)
        rec.update(m=int(np.size(y)), n_native=int(np.asarray(truth.tau).size),
                   runtime_s=round(dt, 1))
        print(f"[{index}] {flight} tau={truth.tau_bot:.1f}: y={np.size(y)} "
              f"(native {np.asarray(truth.tau).size} nodes) in {dt:.0f}s | sig={sig_hash}",
              flush=True)
    except Exception as e:                                          # noqa: BLE001
        rec["skipped"] = str(e)[:200]
        print(f"[{index}] {flight}: SKIPPED {rec['skipped']}", flush=True)
    (out_dir / f"osse_{int(index)}.json").write_text(json.dumps(rec))


def consolidate(out_dir, out_npz):
    """Merge per-index sidecars into one signed radiance cache (osse_config.load_radiance
    reads this). Asserts every sidecar shares the current observing-system signature."""
    out_dir = Path(out_dir)
    _, want = oc.signature()
    merged, present, skipped, tols = {"signature_hash": want}, [], [], set()
    for f in sorted(out_dir.glob("osse_*.npz")):
        d = np.load(f, allow_pickle=True)
        idx = int(d["index"])
        if "y" not in d:                                # a skipped/degenerate profile
            skipped.append(idx); continue
        got = str(d["signature_hash"])
        if got != want:
            raise ValueError(f"{f.name}: signature {got} != current {want}; regenerate.")
        tols.add(round(float(d["tol"]), 12) if "tol" in d else None)  # accuracy tag
        for k in FIELDS:
            merged[f"{idx}_{k}"] = d[k]
        present.append(idx)
    if len(tols) > 1:                                   # never mix tolerances in one cache
        raise ValueError(f"sidecars span multiple tol {sorted(tols)}; one tol per cache.")
    merged["tol"] = tols.pop() if tols else None        # the cache's accuracy tag
    np.savez(Path(out_npz), **merged)
    print(f"consolidated {len(present)} profiles -> {out_npz} (sig {want}, "
          f"tol={merged['tol']}); skipped {sorted(skipped)}", flush=True)


def main():
    if sys.argv[1] == "consolidate":
        consolidate(sys.argv[2], sys.argv[3])
    else:
        generate_one(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
