"""Operational precision/tol viability for the retrieval — ONE profile, on GPU.

Accuracy tiers: the OSSE truth (radiances) + IC are high-accuracy (f64, tol*); the
RETRIEVAL is allowed to run at OPERATIONAL precision/tol (cheaper, a known bias, but
realistic). This probe inverts a FIXED gold observation y=F(truth) [f64/tol=1e-5] with
an operational-precision forward, isolating the operational retrieval bias against the
true measurement. 'Higher error is fine; de-stabilization (NaN / max_steps / non-conv)
is not.'

Built on osse_config (NLeg_all=1024, the correct irregular 24-view fan, the nleg1024
table) — NOT on retrieval_worker, whose NLeg_all=128 is the pre-TMS-fix value (would
corrupt the short bands). gauss_newton_oe hyperparameters mirror retrieval_worker.

  retrieval_precision_probe.py gold <idx>
      env: PYDISORT_RICCATI_JAX_X64=1  SOLVER_TOL=1e-5  [MODE_MAP=vmap on GPU]
      Build forward+obs+grid+prior at GOLD; save gold_<idx>.npz. Run ONCE per profile.

  retrieval_precision_probe.py retrieve <idx>
      env: PYDISORT_RICCATI_JAX_X64=0|1  SOLVER_TOL=<op tol>  MODE_MAP=scan|vmap
      Load gold, rebuild forward at (precision, tol) on the FIXED grid+modes, run GN,
      save retrieved x̂ + RMSE-vs-truth + stability flags + provenance.

Provenance is in EVERY output (precision, tol, mode_map, node, gold_tol) and the filename
(probe_<idx>_<precision>_tol<..>_<mode_map>.{npz,json}) — nothing is mixable.
"""
import os
import sys
import json
import time
import platform
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parents[1] / "src"))
import jax.numpy as jnp                                            # noqa: E402
import osse_config as oc                                          # noqa: E402
import retrieval_oe as roe                                        # noqa: E402
import vocals_io as vio                                           # noqa: E402
import noise_model as nm                                          # noqa: E402

PREC = "float64" if jnp.result_type(float) == jnp.float64 else "float32"
TOL = float(os.environ.get("SOLVER_TOL", "1e-3"))
MODE_MAP = os.environ.get("MODE_MAP", "scan")
DATA = os.environ.get("VOCALS_DATA",
                      "/home/jovyan/cloud_profile_retrieval/"
                      "multispectral-retrieval-using-MODIS/VOCALS_REx_data")
OPTICS_CACHE = os.environ["OPTICS_CACHE"]
GOLD_DIR = Path(os.environ.get("GOLD_DIR", _here / "precision_probe_gold"))
OUT_DIR = Path(os.environ.get("PROBE_OUT", _here / "precision_probe_out"))
COST_RTOL = float(os.environ.get("COST_RTOL", "0.01"))
NOISE = nm.oci_swir()
S_REF_MODES = np.linspace(0.0, 1.0, 5)[:-1]
S_COARSE = np.linspace(0.0, 1.0, 6)[:-1]
S_DENSE = np.linspace(0.0, 1.0, 50)


def build_fwd(clim, tol):
    """osse_config forward at a given ODE tol + the env precision/mode_map (24 views)."""
    opt = oc.load_optics(OPTICS_CACHE)
    return oc.build_forward(opt, tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
                            views="retrieval", state_space="log", jac_mode="fwd",
                            tol=tol, mode_map=MODE_MAP)


def main():
    mode, idx = sys.argv[1], int(sys.argv[2])
    profiles = vio.load_all_profiles(DATA)
    truth = profiles[idx]
    flight = getattr(truth, "flight", "?")
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)

    if mode == "gold":
        assert PREC == "float64" and TOL <= 1e-5, "gold must be float64 + SOLVER_TOL<=1e-5"
        fwd = build_fwd(clim, TOL)
        y = np.asarray(roe.osse_observation(fwd, truth.tau, truth.r_e), float)
        Se = np.asarray(roe.make_Se(fwd, y, NOISE), float)
        x_ref = fwd._encode_state(roe.make_climatology_prior(S_REF_MODES, clim)[0])
        roe.select_num_modes(fwd, x_ref, S_REF_MODES, Se)
        x_fg = fwd._encode_state(roe.make_climatology_prior(S_COARSE, clim)[0])
        pb_phys = lambda sn: roe.make_climatology_prior(sn, clim)            # noqa: E731
        s_grid, _, _ = roe.select_retrieval_grid(fwd, x_fg, S_COARSE, k_active=None,
                                                 Se=Se, prior_builder=pb_phys)
        x_a_log, Sa_log = roe.make_climatology_prior(s_grid, clim, log=True)
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(GOLD_DIR / f"gold_{idx}.npz", y=y, Se=Se, s_grid=np.asarray(s_grid),
                 K_list=np.asarray(fwd.K_list, int), x_a_log=np.asarray(x_a_log),
                 Sa_log=np.asarray(Sa_log), gold_tol=TOL, flight=str(flight))
        print(f"[gold {idx}] {flight} y.sum={y.sum():.6f} n_neg={(y < 0).sum()} "
              f"k={len(s_grid)} K={fwd.K_list} -> gold_{idx}.npz", flush=True)
        return

    # mode == retrieve
    g = np.load(GOLD_DIR / f"gold_{idx}.npz", allow_pickle=True)
    y, Se, s_grid = (np.asarray(g[k], float) for k in ("y", "Se", "s_grid"))
    x_a, Sa = np.asarray(g["x_a_log"], float), np.asarray(g["Sa_log"], float)
    fwd = build_fwd(clim, TOL)
    fwd.K_list = [int(k) for k in g["K_list"]]                    # FIX modes to gold's
    pb_log = lambda sn: roe.make_climatology_prior(sn, clim, log=True)   # noqa: E731
    tag = f"{PREC}_tol{TOL:.0e}_{MODE_MAP}"
    rec = dict(index=idx, flight=str(flight), precision=PREC, tol=TOL, mode_map=MODE_MAP,
               node=platform.node(), gold_tol=float(g["gold_tol"]),
               gold_signature="f64_tol1e-5_native")
    t = time.time()
    try:
        res = roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, x0=x_a, n_iter=12, lm=1e-2,
                                  xtol=2e-3, cost_rtol=COST_RTOL, max_n_outer=1,
                                  prior_builder=pb_log, remesh_if_chi2_red_gt=2.0, warn=False)
        post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
        r_nodes, r_base_ret, tau_bot_ret = (np.asarray(v, float)
                                            for v in fwd._split_state(res.x, s_grid))
        tau_bot_ret = float(tau_bot_ret)
        re_ours = np.asarray(fwd.profile(res.x, s_grid, S_DENSE * tau_bot_ret), float)
        s_t = np.asarray(truth.tau, float) / truth.tau_bot
        o = np.argsort(s_t)
        re_truth = np.interp(S_DENSE, s_t[o], np.asarray(truth.r_e, float)[o])
        rmse = float(np.sqrt(np.mean((re_ours - re_truth) ** 2)))
        adia = roe.best_fit_adiabatic(S_DENSE, re_truth, truth.tau_bot, metric="rmse")
        finite = bool(np.all(np.isfinite(res.x)) and np.all(np.isfinite(res.K)))
        rec.update(ran=True, finite=finite, converged=bool(res.converged),
                   n_gn=len(res.cost_history), rmse_ours=round(rmse, 4),
                   rmse_adia=round(float(adia["rmse"]), 4),
                   d_rmse=round(float(adia["rmse"]) - rmse, 4),
                   dofs=round(float(post.dofs), 3), tau_bot_ret=round(tau_bot_ret, 3),
                   runtime_s=round(time.time() - t, 1))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(OUT_DIR / f"probe_{idx}_{tag}.npz", x_hat_log=np.asarray(res.x),
                 re_ours_dense=re_ours, re_truth_dense=re_truth, s_dense=S_DENSE,
                 s_grid=s_grid, K_list=np.asarray(fwd.K_list, int),
                 index=idx, precision=PREC, tol=TOL, mode_map=MODE_MAP, gold_tol=rec["gold_tol"])
    except Exception as e:                                        # noqa: BLE001
        rec.update(ran=False, finite=False,
                   error=f"{type(e).__name__}: {str(e)[:200]}", runtime_s=round(time.time() - t, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / f"probe_{idx}_{tag}.json").write_text(json.dumps(rec))
    print(f"[probe {idx} {tag}] " + json.dumps({k: rec.get(k) for k in
          ("ran", "finite", "converged", "rmse_ours", "d_rmse", "dofs", "n_gn",
           "runtime_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
