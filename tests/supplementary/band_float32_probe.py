"""Pinpoint which band(s) make the dense-truth forward float32-fragile (DESIGN §15).

The 10-band retrieval forward hits the adaptive solver's `max_steps` at float32 on
the dense in-situ truth, whereas the notebook's 2-band `[1.24,2.13]` float32
retrievals and the IC run's 10-band *float64* forwards are fine. Because the forward
solves each band as an **independent** Riccati integration
(`RetrievalForward._band_reflectance` → one `riccati_solve` per band; no roundoff
shared across bands), one hypothesis was a band-*type* effect: a specific band whose
optics make its float32 solve stiff. This probe runs each band's dense-truth forward
**alone, at float32, NQuad=48** and reports PASS / MAX_STEPS.

**Result (idx 105): all 10 forwards PASS — so it is NOT band type and NOT band
count.** The float32 `max_steps` failure is instead in the grid-selection pool
Jacobian (`jacobian_on_grid` — a ~30-column forward-mode AD through the adaptive
solver, whose augmented tangent system the float32 rtol-floor does not cover); see
DESIGN §15. Kept as the committed evidence that ruled out the band hypotheses.

Run (float32 — do NOT set X64):  python band_float32_probe.py [profile_index]
"""
import os
import sys
import json
import time
import warnings
from math import pi
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1] / "src"))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import optics_table as ot                                          # noqa: E402
import jax.numpy as _jnp                                            # noqa: E402

if _jnp.result_type(float) != _jnp.float32:
    print("NOTE: this probe is meant to run at FLOAT32 (unset PYDISORT_RICCATI_JAX_X64) "
          "to reproduce the failure; currently float64.", flush=True)

DATA = os.environ.get('VOCALS_DATA', '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_10band.npz'))
NQ = int(os.environ.get('ENSEMBLE_NQUAD', '48'))
# suspicion order: strongly-absorbing MWIR first, then short-λ VIS, then the rest,
# with the notebook's known-safe pair [1.24, 2.13] last as controls.
BANDS = [3.7, 4.05, 0.55, 0.67, 0.86, 1.038, 1.64, 2.26, 1.24, 2.13]
mu0, NLeg_all, v_eff = 0.9, 128, 0.10
VIEW_MU, VIEW_PHI = np.array([0.9, 0.6, 0.4]), np.full(3, pi)


def probe_band(lam, truth):
    table = ot.build_or_load_table([lam], 2.0, 25.0, 32, v_eff,
                                   cache_path=_here / f"_probe_optics_{lam}.npz",
                                   NLeg=NLeg_all, n_radii=600)
    opt = [ot.select_channel(table, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = roe.RetrievalForward(
            opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
            tau_bot=truth.tau_bot, r_base=truth.r_base, view_mu=VIEW_MU, view_phi=VIEW_PHI,
            BDRF_bands=[[0.06]], NLeg_all=NLeg_all, retrieve_tau_bot=True,
            retrieve_r_base=True, jac_mode='fwd')
    t0 = time.time()
    try:
        y = roe.osse_observation(fwd, truth.tau, truth.r_e)        # the dense-truth forward
        return dict(band=lam, status="PASS", dt=round(time.time() - t0, 1),
                    y0=float(np.asarray(y)[0]))
    except Exception as e:                                          # noqa: BLE001
        msg = str(e)
        kind = "MAX_STEPS" if "max" in msg.lower() and "step" in msg.lower() else "ERROR"
        return dict(band=lam, status=kind, dt=round(time.time() - t0, 1), err=msg[:80])


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 105
    truth = vio.load_all_profiles(DATA)[idx]
    print(f"probe: idx {idx} {truth.flight} tau_bot={truth.tau_bot:.2f} npts={len(truth.tau)} "
          f"@ NQuad={NQ} float32={_jnp.result_type(float)==_jnp.float32}\n", flush=True)
    rows = []
    for lam in BANDS:
        r = probe_band(lam, truth)
        rows.append(r)
        flag = "  <-- FRAGILE" if r["status"] != "PASS" else ""
        print(f"  {lam:>6} µm : {r['status']:<9} [{r['dt']:>5.1f}s]{flag}", flush=True)
    frag = [r["band"] for r in rows if r["status"] != "PASS"]
    print(f"\nFRAGILE at float32: {frag}   (safe: {[r['band'] for r in rows if r['status']=='PASS']})")
    out = _here.parents[1] / "docs" / "cached_results" / "band_float32_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(index=idx, flight=truth.flight, NQuad=NQ,
                                   tau_bot=float(truth.tau_bot), rows=rows, fragile=frag), indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
