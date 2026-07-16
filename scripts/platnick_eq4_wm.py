"""Platnick (2000) Eq-(4) maximum-penetration weighting w_m from the PRODUCTION solver.

w_m(tau; band, view) = (1/R(tau_c)) * dR(tau)/dtau, built exactly as Pla2000 §4.1 (and
the user's retrieval_plots.ipynb): sweep cloud thickness by TRUNCATING the profile at
tau_cut (layers added at the base), one forward per truncation, finite-difference over
the truncation grid. Differences vs the user's notebook: the real per-band
omega(lambda, r_e(tau)) and phase function from the production optics table (not a
fixed near-conservative omega), all 10 bands x 32 views per solve, and the same
linearization profile as the cached Eq-(3) kernels — so w_m and the sensitivity kernel
are directly comparable per (profile, band, view). Surface is the production Lambertian
0.06 (near-black at SWIR; Pla2000's w_m assumes black — noted, not corrected).

Smooth by construction (d/dtau of a cumulative reflectance) — the Fig-0a wiggle lives
only in the d/dr_e channel and cannot appear here.

Usage: platnick_eq4_wm.py <idx> <out_dir>     Env: as platnick_eq3_validation.py.
"""
import sys
import os
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pydisort_riccati_jax import runtime_setup                       # noqa: E402
runtime_setup.setup()
from pydisort_riccati_jax import vocals_io as vio                    # noqa: E402
from pydisort_riccati_jax import retrieval_oe as roe                 # noqa: E402
from pydisort_riccati_jax import noise_model as nm                   # noqa: E402
from pydisort_riccati_jax import osse_config as oc                   # noqa: E402

NQ = int(os.environ.get("ENSEMBLE_NQUAD", "48"))
N_TRUNC = int(os.environ.get("N_TRUNC", "40"))       # thickness-sweep points
N_TAUPTS = 51                                        # FIXED per-solve grid length (one compile)
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

idx, out_dir = int(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
wig = np.load(f"runs/_wiggle_mie_parts/{idx}.npz", allow_pickle=True)
s_grid = np.asarray(wig["s_grid"], float)
re_lin = np.asarray(wig["re_grid"], float)           # linearization profile r_e(s)
tau_bot = float(wig["tau_bot"])

profiles = vio.load_all_profiles(oc.VOCALS_DATA)
truth = profiles[idx]
clim = vio.vocals_climatology(profiles, exclude_flight=getattr(truth, "flight", "?"))
opt = oc.load_optics(oc.OPTICS_CACHE)
fwd = oc.build_forward(opt, tau_bot=tau_bot, r_base=float(truth.r_base),
                       views="full", jac_mode="fwd",
                       mode_map=os.environ.get("MODE_MAP", "scan"))
xa_sref = np.asarray(roe.make_climatology_prior(s_ref, clim)[0])
x_lin = fwd._encode_state(np.concatenate([xa_sref[:len(s_ref) + 1], [tau_bot]]))
y_probe = np.asarray(roe.osse_observation(fwd, truth.tau, truth.r_e))
roe.select_num_modes(fwd, x_lin, s_ref, np.diag(nm.oci_swir().sigma(y_probe) ** 2))

# quadratic-spaced truncation edges (denser near the top, like the user's notebook)
tau_cuts = np.linspace(np.sqrt(tau_bot / N_TRUNC), np.sqrt(tau_bot), N_TRUNC) ** 2
t0 = time.time()
R = np.empty((N_TRUNC, y_probe.size))
for i, tc in enumerate(tau_cuts):
    tau_pts = np.linspace(0.0, tc, N_TAUPTS)         # fixed length -> single compile
    re_pts = np.interp(tau_pts, s_grid * tau_bot, re_lin)
    R[i] = np.asarray(roe.osse_observation(fwd, tau_pts, re_pts))
    print(f"[{idx}] truncation {i+1}/{N_TRUNC} (tau_cut={tc:.2f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

np.savez_compressed(out_dir / f"eq4_{idx}.npz",
                    index=idx, tau_bot=tau_bot, tau_cuts=tau_cuts, R=R,
                    s_grid=s_grid, re_lin=re_lin,
                    bands=np.asarray(oc.BANDS), view_mu=np.asarray(oc.VIEW_MU_FULL),
                    K_list=np.asarray(fwd.K_list))
Path(out_dir / f"eq4_{idx}.json").write_text(json.dumps(dict(
    index=idx, tau_bot=tau_bot, n_trunc=N_TRUNC,
    runtime_s=round(time.time() - t0, 1))))
print(f"[{idx}] DONE [{time.time()-t0:.0f}s] -> eq4_{idx}.npz", flush=True)
