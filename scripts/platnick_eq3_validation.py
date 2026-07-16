"""Platnick (2000) Eq-(3) alignment validation of the Fig-0a weighting functions.

Pla2000 defines the depth-penetration weighting OPERATIONALLY (his Eq. 3): the
homogeneous-equivalent retrieved radius must satisfy r_e* = ∫ r_e(τ) w(τ) dτ, and he
validates his photon-transport kernels (w_m, w_N) against actual retrievals to
~0.1-0.3 µm (his Tables 3a/3b). This script runs the SAME test on our derivative
kernels: for the smooth prior-mean linearization profile of each probe index
(the profile the cached wiggle Jacobians are linearized at — analogous to his smooth
analytic profiles), compute per (band, view)

    r*_hom     : the actual homogeneous-equivalent radius — R_hom(r*_hom) matches the
                 profile-cloud radiance, by sweep + monotone interpolation
                 (his "retrieval code" column),
    r*_signed  : Σ_j K_j r_e(s_j) / Σ_j K_j  — the exact first-order Eq-(3) kernel,
    r*_abs     : the same with |K| (what Fig 0a currently plots).

PASS = r*_signed reproduces r*_hom at Pla2000's ~0.1-0.3 µm level on the absorbing
bands; the r*_abs column quantifies the |K| bias (expected worst at 3.7/4.05 µm where
~1/3 of nodes flip sign). Compute phase writes <out>/eq3_<idx>.npz (this script, GPU);
the analysis table is a trivial post-hoc read.

Usage: platnick_eq3_validation.py <idx> <out_dir>
Env:   identical to ic_worker_wiggle.py 'mie' cells (v_e=0.10/re20 defaults,
       ENSEMBLE_NQUAD=48, SOLVER_TOL=1e-4, OPTICS_CACHE=re20 production table).
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
RE_SWEEP = np.linspace(3.0, 18.0, 16)                # homogeneous r_e sweep (µm)
s_ref = np.linspace(0.0, 1.0, 6)[:-1]                # = ic_worker_wiggle linearization grid

idx, out_dir = int(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
wig = np.load(f"runs/_wiggle_mie_parts/{idx}.npz", allow_pickle=True)
s_grid = np.asarray(wig["s_grid"], float)
re_lin = np.asarray(wig["re_grid"], float)           # the linearization profile r_e(s)
tau_bot = float(wig["tau_bot"])

profiles = vio.load_all_profiles(oc.VOCALS_DATA)
truth = profiles[idx]
clim = vio.vocals_climatology(profiles, exclude_flight=getattr(truth, "flight", "?"))
opt = oc.load_optics(oc.OPTICS_CACHE)
fwd = oc.build_forward(opt, tau_bot=tau_bot, r_base=float(truth.r_base),
                       views="full", jac_mode="fwd",
                       mode_map=os.environ.get("MODE_MAP", "scan"))
# identical linearization + mode selection to the cached Jacobians
xa_sref = np.asarray(roe.make_climatology_prior(s_ref, clim)[0])
x_lin = fwd._encode_state(np.concatenate([xa_sref[:len(s_ref) + 1], [tau_bot]]))
y_probe = np.asarray(roe.osse_observation(fwd, truth.tau, truth.r_e))
roe.select_num_modes(fwd, x_lin, s_ref, np.diag(nm.oci_swir().sigma(y_probe) ** 2))

t0 = time.time()
# radiances of the LINEARIZATION profile itself (the "vertically structured cloud")
y_lin = np.asarray(roe.osse_observation(fwd, s_grid * tau_bot, re_lin))
print(f"[{idx}] linearization-profile radiances done [{time.time()-t0:.0f}s]", flush=True)
# homogeneous sweep at the same tau_bot
y_hom = np.empty((len(RE_SWEEP), y_lin.size))
for i, c in enumerate(RE_SWEEP):
    y_hom[i] = np.asarray(roe.osse_observation(
        fwd, np.array([0.0, tau_bot]), np.array([c, c])))
    print(f"[{idx}] hom sweep {i+1}/{len(RE_SWEEP)} (r_e={c:.1f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

np.savez_compressed(out_dir / f"eq3_{idx}.npz",
                    index=idx, tau_bot=tau_bot, re_sweep=RE_SWEEP,
                    y_lin=y_lin, y_hom=y_hom, s_grid=s_grid, re_lin=re_lin,
                    bands=np.asarray(oc.BANDS), view_mu=np.asarray(oc.VIEW_MU_FULL),
                    K_list=np.asarray(fwd.K_list))
Path(out_dir / f"eq3_{idx}.json").write_text(json.dumps(dict(
    index=idx, tau_bot=tau_bot, runtime_s=round(time.time() - t0, 1),
    n_sweep=len(RE_SWEEP))))
print(f"[{idx}] DONE [{time.time()-t0:.0f}s] -> eq3_{idx}.npz", flush=True)
