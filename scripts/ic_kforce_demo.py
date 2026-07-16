"""DOFS k-saturation + linearization-point demo on RETRIEVAL-STYLE grids (2026-07-15).

The retrieval-grid IC reframing (user-approved) computes information content on the
QRCP-selected operational grids (p = k+2 small -> texture-converged at production tol,
per the 2026-07-14/15 row-count law). Its sharpest attack: "DOFS <= p by construction,
and QRCP chose k with the same noise/prior — is DOFS ~4.4 a physics number or a grid
echo?" This demo answers it: force k well past QRCP's choice (k = 4..16) and show DOFS
PLATEAUS instead of tracking k. Also runs BOTH linearization points (user nuance A):
  'prior'     — climatological prior-mean r_e, tau_bot at the RETRIEVED value
                (the "prior after tau_bot retrieval" state)
  'retrieved' — the canonical retrieval x_hat, interpolated onto each forced grid.
Each (lin, k): QRCP-select the grid at that state (k_active=k), rebuild the LOO prior
on it, K = fwd.jacobian (state-space log, operational 24-view fan, sigma from the
canonical sidecar), DOFS/SIC via posterior_diagnostics (pure OE prior — no curvature).

Run at SOLVER_TOL=1e-4 AND 1e-6 (separate out files): the tol-agreement of every
(lin, k) DOFS is the row-count-law verification gate for the whole reframing.

Usage: ic_kforce_demo.py <profile_index> <out.json>
Env:   campaign env (OSSE_VEFF/OSSE_RE_MAX/OPTICS_CACHE/... as the ve046 fq campaign),
       SOLVER_TOL, CANON_DIR (default runs/_ve046_tik_fr_parts), MODE_MAP.
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
from pydisort_riccati_jax import osse_config as oc                   # noqa: E402

CANON = Path(os.environ.get("CANON_DIR", "runs/_ve046_tik_fr_parts"))
K_LIST = (4, 6, 8, 12, 16)
S_REF = np.linspace(0.0, 1.0, 6)[:-1]

idx, out = int(sys.argv[1]), sys.argv[2]
profiles = vio.load_all_profiles(oc.VOCALS_DATA)
truth = profiles[idx]
clim = vio.vocals_climatology(profiles, exclude_flight=getattr(truth, "flight", "?"))

side = dict(np.load(CANON / f"{idx}_A.npz", allow_pickle=True))
tau_bot_ret = float(side["tau_bot_ret"])
r_base_ret = float(side["r_base_ret"])
sigma = np.asarray(side["sigma"], float)              # operational 10x24 noise vector
x_hat_log = np.asarray(side["x_hat_log"], float)
s_grid_hat = np.asarray(side["s_grid"], float)        # final retrieval grid of the record

opt = oc.load_optics(oc.OPTICS_CACHE)
fwd = oc.build_forward(opt, tau_bot=float(clim["tau_bot_mean"]), r_base=float(clim["r_base_mean"]),
                       views="retrieval", jac_mode="fwd",
                       mode_map=os.environ.get("MODE_MAP", "scan"))
Se = np.diag(sigma ** 2)

def prior_builder(s_nodes):
    return roe.make_climatology_prior(np.asarray(s_nodes), clim, log=True)

def encode_on(s_nodes, re_of_s, tau_bot, r_base):
    """Physical (r_e nodes, r_base, tau_bot) -> log state on s_nodes."""
    re_vals = np.asarray(re_of_s(np.asarray(s_nodes)), float)
    return fwd._encode_state(np.concatenate([re_vals, [r_base, tau_bot]]))

# linearization states as r_e(s) callables (normalized depth)
xa_ref, _ = prior_builder(S_REF)
re_prior_ref = np.exp(np.asarray(xa_ref[:len(S_REF)]))               # log prior -> physical
def re_prior(s):
    return np.interp(np.asarray(s), S_REF, re_prior_ref)
def re_retrieved(s):
    return np.asarray(fwd.profile(x_hat_log, s_grid_hat, np.asarray(s) * tau_bot_ret))

rec = dict(index=idx, tol=float(os.environ.get("SOLVER_TOL", "1e-3")),
           sidecar=dict(k=int(side["k"]), dofs=float(side["dofs"]), sic=float(side["sic"])),
           runs=[])
t0 = time.time()
for lin, re_fn in (("prior", re_prior), ("retrieved", re_retrieved)):
    x_ref = encode_on(S_REF, re_fn, tau_bot_ret, r_base_ret)
    for k in K_LIST:
        s_nodes = np.asarray(roe.select_retrieval_grid(
            fwd, x_ref, S_REF, k_active=k, Se=Se,
            prior_builder=prior_builder, k_max=max(K_LIST))[0])
        x_a, Sa = prior_builder(s_nodes)
        x_lin = encode_on(s_nodes, re_fn, tau_bot_ret, r_base_ret)
        K = np.asarray(fwd.jacobian(x_lin, s_nodes))
        post = roe.posterior_diagnostics(K, Sa, Se)
        n_prof = len(s_nodes)
        entry = dict(lin=lin, k=int(k), k_got=int(n_prof), p=int(K.shape[1]),
                     dofs=float(post.dofs), sic=float(post.sic),
                     dofs_profile=float(np.trace(np.asarray(post.A)[:n_prof, :n_prof])),
                     s_nodes=[round(float(s), 4) for s in s_nodes])
        rec["runs"].append(entry)
        Path(out).write_text(json.dumps(rec))
        print(f"[{idx}] lin={lin} k={k} (got {n_prof}) p={K.shape[1]} "
              f"DOFS={entry['dofs']:.3f} (profile {entry['dofs_profile']:.3f}) "
              f"SIC={entry['sic']:.2f} [{time.time()-t0:.0f}s]", flush=True)

print("saved", out)
