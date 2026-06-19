"""First-guess SIC(k) + DOFS(k) sweep — validates the SIC-peak node-count criterion.

Both DOFS=tr(A) and SIC=½log2|Sa S_hat^-1| are LINEARIZED functionals of (K, Sa, Se) at a
single point, so they need NO Gauss-Newton retrieval and NO ground truth. This evaluates them
at the FIRST GUESS (prior mean) on each candidate k-grid and reports the curves, to confirm the
first-guess SIC peak lands at the GN-sweep RMSE-optimal k (thin~4, RF03=3, RF08=4). If it does,
SIC-peak selection is operationally viable (cost = a few first-guess Jacobian evals, no GN).

    /tmp/jaxve/bin/python tests/supplementary/fg_sic_dofs_sweep.py
"""
import sys
import json
from pathlib import Path
from math import pi

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQuad, NLeg_all, v_eff = 16, 128, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.linspace(0.95, 0.25, 8)
view_phi = np.full(view_mu.size, pi)

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


def build_case(truth, bands):
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    opt = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
           for i in range(len(bands))]
    prior_builder = (lambda sn: roe.make_marine_sc_prior(
        sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean']))
    fwd = roe.RetrievalForward(
        opt, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=clim['tau_bot_mean'], r_base=clim['r_base_mean'],
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[0.06]] * len(bands),
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
    s_ref = np.linspace(0.0, 1.0, 6)[:-1]
    x_ref, _ = prior_builder(s_ref)
    roe.select_num_modes(fwd, x_ref, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    return prior_builder, fwd, s_ref, x_ref, Se


CASES = [
    ("THIN ", vio.pick_profile(profiles, target_tau=1.0), [1.24, 2.13], [2, 3, 4, 5], 4),
    ("RF03 ", vio.pick_profile([p for p in profiles if p.flight == 'RF03'], 23.3),
     [1.24, 1.64, 2.13], [3, 4, 5, 6], 3),
    ("RF08 ", vio.pick_profile([p for p in profiles if p.flight == 'RF08'], 26.3),
     [1.24, 1.64, 2.13], [3, 4, 5, 6], 4),
]

out = []
for label, truth, bands, ks, rmse_opt_k in CASES:
    prior_builder, fwd, s_ref, x_ref, Se = build_case(truth, bands)
    print(f"\n{label} {truth.flight} tau_bot={truth.tau_bot:.1f}  (GN-sweep RMSE-optimal k={rmse_opt_k})",
          flush=True)
    rows = []
    for k in ks:
        s_grid, _, _ = roe.select_retrieval_grid(fwd, x_ref, s_ref, int(k))
        x0, Sa = prior_builder(s_grid)                       # first guess = prior mean
        K = np.asarray(fwd.jacobian(x0, s_grid))             # first-guess JOINT Jacobian (no GN)
        post = roe.posterior_diagnostics(K, Sa, Se)
        rows.append(dict(k=int(k), dofs=float(post.dofs), sic=float(post.sic)))
        print(f"  k={k}: DOFS {post.dofs:.2f}   SIC {post.sic:.2f}", flush=True)
    sic_peak_k = max(rows, key=lambda r: r['sic'])['k']
    ok = "MATCH" if sic_peak_k == rmse_opt_k else "MISMATCH"
    print(f"  -> first-guess SIC peak at k={sic_peak_k}; RMSE-optimal k={rmse_opt_k}  [{ok}]",
          flush=True)
    out.append(dict(label=label.strip(), flight=truth.flight, rmse_opt_k=rmse_opt_k,
                    sic_peak_k=int(sic_peak_k), match=(sic_peak_k == rmse_opt_k), rows=rows))

p = Path(__file__).resolve().parents[2] / "docs" / "fg_sic_dofs_sweep.json"
p.write_text(json.dumps(out, indent=2))
print(f"\nwrote {p}", flush=True)
