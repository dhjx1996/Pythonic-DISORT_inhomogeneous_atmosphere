"""Tune the default ``filter_threshold`` for ``auto_k_active`` / ``select_retrieval_grid``.

Sweeps the retrieval-node count ``k_active`` on the headline thin + thick VOCALS OSSE
configs (matching docs/riccati_solver_VOCALS_retrieval.ipynb), reporting profile-recovery
RMSE (node + dense), posterior DOFS/SIC, and reduced χ² per k — plus the noise-aware
filter spectrum f_i and the (``filter_threshold`` -> k_active) mapping. The user picks the
default from the printed table: the most conservative (lowest threshold / most nodes) value
that does not worsen RMSE — a node is not useless just because <50 % of its information is
expected from the prior.

    /tmp/jaxve/bin/python tests/supplementary/tune_filter_threshold.py
"""
import sys
import json
import time
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
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]     # data-fraction cuts to map -> k_active (margin=1)

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
    return clim, prior_builder, fwd, s_ref, x_ref, y, Se


def sweep(label, truth, bands, k_values):
    clim, prior_builder, fwd, s_ref, x_ref, y, Se = build_case(truth, bands)
    # filter spectrum (threshold-independent) — computed once on the full ODE grid
    _, _, info = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                           prior_builder=prior_builder, filter_threshold=0.0)
    f = np.sort(np.array(info['filter_factors']))[::-1]
    pool_n = int(info['s_pool'].size)
    Se_inv = np.linalg.inv(Se)
    print(f"\n===== {label}: {truth.flight} tau_bot={truth.tau_bot:.1f}, "
          f"{len(bands)} bands x {view_mu.size} views = {fwd.m} obs; ODE pool {pool_n} =====",
          flush=True)
    print(f"  filter spectrum f_i (descending): {np.round(f, 3).tolist()}", flush=True)
    print("  threshold -> k_active (margin=1): "
          + ", ".join(f"f>={t}:{int(np.count_nonzero(f >= t)) + 1}" for t in THRESHOLDS),
          flush=True)
    rows = []
    for k in k_values:
        t0 = time.time()
        s_grid, _, _ = roe.select_retrieval_grid(fwd, x_ref, s_ref, int(k))
        x_a, Sa = prior_builder(s_grid)
        res = roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, n_iter=15, lm=1e-2,
                                  xtol=2e-3, max_n_outer=1, warn=False)
        post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
        kk = len(res.tau_nodes)
        s = np.asarray(res.tau_nodes)
        _, _, tb = fwd._split_state(res.x, s)
        tb = float(tb)
        truth_at = np.interp(s * tb, truth.tau, truth.r_e)
        rmse_node = float(np.sqrt(np.mean((np.asarray(res.x[:kk]) - truth_at) ** 2)))
        sd = np.linspace(0.0, 1.0, 100)
        ret = np.asarray(fwd.profile(res.x, s, sd * tb))
        tru = np.interp(sd * tb, truth.tau, truth.r_e)
        rmse_dense = float(np.sqrt(np.mean((ret - tru) ** 2)))
        r0 = res.y - res.Fx
        chi2_red = float(r0 @ Se_inv @ r0) / fwd.m
        dt = time.time() - t0
        rows.append(dict(k=int(k), k_final=int(kk), rmse_node=rmse_node, rmse_dense=rmse_dense,
                         dofs=float(post.dofs), sic=float(post.sic), chi2_red=chi2_red,
                         tau_bot=tb, sec=dt))
        print(f"  k={k} (final {kk}): RMSE node {rmse_node:.3f} / dense {rmse_dense:.3f} um; "
              f"DOFS {post.dofs:.2f}, SIC {post.sic:.1f}; chi2_red {chi2_red:.2f}; "
              f"tau_bot {tb:.1f} ({dt:.0f}s)", flush=True)
    return dict(label=label, flight=truth.flight, tau_bot=float(truth.tau_bot),
                m=int(fwd.m), pool_n=pool_n, filter_factors=f.tolist(),
                thresholds={str(t): int(np.count_nonzero(f >= t)) + 1 for t in THRESHOLDS},
                rows=rows)


results = []
thin = vio.pick_profile(profiles, target_tau=1.0)
results.append(sweep("THIN ", thin, [1.24, 2.13], [2, 3, 4, 5]))
thick = vio.pick_profile([p for p in profiles if p.flight == 'RF03'], target_tau=23.3)
results.append(sweep("THICK", thick, [1.24, 1.64, 2.13], [3, 4, 5, 6]))

out = Path(__file__).resolve().parents[2] / "docs" / "filter_threshold_sweep.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}", flush=True)
