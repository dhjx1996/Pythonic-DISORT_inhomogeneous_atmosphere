"""Second thick-cloud filter_threshold stress-test.

Picks a thick VOCALS profile chosen to be SHAPE-DIFFERENT from RF03 τ23.3 (the one already
swept), to confirm the k=3 / filter_threshold=0.5 finding generalizes rather than being a
quirk of one profile. Same OSSE config as tune_filter_threshold.py.

    /tmp/jaxve/bin/python tests/supplementary/thick_sweep2.py
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
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
BANDS = [1.24, 1.64, 2.13]

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)

# --- choose a thick profile most SHAPE-DIFFERENT from RF03 τ23.3 (a different flight) ---
ref = vio.pick_profile([p for p in profiles if p.flight == 'RF03'], target_tau=23.3)
sgrid = np.linspace(0.05, 0.95, 19)


def nshape(p):
    v = np.interp(sgrid, p.tau / p.tau_bot, p.r_e)
    return v / v.mean()                              # normalized shape (magnitude-free)


ref_n = nshape(ref)
cands = []
for p in profiles:
    if p.flight == 'RF03':
        continue
    if not (15.0 <= p.tau_bot <= 40.0):              # genuinely thick
        continue
    if p.r_e.min() < 2.5 or p.r_e.max() > 24.0:      # inside the Mie table
        continue
    d = float(np.sqrt(np.mean((nshape(p) - ref_n) ** 2)))
    cands.append((d, p))
cands.sort(key=lambda t: t[0], reverse=True)

print(f"reference RF03: tau_bot={ref.tau_bot:.1f} r_top={ref.r_top:.1f} r_base={ref.r_base:.1f} "
      f"ratio={ref.r_base / ref.r_top:.2f} re[{ref.r_e.min():.1f},{ref.r_e.max():.1f}]")
print("thick candidates (shape-diff from RF03 τ23.3, desc):")
for d, p in cands[:6]:
    print(f"  {p.flight} tau_bot={p.tau_bot:.1f} r_top={p.r_top:.1f} r_base={p.r_base:.1f} "
          f"ratio={p.r_base / p.r_top:.2f} re[{p.r_e.min():.1f},{p.r_e.max():.1f}] shape_diff={d:.3f}")
truth = cands[0][1]
print(f"\nCHOSEN (most shape-different thick): {truth.flight} tau_bot={truth.tau_bot:.1f}\n", flush=True)


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


clim, prior_builder, fwd, s_ref, x_ref, y, Se = build_case(truth, BANDS)
_, _, info = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                       prior_builder=prior_builder, filter_threshold=0.0)
f = np.sort(np.array(info['filter_factors']))[::-1]
Se_inv = np.linalg.inv(Se)
print(f"===== {truth.flight} tau_bot={truth.tau_bot:.1f}, {len(BANDS)} bands x {view_mu.size} "
      f"views = {fwd.m} obs; ODE pool {info['s_pool'].size} =====", flush=True)
print(f"  filter spectrum f_i (descending): {np.round(f, 3).tolist()}", flush=True)
print("  threshold -> k_active (margin=1): "
      + ", ".join(f"f>={t}:{int(np.count_nonzero(f >= t)) + 1}" for t in THRESHOLDS), flush=True)

rows = []
for k in (3, 4, 5, 6):
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
                     dofs=float(post.dofs), sic=float(post.sic), chi2_red=chi2_red, tau_bot=tb))
    print(f"  k={k} (final {kk}): RMSE node {rmse_node:.3f} / dense {rmse_dense:.3f} um; "
          f"DOFS {post.dofs:.2f}, SIC {post.sic:.1f}; chi2_red {chi2_red:.2f}; "
          f"tau_bot {tb:.1f} ({dt:.0f}s)", flush=True)

out = Path(__file__).resolve().parents[2] / "docs" / "thick_sweep2.json"
out.write_text(json.dumps(dict(flight=truth.flight, tau_bot=float(truth.tau_bot),
                               r_top=float(truth.r_top), r_base=float(truth.r_base),
                               filter_factors=f.tolist(), rows=rows), indent=2))
print(f"\nwrote {out}", flush=True)
