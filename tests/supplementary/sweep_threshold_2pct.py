"""filter_threshold 0.25 vs 0.5 at the NEW 2% (OCI) noise — decision evidence for DESIGN §10f.

Question (post noise change 3%->2%): does the Rodgers data/prior crossover (f>=0.5) fix the
§13 sub-adiabatic OVERFIT without under-resolving a structured cloud? Runs, for each case, the
threshold-independent filter spectrum f_i and then a full retrieval at thr in {0.25, 0.5}:
  - headline thin (RF11) + thick (RF03 tau=23), production (tight) prior;
  - §13 sub-adiabatic detection (RF03 tau=1.5 visible drop; RF10 tau=4.9 shielded), LOOSENED
    base prior (sigma_base=8) — the configuration that over-fits.
Reports k, dense RMSE (+ near-base), r_base recovery, drop-recovery cap%, DOFS/SIC, reduced chi2.

    /tmp/jaxve/bin/python tests/supplementary/sweep_threshold_2pct.py
"""
import sys
import time
from pathlib import Path
from math import pi

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQuad, NLeg_all, v_eff = 16, 128, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.linspace(0.95, 0.25, 8)
view_phi = np.full(view_mu.size, pi)
THRESHOLDS = [0.25, 0.5]

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


def make_fwd(truth, bands, sigma_base=None):
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    opt = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
           for i in range(len(bands))]
    extra = {'sigma_base': sigma_base} if sigma_base is not None else {}
    pb = (lambda sn: roe.make_marine_sc_prior(
        sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean'], **extra))
    fwd = roe.RetrievalForward(
        opt, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=clim['tau_bot_mean'], r_base=clim['r_base_mean'],
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[0.06]] * len(bands),
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
    s_ref = np.linspace(0.0, 1.0, 6)[:-1]
    x_ref, _ = pb(s_ref)
    roe.select_num_modes(fwd, x_ref, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = roe.make_Se(fwd, y, nm.oci_swir())                 # NEW 2% OCI noise
    return clim, pb, fwd, s_ref, x_ref, y, Se


def run_case(label, truth, bands, sigma_base=None, max_n_outer=2):
    try:
        clim, pb, fwd, s_ref, x_ref, y, Se = make_fwd(truth, bands, sigma_base)
        _, _, info = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                               prior_builder=pb, filter_threshold=0.0)
        f = np.sort(np.array(info['filter_factors']))[::-1]
        Se_inv = np.linalg.inv(Se)
        kmap = ", ".join(f"f>={t}:{int(np.count_nonzero(f >= t)) + 1}" for t in (0.25, 0.5))
        print(f"\n===== {label}: {truth.flight} tau={truth.tau_bot:.2f}, m={fwd.m}, "
              f"pool={info['s_pool'].size} =====", flush=True)
        print(f"  filter spectrum f_i = {np.round(f, 3).tolist()}", flush=True)
        print(f"  threshold->k (margin=1): {kmap}", flush=True)
        for thr in THRESHOLDS:
            t0 = time.time()
            s_grid, _, _ = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                                     prior_builder=pb, filter_threshold=thr)
            x_a, Sa = pb(s_grid)
            res = roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, n_iter=15, lm=1e-2,
                                      xtol=2e-3, max_n_outer=max_n_outer,
                                      prior_builder=pb, warn=False)
            post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
            kk = len(res.tau_nodes)
            s = np.asarray(res.tau_nodes)
            _, rb, tb = fwd._split_state(res.x, s)
            rb, tb = float(rb), float(tb)
            sd = np.linspace(0.0, 1.0, 200)
            ret = np.asarray(fwd.profile(res.x, s, sd * tb))
            tru = np.interp(sd * tb, truth.tau, truth.r_e)
            rmse = float(np.sqrt(np.mean((ret - tru) ** 2)))
            near = sd > 0.7
            rmse_near = float(np.sqrt(np.mean((ret[near] - tru[near]) ** 2)))
            drop_ret, drop_tru = float(ret.max() - ret[-1]), float(tru.max() - tru[-1])
            cap = drop_ret / drop_tru if drop_tru > 1e-9 else float('nan')
            r0 = res.y - res.Fx
            chi2 = float(r0 @ Se_inv @ r0) / fwd.m
            print(f"  thr={thr}: k={kk} | RMSE {rmse:.2f} (near-base {rmse_near:.2f}) um | "
                  f"r_base {rb:.1f} (truth {truth.r_base:.1f}) | drop cap {cap * 100:.0f}% | "
                  f"DOFS {post.dofs:.2f} SIC {post.sic:.1f} | chi2 {chi2:.2f} | "
                  f"tau_bot {tb:.1f} (truth {truth.tau_bot:.1f}) ({time.time() - t0:.0f}s)",
                  flush=True)
    except Exception as e:
        import traceback
        print(f"  !! {label} failed: {e}", flush=True)
        traceback.print_exc()


run_case("THIN  RF11", vio.pick_profile(profiles, target_tau=1.0), [1.24, 2.13])
run_case("THICK RF03",
         vio.pick_profile([p for p in profiles if p.flight == 'RF03'], target_tau=23.3),
         [1.24, 1.64, 2.13])
run_case("SUBADIA RF03 (visible)",
         vio.pick_profile([p for p in profiles if p.flight == 'RF03'], target_tau=1.53),
         [1.24, 1.64, 2.13], sigma_base=8.0)
run_case("SUBADIA RF10 (shielded)",
         vio.pick_profile([p for p in profiles if p.flight == 'RF10'], target_tau=4.94),
         [1.24, 1.64, 2.13], sigma_base=8.0)
print("\nDONE", flush=True)
