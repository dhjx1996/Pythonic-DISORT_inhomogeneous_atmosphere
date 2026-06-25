"""DEFINITIVE all-VOCALS IC worker — one profile BY INDEX, raw Jacobian cached.

Usage:  ic_worker_profile.py <profile_index> <out.json>
        (also writes the sidecar <out.npz> with the raw Jacobians — the real product)
Env:    ENSEMBLE_NQUAD (default 48 — converged operating point; N = NQuad//2 streams).
        IC_MODE        (default 'priormean') — the LINEARIZATION POINT:
                         priormean : at the LOO prior MEAN (3-param adiabatic curve)
                                     -> priors {loo (iv) HEADLINE, weak (i; σ≈10), loo2x (vi; ℓ=2×)}
                         draw      : at a LOO adiabatic realization (r_top,r_base~LOO, τ_bot=truth)
                                     -> prior {loo (v)}  (robustness)
        IC_SIGMA_WEAK  (default 10) — σ of the weak ~flat diagonal prior (set i; King & Vaughan 2012).
        OPTICS_CACHE   — path to the cached miepython optics table (.npz). Defaults beside this file;
                         the table is profile-INDEPENDENT, so it is built once and shared by all tasks.

What changed vs the pilot (DEFINITIVE run, 2026-06-24; DESIGN_DECISIONS §13):
  * BANDS — the 9-band instrument-sourced superset (HARP2 VIS/NIR + OCI SWIR + NK1990 3.7 µm); no
    band order is baked in here (the value-/data-greedy ordering is a free post-hoc choice on K_full).
  * IC STATE = ALL r_e(τ) nodes INCLUDING THE BASE (r_base / s=1); τ_bot held KNOWN (re-examined &
    retained — DESIGN §13). info_content.jacobian_on_ode_grid(..., include_base=True).
  * Sₑ = OCI 2 % calibration-relative (noise_model.oci_swir), radiance AND flux — matching the
    pre-§15 retrievals (the pilot's legacy 3 % is the inconsistency being fixed).
  * OPTICS = miepython table (optics_table.build_or_load_table); JAX-Mie retired (autodiff does not
    flow through Mie — it enters only the differentiable table_lookup). Validated vs miejax_lite.
  * TRUTH-linearization RETIRED (the pilot's set ii); modes are {priormean, draw}.
  * RAW K CACHED — K_full (all bands × all views), K_flux, s_int, the noise σ, the prior covariances,
    and the linearization metadata go to <out.npz>. The (n_bands, n_view) trade-off grid, every band
    ordering/subset, and Δ_ang/Δ_spec are then ~ms SVDs of row-subsets of one K_full, done post-hoc by
    ic_analysis_definitive.py — so the figures are built off the cache, not re-run.

Two robustness levers (DESIGN §11): (a) WHERE we linearize (IC_MODE), (b) the PRIOR (loo =
make_climatology_prior, leave-one-flight-out; weak = σ≈10 ~flat diagonal, KV2012; loo2x = LOO with 2×
the depth-correlation length). Sₑ is held at the truth's radiance/flux for every mode (so only K and
Sa vary). τ_bot is FIXED at the truth for every mode (fully measured, A≈1; the τ_bot-unknown case is a
separate one-off sensitivity check, ic_tau_bot_check.py). Index-addressed (no tau-dedup); degenerate
profiles write {skipped:...}.
"""
import sys
import os
import json
import time
from pathlib import Path
from math import pi

import numpy as np

_here = Path(__file__).resolve().parent
_src = _here.parents[1] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import optics_table as ot                                          # noqa: E402
from info_content import (jacobian_on_ode_grid, flux_jacobian_on_ode_grid,  # noqa: E402
                          info_spectrum)

DATA = os.environ.get('VOCALS_DATA',                                # HPC: export VOCALS_DATA=/burg-archive/...
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQ = int(os.environ.get('ENSEMBLE_NQUAD', '48'))
IC_MODE = os.environ.get('IC_MODE', 'priormean')                   # priormean | draw
SIGMA_WEAK = float(os.environ.get('IC_SIGMA_WEAK', '10'))         # set i: weak ~flat prior, σ≈10 µm (KV2012)
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_10band.npz'))
N_PHYS = NQ // 2
NV_MAX = N_PHYS + 8                                                 # views to N + a few beyond (TMS)
mu0, NLeg_all, v_eff = 0.9, 128, 0.10
# 10-band instrument superset (ascending λ; ordering is a post-hoc choice, NOT baked in):
# 0.55/0.67/0.86 HARP2 VIS/NIR (0.67 = 60 view angles) · 1.038 OCI window · 1.24/1.64/2.13/2.26 OCI SWIR
# · 3.7 NK1990 strong-absorption · 4.05 VIIRS M13 (ω≈0.87, operational MWIR, slightly more absorbing than
# 3.7 — tests spectral headroom beyond 3.7).
BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]
NB = len(BANDS)
NOISE = nm.oci_swir()                                              # OCI 2 % calibration-relative + 1e-3 floor
VIEW_MU, VIEW_PHI = np.linspace(0.95, 0.25, NV_MAX), np.full(NV_MAX, pi)
s_ref = np.linspace(0.0, 1.0, 6)[:-1]                              # retrieval grid s=[0,.2,.4,.6,.8]


def spread_idx(k):
    return np.unique(np.linspace(0, NV_MAX - 1, k).round().astype(int))


idx, out = int(sys.argv[1]), sys.argv[2]
profiles = vio.load_all_profiles(DATA)
truth = profiles[idx]
flight = getattr(truth, 'flight', '?')

try:
    if not (0.3 <= float(truth.tau_bot) <= 100.0) or len(np.asarray(truth.tau)) < 5:
        raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f}, npts={len(truth.tau)})")
    if IC_MODE not in ('priormean', 'draw'):
        raise ValueError(f"bad IC_MODE={IC_MODE!r}")
    # Profile-independent miepython optics table: build once (cached), then load.
    re_table = ot.build_or_load_table(BANDS, 2.0, 25.0, 32, v_eff,
                                      cache_path=OPTICS_CACHE, NLeg=NLeg_all, n_radii=600)
    opt = [ot.select_channel(re_table, i) for i in range(NB)]
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    fwd = roe.RetrievalForward(
        opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0, tau_bot=truth.tau_bot, r_base=truth.r_base,
        view_mu=VIEW_MU, view_phi=VIEW_PHI, BDRF_bands=[[0.06]] * NB,
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')

    ts = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(ts)
    x_tru = np.concatenate([np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o]),
                            [truth.r_base], [truth.tau_bot]])
    xa_sref = np.asarray(roe.make_climatology_prior(s_ref, clim)[0])  # LOO prior MEAN on s_ref
    nb6 = len(s_ref) + 1                                           # r_e nodes + r_base (drop tau_bot)

    lin = {}
    if IC_MODE == 'priormean':                                    # set iv (HEADLINE): LOO prior mean
        x_lin = np.concatenate([xa_sref[:nb6], [truth.tau_bot]])
    else:                                                         # draw (v): a physical 3-param ADIABATIC
        x_lin, lin = roe.draw_climatology_realization(            # realization (r_top,r_base~LOO; τ_bot=truth)
            clim, s_ref, rng=np.random.default_rng(1000 + idx), tau_bot=truth.tau_bot)

    t0 = time.time()
    print(f"[{idx}] {flight} tau={truth.tau_bot:.1f} mode={IC_MODE}: optics ready, building Jacobian...", flush=True)
    roe.select_num_modes(fwd, x_lin, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    # include_base=True -> K columns span ALL r_e nodes INCLUDING the s=1 base (r_base is an r_e value).
    K_full, s_int = jacobian_on_ode_grid(fwd, x_lin, s_ref, include_base=True)
    print(f"[{idx}] {flight}: radiance Jacobian done in {time.time()-t0:.0f}s "
          f"(n_int={s_int.size}, base incl.); flux Jacobian...", flush=True)
    K_flux, _ = flux_jacobian_on_ode_grid(fwd, x_lin, s_ref, include_base=True)
    n = s_int.size
    s = np.array(s_int)
    s_interior = s[:-1]                                           # base re-appended by make_climatology_prior

    # priors on the ODE grid (r_e+base block; the n×n block excludes only the τ_bot scalar)
    Sa_loo = np.asarray(roe.make_climatology_prior(s_interior, clim)[1])[:n, :n]
    priors = {'loo': Sa_loo}
    if IC_MODE == 'priormean':
        priors['weak'] = (SIGMA_WEAK ** 2) * np.eye(n)            # set i: weak ~flat diagonal (σ≈10, KV2012)
        priors['loo2x'] = np.asarray(roe.make_climatology_prior(  # set vi: LOO with ℓ=1.0 (2× the 0.5 default)
            s_interior, clim, corr_length=1.0)[1])[:n, :n]

    # Sₑ at the truth's radiance/flux via the OCI 2 % model (radiance + flux); store σ for post-hoc subsets.
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    sig_full = NOISE.sigma(y, n_bands=NB)
    Se_full = np.diag(sig_full ** 2)
    y_flux = fwd.flux_reflectance(x_tru, s_ref)
    sig_flux = NOISE.sigma(y_flux, n_bands=NB)
    Se_flux = np.diag(sig_flux ** 2)

    # --- raw-Jacobian sidecar (the product; all ordering/subset/trade-off analysis is post-hoc) -----
    npz_path = Path(out).with_suffix('.npz')
    cache = dict(index=idx, flight=flight, tau_bot=float(truth.tau_bot), NQuad=NQ, ic_mode=IC_MODE,
                 n_bands=NB, nv_max=NV_MAX, n_phys=N_PHYS, bands=np.asarray(BANDS), view_mu=VIEW_MU,
                 s_int=s, n_int=n, K_full=np.asarray(K_full), K_flux=np.asarray(K_flux),
                 sigma_full=sig_full, sigma_flux=sig_flux,
                 y_full=np.asarray(y), y_flux=np.asarray(y_flux),   # reflectance -> rebuild Se at any noise
                 x_lin=np.asarray(x_lin),
                 lin=json.dumps(lin), sigma_weak=(SIGMA_WEAK if IC_MODE == 'priormean' else np.nan))
    for nmk, Sa in priors.items():
        cache[f'Sa_{nmk}'] = Sa
    np.savez(npz_path, **cache)

    # --- slim JSON: headline scalars at the LOO prior, for monitoring + quick aggregation -----------
    nadir = np.array([b * NV_MAX + 0 for b in range(NB)])
    full = np.array([b * NV_MAX + v for b in range(NB) for v in spread_idx(N_PHYS)])

    def metrics(K, sig, Sa):
        Se = np.diag(sig ** 2)
        post = roe.posterior_diagnostics(K, Sa, Se)
        spec = info_spectrum(K, Sa, Se)
        ix = np.where(post.data_fraction >= 0.5)[0]
        return float(post.dofs), float(spec.sic), (float(s[ix.max()]) if len(ix) else 0.0)

    d_a, sic_a, dep_a = metrics(K_flux, sig_flux, Sa_loo)
    d_n, sic_n, dep_n = metrics(K_full[nadir], sig_full[nadir], Sa_loo)
    d_v, sic_v, dep_v = metrics(K_full[full], sig_full[full], Sa_loo)
    rec = dict(index=idx, flight=flight, tau_bot=float(truth.tau_bot), n_int=n, NQuad=NQ,
               n_phys=N_PHYS, nv_max=NV_MAX, bands=BANDS, ic_mode=IC_MODE,
               sigma_weak=(SIGMA_WEAK if IC_MODE == 'priormean' else None), lin=lin,
               npz=npz_path.name, priors=sorted(priors),
               loo=dict(dofs_albedo=d_a, dofs_nadir=d_n, dofs_fullview=d_v,
                        sic_albedo=sic_a, sic_nadir=sic_n, sic_fullview=sic_v,
                        depth_albedo=dep_a, depth_nadir=dep_n, depth_fullview=dep_v))
    print(f"[{idx}] {flight} tau={truth.tau_bot:.1f} mode={IC_MODE} loo -> "
          f"alb={d_a:.2f} nadir={d_n:.2f} full={d_v:.2f} DOFS | "
          f"SIC {sic_a:.1f}/{sic_n:.1f}/{sic_v:.1f} | cached {npz_path.name}", flush=True)
except Exception as e:                                              # noqa: BLE001
    rec = dict(index=idx, flight=flight, tau_bot=float(getattr(truth, 'tau_bot', 0.0)),
               NQuad=NQ, ic_mode=IC_MODE, skipped=str(e)[:160])
    print(f"[{idx}] {flight}: SKIPPED {rec['skipped']}", flush=True)

Path(out).write_text(json.dumps(rec))
