"""Stage 1 — the information-content matrix (experiments A-E), linearized AT THE TRUTH.

One full Jacobian per regime, then the whole A-E matrix by exact ROW-SUBSETTING:
each ToA observation is an independent row of K, so a config with fewer bands/views
is just fewer rows. We compute K_full once per regime for the FULL band ladder
(6 bands) x FULL view set (16 spread μ, principal plane) at NQuad=32, then derive:

  A  band saturation  : n_bands = 1..6 at nadir (the most-nadir view)         -> Q0/Q1
  B  angular sweep    : n_view  index-subsets of the 16 quadrature-spread μ    -> Q2
  C  spectral x angular: the full (n_bands, n_view) grid -> AK-row localization -> Q3
  D  regime           : thin RF11 / mid RF10 / thick RF03 (separate K_full)    -> Q6
  E  prior ladder     : uninformative -> marine_sc -> climatology (free; only Sa) -> Q5

Linearization point: the TRUTH joint state (truth r_e at the s_ref nodes + truth
r_base + truth tau_bot), so K reflects each regime's ACTUAL optical thickness (the
committed nquad/sic diagnostics used the climatological-mean cloud ~tau10 for every
regime; this supersedes their absolute regime numbers). Observation layout is
BAND-MAJOR (y = [band0 x all views, band1 x all views, ...]; see _forward_raw).

    JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 \
        /tmp/jaxve/bin/python tests/supplementary/info_content_stage1.py
"""
import sys
import json
from pathlib import Path
from math import pi

import numpy as np
import jax

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
from info_content import jacobian_on_ode_grid, info_spectrum        # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OUT = Path(__file__).resolve().parents[2] / "docs" / "cached_results" / "info_content_stage1.json"

NQ, mu0, NLeg_all, v_eff = 32, 0.9, 128, 0.10
BANDS_FULL = [0.66, 0.86, 1.01, 1.24, 1.64, 2.13]          # nested VIS-anchor + SWIR ladder
NVIEW_FULL = 16
VIEW_MU_FULL = np.linspace(0.95, 0.25, NVIEW_FULL)         # nadir -> oblique, principal plane
VIEW_PHI_FULL = np.full(NVIEW_FULL, pi)
REGIMES = [("THIN", "RF11", 1.2), ("MID", "RF10", 4.9), ("THICK", "RF03", 23.3)]
BAND_LADDER = [1, 2, 3, 4, 5, 6]                           # first B bands
# spread index-subsets of the 16-view grid (all subsets of {0..15} -> exact row-subset)
VIEW_LADDER = {1: [0], 2: [0, 15], 3: [0, 7, 15], 4: [0, 5, 10, 15],
               6: [0, 3, 6, 9, 12, 15], 8: [0, 2, 4, 6, 9, 11, 13, 15],
               16: list(range(16))}
s_ref = np.linspace(0.0, 1.0, 6)[:-1]                      # 5 reference nodes

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
opt_full = [select_channel(build_re_table(BANDS_FULL, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
            for i in range(len(BANDS_FULL))]


def make_priors(clim):
    """The Q5 informativeness ladder (means leak-free; IC depends only on Sa)."""
    rt, tb = float(clim['r_top_mean']), float(clim['tau_bot_mean'])
    return {
        'uninformative': (lambda sn: roe.make_joint_prior(
            sn, tau_bot_prior=tb, r_top_prior=rt, r_base_prior=0.65 * rt,
            sigma_top=15.0, sigma_base=15.0, sigma_tau_bot=2.0 * tb)),
        'marine_sc': (lambda sn: roe.make_marine_sc_prior(
            sn, r_top_prior=rt, tau_bot_prior=tb)),
        'climatology': (lambda sn: roe.make_climatology_prior(sn, clim)),
    }


def truth_state(truth):
    """Joint state at the TRUTH: r_e(s_ref) interpolated from the in-situ profile,
    plus the truth r_base and tau_bot — so K linearizes at the real cloud."""
    ts = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(ts)
    re_nodes = np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o])
    return np.concatenate([re_nodes, [truth.r_base], [truth.tau_bot]])


out = {'config': dict(NQuad=NQ, mu0=mu0, bands_full=BANDS_FULL, nview_full=NVIEW_FULL,
                      view_mu_full=VIEW_MU_FULL.tolist(), band_ladder=BAND_LADDER,
                      view_ladder={str(k): v for k, v in VIEW_LADDER.items()},
                      linearization='truth', noise='radiometric 3% (max(|y|,0.02))'),
       'regimes': []}

for label, flight, ttau in REGIMES:
    jax.clear_caches()
    truth = vio.pick_profile([p for p in profiles if p.flight == flight], ttau)
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    fwd = roe.RetrievalForward(
        opt_full, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
        tau_bot=truth.tau_bot, r_base=truth.r_base,
        view_mu=VIEW_MU_FULL, view_phi=VIEW_PHI_FULL, BDRF_bands=[[0.06]] * len(BANDS_FULL),
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
    x_tru = truth_state(truth)
    roe.select_num_modes(fwd, x_tru, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    K_full, s_int = jacobian_on_ode_grid(fwd, x_tru, s_ref)        # (96, n_int)
    n = s_int.size
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se_full = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    print(f"\n{label} {flight}: truth.tau_bot={truth.tau_bot:.1f}  K{K_full.shape}  "
          f"n_int={n}  s_int[-1]={s_int[-1]:.3f}", flush=True)

    priors = make_priors(clim)
    Sa_re = {pn: np.asarray(pb(s_int)[1])[:n, :n] for pn, pb in priors.items()}
    cells, AK_marine = [], {}
    for B in BAND_LADDER:
        for V, vidx in VIEW_LADDER.items():
            rows = np.array([b * NVIEW_FULL + v for b in range(B) for v in vidx])
            K = K_full[rows]
            Se = Se_full[np.ix_(rows, rows)]
            for pn in priors:
                post = roe.posterior_diagnostics(K, Sa_re[pn], Se)
                spec = info_spectrum(K, Sa_re[pn], Se)
                cells.append(dict(n_bands=B, n_view=V, prior=pn,
                                  dofs=float(post.dofs), sic=float(spec.sic),
                                  singular_values=spec.singular_values.tolist(),
                                  data_fraction=post.data_fraction.tolist()))
                if pn == 'marine_sc':
                    AK_marine[f"{B}_{V}"] = post.A.tolist()
    # quick console summary: DOFS(marine_sc) at nadir-1band, full-band-nadir, full-band-full-view
    def d(B, V, pn='marine_sc'):
        return next(c['dofs'] for c in cells if c['n_bands'] == B and c['n_view'] == V and c['prior'] == pn)
    print(f"   marine_sc DOFS: 1band/nadir={d(1,1):.2f}  6band/nadir={d(6,1):.2f}  "
          f"6band/16view={d(6,16):.2f}   SIC(6,16)="
          f"{next(c['sic'] for c in cells if c['n_bands']==6 and c['n_view']==16 and c['prior']=='marine_sc'):.2f}",
          flush=True)
    out['regimes'].append(dict(label=label, flight=flight, tau_bot=float(truth.tau_bot),
                               n_int=n, s_int=s_int.tolist(), cells=cells, AK_marine=AK_marine))
    OUT.write_text(json.dumps(out))                                # incremental
    jax.clear_caches()

OUT.write_text(json.dumps(out))
print(f"\nwrote {OUT}", flush=True)
