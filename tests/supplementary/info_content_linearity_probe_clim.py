"""Secondary linearity-robustness probe — same as info_content_linearity_probe.py but
using make_climatology_prior (the strongest, per-flight LOO prior) instead of
make_marine_sc_prior. Confirms the truth-vs-prior IC gap is small under the strong prior too.

  - truth : the in-situ profile sampled at s_ref
  - prior : the climatology prior mean (make_climatology_prior) with tau_bot fixed to truth.

    JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 \
        /tmp/jaxve/bin/python tests/supplementary/info_content_linearity_probe_clim.py
"""
import sys
import os
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

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OUT = Path(__file__).resolve().parents[2] / "docs" / "cached_results" / "info_content_linearity_probe_clim.json"

NQ, mu0, NLeg_all, v_eff = 32, 0.9, 128, 0.10
BANDS = [1.24, 1.64, 2.13]
NVIEW = 16
VIEW_MU = np.linspace(0.95, 0.25, NVIEW)
VIEW_PHI = np.full(NVIEW, pi)
REGIMES = [("THIN", "RF11", 1.2), ("MID", "RF10", 4.9), ("THICK", "RF03", 23.3)]
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
opt = [select_channel(build_re_table(BANDS, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
       for i in range(len(BANDS))]


def truth_state(truth):
    ts = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(ts)
    re_nodes = np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o])
    return np.concatenate([re_nodes, [truth.r_base], [truth.tau_bot]])


def prior_mean_state(clim, tau_bot):
    """Climatology prior mean with tau_bot KNOWN (= truth thickness)."""
    x_a = np.asarray(roe.make_climatology_prior(s_ref, clim)[0], float).copy()
    x_a[-1] = float(tau_bot)
    return x_a


def ic_at(fwd, x, pb, Se):
    K, s_int = jacobian_on_ode_grid(fwd, x, s_ref)
    n = s_int.size
    Sa = np.asarray(pb(s_int)[1])[:n, :n]
    post = roe.posterior_diagnostics(K, Sa, Se)
    spec = info_spectrum(K, Sa, Se)
    return float(post.dofs), float(spec.sic), n


out = []
for label, flight, ttau in REGIMES:
    jax.clear_caches()
    truth = vio.pick_profile([p for p in profiles if p.flight == flight], ttau)
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    fwd = roe.RetrievalForward(
        opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
        tau_bot=truth.tau_bot, r_base=truth.r_base,
        view_mu=VIEW_MU, view_phi=VIEW_PHI, BDRF_bands=[[0.06]] * len(BANDS),
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
    x_tru = truth_state(truth)
    x_pri = prior_mean_state(clim, truth.tau_bot)
    roe.select_num_modes(fwd, x_tru, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    pb = (lambda sn: roe.make_climatology_prior(sn, clim))

    d_t, s_t, n_t = ic_at(fwd, x_tru, pb, Se)
    d_p, s_p, n_p = ic_at(fwd, x_pri, pb, Se)
    out.append(dict(label=label, flight=flight, tau_bot=float(truth.tau_bot),
                    dofs_truth=d_t, dofs_prior=d_p, dofs_gap=d_t - d_p,
                    sic_truth=s_t, sic_prior=s_p, sic_gap=s_t - s_p,
                    n_int_truth=n_t, n_int_prior=n_p))
    print(f"{label} {flight} tau_bot={truth.tau_bot:5.1f}: "
          f"DOFS truth={d_t:.2f} prior={d_p:.2f} (Δ={d_t-d_p:+.2f})   "
          f"SIC truth={s_t:.2f} prior={s_p:.2f} (Δ={s_t-s_p:+.2f})", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    jax.clear_caches()

print(f"\nwrote {OUT}  (climatology prior)", flush=True)
