"""Ensemble worker: one profile's headline IC. Usage: ic_worker_ensemble.py <target_tau> <out.json>
Computes nadir vs 16-view DOFS/SIC + depth-reach (6 bands x 16 views, truth-linearized,
NQuad=32, marine_sc). Driven in parallel by _ic_parallel.py."""
import sys
import os
import json
from pathlib import Path
from math import pi

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
from info_content import (jacobian_on_ode_grid, flux_jacobian_on_ode_grid,  # noqa: E402
                          info_spectrum)
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQ, mu0, NLeg_all, v_eff = 32, 0.9, 128, 0.10
BANDS_FULL = [0.66, 0.86, 1.01, 1.24, 1.64, 2.13]
NVIEW = 16
VIEW_MU, VIEW_PHI = np.linspace(0.95, 0.25, NVIEW), np.full(NVIEW, pi)
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

target_tau, out = float(sys.argv[1]), sys.argv[2]
profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
opt = [select_channel(build_re_table(BANDS_FULL, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
       for i in range(len(BANDS_FULL))]
truth = vio.pick_profile(profiles, target_tau)
clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
fwd = roe.RetrievalForward(opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
                           tau_bot=truth.tau_bot, r_base=truth.r_base, view_mu=VIEW_MU, view_phi=VIEW_PHI,
                           BDRF_bands=[[0.06]] * len(BANDS_FULL), NLeg_all=NLeg_all,
                           retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
ts = np.asarray(truth.tau, float) / truth.tau_bot
o = np.argsort(ts)
x_tru = np.concatenate([np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o]),
                        [truth.r_base], [truth.tau_bot]])
roe.select_num_modes(fwd, x_tru, s_ref, (0.005 ** 2) * np.eye(fwd.m))
K_full, s_int = jacobian_on_ode_grid(fwd, x_tru, s_ref)             # radiance (n_bands*16, n)
K_flux, _ = flux_jacobian_on_ode_grid(fwd, x_tru, s_ref)            # albedo   (n_bands, n)
n = s_int.size
s = np.array(s_int)
Sa = np.asarray(roe.make_marine_sc_prior(
    s_int, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean'])[1])[:n, :n]
y = roe.osse_observation(fwd, truth.tau, truth.r_e)                 # radiance obs
Se_full = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
y_flux = fwd.flux_reflectance(x_tru, s_ref)                         # albedo obs
Se_flux = np.diag((0.03 * np.maximum(np.abs(y_flux), 0.02)) ** 2)
nadir_rows = np.array([b * NVIEW + 0 for b in range(len(BANDS_FULL))])      # mu=0.95
i06 = int(np.argmin(np.abs(VIEW_MU - 0.6)))                                # view nearest mu=0.6
v06_rows = np.array([b * NVIEW + i06 for b in range(len(BANDS_FULL))])     # oblique single-view check


def metrics(K, Se):
    post = roe.posterior_diagnostics(K, Sa, Se)
    spec = info_spectrum(K, Sa, Se)
    idx = np.where(post.data_fraction >= 0.5)[0]
    return float(post.dofs), float(spec.sic), (float(s[idx.max()]) if len(idx) else 0.0)


# albedo (CPV flux baseline) | single radiance at nadir (0.95) & oblique (~0.6) | full 16-view
d_a, sic_a, dep_a = metrics(K_flux, Se_flux)
d_n, sic_n, dep_n = metrics(K_full[nadir_rows], Se_full[np.ix_(nadir_rows, nadir_rows)])
d_6, sic_6, dep_6 = metrics(K_full[v06_rows], Se_full[np.ix_(v06_rows, v06_rows)])
d_v, sic_v, dep_v = metrics(K_full, Se_full)
rec = dict(flight=truth.flight, tau_bot=float(truth.tau_bot), n_int=n,
           dofs_albedo=d_a, dofs_nadir=d_n, dofs_view06=d_6, dofs_16view=d_v,
           angular_gain=d_v - d_a, view06_mu=float(VIEW_MU[i06]),
           sic_albedo=sic_a, sic_nadir=sic_n, sic_view06=sic_6, sic_16view=sic_v,
           depth_albedo=dep_a, depth_nadir=dep_n, depth_view06=dep_6, depth_16view=dep_v)
Path(out).write_text(json.dumps(rec))
print(f"{truth.flight} tau={truth.tau_bot:.1f}: DOFS albedo={d_a:.2f} nadir(0.95)={d_n:.2f} "
      f"view({VIEW_MU[i06]:.2f})={d_6:.2f} 16view={d_v:.2f}", flush=True)
