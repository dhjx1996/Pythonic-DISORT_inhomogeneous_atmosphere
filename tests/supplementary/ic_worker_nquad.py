"""NQuad worker: one (profile, NQuad) full-ODE-grid IC at the truth.
Usage: ic_worker_nquad.py <target_tau> <NQuad> <out.json>  (3 SWIR bands, n_view=NQuad//2,
marine_sc). Driven in parallel by _ic_parallel.py."""
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
from info_content import info_content_on_ode_grid                   # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
mu0, NLeg_all, v_eff = 0.9, 128, 0.10
BANDS = [1.24, 1.64, 2.13]
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

target_tau, NQ, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
nv = NQ // 2
profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
opt = [select_channel(build_re_table(BANDS, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
       for i in range(len(BANDS))]
truth = vio.pick_profile(profiles, target_tau)
clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
vmu, vphi = np.linspace(0.95, 0.25, nv), np.full(nv, pi)
fwd = roe.RetrievalForward(opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
                           tau_bot=truth.tau_bot, r_base=truth.r_base, view_mu=vmu, view_phi=vphi,
                           BDRF_bands=[[0.06]] * len(BANDS), NLeg_all=NLeg_all,
                           retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
ts = np.asarray(truth.tau, float) / truth.tau_bot
o = np.argsort(ts)
x_tru = np.concatenate([np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o]),
                        [truth.r_base], [truth.tau_bot]])
roe.select_num_modes(fwd, x_tru, s_ref, (0.005 ** 2) * np.eye(fwd.m))
pb = (lambda sn: roe.make_marine_sc_prior(
    sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean']))
y = roe.osse_observation(fwd, truth.tau, truth.r_e)
Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
post, _ = info_content_on_ode_grid(fwd, x_tru, s_ref, pb, Se)
rec = dict(flight=truth.flight, tau_bot=float(truth.tau_bot), NQuad=NQ, n_view=nv,
           dofs=float(post.dofs), sic=float(post.sic))
Path(out).write_text(json.dumps(rec))
print(f"{truth.flight} tau={truth.tau_bot:.1f} NQuad={NQ}: DOFS={post.dofs:.2f} SIC={post.sic:.2f}", flush=True)
