"""Q2 follow-up — the single missing THICK RF03 NQuad=48 point.

The main sweep (nquad_saturation.py) capped THICK at NQuad=32 (the heavy 3-band
forward-mode jacfwd compile); THICK was still gently decelerating there, so this adds
the one NQuad=48 row to confirm the THICK plateau. Identical setup to the THICK case in
nquad_saturation.py (float64, n_view=NQuad//2, full-ODE-grid info content at the prior
mean); merges the new row into the existing nquad_saturation.json (THIN + THICK 16/24/32
preserved). clear_caches() guards the large compile.

    JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 \
        /tmp/jaxve/bin/python tests/supplementary/nquad_saturation_thick48.py
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
from info_content import info_content_on_ode_grid                   # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NLeg_all, v_eff = 128, 0.10
mu0, I0, phi0 = 0.9, 1.0, 0.0
NQ = 48
OUT = Path(__file__).resolve().parents[2] / "docs" / "cached_results" / "nquad_saturation.json"

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)

label = "THICK RF03"
bands = [1.24, 1.64, 2.13]
truth = vio.pick_profile([p for p in profiles if p.flight == 'RF03'], 23.3)
clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
opt = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
       for i in range(len(bands))]
prior_builder = (lambda sn: roe.make_marine_sc_prior(
    sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean']))
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

print(f"{label}  tau_bot={truth.tau_bot:.1f}  NQuad={NQ}  n_view={NQ // 2}", flush=True)
jax.clear_caches()
nv = NQ // 2
vmu = np.linspace(0.95, 0.25, nv)
vphi = np.full(nv, pi)
fwd = roe.RetrievalForward(
    opt, NQuad=NQ, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=clim['tau_bot_mean'], r_base=clim['r_base_mean'],
    view_mu=vmu, view_phi=vphi, BDRF_bands=[[0.06]] * len(bands),
    NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
x_ref, _ = prior_builder(s_ref)
roe.select_num_modes(fwd, x_ref, s_ref, (0.005 ** 2) * np.eye(fwd.m))
y = roe.osse_observation(fwd, truth.tau, truth.r_e)
Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
post, _ = info_content_on_ode_grid(fwd, x_ref, s_ref, prior_builder, Se)
row = dict(NQuad=int(NQ), n_view=int(nv), dofs=float(post.dofs), sic=float(post.sic))
print(f"  -> DOFS={row['dofs']:.4f}  SIC={row['sic']:.4f}", flush=True)
jax.clear_caches()

# --- merge into the existing JSON (preserve THIN + THICK 16/24/32) -------------
out = json.loads(OUT.read_text())
thick = next(c for c in out if c['label'] == label)
thick['rows'] = [r for r in thick['rows'] if r['NQuad'] != NQ] + [row]
thick['rows'].sort(key=lambda r: r['NQuad'])
OUT.write_text(json.dumps(out, indent=2))
print(f"\nmerged into {OUT}", flush=True)
d = [r['dofs'] for r in thick['rows']]
nq = [r['NQuad'] for r in thick['rows']]
print(f"THICK DOFS {' -> '.join(f'{v:.2f}' for v in d)} over NQuad {nq}", flush=True)
