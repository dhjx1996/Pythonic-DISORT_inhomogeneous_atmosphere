"""Q2 — does r_e information saturate as the stream count NQuad grows?

At each NQuad we set n_view = NQuad//2 (enough views to capture the full N=NQuad//2-dim
emergent field) and compute the full-ODE-grid information content (DOFS/SIC) at the prior
mean. If DOFS/SIC plateau as NQuad 16->24->32->48, then finer angular resolution (more
streams) carries no additional r_e information — the angular ceiling is set by the
low-order angular content of the r_e signal, not by sampling density.

clear_caches() between NQuad values (the NQuad=48 forward-mode jacfwd compile is large);
JSON is written incrementally so a heavy-NQuad crash keeps the lighter results.

    JAX_PLATFORMS=cpu /tmp/jaxve/bin/python tests/supplementary/nquad_saturation.py
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
OUT = Path(__file__).resolve().parents[2] / "docs" / "cached_results" / "nquad_saturation.json"

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)

CASES = [
    ("THIN RF11", vio.pick_profile([p for p in profiles if p.flight == 'RF11'], 1.2),
     [1.24, 2.13], [16, 24, 32, 48]),
    ("THICK RF03", vio.pick_profile([p for p in profiles if p.flight == 'RF03'], 23.3),
     [1.24, 1.64, 2.13], [16, 24, 32]),
]

out = []
for label, truth, bands, nquads in CASES:
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    opt = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
           for i in range(len(bands))]
    prior_builder = (lambda sn: roe.make_marine_sc_prior(
        sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean']))
    s_ref = np.linspace(0.0, 1.0, 6)[:-1]
    print(f"\n{label}  tau_bot={truth.tau_bot:.1f}", flush=True)
    print(f"  {'NQuad':>6}{'n_view':>7}{'DOFS':>8}{'SIC':>8}", flush=True)
    rows = []
    case = dict(label=label, flight=truth.flight, bands=bands, rows=rows)
    out.append(case)
    for NQ in nquads:
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
        rows.append(dict(NQuad=int(NQ), n_view=int(nv),
                         dofs=float(post.dofs), sic=float(post.sic)))
        print(f"  {NQ:>6}{nv:>7}{post.dofs:>8.2f}{post.sic:>8.2f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))          # incremental
        jax.clear_caches()
    if len(rows) >= 2:
        d = [r['dofs'] for r in rows]
        print(f"  -> DOFS {d[0]:.2f}->{d[-1]:.2f} over NQuad {nquads[0]}->{nquads[len(rows)-1]} "
              f"(Δ={d[-1]-d[0]:+.2f}); SIC {rows[0]['sic']:.2f}->{rows[-1]['sic']:.2f}", flush=True)

print(f"\nwrote {OUT}", flush=True)
