"""Settle the SIC-vs-node-count shape: plateau, monotone, or n-shaped?

Sweeps the retrieval node count k from small up to (near) the FULL m=0 ODE grid,
recomputing the Jacobian at each k (so per-node leverage is the *actual* one for that
grid, which is what makes SIC potentially n-shaped). DOFS is expected to plateau; the
open question is whether SIC turns over at high k. Run at the locked mu0=0.9. No
Gauss-Newton retrieval (info content = Jacobian @ truth).

MEMORY: the thick full-grid forward-mode jacfwd compiles are large and *accumulate*
into an LLVM "Unable to allocate section memory" OOM. ``jax.clear_caches()`` between
cases and after each k keeps only one compiled executable resident; the thick k-list is
also capped (the n-shape is visible well below the full pool).

    JAX_PLATFORMS=cpu /tmp/jaxve/bin/python tests/supplementary/sic_dofs_vs_k_fullgrid.py
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
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQuad, NLeg_all, v_eff = 16, 128, 0.10
mu0, I0, phi0 = 0.9, 1.0, 0.0                                       # <-- locked near-nadir
view_mu = np.linspace(0.95, 0.25, 8)
view_phi = np.full(view_mu.size, pi)

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
    tau_grid = fwd.ode_grid(x_ref, s_ref)
    s_full = np.unique(np.clip(tau_grid / clim['tau_bot_mean'], 0, 1))
    n_pool = int((s_full < 1 - 1e-6).sum())
    return prior_builder, fwd, s_ref, x_ref, Se, n_pool


CASES = [
    ("THIN RF11", vio.pick_profile([p for p in profiles if p.flight == 'RF11'], 1.2), [1.24, 2.13]),
    ("THICK RF03", vio.pick_profile([p for p in profiles if p.flight == 'RF03'], 23.3),
     [1.24, 1.64, 2.13]),
]

out = []
for label, truth, bands in CASES:
    jax.clear_caches()                              # free the prior case's compiles (LLVM OOM guard)
    prior_builder, fwd, s_ref, x_ref, Se, n_pool = build_case(truth, bands)
    kmax = n_pool if label.startswith("THIN") else min(n_pool, 12)   # cap heaviest thick compile
    ks = sorted(set([2, 3, 4, 5, 6, 8, kmax]))
    ks = [k for k in ks if 2 <= k <= kmax]
    print(f"\n{label}  tau_bot={truth.tau_bot:.1f}  (full ODE-grid interior nodes={n_pool}, k≤{kmax})",
          flush=True)
    print(f"  {'k':>3} {'DOFS':>7} {'SIC':>8}", flush=True)
    rows = []
    for k in ks:
        s_sel, _, _ = roe.select_retrieval_grid(fwd, x_ref, s_ref, int(k))
        x0, Sa = prior_builder(s_sel)
        K = np.asarray(fwd.jacobian(x0, s_sel))     # Jacobian for THIS grid (true leverage)
        post = roe.posterior_diagnostics(K, Sa, Se)
        rows.append(dict(k=int(k), dofs=float(post.dofs), sic=float(post.sic)))
        print(f"  {k:>3} {post.dofs:7.2f} {post.sic:8.2f}", flush=True)
        jax.clear_caches()                          # free this k's jacfwd compile
    sics = [r['sic'] for r in rows]
    peak_k = rows[int(np.argmax(sics))]['k']
    shape = ("MONOTONE-RISING" if np.all(np.diff(sics) > -1e-3)
             else ("N-SHAPED (peak interior)" if 2 < peak_k < ks[-1] else "DECLINING"))
    print(f"  -> SIC shape: {shape};  SIC-peak k={peak_k} of [{ks[0]}..{ks[-1]}]; "
          f"DOFS plateau≈{rows[-1]['dofs']:.2f}", flush=True)
    out.append(dict(label=label, flight=truth.flight, n_pool=n_pool, k_max=kmax,
                    sic_peak_k=int(peak_k), shape=shape, rows=rows))

p = Path(__file__).resolve().parents[2] / "docs" / "cached_results" / "sic_dofs_vs_k_fullgrid.json"
p.write_text(json.dumps(out, indent=2))
print(f"\nwrote {p}", flush=True)
