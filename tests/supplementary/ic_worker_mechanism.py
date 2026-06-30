"""Mechanism worker (DEFINITIVE): direct-vs-prior-propagated depth-reach, BY INDEX.

Usage: ic_worker_mechanism.py <profile_index> <out.json>
Env:   ENSEMBLE_NQUAD (default 48 — N = NQuad//2 views), OPTICS_CACHE (shared miepython table).

single-view DOFS at all N angles + depth-reach (correlated vs DIAGONAL S_a, nadir vs full-view)
+ AK-row peaks + per-view weighting functions, over the 9-band superset × N views. The
diagonal-S_a reach is the DIRECT (un-propagated) depth; the gap to the correlated reach is what
the prior extrapolates into the radiatively-shielded base. This is the data behind §15 fig 5.

DEFINITIVE config (matches ic_worker_profile.py; DESIGN §13): 9-band instrument superset, miepython
optics (JAX-Mie retired), OCI 2 % noise (oci_swir), **linearized at the LOO prior MEAN** (truth-
linearization retired), include_base=True (r_e state spans the s=1 base). Index-addressed; degenerate
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
sys.path.insert(0, str(_here))
import runtime_setup                                                 # noqa: E402
runtime_setup.setup()                                              # affinity pin BEFORE JAX
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import osse_config as oc                                            # noqa: E402
from info_content import jacobian_on_ode_grid, flux_jacobian_on_ode_grid  # noqa: E402

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OPTICS_CACHE   = os.environ.get('OPTICS_CACHE',
                                str(_here / 'optics_table_10band_nleg1536_re20.npz'))
RADIANCE_CACHE = os.environ.get('RADIANCE_CACHE', '')             # required; asserted by load_radiance
NQ    = int(os.environ.get('ENSEMBLE_NQUAD', '48'))
BANDS = oc.BANDS
NB    = oc.NB
NVIEW = oc.N_PHYS           # 24 operational retrieval views
NOISE = nm.oci_swir()
s_ref = np.linspace(0.0, 1.0, 6)[:-1]

idx, out = int(sys.argv[1]), sys.argv[2]
profiles = vio.load_all_profiles(DATA)
truth = profiles[idx]
flight = getattr(truth, 'flight', '?')

try:
    if not (0.3 <= float(truth.tau_bot) <= 100.0) or len(np.asarray(truth.tau)) < 5:
        raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f}, npts={len(truth.tau)})")
    opt = oc.load_optics(OPTICS_CACHE)
    t0 = time.time()
    print(f"[{idx}] {flight} tau={truth.tau_bot:.1f}: optics ready, building Jacobian...", flush=True)
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    fwd = oc.build_forward(opt, tau_bot=float(truth.tau_bot), r_base=float(truth.r_base),
                           views='retrieval', jac_mode='fwd',
                           mode_map=os.environ.get('MODE_MAP', 'scan'))
    ts = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(ts)
    x_tru = fwd._encode_state(np.concatenate([                   # physical -> forward state space (log)
        np.interp(s_ref, ts[o], np.asarray(truth.r_e, float)[o]),
        [truth.r_base], [truth.tau_bot]]))
    # LOO prior-MEAN linearization point (truth-linearization retired)
    xa = np.asarray(roe.make_climatology_prior(s_ref, clim)[0])   # physical
    x_lin_phys = np.concatenate([xa[:len(s_ref) + 1], [truth.tau_bot]])
    # STATE-SPACE MATCH (batch-2 null-K fix): forward is LOG-state; a physical x_lin gets
    # exp()'d to ~thousands of µm -> off the [2,20] table -> clamp -> EXACTLY-ZERO Jacobian.
    x_lin = fwd._encode_state(x_lin_phys)
    roe.select_num_modes(fwd, x_lin, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    K_full, s_int = jacobian_on_ode_grid(fwd, x_lin, s_ref, include_base=True)
    print(f"[{idx}] {flight}: radiance Jacobian done in {time.time()-t0:.0f}s "
          f"(n_int={s_int.size}, base incl.); diagnostics...", flush=True)
    n = s_int.size
    s = np.array(s_int)
    s_interior = s[:-1]
    _rrec = oc.load_radiance(RADIANCE_CACHE, idx)                 # cache IS the measurement (sets Se)
    truth_tol = _rrec.get("tol")                                  # cache accuracy tag (gen tol)
    _exp_tol = os.environ.get("RADIANCE_TOL")                     # expected truth tol (optional gate)
    if _exp_tol is not None and truth_tol is not None and abs(truth_tol - float(_exp_tol)) > 1e-12:
        raise ValueError(f"radiance cache tol {truth_tol} != expected RADIANCE_TOL {_exp_tol} "
                         f"— wrong-accuracy cache; refusing (rigor over results).")
    y = oc.select_retrieval_views(np.asarray(_rrec["y"]))
    Se_full = NOISE.Se(y, n_bands=NB)
    Sa_corr = np.asarray(roe.make_climatology_prior(s_interior, clim)[1])[:n, :n]   # LOO prior (r_e+base)
    Sa_diag = np.diag(np.diag(Sa_corr))

    def deepest(df):
        ix = np.where(df >= 0.5)[0]
        return float(s[ix.max()]) if len(ix) else 0.0

    def rows_k(k):
        return [b * NVIEW + k for b in range(NB)]

    sv_dofs = [float(roe.posterior_diagnostics(
        K_full[rows_k(k)], Sa_corr, Se_full[np.ix_(rows_k(k), rows_k(k))]).dofs)
        for k in range(NVIEW)]

    K_flux, _ = flux_jacobian_on_ode_grid(fwd, x_lin, s_ref, include_base=True)
    if not (np.abs(K_full).max() > 0 and np.abs(K_flux).max() > 0):   # rigor: null deliverable = FAIL
        raise ValueError(f"NULL Jacobian (maxabs K_full={np.abs(K_full).max():.2e}, "
                         f"K_flux={np.abs(K_flux).max():.2e}) — linearization off-table / state-space bug.")
    y_flux = fwd.flux_reflectance(x_tru, s_ref)
    Se_flux = NOISE.Se(y_flux, n_bands=NB)
    nadir = np.array(rows_k(0))
    allr = np.arange(K_full.shape[0])

    def reach(K, Se, Sa):
        return deepest(roe.posterior_diagnostics(K, Sa, Se).data_fraction)

    depth = {'albedo_corr': reach(K_flux, Se_flux, Sa_corr),
             'albedo_diag': reach(K_flux, Se_flux, Sa_diag),
             'corr_nadir': reach(K_full[nadir], Se_full[np.ix_(nadir, nadir)], Sa_corr),
             'corr_fullview': reach(K_full[allr], Se_full, Sa_corr),
             'diag_nadir': reach(K_full[nadir], Se_full[np.ix_(nadir, nadir)], Sa_diag),
             'diag_fullview': reach(K_full[allr], Se_full, Sa_diag)}
    A_n = np.asarray(roe.posterior_diagnostics(K_full[nadir], Sa_corr, Se_full[np.ix_(nadir, nadir)]).A)
    A_f = np.asarray(roe.posterior_diagnostics(K_full[allr], Sa_corr, Se_full).A)
    W = np.array([np.abs(K_full[rows_k(k)]).sum(0) for k in range(NVIEW)])
    rec = dict(index=idx, flight=flight, tau_bot=float(truth.tau_bot), n_int=n, NQuad=NQ, n_view=NVIEW,
               bands=BANDS, ic_mode='priormean', radiance_tol=truth_tol,
               s_int=s.tolist(), view_mu=oc.VIEW_MU.tolist(),
               single_view_dofs=sv_dofs, depth=depth,
               ak_peak_nadir=s[np.argmax(np.abs(A_n), 1)].tolist(),
               ak_peak_fullview=s[np.argmax(np.abs(A_f), 1)].tolist(),
               wf_peak_s=s[np.argmax(W, 1)].tolist())
    gc = depth['corr_fullview'] - depth['albedo_corr']
    gd = depth['diag_fullview'] - depth['albedo_diag']
    print(f"[{idx}] {flight} tau={truth.tau_bot:.1f}: depth-gain (fullview over albedo) corr +{gc:.2f} "
          f"vs diag +{gd:.2f} -> {'PRIOR-PROPAGATED' if gd < 0.5 * gc else 'partly DIRECT'}", flush=True)
except Exception as e:                                              # noqa: BLE001
    rec = dict(index=idx, flight=flight,
               tau_bot=float(getattr(truth, 'tau_bot', 0.0)), NQuad=NQ, skipped=str(e)[:160])
    print(f"[{idx}] {flight}: SKIPPED {rec['skipped']}", flush=True)

Path(out).write_text(json.dumps(rec))
