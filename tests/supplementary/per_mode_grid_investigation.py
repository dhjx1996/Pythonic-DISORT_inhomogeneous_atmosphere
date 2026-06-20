"""Per-mode ODE grids as a retrieval-grid opportunity (OUTSTANDING §G).

Question (user): the candidate **pool** for the retrieval grid is, in production, the
**m=0** ODE grid alone (the solver computes a grid per Fourier mode but returns only
``tau_grid_m0``, discarding m>=1). Do the m>=1 grids place adaptive steps the m=0 grid
*misses*, and would folding them into the pool give a **better retrieval grid**?

Method — three escalating tests, per VOCALS case:
  1. PLACEMENT. Retain every mode's adaptive tau-grid (faithful, forward-sweep-only
     monkeypatch of ``_fourier_solve`` running each mode with ``save_grid=True``;
     production's pool *is* the forward grid, so the backward/BC solves are skipped).
     Report per mode: grid size, ToA reflectance amplitude (``mode_amplitudes``, which
     defines the production Cauchy-K = ``fwd.K_list``), and the near-ToA/mid/deep split.
  2. POOL. Union of the non-negligible modes' grids vs m=0: how many nodes it adds and
     *where in s=tau/tau_bot* (intermediate informative depths vs the universal near-BoA
     imbedding layer).
  3. SELECTION + RECOVERY (decisive). Run the production ``select_retrieval_grid`` on
     the m=0 pool vs the union pool (inject each as ``fwd.ode_grid``). Does QRCP *select*
     a union-only node? Does k or the placement change? Then run the full
     ``gauss_newton_oe`` retrieval on each grid and compare dense RMSE / drop-cap / chi2
     — the end-to-end "is there a real opportunity" answer.

Run ONE case per process (the per-mode save_grid solves each compile a large program;
running cases in one process exhausts the XLA compile cache):

    /tmp/jaxve/bin/python tests/supplementary/per_mode_grid_investigation.py {thin|thick|rf10}
"""
import sys
import gc
import time
from pathlib import Path
from math import pi

import numpy as np
import jax

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import pydisort_riccati_jax as P                                    # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQuad, NLeg_all, v_eff = 16, 128, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.linspace(0.95, 0.25, 8)
view_phi = np.full(view_mu.size, pi)

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


# ---------------------------------------------------------------------------
# Faithful monkeypatch: retain EVERY mode's adaptive **forward** tau-grid (the
# production pool). Copy of pydisort_riccati_jax._fourier_solve, return_grid path,
# with the lax.scan replaced by a Python loop over modes; only the forward sweep
# is run (save_grid=True) — backward/BC are skipped (u_modes -> zeros; ode_grid
# discards u). Per-mode forward solve is numerically identical to production.
# ---------------------------------------------------------------------------
_ORIG_FS = P._fourier_solve
PER_MODE = {}        # {'grids':[np(tau)], 'tau_bot':float}


def _fourier_solve_record(setup, omega_func, Leg_coeffs_func, tau_bot,
                          *, num_modes, return_grid):
    if not return_grid:
        return _ORIG_FS(setup, omega_func, Leg_coeffs_func, tau_bot,
                        num_modes=num_modes, return_grid=False)
    import jax.numpy as jnp
    N, NLeg, K = setup.N, setup.NLeg, num_modes
    if setup.delta_M_scaling:
        f_of_tau = lambda tau: Leg_coeffs_func(tau)[NLeg]
        tau_star_eval, _ = P._compute_tau_star(omega_func, f_of_tau, tau_bot)
    else:
        tau_star_eval = None

    def fwd_grid(wp_m, ap_pos_m, ap_neg_m, amu0_m, mz):
        alpha_func, beta_func = P._make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m,
            setup.W_jax, setup.M_inv, N, NLeg, setup.delta_M_scaling)
        q_up, q_down = P._make_q_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m, amu0_m,
            setup.M_inv, setup.mu0, setup.I0_div_4pi, mz, N, NLeg,
            setup.delta_M_scaling, tau_star_eval)
        _, _, _, tau_grid_m = P._riccati_forward_jax(
            alpha_func, beta_func, tau_bot, N, setup.tol,
            q_up_func=q_up, q_down_func=q_down, save_grid=True,
            max_steps=512, adjoint=setup.adjoint)
        return tau_grid_m

    stacks = (setup.weighted_poch_modes[:K], setup.asso_leg_pos_modes[:K],
              setup.asso_leg_neg_modes[:K], setup.asso_leg_mu0_modes[:K],
              setup.m_is_zero[:K])
    grids = [np.asarray(fwd_grid(*(s[j] for s in stacks)), float)
             for j in range(K)]
    PER_MODE['grids'] = grids
    PER_MODE['tau_bot'] = float(tau_bot)

    tms_data = None
    if setup.NT_cor:
        tms_data = P._precompute_tms(
            omega_func, Leg_coeffs_func, tau_star_eval, tau_bot,
            setup.mu0, setup.phi0, setup.I0_orig_div_4pi, NLeg,
            setup.NLeg_all, setup.NT_quad_order)
    u_modes_arr = jnp.zeros((K, N))
    return P.SolveResult(u_modes=u_modes_arr, tms_data=tms_data,
                         tau_grid=grids[0])


P._fourier_solve = _fourier_solve_record


# ---------------------------------------------------------------------------
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
    Se = roe.make_Se(fwd, y, nm.oci_swir())
    return clim, pb, fwd, s_ref, x_ref, y, Se


def merge_close(tau_sorted, tol):
    keep = [tau_sorted[0]]
    for t in tau_sorted[1:]:
        if t - keep[-1] > tol:
            keep.append(t)
    return np.asarray(keep)


def union_only(s_query, s_ref_pool, tol):
    return np.array([np.min(np.abs(s_ref_pool - sq)) > tol for sq in s_query])


def grid_report(label, truth, bands, sigma_base=None, do_recovery=True):
    clim, pb, fwd, s_ref, x_ref, y, Se = make_fwd(truth, bands, sigma_base)
    Kc = int(max(fwd.K_list))
    tb = float(fwd._split_state(x_ref, s_ref)[2])
    amps = np.asarray(fwd.mode_amplitudes(x_ref, s_ref))[0]          # band-0 per-mode amp

    _ = fwd.ode_grid(x_ref, s_ref)                                   # record (once)
    grids = PER_MODE['grids']
    jax.clear_caches(); gc.collect()                                # free save_grid programs
    s_grids = [np.unique(np.clip(g / tb, 0.0, 1.0)) for g in grids]

    print(f"\n========== {label}: {truth.flight} tau={truth.tau_bot:.2f}, "
          f"m={fwd.m}, K_list={fwd.K_list} (Cauchy K={Kc}) ==========", flush=True)
    print("  --- TEST 1: per-mode grid placement ---", flush=True)
    amax = amps.max()
    print("   mode | size | amp/amp0 | near-ToA(s<.3) | mid(.3-.85) | deep(s>.85)",
          flush=True)
    for m in range(len(grids)):
        sg = s_grids[m]
        ntoa = int(np.count_nonzero(sg < 0.30))
        nmid = int(np.count_nonzero((sg >= 0.30) & (sg <= 0.85)))
        ndeep = int(np.count_nonzero(sg > 0.85))
        flag = "  (>noise)" if m < Kc else ""
        amp_s = f"{amps[m] / amax:8.2e}" if m < len(amps) else "    --  "
        print(f"   m={m:<2d} | {sg.size:4d} | {amp_s} | {ntoa:11d} | "
              f"{nmid:8d} | {ndeep:7d}{flag}", flush=True)

    s0 = s_grids[0]
    merge_tol_s = 0.005
    raw_union = np.unique(np.concatenate(s_grids[:Kc]))
    union = merge_close(raw_union, merge_tol_s)
    extra_mask = union_only(union, s0, merge_tol_s)
    s_extra = union[extra_mask]
    print("  --- TEST 2: union pool (non-negligible modes) vs m=0 ---", flush=True)
    print(f"   |m=0 grid| = {s0.size};  |union merged| = {union.size};  "
          f"union-only nodes = {int(extra_mask.sum())}", flush=True)
    if s_extra.size:
        ex_toa = int(np.count_nonzero(s_extra < 0.30))
        ex_mid = int(np.count_nonzero((s_extra >= 0.30) & (s_extra <= 0.85)))
        ex_deep = int(np.count_nonzero(s_extra > 0.85))
        print(f"   union-only s-locations: near-ToA {ex_toa}, mid {ex_mid}, deep {ex_deep}",
              flush=True)

    print("  --- TEST 3: QRCP selection, m=0 pool vs union pool ---", flush=True)
    m0_abs = grids[0]
    union_abs = union * tb
    _orig_ode = fwd.ode_grid
    fwd.ode_grid = lambda x, s_nodes: m0_abs                         # inject production m=0 pool
    try:
        sA, _, infoA = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                                 prior_builder=pb, filter_threshold=0.5)
        fwd.ode_grid = lambda x, s_nodes: union_abs                  # inject union pool
        sB, _, infoB = roe.select_retrieval_grid(fwd, x_ref, s_ref, None, Se=Se,
                                                 prior_builder=pb, filter_threshold=0.5)
    finally:
        fwd.ode_grid = _orig_ode
    selB_extra = union_only(sB, s0, merge_tol_s)
    print(f"   m=0 pool  : k={infoA['k_active']}, nodes s={np.round(sA, 3).tolist()}",
          flush=True)
    print(f"   union pool: k={infoB['k_active']}, nodes s={np.round(sB, 3).tolist()}",
          flush=True)
    print(f"   union pool selected any union-only node? "
          f"{'YES -> ' + str(np.round(sB[selB_extra], 3).tolist()) if selB_extra.any() else 'no'}",
          flush=True)

    if not do_recovery:
        print("\nDONE", flush=True)
        return

    Se_inv = np.linalg.inv(Se)

    def retrieve(s_grid, tag):
        x_a, Sa = pb(s_grid)
        t0 = time.time()
        res = roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, n_iter=15, lm=1e-2,
                                  xtol=2e-3, max_n_outer=1, prior_builder=pb, warn=False)
        kk = len(res.tau_nodes)
        s = np.asarray(res.tau_nodes)
        _, rb, tbr = fwd._split_state(res.x, s)
        rb, tbr = float(rb), float(tbr)
        sd = np.linspace(0.0, 1.0, 200)
        ret = np.asarray(fwd.profile(res.x, s, sd * tbr))
        tru = np.interp(sd * tbr, truth.tau, truth.r_e)
        rmse = float(np.sqrt(np.mean((ret - tru) ** 2)))
        near = sd > 0.7
        rmse_near = float(np.sqrt(np.mean((ret[near] - tru[near]) ** 2)))
        cap = (float(ret.max() - ret[-1]) / max(float(tru.max() - tru[-1]), 1e-9))
        r0 = res.y - res.Fx
        chi2 = float(r0 @ Se_inv @ r0) / fwd.m
        print(f"   {tag}: k={kk} | RMSE {rmse:.3f} (near-base {rmse_near:.3f}) um | "
              f"r_base {rb:.1f} (truth {truth.r_base:.1f}) | drop cap {cap*100:.0f}% | "
              f"chi2 {chi2:.2f} | tau_bot {tbr:.1f} ({time.time()-t0:.0f}s)", flush=True)

    print("  --- TEST 3b: recovery on each selected grid ---", flush=True)
    retrieve(sA, "m=0 pool  ")
    if not np.array_equal(np.round(sA, 4), np.round(sB, 4)):
        retrieve(sB, "union pool")
    else:
        print("   union grid == m=0 grid -> recovery identical (skipped)", flush=True)
    print("\nDONE", flush=True)


CASES = {
    'thin': lambda: grid_report(
        "THIN  RF11", vio.pick_profile(profiles, target_tau=1.0),
        [1.24, 2.13], do_recovery=True),
    'thick': lambda: grid_report(
        "THICK RF03",
        vio.pick_profile([p for p in profiles if p.flight == 'RF03'], target_tau=23.3),
        [1.24, 1.64, 2.13]),
    'rf10': lambda: grid_report(
        "SUBADIA RF10 (shielded)",
        vio.pick_profile([p for p in profiles if p.flight == 'RF10'], target_tau=4.94),
        [1.24, 1.64, 2.13], sigma_base=8.0),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which in CASES:
        CASES[which]()
    else:
        for fn in CASES.values():       # all (may OOM in one process; prefer per-case)
            fn()
