"""QRCP tau_grid trimming prototype.

Usage: python _qrcp_trimming.py <outfile.npz> [--sweep]

Column-pivoted QR on the multi-mode stacked Jacobian J = [J_omega_m; J_g_m]
evaluated at the solver's adaptive tau_grid nodes.  Ranks grid points by
information content for retrieval of optical properties from ToA radiance.

NQuad=16, NFourier=16 (all modes).  Uses GPU if available.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from math import pi
import diffrax
from scipy.linalg import qr as scipy_qr

from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
)
from _solve_bc_riccati_jax import _solve_bc_riccati_jax
from PythonicDISORT import subroutines
from pydisort_riccati_jax import pydisort_riccati_jax

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)

# ================================================================
# Parameters
# ================================================================
tau_bot = 30.0
omega_top, omega_bot = 0.85, 0.96
g_top, g_bot = 0.865, 0.820
NQuad = 16
NLeg = NQuad
NFourier = NQuad
N = NQuad // 2
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tol = 1e-3
sigma_spike = 0.5
d_omega_spike = -0.15
d_g_spike = 0.04

# ================================================================
# Quadrature setup (once)
# ================================================================
mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos = jnp.array(mu_arr_pos_np)
W_jax = jnp.array(W_np)
M_inv = 1.0 / mu_arr_pos
I0_div_4pi = I0 / (4 * pi)


# ================================================================
# Profile constructors
# ================================================================
def make_profiles(tau_spike=None):
    def omega_func(tau):
        base = omega_top + (omega_bot - omega_top) * tau / tau_bot
        if tau_spike is not None:
            base = base + d_omega_spike * jnp.exp(
                -0.5 * ((tau - tau_spike) / sigma_spike)**2)
        return base

    def g_func(tau):
        base = g_top + (g_bot - g_top) * tau / tau_bot
        if tau_spike is not None:
            base = base + d_g_spike * jnp.exp(
                -0.5 * ((tau - tau_spike) / sigma_spike)**2)
        return base

    def Leg_coeffs_func(tau):
        return g_func(tau) ** jnp.arange(NLeg)

    return omega_func, g_func, Leg_coeffs_func


# ================================================================
# AD-traceable Riccati integrator
# ================================================================
def _integrate_final(alpha_func, beta_func, sigma_end, q1_func, q2_func):
    def vf(sigma, state, args):
        R, T, s = state['R'], state['T'], state['s']
        alpha = alpha_func(sigma)
        beta = beta_func(sigma)
        A = alpha + R @ beta
        dR = alpha @ R + R @ alpha + R @ beta @ R + beta
        dT = A @ T
        ds = A @ s + R @ q1_func(sigma) + q2_func(sigma)
        return {'R': dR, 'T': dT, 's': ds}

    y0 = {'R': jnp.zeros((N, N)), 'T': jnp.eye(N), 's': jnp.zeros(N)}
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(),
        t0=0.0, t1=float(sigma_end), dt0=None, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol * 1e-3),
        saveat=diffrax.SaveAt(t1=True), max_steps=4096)
    return sol.ys['R'][0], sol.ys['T'][0], sol.ys['s'][0]


# ================================================================
# Per-mode forward model builder
# ================================================================
def make_solve_m(omega_func, g_func, tau_nodes_jax, m, leg_data_m):
    """Build AD-traceable solver for Fourier mode m."""
    m_eq_0 = (m == 0)

    def solve_m(delta_omega_vec, delta_g_vec):
        def omega_pert(tau):
            return omega_func(tau) + jnp.interp(tau, tau_nodes_jax, delta_omega_vec)
        def Leg_coeffs_pert(tau):
            g_pert = g_func(tau) + jnp.interp(tau, tau_nodes_jax, delta_g_vec)
            return g_pert ** jnp.arange(NLeg)

        alpha_f, beta_f = _make_alpha_beta_funcs_jax(
            omega_pert, Leg_coeffs_pert, m, leg_data_m,
            mu_arr_pos, W_jax, M_inv, N)
        q_up, q_down = _make_q_funcs_jax(
            omega_pert, Leg_coeffs_pert, m, leg_data_m,
            mu_arr_pos, M_inv, mu0, I0_div_4pi, m_eq_0, N)

        tb = float(tau_bot)
        R_up, T_up, s_up = _integrate_final(
            lambda s: alpha_f(tb - s), lambda s: beta_f(tb - s), tb,
            lambda s: q_down(tb - s), lambda s: q_up(tb - s))
        R_dn, T_dn, s_dn = _integrate_final(
            alpha_f, beta_f, tb, q_up, q_down)

        return _solve_bc_riccati_jax(
            R_up, T_up, T_dn, R_dn, s_up, s_dn,
            N, jnp.zeros(N), jnp.zeros(N),
            None, mu_arr_pos, W_jax, m, mu0, I0_div_4pi, tau_bot, True)

    return solve_m


# ================================================================
# QRCP with forced boundary columns
# ================================================================
def qrcp_selection(J, forced_indices):
    """QRCP column selection with forced boundary columns.

    Returns (ordering, R_diag) where ordering[0:len(forced)] are forced,
    then QRCP pivots on the residual.
    """
    M, K = J.shape
    forced = list(forced_indices)

    J_forced = J[:, forced]
    Q_f, _ = np.linalg.qr(J_forced)
    J_proj = J - Q_f @ (Q_f.T @ J)
    J_proj[:, forced] = 0.0

    _, R, piv = scipy_qr(J_proj, pivoting=True)
    R_diag = np.abs(np.diag(R))

    ordering = forced + [int(p) for p in piv if p not in set(forced)]
    return ordering, R_diag


# ================================================================
# Sensitivity ranking (diagnostic)
# ================================================================
def sensitivity_ranking(J):
    norms = np.sqrt(np.sum(J**2, axis=0))
    order = list(np.argsort(-norms))
    return order, norms


# ================================================================
# Analysis
# ================================================================
BIN_EDGES = [0, 5, 10, 15, 20, 25, 30.01]

def analyze_selection(tau_grid, selected, n_show, tau_spike=None, label=""):
    tau_sel = tau_grid[selected[:n_show]]
    counts = np.histogram(tau_sel, bins=BIN_EDGES)[0]

    print(f"\n  Density ({label}, first {n_show} selections):")
    parts = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        tag = f"[{lo:.0f},{hi:.0f})" if hi < 30.01 else f"[{lo:.0f},{tau_grid[-1]:.0f}]"
        parts.append(f"{tag}: {counts[i]}")
    print("    " + "   ".join(parts))

    if tau_spike is not None:
        first_spike_rank = None
        for rank, idx in enumerate(selected):
            if abs(tau_grid[idx] - tau_spike) < 2 * sigma_spike:
                first_spike_rank = rank
                break
        n_in_spike = int(np.sum(np.abs(tau_sel - tau_spike) < 2 * sigma_spike))
        print(f"  Spike region (|tau-{tau_spike:.0f}|<{2*sigma_spike}): "
              f"{n_in_spike}/{n_show} points", end="")
        if first_spike_rank is not None:
            print(f", first at rank {first_spike_rank}")
        else:
            print(f", first at rank >{len(selected)} (not selected)")


# ================================================================
# Run one case
# ================================================================
def run_case(label, tau_spike=None):
    print(f"\n{'=' * 65}")
    spike_str = f", spike at tau={tau_spike}" if tau_spike is not None else ""
    print(f"Case: {label} (NQuad={NQuad}, NFourier={NFourier}{spike_str})")
    print(f"{'=' * 65}")

    # --- Nominal solve ---
    omega_f, g_f, Leg_f = make_profiles(tau_spike=tau_spike)
    t0 = time.time()
    mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
        tau_bot, omega_f, Leg_f, NQuad, mu0, I0, phi0, tol=tol)
    tau_grid = np.asarray(tau_grid)
    K = len(tau_grid)
    print(f"  Nominal solve: {time.time() - t0:.1f}s, {K} grid points")
    n_toa = int(np.sum(tau_grid < 5))
    n_boa = int(np.sum(tau_grid > 25))
    print(f"  Grid: {n_toa} ToA (<5), {n_boa} BoA (>25)")

    tau_nodes_jax = jnp.array(tau_grid)
    zero_K = jnp.zeros(K)

    # --- Per-mode Jacobian loop ---
    J_omega_modes = []
    J_g_modes = []
    t_total = time.time()

    for m in range(NFourier):
        t0 = time.time()
        leg_data_m = _precompute_legendre(m, NLeg, mu_arr_pos, mu0)
        n_ells = NLeg - m
        if n_ells <= 0:
            J_omega_modes.append(np.zeros((N, K)))
            J_g_modes.append(np.zeros((N, K)))
            print(f"  m={m:2d}: skipped (no Legendre terms)", flush=True)
            continue

        solve = make_solve_m(omega_f, g_f, tau_nodes_jax, m, leg_data_m)

        J_om = np.asarray(jax.jacrev(solve, argnums=0)(zero_K, zero_K))
        t_om = time.time() - t0

        t1 = time.time()
        J_gm = np.asarray(jax.jacrev(solve, argnums=1)(zero_K, zero_K))
        t_gm = time.time() - t1

        J_omega_modes.append(J_om)
        J_g_modes.append(J_gm)

        sens_om = np.sqrt(np.sum(J_om**2, axis=0))
        sens_gm = np.sqrt(np.sum(J_gm**2, axis=0))
        print(f"  m={m:2d}: J_omega {t_om:.1f}s, J_g {t_gm:.1f}s  "
              f"||J_om||=[{sens_om.min():.2e},{sens_om.max():.2e}]  "
              f"||J_gm||=[{sens_gm.min():.2e},{sens_gm.max():.2e}]", flush=True)

    t_jac = time.time() - t_total
    print(f"\n  Total Jacobian time: {t_jac:.1f}s")

    # --- Stack ---
    J_omega_full = np.vstack(J_omega_modes)
    J_g_full = np.vstack(J_g_modes)
    J = np.vstack([J_omega_full, J_g_full])
    print(f"  J_full: {J.shape}")

    S_vals = np.linalg.svd(J, compute_uv=False)
    eff_rank_1 = int(np.sum(S_vals > 0.01 * S_vals[0]))
    eff_rank_10 = int(np.sum(S_vals > 0.10 * S_vals[0]))
    print(f"  Effective rank: {eff_rank_1} (>1% of sigma_1), {eff_rank_10} (>10%)")
    print(f"  Top 10 singular values: {', '.join(f'{s:.3e}' for s in S_vals[:10])}")

    # --- Per-angle sensitivity (m=0) ---
    J_omega_m0 = J_omega_modes[0]
    print(f"\n  Per-angle sensitivity (m=0, ||J_omega[i,:]||):")
    for i in range(N):
        norm_i = np.sqrt(np.sum(J_omega_m0[i, :]**2))
        print(f"    mu={float(mu_arr_pos[i]):.4f}: {norm_i:.3e}")

    # --- QRCP selection ---
    print(f"\n  QRCP selection (forced: tau=0.0, tau={tau_grid[-1]:.1f}):")
    t0 = time.time()
    ordering, R_diag = qrcp_selection(J, [0, K - 1])
    print(f"    {time.time() - t0:.2f}s")

    for rank in range(min(30, len(ordering))):
        idx = ordering[rank]
        rd = R_diag[rank] if rank < len(R_diag) else 0.0
        print(f"    rank {rank:2d}: idx={idx:3d}, tau={tau_grid[idx]:7.3f}  |R_diag|={rd:.3e}")

    # --- Sensitivity ranking ---
    sens_order, sens_norms = sensitivity_ranking(J)
    print(f"\n  Sensitivity ranking (||J[:,k]|| — diagnostic, top 10):")
    for rank in range(min(10, len(sens_order))):
        idx = sens_order[rank]
        print(f"    rank {rank:2d}: idx={idx:3d}, tau={tau_grid[idx]:7.3f}, "
              f"||J||={sens_norms[idx]:.3e}")

    dopt_set = set(ordering[:15])
    sens_set = set(sens_order[:15])
    print(f"\n  Top-15 overlap (QRCP vs sensitivity): {len(dopt_set & sens_set)}/15")

    # --- Density analysis ---
    n_show = min(20, len(ordering))
    analyze_selection(tau_grid, ordering, n_show, tau_spike, "QRCP")
    analyze_selection(tau_grid, sens_order, n_show, tau_spike, "sensitivity")

    result = {
        'tau_grid': tau_grid,
        'J_omega_full': J_omega_full,
        'J_g_full': J_g_full,
        'qrcp_order': np.array(ordering),
        'R_diag': R_diag,
        'sens_order': np.array(sens_order),
        'sens_norms': sens_norms,
        'u0': np.asarray(u0),
        'mu_arr': np.asarray(mu_arr),
        'tau_spike': tau_spike if tau_spike is not None else np.nan,
        'singular_values': S_vals,
    }
    for m_idx in range(NFourier):
        result[f'J_omega_m{m_idx}'] = J_omega_modes[m_idx]
        result[f'J_g_m{m_idx}'] = J_g_modes[m_idx]
    return result


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    outfile = sys.argv[1]
    do_sweep = '--sweep' in sys.argv

    results = {}

    results['adiabatic'] = run_case('adiabatic')
    jax.clear_caches()

    results['drizzle'] = run_case('drizzle', tau_spike=15.0)
    jax.clear_caches()

    if do_sweep:
        sweep_positions = [3, 5, 10, 15, 20, 25]
        first_spike_ranks = []

        for ts in sweep_positions:
            key = f'spike_{ts}'
            results[key] = run_case(key, tau_spike=float(ts))
            jax.clear_caches()

            tau_grid = results[key]['tau_grid']
            qrcp_order = results[key]['qrcp_order']
            first_rank = None
            for rank, idx in enumerate(qrcp_order):
                if abs(tau_grid[idx] - ts) < 2 * sigma_spike:
                    first_rank = rank
                    break
            first_spike_ranks.append(first_rank)

        print(f"\n{'=' * 65}")
        print("Spike sweep — first QRCP rank in spike region")
        print(f"{'=' * 65}")
        for ts, rank in zip(sweep_positions, first_spike_ranks):
            if rank is not None:
                print(f"  tau_spike={ts:2d}: rank {rank:2d}")
            else:
                n_sel = len(results[f'spike_{ts}']['qrcp_order'])
                print(f"  tau_spike={ts:2d}: rank >{n_sel} (not selected)")

    # Save
    save_dict = {}
    for key, res in results.items():
        for field, val in res.items():
            save_dict[f'{key}__{field}'] = val
    np.savez(outfile, **save_dict)
    print(f"\nSaved to {outfile}")
