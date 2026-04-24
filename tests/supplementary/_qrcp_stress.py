"""QRCP rank-4 ceiling stress test — single experiment.

Usage:
  python _qrcp_stress.py <outfile.npz> --atm {absorbing,conservative}
         [--nquad 16] [--tol 1e-3] [--spike-tau 1.0]

Stress-tests whether effective rank > 4 under varied conditions:
  - Tighter ODE tolerance (denser tau_grid)
  - Gaussian spike at ToA (forced grid refinement in [0,5])
  - Higher NQuad/NFourier (finer angular resolution)
"""
import sys, os, time, traceback, argparse
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

# Fixed atmosphere parameters
tau_bot = 30.0
g_top, g_bot = 0.865, 0.820
omega_top_abs, omega_bot_abs = 0.85, 0.96
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
sigma_spike = 0.5
d_omega_spike = -0.15
d_g_spike = 0.04

BIN_EDGES = [0, 5, 10, 15, 20, 25, 30.01]


def qrcp_selection(J, forced_indices):
    forced = list(forced_indices)
    J_forced = J[:, forced]
    Q_f, _ = np.linalg.qr(J_forced)
    J_proj = J - Q_f @ (Q_f.T @ J)
    J_proj[:, forced] = 0.0
    _, R, piv = scipy_qr(J_proj, pivoting=True)
    R_diag = np.abs(np.diag(R))
    ordering = forced + [int(p) for p in piv if p not in set(forced)]
    return ordering, R_diag


def run_experiment(atm, nquad, tol, spike_tau, outfile='/tmp/_qrcp_stress_default.npz', phi_obs=0.0,
                   mode_start=None, mode_end=None):
    is_conservative = (atm == 'conservative')
    N = nquad // 2
    NLeg = nquad
    NFourier = nquad
    I0_div_4pi = I0 / (4 * pi)
    if mode_start is None:
        mode_start = 0
    if mode_end is None:
        mode_end = NFourier
    partial = (mode_start != 0 or mode_end != NFourier)

    label = f"{atm}, NQuad={nquad}, tol={tol:.0e}"
    if spike_tau is not None:
        label += f", spike@tau={spike_tau}"
    if partial:
        label += f", modes {mode_start}-{mode_end-1}"
    print(f"\n{'=' * 70}")
    print(f"Experiment: {label}")
    print(f"{'=' * 70}", flush=True)

    # Quadrature
    mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
    mu_arr_pos = jnp.array(mu_arr_pos_np)
    W_jax = jnp.array(W_np)
    M_inv = 1.0 / mu_arr_pos

    # Profiles
    if is_conservative:
        def omega_func(tau):
            return 1.0 + 0.0 * tau
    else:
        if spike_tau is not None:
            def omega_func(tau):
                return (omega_top_abs + (omega_bot_abs - omega_top_abs) * tau / tau_bot
                        + d_omega_spike * jnp.exp(-0.5 * ((tau - spike_tau) / sigma_spike)**2))
        else:
            def omega_func(tau):
                return omega_top_abs + (omega_bot_abs - omega_top_abs) * tau / tau_bot

    if spike_tau is not None:
        def g_func(tau):
            return (g_top + (g_bot - g_top) * tau / tau_bot
                    + d_g_spike * jnp.exp(-0.5 * ((tau - spike_tau) / sigma_spike)**2))
    else:
        def g_func(tau):
            return g_top + (g_bot - g_top) * tau / tau_bot

    def Leg_coeffs_func(tau):
        return g_func(tau) ** jnp.arange(NLeg)

    # Riccati integrator
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
            saveat=diffrax.SaveAt(t1=True), max_steps=16384)
        return sol.ys['R'][0], sol.ys['T'][0], sol.ys['s'][0]

    # Nominal solve
    t0 = time.time()
    fallback_omega = None
    try:
        mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
            tau_bot, omega_func, Leg_coeffs_func, nquad, mu0, I0, phi0, tol=tol)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc()
        if is_conservative:
            print("  Falling back to omega=0.999 ...", flush=True)
            fallback_omega = 0.999
            def omega_func(tau):
                return 0.999 + 0.0 * tau
            def Leg_coeffs_func(tau):
                return g_func(tau) ** jnp.arange(NLeg)
            mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
                tau_bot, omega_func, Leg_coeffs_func, nquad, mu0, I0, phi0, tol=tol)
        else:
            raise

    tau_grid = np.asarray(tau_grid)
    K = len(tau_grid)
    print(f"  Nominal solve: {time.time() - t0:.1f}s, K={K} grid points")
    n_toa = int(np.sum(tau_grid < 5))
    n_boa = int(np.sum(tau_grid > 25))
    print(f"  Grid: {n_toa} ToA (<5), {n_boa} BoA (>25)", flush=True)

    tau_nodes_jax = jnp.array(tau_grid)
    zero_K = jnp.zeros(K)

    # Per-mode forward model
    def make_solve_m(m, leg_data_m):
        m_eq_0 = (m == 0)
        if is_conservative:
            def solve_m(delta_g_vec):
                def Leg_coeffs_pert(tau):
                    g_pert = g_func(tau) + jnp.interp(tau, tau_nodes_jax, delta_g_vec)
                    return g_pert ** jnp.arange(NLeg)
                alpha_f, beta_f = _make_alpha_beta_funcs_jax(
                    omega_func, Leg_coeffs_pert, m, leg_data_m,
                    mu_arr_pos, W_jax, M_inv, N)
                q_up, q_down = _make_q_funcs_jax(
                    omega_func, Leg_coeffs_pert, m, leg_data_m,
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
        else:
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

    # Per-mode Jacobian (with incremental checkpointing)
    J_g_dict = {}
    J_omega_dict = {} if not is_conservative else None
    t_total = time.time()

    ckpt_path = outfile + '.ckpt.npz'
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        for key in ckpt.files:
            if key.startswith('J_g_m'):
                J_g_dict[key] = ckpt[key]
            elif key.startswith('J_omega_m') and not is_conservative:
                J_omega_dict[key] = ckpt[key]
        loaded = [k for k in sorted(J_g_dict) if k.startswith('J_g_m')]
        print(f"  Resumed from checkpoint: {len(loaded)} modes loaded", flush=True)

    for m in range(mode_start, mode_end):
        if f'J_g_m{m}' in J_g_dict:
            print(f"  m={m:2d}: already in checkpoint, skipping", flush=True)
            continue
        t0 = time.time()
        leg_data_m = _precompute_legendre(m, NLeg, mu_arr_pos, mu0)
        n_ells = NLeg - m
        if n_ells <= 0:
            J_g_dict[f'J_g_m{m}'] = np.zeros((N, K))
            if not is_conservative:
                J_omega_dict[f'J_omega_m{m}'] = np.zeros((N, K))
            print(f"  m={m:2d}: skipped", flush=True)
        else:
            solve = make_solve_m(m, leg_data_m)

            if is_conservative:
                J_gm = np.asarray(jax.jacrev(solve)(zero_K))
                dt = time.time() - t0
                J_g_dict[f'J_g_m{m}'] = J_gm
                sens = np.sqrt(np.sum(J_gm**2, axis=0))
                print(f"  m={m:2d}: {dt:.1f}s  ||J_g||=[{sens.min():.2e},{sens.max():.2e}]", flush=True)
            else:
                J_om = np.asarray(jax.jacrev(solve, argnums=0)(zero_K, zero_K))
                dt0 = time.time() - t0
                t1 = time.time()
                J_gm = np.asarray(jax.jacrev(solve, argnums=1)(zero_K, zero_K))
                dt1 = time.time() - t1
                J_omega_dict[f'J_omega_m{m}'] = J_om
                J_g_dict[f'J_g_m{m}'] = J_gm
                print(f"  m={m:2d}: J_om {dt0:.1f}s, J_g {dt1:.1f}s", flush=True)

        # Incremental checkpoint
        ckpt_dict = {'tau_grid': tau_grid, 'nquad': np.array(nquad),
                     'tol': np.array(tol), 'u0': np.asarray(u0),
                     'mu_arr': np.asarray(mu_arr)}
        ckpt_dict.update(J_g_dict)
        if not is_conservative:
            ckpt_dict.update(J_omega_dict)
        np.savez(ckpt_path, **ckpt_dict)
        print(f"    [checkpoint saved: {len(J_g_dict)}/{NFourier} modes]", flush=True)

    t_jac = time.time() - t_total
    print(f"\n  Jacobian time (modes {mode_start}-{mode_end-1}): {t_jac:.1f}s")

    # Build ordered mode lists from dict
    J_g_modes = [J_g_dict[f'J_g_m{m}'] for m in range(NFourier) if f'J_g_m{m}' in J_g_dict]
    have_all = (len(J_g_modes) == NFourier)

    if partial and not have_all:
        print(f"\n  Partial run: {len(J_g_modes)}/{NFourier} modes. "
              f"Merge with other partial to get full analysis.", flush=True)
        result = {'tau_grid': tau_grid, 'nquad': np.array(nquad),
                  'tol': np.array(tol), 'u0': np.asarray(u0),
                  'mu_arr': np.asarray(mu_arr),
                  'spike_tau': np.array(spike_tau if spike_tau is not None else np.nan),
                  'fallback_omega': np.array(fallback_omega if fallback_omega is not None else np.nan)}
        result.update(J_g_dict)
        if not is_conservative:
            result.update(J_omega_dict)
        return result

    # Physical Jacobian: cosine-weighted sum over Fourier modes
    J_g_phys = J_g_dict['J_g_m0'].copy()
    for m in range(1, NFourier):
        w_m = 2.0 * np.cos(m * (phi_obs - phi0))
        J_g_phys += w_m * J_g_dict[f'J_g_m{m}']

    if is_conservative:
        J = J_g_phys
    else:
        J_omega_phys = J_omega_dict['J_omega_m0'].copy()
        for m in range(1, NFourier):
            w_m = 2.0 * np.cos(m * (phi_obs - phi0))
            J_omega_phys += w_m * J_omega_dict[f'J_omega_m{m}']
        J = np.vstack([J_omega_phys, J_g_phys])
    print(f"  J_physical (phi={phi_obs:.2f}): {J.shape}")

    # SVD
    S_vals = np.linalg.svd(J, compute_uv=False)
    eff_rank = int(np.sum(S_vals > 0.01 * S_vals[0]))
    print(f"  Effective rank (>1% of sigma_1): {eff_rank}")
    n_show = min(15, len(S_vals))
    print(f"  Top {n_show} SVs: {', '.join(f'{s:.3e}' for s in S_vals[:n_show])}")

    # Per-mode BoA sensitivity
    print(f"\n  Per-mode BoA sensitivity (||J_g_m[:,-1]||):")
    boa_norms = []
    for m_idx in range(NFourier):
        J_m = J_g_dict[f'J_g_m{m_idx}']
        boa_norms.append(np.sqrt(np.sum(J_m[:, -1]**2)))
    for m_idx in range(NFourier):
        end = "   " if (m_idx + 1) % 4 != 0 else "\n"
        print(f"    m={m_idx:2d}: {boa_norms[m_idx]:.2e}", end=end, flush=True)
    if NFourier % 4 != 0:
        print()

    # QRCP
    print(f"\n  QRCP selection (forced: tau=0.0, tau={tau_grid[-1]:.1f}):")
    ordering, R_diag = qrcp_selection(J, [0, K - 1])
    for rank in range(min(25, len(ordering))):
        idx = ordering[rank]
        rd = R_diag[rank] if rank < len(R_diag) else 0.0
        print(f"    rank {rank:2d}: idx={idx:3d}, tau={tau_grid[idx]:7.3f}  |R_diag|={rd:.3e}")

    # Density
    n_sel = min(20, len(ordering))
    tau_sel = tau_grid[ordering[:n_sel]]
    counts = np.histogram(tau_sel, bins=BIN_EDGES)[0]
    print(f"\n  Density (QRCP, first {n_sel}):")
    parts = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        tag = f"[{lo:.0f},{hi:.0f})" if hi < 30.01 else f"[{lo:.0f},{tau_grid[-1]:.0f}]"
        parts.append(f"{tag}: {counts[i]}")
    print("    " + "   ".join(parts))

    # tau_grid in [0,5] detail
    toa_mask = tau_grid < 5
    toa_taus = tau_grid[toa_mask]
    print(f"\n  tau_grid nodes in [0,5] ({len(toa_taus)} points):")
    print(f"    {', '.join(f'{t:.3f}' for t in toa_taus)}")

    print(f"\n  SUMMARY: atm={atm}, NQuad={nquad}, tol={tol:.0e}, "
          f"spike={spike_tau}, K={K}, eff_rank={eff_rank}", flush=True)

    result = {
        'tau_grid': tau_grid,
        'singular_values': S_vals,
        'eff_rank': np.array(eff_rank),
        'qrcp_order': np.array(ordering),
        'R_diag': R_diag,
        'boa_norms': np.array(boa_norms),
        'J_shape': np.array(J.shape),
        'J_g_phys': J_g_phys,
        'phi_obs': np.array(phi_obs),
        'nquad': np.array(nquad),
        'tol': np.array(tol),
        'spike_tau': np.array(spike_tau if spike_tau is not None else np.nan),
        'fallback_omega': np.array(fallback_omega if fallback_omega is not None else np.nan),
        'u0': np.asarray(u0),
        'mu_arr': np.asarray(mu_arr),
    }
    result.update(J_g_dict)
    if not is_conservative:
        result['J_omega_phys'] = J_omega_phys
        result.update(J_omega_dict)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QRCP rank-4 stress test')
    parser.add_argument('outfile')
    parser.add_argument('--atm', choices=['absorbing', 'conservative'], required=True)
    parser.add_argument('--nquad', type=int, default=16)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--spike-tau', type=float, default=None)
    parser.add_argument('--phi', type=float, default=0.0,
                        help='Observation azimuth (default: 0.0, principal plane)')
    parser.add_argument('--mode-start', type=int, default=None)
    parser.add_argument('--mode-end', type=int, default=None)
    args = parser.parse_args()

    results = run_experiment(args.atm, args.nquad, args.tol, args.spike_tau, args.outfile, args.phi,
                            args.mode_start, args.mode_end)
    # Merge with existing partial results if present
    if os.path.exists(args.outfile):
        existing = dict(np.load(args.outfile, allow_pickle=True))
        for key, val in results.items():
            existing[key] = val
        results = existing
    np.savez(args.outfile, **results)
    print(f"\nSaved to {args.outfile}")
    ckpt_path = args.outfile + '.ckpt.npz'
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  Checkpoint removed: {ckpt_path}")
