"""QRCP tau_grid trimming — conservative atmosphere (omega=1.0), J_g only.

Usage: python _qrcp_conservative.py <outfile.npz>

Column-pivoted QR on the multi-mode stacked Jacobian J_g = [J_g_m0; ...; J_g_m15]
evaluated at the solver's adaptive tau_grid nodes.  Tests whether perfectly
conservative scattering increases effective rank compared to the absorbing case.

NQuad=16, NFourier=16 (all modes).  Uses GPU if available.
"""
import sys, os, time, traceback
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
omega_val = 1.0
g_top, g_bot = 0.865, 0.820
NQuad = 16
NLeg = NQuad
NFourier = NQuad
N = NQuad // 2
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tol = 1e-3

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
def make_profiles():
    def omega_func(tau):
        return omega_val + 0.0 * tau

    def g_func(tau):
        return g_top + (g_bot - g_top) * tau / tau_bot

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
# Per-mode forward model builder (J_g only — single argument)
# ================================================================
def make_solve_m(omega_func, g_func, tau_nodes_jax, m, leg_data_m):
    m_eq_0 = (m == 0)

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

    return solve_m


# ================================================================
# QRCP with forced boundary columns
# ================================================================
def qrcp_selection(J, forced_indices):
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

def analyze_selection(tau_grid, selected, n_show, label=""):
    tau_sel = tau_grid[selected[:n_show]]
    counts = np.histogram(tau_sel, bins=BIN_EDGES)[0]

    print(f"\n  Density ({label}, first {n_show} selections):")
    parts = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        tag = f"[{lo:.0f},{hi:.0f})" if hi < 30.01 else f"[{lo:.0f},{tau_grid[-1]:.0f}]"
        parts.append(f"{tag}: {counts[i]}")
    print("    " + "   ".join(parts))


# ================================================================
# Run one case
# ================================================================
def run_case(label):
    global omega_val

    print(f"\n{'=' * 65}")
    print(f"Case: {label} (NQuad={NQuad}, NFourier={NFourier}, omega={omega_val})")
    print(f"{'=' * 65}")

    # --- Nominal solve ---
    omega_f, g_f, Leg_f = make_profiles()
    t0 = time.time()
    try:
        mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
            tau_bot, omega_f, Leg_f, NQuad, mu0, I0, phi0, tol=tol)
    except Exception as e:
        print(f"  FAILED at omega={omega_val}: {e}", flush=True)
        traceback.print_exc()
        if omega_val == 1.0:
            print(f"\n  Falling back to omega=0.999 ...", flush=True)
            omega_val = 0.999
            omega_f, g_f, Leg_f = make_profiles()
            mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
                tau_bot, omega_f, Leg_f, NQuad, mu0, I0, phi0, tol=tol)
        else:
            raise

    tau_grid = np.asarray(tau_grid)
    K = len(tau_grid)
    print(f"  Nominal solve: {time.time() - t0:.1f}s, {K} grid points")
    n_toa = int(np.sum(tau_grid < 5))
    n_boa = int(np.sum(tau_grid > 25))
    print(f"  Grid: {n_toa} ToA (<5), {n_boa} BoA (>25)")

    tau_nodes_jax = jnp.array(tau_grid)
    zero_K = jnp.zeros(K)

    # --- Per-mode Jacobian loop (J_g only) ---
    J_g_modes = []
    t_total = time.time()

    for m in range(NFourier):
        t0 = time.time()
        leg_data_m = _precompute_legendre(m, NLeg, mu_arr_pos, mu0)
        n_ells = NLeg - m
        if n_ells <= 0:
            J_g_modes.append(np.zeros((N, K)))
            print(f"  m={m:2d}: skipped (no Legendre terms)", flush=True)
            continue

        solve = make_solve_m(omega_f, g_f, tau_nodes_jax, m, leg_data_m)

        J_gm = np.asarray(jax.jacrev(solve)(zero_K))
        t_gm = time.time() - t0

        J_g_modes.append(J_gm)

        sens_gm = np.sqrt(np.sum(J_gm**2, axis=0))
        print(f"  m={m:2d}: J_g {t_gm:.1f}s  "
              f"||J_gm||=[{sens_gm.min():.2e},{sens_gm.max():.2e}]", flush=True)

    t_jac = time.time() - t_total
    print(f"\n  Total Jacobian time: {t_jac:.1f}s")

    # --- Stack (J_g only) ---
    J = np.vstack(J_g_modes)
    print(f"  J_full: {J.shape}")

    S_vals = np.linalg.svd(J, compute_uv=False)
    eff_rank_1 = int(np.sum(S_vals > 0.01 * S_vals[0]))
    eff_rank_10 = int(np.sum(S_vals > 0.10 * S_vals[0]))
    print(f"  Effective rank: {eff_rank_1} (>1% of sigma_1), {eff_rank_10} (>10%)")
    print(f"  Top 10 singular values: {', '.join(f'{s:.3e}' for s in S_vals[:10])}")

    # --- Per-mode BoA sensitivity ---
    print(f"\n  Per-mode BoA sensitivity (||J_g_m[:,-1]||):")
    boa_norms = []
    for m_idx, J_m in enumerate(J_g_modes):
        boa_norm = np.sqrt(np.sum(J_m[:, -1]**2))
        boa_norms.append(boa_norm)
    for m_idx in range(NFourier):
        end = "   " if (m_idx + 1) % 4 != 0 else "\n"
        print(f"    m={m_idx:2d}: {boa_norms[m_idx]:.2e}", end=end, flush=True)
    if NFourier % 4 != 0:
        print()

    # --- Cosine similarity with BoA column ---
    print(f"\n  Cosine similarity with BoA column (full J):")
    boa_col = J[:, -1]
    boa_col_norm = np.linalg.norm(boa_col)
    cos_sim = np.zeros(K)
    for j in range(K):
        col = J[:, j]
        cos_sim[j] = abs(np.dot(col, boa_col)) / (np.linalg.norm(col) * boa_col_norm + 1e-30)
    tau_np = tau_grid
    regions = [
        ("[0,5)",     tau_np < 5),
        ("[5,10)",    (tau_np >= 5) & (tau_np < 10)),
        ("[10,15)",   (tau_np >= 10) & (tau_np < 15)),
        ("[15,20)",   (tau_np >= 15) & (tau_np < 20)),
        ("[20,25)",   (tau_np >= 20) & (tau_np < 25)),
        ("[25,30]",   tau_np >= 25),
    ]
    for name, mask in regions:
        v = cos_sim[mask]
        if len(v) > 0:
            print(f"    {name}: mean={v.mean():.6f}  min={v.min():.6f}  max={v.max():.6f}")

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
    analyze_selection(tau_grid, ordering, n_show, "QRCP")
    analyze_selection(tau_grid, sens_order, n_show, "sensitivity")

    return {
        'tau_grid': tau_grid,
        'J_g_full': J,
        'qrcp_order': np.array(ordering),
        'R_diag': R_diag,
        'sens_order': np.array(sens_order),
        'sens_norms': sens_norms,
        'u0': np.asarray(u0),
        'mu_arr': np.asarray(mu_arr),
        'singular_values': S_vals,
        'omega_val': omega_val,
        'boa_norms_per_mode': np.array(boa_norms),
        'cos_sim': cos_sim,
    }


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    outfile = sys.argv[1]

    results = run_case('conservative')

    save_dict = {}
    for field, val in results.items():
        save_dict[field] = val
    # Save per-mode Jacobians
    J_g_full = results['J_g_full']
    K = len(results['tau_grid'])
    for m_idx in range(NFourier):
        save_dict[f'J_g_m{m_idx}'] = J_g_full[m_idx * N:(m_idx + 1) * N, :]
    np.savez(outfile, **save_dict)
    print(f"\nSaved to {outfile}")
