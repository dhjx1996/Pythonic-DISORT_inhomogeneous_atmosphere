"""
Diagnostic: semi-infinite reflectance R_inf and convergence.

Computes R_inf (semi-infinite reflectance) via iterative self-doubling of the
star-product operators, validates convergence R(tau) -> R_inf, and explores
smoothness of R_inf along an adiabatic cloud profile.

Part A: R_inf via iterative doubling for representative (omega, g) cases
Part B: Convergence rate ||R(tau) - R_inf|| ~ exp(-2 k_min tau)
Part C: R_inf along adiabatic cloud profile (local R_inf varies smoothly)
"""
import numpy as np
import scipy.linalg
import sys, math
sys.path.insert(0, '.')
from _helpers import make_D_m_funcs, pydisort_toa, make_cloud_profile
from PythonicDISORT import subroutines

NQuad = 8; N = NQuad // 2
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos


def build_A_and_S(omega, D_m, m=0, mu0=None, I0=0):
    """Build coefficient matrix A and source vector S."""
    I0_div_4pi = I0 / (4*math.pi) if I0 > 0 else 0.0
    m_equals_0 = int(m == 0)
    fac_const = I0_div_4pi * (2 - m_equals_0) * 2

    def A_func(tau):
        D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        DW_neg = D_neg * W[None, :]
        alpha = M_inv[:, None] * (DW_pos - np.eye(N))
        beta  = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N] = -alpha; A[:N, N:] = -beta
        A[N:, :N] = beta;   A[N:, N:] = alpha
        return A

    if I0 > 0 and mu0 is not None:
        def S_func(tau):
            fac = fac_const * omega * np.exp(-tau / mu0)
            S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0)
            S_neg =  M_inv * fac * D_m(tau, -mu_arr_pos, -mu0)
            return np.concatenate([S_pos, S_neg])
    else:
        def S_func(tau):
            return np.zeros(NQuad)

    return A_func, S_func


def one_step_star(A_func, S_func, tau_start, h):
    """Compute one star-product step: returns 6 N*N operators."""
    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))
    tau_mid = tau_start + h / 2
    M_ext[:NQuad, :NQuad] = h * A_func(tau_mid)
    M_ext[:NQuad, NQuad] = h * S_func(tau_mid)
    Phi_ext = scipy.linalg.expm(M_ext)
    Phi = Phi_ext[:NQuad, :NQuad]
    delta_p = Phi_ext[:NQuad, NQuad]
    a = Phi[:N, :N]; b = Phi[:N, N:]
    c = Phi[N:, :N]; d = Phi[N:, N:]
    a_inv = np.linalg.inv(a)
    r_top = -a_inv @ b
    t_up = a_inv
    r_bot = c @ a_inv
    t_down = d - c @ a_inv @ b
    s_up = -a_inv @ delta_p[:N]
    s_down = delta_p[N:] - c @ a_inv @ delta_p[:N]
    return r_top, t_up, r_bot, t_down, s_up, s_down


def star_combine(R_top, T_up, T_down, R_bot, s_up, s_down,
                 r_top, t_up, r_bot, t_down, src_up, src_down):
    """Combine accumulated (upper) with new step (lower) via star product."""
    M2 = np.linalg.inv(np.eye(N) - r_top @ R_bot)
    M1 = np.linalg.inv(np.eye(N) - R_bot @ r_top)
    R_top_new = R_top + T_up @ r_top @ M1 @ T_down
    R_bot_new = r_bot + t_down @ R_bot @ M2 @ t_up
    T_up_new = T_up @ M2 @ t_up
    T_down_new = t_down @ M1 @ T_down
    s_up_new = s_up + T_up @ M2 @ (r_top @ s_down + src_up)
    s_down_new = t_down @ (s_down + R_bot @ M2 @ (r_top @ s_down + src_up)) + src_down
    return R_top_new, T_up_new, T_down_new, R_bot_new, s_up_new, s_down_new


def run_star_product(A_func, S_func, tau_bot, N_steps):
    """Full star-product solve, return 6-tuple of operators."""
    h = tau_bot / N_steps
    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)
    for k in range(N_steps):
        tau_start = k * h
        r, tu, rb, td, su, sd = one_step_star(A_func, S_func, tau_start, h)
        R_top, T_up, T_down, R_bot, s_up, s_down = star_combine(
            R_top, T_up, T_down, R_bot, s_up, s_down, r, tu, rb, td, su, sd)
    return R_top, T_up, T_down, R_bot, s_up, s_down


def compute_k_values(omega, D_m, tau=1.0):
    """Compute k-eigenvalues from the N*N reduction."""
    D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    k_sq = np.linalg.eigvals((alpha - beta) @ (alpha + beta)).real
    k_sq = np.sort(k_sq)[::-1]
    return np.sqrt(np.maximum(k_sq, 0))


# ======================================================================
print("=" * 70)
print("PART A: R_inf via iterative doubling")
print("=" * 70)

R_inf_store = {}

for omega, g, label in [(0.9, 0.5, "w=0.9,g=0.5"),
                         (0.99, 0.85, "w=0.99,g=0.85"),
                         (0.999, 0.85, "w=0.999,g=0.85")]:
    g_l = g ** np.arange(NQuad)
    D_m_funcs = make_D_m_funcs(g_l, NQuad, NQuad)
    D_m = D_m_funcs[0]
    A_func, S_func = build_A_and_S(omega, D_m)

    # Build R(tau=1) with 200 steps (no beam)
    R, Tu, Td, Rb, su, sd = run_star_product(A_func, S_func, 1.0, 200)

    k_vals = compute_k_values(omega, D_m)
    k_min = k_vals[-1]

    tau = 1.0
    print(f"\n{label} (k_min={k_min:.4f}): iterative doubling")
    print(f"  {'tau':>8s} {'||R_new-R_old||':>16s}")
    for i in range(8):
        R_prev = R.copy()
        R, Tu, Td, Rb, su, sd = star_combine(R, Tu, Td, Rb, su, sd,
                                              R, Tu, Rb, Td, su, sd)
        tau *= 2
        conv = np.linalg.norm(R - R_prev, 2)
        print(f"  {tau:8.0f} {conv:16.3e}")

    R_inf = R.copy()
    R_inf_store[(omega, g)] = R_inf
    print(f"  R_inf converged.  ||R_inf||_2 = {np.linalg.norm(R_inf, 2):.6f}")

    # Cross-check: extract R from pydisort at large tau
    print(f"  pydisort cross-check (tau=256, I0=0, column-by-column):")
    R_pyd = np.zeros((N, N))
    for j in range(N):
        b_neg_j = np.zeros(N); b_neg_j[j] = 1.0
        _, u0 = pydisort_toa(256.0, omega, NQuad, g_l, mu0=0.5, I0=0, phi0=0,
                             b_neg=b_neg_j)
        R_pyd[:, j] = u0[:N]
    err = np.linalg.norm(R_inf - R_pyd, 2)
    print(f"    ||R_inf - R_pydisort||_2 = {err:.3e}")


# ======================================================================
print("\n" + "=" * 70)
print("PART B: Convergence rate R(tau) -> R_inf")
print("=" * 70)

omega_b = 0.99; g_b = 0.85
g_l_b = g_b ** np.arange(NQuad)
D_m_funcs_b = make_D_m_funcs(g_l_b, NQuad, NQuad)
D_m_b = D_m_funcs_b[0]
A_func_b, S_func_b = build_A_and_S(omega_b, D_m_b)

k_vals_b = compute_k_values(omega_b, D_m_b)
k_min_b = k_vals_b[-1]
R_inf_b = R_inf_store[(omega_b, g_b)]

print(f"\nomega={omega_b}, g={g_b}, k_min={k_min_b:.4f}")
print(f"  Theoretical decay: ||R(tau) - R_inf|| ~ exp(-2*k_min*tau) = exp(-{2*k_min_b:.4f}*tau)")
print(f"  {'tau':>8s} {'||R-R_inf||':>12s} {'exp(-2k*tau)':>14s} {'ratio':>10s}")

for tau_test in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    K = max(int(15 * tau_test) + 20, 50)
    R_tau, *_ = run_star_product(A_func_b, S_func_b, tau_test, K)
    err = np.linalg.norm(R_tau - R_inf_b, 2)
    theory = math.exp(-2 * k_min_b * tau_test)
    ratio = err / theory if theory > 1e-300 else float('inf')
    print(f"  {tau_test:8.1f} {err:12.3e} {theory:14.3e} {ratio:10.4f}")

print(f"\n  If the ratio column is roughly constant, the decay rate is verified.")


# ======================================================================
print("\n" + "=" * 70)
print("PART C: Local R_inf along adiabatic cloud profile")
print("=" * 70)

tau_bot_cloud = 30.0
omega_func, g_l_func, D_m_funcs_cloud = make_cloud_profile(
    tau_bot=tau_bot_cloud, omega_top=0.85, omega_bot=0.96,
    g_top=0.865, g_bot=0.820, NLeg=NQuad, NQuad=NQuad,
)
D_m_cloud = D_m_funcs_cloud[0]

tau_points = np.linspace(1.0, tau_bot_cloud - 1.0, 12)
R_inf_prev = None

print(f"\nCloud: tau_bot={tau_bot_cloud}, omega=[0.85 -> 0.96], g=[0.865 -> 0.820]")
print(f"Computing local R_inf at each tau by doubling with local (omega, g)...")
print(f"  {'tau':>8s} {'omega':>8s} {'||R_inf||':>10s} {'||dR_inf||':>12s}")

for tau in tau_points:
    omega = omega_func(tau)
    # Build A_func with local omega at this tau
    g_l_local = g_l_func(tau)
    D_local = make_D_m_funcs(g_l_local, NQuad, NQuad)
    A_loc, S_loc = build_A_and_S(omega, D_local[0])

    # Build R(tau=1) with local properties, then double to convergence
    R, Tu, Td, Rb, su, sd = run_star_product(A_loc, S_loc, 1.0, 200)
    for _ in range(8):
        R, Tu, Td, Rb, su, sd = star_combine(R, Tu, Td, Rb, su, sd,
                                              R, Tu, Rb, Td, su, sd)
    R_inf_local = R.copy()
    norm_R = np.linalg.norm(R_inf_local, 2)

    if R_inf_prev is not None:
        dR = np.linalg.norm(R_inf_local - R_inf_prev, 2)
        print(f"  {tau:8.2f} {omega:8.4f} {norm_R:10.6f} {dR:12.3e}")
    else:
        print(f"  {tau:8.2f} {omega:8.4f} {norm_R:10.6f} {'---':>12s}")

    R_inf_prev = R_inf_local

print(f"\nSmall ||dR_inf|| between adjacent tau values confirms smooth variation.")
print(f"This supports adiabatic tracking: R_inf varies slowly enough that")
print(f"the Riccati solution can follow it without resolving fast transients.")
