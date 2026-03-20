"""
Diagnostic: investigate adaptive stepping for the star-product approach.
1. Maximum step size before block extraction loses accuracy
2. Higher-order Magnus (4th order commutator-free) step savings
3. Local error estimation via step doubling
4. Error distribution across the domain
"""
import numpy as np
import scipy.linalg
import sys, math
sys.path.insert(0, '.')
from _helpers import make_D_m_funcs, pydisort_toa
from PythonicDISORT import subroutines

NQuad = 8; N = NQuad // 2
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos
g_l = np.zeros(NQuad); g_l[0] = 1.0
D_m_funcs = make_D_m_funcs(g_l, NQuad, NQuad)
mu0 = 0.1; I0 = math.pi / mu0; phi0 = math.pi


def build_A_and_S(omega, D_m, m):
    def A_func(tau):
        D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        DW_neg = D_neg * W[None, :]
        alpha = M_inv[:, None] * (DW_pos - np.eye(N))
        beta = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N] = -alpha; A[:N, N:] = -beta
        A[N:, :N] = beta; A[N:, N:] = alpha
        return A

    I0_div_4pi = I0 / (4*math.pi)
    fac_const = I0_div_4pi * (2 - 1) * 2  # m=0: (2 - delta_{m0}) * 2

    def S_func(tau):
        fac = fac_const * omega * np.exp(-tau / mu0)
        S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0)
        S_neg = M_inv * fac * D_m(tau, -mu_arr_pos, -mu0)
        return np.concatenate([S_pos, S_neg])

    return A_func, S_func


def one_step_star(A_func, S_func, tau_start, h, NQuad, N):
    """Compute one star-product step: returns (r_top, t_up, r_bot, t_down, s_up, s_down)."""
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
                 r_top, t_up, r_bot, t_down, src_up, src_down, N):
    """Combine accumulated (1) with new step (2) via star product."""
    M2 = np.linalg.inv(np.eye(N) - r_top @ R_bot)
    M1 = np.linalg.inv(np.eye(N) - R_bot @ r_top)
    R_top_new = R_top + T_up @ r_top @ M1 @ T_down
    R_bot_new = r_bot + t_down @ R_bot @ M2 @ t_up
    T_up_new = T_up @ M2 @ t_up
    T_down_new = t_down @ M1 @ T_down
    s_up_new = s_up + T_up @ M2 @ (r_top @ s_down + src_up)
    s_down_new = t_down @ (s_down + R_bot @ M2 @ (r_top @ s_down + src_up)) + src_down
    return R_top_new, T_up_new, T_down_new, R_bot_new, s_up_new, s_down_new


def run_star_product(A_func, S_func, tau_bot, N_steps, N, NQuad):
    """Full star-product solve, return I+(0)."""
    h = tau_bot / N_steps
    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)
    for k in range(N_steps):
        tau_start = k * h
        r, tu, rb, td, su, sd = one_step_star(A_func, S_func, tau_start, h, NQuad, N)
        R_top, T_up, T_down, R_bot, s_up, s_down = star_combine(
            R_top, T_up, T_down, R_bot, s_up, s_down, r, tu, rb, td, su, sd, N)
    return s_up  # I+(0) for b_neg=0, b_pos=0


def one_step_star_order4(A_func, S_func, tau_start, h, NQuad, N):
    """
    4th-order commutator-free Magnus step.
    Uses two Gauss-Legendre quadrature points to approximate Omega.
    Omega = h * (w1*A(tau1) + w2*A(tau2))  -- but this is only 2nd order.

    For TRUE 4th order, use the CF4 (commutator-free) method:
    Phi = expm(h * (a1*A1 + a2*A2)) @ expm(h * (a2*A1 + a1*A2))
    where A1 = A(tau + c1*h), A2 = A(tau + c2*h),
    c1 = 1/2 - sqrt(3)/6, c2 = 1/2 + sqrt(3)/6,
    a1 = 1/4 + sqrt(3)/6, a2 = 1/4 - sqrt(3)/6.
    """
    c1 = 0.5 - math.sqrt(3)/6
    c2 = 0.5 + math.sqrt(3)/6
    a1 = 0.25 + math.sqrt(3)/6
    a2 = 0.25 - math.sqrt(3)/6

    A1 = A_func(tau_start + c1*h)
    A2 = A_func(tau_start + c2*h)
    S1 = S_func(tau_start + c1*h)
    S2 = S_func(tau_start + c2*h)

    ext = NQuad + 1

    # First exponential
    M1 = np.zeros((ext, ext))
    M1[:NQuad, :NQuad] = h * (a1*A1 + a2*A2)
    M1[:NQuad, NQuad] = h * (a1*S1 + a2*S2)
    Phi1_ext = scipy.linalg.expm(M1)

    # Second exponential
    M2 = np.zeros((ext, ext))
    M2[:NQuad, :NQuad] = h * (a2*A1 + a1*A2)
    M2[:NQuad, NQuad] = h * (a2*S1 + a1*S2)
    Phi2_ext = scipy.linalg.expm(M2)

    # Combined: Phi = Phi2 @ Phi1 (order matters)
    Phi_ext = Phi2_ext @ Phi1_ext
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


def run_star_product_order4(A_func, S_func, tau_bot, N_steps, N, NQuad):
    """Full star-product solve with 4th-order Magnus."""
    h = tau_bot / N_steps
    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)
    for k in range(N_steps):
        tau_start = k * h
        r, tu, rb, td, su, sd = one_step_star_order4(A_func, S_func, tau_start, h, NQuad, N)
        R_top, T_up, T_down, R_bot, s_up, s_down = star_combine(
            R_top, T_up, T_down, R_bot, s_up, s_down, r, tu, rb, td, su, sd, N)
    return s_up


# =========================================================================
print("=" * 70)
print("PART 1: Maximum step size -- cond(a) vs h*lambda_max")
print("=" * 70)

for omega in [0.2, 0.5, 0.9, 0.99]:
    A_func, S_func = build_A_and_S(omega, D_m_funcs[0], m=0)
    A = A_func(1.0)
    lam_max = max(np.linalg.eigvals(A).real)
    print(f"\nomega={omega}, lambda_max={lam_max:.2f}")
    print(f"  {'h':>8s} {'lam*h':>8s} {'cond(a)':>10s} {'cond(Phi)':>10s} {'||r_top||':>10s}")
    for h in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        Phi = scipy.linalg.expm(A * h)
        a = Phi[:N, :N]
        cond_a = np.linalg.cond(a)
        cond_Phi = np.linalg.cond(Phi)
        a_inv = np.linalg.inv(a)
        r_top = -a_inv @ Phi[:N, N:]
        print(f"  {h:8.3f} {lam_max*h:8.3f} {cond_a:10.2e} {cond_Phi:10.2e} {np.linalg.norm(r_top,2):10.4f}")


# =========================================================================
print("\n" + "=" * 70)
print("PART 2: 4th-order Magnus convergence comparison")
print("=" * 70)

test_cases = [
    (5.0, 0.5, "2c"),
    (32.0, 0.2, "1d"),
    (32.0, 0.99, "1f"),
]

for tau_bot, omega, label in test_cases:
    A_func, S_func = build_A_and_S(omega, D_m_funcs[0], m=0)
    flux_ref, u0_ref = pydisort_toa(tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u0_pos_ref = u0_ref[:N]

    print(f"\n{label} (tau={tau_bot}, omega={omega}), ref flux={flux_ref:.6e}")
    print(f"  {'K':>6s} {'Order':>6s} {'flux':>13s} {'rel_err':>10s} {'ratio':>8s} {'expm_calls':>10s}")

    # 2nd order
    prev_err = None
    for K in [25, 50, 100, 200, 400]:
        u_pos = run_star_product(A_func, S_func, tau_bot, K, N, NQuad)
        flux = 2 * math.pi * np.dot(mu_arr_pos * W, u_pos)
        err = abs(flux - flux_ref) / max(abs(flux_ref), 1e-10)
        ratio = prev_err / err if prev_err and err > 0 else float('nan')
        print(f"  {K:6d} {'2nd':>6s} {flux:13.6e} {err:10.3e} {ratio:8.2f} {K:10d}")
        prev_err = err

    # 4th order
    prev_err = None
    for K in [10, 20, 40, 80, 160]:
        u_pos = run_star_product_order4(A_func, S_func, tau_bot, K, N, NQuad)
        flux = 2 * math.pi * np.dot(mu_arr_pos * W, u_pos)
        err = abs(flux - flux_ref) / max(abs(flux_ref), 1e-10)
        ratio = prev_err / err if prev_err and err > 0 else float('nan')
        print(f"  {K:6d} {'4th':>6s} {flux:13.6e} {err:10.3e} {ratio:8.2f} {2*K:10d}")
        prev_err = err


# =========================================================================
print("\n" + "=" * 70)
print("PART 3: Error distribution -- where does the error concentrate?")
print("(Compare full domain vs first-half vs second-half accuracy)")
print("=" * 70)

omega = 0.5; tau_bot = 5.0
A_func, S_func = build_A_and_S(omega, D_m_funcs[0], m=0)
flux_ref, _ = pydisort_toa(tau_bot, omega, NQuad, g_l, mu0, I0, phi0)

# Step-doubling error estimator: compare K steps vs K/2 steps
print(f"\ntau={tau_bot}, omega={omega}")
print(f"  {'K':>6s} {'flux_K':>13s} {'flux_K/2':>13s} {'|diff|/|flux|':>14s} {'true_err_K':>11s}")
for K in [50, 100, 200, 400]:
    u_K = run_star_product(A_func, S_func, tau_bot, K, N, NQuad)
    u_K2 = run_star_product(A_func, S_func, tau_bot, K//2, N, NQuad)
    flux_K = 2 * math.pi * np.dot(mu_arr_pos * W, u_K)
    flux_K2 = 2 * math.pi * np.dot(mu_arr_pos * W, u_K2)
    est_err = abs(flux_K - flux_K2) / max(abs(flux_K), 1e-10)
    true_err = abs(flux_K - flux_ref) / max(abs(flux_ref), 1e-10)
    print(f"  {K:6d} {flux_K:13.6e} {flux_K2:13.6e} {est_err:14.3e} {true_err:11.3e}")


# =========================================================================
print("\n" + "=" * 70)
print("PART 4: Step-doubling as local error estimator")
print("Compare one step of size h vs two steps of size h/2")
print("=" * 70)

omega = 0.5; tau_bot = 5.0
A_func, S_func = build_A_and_S(omega, D_m_funcs[0], m=0)

# For each test h, compare the R,T from one step of h vs two steps of h/2
print(f"\ntau_start=0.0, omega={omega}")
print(f"  {'h':>8s} {'||R1-R2||/||R2||':>16s} {'||T1-T2||/||T2||':>16s} {'||s1-s2||':>12s}")
for h in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    # One step of size h
    r1, tu1, rb1, td1, su1, sd1 = one_step_star(A_func, S_func, 0.0, h, NQuad, N)

    # Two steps of size h/2
    r2a, tu2a, rb2a, td2a, su2a, sd2a = one_step_star(A_func, S_func, 0.0, h/2, NQuad, N)
    r2b, tu2b, rb2b, td2b, su2b, sd2b = one_step_star(A_func, S_func, h/2, h/2, NQuad, N)
    # Combine
    R2, TU2, TD2, RB2, SU2, SD2 = star_combine(
        r2a, tu2a, td2a, rb2a, su2a, sd2a,
        r2b, tu2b, rb2b, td2b, su2b, sd2b, N)

    r_err = np.linalg.norm(r1 - R2, 2) / max(np.linalg.norm(R2, 2), 1e-15)
    t_err = np.linalg.norm(tu1 - TU2, 2) / max(np.linalg.norm(TU2, 2), 1e-15)
    s_err = np.linalg.norm(su1 - SU2)
    print(f"  {h:8.3f} {r_err:16.3e} {t_err:16.3e} {s_err:12.3e}")

# Same for 4th-order
print(f"\n  4th-order Magnus:")
print(f"  {'h':>8s} {'||R1-R2||/||R2||':>16s} {'||T1-T2||/||T2||':>16s} {'||s1-s2||':>12s}")
for h in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    r1, tu1, rb1, td1, su1, sd1 = one_step_star_order4(A_func, S_func, 0.0, h, NQuad, N)
    r2a, tu2a, rb2a, td2a, su2a, sd2a = one_step_star_order4(A_func, S_func, 0.0, h/2, NQuad, N)
    r2b, tu2b, rb2b, td2b, su2b, sd2b = one_step_star_order4(A_func, S_func, h/2, h/2, NQuad, N)
    R2, TU2, TD2, RB2, SU2, SD2 = star_combine(
        r2a, tu2a, td2a, rb2a, su2a, sd2a,
        r2b, tu2b, rb2b, td2b, su2b, sd2b, N)
    r_err = np.linalg.norm(r1 - R2, 2) / max(np.linalg.norm(R2, 2), 1e-15)
    t_err = np.linalg.norm(tu1 - TU2, 2) / max(np.linalg.norm(TU2, 2), 1e-15)
    s_err = np.linalg.norm(su1 - SU2)
    print(f"  {h:8.3f} {r_err:16.3e} {t_err:16.3e} {s_err:12.3e}")
