"""
Diagnostic: Riccati-based propagation vs star-product vs pydisort.
Tests whether the Riccati update R_new = (aR + b)(cR + d)^{-1}
gives the same result as the full star product.
Also tests the condition number of intermediate quantities.
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
        beta  = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N] = -alpha; A[:N, N:] = -beta
        A[N:, :N] = beta;   A[N:, N:] = alpha
        return A

    I0_div_4pi = I0 / (4*math.pi)
    m_equals_0 = int(m == 0)
    fac_const = I0_div_4pi * (2 - m_equals_0) * 2

    def S_func(tau):
        fac = fac_const * omega * np.exp(-tau / mu0)
        S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0)
        S_neg =  M_inv * fac * D_m(tau, -mu_arr_pos, -mu0)
        return np.concatenate([S_pos, S_neg])

    return A_func, S_func


print("=" * 70)
print("PART A: Condition numbers of intermediate quantities at each step")
print("=" * 70)

for tau_bot, omega, label in [(5.0, 0.5, "2c"), (32.0, 0.2, "1d"), (32.0, 0.99, "1f")]:
    D_m = D_m_funcs[0]
    A_func, S_func = build_A_and_S(omega, D_m, m=0)
    N_steps = 1600
    h = tau_bot / N_steps

    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))

    max_cond_a = 0
    max_cond_resolvent = 0
    max_norm_R = 0
    max_norm_T = 0

    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)

    for k in range(N_steps):
        tau_mid = (k + 0.5) * h
        M_ext[:NQuad, :NQuad] = h * A_func(tau_mid)
        M_ext[:NQuad, NQuad] = h * S_func(tau_mid)
        M_ext[NQuad, :] = 0.0
        Phi_ext = scipy.linalg.expm(M_ext)
        Phi = Phi_ext[:NQuad, :NQuad]
        delta_p = Phi_ext[:NQuad, NQuad]

        a = Phi[:N, :N]; b = Phi[:N, N:]
        c = Phi[N:, :N]; d = Phi[N:, N:]

        cond_a = np.linalg.cond(a)
        max_cond_a = max(max_cond_a, cond_a)

        a_inv = np.linalg.inv(a)
        r_top_k = -a_inv @ b
        t_up_k = a_inv
        r_bot_k = c @ a_inv
        t_down_k = d - c @ a_inv @ b
        src_up_k = -a_inv @ delta_p[:N]
        src_down_k = delta_p[N:] - c @ a_inv @ delta_p[:N]

        resolvent = np.eye(N) - r_top_k @ R_bot
        cond_res = np.linalg.cond(resolvent)
        max_cond_resolvent = max(max_cond_resolvent, cond_res)

        M2 = np.linalg.inv(resolvent.T)  # actually inv of (I - R_bot @ r_top_k)
        M2 = np.linalg.inv(np.eye(N) - r_top_k @ R_bot)
        M1 = np.linalg.inv(np.eye(N) - R_bot @ r_top_k)

        R_top_new = R_top + T_up @ r_top_k @ M1 @ T_down
        R_bot_new = r_bot_k + t_down_k @ R_bot @ M2 @ t_up_k
        T_up_new = T_up @ M2 @ t_up_k
        T_down_new = t_down_k @ M1 @ T_down
        s_up_new = s_up + T_up @ M2 @ (r_top_k @ s_down + src_up_k)
        s_down_new = (t_down_k @ (s_down + R_bot @ M2 @ (r_top_k @ s_down + src_up_k))
                      + src_down_k)

        R_top = R_top_new; R_bot = R_bot_new
        T_up = T_up_new; T_down = T_down_new
        s_up = s_up_new; s_down = s_down_new

        max_norm_R = max(max_norm_R, np.linalg.norm(R_top, 2), np.linalg.norm(R_bot, 2))
        max_norm_T = max(max_norm_T, np.linalg.norm(T_up, 2), np.linalg.norm(T_down, 2))

    print(f"\n{label} (tau={tau_bot}, omega={omega}, N_steps={N_steps}, h={h:.4e}):")
    print(f"  max cond(a_step) = {max_cond_a:.4e}")
    print(f"  max cond(resolvent) = {max_cond_resolvent:.4e}")
    print(f"  max ||R||_2 = {max_norm_R:.6e}")
    print(f"  max ||T||_2 = {max_norm_T:.6e}")
    print(f"  final ||R_top||_2 = {np.linalg.norm(R_top, 2):.6e}")
    print(f"  final ||T_down||_2 = {np.linalg.norm(T_down, 2):.6e}")

    u_pos_0 = s_up
    flux_up = 2 * math.pi * np.dot(mu_arr_pos * W, u_pos_0)
    flux_ref, u0_ref = pydisort_toa(tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    rel_err = abs(flux_up - flux_ref) / max(abs(flux_ref), 1e-10)
    print(f"  flux_star = {flux_up:.6e}, ref = {flux_ref:.6e}, rel_err = {rel_err:.3e}")


print("\n" + "=" * 70)
print("PART B: Verify eigenvalue structure of A")
print("=" * 70)

for omega in [0.2, 0.5, 0.9, 0.99]:
    D_m = D_m_funcs[0]
    A_func, _ = build_A_and_S(omega, D_m, m=0)
    A = A_func(1.0)
    evals = np.sort(np.linalg.eigvals(A).real)[::-1]
    lam_max = evals[0]
    # For small h, cond(a) ~ cond(exp(A*h)[:N,:N])
    # The condition number of exp(A*h) is exp((lam_max - lam_min)*h) = exp(2*lam_max*h)
    print(f"\nomega={omega}: lam_max={lam_max:.4f}")
    for h in [0.01, 0.02, 0.05, 0.1, 0.32]:
        Phi = scipy.linalg.expm(A * h)
        a = Phi[:N, :N]
        print(f"  h={h:.2f}: cond(Phi)={np.linalg.cond(Phi):.2e}, cond(a)={np.linalg.cond(a):.2e}, lam*h={lam_max*h:.2f}")
