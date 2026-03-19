"""
Diagnostic: convergence of star-product approach with increasing N_steps.
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
    m_equals_0 = int(m == 0)
    fac_const = I0_div_4pi * (2 - m_equals_0) * 2

    def S_func(tau):
        fac = fac_const * omega * np.exp(-tau / mu0)
        S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0)
        S_neg = M_inv * fac * D_m(tau, -mu_arr_pos, -mu0)
        return np.concatenate([S_pos, S_neg])

    return A_func, S_func


def star_product_solve(A_func, S_func, tau_bot, N_steps, N, NQuad):
    h = tau_bot / N_steps
    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)
    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))

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

        a_inv = np.linalg.inv(a)
        r_top_k = -a_inv @ b
        t_up_k = a_inv
        r_bot_k = c @ a_inv
        t_down_k = d - c @ a_inv @ b
        src_up_k = -a_inv @ delta_p[:N]
        src_down_k = delta_p[N:] - c @ a_inv @ delta_p[:N]

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

    return R_top, T_up, T_down, R_bot, s_up, s_down


test_cases = [
    (5.0, 0.5, "2c"),
    (32.0, 0.2, "1d"),
    (32.0, 0.99, "1f"),
]

step_counts = [100, 200, 400, 800, 1600]

print("=" * 70)
print("Star-product convergence with N_steps")
print("=" * 70)

for tau_bot, omega, label in test_cases:
    D_m = D_m_funcs[0]
    A_func, S_func = build_A_and_S(omega, D_m, m=0)
    flux_ref, u0_ref = pydisort_toa(tau_bot, omega, NQuad, g_l, mu0, I0, phi0)

    print(f"\n{label} (tau={tau_bot}, omega={omega}), ref flux = {flux_ref:.6e}")
    print(f"  {'N_steps':>8s} {'flux':>13s} {'rel_err':>10s} {'ratio':>8s}")

    prev_err = None
    for ns in step_counts:
        R_top, T_up, T_down, R_bot, s_up, s_down = star_product_solve(
            A_func, S_func, tau_bot, ns, N, NQuad
        )
        u_pos_0 = s_up
        flux_up = 2 * math.pi * np.dot(mu_arr_pos * W, u_pos_0)
        rel_err = abs(flux_up - flux_ref) / max(abs(flux_ref), 1e-10)

        if prev_err is not None and rel_err > 0:
            ratio = prev_err / rel_err
        else:
            ratio = float('nan')
        print(f"  {ns:8d} {flux_up:13.6e} {rel_err:10.3e} {ratio:8.2f}")
        prev_err = rel_err
