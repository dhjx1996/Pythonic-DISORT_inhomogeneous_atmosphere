"""
Diagnostic: verify star-product handles BDRF, b_pos, b_neg correctly.
Tests the full BC solve using star-product output.
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


def build_A_and_S(omega, D_m, m, mu0, I0):
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
        if I0 == 0:
            return np.zeros(NQuad)
        fac = fac_const * omega * np.exp(-tau / mu0)
        S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0)
        S_neg =  M_inv * fac * D_m(tau, -mu_arr_pos, -mu0)
        return np.concatenate([S_pos, S_neg])

    return A_func, S_func


def star_product_full(A_func, S_func, tau_bot, N_steps, N, NQuad):
    """Returns R_top, T_up, T_down, R_bot, s_up, s_down."""
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


def solve_with_star_product(R_top, T_up, T_down, R_bot, s_up, s_down,
                            b_neg, b_pos, R_surface, tau_bot, mu0, I0,
                            BDRF_mode, there_is_beam):
    """
    Given star-product output, solve the BVP:
    I+(0) = R_top * I-(0) + T_up * I+(tau_bot) + s_up
    I-(tau_bot) = T_down * I-(0) + R_bot * I+(tau_bot) + s_down

    Bottom BC: I+(tau_bot) = R_surface * I-(tau_bot) + b_pos_eff

    where R_surface is from BDRF and b_pos_eff includes surface beam reflection.
    """
    # Build R_surface and b_pos_eff
    mu_W = mu_arr_pos * W
    if BDRF_mode is not None:
        if np.isscalar(BDRF_mode):
            R_surf = 2 * BDRF_mode * mu_W[None, :]
        else:
            R_surf = 2 * BDRF_mode(mu_arr_pos, mu_arr_pos) * mu_W[None, :]
        if there_is_beam:
            if np.isscalar(BDRF_mode):
                beam_refl = mu0 * I0 / (4*math.pi) * 4 * BDRF_mode * np.ones(N)
            else:
                beam_refl = mu0 * I0 / (4*math.pi) * 4 * np.asarray(
                    BDRF_mode(mu_arr_pos, mu0)).ravel()
            b_pos_eff = b_pos + beam_refl * np.exp(-tau_bot / mu0)
        else:
            b_pos_eff = b_pos
    else:
        R_surf = np.zeros((N, N))
        b_pos_eff = b_pos

    # Solve: I+(tau_bot) = R_surf * (T_down * b_neg + R_bot * I+(tau_bot) + s_down) + b_pos_eff
    # (I - R_surf*R_bot) * I+(tau_bot) = R_surf*(T_down*b_neg + s_down) + b_pos_eff
    LHS = np.eye(N) - R_surf @ R_bot
    RHS = R_surf @ (T_down @ b_neg + s_down) + b_pos_eff
    I_plus_bot = np.linalg.solve(LHS, RHS)

    # I+(0) = R_top * b_neg + T_up * I_plus_bot + s_up
    I_plus_0 = R_top @ b_neg + T_up @ I_plus_bot + s_up
    return I_plus_0


# Test cases
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0
g_l_hg5 = 0.5 ** np.arange(NQuad)
g_l_hg75 = 0.75 ** np.arange(NQuad)

D_m_iso = make_D_m_funcs(g_l_iso, NQuad, NQuad)
D_m_hg5 = make_D_m_funcs(g_l_hg5, NQuad, NQuad)
D_m_hg75 = make_D_m_funcs(g_l_hg75, NQuad, NQuad)

tests = [
    # (label, tau, omega, mu0, I0, phi0, b_pos, b_neg, BDRF, g_l, D_m)
    ("4a: b_neg only", 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0/math.pi, None, g_l_iso, D_m_iso),
    ("4b: beam+b_pos", 1.0, 0.8, 0.5, 1.0, 0.0, 0.5, 0.0, None, g_l_hg75, D_m_hg75),
    ("4c: both BCs", 2.0, 0.5, 0.6, math.pi/0.6, 0.5*math.pi, 0.3, 0.1, None, g_l_iso, D_m_iso),
    ("5a: BDRF=0.1", 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1/math.pi, g_l_iso, D_m_iso),
    ("5b: BDRF=0.5", 1.0, 0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5/math.pi, g_l_hg75, D_m_hg75),
]

print("=" * 70)
print("Star-product with full BC solve vs pydisort reference")
print("(m=0 mode only)")
print("=" * 70)

for label, tau_bot, omega, mu0, I0, phi0, b_pos_val, b_neg_val, bdrf, g_l, D_m_funcs_all in tests:
    D_m = D_m_funcs_all[0]
    A_func, S_func = build_A_and_S(omega, D_m, m=0, mu0=mu0, I0=I0)

    R_top, T_up, T_down, R_bot, s_up, s_down = star_product_full(
        A_func, S_func, tau_bot, N_steps=400, N=N, NQuad=NQuad
    )

    b_neg_arr = np.full(N, b_neg_val)
    b_pos_arr = np.full(N, b_pos_val)

    I_plus_0 = solve_with_star_product(
        R_top, T_up, T_down, R_bot, s_up, s_down,
        b_neg_arr, b_pos_arr, None, tau_bot, mu0, I0,
        bdrf, I0 > 0
    )

    flux_up = 2 * math.pi * np.dot(mu_arr_pos * W, I_plus_0)

    BDRF_list = [bdrf] if bdrf is not None else []
    flux_ref, u0_ref = pydisort_toa(
        tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos_val, b_neg=b_neg_val, BDRF_Fourier_modes=BDRF_list
    )

    rel_err = abs(flux_up - flux_ref) / max(abs(flux_ref), 1e-10)
    print(f"  {label:20s} | flux={flux_up:12.6e} ref={flux_ref:12.6e} rel_err={rel_err:.3e}")
