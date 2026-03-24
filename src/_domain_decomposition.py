"""
Domain decomposition for the hybrid Magnus4-ROS2 solver.

Uses eigenvalue-gap / beam-negligibility criterion to split thick atmospheres
into three domains: two non-diffusive domains (I, III) handled by adaptive
Magnus4, and a central diffusion domain (II) handled by the ROS2 Riccati
solver.  The three domains are coupled via Redheffer star product.

Top boundary (tau_1):  beam attenuates as exp(-tau/mu0).  At tau_1 the beam
residual equals tol, so tau_1 = -mu0 * ln(tol).

Bottom boundary (tau_2):  the surface BC influences the field over a
penetration depth ~1/k_diff, where k_diff is the smallest eigenvalue
magnitude of the DISORT A matrix.  tau_2 = tau_bot - c/k_diff with c ~ 2.
"""

import numpy as np
from _magnus_propagator import _compute_magnus_propagator_adaptive, _star_product
from _ros2_riccati import _ros2_forward, _ros2_backward


_C_BL = 2.0   # safety factor for bottom boundary layer (units of 1/k_diff)


def _detect_domains(tau_bot, mu0, tol, omega_func, D_m0, mu_arr_pos, W, N):
    """
    Eigenvalue-gap / beam-negligibility criterion for domain detection.

    Returns (tau1, tau2) or (None, None) if no diffusion domain.
    """
    # --- Top boundary: beam negligible at exp(-tau1/mu0) = tol ---
    tau1 = -mu0 * np.log(tol)

    # --- Bottom boundary: need k_diff from A matrix at midpoint ---
    NQuad = 2 * N
    tau_mid = tau_bot / 2.0
    omega = omega_func(tau_mid)
    M_inv = 1.0 / mu_arr_pos
    I_N = np.eye(N)
    D_pos = omega * D_m0(tau_mid, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m0(tau_mid, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    alpha = M_inv[:, None] * (D_pos * W[None, :] - I_N)
    beta = M_inv[:, None] * (D_neg * W[None, :])

    A = np.empty((NQuad, NQuad))
    A[:N, :N] = -alpha
    A[:N, N:] = -beta
    A[N:, :N] = beta
    A[N:, N:] = alpha

    eigvals = np.linalg.eigvals(A)
    abs_real = np.abs(eigvals.real)
    abs_real_pos = abs_real[abs_real > 1e-10]
    if len(abs_real_pos) == 0:
        return None, None
    k_diff = float(np.min(abs_real_pos))

    tau2 = tau_bot - _C_BL / k_diff

    if tau1 >= tau2:
        return None, None
    return tau1, tau2


def _make_alpha_beta_funcs(omega_func, D_m, mu_arr_pos, W, M_inv, N):
    """
    Build alpha(tau) and beta(tau) callables for the Riccati solver.

    These are the N*N blocks of the 2N*2N coefficient matrix A:
        A = [[-alpha, -beta], [beta, alpha]]
    """
    I_N = np.eye(N)

    def alpha_func(tau):
        omega = omega_func(tau)
        D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        return M_inv[:, None] * (DW_pos - I_N)

    def beta_func(tau):
        omega = omega_func(tau)
        D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_neg = D_neg * W[None, :]
        return M_inv[:, None] * DW_neg

    return alpha_func, beta_func


def _hybrid_propagator(A_func, S_func, omega_func, D_m,
                       tau1, tau2, tau_bot,
                       mu_arr_pos, W, M_inv, NQuad, N, tol):
    """
    Three-domain hybrid: Magnus4(I) * ROS2(II) * Magnus4(III).

    Domain I:   [0, tau1]       -- adaptive Magnus4
    Domain II:  [tau1, tau2]    -- ROS2 forward + backward (beam negligible)
    Domain III: [tau2, tau_bot] -- adaptive Magnus4

    Returns (R_up, T_up, T_down, R_down, s_up, s_down, tau_grid).
    """
    # --- Domain I: [0, tau1] via adaptive Magnus4 ---
    R_I, T_I, Td_I, Rd_I, s_I, sd_I, grid_I = \
        _compute_magnus_propagator_adaptive(A_func, S_func, tau1, NQuad, tol)

    # --- Domain II: [tau1, tau2] via ROS2 Riccati ---
    alpha_func, beta_func = _make_alpha_beta_funcs(
        omega_func, D_m, mu_arr_pos, W, M_inv, N,
    )
    R_up_II, T_up_II, grid_fwd = _ros2_forward(
        alpha_func, beta_func, tau1, tau2, N, tol,
    )
    R_down_II, T_down_II, _ = _ros2_backward(
        alpha_func, beta_func, tau1, tau2, N, tol,
    )
    s_up_II = np.zeros(N)
    s_down_II = np.zeros(N)

    # --- Domain III: [tau2, tau_bot] via adaptive Magnus4 ---
    # Shift A_func, S_func to start from tau=0 (internal convention)
    tau2_val = tau2
    A_func_III = lambda tau: A_func(tau2_val + tau)
    S_func_III = lambda tau: S_func(tau2_val + tau)
    R_III, T_III, Td_III, Rd_III, s_III, sd_III, grid_III = \
        _compute_magnus_propagator_adaptive(
            A_func_III, S_func_III, tau_bot - tau2, NQuad, tol,
        )

    # --- Couple: (I * II) * III ---
    R12, T12, Td12, Rd12, s12, sd12 = _star_product(
        R_I, T_I, Td_I, Rd_I, s_I, sd_I,
        R_up_II, T_up_II, T_down_II, R_down_II, s_up_II, s_down_II, N,
    )
    R_up, T_up, T_down, R_down, s_up, s_down = _star_product(
        R12, T12, Td12, Rd12, s12, sd12,
        R_III, T_III, Td_III, Rd_III, s_III, sd_III, N,
    )

    # --- Combined tau_grid ---
    grid_III_shifted = grid_III + tau2
    tau_grid = np.concatenate([grid_I, grid_fwd[1:], grid_III_shifted[1:]])

    return R_up, T_up, T_down, R_down, s_up, s_down, tau_grid
