import numpy as np


def _solve_bc_magnus(
    R_up, T_up, T_down, R_down, s_up, s_down,   # star-product scattering operators
    N,
    b_pos_m, b_neg_m,
    BDRF_Fourier_mode, mu_arr_pos, W,
    m, mu0, I0_div_4pi_scaled, tau_bot,
    there_is_beam_source,
):
    """
    Solve the N×N boundary-condition system from star-product operators.

    The scattering-matrix representation gives:
        I⁺(0)     = R_up   · b_neg + T_up   · I⁺(τ_bot) + s_up
        I⁻(τ_bot) = T_down · b_neg + R_down · I⁺(τ_bot) + s_down

    With surface BC:  I⁺(τ_bot) = R_surf · I⁻(τ_bot) + b_pos_eff

    Substituting yields the N×N system:
        (I − R_surf · R_down) · I⁺(τ_bot) = R_surf · (T_down · b_neg + s_down) + b_pos_eff

    Then recover:
        I⁺(0) = R_up · b_neg + T_up · I⁺(τ_bot) + s_up

    Returns
    -------
    u0 : (2N,) ndarray
        Intensity at τ=0: first N entries are upward (u⁺(0)), last N are
        downward (= b_neg_m by construction).
    """
    m_equals_0         = int(m == 0)
    mu_arr_pos_times_W = mu_arr_pos * W  # (N,)

    # ------------------------------------------------------------------
    # Build BDRF matrix R_surf and direct-beam surface reflection term
    # ------------------------------------------------------------------
    if BDRF_Fourier_mode is not None:
        if np.isscalar(BDRF_Fourier_mode):
            R_surf = (1 + m_equals_0) * BDRF_Fourier_mode * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * BDRF_Fourier_mode * np.ones(N)
            else:
                mathscr_X_pos = np.zeros(N)
        else:
            R_surf = (1 + m_equals_0) * BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos) * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * np.asarray(
                    BDRF_Fourier_mode(mu_arr_pos, mu0)
                ).ravel()
            else:
                mathscr_X_pos = np.zeros(N)

        beam_surface_term = mathscr_X_pos * np.exp(-tau_bot / mu0) if there_is_beam_source else np.zeros(N)
        b_pos_eff = b_pos_m + beam_surface_term
    else:
        R_surf    = np.zeros((N, N))
        b_pos_eff = b_pos_m

    # ------------------------------------------------------------------
    # N×N BC system
    # ------------------------------------------------------------------
    LHS = np.eye(N) - R_surf @ R_down
    rhs = R_surf @ (T_down @ b_neg_m + s_down) + b_pos_eff
    I_plus_bot = np.linalg.solve(LHS, rhs)

    # Recovery: I⁺(0) = R_up · b_neg + T_up · I⁺_bot + s_up
    I_plus_top = (R_up @ b_neg_m + T_up @ I_plus_bot + s_up).real

    return np.concatenate([I_plus_top, b_neg_m])  # (2N,)
