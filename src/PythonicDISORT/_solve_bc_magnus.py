import numpy as np


def _solve_bc_magnus(
    Phi_hom,              # (2N, 2N) full-domain homogeneous propagator
    phi_part,             # (2N,)    full-domain particular-solution increment
    N,                    # int, NQuad // 2
    b_pos_m,              # (N,) upward BC at bottom (diffuse surface emission / BC)
    b_neg_m,              # (N,) downward BC at top  (diffuse top-of-atmosphere BC)
    BDRF_Fourier_mode,    # None, scalar, or callable(mu_i, mu_j) -> ndarray
    mu_arr_pos,           # (N,) positive quadrature cosines
    W,                    # (N,) quadrature weights
    m,                    # int, Fourier mode index
    mu0,                  # float, cosine of solar zenith angle
    I0_div_4pi_scaled,    # float, I0 / (4*pi) after source rescaling
    tau_bot,              # float, optical depth of the bottom boundary
    there_is_beam_source, # bool
):
    """
    Solve the N×N boundary-value linear system arising from the Magnus propagator.

    The propagator satisfies

        u(tau_bot) = Phi_hom @ u(0) + phi_part,

    where u = [u^+(tau), u^-(tau)] stacked (positive then negative hemisphere).

    Boundary conditions:
        u^-(0)          = b_neg_m          (top: prescribed downward diffuse)
        u^+(tau_bot)    = R @ u^-(tau_bot) + b_pos_eff    (bottom: surface)

    where b_pos_eff includes the direct-beam surface reflection if BDRF is present
    and a beam source exists:
        b_pos_eff = b_pos_m + mathscr_X_pos * exp(-tau_bot / mu0)

    The BDRF matrix R follows the same convention as _solve_for_coeffs.py:
        R[i, j] = (1 + delta_{m,0}) * rho(mu_i, mu_j) * mu_j * w_j

    Returns
    -------
    u0 : (2N,) ndarray
        Intensity at tau=0: first N entries are upward (u^+(0)), last N are
        downward (= b_neg_m by construction).
    """
    # Block the propagator
    Phi_pp = Phi_hom[:N, :N]
    Phi_pm = Phi_hom[:N, N:]
    Phi_mp = Phi_hom[N:, :N]
    Phi_mm = Phi_hom[N:, N:]
    p_p = phi_part[:N]
    p_m = phi_part[N:]

    m_equals_0 = int(m == 0)
    mu_arr_pos_times_W = mu_arr_pos * W  # (N,)

    # Build BDRF matrix R and direct-beam surface reflection term
    if BDRF_Fourier_mode is not None:
        if np.isscalar(BDRF_Fourier_mode):
            # Scalar (Lambertian) BDRF
            R = (1 + m_equals_0) * BDRF_Fourier_mode * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * BDRF_Fourier_mode * np.ones(N)
            else:
                mathscr_X_pos = np.zeros(N)
        else:
            # Callable BDRF
            R = (1 + m_equals_0) * BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos) * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                # Call with scalar mu0 so the result is always shape (N,)
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * np.asarray(
                    BDRF_Fourier_mode(mu_arr_pos, mu0)
                ).ravel()
            else:
                mathscr_X_pos = np.zeros(N)

        beam_surface_term = mathscr_X_pos * np.exp(-tau_bot / mu0) if there_is_beam_source else np.zeros(N)
        b_pos_eff = b_pos_m + beam_surface_term
    else:
        R = np.zeros((N, N))
        b_pos_eff = b_pos_m

    # Solve the N×N system for u^+(0)
    # u^+(tau_bot) = Phi_pp @ u^+(0) + Phi_pm @ b_neg_m + p_p   (propagator identity)
    # u^+(tau_bot) = R @ u^-(tau_bot) + b_pos_eff                (surface BC)
    # u^-(tau_bot) = Phi_mp @ u^+(0) + Phi_mm @ b_neg_m + p_m   (propagator identity)
    #
    # Substituting:
    #   Phi_pp @ u^+(0) + Phi_pm @ b_neg_m + p_p
    #     = R @ (Phi_mp @ u^+(0) + Phi_mm @ b_neg_m + p_m) + b_pos_eff
    #
    # => (Phi_pp - R @ Phi_mp) @ u^+(0)
    #     = b_pos_eff + R @ (Phi_mm @ b_neg_m + p_m) - Phi_pm @ b_neg_m - p_p

    LHS = Phi_pp - R @ Phi_mp
    RHS = (
        b_pos_eff
        + R @ (Phi_mm @ b_neg_m + p_m)
        - Phi_pm @ b_neg_m
        - p_p
    )

    u_pos_0 = np.linalg.solve(LHS, RHS)  # (N,)

    return np.concatenate([u_pos_0, b_neg_m])  # (2N,)
