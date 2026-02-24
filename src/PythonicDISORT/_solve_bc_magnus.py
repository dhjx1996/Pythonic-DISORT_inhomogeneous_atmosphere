import numpy as np


def _solve_bc_magnus(
    U,                    # (NQuad, r_live) left singular vectors of live modes
    Sigma,                # (r_live,)       singular values, descending; sigma>1 growing
    Vt,                   # (NQuad, NQuad)  right singular vectors, ALL modes
    q_scaled,             # (NQuad,)        scaled particular-solution coords, ALL modes
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
    Solve for the upward intensity at tau=0 from the Magnus propagator SVD output.

    The propagator satisfies (exactly when no modes are frozen, approximately otherwise):

        u(tau_bot) ≈ U @ diag(Sigma) @ Vt_live @ u(0) + phi_part

    where phi_part ≈ U @ diag(Sigma) @ q_scaled_live.  Frozen decaying modes have
    Sigma ≈ 0 so their contribution to u(tau_bot) is negligible, but their Vt rows
    and q_scaled entries still appear in the recovery of u^+(0).

    The Stamnes-Conklin-analog system is always 2N x 2N:
      - Unknowns: [Xi_dec (N,), Xi_gro_bot (N,)]
        where Xi_dec covers all N decaying-mode unknowns (live + frozen).
      - Top BC (u^-(0) = b_neg_m): uses full Vt_dec and Vt_gro, always N equations.
      - Bottom BC (u^+(tau_bot) = R @ u^-(tau_bot) + b_pos_eff): live-only contribution
        via EC_dec_live * Sigma_dec_live; frozen modes contribute ~0 (Sigma ≈ 0).

    Ordering of Vt rows (from _compute_magnus_propagator):
      rows 0 .. r_gro-1    : growing live modes (Sigma > 1), r_gro = N always
      rows r_gro .. r_live-1: decaying live modes (FREEZE_THRESH < Sigma < 1)
      rows r_live .. NQuad-1: frozen decaying modes (Sigma was < FREEZE_THRESH)
    So Vt[r_gro:, :] = all N decaying-mode rows (live + frozen), always N rows.

    Returns
    -------
    u0 : (2N,) ndarray
        Intensity at tau=0: first N entries are upward (u^+(0)), last N are
        downward (= b_neg_m by construction).
    """
    m_equals_0         = int(m == 0)
    mu_arr_pos_times_W = mu_arr_pos * W  # (N,)

    # ------------------------------------------------------------------
    # Build BDRF matrix R and direct-beam surface reflection term
    # ------------------------------------------------------------------
    if BDRF_Fourier_mode is not None:
        if np.isscalar(BDRF_Fourier_mode):
            R = (1 + m_equals_0) * BDRF_Fourier_mode * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * BDRF_Fourier_mode * np.ones(N)
            else:
                mathscr_X_pos = np.zeros(N)
        else:
            R = (1 + m_equals_0) * BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos) * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * np.asarray(
                    BDRF_Fourier_mode(mu_arr_pos, mu0)
                ).ravel()
            else:
                mathscr_X_pos = np.zeros(N)

        beam_surface_term = mathscr_X_pos * np.exp(-tau_bot / mu0) if there_is_beam_source else np.zeros(N)
        b_pos_eff = b_pos_m + beam_surface_term
    else:
        R         = np.zeros((N, N))
        b_pos_eff = b_pos_m

    # ------------------------------------------------------------------
    # Mode split (inferred from live-only Sigma and full-rank Vt)
    #
    # r_gro = N always (DISORT: N growing modes, N decaying modes).
    # r_dec_live = live decaying modes (may be < N for thick atmospheres).
    # r_dec_frozen = frozen decaying modes = N - r_dec_live.
    # ------------------------------------------------------------------
    r_live       = U.shape[1]
    r_gro        = int(np.sum(Sigma > 1.0))   # = N for DISORT
    r_dec_live   = r_live - r_gro              # >= 0

    U_gro          = U[:, :r_gro]             # (NQuad, N)
    U_dec_live     = U[:, r_gro:]             # (NQuad, r_dec_live); may be empty

    Sigma_gro      = Sigma[:r_gro]            # (N,)
    Sigma_dec_live = Sigma[r_gro:]            # (r_dec_live,); may be empty

    D_gro_inv = 1.0 / Sigma_gro               # (N,); entries < 1; no overflow

    # Vt rows: [gro (N rows), dec_live (r_dec_live rows), frozen (r_dec_frozen rows)]
    # Vt[r_gro:, :] = all N decaying rows (live decaying + frozen decaying)
    Vt_gro = Vt[:r_gro, :]                    # (N, NQuad)
    Vt_dec = Vt[r_gro:, :]                    # (N, NQuad) — all N decaying modes

    # Hemisphere splits: rows 0:N = upwelling (+mu), rows N:2N = downwelling (-mu)
    U_gro_top = U_gro[:N, :]                  # (N, N)
    U_gro_bot = U_gro[N:, :]                  # (N, N)

    # Columns 0:N = upwelling streams, columns N:2N = downwelling streams
    Vt_gro_top = Vt_gro[:, :N]               # (N, N)
    Vt_gro_bot = Vt_gro[:, N:]               # (N, N)
    Vt_dec_top = Vt_dec[:, :N]               # (N, N)
    Vt_dec_bot = Vt_dec[:, N:]               # (N, N)

    # ------------------------------------------------------------------
    # Stamnes-Conklin-analog 2N x 2N system
    #
    # Unknowns: [Xi_dec (N,), Xi_gro_bot (N,)]
    #   Xi_dec:     xi_dec + q_scaled_dec  for all N decaying modes
    #   Xi_gro_bot: xi_gro_bot + Sigma_gro * q_scaled_gro
    #
    # Top BC (u^-(0) = b_neg_m):
    #   Vt_dec_bot.T @ Xi_dec + Vt_gro_bot.T @ diag(D_gro_inv) @ Xi_gro_bot
    #   = b_neg_m + Vt[:, N:].T @ q_scaled
    #
    # Bottom BC (u^+(tau_bot) = R @ u^-(tau_bot) + b_pos_eff):
    #   bot_left @ Xi_dec + EC_gro @ Xi_gro_bot = b_pos_eff
    #   where bot_left has nonzero entries only for live decaying modes
    #   (frozen dec modes: Sigma ≈ 0 => their Sigma * Xi_dec contribution is ~0)
    #
    # All LHS entries O(1): D_gro_inv < 1, Sigma_dec_live < 1. No positive exponents.
    # System is always square (2N x 2N); always use np.linalg.solve.
    # ------------------------------------------------------------------
    top_left  = Vt_dec_bot.T                               # (N, N)
    top_right = Vt_gro_bot.T * D_gro_inv[None, :]         # (N, N)

    EC_gro    = U_gro_top - R @ U_gro_bot                 # (N, N)
    bot_right = EC_gro

    # Frozen dec modes contribute 0 to bottom BC (their Sigma ≈ 0)
    bot_left = np.zeros((N, N))
    if r_dec_live > 0:
        U_dec_live_top = U_dec_live[:N, :]                 # (N, r_dec_live)
        U_dec_live_bot = U_dec_live[N:, :]                 # (N, r_dec_live)
        EC_dec_live    = U_dec_live_top - R @ U_dec_live_bot   # (N, r_dec_live)
        bot_left[:, :r_dec_live] = EC_dec_live * Sigma_dec_live[None, :]

    LHS_SC = np.block([[top_left,  top_right],
                       [bot_left,  bot_right]])             # (2N, 2N)

    # RHS uses full NQuad Vt and q_scaled (live + frozen contributions)
    RHS_SC = np.concatenate([
        b_neg_m + Vt[:, N:].T @ q_scaled,
        b_pos_eff,
    ])   # (2N,)

    # System is always square — always use direct solve
    xi         = np.linalg.solve(LHS_SC, RHS_SC)
    Xi_dec     = xi[:N]    # (N,)
    Xi_gro_bot = xi[N:]    # (N,)

    # ------------------------------------------------------------------
    # Recover u^+(0)
    #
    # u(0) = Vt_dec.T @ Xi_dec + Vt_gro.T @ diag(D_gro_inv) @ Xi_gro_bot
    #        - Vt.T @ q_scaled
    #
    # (Derivation: substituting back the SC shift and using D_gro_inv * Sigma_gro = 1.)
    # u^+(0) is the upwelling (first N components).
    # Take .real to discard tiny imaginary parts from numerical noise.
    # ------------------------------------------------------------------
    u_pos_0 = (
        Vt_dec_top.T @ Xi_dec
        + (Vt_gro_top.T * D_gro_inv[None, :]) @ Xi_gro_bot
        - Vt[:, :N].T @ q_scaled
    ).real

    return np.concatenate([u_pos_0, b_neg_m])  # (2N,)
