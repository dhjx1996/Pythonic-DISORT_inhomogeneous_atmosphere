"""
BC solver for the Riccati forward solver — JAX port.

Port of _solve_bc_riccati.py (numpy) to JAX (jnp).
Solves the N x N boundary-condition system from star-product operators.
"""

import jax.numpy as jnp
import numpy as np


def _solve_bc_riccati_jax(
    R_up, T_up, T_down, R_down, s_up, s_down,
    N,
    b_pos_m, b_neg_m,
    BDRF_Fourier_mode, mu_arr_pos, W,
    m, mu0, I0_div_4pi_scaled, tau_bot,
    there_is_beam_source,
):
    """
    Solve the N x N boundary-condition system from star-product operators.

    The scattering-matrix representation gives:
        I+(0)     = R_up   b_neg + T_up   I+(tau_bot) + s_up
        I-(tau_bot) = T_down b_neg + R_down I+(tau_bot) + s_down

    With surface BC:  I+(tau_bot) = R_surf I-(tau_bot) + b_pos_eff

    Returns
    -------
    I_plus_top : (N,) JAX array
        Upwelling intensity at tau=0.
    """
    m_equals_0 = int(m == 0)
    mu_arr_pos_times_W = mu_arr_pos * W

    # ------------------------------------------------------------------
    # Build BDRF matrix R_surf and direct-beam surface reflection term
    # ------------------------------------------------------------------
    if BDRF_Fourier_mode is not None:
        if callable(BDRF_Fourier_mode):
            mu_pos_np = np.asarray(mu_arr_pos)
            R_surf_vals = BDRF_Fourier_mode(mu_pos_np, mu_pos_np)
            R_surf = (1 + m_equals_0) * jnp.asarray(R_surf_vals) * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                bdrf_beam = jnp.asarray(
                    BDRF_Fourier_mode(mu_pos_np, mu0)
                ).ravel()
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * bdrf_beam
            else:
                mathscr_X_pos = jnp.zeros(N)
        else:
            R_surf = (1 + m_equals_0) * BDRF_Fourier_mode * mu_arr_pos_times_W[None, :]
            if there_is_beam_source:
                mathscr_X_pos = (mu0 * I0_div_4pi_scaled * 4) * BDRF_Fourier_mode * jnp.ones(N)
            else:
                mathscr_X_pos = jnp.zeros(N)

        if there_is_beam_source:
            beam_surface_term = mathscr_X_pos * jnp.exp(-tau_bot / mu0)
        else:
            beam_surface_term = jnp.zeros(N)
        b_pos_eff = b_pos_m + beam_surface_term
    else:
        R_surf = jnp.zeros((N, N))
        b_pos_eff = b_pos_m

    # ------------------------------------------------------------------
    # N x N BC system
    # ------------------------------------------------------------------
    LHS = jnp.eye(N) - R_surf @ R_down
    RHS = R_surf @ (T_down @ b_neg_m + s_down) + b_pos_eff
    I_plus_bot = jnp.linalg.solve(LHS, RHS)

    # Recovery: I+(0) = R_up b_neg + T_up I+_bot + s_up
    I_plus_top = (R_up @ b_neg_m + T_up @ I_plus_bot + s_up).real

    return I_plus_top
