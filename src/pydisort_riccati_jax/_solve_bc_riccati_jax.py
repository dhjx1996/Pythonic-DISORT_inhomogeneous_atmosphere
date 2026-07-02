"""
BC solver for the Riccati forward solver — JAX port (mode-map uniform).

Solves the N x N boundary-condition system from star-product operators. Written
to be **uniform across Fourier modes** so it runs inside the `lax.scan` over modes
(`_fourier_solve`): the surface BDRF enters as *precomputed*, static-mu0 arrays
(`R_raw`, `beam_raw`) and the (m==0) special case as a float `m_is_zero` — no
per-mode Python branching, no `int(m==0)` on a scanned index, no callable BDRF in
the trace (it is evaluated host-side at the static mu0 in `riccati_setup`).
"""

import jax.numpy as jnp


def _solve_bc_riccati_jax(
    R_up, T_up, T_down, R_down, s_up, s_down,
    N,
    b_pos_m, b_neg_m,
    R_raw, beam_raw, m_is_zero,
    mu_arr_pos, W, I0_div_4pi, mu0, tau_star_bot,
):
    """
    Solve the N x N boundary-condition system from star-product operators.

    The scattering-matrix representation gives:
        I+(0)       = R_up   b_neg + T_up   I+(tau_bot) + s_up
        I-(tau_bot) = T_down b_neg + R_down I+(tau_bot) + s_down

    With surface BC:  I+(tau_bot) = R_surf I-(tau_bot) + b_pos_eff

    Parameters
    ----------
    R_raw : (N, N) JAX array
        The mode's raw BDRF reflectance (BDRF Fourier coefficient evaluated at the
        quadrature pairs; zeros for modes/​cases with no surface). Precomputed
        host-side at the static mu0 in :func:`riccati_setup`. ``R_surf`` is then
        ``(1 + m_is_zero) * R_raw * (mu_arr_pos * W)``.
    beam_raw : (N,) JAX array
        The mode's raw direct-beam surface reflectance ``BDRF(mu_i, mu0)`` (zeros
        for no-surface / no-beam). Drives the direct-beam surface reflection
        ``(mu0 * I0_div_4pi * 4) * beam_raw * exp(-tau_star_bot / mu0)``.
    m_is_zero : scalar (1.0 for mode 0, else 0.0)
        Replaces ``int(m == 0)``; gives the Fourier (2 - delta_{m0}) → (1 + .) factor.
    I0_div_4pi : float
        Rescaled I0/(4 pi). Zero when there is no beam source, which zeroes the
        beam-surface term uniformly (no separate ``there_is_beam_source`` branch).
    mu0 : float (static), tau_star_bot : JAX scalar
        Beam cosine and scaled cumulative depth at the bottom (delta-M); the
        direct-beam surface reflection attenuates as exp(-tau_star_bot / mu0).

    Returns
    -------
    I_plus_top : (N,) JAX array — upwelling intensity at tau=0.
    """
    muW = mu_arr_pos * W

    # Surface reflection matrix and direct-beam surface reflection (both zero
    # when R_raw / beam_raw are zero, i.e. no surface / no beam).
    R_surf = (1.0 + m_is_zero) * R_raw * muW[None, :]          # (N, N)
    beam_surface_term = ((mu0 * I0_div_4pi * 4.0) * beam_raw
                         * jnp.exp(-tau_star_bot / mu0))        # (N,)
    b_pos_eff = b_pos_m + beam_surface_term

    # N x N BC system.
    LHS = jnp.eye(N) - R_surf @ R_down
    RHS = R_surf @ (T_down @ b_neg_m + s_down) + b_pos_eff
    I_plus_bot = jnp.linalg.solve(LHS, RHS)

    # Recovery: I+(0) = R_up b_neg + T_up I+_bot + s_up
    I_plus_top = (R_up @ b_neg_m + T_up @ I_plus_bot + s_up).real
    return I_plus_top
