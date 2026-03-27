"""
pydisort_riccati_jax — JAX + diffrax port of the Riccati forward solver.

Port of pydisort_riccati.py (scipy Radau IIA) to JAX + diffrax (Kvaerno5).
Solves the 1-D RTE for atmospheres with continuously tau-varying
single-scattering albedo omega(tau) and phase function g_l(tau).

Uses invariant-imbedding Riccati ODE integrated via diffrax Kvaerno5
(L-stable ESDIRK, order 5, adaptive PIDController step-size).

See CLAUDE.md for design rationale and deferred features.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from math import pi

from PythonicDISORT import subroutines
from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
    _riccati_forward_jax,
    _riccati_backward_jax,
)
from _solve_bc_riccati_jax import _solve_bc_riccati_jax


def pydisort_riccati_jax(
    tau_bot,
    omega_func,
    Leg_coeffs_func,
    NQuad,
    NLeg,
    NFourier,
    mu0,
    I0,
    phi0,
    tol=1e-3,
    b_pos=0,
    b_neg=0,
    BDRF_Fourier_modes=(),
):
    """
    Riccati forward solver for a single atmospheric column with
    continuously tau-varying single-scattering albedo and phase function.

    Parameters
    ----------
    tau_bot : float
        Optical depth of the bottom boundary (> 0).
    omega_func : callable
        tau -> omega (float in [0, 1)).
    Leg_coeffs_func : callable
        tau -> (NLeg,) array of Legendre coefficients g_l.
    NQuad : int
        Number of quadrature streams (even, >= 2).
    NLeg : int
        Number of Legendre terms.
    NFourier : int
        Number of Fourier azimuthal modes.
    mu0 : float
        Cosine of the beam zenith angle, in (0, 1].
    I0 : float
        Beam intensity (>= 0).
    phi0 : float
        Beam azimuthal angle, in [0, 2pi).
    tol : float, optional
        Relative tolerance for adaptive Kvaerno5 (default 1e-3).
    b_pos : float or (N,) or (N, NFourier), optional
        Upward diffuse intensity at the bottom boundary.
    b_neg : float or (N,) or (N, NFourier), optional
        Downward diffuse intensity at the top boundary.
    BDRF_Fourier_modes : sequence, optional
        Bidirectional reflectance Fourier mode coefficients.

    Returns
    -------
    mu_arr : (NQuad,) ndarray
    flux_up_ToA : float
    u0_ToA : (NQuad,) ndarray
    u_ToA_func : callable  phi -> (NQuad,) or (NQuad, len(phi))
    tau_grid : ndarray  [0, ..., tau_bot]
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if tau_bot <= 0:
        raise ValueError("`tau_bot` must be positive.")
    if NQuad < 2 or NQuad % 2 != 0:
        raise ValueError("`NQuad` must be a positive even integer.")
    if I0 < 0:
        raise ValueError("`I0` cannot be negative.")
    there_is_beam_source = I0 > 0
    if there_is_beam_source:
        if not (0 < mu0 <= 1):
            raise ValueError("`mu0` must be in (0, 1].")
        if not (0 <= phi0 < 2 * pi):
            raise ValueError("`phi0` must be in [0, 2pi).")
    if tol <= 0:
        raise ValueError("`tol` must be positive.")
    if NLeg < 1:
        raise ValueError("`NLeg` must be >= 1.")
    if NFourier < 1:
        raise ValueError("`NFourier` must be >= 1.")

    N = NQuad // 2

    # ------------------------------------------------------------------
    # Quadrature setup
    # ------------------------------------------------------------------
    mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
    mu_arr = np.concatenate([mu_arr_pos_np, -mu_arr_pos_np])

    # JAX arrays for the solver internals
    mu_arr_pos = jnp.array(mu_arr_pos_np)
    W = jnp.array(W_np)
    M_inv = 1.0 / mu_arr_pos

    # ------------------------------------------------------------------
    # Boundary condition shape handling
    # ------------------------------------------------------------------
    b_pos_1d = np.atleast_1d(b_pos)
    if b_pos_1d.size == 1:
        b_pos_arr = np.full(N, float(b_pos))
        b_pos_matrix = np.tile(b_pos_arr[:, None], (1, NFourier))
    elif b_pos_1d.shape == (N,):
        b_pos_arr = b_pos_1d.astype(float)
        b_pos_matrix = np.tile(b_pos_arr[:, None], (1, NFourier))
    elif b_pos_1d.shape == (N, NFourier):
        b_pos_matrix = b_pos_1d.astype(float)
    else:
        raise ValueError("Shape of `b_pos` is incorrect.")

    b_neg_1d = np.atleast_1d(b_neg)
    if b_neg_1d.size == 1:
        b_neg_arr = np.full(N, float(b_neg))
        b_neg_matrix = np.tile(b_neg_arr[:, None], (1, NFourier))
    elif b_neg_1d.shape == (N,):
        b_neg_arr = b_neg_1d.astype(float)
        b_neg_matrix = np.tile(b_neg_arr[:, None], (1, NFourier))
    elif b_neg_1d.shape == (N, NFourier):
        b_neg_matrix = b_neg_1d.astype(float)
    else:
        raise ValueError("Shape of `b_neg` is incorrect.")

    # ------------------------------------------------------------------
    # Source rescaling for numerical stability
    # ------------------------------------------------------------------
    rescale_factor = np.max((I0, np.max(b_pos_matrix), np.max(b_neg_matrix)))
    if rescale_factor > 0:
        I0_scaled = I0 / rescale_factor
        b_pos_matrix = b_pos_matrix / rescale_factor
        b_neg_matrix = b_neg_matrix / rescale_factor
    else:
        I0_scaled = 0.0
    I0_div_4pi_scaled = I0_scaled / (4 * pi)

    # ------------------------------------------------------------------
    # BDRF mode list
    # ------------------------------------------------------------------
    BDRF_list = list(BDRF_Fourier_modes)
    NBDRF = len(BDRF_list)

    # ------------------------------------------------------------------
    # Fourier mode loop
    # ------------------------------------------------------------------
    u_modes = []
    tau_grid_m0 = None

    for m in range(NFourier):
        m_equals_0 = (m == 0)

        # ---- Pre-compute Legendre products for this mode ---------------
        leg_data = _precompute_legendre(m, NLeg, mu_arr_pos, mu0)

        # ---- Build alpha(tau), beta(tau) --------------------------------
        alpha_m_func, beta_m_func = _make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, m, leg_data, mu_arr_pos, W, M_inv, N,
        )

        # ---- Beam-source q functions -----------------------------------
        if there_is_beam_source:
            fac_const = I0_div_4pi_scaled * (2 - int(m_equals_0)) * 2
            q_up_m, q_down_m = _make_q_funcs_jax(
                omega_func, Leg_coeffs_func, m, leg_data,
                mu_arr_pos, M_inv, mu0, fac_const, N,
            )
        else:
            q_up_m = q_down_m = None

        # ---- Forward sweep: R_up, T_up, s_up --------------------------
        R_up_m, T_up_m, s_up_m, grid_fwd = _riccati_forward_jax(
            alpha_m_func, beta_m_func, tau_bot, N, tol,
            q_up_func=q_up_m, q_down_func=q_down_m,
        )

        # ---- Backward sweep: R_down, T_down, s_down -------------------
        R_down_m, T_down_m, s_down_m, _ = _riccati_backward_jax(
            alpha_m_func, beta_m_func, tau_bot, N, tol,
            q_up_func=q_up_m, q_down_func=q_down_m,
        )

        if m == 0:
            tau_grid_m0 = grid_fwd

        # ---- Boundary conditions ----------------------------------------
        BDRF_mode_m = BDRF_list[m] if m < NBDRF else None
        b_pos_m = jnp.array(b_pos_matrix[:, m])
        b_neg_m = jnp.array(b_neg_matrix[:, m])

        u_m = _solve_bc_riccati_jax(
            R_up_m, T_up_m, T_down_m, R_down_m, s_up_m, s_down_m,
            N, b_pos_m, b_neg_m,
            BDRF_mode_m, mu_arr_pos, W,
            m, mu0, I0_div_4pi_scaled, tau_bot,
            there_is_beam_source,
        )

        u_modes.append(u_m)

    # ------------------------------------------------------------------
    # Assemble outputs (rescale back, convert to numpy)
    # ------------------------------------------------------------------
    u_modes_arr = np.array([np.asarray(u) for u in u_modes])  # (NFourier, 2N)
    if rescale_factor > 0:
        u_modes_arr *= rescale_factor

    u0_ToA = u_modes_arr[0]  # (2N,) zeroth Fourier mode at tau=0

    # Upward diffuse flux at ToA: 2pi * sum_i w_i mu_i u+(0)_i
    flux_up_ToA = float(2 * pi * np.dot(mu_arr_pos_np * W_np, u0_ToA[:N]))

    # Full-intensity function at tau=0
    def u_ToA_func(phi):
        phi = np.atleast_1d(phi)
        m_arr = np.arange(NFourier)
        cos_phases = np.cos(np.outer(m_arr, phi0 - phi))  # (NFourier, len(phi))
        result = u_modes_arr.T @ cos_phases  # (2N, len(phi))
        if result.shape[1] == 1:
            return result[:, 0]
        return result

    return mu_arr, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid_m0
