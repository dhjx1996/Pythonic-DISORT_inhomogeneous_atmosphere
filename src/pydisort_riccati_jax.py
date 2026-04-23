"""
pydisort_riccati_jax — Riccati forward solver for radiative transfer (JAX + diffrax).

Solves the 1-D RTE for atmospheres with continuously tau-varying
single-scattering albedo omega(tau) and phase function g_l(tau),
using invariant-imbedding Riccati ODE integrated via diffrax Kvaerno5
(L-stable ESDIRK, order 5, adaptive PIDController step-size).

See CLAUDE.md for design rationale and deferred features.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
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
    mu0,
    I0,
    phi0,
    NLeg=None,
    NFourier=None,
    tol=1e-3,
    b_pos=0,
    b_neg=0,
    BDRF_Fourier_modes=[],
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
        tau -> (NLeg,) array of Legendre coefficients g_l(tau).
    NQuad : int
        Number of quadrature streams (even, >= 2).
    mu0 : float
        Cosine of the beam zenith angle, in (0, 1].
    I0 : float
        Beam intensity (>= 0).
    phi0 : float
        Beam azimuthal angle, in [0, 2pi).
    NLeg : int or None, optional
        Number of Legendre terms (default: NQuad).
    NFourier : int or None, optional
        Number of azimuthal Fourier modes (default: NQuad).
    tol : float, optional
        Relative tolerance for adaptive Kvaerno5 (default 1e-3).
    b_pos : float or (N,) or (N, NFourier), optional
        Upward diffuse intensity at the bottom boundary.
    b_neg : float or (N,) or (N, NFourier), optional
        Downward diffuse intensity at the top boundary.
    BDRF_Fourier_modes : list, optional
        Bidirectional reflectance Fourier mode coefficients.

    Returns
    -------
    mu_arr_pos : (N,) ndarray  — positive quadrature cosines (upwelling)
    flux_up_ToA : float
    u0_ToA : (N,) ndarray  — upwelling intensity at ToA (zeroth Fourier mode)
    u_ToA_func : callable  phi -> (N,) or (N, len(phi))
    tau_grid : ndarray  [0, ..., tau_bot]

    Notable internal variables
    --------------------------
    N                    : NQuad // 2 (half-stream count)
    mu_arr_pos           : (N,) positive quadrature cosines
    W                    : (N,) quadrature weights
    M_inv                : (N,) 1 / mu_arr_pos
    I0_div_4pi           : I0 / (4 pi), after rescaling
    there_is_beam_source : bool, I0 > 0
    NBDRF                : len(BDRF_Fourier_modes)
    """

    ######################################################################
    #          Setup (cf. pydisort, section "Setup")                     #
    ######################################################################

    # NLeg and NFourier default to NQuad (same convention as pydisort)
    if NLeg is None:
        NLeg = NQuad
    if NFourier is None:
        NFourier = NQuad

    N = NQuad // 2
    there_is_beam_source = I0 > 0
    NBDRF = len(BDRF_Fourier_modes)

    ######################################################################
    #       Input checks (cf. pydisort, section "Input checks")         #
    ######################################################################

    # Optical depth must be positive
    if tau_bot <= 0:
        raise ValueError("tau values cannot be non-positive.")
    # Number of streams must be even and >= 2
    if not NQuad >= 2:
        raise ValueError("There must be at least two streams.")
    if not NQuad % 2 == 0:
        raise ValueError("The number of streams must be even.")
    # NLeg and NFourier constraints: NFourier <= NLeg <= NQuad
    if not NLeg > 0:
        raise ValueError(
            "The number of phase function Legendre coefficients must be positive."
        )
    if not NFourier > 0:
        raise ValueError(
            "The number of Fourier modes to use in the solution must be positive."
        )
    if not NFourier <= NLeg:
        raise ValueError(
            "The number of Fourier modes to use in the solution must be "
            "less than or equal to the number of phase function Legendre "
            "coefficients used."
        )
    if not NLeg <= NQuad:
        raise ValueError(
            "There should be more streams than the number of phase function "
            "Legendre coefficients used."
        )
    if NFourier > 64:
        warnings.warn(
            "`NFourier` is large and may cause errors, consider decreasing "
            "`NFourier` to 64 and it probably should be even less. "
            "By default `NFourier` equals `NQuad`."
        )
    # Beam source checks
    if I0 < 0:
        raise ValueError("The intensity of the incident beam cannot be negative.")
    if there_is_beam_source:
        if not (0 < mu0 and mu0 <= 1):
            raise ValueError(
                "The cosine of the polar angle of the incident beam must be "
                "between 0 and 1, excluding 0."
            )
        if not (0 <= phi0 and phi0 < 2 * pi):
            raise ValueError(
                "Provide the principal azimuthal angle for the incident beam "
                "(must be between 0 and 2pi, excluding 2pi)."
            )
    # Tolerance must be positive
    if tol <= 0:
        raise ValueError("`tol` must be positive.")
    # BC shape checks (cf. pydisort: scalar/vector only contribute to m=0)
    b_pos_is_scalar = False
    b_neg_is_scalar = False
    b_pos_is_vector = False
    b_neg_is_vector = False
    if len(np.atleast_1d(b_pos)) == 1:
        b_pos_is_scalar = True
    elif len(b_pos) == N:
        b_pos_is_vector = True
    elif not np.shape(b_pos) == (N, NFourier):
        raise ValueError("The shape of the bottom boundary condition is incorrect.")
    if len(np.atleast_1d(b_neg)) == 1:
        b_neg_is_scalar = True
    elif len(b_neg) == N:
        b_neg_is_vector = True
    elif not np.shape(b_neg) == (N, NFourier):
        raise ValueError("The shape of the top boundary condition is incorrect.")

    ######################################################################
    # Generation of "double-Gauss" quadrature weights and points        #
    # (cf. section 3.4 of the Comprehensive Documentation)              #
    ######################################################################

    mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)

    # JAX arrays for the Riccati ODE vector field
    mu_arr_pos_jax = jnp.array(mu_arr_pos)
    W_jax = jnp.array(W)
    M_inv = 1.0 / mu_arr_pos_jax

    ######################################################################
    #       Rescale of sources (cf. pydisort, section "Rescale")        #
    ######################################################################

    rescale_factor = np.max((I0, np.max(b_pos), np.max(b_neg)))
    if rescale_factor > 0:
        I0 = I0 / rescale_factor
        # b_pos, b_neg rescaled per-mode below
    else:
        I0 = 0.0
    I0_div_4pi = I0 / (4 * pi)

    ######################################################################
    #  Fourier mode loop                                                #
    # (cf. pydisort / _solve_for_gen_and_part_sols / _solve_for_coeffs) #
    ######################################################################

    u_modes = []
    tau_grid_m0 = None

    for m in range(NFourier):
        m_equals_0 = (m == 0)

        # BC dispatch for mode m
        # (cf. pydisort _solve_for_coeffs: scalars and vectors only
        #  contribute to the zeroth Fourier mode; higher modes get zeros)
        # ----------------------------------------------------------
        if b_pos_is_scalar and m_equals_0:
            b_pos_m = np.full(N, float(b_pos))
        elif b_pos_is_vector and m_equals_0:
            b_pos_m = np.asarray(b_pos, dtype=float)
        elif (b_pos_is_scalar or b_pos_is_vector) and not m_equals_0:
            b_pos_m = np.zeros(N)
        else:
            b_pos_m = np.asarray(b_pos)[:, m]

        if b_neg_is_scalar and m_equals_0:
            b_neg_m = np.full(N, float(b_neg))
        elif b_neg_is_vector and m_equals_0:
            b_neg_m = np.asarray(b_neg, dtype=float)
        elif (b_neg_is_scalar or b_neg_is_vector) and not m_equals_0:
            b_neg_m = np.zeros(N)
        else:
            b_neg_m = np.asarray(b_neg)[:, m]

        if rescale_factor > 0:
            b_pos_m = b_pos_m / rescale_factor
            b_neg_m = b_neg_m / rescale_factor

        b_pos_m = jnp.array(b_pos_m)
        b_neg_m = jnp.array(b_neg_m)

        # Pre-compute Legendre polynomial products for this mode
        # (replaces the inline scipy.special calls in _solve_for_gen_and_part_sols;
        #  pre-computation is necessary for JAX traceability of the ODE vector field)
        # ----------------------------------------------------------
        leg_data_m = _precompute_legendre(m, NLeg, mu_arr_pos_jax, mu0)

        # Build alpha(tau), beta(tau) for the Riccati ODE
        # (cf. section 3.4.2 of the Comprehensive Documentation)
        # ----------------------------------------------------------
        alpha_m_func, beta_m_func = _make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, m, leg_data_m,
            mu_arr_pos_jax, W_jax, M_inv, N,
        )

        # Beam-source q functions
        # (cf. section 3.6.1 of the Comprehensive Documentation)
        # ----------------------------------------------------------
        if there_is_beam_source:
            q_up_m, q_down_m = _make_q_funcs_jax(
                omega_func, Leg_coeffs_func, m, leg_data_m,
                mu_arr_pos_jax, M_inv, mu0, I0_div_4pi, m_equals_0, N,
            )
        else:
            q_up_m = q_down_m = None

        # Forward sweep: R_up, T_up, s_up
        # (build slab from bottom upward; see report section 2.3)
        # ----------------------------------------------------------
        R_up_m, T_up_m, s_up_m, tau_grid_m = _riccati_forward_jax(
            alpha_m_func, beta_m_func, tau_bot, N, tol,
            q_up_func=q_up_m, q_down_func=q_down_m,
        )

        # Backward sweep: R_down, T_down, s_down
        # (build slab from top downward; see report section 2.4)
        # ----------------------------------------------------------
        R_down_m, T_down_m, s_down_m, _ = _riccati_backward_jax(
            alpha_m_func, beta_m_func, tau_bot, N, tol,
            q_up_func=q_up_m, q_down_func=q_down_m,
        )

        if m == 0:
            tau_grid_m0 = tau_grid_m

        # Boundary-condition solve
        # (cf. report section 3; replaces the 2N x 2N system in pydisort)
        # ----------------------------------------------------------
        there_is_BDRF_mode = (NBDRF > m)
        BDRF_Fourier_mode_m = BDRF_Fourier_modes[m] if there_is_BDRF_mode else None

        u_m = _solve_bc_riccati_jax(
            R_up_m, T_up_m, T_down_m, R_down_m, s_up_m, s_down_m,
            N, b_pos_m, b_neg_m,
            BDRF_Fourier_mode_m, mu_arr_pos_jax, W_jax,
            m, mu0, I0_div_4pi, tau_bot,
            there_is_beam_source,
        )

        u_modes.append(u_m)

    ######################################################################
    #       Assemble outputs (rescale back, JAX-traceable)              #
    ######################################################################

    u_modes_arr = jnp.stack(u_modes)  # (NFourier, N)
    if rescale_factor > 0:
        u_modes_arr = u_modes_arr * rescale_factor

    u0_ToA = u_modes_arr[0]  # (N,) zeroth Fourier mode at tau=0

    # Upward diffuse flux at ToA: 2pi * sum_i w_i mu_i u+(0)_i
    flux_up_ToA = float(2 * pi * jnp.dot(mu_arr_pos_jax * W_jax, u0_ToA))

    # Upwelling intensity function at tau=0
    def u_ToA_func(phi):
        phi = jnp.atleast_1d(jnp.asarray(phi, dtype=float))
        m_arr = jnp.arange(NFourier)
        cos_phases = jnp.cos(jnp.outer(m_arr, phi0 - phi))  # (NFourier, len(phi))
        result = u_modes_arr.T @ cos_phases  # (N, len(phi))
        if result.shape[1] == 1:
            return result[:, 0]
        return result

    return mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid_m0


# ======================================================================
# Barycentric Lagrange interpolation in mu (JAX-traceable)
# ======================================================================

def _compute_bary_weights(nodes):
    """Barycentric weights for Lagrange interpolation.

    Parameters
    ----------
    nodes : (N,) numpy array of distinct interpolation nodes.

    Returns
    -------
    weights : (N,) numpy array, w_j = 1 / prod_{k != j} (x_j - x_k).
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    weights = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                weights[j] /= (nodes[j] - nodes[k])
    return weights


def _barycentric_interpolate(mu_query, mu_nodes, values, bary_weights):
    """Barycentric Lagrange interpolation — JAX-traceable.

    Parameters
    ----------
    mu_query : (M,) JAX array of query points.
    mu_nodes : (N,) JAX array of interpolation nodes.
    values : (N,) or (N, K) JAX array of function values at nodes.
    bary_weights : (N,) JAX array of precomputed barycentric weights.

    Returns
    -------
    result : (M,) or (M, K) interpolated values.
    """
    # diff[i, j] = mu_query[i] - mu_nodes[j], shape (M, N)
    diff = mu_query[:, None] - mu_nodes[None, :]

    # Detect exact node matches (avoid division by zero).
    # Both branches of jnp.where must be NaN-free for clean JAX gradients.
    is_node = jnp.abs(diff) < 1e-14
    safe_diff = jnp.where(is_node, 1.0, diff)

    # Kernel: w_j / (mu - mu_j), zeroed at exact matches
    kernel = jnp.where(is_node, 0.0, bary_weights[None, :] / safe_diff)  # (M, N)

    if values.ndim == 1:
        numer = kernel @ values                 # (M,)
        denom = kernel.sum(axis=1)              # (M,)
        interp = numer / denom                  # (M,)
        # At exact nodes: pick the node value directly
        node_val = is_node @ values             # (M,) — at most one True per row
    else:
        numer = kernel @ values                 # (M, K)
        denom = kernel.sum(axis=1, keepdims=True)  # (M, 1)
        interp = numer / denom                  # (M, K)
        node_val = is_node.astype(values.dtype) @ values  # (M, K)

    any_exact = is_node.any(axis=1)  # (M,)
    if values.ndim == 1:
        return jnp.where(any_exact, node_val, interp)
    else:
        return jnp.where(any_exact[:, None], node_val, interp)


def interpolate(u_ToA_func, mu_arr_pos):
    """Barycentric interpolation in mu for ToA upwelling intensity.

    Analog of ``PythonicDISORT.subroutines.interpolate``, restricted to
    tau=0 (ToA) and positive mu (upwelling hemisphere). JAX-traceable for
    autodiff through the entire forward model chain.

    Parameters
    ----------
    u_ToA_func : callable
        ``phi -> (N,)`` or ``phi -> (N, len(phi))``, as returned by
        ``pydisort_riccati_jax``.
    mu_arr_pos : (N,) ndarray
        Positive Gauss-Legendre quadrature cosines, as returned by
        ``pydisort_riccati_jax``.

    Returns
    -------
    u_interp : callable
        ``(mu, phi) -> intensity`` where *mu* is a scalar or 1-D array
        of positive cosines in (0, 1] and *phi* is a scalar or 1-D array
        of azimuthal angles.  Return shape follows broadcasting:
        scalar mu & scalar phi -> scalar; array mu & scalar phi -> (M,);
        scalar mu & array phi -> (K,); array mu & array phi -> (M, K).
    """
    bary_weights = jnp.asarray(_compute_bary_weights(np.asarray(mu_arr_pos)))
    mu_nodes = jnp.asarray(mu_arr_pos)

    def u_interp(mu, phi):
        mu_q = jnp.atleast_1d(jnp.asarray(mu, dtype=float))
        vals = u_ToA_func(phi)  # (N,) or (N, K)
        result = _barycentric_interpolate(mu_q, mu_nodes, vals, bary_weights)
        # Squeeze singleton dimensions to match scalar inputs
        if jnp.ndim(mu) == 0 and result.ndim >= 1:
            result = result[0]
        return result

    return u_interp
