"""
pydisort_magnus — Magnus-integration forward solver for τ-dependent optical properties.

This is an alternative entry point to PythonicDISORT designed for atmospheres where
the single-scattering albedo ω(τ) and the phase-function kernels D^m(τ, μ_i, μ_j)
vary continuously with optical depth τ.  It solves the same 1D RTE as `pydisort`
but uses first-order Magnus integration (midpoint-rule matrix exponential) instead
of per-layer eigendecomposition, enabling an analytically exact treatment of
variable-coefficient layers.

See CLAUDE.md section "Magnus forward solver" for the design rationale and a list
of features that are intentionally deferred (delta-M scaling, NT corrections, etc.).
"""

import warnings
from math import pi

import numpy as np

from PythonicDISORT import subroutines
from PythonicDISORT._magnus_propagator import _compute_magnus_propagator
from PythonicDISORT._solve_bc_magnus import _solve_bc_magnus


def pydisort_magnus(
    tau_bot,
    omega_func,
    D_m_funcs,
    NQuad,
    mu0,
    I0,
    phi0,
    N_magnus_steps=100,
    b_pos=0,
    b_neg=0,
    BDRF_Fourier_modes=(),
):
    """
    Magnus-integration forward solver for a single atmospheric column with
    continuously τ-varying single-scattering albedo ω(τ) and phase function.

    Parameters
    ----------
    tau_bot : float
        Optical depth of the bottom boundary (must be > 0).
    omega_func : callable
        τ (float) → ω (float in [0, 1)).  Single-scattering albedo as a
        function of optical depth.
    D_m_funcs : list of callable
        Length = NFourier.  D_m_funcs[m](τ, mu_i, mu_j) returns the
        azimuthal-mode-m phase-function kernel WITHOUT the ω factor:

            D^m_pure(μ_i, μ_j; τ) = (1/2) Σ_l (2l+1) poch_l g_l^m(τ)
                                     × P_l^m(μ_i) P_l^m(μ_j)

        μ_i and μ_j may be negative; the callable must handle arbitrary signs
        and support broadcasting to (N,) arrays.
    NQuad : int
        Number of quadrature streams (must be even and ≥ 2).
    mu0 : float
        Cosine of the polar angle of the direct (collimated) solar beam,
        in (0, 1].  Set I0 = 0 and mu0 = 1 if there is no beam source.
    I0 : float
        Intensity of the incident collimated beam (≥ 0).
    phi0 : float
        Azimuthal angle of the incident beam, in [0, 2π).
    N_magnus_steps : int, optional
        Number of equidistant Magnus integration steps (default 100).
        Higher values yield more accurate results at higher cost.
    b_pos : float or (N,) or (N, NFourier) array, optional
        Upward diffuse intensity at the bottom boundary (default 0).
        Represents thermal / isotropic emission from the surface beyond
        BDRF reflection.
    b_neg : float or (N,) or (N, NFourier) array, optional
        Downward diffuse intensity at the top boundary (default 0).
    BDRF_Fourier_modes : sequence, optional
        Bidirectional Reflectance Distribution Function expressed as a list
        of NFourier Fourier mode coefficients.  Each entry is either a scalar
        (Lambertian) or a callable(mu_i, mu_j) → ndarray.  Leave empty for a
        purely absorbing / black surface.

    Returns
    -------
    mu_arr : (NQuad,) ndarray
        Quadrature cosines [μ₁ … μ_N, −μ₁ … −μ_N] (positive then negative).
    flux_up_ToA : float
        Upward diffuse flux at τ = 0:  2π × Σ_i w_i μ_i u0_ToA[i]
        (sum over the positive hemisphere only).
    u0_ToA : (NQuad,) ndarray
        Zeroth Fourier azimuthal mode of the intensity field at τ = 0
        (both hemispheres).
    u_ToA_func : callable
        phi → (NQuad,) ndarray  (or (NQuad, len(phi)) for array phi).
        Full intensity at τ = 0 reconstructed from all Fourier modes:
            u(0, φ) = Σ_m u_m(0) × cos(m (φ₀ − φ)).
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
            raise ValueError("`phi0` must be in [0, 2π).")
    if N_magnus_steps < 1:
        raise ValueError("`N_magnus_steps` must be ≥ 1.")
    if len(D_m_funcs) == 0:
        raise ValueError("`D_m_funcs` must contain at least one callable (m = 0).")
    # ------------------------------------------------------------------

    NFourier = len(D_m_funcs)
    N = NQuad // 2

    # ------------------------------------------------------------------
    # Quadrature setup (double-Gauss, section 3.4 of the Comprehensive Documentation)
    # ------------------------------------------------------------------
    mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
    mu_arr = np.concatenate([mu_arr_pos, -mu_arr_pos])
    M_inv = 1.0 / mu_arr_pos

    # ------------------------------------------------------------------
    # Boundary condition shape handling
    # ------------------------------------------------------------------
    b_pos_1d = np.atleast_1d(b_pos)
    if b_pos_1d.size == 1:
        b_pos_arr = np.full(N, float(b_pos))
        b_pos_matrix = np.tile(b_pos_arr[:, None], (1, NFourier))  # (N, NFourier)
    elif b_pos_1d.shape == (N,):
        b_pos_arr = b_pos_1d.astype(float)
        b_pos_matrix = np.tile(b_pos_arr[:, None], (1, NFourier))
    elif b_pos_1d.shape == (N, NFourier):
        b_pos_matrix = b_pos_1d.astype(float)
        b_pos_arr = b_pos_matrix[:, 0]
    else:
        raise ValueError("Shape of `b_pos` is incorrect.")

    b_neg_1d = np.atleast_1d(b_neg)
    if b_neg_1d.size == 1:
        b_neg_arr = np.full(N, float(b_neg))
        b_neg_matrix = np.tile(b_neg_arr[:, None], (1, NFourier))  # (N, NFourier)
    elif b_neg_1d.shape == (N,):
        b_neg_arr = b_neg_1d.astype(float)
        b_neg_matrix = np.tile(b_neg_arr[:, None], (1, NFourier))
    elif b_neg_1d.shape == (N, NFourier):
        b_neg_matrix = b_neg_1d.astype(float)
        b_neg_arr = b_neg_matrix[:, 0]
    else:
        raise ValueError("Shape of `b_neg` is incorrect.")

    # ------------------------------------------------------------------
    # Source rescaling for numerical stability (section 1.4 of the Documentation)
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
    # BDRF mode list padding
    # ------------------------------------------------------------------
    BDRF_list = list(BDRF_Fourier_modes)  # may be shorter than NFourier
    NBDRF = len(BDRF_list)

    # ------------------------------------------------------------------
    # Fourier mode loop
    # ------------------------------------------------------------------
    u_modes = []  # u_modes[m] is (2N,) intensity at tau=0 for mode m

    for m in range(NFourier):
        m_equals_0 = (m == 0)
        D_m = D_m_funcs[m]

        # ---- Coefficient matrix A(τ) --------------------------------
        def make_A_func(D_m_inner):
            def A_func(tau):
                omega = omega_func(tau)
                D_pos = omega * D_m_inner(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])  # (N,N)
                D_neg = omega * D_m_inner(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :]) # (N,N)
                DW_pos = D_pos * W[None, :]
                DW_neg = D_neg * W[None, :]
                alpha = M_inv[:, None] * (DW_pos - np.eye(N))
                beta  = M_inv[:, None] * DW_neg
                A = np.empty((NQuad, NQuad))
                A[:N, :N]  = -alpha
                A[:N, N:]  = -beta
                A[N:, :N]  =  beta
                A[N:, N:]  =  alpha
                return A
            return A_func

        A_m_func = make_A_func(D_m)

        # ---- Beam source vector S(τ) --------------------------------
        if there_is_beam_source:
            # Factor of 2 accounts for the 1/2 built into D_m_funcs convention:
            #   D^m_pure = (1/2) sum_l ..., but source needs the full sum_l ...
            # Derivation: S[:N] = -M_inv * I0_div_4pi * (2-δ_{m0}) * ω * sum_l ... * exp(-τ/μ0)
            #           = -M_inv * [2 * I0_div_4pi * (2-δ_{m0})] * ω * D^m_pure * exp(-τ/μ0)
            # (compare X_arr in _solve_for_gen_and_part_sols.py lines 141-153)
            fac_const = I0_div_4pi_scaled * (2 - int(m_equals_0)) * 2

            def make_S_func(D_m_inner, fac_const_inner):
                def S_func(tau):
                    omega = omega_func(tau)
                    fac = fac_const_inner * omega * np.exp(-tau / mu0)
                    # S[:N][i] = -M_inv[i] * fac * D_m(τ, mu_arr_pos[i], -mu0)
                    # S[N:][i] = +M_inv[i] * fac * D_m(τ, -mu_arr_pos[i], -mu0)
                    # (sign pattern from X_arr in _solve_for_gen_and_part_sols.py lines 151-152)
                    S_pos = -M_inv * fac * D_m_inner(tau, mu_arr_pos, -mu0)
                    S_neg =  M_inv * fac * D_m_inner(tau, -mu_arr_pos, -mu0)
                    return np.concatenate([S_pos, S_neg])
                return S_func

            S_m_func = make_S_func(D_m, fac_const)
        else:
            def S_m_func(_tau):
                return np.zeros(NQuad)

        # ---- Magnus propagator (Redheffer star product, unconditionally stable) ----
        R_up_m, T_up_m, T_down_m, R_down_m, s_up_m, s_down_m = \
            _compute_magnus_propagator(A_m_func, S_m_func, tau_bot, N_magnus_steps, NQuad)

        # ---- Boundary conditions --------------------------------------
        BDRF_mode_m = BDRF_list[m] if m < NBDRF else None
        b_pos_m = b_pos_matrix[:, m]
        b_neg_m = b_neg_matrix[:, m]

        u_m = _solve_bc_magnus(
            R_up_m, T_up_m, T_down_m, R_down_m, s_up_m, s_down_m,
            N, b_pos_m, b_neg_m,
            BDRF_mode_m, mu_arr_pos, W,
            m, mu0, I0_div_4pi_scaled, tau_bot,
            there_is_beam_source,
        )

        u_modes.append(u_m)

    # ------------------------------------------------------------------
    # Assemble outputs (rescale back)
    # ------------------------------------------------------------------
    u_modes_arr = np.array(u_modes)  # (NFourier, 2N)
    if rescale_factor > 0:
        u_modes_arr *= rescale_factor

    u0_ToA = u_modes_arr[0]  # (2N,) zeroth Fourier mode at τ=0

    # Upward diffuse flux at ToA: 2π Σ_i w_i μ_i u^+(0)_i
    flux_up_ToA = float(2 * pi * np.dot(mu_arr_pos * W, u0_ToA[:N]))

    # Full-intensity function at τ=0
    def u_ToA_func(phi):
        phi = np.atleast_1d(phi)
        scalar_input = phi.ndim == 0 or phi.shape == ()
        # cos(m*(phi0 - phi)) for each mode, shape (NFourier, len(phi))
        m_arr = np.arange(NFourier)
        cos_phases = np.cos(np.outer(m_arr, phi0 - phi))  # (NFourier, len(phi))
        # u_modes_arr: (NFourier, 2N); result: (2N, len(phi))
        result = u_modes_arr.T @ cos_phases
        if result.shape[1] == 1:
            return result[:, 0]  # (2N,)
        return result  # (2N, len(phi))

    return mu_arr, flux_up_ToA, u0_ToA, u_ToA_func
