"""
Test suite 9: Thick atmospheres with tau-varying optical properties (convergence).

Exercises the star-product solver for optically thick atmospheres with
continuously varying omega(tau) and/or g(tau).

Verification strategy: multi-layer pydisort (50 / 500 layers, 10x refinement)
must converge toward the Riccati reference (tol=1e-8) at O(h^2).
Theoretical convergence ratio for 10x refinement: 100.
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import (
    multilayer_pydisort_toa_full_phi, assert_convergence_phi, PHI_VALUES,
)

NQuad = 8
NLeg  = NQuad
NFourier = NQuad
N = NQuad // 2


def _u_phi(func, *args):
    """Extract (N, n_phi) array from a callable at PHI_VALUES."""
    return np.column_stack([func(*args, phi)[:N] for phi in PHI_VALUES])


def _ref_and_layers(tau_bot, omega_func, g_func, mu0, I0, phi0,
                    b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    """Run Riccati@tol=1e-8 (reference), pydisort@50 (coarse), pydisort@500 (fine)."""
    def Leg_coeffs_func(tau):
        g = g_func(tau)
        return g ** np.arange(NLeg)

    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-8,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    _, _, uf_c = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 50, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    _, _, uf_f = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 500, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    u_ref = _u_phi(u_ToA_func)
    u_coarse = _u_phi(uf_c, 0)
    u_fine = _u_phi(uf_f, 0)
    return u_ref, u_coarse, u_fine


def test_9a():
    """Thick tau=5, linear omega 0.90->0.40, isotropic."""
    print("\n--- Test 9a ---")
    tau_bot = 5.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.90 - 0.50 * tau / tau_bot
    g_func     = lambda tau: 0.0  # isotropic

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


def test_9b():
    """Thick tau=5, linear omega 0.95->0.70, linear g 0.80->0.40."""
    print("\n--- Test 9b ---")
    tau_bot = 5.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.95 - 0.25 * tau / tau_bot
    g_func     = lambda tau: 0.80 - 0.40 * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


def test_9c():
    """Thick tau=10, linear omega 0.90->0.60, linear g 0.75->0.40, BDRF rho=0.05."""
    print("\n--- Test 9c ---")
    tau_bot = 10.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.90 - 0.30 * tau / tau_bot
    g_func     = lambda tau: 0.75 - 0.35 * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


def test_9d():
    """Very thick tau=30, near-conservative omega 0.99->0.95, const g=0.85, BDRF rho=0.05."""
    print("\n--- Test 9d ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.99 - 0.04 * tau / tau_bot
    g_func     = lambda tau: 0.85  # constant

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)
