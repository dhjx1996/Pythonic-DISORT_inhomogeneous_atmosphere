"""
Test suite 7: tau-varying phase function and combined variation.

Exercises the second key new capability: a phase function whose Henyey-
Greenstein asymmetry parameter g changes continuously with optical depth.
Also tests combined tau-variation of both omega and g, and the BDRF path
with tau-varying optical properties.

Verification strategy: multi-layer pydisort converges to Riccati reference
(tol=1e-5), demonstrating O(h^2) convergence.
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
    """
    Run Riccati at tight tolerance (reference) and pydisort at 10 / 100 layers.
    Both omega and g are treated as tau-varying.
    """
    def Leg_coeffs_func(tau):
        g = g_func(tau)
        return g ** np.arange(NLeg)

    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-5,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    _, _, uf_c = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 10, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    _, _, uf_f = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 100, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    u_ref = _u_phi(u_ToA_func)
    u_coarse = _u_phi(uf_c, 0)
    u_fine = _u_phi(uf_f, 0)
    return u_ref, u_coarse, u_fine


def test_7a():
    """Constant omega=0.8, linear g(tau): 0.80 -> 0.40, tau_bot=2."""
    print("\n--- Test 7a: tau-varying g, constant omega ---")
    tau_bot = 2.0
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.8
    g_func     = lambda tau: 0.80 + (0.40 - 0.80) * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)


def test_7b():
    """Linear omega (0.95->0.70) AND linear g (0.80->0.40), tau_bot=2."""
    print("\n--- Test 7b: tau-varying omega AND g ---")
    tau_bot = 2.0
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot
    g_func     = lambda tau: 0.80 + (0.40 - 0.80) * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)


def test_7c():
    """tau-varying omega AND g with Lambertian BDRF surface (rho=0.3), tau_bot=1."""
    print("\n--- Test 7c: tau-varying omega+g, BDRF ---")
    tau_bot = 1.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.3
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.90 + (0.60 - 0.90) * tau / tau_bot
    g_func     = lambda tau: 0.70 + (0.30 - 0.70) * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)
