"""
Test suite 6: tau-varying single-scattering albedo omega(tau).

This is the primary test of the new capability: a continuously varying omega
that cannot be handled by the standard pydisort eigendecomposition.

Verification strategy: multi-layer pydisort with piecewise-constant omega
(midpoint rule) must converge to the Magnus solution as NLayers increases.
The Magnus solver (N_magnus_steps=2000) is the reference.

Expected convergence: O(h^2) in the layer thickness h = tau_bot / NLayers.
"""
import numpy as np
from math import pi
from pydisort_magnus import pydisort_magnus
from _helpers import make_D_m_funcs, multilayer_pydisort_toa, assert_convergence

NQuad = 8
NLeg  = NQuad


def _ref_and_layers(tau_bot, omega_func, g_const, mu0, I0, phi0,
                    b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    """Run Magnus at 2000 steps (reference) and pydisort at 10 / 100 layers."""
    g_l = g_const ** np.arange(NLeg)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
    g_l_func = lambda tau: g_l  # constant phase function

    _, flux_ref, u0_ref, _, _ = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=2000,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    flux_c, u0_c = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, 10, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    flux_f, u0_f = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, 100, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    return flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f


def test_6a():
    """Linear omega(tau): 0.90 -> 0.40, isotropic scattering, tau_bot=1."""
    print("\n--- Test 6a: linear omega, isotropic ---")
    tau_bot, g = 1.0, 0.0   # isotropic: g=0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.90 + (0.40 - 0.90) * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)


def test_6b():
    """Linear omega(tau): 0.95 -> 0.70, HG g=0.75, tau_bot=2."""
    print("\n--- Test 6b: linear omega, HG g=0.75 ---")
    tau_bot, g = 2.0, 0.75
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)


def test_6c():
    """Sinusoidal omega(tau), HG g=0.5, tau_bot=1 — non-monotone variation."""
    print("\n--- Test 6c: sinusoidal omega, HG g=0.5 ---")
    tau_bot, g = 1.0, 0.5
    mu0, I0, phi0 = 0.7, 1.0, 0.0
    omega_func = lambda tau: 0.70 + 0.20 * np.sin(2 * tau)

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)
