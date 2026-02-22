"""
Test suite 7: tau-varying phase function D^m(tau) and combined variation.

Exercises the second key new capability: a phase function whose Henyey-
Greenstein asymmetry parameter g changes continuously with optical depth.
Also tests combined tau-variation of both omega and g, and the BDRF path
with tau-varying optical properties.

Verification strategy: multi-layer pydisort converges to Magnus reference
(2000 steps), demonstrating O(h^2) convergence.
"""
import numpy as np
from math import pi
import PythonicDISORT
from _helpers import make_D_m_funcs, multilayer_pydisort_toa, assert_convergence

NQuad = 8
NLeg  = NQuad


def _ref_and_layers(tau_bot, omega_func, g_func, mu0, I0, phi0,
                    b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    """
    Run Magnus at 2000 steps (reference) and pydisort at 10 / 100 layers.
    Both omega and g are treated as tau-varying.
    """
    def D_m_funcs_varying():
        return make_D_m_funcs(
            lambda tau: g_func(tau) ** np.arange(NLeg), NLeg, NQuad
        )

    def g_l_func(tau):
        g = g_func(tau)
        return g ** np.arange(NLeg)

    D_m_funcs = D_m_funcs_varying()

    _, flux_ref, u0_ref, _ = PythonicDISORT.pydisort_magnus(
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


def test_7a():
    """Constant omega=0.8, linear g(tau): 0.80 -> 0.40, tau_bot=2."""
    print("\n--- Test 7a: tau-varying g, constant omega ---")
    tau_bot = 2.0
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.8
    g_func     = lambda tau: 0.80 + (0.40 - 0.80) * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)


def test_7b():
    """Linear omega (0.95->0.70) AND linear g (0.80->0.40), tau_bot=2."""
    print("\n--- Test 7b: tau-varying omega AND g ---")
    tau_bot = 2.0
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot
    g_func     = lambda tau: 0.80 + (0.40 - 0.80) * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)


def test_7c():
    """tau-varying omega AND g with Lambertian BDRF surface (rho=0.3), tau_bot=1."""
    print("\n--- Test 7c: tau-varying omega+g, BDRF ---")
    tau_bot = 1.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.3
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.90 + (0.60 - 0.90) * tau / tau_bot
    g_func     = lambda tau: 0.70 + (0.30 - 0.70) * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f)
