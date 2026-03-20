"""
Test suite 9: Thick atmospheres with tau-varying optical properties (convergence).

Exercises the diffusion-domain solver for optically thick atmospheres with
continuously varying omega(tau) and/or g(tau).  Expected to fail with
NotImplementedError until _solve_diffusion_domain is implemented.

Verification strategy: multi-layer pydisort (20 / 200 layers) must converge
toward the Magnus reference (2000 steps).
"""
import numpy as np
from math import pi
import PythonicDISORT
from _helpers import make_D_m_funcs, multilayer_pydisort_toa, assert_convergence

NQuad = 8
NLeg  = NQuad


def _ref_and_layers(tau_bot, omega_func, g_func, mu0, I0, phi0,
                    b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    """Run Magnus@2000 (reference), pydisort@20 (coarse), pydisort@200 (fine)."""
    def g_l_func(tau):
        g = g_func(tau)
        return g ** np.arange(NLeg)

    D_m_funcs = make_D_m_funcs(
        lambda tau: g_func(tau) ** np.arange(NLeg), NLeg, NQuad
    )

    _, flux_ref, u0_ref, _ = PythonicDISORT.pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=2000,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    flux_c, u0_c = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, 20, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    flux_f, u0_f = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, 200, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    return flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f


def test_9a():
    """Thick tau=5, linear omega 0.90->0.40, isotropic."""
    print("\n--- Test 9a ---")
    tau_bot = 5.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.90 - 0.50 * tau / tau_bot
    g_func     = lambda tau: 0.0  # isotropic

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)


def test_9b():
    """Thick tau=5, linear omega 0.95->0.70, linear g 0.80->0.40."""
    print("\n--- Test 9b ---")
    tau_bot = 5.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.95 - 0.25 * tau / tau_bot
    g_func     = lambda tau: 0.80 - 0.40 * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)


def test_9c():
    """Thick tau=10, linear omega 0.90->0.60, linear g 0.75->0.40, BDRF rho=0.05."""
    print("\n--- Test 9c ---")
    tau_bot = 10.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.90 - 0.30 * tau / tau_bot
    g_func     = lambda tau: 0.75 - 0.35 * tau / tau_bot

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)


def test_9d():
    """Very thick tau=30, near-conservative omega 0.99->0.95, const g=0.85, BDRF rho=0.05."""
    print("\n--- Test 9d ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func = lambda tau: 0.99 - 0.04 * tau / tau_bot
    g_func     = lambda tau: 0.85  # constant

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, g_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)
