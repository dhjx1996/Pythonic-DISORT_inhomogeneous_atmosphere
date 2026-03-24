"""
Test suite 12: Direct solver convergence rate (K-sweep).

Varies the solver's own step count K on a fixed atmosphere, measures error
against a multi-layer pydisort reference, and checks BOTH:
  - Convergence order: err(K)/err(2K) ≈ 2^p
  - Correctness: err(finest K) < 0.5%

Reference: multilayer_pydisort_toa at high NLayers (5000–6000).

Expected convergence ratio ≈ 16 (O(h⁴)) for the 4th-order Magnus integrator.
"""
import numpy as np
from math import pi, ceil
from pydisort_magnus import pydisort_magnus
from _helpers import (
    make_D_m_funcs, make_cloud_profile, multilayer_pydisort_toa,
    assert_convergence_and_accuracy,
)

NQuad = 8
NLeg = NQuad


def _ksweep(tau_bot, omega_func, D_m_funcs, mu0, I0, phi0,
            K_values, NLayers_ref, g_l_func,
            BDRF_Fourier_modes=(), expected_order=4, noise_floor=1e-4):
    """Run solver at multiple K values and check convergence + accuracy."""

    # Reference: multi-layer pydisort
    flux_ref, u0_ref = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    flux_results = []
    u0_results = []
    for K in K_values:
        _, flux_k, u0_k, _, _ = pydisort_magnus(
            tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
            N_magnus_steps=K,
            BDRF_Fourier_modes=BDRF_Fourier_modes,
        )
        flux_results.append(flux_k)
        u0_results.append(u0_k)

    assert_convergence_and_accuracy(
        K_values, flux_results, u0_results,
        flux_ref, u0_ref,
        expected_order=expected_order,
        noise_floor=noise_floor,
    )


def test_12a():
    """K-sweep on profile 6b: tau=2, linear omega 0.95->0.70, HG g=0.75."""
    print("\n--- Test 12a: K-sweep, profile 6b (tau=2, varying omega) ---")
    tau_bot, g = 2.0, 0.75
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot

    g_l = g ** np.arange(NLeg)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
    g_l_func = lambda tau: g_l

    _ksweep(tau_bot, omega_func, D_m_funcs, mu0, I0, phi0,
            K_values=[4, 8, 16, 32],
            NLayers_ref=5000, g_l_func=g_l_func)


def test_12b():
    """K-sweep on profile 10b: tau=30, adiabatic cloud, BDRF rho=0.05."""
    print("\n--- Test 12b: K-sweep, profile 10b (tau=30, cloud) ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]

    omega_func, g_l_func, D_m_funcs = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )

    _ksweep(tau_bot, omega_func, D_m_funcs, mu0, I0, phi0,
            K_values=[100, 200, 400, 800],
            NLayers_ref=6000, g_l_func=g_l_func,
            BDRF_Fourier_modes=BDRF, noise_floor=1e-6)
