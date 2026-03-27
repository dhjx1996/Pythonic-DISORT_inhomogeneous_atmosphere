"""
Test suite 10: Adiabatic cloud profiles (convergence).

Mimics realistic cloud microphysics where the effective radius r_e varies with
optical depth, causing both omega(tau) and g(tau) to vary.  Uses linear
interpolation profiles built by make_cloud_profile.

Verification strategy: multi-layer pydisort (20 / 200 layers) must converge
toward the Riccati reference (tol=1e-5).
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import make_cloud_profile, multilayer_pydisort_toa, assert_convergence

NQuad = 8
NLeg  = NQuad
NFourier = NQuad


def _ref_and_layers(tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
                    BDRF_Fourier_modes=()):
    """Run Riccati@tol=1e-5 (reference), pydisort@20 (coarse), pydisort@200 (fine)."""
    _, flux_ref, u0_ref, _, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        tol=1e-5,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    flux_c, u0_c = multilayer_pydisort_toa(
        tau_bot, omega_func, Leg_coeffs_func, 20, NQuad, NLeg, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    flux_f, u0_f = multilayer_pydisort_toa(
        tau_bot, omega_func, Leg_coeffs_func, 200, NQuad, NLeg, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    return flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f


def test_10a():
    """2.1 um channel, tau=10: omega 0.85->0.96, g 0.865->0.820, BDRF rho=0.05."""
    print("\n--- Test 10a ---")
    tau_bot = 10.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)


def test_10b():
    """2.1 um channel, thick tau=30: omega 0.85->0.96, g 0.865->0.820, BDRF rho=0.05."""
    print("\n--- Test 10b ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)


def test_10c():
    """0.65 um channel, tau=10: omega~1 (const), g 0.87->0.83, BDRF rho=0.06."""
    print("\n--- Test 10c ---")
    tau_bot = 10.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.06
    BDRF = [rho / pi]
    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.99995, omega_bot=0.99995,
        g_top=0.87, g_bot=0.83, NLeg=NLeg, NQuad=NQuad,
    )

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=2.5)


def test_10d():
    """2.1 um channel, thick tau=30, land surface: omega 0.85->0.96, g 0.865->0.820, BDRF rho=0.3."""
    print("\n--- Test 10d ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.3
    BDRF = [rho / pi]
    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )

    flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence(flux_ref, flux_c, flux_f, u0_ref, u0_c, u0_f,
                       min_ratio=3.0)
