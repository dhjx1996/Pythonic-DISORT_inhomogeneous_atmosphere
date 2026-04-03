"""
Test suite 10: Adiabatic cloud profiles (convergence).

Mimics realistic cloud microphysics where the effective radius r_e varies with
optical depth, causing both omega(tau) and g(tau) to vary.  Uses linear
interpolation profiles built by make_cloud_profile.

Verification strategy: multi-layer pydisort (50 / 500 layers, 10x refinement)
must converge toward the Riccati reference (tol=1e-8) at O(h^2).
Theoretical convergence ratio for 10x refinement: 100.
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import (
    make_cloud_profile, multilayer_pydisort_toa_full_phi,
    assert_convergence_phi, PHI_VALUES,
)

NQuad = 8
NLeg  = NQuad
NFourier = NQuad
N = NQuad // 2


def _u_phi(func, *args):
    """Extract (N, n_phi) array from a callable at PHI_VALUES."""
    return np.column_stack([func(*args, phi)[:N] for phi in PHI_VALUES])


def _ref_and_layers(tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
                    BDRF_Fourier_modes=()):
    """Run Riccati@tol=1e-8 (reference), pydisort@50 (coarse), pydisort@500 (fine)."""
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-8,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    _, _, uf_c = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 50, NQuad, NLeg, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    _, _, uf_f = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 500, NQuad, NLeg, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    u_ref = _u_phi(u_ToA_func)
    u_coarse = _u_phi(uf_c, 0)
    u_fine = _u_phi(uf_f, 0)
    return u_ref, u_coarse, u_fine


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

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


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

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


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

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)


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

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, Leg_coeffs_func, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=50, abs_tol=1e-3)
