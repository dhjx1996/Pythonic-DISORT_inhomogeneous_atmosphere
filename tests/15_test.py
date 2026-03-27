"""
Test suite 15: Full-domain Riccati integration tests.

Validates the Kvaerno5 Riccati solver end-to-end via pydisort_riccati_jax(tol=...):
  15a: Thick cloud (tau=30) -- accuracy vs multilayer reference.
  15b: Thin atmosphere (tau=1) -- reproducibility.
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import (
    make_cloud_profile, multilayer_pydisort_toa,
    assert_close_to_reference,
)

NQuad = 8
NLeg = NQuad
NFourier = NQuad


def test_15a():
    """Riccati on profile 10b: tau=30, adiabatic cloud, BDRF rho=0.05."""
    print("\n--- Test 15a: Riccati, tau=30, cloud ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]

    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )
    # Reference: high-resolution multilayer pydisort
    NLayers_ref = 6000
    flux_ref, u0_ref = multilayer_pydisort_toa(
        tau_bot, omega_func, Leg_coeffs_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )

    # Riccati solve
    _, flux_hyb, u0_hyb, _, tau_grid = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        tol=1e-3,
        BDRF_Fourier_modes=BDRF,
    )

    n_steps = len(tau_grid) - 1
    print(f"  Riccati grid points: {n_steps}")
    assert_close_to_reference(flux_hyb, u0_hyb, flux_ref, u0_ref, rel_tol=5e-3)


def test_15b():
    """Thin atmosphere (tau=1): reproducibility of Riccati solver."""
    print("\n--- Test 15b: Riccati on thin atmosphere (reproducibility) ---")
    tau_bot = 1.0
    omega = 0.9
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = np.zeros(NLeg)
    g_l[0] = 1.0
    Leg_coeffs_func = lambda tau: g_l

    # First Riccati solve
    _, flux_adap, u0_adap, _, grid_adap = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        tol=1e-3,
    )

    # Same call again — should reproduce identically
    _, flux_hyb, u0_hyb, _, grid_hyb = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        tol=1e-3,
    )

    # Results should be identical (same code path)
    assert np.allclose(flux_adap, flux_hyb, atol=1e-14)
    assert np.allclose(u0_adap, u0_hyb, atol=1e-14)
    n_steps = len(grid_hyb) - 1
    print(f"  steps: {n_steps} (Riccati)")
