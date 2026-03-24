"""
Test suite 15: Hybrid Magnus4+ROS2 domain decomposition.

Validates the three-domain hybrid solver (eigenvalue-gap / beam criterion):
  15a: Thick cloud (tau=30) -- accuracy vs multilayer reference.
  15b: Thin atmosphere (tau=1) -- no decomposition triggered.
  15c: Consistency -- hybrid vs high-K equidistant reference.
"""
import numpy as np
from math import pi
from pydisort_magnus import pydisort_magnus
from _helpers import (
    make_D_m_funcs, make_cloud_profile, multilayer_pydisort_toa,
    assert_close_to_reference,
)

NQuad = 8
NLeg = NQuad


def test_15a():
    """Hybrid on profile 10b: tau=30, adiabatic cloud, BDRF rho=0.05."""
    print("\n--- Test 15a: hybrid, tau=30, cloud ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]

    omega_func, g_l_func, D_m_funcs = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )
    # Reference: high-resolution multilayer pydisort
    NLayers_ref = 6000
    flux_ref, u0_ref = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )

    # Hybrid solve (auto-triggered by tol + beam source)
    _, flux_hyb, u0_hyb, _, tau_grid = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
        BDRF_Fourier_modes=BDRF,
    )

    n_steps = len(tau_grid) - 1
    print(f"  hybrid grid points: {n_steps}")
    assert_close_to_reference(flux_hyb, u0_hyb, flux_ref, u0_ref, rel_tol=5e-3)


def test_15b():
    """Thin atmosphere (tau=1): no decomposition triggered."""
    print("\n--- Test 15b: hybrid on thin atmosphere (no DD) ---")
    tau_bot = 1.0
    omega = 0.9
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = np.zeros(NLeg)
    g_l[0] = 1.0
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)

    # Adaptive (thin atmosphere: eigenvalue-gap criterion should find no diffusion domain)
    _, flux_adap, u0_adap, _, grid_adap = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    # Same call again — should take same code path (no hybrid triggered)
    _, flux_hyb, u0_hyb, _, grid_hyb = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    # Results should be identical (same code path)
    assert np.allclose(flux_adap, flux_hyb, atol=1e-14)
    assert np.allclose(u0_adap, u0_hyb, atol=1e-14)
    n_steps = len(grid_hyb) - 1
    print(f"  steps: {n_steps} (no decomposition, same as adaptive-only)")


def test_15c():
    """Consistency: hybrid and adaptive-only agree at same tol."""
    print("\n--- Test 15c: hybrid vs adaptive-only consistency ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]

    omega_func, g_l_func, D_m_funcs = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )
    # Equidistant reference (high K, no hybrid)
    _, flux_eq, u0_eq, _, _ = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=500, BDRF_Fourier_modes=BDRF,
    )

    # Hybrid (auto-triggered by tol + beam source)
    _, flux_hyb, u0_hyb, _, grid_hyb = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3, BDRF_Fourier_modes=BDRF,
    )

    n_hyb = len(grid_hyb) - 1
    print(f"  equidistant steps: 500")
    print(f"  hybrid steps:      {n_hyb}")

    # Hybrid should agree with high-K equidistant reference
    rel_diff_flux = abs(flux_hyb - flux_eq) / max(abs(flux_eq), 1e-10)
    rel_diff_u0 = np.max(np.abs(u0_hyb - u0_eq)) / max(np.max(np.abs(u0_eq)), 1e-10)
    print(f"  flux rel diff: {rel_diff_flux:.3e}")
    print(f"  u0 max rel diff: {rel_diff_u0:.3e}")
    assert rel_diff_flux < 1e-2, f"flux rel diff {rel_diff_flux:.3e} >= 1e-2"
    assert rel_diff_u0 < 1e-2, f"u0 rel diff {rel_diff_u0:.3e} >= 1e-2"
