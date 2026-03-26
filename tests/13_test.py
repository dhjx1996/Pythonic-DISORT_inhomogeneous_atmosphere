"""
Test suite 13: Adaptive Riccati solver (full-domain).

Validates the Radau IIA Riccati solver via pydisort_magnus(tol=...):
  13a: tau-varying omega on thin atmosphere — accuracy vs multilayer reference.
  13b: thick cloud — accuracy + verifies Riccati uses fewer steps than equidistant.
  13c: constant omega — verifies few steps needed.
"""
import numpy as np
from math import pi, ceil
from pydisort_magnus import pydisort_magnus
from _helpers import (
    make_D_m_funcs, make_cloud_profile, multilayer_pydisort_toa,
    assert_close_to_reference,
)

NQuad = 8
NLeg = NQuad


def test_13a():
    """Adaptive on profile 6b: tau=2, linear omega 0.95->0.70, HG g=0.75."""
    print("\n--- Test 13a: Riccati, tau=2, varying omega ---")
    tau_bot, g = 2.0, 0.75
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot

    g_l = g ** np.arange(NLeg)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
    g_l_func = lambda tau: g_l

    # Reference: high-resolution multilayer pydisort
    NLayers_ref = 5000
    flux_ref, u0_ref = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0,
    )

    # Adaptive solve
    _, flux_mag, u0_mag, _, tau_grid = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    n_steps = len(tau_grid) - 1
    print(f"  Riccati steps: {n_steps}")
    print(f"  tau_grid endpoints: [{tau_grid[0]:.4f}, ..., {tau_grid[-1]:.4f}]")
    assert tau_grid is not None
    assert abs(tau_grid[0]) < 1e-14
    assert abs(tau_grid[-1] - tau_bot) < 1e-12
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref, rel_tol=5e-3)


def test_13b():
    """Adaptive on profile 10b: tau=30, adiabatic cloud, BDRF rho=0.05."""
    print("\n--- Test 13b: Riccati, tau=30, cloud ---")
    tau_bot = 30.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]

    omega_func, g_l_func, D_m_funcs = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
    )

    # Reference
    NLayers_ref = 6000
    flux_ref, u0_ref = multilayer_pydisort_toa(
        tau_bot, omega_func, g_l_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )

    # Adaptive solve
    _, flux_mag, u0_mag, _, tau_grid = pydisort_magnus(
        tau_bot, omega_func, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
        BDRF_Fourier_modes=BDRF,
    )

    n_steps = len(tau_grid) - 1
    print(f"  Riccati steps: {n_steps}")
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref, rel_tol=5e-3)
    # Riccati should use far fewer than 2000 equidistant steps
    assert n_steps < 2000, f"Riccati used {n_steps} steps, expected << 2000"


def test_13c():
    """Adaptive on constant omega — stability-limited, few steps expected."""
    print("\n--- Test 13c: Riccati, constant omega ---")
    tau_bot = 1.0
    omega = 0.9
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = np.zeros(NLeg)
    g_l[0] = 1.0
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)

    # Adaptive solve (commutator = 0 for constant A, so only stability ceiling)
    _, flux_mag, u0_mag, _, tau_grid = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    n_steps = len(tau_grid) - 1
    print(f"  Riccati steps: {n_steps}")

    # Reference: tight-tolerance Riccati
    _, flux_ref, u0_ref, _, _ = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        tol=1e-6,
    )

    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref, rel_tol=5e-3)
    assert n_steps < 50, f"constant-omega used {n_steps} steps, expected few"
