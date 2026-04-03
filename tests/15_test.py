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
    make_cloud_profile, multilayer_pydisort_toa_full_phi,
    assert_close_to_reference_phi, PHI_VALUES,
)

NQuad = 8
NLeg = NQuad
NFourier = NQuad
N = NQuad // 2


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
    _, _, uf_ref = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )

    # Riccati solve (tol=1e-4 needed for u(phi) to clear 5e-3 at all angles)
    _, _, _, u_ToA_func, tau_grid = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-4,
        BDRF_Fourier_modes=BDRF,
    )

    n_steps = len(tau_grid) - 1
    print(f"  Riccati grid points: {n_steps}")
    assert_close_to_reference_phi(u_ToA_func, uf_ref, PHI_VALUES, N)


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
    _, _, _, u_func_1, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    # Same call again — should reproduce identically
    _, _, _, u_func_2, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )

    # Results should be identical at all phi values
    for phi in PHI_VALUES:
        u1 = u_func_1(phi)[:N]
        u2 = u_func_2(phi)[:N]
        assert np.allclose(u1, u2, atol=1e-14), (
            f"phi={phi:.4f}: reproducibility failed, max diff={np.max(np.abs(u1 - u2)):.2e}"
        )
