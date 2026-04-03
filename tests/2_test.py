"""
Test suite 2: Rayleigh-like scattering, beam source.

Corresponds to Stamnes Test Problem 2.
Phase function has only l=0 and l=2 Legendre coefficients (g_2 = 0.1),
approximating Rayleigh scattering.  Tests thin (tau=0.2) and moderate
(tau=1.5) single-layer atmospheres.

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/2{a-d}.npz
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import get_reference, assert_close_to_reference_phi, PHI_VALUES

NQuad = 8
NLeg  = NQuad
NFourier = NQuad
N = NQuad // 2
mu0   = 0.080442
I0    = pi
phi0  = pi

# Rayleigh-like: l=0 and l=2 terms only.
g_l = np.zeros(NLeg)
g_l[0] = 1.0
g_l[2] = 0.1
Leg_coeffs_func = lambda tau: g_l


def _run(tau_bot, omega):
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    )
    return u_ToA_func


def test_2a():
    """Thin atmosphere (tau=0.2), moderate scattering (omega=0.5)."""
    print("\n--- Test 2a ---")
    tau_bot, omega = 0.2, 0.5
    u_func_ref = get_reference("2a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_2b():
    """Thin atmosphere (tau=0.2), conservative scattering (omega~1)."""
    print("\n--- Test 2b ---")
    tau_bot, omega = 0.2, 1 - 1e-6
    u_func_ref = get_reference("2b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_2c():
    """Thick atmosphere (tau=5), moderate scattering (omega=0.5)."""
    print("\n--- Test 2c ---")
    tau_bot, omega = 5.0, 0.5
    u_func_ref = get_reference("2c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_2d():
    """Thick atmosphere (tau=5), conservative scattering (omega~1)."""
    print("\n--- Test 2d ---")
    tau_bot, omega = 5.0, 1 - 1e-6
    u_func_ref = get_reference("2d", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)
