"""
Test suite 1: Isotropic scattering, beam source.

Corresponds to Stamnes Test Problem 1.
Tests 1a–1c: thin atmosphere (tau_bot = 0.03125).
Tests 1d–1f: thick atmosphere (tau_bot = 32).

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/1{a-f}.npz
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import get_reference, assert_close_to_reference_phi, PHI_VALUES

NQuad = 8
NLeg  = NQuad
NFourier = NQuad
N = NQuad // 2
mu0   = 0.1
I0    = pi / mu0
phi0  = pi

# Isotropic: only the l=0 Legendre coefficient is non-zero.
g_l = np.zeros(NLeg)
g_l[0] = 1.0
Leg_coeffs_func = lambda tau: g_l


def _run(tau_bot, omega):
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    )
    return u_ToA_func


# ── thin atmosphere ─────────────────────────────────────────────────────────

def test_1a():
    """Thin atmosphere (tau=0.03125), low scattering (omega=0.2)."""
    print("\n--- Test 1a ---")
    tau_bot, omega = 0.03125, 0.2
    u_func_ref = get_reference("1a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_1b():
    """Thin atmosphere (tau=0.03125), conservative scattering (omega~1)."""
    print("\n--- Test 1b ---")
    tau_bot, omega = 0.03125, 1 - 1e-6
    u_func_ref = get_reference("1b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_1c():
    """Thin atmosphere (tau=0.03125), high scattering (omega=0.99)."""
    print("\n--- Test 1c ---")
    tau_bot, omega = 0.03125, 0.99
    u_func_ref = get_reference("1c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


# ── moderate atmosphere ─────────────────────────────────────────────────────

def test_1d():
    """Thick atmosphere (tau=32), low scattering (omega=0.2)."""
    print("\n--- Test 1d ---")
    tau_bot, omega = 32.0, 0.2
    u_func_ref = get_reference("1d", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_1e():
    """Thick atmosphere (tau=32), conservative scattering (omega~1)."""
    print("\n--- Test 1e ---")
    tau_bot, omega = 32.0, 1 - 1e-6
    u_func_ref = get_reference("1e", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)


def test_1f():
    """Thick atmosphere (tau=32), high scattering (omega=0.99)."""
    print("\n--- Test 1f ---")
    tau_bot, omega = 32.0, 0.99
    u_func_ref = get_reference("1f", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    u_ToA_func = _run(tau_bot, omega)
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, PHI_VALUES, N)
