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
from pydisort_magnus import pydisort_magnus
from _helpers import make_D_m_funcs, get_reference, assert_close_to_reference

NQuad = 8
NLeg  = NQuad
mu0   = 0.1
I0    = pi / mu0
phi0  = pi

# Isotropic: only the l=0 Legendre coefficient is non-zero.
g_l = np.zeros(NLeg)
g_l[0] = 1.0
D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)


def _run(tau_bot, omega, N_steps):
    _, flux_up, u0_ToA, _ = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=N_steps,
    )
    return flux_up, u0_ToA


# ── thin atmosphere ─────────────────────────────────────────────────────────

def test_1a():
    """Thin atmosphere (tau=0.03125), low scattering (omega=0.2)."""
    print("\n--- Test 1a ---")
    tau_bot, omega = 0.03125, 0.2
    flux_ref, u0_ref = get_reference("1a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=100)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_1b():
    """Thin atmosphere (tau=0.03125), conservative scattering (omega~1)."""
    print("\n--- Test 1b ---")
    tau_bot, omega = 0.03125, 1 - 1e-6
    flux_ref, u0_ref = get_reference("1b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=100)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_1c():
    """Thin atmosphere (tau=0.03125), high scattering (omega=0.99)."""
    print("\n--- Test 1c ---")
    tau_bot, omega = 0.03125, 0.99
    flux_ref, u0_ref = get_reference("1c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=100)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


# ── moderate atmosphere ─────────────────────────────────────────────────────

def test_1d():
    """Thick atmosphere (tau=32), low scattering (omega=0.2)."""
    print("\n--- Test 1d ---")
    tau_bot, omega = 32.0, 0.2
    flux_ref, u0_ref = get_reference("1d", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=2500)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_1e():
    """Thick atmosphere (tau=32), conservative scattering (omega~1)."""
    print("\n--- Test 1e ---")
    tau_bot, omega = 32.0, 1 - 1e-6
    flux_ref, u0_ref = get_reference("1e", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=2500)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_1f():
    """Thick atmosphere (tau=32), high scattering (omega=0.99)."""
    print("\n--- Test 1f ---")
    tau_bot, omega = 32.0, 0.99
    flux_ref, u0_ref = get_reference("1f", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, N_steps=2500)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)
