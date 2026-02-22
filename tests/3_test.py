"""
Test suite 3: Henyey-Greenstein scattering, beam source.

Corresponds to Stamnes Test Problem 3 (without delta-M / NT corrections,
which are deferred features for pydisort_magnus).
Covers different asymmetry parameters and optical depths.

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/3{a-c}_test.npz
"""
import numpy as np
from math import pi
import PythonicDISORT
from _helpers import make_D_m_funcs, get_reference, assert_close_to_reference

NQuad = 8
NLeg  = NQuad


def _make(g):
    """HG Legendre coefficients: g_l = g^l."""
    return g ** np.arange(NLeg)


def _run(tau_bot, omega, g, mu0, I0, phi0, N_steps):
    g_l = _make(g)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
    _, flux_up, u0_ToA, _ = PythonicDISORT.pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=N_steps,
    )
    return flux_up, u0_ToA


def test_3a():
    """tau=1, g=0.75, omega~1, nadir incidence (mu0=1)."""
    print("\n--- Test 3a ---")
    tau_bot, omega, g = 1.0, 1 - 1e-6, 0.75
    mu0, I0, phi0 = 1.0, pi, pi
    g_l = _make(g)
    flux_ref, u0_ref = get_reference("3a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, g, mu0, I0, phi0, N_steps=200)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_3b():
    """tau=1, g=0.75, omega=0.9, oblique incidence (mu0=0.5)."""
    print("\n--- Test 3b ---")
    tau_bot, omega, g = 1.0, 0.9, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    g_l = _make(g)
    flux_ref, u0_ref = get_reference("3b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, g, mu0, I0, phi0, N_steps=200)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_3c():
    """tau=2, g=0.5, omega=0.8, mu0=0.6."""
    print("\n--- Test 3c ---")
    tau_bot, omega, g = 2.0, 0.8, 0.5
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.9 * pi
    g_l = _make(g)
    flux_ref, u0_ref = get_reference("3c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0)
    flux_mag, u0_mag = _run(tau_bot, omega, g, mu0, I0, phi0, N_steps=300)
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)
