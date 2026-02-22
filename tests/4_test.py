"""
Test suite 4: Non-zero diffuse boundary conditions.

Corresponds to aspects of Stamnes Test Problems 8 and 9 (absorbing /
isotropic-scattering media with diffuse top / bottom illumination).
All tests use single-layer atmospheres with constant optical properties.

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/4{a-c}_test.npz
"""
import numpy as np
from math import pi
import PythonicDISORT
from _helpers import make_D_m_funcs, get_reference, assert_close_to_reference

NQuad = 8
NLeg  = NQuad


def _make_isotropic():
    g_l = np.zeros(NLeg); g_l[0] = 1.0
    return g_l, make_D_m_funcs(g_l, NLeg, NQuad)


def _make_HG(g):
    g_l = g ** np.arange(NLeg)
    return g_l, make_D_m_funcs(g_l, NLeg, NQuad)


def test_4a():
    """No beam, isotropic scattering, downward diffuse top BC (b_neg=1/pi)."""
    print("\n--- Test 4a ---")
    tau_bot, omega = 0.5, 0.5
    mu0, I0, phi0  = 0.5, 0.0, 0.0   # I0=0  -> no beam; mu0 ignored
    b_neg = 1.0 / pi
    g_l, D_m_funcs = _make_isotropic()

    flux_ref, u0_ref = get_reference(
        "4a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0, b_neg=b_neg
    )
    _, flux_mag, u0_mag, _ = PythonicDISORT.pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=200, b_neg=b_neg,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_4b():
    """Beam + upward surface emission (b_pos=0.5), HG g=0.75."""
    print("\n--- Test 4b ---")
    tau_bot, omega, g = 1.0, 0.8, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    b_pos = 0.5
    g_l, D_m_funcs = _make_HG(g)

    flux_ref, u0_ref = get_reference(
        "4b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0, b_pos=b_pos
    )
    _, flux_mag, u0_mag, _ = PythonicDISORT.pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=200, b_pos=b_pos,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_4c():
    """Beam + both b_pos=0.3 and b_neg=0.1, isotropic scattering."""
    print("\n--- Test 4c ---")
    tau_bot, omega = 2.0, 0.5
    mu0, I0, phi0  = 0.6, pi / 0.6, 0.5 * pi
    b_pos, b_neg   = 0.3, 0.1
    g_l, D_m_funcs = _make_isotropic()

    flux_ref, u0_ref = get_reference(
        "4c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg,
    )
    _, flux_mag, u0_mag, _ = PythonicDISORT.pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
        N_magnus_steps=300, b_pos=b_pos, b_neg=b_neg,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)
