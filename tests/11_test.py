"""
Test suite 11: NQuad variation and azimuthal output.

Tests 11a-11b: verify that pydisort_magnus works correctly with different
NQuad values (4 and 16) — not just the default NQuad=8.

Test 11c: validates the u_ToA_func(phi) azimuthal reconstruction against
the full pydisort u(0, phi) at several azimuthal angles.

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/11{a,b}.npz  (11c uses on-the-fly only)
"""
import numpy as np
from math import pi
from pydisort_magnus import pydisort_magnus
from _helpers import (
    make_D_m_funcs, get_reference, assert_close_to_reference,
    pydisort_toa_full_phi, assert_close_to_reference_phi,
)


def test_11a():
    """NQuad=4, medium-thick isotropic (tau=3.0, omega=0.5)."""
    print("\n--- Test 11a ---")
    NQuad = 4
    NLeg  = NQuad
    tau_bot, omega = 3.0, 0.5
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = np.zeros(NLeg); g_l[0] = 1.0
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)

    flux_ref, u0_ref = get_reference(
        "11a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_11b():
    """NQuad=16, medium-thick HG (tau=3.0, omega=0.9, g=0.75).
    """
    print("\n--- Test 11b ---")
    NQuad = 16
    NLeg  = NQuad
    tau_bot, omega, g = 3.0, 0.9, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = g ** np.arange(NLeg)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)

    flux_ref, u0_ref = get_reference(
        "11b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_11c():
    """Azimuthal u_ToA_func(phi) validation against pydisort u(0, phi)."""
    print("\n--- Test 11c ---")
    NQuad = 8
    NLeg  = NQuad
    N     = NQuad // 2
    tau_bot, omega, g = 3.0, 0.8, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0

    g_l = g ** np.arange(NLeg)
    D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)

    # Magnus: get u_ToA_func
    _, _, _, u_ToA_func, _ = pydisort_magnus(
        tau_bot, lambda tau: omega, D_m_funcs, NQuad, mu0, I0, phi0,
    )

    # Reference: pydisort with full phi output
    _, _, u_func_ref = pydisort_toa_full_phi(
        tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    )

    phi_values = [0.0, pi / 4, pi / 2, pi, 3 * pi / 2]
    assert_close_to_reference_phi(u_ToA_func, u_func_ref, phi_values, N)
