"""
Test suite 8: Thick atmospheres with boundary conditions (constant omega).

Exercises the Redheffer star-product solver for optically thick atmospheres
combined with surface BDRF and/or bottom sources (b_pos).

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/8{a-f}.npz
"""
import numpy as np
from math import pi
from pydisort_magnus_jax import pydisort_magnus_jax
from _helpers import get_reference, assert_close_to_reference

NQuad = 8
NLeg  = NQuad
NFourier = NQuad


def _make_isotropic():
    g_l = np.zeros(NLeg); g_l[0] = 1.0
    return g_l


def _make_HG(g):
    return g ** np.arange(NLeg)


def _make_Rayleigh():
    g_l = np.zeros(NLeg); g_l[0] = 1.0; g_l[2] = 0.1
    return g_l


def test_8a():
    """Thick cloud (tau=32), omega=0.99, isotropic, BDRF rho=0.05 (ocean)."""
    print("\n--- Test 8a ---")
    tau_bot, omega = 32.0, 0.99
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.05
    BDRF = [rho / pi]
    g_l = _make_isotropic()
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_8b():
    """Thick cloud (tau=32), omega=0.99, isotropic, BDRF rho=0.3 (land)."""
    print("\n--- Test 8b ---")
    tau_bot, omega = 32.0, 0.99
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.3
    BDRF = [rho / pi]
    g_l = _make_isotropic()
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_8c():
    """Thick atmosphere (tau=10), omega=0.9, HG g=0.75, high BDRF rho=0.85."""
    print("\n--- Test 8c ---")
    tau_bot, omega, g = 10.0, 0.9, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.85
    BDRF = [rho / pi]
    g_l = _make_HG(g)
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_8d():
    """Thick (tau=32), omega=0.5, isotropic, b_pos=0.5."""
    print("\n--- Test 8d ---")
    tau_bot, omega = 32.0, 0.5
    mu0, I0, phi0 = 0.1, pi / 0.1, pi
    b_pos = 0.5
    g_l = _make_isotropic()
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8d", tau_bot, omega, NQuad, g_l, mu0, I0, phi0, b_pos=b_pos,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        b_pos=b_pos,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_8e():
    """Thick (tau=5), omega=0.8, HG g=0.5, callable BDRF rho=0.3 + b_pos=0.2."""
    print("\n--- Test 8e ---")
    tau_bot, omega, g = 5.0, 0.8, 0.5
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.9 * pi
    rho = 0.3
    b_pos = 0.2
    BDRF_callable = [lambda mu, mup, _r=rho: np.full(
        np.broadcast_shapes(np.shape(mu), np.shape(mup)), _r / pi
    )]
    g_l = _make_HG(g)
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8e", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, BDRF_Fourier_modes=[rho / pi],
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        b_pos=b_pos, BDRF_Fourier_modes=BDRF_callable,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_8f():
    """Conservative thick (tau=5, omega~1), Rayleigh, BDRF rho=0.1."""
    print("\n--- Test 8f ---")
    tau_bot, omega = 5.0, 1 - 1e-6
    mu0, I0, phi0 = 0.080442, pi, pi
    rho = 0.1
    BDRF = [rho / pi]
    g_l = _make_Rayleigh()
    g_l_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "8f", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_magnus_jax(
        tau_bot, lambda tau: omega, g_l_func, NQuad, NLeg, NFourier, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)
