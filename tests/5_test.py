"""
Test suite 5: Lambertian BDRF surface, beam source.

Corresponds to aspects of Stamnes Test Problems 7d and 11 (Lambertian
surface reflectance with direct-beam + diffuse scattering).
Exercises both the scalar and callable BDRF code paths in _solve_bc_riccati.

Reference: pydisort (single-layer, exact eigendecomposition).
Fallback:  reference_results/5{a-c}_test.npz
"""
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import get_reference, assert_close_to_reference

NQuad = 8
NLeg  = NQuad
NFourier = NQuad


def _make_isotropic():
    g_l = np.zeros(NLeg); g_l[0] = 1.0
    return g_l


def _make_HG(g):
    return g ** np.arange(NLeg)


def test_5a():
    """Isotropic scattering, scalar (Lambertian) BDRF with albedo=0.1."""
    print("\n--- Test 5a ---")
    tau_bot, omega = 0.5, 0.5
    mu0, I0, phi0  = 0.5, 1.0, 0.0
    rho = 0.1
    # pydisort BDRF convention: scalar mode-0 coefficient = rho/pi
    BDRF = [rho / pi]
    g_l = _make_isotropic()
    Leg_coeffs_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "5a", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_5b():
    """HG g=0.75, scalar Lambertian BDRF with albedo=0.5."""
    print("\n--- Test 5b ---")
    tau_bot, omega, g = 1.0, 0.8, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.5
    BDRF = [rho / pi]
    g_l = _make_HG(g)
    Leg_coeffs_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "5b", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_5c():
    """HG g=0.5, callable Lambertian BDRF (rho=0.3) — exercises callable path."""
    print("\n--- Test 5c ---")
    tau_bot, omega, g = 2.0, 0.7, 0.5
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    rho = 0.3

    # pydisort mode-m callable: BDRF^m(mu_i, mu_j) = rho/pi for m=0, 0 otherwise.
    # Lambertian surface has only an m=0 contribution.
    BDRF_callable = [lambda mu, mup, _r=rho: np.full(
        np.broadcast_shapes(np.shape(mu), np.shape(mup)), _r / pi
    )]
    # Scalar equivalent for the pydisort reference
    BDRF_scalar = [rho / pi]

    g_l = _make_HG(g)
    Leg_coeffs_func = lambda tau: g_l

    # Reference uses scalar BDRF (equivalent result for Lambertian surface)
    flux_ref, u0_ref = get_reference(
        "5c", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_scalar,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF_callable,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_5d():
    """Combined BDRF + b_pos + b_neg, HG g=0.5, thin atmosphere."""
    print("\n--- Test 5d ---")
    tau_bot, omega, g = 1.0, 0.7, 0.5
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.3
    b_pos, b_neg = 0.2, 0.1
    BDRF = [rho / pi]
    g_l = _make_HG(g)
    Leg_coeffs_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "5d", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)


def test_5e():
    """High surface albedo (rho=0.9), HG g=0.75, thin atmosphere."""
    print("\n--- Test 5e ---")
    tau_bot, omega, g = 0.5, 0.9, 0.75
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    rho = 0.9
    BDRF = [rho / pi]
    g_l = _make_HG(g)
    Leg_coeffs_func = lambda tau: g_l

    flux_ref, u0_ref = get_reference(
        "5e", tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    _, flux_mag, u0_mag, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        BDRF_Fourier_modes=BDRF,
    )
    assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref)
