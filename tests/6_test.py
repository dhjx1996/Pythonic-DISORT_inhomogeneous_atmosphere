"""
Test suite 6: tau-varying single-scattering albedo omega(tau).

This is the primary test of the new capability: a continuously varying omega
that cannot be handled by the standard pydisort eigendecomposition.

Verification strategy: multi-layer pydisort with piecewise-constant omega
(midpoint rule) must converge to the Riccati solution as NLayers increases.
The Riccati solver (tol=1e-5) is the reference.

Expected convergence: O(h^2) in the layer thickness h = tau_bot / NLayers.
"""
import numpy as np
import jax.numpy as jnp
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import (
    multilayer_pydisort_toa_full_phi, assert_convergence_phi, PHI_VALUES,
)

NQuad = 8
NLeg  = NQuad
NFourier = NQuad
N = NQuad // 2


def _u_phi(func, *args):
    """Extract (N, n_phi) array from a callable at PHI_VALUES."""
    return np.column_stack([func(*args, phi)[:N] for phi in PHI_VALUES])


def _ref_and_layers(tau_bot, omega_func, g_const, mu0, I0, phi0,
                    b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    """Run Riccati at tight tolerance (reference) and pydisort at 10 / 100 layers."""
    g_l = g_const ** np.arange(NLeg)
    Leg_coeffs_func = lambda tau: g_l

    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=1e-5,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    _, _, uf_c = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 10, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    _, _, uf_f = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 100, NQuad, NLeg, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )

    u_ref = _u_phi(u_ToA_func)
    u_coarse = _u_phi(uf_c, 0)
    u_fine = _u_phi(uf_f, 0)
    return u_ref, u_coarse, u_fine


def test_6a():
    """Linear omega(tau): 0.90 -> 0.40, isotropic scattering, tau_bot=1."""
    print("\n--- Test 6a: linear omega, isotropic ---")
    tau_bot, g = 1.0, 0.0   # isotropic: g=0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    omega_func = lambda tau: 0.90 + (0.40 - 0.90) * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)


def test_6b():
    """Linear omega(tau): 0.95 -> 0.70, HG g=0.75, tau_bot=2."""
    print("\n--- Test 6b: linear omega, HG g=0.75 ---")
    tau_bot, g = 2.0, 0.75
    mu0, I0, phi0 = 0.6, pi / 0.6, 0.0
    omega_func = lambda tau: 0.95 + (0.70 - 0.95) * tau / tau_bot

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)


def test_6c():
    """Sinusoidal omega(tau), HG g=0.5, tau_bot=1 — non-monotone variation."""
    print("\n--- Test 6c: sinusoidal omega, HG g=0.5 ---")
    tau_bot, g = 1.0, 0.5
    mu0, I0, phi0 = 0.7, 1.0, 0.0
    omega_func = lambda tau: 0.70 + 0.20 * jnp.sin(2 * tau)

    u_ref, u_coarse, u_fine = _ref_and_layers(
        tau_bot, omega_func, g, mu0, I0, phi0
    )
    assert_convergence_phi(u_ref, u_coarse, u_fine)
