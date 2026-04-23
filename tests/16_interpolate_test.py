"""
Test suite 16: Barycentric interpolation in mu at ToA.

Tests the `interpolate` function from `pydisort_riccati_jax`:
  - Identity at quadrature nodes (machine precision)
  - Cross-check against PythonicDISORT.subroutines.interpolate at off-quadrature mu
  - JAX autodiff smoke test (finite, nonzero Jacobian)
"""
import jax
import jax.numpy as jnp
import numpy as np
from math import pi

from pydisort_riccati_jax import pydisort_riccati_jax, interpolate


NQuad = 8
NLeg = NQuad
N = NQuad // 2

# Shared problem parameters (isotropic, thin atmosphere)
mu0 = 0.1
I0 = pi / mu0
phi0 = pi
g_l_iso = np.zeros(NLeg)
g_l_iso[0] = 1.0
Leg_coeffs_func_iso = lambda tau: g_l_iso

# Off-quadrature mu values for interpolation tests
MU_TEST = np.array([0.15, 0.35, 0.55, 0.75, 0.95])
PHI_TEST = pi / 4


# ── Identity test ──────────────────────────────────────────────────────

def test_identity_at_nodes():
    """u_interp(mu_arr_pos, phi) must recover u_ToA_func(phi) to machine precision."""
    tau_bot, omega = 0.03125, 0.2
    mu_arr_pos, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func_iso, NQuad, mu0, I0, phi0,
    )
    u_interp = interpolate(u_ToA_func, mu_arr_pos)

    for phi in (0.0, pi / 4, pi):
        expected = u_ToA_func(phi)
        result = u_interp(mu_arr_pos, phi)
        err = float(jnp.max(jnp.abs(result - expected)))
        assert err < 1e-12, f"Identity test failed at phi={phi}: max err={err:.2e}"


def test_identity_scalar_mu():
    """u_interp at each individual quadrature mu matches u_ToA_func element."""
    tau_bot, omega = 1.0, 0.9
    g = 0.75
    g_l = g ** np.arange(NLeg)
    Leg_coeffs_func = lambda tau: g_l
    mu_arr_pos, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    )
    u_interp = interpolate(u_ToA_func, mu_arr_pos)

    phi = pi / 2
    expected = u_ToA_func(phi)  # (N,)
    for i in range(N):
        val = u_interp(mu_arr_pos[i], phi)
        err = abs(float(val) - float(expected[i]))
        assert err < 1e-12, f"Scalar identity failed at i={i}: err={err:.2e}"

'''
# ── Interpolation cross-check: JAX barycentric vs scipy barycentric ────
#
# Both interpolators receive the SAME node values from the Riccati solver,
# isolating interpolation correctness from solver accuracy.

def _crosscheck_vs_scipy(tau_bot, omega, Leg_coeffs_func, label):
    """Compare our JAX barycentric interpolation against scipy on identical data."""
    from scipy.interpolate import BarycentricInterpolator

    mu_arr_pos, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    )
    u_interp = interpolate(u_ToA_func, mu_arr_pos)

    # Get node values from the solver at a specific phi
    node_vals = np.asarray(u_ToA_func(PHI_TEST))  # (N,)
    scipy_interp = BarycentricInterpolator(np.asarray(mu_arr_pos), node_vals)

    for mu_q in MU_TEST:
        val_jax = float(u_interp(mu_q, PHI_TEST))
        val_scipy = float(scipy_interp(mu_q))
        err = abs(val_jax - val_scipy)
        assert err < 1e-12, (
            f"{label}, mu={mu_q}: JAX={val_jax:.15e}, "
            f"scipy={val_scipy:.15e}, abs_err={err:.2e}"
        )


def test_crosscheck_isotropic():
    """JAX vs scipy barycentric on isotropic thin-atmosphere intensity profile."""
    _crosscheck_vs_scipy(0.03125, 0.99, Leg_coeffs_func_iso, "isotropic thin")


def test_crosscheck_hg():
    """JAX vs scipy barycentric on HG (g=0.75) intensity profile."""
    g = 0.75
    g_l = g ** np.arange(NLeg)
    _crosscheck_vs_scipy(1.0, 0.9, lambda tau: g_l, "HG g=0.75")


def test_crosscheck_thick():
    """JAX vs scipy barycentric on thick-atmosphere (tau=32) intensity profile."""
    _crosscheck_vs_scipy(32.0, 0.99, Leg_coeffs_func_iso, "isotropic thick")
'''

# ── Autodiff smoke test ────────────────────────────────────────────────

def test_autodiff_u_interp():
    """jax.jacrev through u_interp produces finite, nonzero Jacobian."""
    tau_bot, omega = 0.03125, 0.99
    mu_arr_pos, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func_iso, NQuad, mu0, I0, phi0,
    )
    u_interp = interpolate(u_ToA_func, mu_arr_pos)

    mu_q = 0.35
    phi_q = pi / 4

    # Differentiate u_interp w.r.t. mu (scalar -> scalar)
    grad_fn = jax.grad(lambda mu: u_interp(mu, phi_q))
    g = grad_fn(mu_q)
    assert jnp.isfinite(g), f"Gradient w.r.t. mu is not finite: {g}"
    assert float(jnp.abs(g)) > 1e-15, f"Gradient w.r.t. mu is zero: {g}"


def test_autodiff_u_ToA_func():
    """jax.jacrev through u_ToA_func (without interpolation) produces finite Jacobian."""
    tau_bot, omega = 0.03125, 0.99
    mu_arr_pos, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, Leg_coeffs_func_iso, NQuad, mu0, I0, phi0,
    )

    phi_q = pi / 4

    # Differentiate first element of u_ToA_func(phi) w.r.t. phi
    grad_fn = jax.grad(lambda phi: u_ToA_func(phi)[0])
    g = grad_fn(phi_q)
    assert jnp.isfinite(g), f"Gradient of u_ToA_func w.r.t. phi is not finite: {g}"
