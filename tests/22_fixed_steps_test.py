"""Frozen-step (StepTo) integration mode — verification (2026-07-14).

``fixed_n_steps`` replaces the adaptive PIDController with a uniform StepTo
σ-grid (SetupData.fixed_n_steps). Purpose: texture-free IC Jacobians — AD no
longer differentiates through the controller's accepted-step decisions, the
source of the high-frequency Jacobian texture that inflates DOFS/SIC at any
affordable adaptive tolerance (the 2026-07-14 idx98 tolerance-ladder finding).

Verified here (float32-friendly tolerances; the IC production use is float64):
  1. Radiance agreement: frozen-step u0_ToA matches the adaptive solve.
  2. Step-count convergence: doubling fixed_n_steps moves u0 less than the
     frozen-vs-adaptive gap (the mode converges to the same limit).
  3. Differentiability: jacfwd (ForwardMode) and jax.grad both run and agree
     with the adaptive-solve gradients to a loose tolerance.
  4. The default path (fixed_n_steps=None) is bit-identical to before (the
     controller branch is untouched).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jax
import jax.numpy as jnp
import diffrax

from pydisort_riccati_jax import (
    pydisort_riccati_jax, riccati_setup, riccati_solve, eval_radiance,
)

NQUAD, NLEG, TAU_BOT, MU0 = 16, 16, 3.0, 0.6


def _omega_func(tau):
    return 0.9 - 0.05 * tau / TAU_BOT


def _leg_func(tau):
    g = 0.75 + 0.05 * tau / TAU_BOT
    return g ** jnp.arange(NLEG)


def _one_shot(**kw):
    _, flux, u0, _, _ = pydisort_riccati_jax(
        TAU_BOT, _omega_func, _leg_func, NQUAD, MU0, 1.0, 0.0,
        NLeg=NLEG, NFourier=4, tol=1e-3, **kw)
    return np.asarray(u0), float(flux)


def test_frozen_matches_adaptive_and_converges():
    u_ad, f_ad = _one_shot()
    u_fx, f_fx = _one_shot(fixed_n_steps=64)
    u_fx2, f_fx2 = _one_shot(fixed_n_steps=128)
    scale = np.abs(u_ad).max()
    gap = np.abs(u_fx - u_ad).max() / scale
    gap2 = np.abs(u_fx2 - u_ad).max() / scale
    assert gap < 5e-3, f"frozen-64 vs adaptive rel gap {gap:.2e}"
    # doubling the steps converges toward the adaptive answer
    assert gap2 < gap or gap2 < 5e-4, (gap, gap2)
    assert abs(f_fx - f_ad) / abs(f_ad) < 5e-3


def test_default_path_unchanged():
    # fixed_n_steps=None must reproduce the adaptive solve exactly (same branch).
    u_a, _ = _one_shot()
    u_b, _ = _one_shot(fixed_n_steps=None)
    np.testing.assert_array_equal(u_a, u_b)


@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_frozen_step_gradients(mode):
    adj = diffrax.ForwardMode() if mode == "fwd" else None

    def make_obs(**setup_kw):
        setup = riccati_setup(NQUAD, 1.0, 0.0, MU0, NLeg=NLEG, NFourier=2,
                              tol=1e-3, adjoint=adj, **setup_kw)

        def obs(scale):
            of = lambda tau: scale * _omega_func(tau)
            res = riccati_solve(setup, of, _leg_func, TAU_BOT)
            return jnp.sum(eval_radiance(setup, res, jnp.asarray([0.8]),
                                         jnp.asarray([0.0])))
        return obs

    d = jax.jacfwd if mode == "fwd" else jax.grad
    g_ad = float(d(make_obs())(1.0))
    g_fx = float(d(make_obs(fixed_n_steps=64))(1.0))
    assert np.isfinite(g_fx) and abs(g_fx) > 0
    assert abs(g_fx - g_ad) / abs(g_ad) < 5e-2, (g_fx, g_ad)
    # Gradient-frozen adaptive mode: primal bit-identical to plain adaptive,
    # gradient finite and near the adaptive one (the controller-channel term
    # it removes is a small correction on a smooth problem).
    obs_sg, obs_pl = make_obs(freeze_step_grads=True), make_obs()
    assert float(obs_sg(1.0)) == float(obs_pl(1.0))
    g_sg = float(d(obs_sg)(1.0))
    assert np.isfinite(g_sg) and abs(g_sg) > 0
    assert abs(g_sg - g_ad) / abs(g_ad) < 5e-2, (g_sg, g_ad)
