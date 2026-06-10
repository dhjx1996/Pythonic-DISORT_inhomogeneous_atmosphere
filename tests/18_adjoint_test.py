"""
Test suite 18: discrete adjoint (gradient through the solve).

The retrieval needs the Jacobian d(observable)/d(optical properties). This is
free reverse-mode AD through diffrax's default RecursiveCheckpointAdjoint — the
"discrete adjoint" the report recommends. This test is the report's prescribed
validation experiment: `jax.grad` through `pydisort_riccati_jax` vs finite
differences.

The differentiated quantity is the package's **upwelling flux output**
`flux_up_ToA = 2*pi * sum_i w_i mu_i u+(0)_i` (the Gauss-Legendre quadrature of
the upwelling field) — the operationally relevant retrieval scalar, and the same
output whose traceability the float() fix restored. An eager `float(flux_up_ToA)`
inside the solver would concretize and make the whole function non-differentiable
(ConcretizationTypeError), so this `grad` would fail to even trace.

Partitioning:
  - The finite-difference cross-check lives in the **float64 partition**
    (`@pytest.mark.float64`): FD roundoff is ~eps/h, so in float32 the check is
    not meaningful (cf. miejax_lite's opt-in float64 `--run-gradients`). It is
    also slow (one reverse-mode solve compile, ~minutes). Run with
    `PYDISORT_RICCATI_JAX_X64=1 pytest -m float64 18_adjoint_test.py`.
  - The cheap `flux_up_ToA` type guard stays in the default float32 run.
"""
import numpy as np
from math import pi
import jax
import jax.numpy as jnp
import pytest

from pydisort_riccati_jax import (
    pydisort_riccati_jax, riccati_setup, riccati_solve,
)

NQuad = 6          # >= 6 (ARE ill-conditioned at NQuad=4)
NLeg = NQuad
tau_bot = 0.2      # thin -> few adaptive steps -> small reverse-mode graph
mu0, I0, phi0 = 0.5, 1.0, 0.0
_g_iso = np.zeros(NLeg); _g_iso[0] = 1.0


def _flux_up_ToA(omega, tol=1e-3):
    """The package's upwelling diffuse flux at ToA (2nd return value): the
    Gauss-Legendre weighted sum 2*pi * sum_i w_i mu_i u+(0)_i. Differentiable
    through the full Riccati solve, and the operationally relevant retrieval
    observable."""
    _, flux_up_ToA, _, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: omega, lambda tau: _g_iso, NQuad, mu0, I0, phi0,
        tol=tol,
    )
    return flux_up_ToA


@pytest.mark.float64
def test_grad_through_solve_matches_finite_difference():
    """jax.grad of the upwelling flux agrees with central finite differences
    w.r.t. omega — confirms the discrete-adjoint AD pipeline. Float64 only: FD
    roundoff (~eps/h) makes this meaningless in float32; in float64 we can demand
    tight agreement.

    Routed through the jit-able composable seam (``riccati_setup`` +
    ``riccati_solve``, the §C resolution) and **jitted**, so the forward compiles
    once and is reused across the gradient and both FD perturbations — instead of
    re-tracing the (non-jit-able) one-shot entry three times. The flux observable
    is identical to the legacy entry's (verified bit-for-bit by 21b)."""
    assert jax.config.jax_enable_x64, "FD gradient check requires float64"
    om0 = 0.8

    # Setup once; tol=1e-8 so the adaptive grid is stable and F is smooth in
    # omega -> FD reaches its float64 floor (~1e-10) and adjoint-vs-FD is a
    # genuine accuracy check, not noise.
    setup = riccati_setup(NQuad, I0, phi0, mu0, tol=1e-8)

    def flux_of_omega(omega):
        res = riccati_solve(setup, lambda tau: omega, lambda tau: _g_iso,
                            tau_bot)
        return 2 * pi * jnp.dot(setup.mu_arr_pos_jax * setup.W_jax, res.u_modes[0])

    f = jax.jit(flux_of_omega)                 # compile once, reuse for FD
    grad_f = jax.jit(jax.grad(flux_of_omega))  # compile once (reverse adjoint)

    g_ad = float(grad_f(om0))
    assert np.isfinite(g_ad), f"adjoint gradient (d flux_up / d omega) not finite: {g_ad}"

    h = 1e-6                                        # near-optimal for float64 central FD
    g_fd = (float(f(om0 + h)) - float(f(om0 - h))) / (2 * h)

    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
    print(f"\n  adjoint={g_ad:.10e}  fd={g_fd:.10e}  rel={rel:.2e}")
    assert rel < 1e-6, (
        f"adjoint vs finite-diff disagree: grad={g_ad:.9e}, fd={g_fd:.9e}, "
        f"rel={rel:.2e}"
    )


def test_flux_is_jax_scalar():
    """flux_up_ToA must be a traceable JAX scalar, NOT a Python float — an eager
    float() would have broken every grad-through-solve. Cheap type check (no
    reverse-mode compile)."""
    _, flux_up, _, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: 0.8, lambda tau: _g_iso, NQuad, mu0, I0, phi0,
        tol=1e-3,
    )
    assert isinstance(flux_up, jax.Array), f"flux_up is {type(flux_up)}, not jax.Array"
    assert jnp.ndim(flux_up) == 0 and np.isfinite(float(flux_up))
