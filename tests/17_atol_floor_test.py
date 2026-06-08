"""
Test suite 17: dtype-aware tolerance floor (float32 robustness).

The adaptive PIDController error test is ||error / (atol + rtol*|y|)|| ~ 1.
Float32 roundoff is amplified by the nonlinear Riccati matrix products to an
effective ~1e-3 accuracy floor on thick atmospheres, so a `tol` below that is
unsatisfiable and the controller shrinks dt until it raises `max_steps`.
`_floored_tolerances` clamps `tol` to the dtype's achievable accuracy (1e-3 in
float32; a few*eps in float64) so a too-tight request *caps accuracy* instead of
crashing, keeping the established `atol = tol*1e-3` coupling.

Fast (one thin/thick solve + pure-function checks); default float32 partition.
"""
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from _riccati_solver_jax import _floored_tolerances, _RTOL_FLOOR_F32
from pydisort_riccati_jax import pydisort_riccati_jax

NQuad = 8
NLeg = NQuad
N = NQuad // 2
_g_iso = np.zeros(NLeg); _g_iso[0] = 1.0
Leg_coeffs_func = lambda tau: _g_iso
_F32 = (jnp.result_type(float) == jnp.float32)


def test_floor_inactive_at_production_tol():
    """At the float32 production tol=1e-3 the floor must NOT bind: behaviour is
    bit-identical to the original `rtol = tol`, `atol = tol*1e-3` coupling."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any clamp warning would fail here
        rtol, atol = _floored_tolerances(1e-3)
    assert rtol == 1e-3, f"rtol changed at production tol: {rtol}"
    assert atol == 1e-3 * 1e-3, f"atol changed at production tol: {atol}"


def test_floor_clamps_below_production_in_float32():
    """A sub-floor tol is clamped to the dtype accuracy floor (with a warning)
    in float32; in float64 the same tol passes through untouched."""
    if _F32:
        # Deterministic clamp behaviour (independent of warning/global state):
        rtol, atol = _floored_tolerances(1e-4)
        assert rtol == _RTOL_FLOOR_F32 == 1e-3
        assert atol == _RTOL_FLOOR_F32 * 1e-3
        # ...and it warns. Clear this module's warning registry and force the
        # "always" filter so suite ordering can't dedup the warning away.
        globals().pop("__warningregistry__", None)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _floored_tolerances(1e-4)
        assert any(issubclass(w.category, UserWarning)
                   and "accuracy floor" in str(w.message) for w in rec), \
            "sub-floor tol did not warn"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rtol, atol = _floored_tolerances(1e-4)
        assert rtol == 1e-4 and atol == 1e-4 * 1e-3   # float64: untouched


def test_thick_tiny_tol_returns_not_raises():
    """Headline fix: a thick atmosphere with a too-tight tol now RETURNS a finite
    ToA field (capped at the dtype accuracy floor) instead of raising
    EquinoxRuntimeError(max_steps). Previously tol=1e-6 at tau=32 crashed."""
    tau_bot = 32.0
    mu0, I0, phi0 = 0.5, 1.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # float32 clamp warning is expected
        mu_arr_pos, flux_up, u0_tiny, _, _ = pydisort_riccati_jax(
            tau_bot, lambda tau: 0.99, Leg_coeffs_func, NQuad, mu0, I0, phi0,
            tol=1e-6,
        )
    u0_tiny = np.asarray(u0_tiny)
    assert u0_tiny.shape == (N,)
    assert np.all(np.isfinite(u0_tiny)), "ToA field not finite"
    assert np.isfinite(flux_up)

    # Clamping means tol=1e-6 runs at the 1e-3 floor, so it agrees with the
    # production tol=1e-3 solve (bit-identical in a clean process; we only assert
    # float32-accuracy agreement here to stay robust to cross-test XLA state).
    # Graceful degradation, not a crash.
    _, _, u0_prod, _, _ = pydisort_riccati_jax(
        tau_bot, lambda tau: 0.99, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-3,
    )
    u0_prod = np.asarray(u0_prod)
    rel = np.max(np.abs(u0_tiny - u0_prod)) / max(np.max(np.abs(u0_prod)), 1e-8)
    assert rel < 1e-2, f"tiny-tol vs production-tol disagree: rel={rel:.2e}"
