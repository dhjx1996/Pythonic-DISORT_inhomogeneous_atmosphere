"""
Test suite 21: jit-ability of the solver (the composable retrieval seam).

Resolves docs/OUTSTANDING.md §C (the retrieval-cost blocker). The composable
seam (`riccati_setup` / `riccati_solve` / `eval_radiance`) splits host-side SciPy
setup from a traceable, jit-able solve of the traced inputs (`tau_bot` and the
optics closures); `mu0` is **static** (baked into `setup`). See
docs/DESIGN_DECISIONS.md §7.

    21b  parity: unjit riccati_solve == jax.jit(riccati_solve) == legacy
         pydisort_riccati_jax; eval_radiance == legacy u_ToA_func / interpolate;
         all vs the pydisort reference. With and without delta_M + NT_cor.

(The former 21a — the traced-mu0 associated-Legendre recurrence — and 21c — the
relative azimuthal Cauchy `calibrate_num_modes` — were removed when mu0 became
static and the Fourier modes moved to a `lax.scan` (OUTSTANDING §H): there is no
in-trace `P_l^m(-mu0)` recurrence any more, and mode truncation is now the
noise-aware `retrieval_oe.select_num_modes`, not a relative partial-sum test.)

Default partition is float32.
"""
from math import pi

import numpy as np
import jax
import jax.numpy as jnp

from pydisort_riccati_jax import (
    pydisort_riccati_jax,
    riccati_setup,
    riccati_solve,
    eval_radiance,
    interpolate,
)
from _helpers import pydisort_toa_full_phi, PHI_VALUES


# ===========================================================================
# 21b — parity: seam == jit(seam) == legacy == pydisort
# ===========================================================================

_NQuad = 8
_mu0, _I0, _phi0, _tau_bot = 0.6, 1.0, 0.0, 8.0
_omega_func = lambda t: 0.95
_g = 0.8
_Leg = lambda t: _g ** jnp.arange(_NQuad)


def _legacy_5tuple(**kw):
    return pydisort_riccati_jax(
        _tau_bot, _omega_func, _Leg, _NQuad, _mu0, _I0, _phi0, **kw
    )


def test_21b_parity_unjit_jit_legacy_noscaling():
    """riccati_solve (unjit) == jax.jit(riccati_solve) == legacy, no scaling."""
    print("\n--- Test 21b: parity (no delta-M) ---")
    setup = riccati_setup(_NQuad, _I0, _phi0, _mu0)

    res = riccati_solve(setup, _omega_func, _Leg, _tau_bot)

    # mu0 is static (in setup); only tau_bot is traced.
    jit_solve = jax.jit(
        lambda tb: riccati_solve(setup, _omega_func, _Leg, tb).u_modes
    )
    u_jit = jit_solve(_tau_bot)
    assert np.allclose(np.asarray(res.u_modes), np.asarray(u_jit), rtol=1e-4, atol=1e-6), \
        "jit(riccati_solve) != unjit riccati_solve"

    # Legacy 5-tuple shares the same core (return_grid=True path). The seam m=0
    # mode comes from save_grid=False (SaveAt(t1), float32 t1) while the legacy u0
    # comes from save_grid=True (SaveAt(steps), float(t1)); in float32 that endpoint
    # difference diverges the m=0 final state by ~1e-4 relative (same effect as the
    # flux parity below), so use rtol=1e-3, not 1e-4.
    mu_pos, flux, u0, uf, grid = _legacy_5tuple()
    assert np.allclose(np.asarray(res.u_modes[0]), np.asarray(u0), rtol=1e-3, atol=1e-6), \
        "seam u_modes[0] != legacy u0_ToA"

    # eval_radiance at the quadrature nodes reproduces legacy u_ToA_func(phi).
    # Scale-relative per phi (not pointwise rtol): the m=0 SaveAt divergence above
    # feeds into u(phi), and at the phi=pi back-azimuth null pointwise rtol is
    # meaningless in float32 (cf. the TMS check below and 19c).
    for phi in PHI_VALUES:
        ev = np.asarray(eval_radiance(setup, res, jnp.asarray(setup.mu_arr_pos), phi))
        leg = np.asarray(uf(phi))[: setup.N]
        scale = max(float(np.max(np.abs(leg))), 1e-8)
        rel = float(np.max(np.abs(ev - leg))) / scale
        assert rel < 1e-3, \
            f"eval_radiance(nodes) != legacy u_ToA_func at phi={phi:.3f}: rel={rel:.2e}"


def test_21b_parity_with_deltaM_NT():
    """Same parity with delta_M_scaling + NT_cor on (TMS through eval_radiance).

    Uses a 16-coefficient phase function consistently for *both* the seam and the
    legacy call (delta-M/TMS needs NLeg_all > NLeg)."""
    print("\n--- Test 21b: parity (delta-M + TMS) ---")
    NLa = 16
    Leg = lambda t: _g ** jnp.arange(NLa)   # NLeg_all=16 (NLeg defaults to NQuad=8)

    setup = riccati_setup(_NQuad, _I0, _phi0, _mu0, NLeg_all=NLa,
                          delta_M_scaling=True, NT_cor=True)
    res = riccati_solve(setup, _omega_func, Leg, _tau_bot)

    # Legacy one-shot with the SAME 16-coefficient phase function.
    mu_pos, flux, u0, uf, grid = pydisort_riccati_jax(
        _tau_bot, _omega_func, Leg, _NQuad, _mu0, _I0, _phi0,
        NLeg_all=NLa, delta_M_scaling=True, NT_cor=True,
    )
    # flux uses only m=0 (delta-M); seam flux from u_modes[0]. The seam m=0 ODE
    # is solved with SaveAt(t1) (inside lax.scan) while the legacy path uses
    # SaveAt(steps); in float32 that per-node ~1e-6 difference accumulates through
    # the quadrature dot product to ~1e-4 relative, so use the file's standard
    # float32 parity tolerance (rtol=1e-3) rather than a tighter 1e-4.
    flux_seam = 2 * pi * float(jnp.dot(setup.mu_arr_pos_jax * setup.W_jax, res.u_modes[0]))
    assert np.allclose(flux_seam, float(flux), rtol=1e-3, atol=1e-6), "flux parity (delta-M)"

    # eval_radiance (smooth interp + analytic TMS) == legacy interpolate at nodes.
    # Scale-relative per phi (pointwise rtol is meaningless at the phi=pi
    # back-azimuth null in float32 -- cf. 19c); the two are the same computation
    # bar the SaveAt(t1) vs SaveAt(steps) final-state difference (~1e-6).
    u_interp = interpolate(uf, mu_pos)
    for phi in PHI_VALUES:
        ev = np.asarray(eval_radiance(setup, res, jnp.asarray(setup.mu_arr_pos), phi))
        leg = np.asarray(u_interp(jnp.asarray(setup.mu_arr_pos), phi))
        scale = max(float(np.max(np.abs(leg))), 1e-8)
        rel = float(np.max(np.abs(ev - leg))) / scale
        assert rel < 1e-3, \
            f"eval_radiance != legacy interpolate (TMS) at phi={phi:.3f}: rel={rel:.2e}"


def test_21b_parity_vs_pydisort_reference():
    """The seam radiance matches the exact pydisort reference (float32 floor)."""
    print("\n--- Test 21b: parity vs pydisort reference ---")
    g_l = _g ** np.arange(_NQuad)
    _, _, uf_ref = pydisort_toa_full_phi(
        _tau_bot, 0.95, _NQuad, g_l, _mu0, _I0, _phi0,
    )
    setup = riccati_setup(_NQuad, _I0, _phi0, _mu0)
    res = riccati_solve(setup, _omega_func, _Leg, _tau_bot)
    N = setup.N
    for phi in PHI_VALUES:
        ev = np.asarray(eval_radiance(setup, res, jnp.asarray(setup.mu_arr_pos), phi))
        ref = np.asarray(uf_ref(0, phi))[:N]
        scale = max(float(np.max(np.abs(ref))), 1e-8)
        rel = float(np.max(np.abs(ev - ref))) / scale
        assert rel < 1e-2, f"seam vs pydisort rel={rel:.2e} at phi={phi:.3f}"
