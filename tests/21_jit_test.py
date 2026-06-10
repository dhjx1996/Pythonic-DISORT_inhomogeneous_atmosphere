"""
Test suite 21: jit-ability of the solver + DISORT azimuthal convergence.

Resolves docs/OUTSTANDING.md §C (the retrieval-cost blocker). The composable
seam (`riccati_setup` / `riccati_solve` / `calibrate_num_modes` /
`eval_radiance`) splits host-side SciPy setup from a traceable, jit-able solve
of the traced inputs (tau_bot, mu0, and the optics closures). See
docs/DESIGN_DECISIONS.md §7.

    21a  custom associated-Legendre recurrence P_l^m(-mu0) vs scipy.lpmv, with a
         profile and a gradient-finiteness check.
    21b  parity: unjit riccati_solve == jax.jit(riccati_solve) == legacy
         pydisort_riccati_jax; eval_radiance == legacy u_ToA_func / interpolate;
         all vs the pydisort reference. With and without delta_M + NT_cor.
    21c  DISORT azimuthal convergence (Cauchy, STWLE2000 §3.7 p.89):
         calibrate_num_modes reproduces the exact stop, and the K-mode radiance
         matches the full-NFourier radiance within tol_azim.

Default partition is float32.
"""
import sys
import time
from math import pi

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import scipy.special as sp

from pydisort_riccati_jax import (
    pydisort_riccati_jax,
    riccati_setup,
    riccati_solve,
    calibrate_num_modes,
    eval_radiance,
    interpolate,
)
from _riccati_solver_jax import _assoc_legendre_neg_mu0_jax
from _helpers import pydisort_toa_full_phi, PHI_VALUES


# ===========================================================================
# 21a — custom associated-Legendre recurrence
# ===========================================================================

def test_21a_assoc_legendre_matches_scipy():
    """P_l^m(-mu0) from the custom JAX recurrence vs scipy.special.lpmv.

    This is *our* code (JAX has no usable associated-Legendre — lpmn is
    deprecated, there is no lpmv). It is what lets mu0 be a traced solve input. 
    Tolerance is the active dtype's floor: ~1e-3 (loose) in float32 
    — the high-m values span ~1e15 so float32 relative roundoff shows — 
    and tight in float64.
    """
    print("\n--- Test 21a: associated-Legendre recurrence vs scipy ---")
    tol = 5e-3 if jnp.result_type(float) == jnp.float32 else 1e-10
    worst = 0.0
    for NLeg in (8, 16):
        for m in range(NLeg):
            for mu0 in np.linspace(0.05, 0.98, 16):
                got = np.asarray(_assoc_legendre_neg_mu0_jax(m, NLeg, float(mu0)))
                ref = np.array([sp.lpmv(m, l, -mu0) for l in range(m, NLeg)])
                if len(ref) == 0:
                    continue
                scale = max(float(np.max(np.abs(ref))), 1e-12)
                worst = max(worst, float(np.max(np.abs(got - ref))) / scale)
    print(f"  worst relative error vs scipy = {worst:.2e} (tol {tol:g})")
    assert worst < tol, f"recurrence disagrees with scipy.lpmv (rel {worst:.2e})"


def test_21a_assoc_legendre_gradient_finite():
    """d/dmu0 of the recurrence is finite and matches central finite differences.

    Gradient finiteness through P_l^m(-mu0) is the prerequisite for traced-mu0
    gradients (OUTSTANDING E). FD is meaningful even in float32 here because the
    function is a smooth polynomial in mu0 (no solver noise)."""
    print("\n--- Test 21a: associated-Legendre gradient finiteness ---")

    def scalar(mu0):
        return jnp.sum(_assoc_legendre_neg_mu0_jax(5, 12, mu0))

    g = float(jax.grad(scalar)(0.6))
    assert np.isfinite(g), "gradient not finite"
    h = 1e-3
    fd = (float(scalar(0.6 + h)) - float(scalar(0.6 - h))) / (2 * h)
    rel = abs(g - fd) / max(abs(fd), 1e-9)
    print(f"  grad={g:.6e}  fd={fd:.6e}  rel={rel:.2e}")
    assert rel < 1e-2, f"gradient disagrees with FD (rel {rel:.2e})"


def test_21a_assoc_legendre_profile():
    """Profile the in-trace cost of the recurrence (jitted, all modes).

    The eager per-call cost is dominated by GPU launch latency and is NOT
    representative; under jit the recurrence is fused into the graph. We record
    the warm (cached) cost to document it is negligible relative to the ODE
    solve. Not an assertion on wall-time (machine-dependent) beyond a sanity
    ceiling."""
    print("\n--- Test 21a: associated-Legendre in-jit profile ---")
    NLeg = 16

    @jax.jit
    def all_modes(mu0):
        return jnp.concatenate(
            [_assoc_legendre_neg_mu0_jax(m, NLeg, mu0) for m in range(NLeg)]
        )

    all_modes(0.6).block_until_ready()  # warm
    ts = []
    for v in (0.5, 0.55, 0.6, 0.65, 0.7):
        t0 = time.perf_counter()
        all_modes(v).block_until_ready()
        ts.append(time.perf_counter() - t0)
    warm_ms = float(np.mean(ts)) * 1e3
    print(f"  warm (cached) cost of all {NLeg} modes = {warm_ms:.3f} ms")
    assert warm_ms < 200.0, "recurrence unexpectedly slow even cached"


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
    setup = riccati_setup(_NQuad, _I0, _phi0, tol_azim=0.0)

    res = riccati_solve(setup, _omega_func, _Leg, _tau_bot, _mu0)

    jit_solve = jax.jit(
        lambda tb, m0: riccati_solve(setup, _omega_func, _Leg, tb, m0).u_modes
    )
    u_jit = jit_solve(_tau_bot, _mu0)
    assert np.allclose(np.asarray(res.u_modes), np.asarray(u_jit), rtol=1e-4, atol=1e-6), \
        "jit(riccati_solve) != unjit riccati_solve"

    # Legacy 5-tuple shares the same core (return_grid=True path).
    mu_pos, flux, u0, uf, grid = _legacy_5tuple()
    assert np.allclose(np.asarray(res.u_modes[0]), np.asarray(u0), rtol=1e-4, atol=1e-6), \
        "seam u_modes[0] != legacy u0_ToA"

    # eval_radiance at the quadrature nodes reproduces legacy u_ToA_func(phi).
    for phi in PHI_VALUES:
        ev = np.asarray(eval_radiance(setup, res, jnp.asarray(setup.mu_arr_pos), phi))
        leg = np.asarray(uf(phi))[: setup.N]
        assert np.allclose(ev, leg, rtol=1e-4, atol=1e-6), \
            f"eval_radiance(nodes) != legacy u_ToA_func at phi={phi:.3f}"


def test_21b_parity_with_deltaM_NT():
    """Same parity with delta_M_scaling + NT_cor on (TMS through eval_radiance).

    Uses a 16-coefficient phase function consistently for *both* the seam and the
    legacy call (delta-M/TMS needs NLeg_all > NLeg)."""
    print("\n--- Test 21b: parity (delta-M + TMS) ---")
    NLa = 16
    Leg = lambda t: _g ** jnp.arange(NLa)   # NLeg_all=16 (NLeg defaults to NQuad=8)

    setup = riccati_setup(_NQuad, _I0, _phi0, NLeg_all=NLa, tol_azim=0.0,
                          delta_M_scaling=True, NT_cor=True)
    res = riccati_solve(setup, _omega_func, Leg, _tau_bot, _mu0)

    # Legacy one-shot with the SAME 16-coefficient phase function.
    mu_pos, flux, u0, uf, grid = pydisort_riccati_jax(
        _tau_bot, _omega_func, Leg, _NQuad, _mu0, _I0, _phi0,
        NLeg_all=NLa, delta_M_scaling=True, NT_cor=True,
    )
    # flux uses only m=0 (delta-M); seam flux from u_modes[0].
    flux_seam = 2 * pi * float(jnp.dot(setup.mu_arr_pos_jax * setup.W_jax, res.u_modes[0]))
    assert np.allclose(flux_seam, float(flux), rtol=1e-4, atol=1e-6), "flux parity (delta-M)"

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
    setup = riccati_setup(_NQuad, _I0, _phi0, tol_azim=0.0)
    res = riccati_solve(setup, _omega_func, _Leg, _tau_bot, _mu0)
    N = setup.N
    for phi in PHI_VALUES:
        ev = np.asarray(eval_radiance(setup, res, jnp.asarray(setup.mu_arr_pos), phi))
        ref = np.asarray(uf_ref(0, phi))[:N]
        scale = max(float(np.max(np.abs(ref))), 1e-8)
        rel = float(np.max(np.abs(ev - ref))) / scale
        assert rel < 1e-2, f"seam vs pydisort rel={rel:.2e} at phi={phi:.3f}"


# ===========================================================================
# 21c — DISORT azimuthal convergence / Cauchy criterion
# ===========================================================================

def _reference_cauchy_K(u_modes, mu_obs, phi_obs, phi0, tol_azim):
    """A second, independent reimplementation of the STWLE2000 p.89 criterion,
    operating directly on already-interpolated per-mode user-angle radiances —
    used to cross-check calibrate_num_modes."""
    NF = u_modes.shape[0]
    if tol_azim <= 0:
        return NF
    I_K = np.zeros((len(mu_obs), len(phi_obs)))
    consecutive, K = 0, NF
    for m in range(NF):
        term = u_modes[m][:, None] * np.cos(m * (phi0 - phi_obs))[None, :]
        I_K = I_K + term
        denom = np.where(np.abs(I_K) < 1e-30, 1e-30, np.abs(I_K))
        if float(np.max(np.abs(term) / denomS)) <= tol_azim:
            consecutive += 1
            if consecutive >= 2:
                K = m + 1
                break
        else:
            consecutive = 0
    return K


def test_21c_cauchy_matches_reference():
    """calibrate_num_modes reproduces the exact p.89 stop (twice-rule, max over
    user angles); tol_azim=0 returns all NFourier modes; K is monotone in
    tol_azim; and the K-mode radiance is within ~tol_azim of the full series.

    Structured to reuse the (expensive, un-jit) full solve: one full solve drives
    both the independent Cauchy reference and the K-vs-full radiance check.
    """
    print("\n--- Test 21c: DISORT azimuthal convergence (Cauchy) ---")
    from pydisort_riccati_jax import _barycentric_interpolate
    NQ = 8
    mu0, I0, phi0, tau_bot = 0.6, 1.0, 0.0, 6.0
    omega_func = lambda t: 0.97
    Leg = lambda t: 0.8 ** jnp.arange(NQ)
    mu_obs = np.array([0.3, 0.55, 0.85])
    phi_obs = np.array([0.0, pi / 4, pi / 2, pi])

    # One full NFourier solve (tol_azim-independent), reused throughout.
    setup_full = riccati_setup(NQ, I0, phi0, tol_azim=0.0)
    full = riccati_solve(setup_full, omega_func, Leg, tau_bot, mu0)
    interp = np.asarray(_barycentric_interpolate(
        jnp.asarray(mu_obs), setup_full.mu_nodes,
        jnp.asarray(np.asarray(full.u_modes).T), setup_full.bary_weights))  # (M, NF)
    r_full = np.asarray(eval_radiance(setup_full, full, mu_obs, phi_obs))     # (M, P)

    # tol_azim = 0 -> all modes (ACCUR=0 semantics).
    assert calibrate_num_modes(setup_full, omega_func, Leg, tau_bot, mu0,
                               mu_obs, phi_obs) == setup_full.NFourier

    Ks = []
    for tol_azim in (1e-2, 1e-3, 1e-4):
        setup = riccati_setup(NQ, I0, phi0, tol_azim=tol_azim)
        K = calibrate_num_modes(setup, omega_func, Leg, tau_bot, mu0,
                                mu_obs, phi_obs)
        K_ref = _reference_cauchy_K(interp.T, mu_obs, phi_obs, phi0, tol_azim)
        print(f"  tol_azim={tol_azim:.0e}: K={K}  K_ref={K_ref}  (NFourier={setup.NFourier})")
        assert K == K_ref, f"Cauchy K mismatch: {K} != {K_ref}"
        assert 1 <= K <= setup.NFourier
        Ks.append(K)

        # The K-mode radiance is within ~tol_azim of the full series.
        partial = riccati_solve(setup, omega_func, Leg, tau_bot, mu0, num_modes=K)
        r_K = np.asarray(eval_radiance(setup, partial, mu_obs, phi_obs))
        scale = max(float(np.max(np.abs(r_full))), 1e-8)
        rel = float(np.max(np.abs(r_K - r_full))) / scale
        assert rel < max(tol_azim * 5, 1e-3), \
            f"K-mode radiance off by {rel:.2e} (tol_azim={tol_azim:.0e})"

    # Monotone: tighter tol_azim keeps no fewer modes.
    assert all(Ks[i] <= Ks[i + 1] for i in range(len(Ks) - 1)), \
        f"K not monotone non-decreasing as tol tightens: {Ks}"
