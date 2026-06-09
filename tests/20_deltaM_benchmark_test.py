"""
Test suite 20: Delta-M + Nakajima-Tanaka (TMS) — stringent float64 benchmarks.

float64 partition (`@pytest.mark.float64`; run with
`PYDISORT_RICCATI_JAX_X64=1 pytest -m float64 20_deltaM_benchmark_test.py -s`).
Curated to bound runtime. Provides the quantitative "intended-effect" evidence
analogous to the PythonicDISORT documentation demonstrations:

  20a tau-varying order  : Design-A rate-based convergence (10x refinement),
                           delta-M + TMS on both sides.
  20b stream convergence : single homogeneous layer — raw(NQuad=16) vs
                           corrected(NQuad=16) vs truth(NQuad=64); corrected
                           error must collapse (the delta-M/NT intended effect).
                           This is the genuinely-new piece (the suite's existing
                           convergence tests target tau-discretization, not streams).
  20c single-layer exact : ours delta-M+TMS == pydisort(1 layer, f_arr, NT_cor)
                           at the SAME NQuad, to tight tolerance.
  20d Mie-coupled        : real water-cloud phase function via miejax_lite
                           (f = Leg_coeffs[NLeg] exercises the retrieval chain).
  20e FD gradient        : jax.grad through delta-M+TMS vs finite differences
                           (~1e-6), preserving the discrete adjoint.
"""
import numpy as np
from math import pi
import jax
import jax.numpy as jnp
import pytest

from pydisort_riccati_jax import (
    pydisort_riccati_jax, interpolate,
    riccati_setup, riccati_solve, eval_radiance,
)
from _helpers import (
    make_cloud_profile, multilayer_pydisort_toa_full_phi, pydisort_toa_full_phi,
    assert_convergence_phi, assert_close_to_reference_phi,
    assert_nonnegative_phi, find_min_radiance_phi, PHI_VALUES,
)

pytestmark = pytest.mark.float64

NQuad = 16
NLeg = NQuad
NLeg_all = 32
N = NQuad // 2


def _u_phi(func, *args):
    """(N, n_phi) array from a callable evaluated at PHI_VALUES."""
    return np.column_stack([np.asarray(func(*args, phi))[:N] for phi in PHI_VALUES])


# ---------------------------------------------------------------------------
# 20a — Design-A rate-based convergence (tau-varying), corrections on
# ---------------------------------------------------------------------------

def test_20a_tau_varying_convergence_design_A():
    """Multilayer pydisort (50/500 layers) converges toward the Riccati
    reference (tol=1e-8) with delta-M + TMS active on both sides."""
    print("\n--- Test 20a: Design-A convergence, delta-M + TMS ---")
    tau_bot = 10.0
    mu0, I0, phi0 = 0.6, 1.0, 0.0
    omega_func, Leg_coeffs_func = make_cloud_profile(
        tau_bot, omega_top=0.85, omega_bot=0.96,
        g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad, NLeg_all=NLeg_all,
    )

    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=True,
    )
    _, _, uf_c = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 50, NQuad, NLeg, mu0, I0, phi0,
        delta_M_scaling=True, NT_cor=True,
    )
    _, _, uf_f = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, 500, NQuad, NLeg, mu0, I0, phi0,
        delta_M_scaling=True, NT_cor=True,
    )

    u_ref = _u_phi(u_ToA_func)
    u_coarse = _u_phi(uf_c, 0)
    u_fine = _u_phi(uf_f, 0)
    # Design-A structure (rate-based, 10x refinement, abs_tol=1e-3). min_ratio is
    # relaxed from 10_key's 50 to 30 because the TMS integrand is sampled at layer
    # midpoints alongside (omega, g); its O(h^2) constant is larger, so the
    # measured ratio sits below the pure multiple-scattering case while remaining
    # clean second-order. (Documented Design-A threshold adjustment.)
    assert_convergence_phi(u_ref, u_coarse, u_fine, min_ratio=30, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# 20b — stream convergence (the delta-M/NT intended effect) — NEW
# ---------------------------------------------------------------------------

def _stream_convergence_case(tau_bot, omega, g, NLa, mu0, I0, phi0,
                             improve_factor, abs_tol):
    """raw(NQuad=16) vs corrected(NQuad=16) vs truth(NQuad=64), single layer."""
    from PythonicDISORT import subroutines

    Leg_full = g ** np.arange(NLa)
    omega_func = lambda tau: omega
    Leg_coeffs_func = lambda tau: g ** jnp.arange(NLa)

    # Truth: high-stream pydisort, full (untruncated) phase function.
    NQuad_truth = 64
    N_truth = NQuad_truth // 2
    _, _, uf_truth = pydisort_toa_full_phi(
        tau_bot, omega, NQuad_truth, Leg_full, mu0, I0, phi0,
        NLeg=NLa, delta_M_scaling=False, NT_cor=False,
    )
    mu_truth_pos = subroutines.Gauss_Legendre_quad(N_truth)[0]
    truth_interp = interpolate(lambda phi: np.asarray(uf_truth(0, phi))[:N_truth],
                               mu_truth_pos)

    # Raw: NQuad=16, no correction.
    mu_pos, _, _, u_raw, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=False, NLeg_all=NLa,
    )
    raw_interp = interpolate(u_raw, mu_pos)

    # Corrected: NQuad=16, delta-M + TMS.
    _, _, _, u_corr, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLa, NT_cor=True,
    )
    corr_interp = interpolate(u_corr, mu_pos)

    mu_eval = np.linspace(0.1, 0.95, 12)
    err_raw = err_corr = 0.0
    for phi in PHI_VALUES:
        t = np.asarray(truth_interp(mu_eval, phi))
        scale = max(float(np.max(np.abs(t))), 1e-8)
        err_raw = max(err_raw,
                      float(np.max(np.abs(np.asarray(raw_interp(mu_eval, phi)) - t))) / scale)
        err_corr = max(err_corr,
                       float(np.max(np.abs(np.asarray(corr_interp(mu_eval, phi)) - t))) / scale)
    print(f"  g={g}, tau={tau_bot}: max-rel-err raw={err_raw:.3e}  corr={err_corr:.3e}"
          f"  (improvement {err_raw / max(err_corr, 1e-15):.1f}x)")
    assert err_corr < err_raw / improve_factor, (
        f"correction did not substantially improve angular accuracy "
        f"(raw={err_raw:.3e}, corr={err_corr:.3e})"
    )
    assert err_corr < abs_tol, f"corrected error {err_corr:.3e} >= {abs_tol}"


def test_20b_stream_convergence_docs_problem():
    """PythonicDISORT-docs-style demonstration: HG g=0.75, near-conservative."""
    print("\n--- Test 20b: stream convergence (docs problem) ---")
    _stream_convergence_case(
        tau_bot=8.0, omega=1 - 1e-6, g=0.75, NLa=64,
        mu0=0.6, I0=100.0, phi0=0.0, improve_factor=3.0, abs_tol=5e-2,
    )


def test_20b_stream_convergence_cloud():
    """Cloud-like forward peak: g=0.85, omega~0.99."""
    print("\n--- Test 20b: stream convergence (cloud-like) ---")
    _stream_convergence_case(
        tau_bot=10.0, omega=0.99, g=0.85, NLa=64,
        mu0=0.6, I0=1.0, phi0=0.0, improve_factor=3.0, abs_tol=5e-2,
    )


# ---------------------------------------------------------------------------
# 20c — single-layer exact apples-to-apples vs pydisort NT_cor
# ---------------------------------------------------------------------------

def test_20c_single_layer_exact():
    """ours delta-M+TMS == pydisort(1 layer, f_arr=g^NLeg, NT_cor) at same NQuad."""
    print("\n--- Test 20c: single-layer exact match vs pydisort NT_cor ---")
    tau_bot, omega, g = 8.0, 0.99, 0.85
    mu0, I0, phi0 = 0.6, 1.0, 0.0
    Leg_full = g ** np.arange(NLeg_all)
    omega_func = lambda tau: omega
    Leg_coeffs_func = lambda tau: g ** jnp.arange(NLeg_all)

    _, _, uf_ref = pydisort_toa_full_phi(
        tau_bot, omega, NQuad, Leg_full, mu0, I0, phi0,
        NLeg=NLeg, delta_M_scaling=True, NT_cor=True,
    )
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=True,
    )
    # Same NQuad (identical mu nodes) and both exact for a homogeneous layer ->
    # tight tolerance. Residual is the GL tau-quadrature of the (smooth) TMS
    # integral vs pydisort's per-layer analytic form.
    assert_close_to_reference_phi(u_ToA_func, uf_ref, PHI_VALUES, N, rel_tol=2e-3)


# ---------------------------------------------------------------------------
# 20d — Mie-coupled phase function (retrieval chain)
# ---------------------------------------------------------------------------

def test_20d_mie_coupled():
    """Real water-cloud phase function from miejax_lite; exact match vs pydisort NT_cor.

    The key correctness check is the apples-to-apples match to PythonicDISORT's own NT_cor with
    the same Mie coefficients (exercises the f=Leg_coeffs[NLeg] retrieval-chain slice). We also
    check the correction does not *worsen* the radiance floor vs raw -- but NOT strict
    non-negativity: a sharp Mie peak under-resolved at NQuad=16 can leave a residual negative lobe
    that pydisort's NT_cor shares (see OUTSTANDING A), so strict positivity is not asserted here.
    """
    print("\n--- Test 20d: Mie-coupled delta-M + TMS ---")
    miejax_lite = pytest.importorskip("miejax_lite")

    NLa = 36
    precomp = miejax_lite.mie_legendre_precompute(max_nstop=512, NLeg=NLa)
    r_eff, wavelength, v_eff = 10.0, 2.13, 0.1
    omega, Leg_coeffs, _ = miejax_lite.mie_avg_legendre(
        r_eff, wavelength, v_eff, precomp
    )
    omega = float(omega)
    Leg_coeffs = np.asarray(Leg_coeffs, dtype=float)
    print(f"  Mie: omega={omega:.5f}, g={Leg_coeffs[1]:.4f}, NLeg_all={len(Leg_coeffs)}")

    tau_bot = 10.0
    mu0, I0, phi0 = 0.6, 1.0, 0.0
    omega_func = lambda tau: omega
    Leg_coeffs_func = lambda tau: jnp.asarray(Leg_coeffs)

    _, _, _, u_raw, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=False, NLeg_all=NLa,
    )
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLa, NT_cor=True,
    )
    raw_min = find_min_radiance_phi(u_raw, PHI_VALUES, N)
    corr_min = find_min_radiance_phi(u_ToA_func, PHI_VALUES, N)
    print(f"  raw min={raw_min:.3e}  corrected min={corr_min:.3e}")
    # Correction must not worsen the radiance floor (no strict-positivity claim).
    assert corr_min >= raw_min - 1e-9, "correction lowered the radiance floor"

    # Key check: apples-to-apples vs pydisort NT_cor at the same NQuad / coefficients.
    _, _, uf_ref = pydisort_toa_full_phi(
        tau_bot, omega, NQuad, Leg_coeffs, mu0, I0, phi0,
        NLeg=NLeg, delta_M_scaling=True, NT_cor=True,
    )
    assert_close_to_reference_phi(u_ToA_func, uf_ref, PHI_VALUES, N, rel_tol=3e-3)


# ---------------------------------------------------------------------------
# 20e — FD gradient through delta-M + TMS (discrete adjoint preserved)
# ---------------------------------------------------------------------------

def test_20e_fd_gradient():
    """jax.grad of flux_up_ToA and of a TMS-bearing radiance scalar vs central FD.

    Routed through the jit-able composable seam (``riccati_setup`` +
    ``riccati_solve`` + ``eval_radiance``, the §C resolution) and **jitted**, so
    each observable's forward compiles once and is reused across the gradient and
    both FD perturbations (4 traces -> 2 compiles per observable). The observables
    are identical to the legacy delta-M+TMS path (verified by 21b)."""
    print("\n--- Test 20e: FD gradient through delta-M + TMS ---")
    assert jax.config.jax_enable_x64, "FD gradient check requires float64"
    NQ, NLa = 8, 16
    tau_bot = 1.0
    mu0, I0, phi0 = 0.6, 1.0, 0.0
    g0 = 0.8

    setup = riccati_setup(
        NQ, I0, phi0, NLeg_all=NLa, delta_M_scaling=True, NT_cor=True,
        tol=1e-8, tol_azim=0.0,
    )
    omega_func = lambda tau: 0.95

    def _res(g):
        return riccati_solve(setup, omega_func, lambda tau: g ** jnp.arange(NLa),
                             tau_bot, mu0)

    def flux_of_g(g):
        return 2 * pi * jnp.dot(setup.mu_arr_pos_jax * setup.W_jax, _res(g).u_modes[0])

    def radiance_scalar_of_g(g):
        # TMS-bearing radiance at the quadrature nodes, phi=0.7 (== legacy
        # u_ToA_func(0.7), summed).
        return jnp.sum(eval_radiance(setup, _res(g), setup.mu_arr_pos_jax, 0.7))

    h = 1e-6
    for name, fn in [("flux_up", flux_of_g), ("u(phi) sum", radiance_scalar_of_g)]:
        f = jax.jit(fn)                  # compile once, reuse for FD
        grad_f = jax.jit(jax.grad(fn))   # compile once (reverse adjoint)
        g_ad = float(grad_f(g0))
        g_fd = (float(f(g0 + h)) - float(f(g0 - h))) / (2 * h)
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
        print(f"  d({name})/dg  adjoint={g_ad:.8e}  fd={g_fd:.8e}  rel={rel:.2e}")
        assert np.isfinite(g_ad), f"{name}: adjoint grad not finite"
        # 1e-4 (vs 18_adjoint_test's 1e-6): the delta-M path adds a tau* fixed-grid
        # interpolation and runs at tol=1e-8, so the central-FD floor sits at
        # ~1e-5 here (measured ~1.1e-5 for flux) rather than the smooth
        # constant-omega case. 1e-4 confirms the discrete adjoint without being
        # defeated by FD truncation/solver-residual noise.
        assert rel < 1e-4, f"{name}: adjoint vs FD disagree rel={rel:.2e}"
