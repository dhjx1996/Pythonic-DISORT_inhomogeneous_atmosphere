"""
Test suite 14: JAX Riccati solver (standalone).

Validates the Kvaerno5 (order 5, L-stable) solver for the Riccati +
companion T + beam-source s equations in isolation (before integration
into pydisort_riccati_jax).

  14a: Homogeneous atmosphere — R_up converges (loose vs tight tol).
  14b: Tol-sweep — error tracks tolerance, steps decrease at loose tol.
  14c: T validation — T_up propagates boundary correctly.
  14d: R_up = R_down for homogeneous slab.
  14e: s_up validation — beam source via loose vs tight tolerance.
  14f: T_up with tau-varying properties — vs multilayer pydisort (catches ODE ordering errors).
"""
import numpy as np
import jax.numpy as jnp
from math import pi
from PythonicDISORT import subroutines
from _riccati_solver_jax import (
    _riccati_forward_jax, _riccati_backward_jax,
    _make_alpha_beta_funcs_jax, _make_q_funcs_jax,
    _precompute_legendre, _riccati_rhs_jax,
)
from pydisort_riccati_jax import pydisort_riccati_jax
from _helpers import multilayer_pydisort_toa_full_phi

NQuad = 16
NLeg = NQuad
N = NQuad // 2

# Setup: homogeneous atmosphere (omega=0.99, g=0.85)
omega, g = 0.99, 0.85
g_l = g ** np.arange(NLeg)
Leg_coeffs_func = lambda tau: g_l
omega_func = lambda tau: omega

mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos = jnp.array(mu_arr_pos)
W = jnp.array(W)
M_inv = 1.0 / mu_arr_pos

mu0_setup = 0.5  # for precompute_legendre
leg_data = _precompute_legendre(0, NLeg, mu_arr_pos, mu0_setup)
alpha_func, beta_func = _make_alpha_beta_funcs_jax(
    omega_func, Leg_coeffs_func, 0, leg_data, mu_arr_pos, W, M_inv, N,
)

# Constant alpha/beta for the homogeneous reference checks
alpha = np.asarray(alpha_func(0.0))
beta = np.asarray(beta_func(0.0))


def test_14a():
    """R_up from Kvaerno5 converges: loose tol vs tight tol."""
    print("\n--- Test 14a: Kvaerno5 R_up convergence ---")
    tau_sub = 20.0

    # Tight-tolerance reference
    R_up_ref, _, _, _ = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-10,
    )

    # Loose tolerance
    R_up, T_up, _, grid = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-4,
    )

    R_up_ref, R_up = np.asarray(R_up_ref), np.asarray(R_up)
    rel_err_R = np.linalg.norm(R_up - R_up_ref, 'fro') / max(
        np.linalg.norm(R_up_ref, 'fro'), 1e-10
    )
    n_steps = len(grid) - 1
    print(f"  Kvaerno5 steps: {n_steps}")
    print(f"  R_up rel_err: {rel_err_R:.3e}")
    assert rel_err_R < 1e-3, f"R_up rel_err {rel_err_R:.3e} >= 1e-3"


def test_14b():
    """Tol-sweep: Kvaerno5 error tracks tolerance, steps decrease at loose tol."""
    print("\n--- Test 14b: Kvaerno5 tol-sweep ---")
    tau_sub = 5.0

    # Tight-tolerance reference
    R_ref, _, _, _ = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-10,
    )
    R_ref = np.asarray(R_ref)

    tols = [1e-2, 1e-3, 1e-4, 1e-5]
    errs = []
    steps_list = []
    for tol in tols:
        R_up, _, _, grid = _riccati_forward_jax(
            alpha_func, beta_func, tau_sub, N, tol=tol,
        )
        R_up = np.asarray(R_up)
        err = np.linalg.norm(R_up - R_ref, 'fro') / max(
            np.linalg.norm(R_ref, 'fro'), 1e-10
        )
        errs.append(err)
        steps_list.append(len(grid) - 1)

    print(f"  {'tol':>8s}  {'err':>10s}  {'steps':>6s}")
    for i in range(len(tols)):
        print(f"  {tols[i]:8.0e}  {errs[i]:10.3e}  {steps_list[i]:6d}")

    # Error should decrease with tolerance
    for i in range(1, len(tols)):
        assert errs[i] <= errs[i - 1], (
            f"Error did not decrease: tol {tols[i]:.0e} err {errs[i]:.3e} "
            f">= tol {tols[i-1]:.0e} err {errs[i-1]:.3e}"
        )

    # Tightest tolerance should be very accurate
    assert errs[-1] < 1e-4, f"err at tol=1e-5: {errs[-1]:.3e} >= 1e-4"


def test_14c():
    """T validation: R_up*b_neg + T_up*b_pos matches tight-tol reference."""
    print("\n--- Test 14c: T validation via boundary propagation ---")
    tau_sub = 10.0

    # Construct b_pos and b_neg test vectors
    b_neg = np.random.RandomState(42).rand(N)
    b_pos = np.random.RandomState(43).rand(N)

    # Tight-tolerance reference
    R_up_ref, T_up_ref, _, _ = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-10,
    )
    R_up_ref, T_up_ref = np.asarray(R_up_ref), np.asarray(T_up_ref)
    I_up_ref = R_up_ref @ b_neg + T_up_ref @ b_pos

    # Loose tolerance
    R_up, T_up, _, grid = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-4,
    )
    R_up, T_up = np.asarray(R_up), np.asarray(T_up)
    I_up = R_up @ b_neg + T_up @ b_pos

    rel_err = np.linalg.norm(I_up - I_up_ref) / max(
        np.linalg.norm(I_up_ref), 1e-10
    )
    n_steps = len(grid) - 1
    print(f"  Kvaerno5 steps: {n_steps}")
    print(f"  I^+(0) rel_err: {rel_err:.3e}")
    assert rel_err < 1e-3, f"I^+(0) rel_err {rel_err:.3e} >= 1e-3"


def test_14d():
    """R_up = R_down for homogeneous slab."""
    print("\n--- Test 14d: R_up = R_down (homogeneous symmetry) ---")
    tau_sub = 10.0

    R_up, _, _, _ = _riccati_forward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-4,
    )
    R_down, _, _, _ = _riccati_backward_jax(
        alpha_func, beta_func, tau_sub, N, tol=1e-4,
    )

    R_up, R_down = np.asarray(R_up), np.asarray(R_down)
    diff = np.linalg.norm(R_up - R_down, 'fro') / max(
        np.linalg.norm(R_up, 'fro'), 1e-10
    )
    print(f"  ||R_up - R_down||/||R_up||: {diff:.3e}")
    assert diff < 1e-3, f"R_up != R_down: rel diff {diff:.3e} >= 1e-3"


def test_14e():
    """s_up validation: beam source via loose vs tight tolerance."""
    print("\n--- Test 14e: s_up convergence ---")
    tau_sub = 5.0
    mu0 = 0.5
    I0_div_4pi = 1.0 / (4 * pi)

    # Build beam-source q functions using the JAX helper
    leg_data_beam = _precompute_legendre(0, NLeg, mu_arr_pos, mu0)
    q_up_func, q_down_func = _make_q_funcs_jax(
        omega_func, Leg_coeffs_func, 0, leg_data_beam, mu_arr_pos, M_inv, mu0,
        I0_div_4pi, True, N,  # m_equals_0=True for m=0
    )

    alpha_func_beam, beta_func_beam = _make_alpha_beta_funcs_jax(
        omega_func, Leg_coeffs_func, 0, leg_data_beam, mu_arr_pos, W, M_inv, N,
    )

    # Tight-tolerance reference
    R_up_ref, T_up_ref, s_up_ref, _ = _riccati_forward_jax(
        alpha_func_beam, beta_func_beam, tau_sub, N, tol=1e-10,
        q_up_func=q_up_func, q_down_func=q_down_func,
    )

    # Loose tolerance
    R_up, T_up, s_up, grid = _riccati_forward_jax(
        alpha_func_beam, beta_func_beam, tau_sub, N, tol=1e-4,
        q_up_func=q_up_func, q_down_func=q_down_func,
    )

    n_steps = len(grid) - 1
    print(f"  Kvaerno5 steps: {n_steps}")

    s_up_ref, s_up = np.asarray(s_up_ref), np.asarray(s_up)
    R_up_ref, R_up = np.asarray(R_up_ref), np.asarray(R_up)

    # Compare s_up
    rel_err_s = np.linalg.norm(s_up - s_up_ref) / max(
        np.linalg.norm(s_up_ref), 1e-10
    )
    print(f"  s_up rel_err: {rel_err_s:.3e}")
    assert rel_err_s < 5e-3, f"s_up rel_err {rel_err_s:.3e} >= 5e-3"

    # Also check R_up consistency
    rel_err_R = np.linalg.norm(R_up - R_up_ref, 'fro') / max(
        np.linalg.norm(R_up_ref, 'fro'), 1e-10
    )
    print(f"  R_up rel_err: {rel_err_R:.3e}")
    assert rel_err_R < 1e-3, f"R_up rel_err {rel_err_R:.3e} >= 1e-3"


# ---------------------------------------------------------------------------
# Tau-varying setup for test_14f
# ---------------------------------------------------------------------------
_tau_14f = 1.0
_NQuad_14f = 8
_NLeg_14f = _NQuad_14f
_N_14f = _NQuad_14f // 2

_omega_func_14f = lambda tau: 0.90 - 0.30 * tau / _tau_14f
_g_func_14f = lambda tau: 0.70 - 0.40 * tau / _tau_14f
_Leg_coeffs_func_14f = lambda tau: _g_func_14f(tau) ** np.arange(_NLeg_14f)
_b_pos_14f = np.random.RandomState(44).rand(_N_14f)


def test_14f():
    """T_up with tau-varying omega/g: I0=0, b_pos-only propagation vs multilayer pydisort.

    With I0=0, b_neg=0, no BDRF, the ToA upwelling reduces to I+(0) = T_up @ b_pos.
    Validates T_up against an independent reference (not self-convergence).
    Would have caught the dT/dσ = T(α+βR) vs (α+Rβ)T ordering bug.
    """
    print("\n--- Test 14f: T_up tau-varying vs multilayer pydisort ---")
    mu0, phi0 = 0.5, 0.0  # dummy (no beam)

    # Riccati solver
    _, _, _, u_ToA_func, _ = pydisort_riccati_jax(
        _tau_14f, _omega_func_14f, _Leg_coeffs_func_14f, _NQuad_14f,
        mu0, 0.0, phi0,      # I0 = 0
        b_pos=_b_pos_14f,
        tol=1e-8,
    )

    # Independent reference: 500-layer multilayer pydisort
    _, _, uf_ref = multilayer_pydisort_toa_full_phi(
        _tau_14f, _omega_func_14f, _Leg_coeffs_func_14f, 500,
        _NQuad_14f, _NLeg_14f, mu0, 0.0, phi0,
        b_pos=_b_pos_14f,
    )

    # Compare at phi=0 (no azimuthal dependence when I0=0)
    u_ric = np.asarray(u_ToA_func(0.0))[:_N_14f]
    u_ref = np.asarray(uf_ref(0, 0.0))[:_N_14f]
    scale = max(float(np.max(np.abs(u_ref))), 1e-8)
    rel_err = float(np.max(np.abs(u_ric - u_ref))) / scale

    print(f"  T_up tau-varying rel_err: {rel_err:.3e}")
    assert rel_err < 5e-3, (
        f"T_up tau-varying: rel_err={rel_err:.3e} >= 5e-3"
    )
