"""
Test suite 14: ROS2 Riccati solver (standalone).

Validates the ROS2 solver for the Riccati + companion T equations in isolation
(before integration into the hybrid domain decomposition).

  14a: Homogeneous atmosphere — R_up matches Magnus star-product at high K.
  14b: K-sweep — O(h²) convergence for fixed-step ROS2.
  14c: T validation — T_up propagates boundary correctly.
  14d: R_up = R_down for homogeneous slab.
"""
import numpy as np
from math import pi
from _helpers import make_D_m_funcs
from PythonicDISORT import subroutines
from _ros2_riccati import _ros2_integrate, _ros2_forward, _ros2_backward
from _magnus_propagator import _compute_magnus_propagator

NQuad = 16
NLeg = NQuad
N = NQuad // 2


def _build_alpha_beta(omega, g, mu_arr_pos, W, N, D_m):
    """Build constant α, β for a homogeneous atmosphere (m=0 mode)."""
    M_inv = 1.0 / mu_arr_pos
    D_pos = omega * D_m(0.0, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m(0.0, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta = M_inv[:, None] * DW_neg
    return alpha, beta


def _build_A_func(alpha, beta):
    """Build the 2N×2N A matrix from α, β (for Magnus comparison)."""
    NQ = 2 * alpha.shape[0]
    N = alpha.shape[0]
    def A_func(tau):
        A = np.empty((NQ, NQ))
        A[:N, :N] = -alpha
        A[:N, N:] = -beta
        A[N:, :N] = beta
        A[N:, N:] = alpha
        return A
    return A_func


# Setup: homogeneous atmosphere (ω=0.99, g=0.85)
omega, g = 0.99, 0.85
g_l = g ** np.arange(NLeg)
D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
D_m = D_m_funcs[0]  # m=0 mode

mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
alpha, beta = _build_alpha_beta(omega, g, mu_arr_pos, W, N, D_m)


def test_14a():
    """R_up from ROS2 matches Magnus star-product at K=2000."""
    print("\n--- Test 14a: ROS2 R_up vs Magnus (homogeneous) ---")
    tau_sub = 20.0

    # Magnus reference (high K)
    A_func = _build_A_func(alpha, beta)
    S_func = lambda tau: np.zeros(NQuad)
    R_up_mag, T_up_mag, _, _, _, _ = _compute_magnus_propagator(
        A_func, S_func, tau_sub, 2000, NQuad,
    )

    # ROS2 (adaptive)
    R_up_ros, T_up_ros, grid = _ros2_forward(
        lambda tau: alpha, lambda tau: beta,
        0.0, tau_sub, N, tol=1e-4,
    )

    rel_err_R = np.linalg.norm(R_up_ros - R_up_mag, 'fro') / max(
        np.linalg.norm(R_up_mag, 'fro'), 1e-10
    )
    n_steps = len(grid) - 1
    print(f"  ROS2 steps: {n_steps}")
    print(f"  R_up rel_err: {rel_err_R:.3e}")
    assert rel_err_R < 1e-3, f"R_up rel_err {rel_err_R:.3e} >= 1e-3"


def test_14b():
    """Fixed-step ROS2 K-sweep: verify O(h²) convergence."""
    print("\n--- Test 14b: ROS2 K-sweep (O(h²) check) ---")
    tau_sub = 5.0

    # Reference at very high K
    A_func = _build_A_func(alpha, beta)
    S_func = lambda tau: np.zeros(NQuad)
    R_ref, _, _, _, _, _ = _compute_magnus_propagator(
        A_func, S_func, tau_sub, 2000, NQuad,
    )

    K_values = [20, 40, 80, 160]
    errs = []
    for K in K_values:
        h = tau_sub / K
        # Manual fixed-step integration
        from _ros2_riccati import _ros2_step
        R = np.zeros((N, N))
        T = np.eye(N)
        for _ in range(K):
            R, T, _ = _ros2_step(R, T, alpha, beta, h, N)
        err = np.linalg.norm(R - R_ref, 'fro') / max(
            np.linalg.norm(R_ref, 'fro'), 1e-10
        )
        errs.append(err)

    print(f"  {'K':>6s}  {'err':>10s}  {'ratio':>8s}")
    for i, K in enumerate(K_values):
        ratio_str = f"{errs[i-1]/max(errs[i],1e-15):.1f}" if i > 0 else ""
        print(f"  {K:6d}  {errs[i]:10.3e}  {ratio_str:>8s}")

    # Check O(h²): ratios should be ~4 for doubling K
    for i in range(1, len(K_values)):
        if errs[i - 1] < 1e-8:
            continue
        ratio = errs[i - 1] / max(errs[i], 1e-15)
        assert ratio > 2.0, (
            f"K={K_values[i-1]}->{K_values[i]}: ratio {ratio:.1f} < 2 "
            f"(expected ~4 for O(h²))"
        )


def test_14c():
    """T validation: R_up·b_neg + T_up·b_pos matches Magnus result."""
    print("\n--- Test 14c: T validation via boundary propagation ---")
    tau_sub = 10.0

    # Construct b_pos and b_neg test vectors
    b_neg = np.random.RandomState(42).rand(N)
    b_pos = np.random.RandomState(43).rand(N)

    # Magnus reference
    A_func = _build_A_func(alpha, beta)
    S_func = lambda tau: np.zeros(NQuad)
    R_up_mag, T_up_mag, T_down_mag, R_down_mag, _, _ = _compute_magnus_propagator(
        A_func, S_func, tau_sub, 2000, NQuad,
    )

    # Compute I^+(0) from Magnus: I^+(0) = R_up·b_neg + T_up·I^+(bot)
    # where I^+(bot) = (I - R_surf·R_down)^{-1} · R_surf·(T_down·b_neg + s_down) + b_pos_eff
    # For no BDRF: I^+(bot) = b_pos, so I^+(0) = R_up·b_neg + T_up·b_pos
    I_up_mag = R_up_mag @ b_neg + T_up_mag @ b_pos

    # ROS2
    R_up_ros, T_up_ros, _ = _ros2_forward(
        lambda tau: alpha, lambda tau: beta,
        0.0, tau_sub, N, tol=1e-4,
    )
    I_up_ros = R_up_ros @ b_neg + T_up_ros @ b_pos

    rel_err = np.linalg.norm(I_up_ros - I_up_mag) / max(
        np.linalg.norm(I_up_mag), 1e-10
    )
    print(f"  I^+(0) rel_err: {rel_err:.3e}")
    assert rel_err < 1e-3, f"I^+(0) rel_err {rel_err:.3e} >= 1e-3"


def test_14d():
    """R_up = R_down for homogeneous slab."""
    print("\n--- Test 14d: R_up = R_down (homogeneous symmetry) ---")
    tau_sub = 10.0

    R_up, _, _ = _ros2_forward(
        lambda tau: alpha, lambda tau: beta,
        0.0, tau_sub, N, tol=1e-4,
    )
    R_down, _, _ = _ros2_backward(
        lambda tau: alpha, lambda tau: beta,
        0.0, tau_sub, N, tol=1e-4,
    )

    diff = np.linalg.norm(R_up - R_down, 'fro') / max(
        np.linalg.norm(R_up, 'fro'), 1e-10
    )
    print(f"  ||R_up - R_down||/||R_up||: {diff:.3e}")
    assert diff < 1e-3, f"R_up ≠ R_down: rel diff {diff:.3e} >= 1e-3"
