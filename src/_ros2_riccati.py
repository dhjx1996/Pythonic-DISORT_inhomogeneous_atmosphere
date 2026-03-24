"""
ROS2 Riccati solver for the diffusion domain.

Integrates the invariant-imbedding Riccati ODE for reflection R and
the companion linear ODE for transmission T:

    dR/dσ = α·R + R·α + R·β·R + β,   R(0) = 0
    dT/dσ = T·(α + β·R),              T(0) = I

All terms have positive sign — no growing exponentials.

Uses a 2-stage L-stable Rosenbrock method (ROS2) with Sylvester solves
for the Riccati stage equations and Crank-Nicolson for the companion T.
Step size is controlled by an embedded ROS2-1 error pair.
"""
import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# ROS2 tableau  (L-stable, Verwer convention)
#
# Stage eqs:  (I - hγJ) k_i  =  h f(...)  +  c₂₁ k₁  (stage 2 only)
# Update:     y₁ = y₀ + b₁ k₁ + b₂ k₂
#
# γ = 1 + 1/√2 gives L-stability (automatically, for any order-2 ROS2).
# c₂₁ is a direct correction (NOT Jacobian-weighted like Hairer-Wanner γ₂₁).
# Order-2 conditions:  b₁+b₂(1+c₂₁)=1,  b₁γ+b₂[(1+2c₂₁)γ+a₂₁]=½.
#
# Ref: Verwer, Spee, Blom, Hundsdorfer (1999).
# ---------------------------------------------------------------------------
_GAMMA = 1.0 + 1.0 / np.sqrt(2.0)             # ≈ 1.7071
_A21   = 1.0                                    # stage 2 evaluates F at R_n + K₁
_C21   = -2.0                                   # direct correction on K₁
_B1    = 1.5                                     # 3/2
_B2    = 0.5                                     # 1/2


def _riccati_rhs(R, alpha, beta):
    """F(R) = α·R + R·α + R·β·R + β."""
    return alpha @ R + R @ alpha + R @ beta @ R + beta


def _ros2_step(R_n, T_n, alpha, beta, h, N):
    """
    One ROS2 step for R, Crank-Nicolson for T.

    Returns (R_new, T_new, err_rel).
    err_rel: relative error estimate from the embedded ROS2-1 pair.
    """
    # Linearisation matrices
    A_n = alpha + R_n @ beta          # N×N
    B_n = alpha + beta @ R_n          # N×N  (also C_n for T equation)

    F_n = _riccati_rhs(R_n, alpha, beta)

    # Sylvester LHS:  P K + K Q = RHS
    P = np.eye(N) - _GAMMA * h * A_n
    Q = -_GAMMA * h * B_n

    # Stage 1:  P K₁ + K₁ Q = h F_n
    K1 = scipy.linalg.solve_sylvester(P, Q, h * F_n)

    # Stage 2:  P K₂ + K₂ Q = h F(R_n + K₁) + c₂₁ K₁
    R_mid = R_n + _A21 * K1
    F_mid = _riccati_rhs(R_mid, alpha, beta)
    K2 = scipy.linalg.solve_sylvester(P, Q, h * F_mid + _C21 * K1)

    # R update:  R_{n+1} = R_n + b₁ K₁ + b₂ K₂
    R_new = R_n + _B1 * K1 + _B2 * K2

    # Embedded error (order-2 vs order-1):  ‖(b₁-1)K₁ + b₂K₂‖
    err_norm = np.linalg.norm((_B1 - 1.0) * K1 + _B2 * K2, 'fro')
    err_rel = err_norm / max(1.0, np.linalg.norm(R_n, 'fro'))

    # T update: Crank-Nicolson  T_{n+1}(I - (h/2)C_{n+1}) = T_n(I + (h/2)C_n)
    C_n = B_n
    C_new = alpha + beta @ R_new
    T_half = T_n @ (np.eye(N) + 0.5 * h * C_n)
    # Right-multiply solve:  X M = B  ⟺  M^T X^T = B^T
    T_new = scipy.linalg.solve(
        (np.eye(N) - 0.5 * h * C_new).T, T_half.T
    ).T

    return R_new, T_new, err_rel


def _ros2_integrate(alpha_func, beta_func, sigma_end, N, tol):
    """
    Adaptive ROS2 integration from σ = 0 to σ = sigma_end.

    R(0) = 0, T(0) = I.

    Parameters
    ----------
    alpha_func, beta_func : callables  σ (float) → (N, N) ndarray
    sigma_end : float > 0
    N : int, matrix dimension
    tol : float > 0

    Returns
    -------
    R, T : (N, N) ndarrays
    sigma_grid : 1-D ndarray of step boundary points [0, σ₁, …, sigma_end]
    """
    alpha0 = alpha_func(0.0)
    beta0 = beta_func(0.0)
    spec = float(np.max(np.abs(np.linalg.eigvals(alpha0 + beta0).real)))
    h = min(sigma_end, 0.5 / max(spec, 1e-10))

    R = np.zeros((N, N))
    T = np.eye(N)
    sigma = 0.0
    sigma_grid = [0.0]

    while sigma < sigma_end - 1e-14:
        final_step = (sigma + h >= sigma_end - 1e-14)
        if final_step:
            h = sigma_end - sigma

        alpha = alpha_func(sigma)
        beta = beta_func(sigma)

        R_new, T_new, err_rel = _ros2_step(R, T, alpha, beta, h, N)

        if err_rel > tol and not final_step:
            factor = max(0.2, 0.9 * (tol / max(err_rel, 1e-15)) ** 0.5)
            h *= factor
            continue

        R, T = R_new, T_new
        sigma += h
        sigma_grid.append(sigma)

        if err_rel > 0:
            factor = min(2.0, max(0.2, 0.9 * (tol / err_rel) ** 0.5))
        else:
            factor = 2.0
        h *= factor

    return R, T, np.array(sigma_grid)


def _ros2_forward(alpha_func, beta_func, tau1, tau2, N, tol):
    """
    Forward Riccati: build slab from bottom (τ₂) upward.

    Integration variable σ ∈ [0, τ₂−τ₁] with α(τ₂−σ), β(τ₂−σ).

    Returns (R_up, T_up, tau_grid).
    """
    sigma_end = tau2 - tau1
    R, T, grid = _ros2_integrate(
        lambda sigma: alpha_func(tau2 - sigma),
        lambda sigma: beta_func(tau2 - sigma),
        sigma_end, N, tol,
    )
    tau_grid = tau2 - grid[::-1]
    return R, T, tau_grid


def _ros2_backward(alpha_func, beta_func, tau1, tau2, N, tol):
    """
    Backward Riccati: build slab from top (τ₁) downward.

    Integration variable σ ∈ [0, τ₂−τ₁] with α(τ₁+σ), β(τ₁+σ).

    Returns (R_down, T_down, tau_grid).
    """
    sigma_end = tau2 - tau1
    R, T, grid = _ros2_integrate(
        lambda sigma: alpha_func(tau1 + sigma),
        lambda sigma: beta_func(tau1 + sigma),
        sigma_end, N, tol,
    )
    tau_grid = tau1 + grid
    return R, T, tau_grid
