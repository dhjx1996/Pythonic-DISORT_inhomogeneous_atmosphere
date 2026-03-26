"""
Riccati solver for full-domain radiative transfer.

Integrates the invariant-imbedding Riccati ODE for reflection R,
the companion linear ODE for transmission T, and the beam-source
companion ODE for source vector s using scipy's Radau IIA solver
(3-stage implicit Runge-Kutta, L-stable, order 5, adaptive).

    dR/dσ = α·R + R·α + R·β·R + β           [N×N, nonlinear Riccati]
    dT/dσ = T·(α + β·R)                      [N×N, linear in T]
    ds/dσ = (α + R·β)·s + R·q₁ + q₂         [N, linear in s]

The state is vectorized as y = [vec(R), vec(T), s] of size 2N² + N.
Radau handles the full coupled system with no order reduction for s
(unlike Rosenbrock methods with frozen/block-diagonal Jacobian).

All terms have positive sign — no growing exponentials (satisfies the
no-positive-exponents invariant).

NQuad ≥ 6 required: The Riccati ARE solution is ill-conditioned for
NQuad=4 (two-stream), with ‖R_stab‖ ≈ 10 causing solver divergence.
NQuad ≥ 6 gives ‖R_stab‖ ≈ 0.35.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _riccati_rhs(R, alpha, beta):
    """F(R) = α·R + R·α + R·β·R + β."""
    return alpha @ R + R @ alpha + R @ beta @ R + beta


def _make_alpha_beta_funcs(omega_func, D_m, mu_arr_pos, W, M_inv, N):
    """
    Build alpha(tau) and beta(tau) callables for the Riccati solver.

    These are the N×N blocks of the 2N×2N coefficient matrix A:
        A = [[-alpha, -beta], [beta, alpha]]
    """
    I_N = np.eye(N)

    def alpha_func(tau):
        omega = omega_func(tau)
        D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        return M_inv[:, None] * (DW_pos - I_N)

    def beta_func(tau):
        omega = omega_func(tau)
        D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_neg = D_neg * W[None, :]
        return M_inv[:, None] * DW_neg

    return alpha_func, beta_func


# ---------------------------------------------------------------------------
# Radau integration
# ---------------------------------------------------------------------------

def _radau_integrate(alpha_func, beta_func, sigma_end, N, tol,
                     q1_func=None, q2_func=None):
    """
    Adaptive Radau IIA (order 5) integration from σ = 0 to σ = sigma_end.

    State vector: y = [vec(R), vec(T), s], size 2N² + N.
    IC: R(0) = 0, T(0) = I, s(0) = 0.

    Parameters
    ----------
    alpha_func, beta_func : callables  σ → (N, N) ndarray
    sigma_end : float > 0
    N : int, half-stream count (matrix dimension)
    tol : float > 0, relative tolerance for Radau
    q1_func, q2_func : callables σ → (N,) ndarray, or None (no beam source)

    Returns
    -------
    R, T : (N, N) ndarrays
    s : (N,) ndarray  (zeros if no beam source)
    sigma_grid : 1-D ndarray of step boundary points [0, σ₁, …, sigma_end]
    """
    has_source = q1_func is not None
    NN = N * N

    def rhs(sigma, y):
        R = y[:NN].reshape(N, N)
        T = y[NN:2 * NN].reshape(N, N)
        s = y[2 * NN:]

        alpha = alpha_func(sigma)
        beta = beta_func(sigma)
        beta_R = beta @ R

        # Riccati: dR/dσ = αR + Rα + RβR + β
        dR = alpha @ R + R @ alpha + R @ beta_R + beta

        # Transmission: dT/dσ = T·(α + βR)
        dT = T @ (alpha + beta_R)

        # Source: ds/dσ = (α + Rβ)s + Rq₁ + q₂
        if has_source:
            ds = (alpha + R @ beta) @ s + R @ q1_func(sigma) + q2_func(sigma)
        else:
            ds = np.zeros(N)

        return np.concatenate([dR.ravel(), dT.ravel(), ds])

    # Initial condition: R=0, T=I, s=0
    y0 = np.zeros(2 * NN + N)
    y0[NN:2 * NN] = np.eye(N).ravel()

    sol = solve_ivp(rhs, [0.0, sigma_end], y0,
                    method='Radau', rtol=tol, atol=tol * 1e-3)

    if not sol.success:
        raise RuntimeError(f"Radau integration failed: {sol.message}")

    # Extract final state
    y_end = sol.y[:, -1]
    R = y_end[:NN].reshape(N, N)
    T = y_end[NN:2 * NN].reshape(N, N)
    s = y_end[2 * NN:]

    return R, T, s, sol.t


# ---------------------------------------------------------------------------
# Forward / backward wrappers
# ---------------------------------------------------------------------------

def _riccati_forward(alpha_func, beta_func, tau_bot, N, tol,
                     q_up_func=None, q_down_func=None):
    """
    Forward Riccati: build slab from bottom (τ_bot) upward to top (0).

    Integration variable σ ∈ [0, τ_bot].
    Coefficient evaluation: α(τ_bot − σ), β(τ_bot − σ).
    Source mapping:
        q1(σ) = q_down(τ_bot − σ)  (downward source reflects off slab → upward)
        q2(σ) = q_up(τ_bot − σ)    (upward source directly → upward)

    Returns (R_up, T_up, s_up, tau_grid).
    """
    tb = float(tau_bot)
    has_source = q_up_func is not None

    if has_source:
        q1 = lambda sigma: q_down_func(tb - sigma)
        q2 = lambda sigma: q_up_func(tb - sigma)
    else:
        q1 = q2 = None

    R, T, s, grid = _radau_integrate(
        lambda sigma: alpha_func(tb - sigma),
        lambda sigma: beta_func(tb - sigma),
        tb, N, tol,
        q1_func=q1, q2_func=q2,
    )
    tau_grid = tb - grid[::-1]    # [0, ..., tau_bot]
    return R, T, s, tau_grid


def _riccati_backward(alpha_func, beta_func, tau_bot, N, tol,
                      q_up_func=None, q_down_func=None):
    """
    Backward Riccati: build slab from top (0) downward to bottom (τ_bot).

    Integration variable σ ∈ [0, τ_bot].
    Coefficient evaluation: α(σ), β(σ).
    Source mapping:
        q1(σ) = q_up(σ)    (upward source reflects off slab → downward)
        q2(σ) = q_down(σ)  (downward source directly → downward)

    Returns (R_down, T_down, s_down, tau_grid).
    """
    tb = float(tau_bot)
    has_source = q_up_func is not None

    if has_source:
        q1 = lambda sigma: q_up_func(sigma)
        q2 = lambda sigma: q_down_func(sigma)
    else:
        q1 = q2 = None

    R, T, s, grid = _radau_integrate(
        lambda sigma: alpha_func(sigma),
        lambda sigma: beta_func(sigma),
        tb, N, tol,
        q1_func=q1, q2_func=q2,
    )
    tau_grid = grid              # [0, ..., tau_bot]
    return R, T, s, tau_grid
