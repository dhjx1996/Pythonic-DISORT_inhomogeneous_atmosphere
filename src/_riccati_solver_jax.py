"""
Riccati solver for full-domain radiative transfer — JAX + diffrax port.

Port of _riccati_solver.py (scipy Radau IIA) to JAX + diffrax (Kvaerno5).
Integrates the invariant-imbedding Riccati ODE using diffrax's Kvaerno5
(L-stable ESDIRK, order 5, adaptive) with PIDController step-size control.

    dR/dσ = α·R + R·α + R·β·R + β           [N×N, nonlinear Riccati]
    dT/dσ = T·(α + β·R)                      [N×N, linear in T]
    ds/dσ = (α + R·β)·s + R·q₁ + q₂         [N, linear in s]

State is a PyTree {'R': (N,N), 'T': (N,N), 's': (N,)} — no flattening needed.
Legendre polynomial products are pre-computed at setup time using scipy.special,
making the ODE vector field fully JAX-traceable (no scipy calls during integration).

All terms have positive sign — no growing exponentials (satisfies the
no-positive-exponents invariant).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import diffrax


# ---------------------------------------------------------------------------
# Pre-computation (scipy, at setup time — NOT inside JAX trace)
# ---------------------------------------------------------------------------

def _precompute_legendre(m, NLeg, mu_arr_pos, mu0):
    """Pre-compute Legendre polynomial products at quadrature points.

    Uses scipy.special at setup time (not JAX-traceable). Returns JAX arrays
    for use in the ODE vector field via einsum contractions.

    Parameters
    ----------
    m : int
        Azimuthal Fourier mode index.
    NLeg : int
        Number of Legendre terms.
    mu_arr_pos : (N,) array
        Positive quadrature cosines.
    mu0 : float
        Cosine of the beam zenith angle.

    Returns
    -------
    dict with JAX arrays:
        base_coeffs : (n_ells,) — (2l+1) * poch(l+m+1, -2m)
        P_pos       : (n_ells, N) — P_l^m(+mu_i)
        P_neg       : (n_ells, N) — P_l^m(-mu_i)
        P_mu0       : (n_ells,) — P_l^m(-mu0)
    """
    mu_np = np.asarray(mu_arr_pos, dtype=np.float64)
    N = len(mu_np)
    ells = np.arange(m, NLeg)
    n_ells = len(ells)

    if n_ells == 0:
        return {
            'base_coeffs': jnp.zeros(0),
            'P_pos': jnp.zeros((0, N)),
            'P_neg': jnp.zeros((0, N)),
            'P_mu0': jnp.zeros(0),
        }

    base_coeffs = (2 * ells + 1) * sp.poch(ells + m + 1, -2.0 * m)

    P_pos = np.zeros((n_ells, N))
    P_neg = np.zeros((n_ells, N))
    for idx, l_val in enumerate(ells):
        P_pos[idx] = sp.lpmv(m, l_val, mu_np)
        P_neg[idx] = sp.lpmv(m, l_val, -mu_np)

    P_mu0 = np.array([sp.lpmv(m, int(l_val), -float(mu0)) for l_val in ells])

    return {
        'base_coeffs': jnp.array(base_coeffs),
        'P_pos': jnp.array(P_pos),
        'P_neg': jnp.array(P_neg),
        'P_mu0': jnp.array(P_mu0),
    }


# ---------------------------------------------------------------------------
# JAX-traceable coefficient functions
# ---------------------------------------------------------------------------

def _make_alpha_beta_funcs_jax(omega_func, g_l_func, m, leg_data,
                                mu_arr_pos, W, M_inv, N):
    """Build JAX-traceable alpha(tau) and beta(tau) for the Riccati ODE.

    Parameters
    ----------
    omega_func : tau -> scalar
    g_l_func   : tau -> (NLeg,) array
    m          : int, Fourier mode index
    leg_data   : dict from _precompute_legendre
    mu_arr_pos : (N,) JAX array
    W          : (N,) JAX array, quadrature weights
    M_inv      : (N,) JAX array, 1/mu_arr_pos
    N          : int, half-stream count

    Returns
    -------
    (alpha_func, beta_func) : each  tau -> (N, N) JAX array
    """
    base_coeffs = leg_data['base_coeffs']
    P_pos = leg_data['P_pos']
    P_neg = leg_data['P_neg']
    I_N = jnp.eye(N)

    def alpha_func(tau):
        omega = omega_func(tau)
        g_l = g_l_func(tau)
        wt = base_coeffs * g_l[m:]
        D_pos = 0.5 * jnp.einsum('l,li,lj->ij', wt, P_pos, P_pos)
        return M_inv[:, None] * (omega * D_pos * W[None, :] - I_N)

    def beta_func(tau):
        omega = omega_func(tau)
        g_l = g_l_func(tau)
        wt = base_coeffs * g_l[m:]
        D_neg = 0.5 * jnp.einsum('l,li,lj->ij', wt, P_pos, P_neg)
        return M_inv[:, None] * (omega * D_neg * W[None, :])

    return alpha_func, beta_func


def _make_q_funcs_jax(omega_func, g_l_func, m, leg_data,
                       mu_arr_pos, M_inv, mu0, fac_const, N):
    """Build JAX-traceable beam-source q functions.

    Parameters
    ----------
    omega_func : tau -> scalar
    g_l_func   : tau -> (NLeg,) array
    m          : int, Fourier mode index
    leg_data   : dict from _precompute_legendre
    mu_arr_pos : (N,) JAX array
    M_inv      : (N,) JAX array
    mu0        : float, beam cosine
    fac_const  : float, (I0/(4pi)) * (2 - delta_m0) * 2
    N          : int

    Returns
    -------
    (q_up_func, q_down_func) : each  tau -> (N,) JAX array
    """
    base_coeffs = leg_data['base_coeffs']
    P_pos = leg_data['P_pos']
    P_neg = leg_data['P_neg']
    P_mu0 = leg_data['P_mu0']

    def q_up_func(tau):
        omega = omega_func(tau)
        g_l = g_l_func(tau)
        wt = base_coeffs * g_l[m:]
        D_beam = 0.5 * jnp.einsum('l,li,l->i', wt, P_pos, P_mu0)
        return M_inv * fac_const * omega * jnp.exp(-tau / mu0) * D_beam

    def q_down_func(tau):
        omega = omega_func(tau)
        g_l = g_l_func(tau)
        wt = base_coeffs * g_l[m:]
        D_beam = 0.5 * jnp.einsum('l,li,l->i', wt, P_neg, P_mu0)
        return M_inv * fac_const * omega * jnp.exp(-tau / mu0) * D_beam

    return q_up_func, q_down_func


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _riccati_rhs_jax(R, alpha, beta):
    """F(R) = alpha R + R alpha + R beta R + beta  (Riccati RHS)."""
    return alpha @ R + R @ alpha + R @ beta @ R + beta


# ---------------------------------------------------------------------------
# Kvaerno5 integration (core)
# ---------------------------------------------------------------------------

def _kvaerno5_integrate(alpha_func, beta_func, sigma_end, N, tol,
                        q1_func=None, q2_func=None, max_steps=4096):
    """Adaptive Kvaerno5 (L-stable ESDIRK, order 5) integration.

    Integrates from sigma=0 to sigma=sigma_end.
    State: PyTree {'R': (N,N), 'T': (N,N), 's': (N,)}.
    IC: R(0)=0, T(0)=I, s(0)=0.

    Parameters
    ----------
    alpha_func, beta_func : sigma -> (N, N) JAX arrays
    sigma_end : float > 0
    N         : int
    tol       : float > 0
    q1_func, q2_func : sigma -> (N,) JAX arrays, or None
    max_steps : int

    Returns
    -------
    R, T : (N, N) JAX arrays at sigma_end
    s    : (N,) JAX array (zeros if no beam source)
    sigma_grid : 1-D numpy array [0, sigma_1, ..., sigma_end]
    """
    has_source = q1_func is not None

    def vector_field(sigma, state, args):
        R = state['R']
        T = state['T']
        s = state['s']

        alpha = alpha_func(sigma)
        beta = beta_func(sigma)
        beta_R = beta @ R

        # Riccati: dR/dsigma = alpha R + R alpha + R beta R + beta
        dR = alpha @ R + R @ alpha + R @ beta_R + beta

        # Transmission: dT/dsigma = T (alpha + beta R)
        dT = T @ (alpha + beta_R)

        # Source: ds/dsigma = (alpha + R beta) s + R q1 + q2
        if has_source:
            ds = (alpha + R @ beta) @ s + R @ q1_func(sigma) + q2_func(sigma)
        else:
            ds = jnp.zeros(N)

        return {'R': dR, 'T': dT, 's': ds}

    y0 = {
        'R': jnp.zeros((N, N)),
        'T': jnp.eye(N),
        's': jnp.zeros(N),
    }

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=tol, atol=tol * 1e-3)
    saveat = diffrax.SaveAt(steps=True)

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=float(sigma_end),
        dt0=None,
        y0=y0,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
    )

    # Extract valid time points (unused slots are padded with inf)
    ts_np = np.asarray(sol.ts)
    valid_mask = np.isfinite(ts_np)
    n_valid = int(np.sum(valid_mask))
    sigma_steps = ts_np[:n_valid]

    # Build grid including sigma=0
    if n_valid == 0 or sigma_steps[0] > 1e-14:
        sigma_grid = np.concatenate([[0.0], sigma_steps])
    else:
        sigma_grid = sigma_steps

    # Final state at last valid step
    R = sol.ys['R'][n_valid - 1]
    T = sol.ys['T'][n_valid - 1]
    s = sol.ys['s'][n_valid - 1]

    return R, T, s, sigma_grid


# ---------------------------------------------------------------------------
# Forward / backward wrappers
# ---------------------------------------------------------------------------

def _riccati_forward_jax(alpha_func, beta_func, tau_bot, N, tol,
                          q_up_func=None, q_down_func=None, max_steps=4096):
    """Forward Riccati: build slab from bottom (tau_bot) upward to top (0).

    Integration variable sigma in [0, tau_bot].
    Coefficient evaluation: alpha(tau_bot - sigma), beta(tau_bot - sigma).
    Source mapping: q1(sigma) = q_down(tau_bot - sigma),
                    q2(sigma) = q_up(tau_bot - sigma).

    Returns (R_up, T_up, s_up, tau_grid).
    """
    tb = float(tau_bot)
    has_source = q_up_func is not None

    if has_source:
        q1 = lambda sigma: q_down_func(tb - sigma)
        q2 = lambda sigma: q_up_func(tb - sigma)
    else:
        q1 = q2 = None

    R, T, s, sigma_grid = _kvaerno5_integrate(
        lambda sigma: alpha_func(tb - sigma),
        lambda sigma: beta_func(tb - sigma),
        tb, N, tol,
        q1_func=q1, q2_func=q2,
        max_steps=max_steps,
    )
    tau_grid = tb - sigma_grid[::-1]
    return R, T, s, tau_grid


def _riccati_backward_jax(alpha_func, beta_func, tau_bot, N, tol,
                           q_up_func=None, q_down_func=None, max_steps=4096):
    """Backward Riccati: build slab from top (0) downward to bottom (tau_bot).

    Integration variable sigma in [0, tau_bot].
    Coefficient evaluation: alpha(sigma), beta(sigma).
    Source mapping: q1(sigma) = q_up(sigma), q2(sigma) = q_down(sigma).

    Returns (R_down, T_down, s_down, tau_grid).
    """
    tb = float(tau_bot)
    has_source = q_up_func is not None

    if has_source:
        q1 = lambda sigma: q_up_func(sigma)
        q2 = lambda sigma: q_down_func(sigma)
    else:
        q1 = q2 = None

    R, T, s, sigma_grid = _kvaerno5_integrate(
        alpha_func, beta_func,
        tb, N, tol,
        q1_func=q1, q2_func=q2,
        max_steps=max_steps,
    )
    return R, T, s, sigma_grid
