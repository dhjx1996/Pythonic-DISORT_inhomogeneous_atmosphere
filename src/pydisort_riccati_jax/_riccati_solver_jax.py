"""
Riccati solver for full-domain radiative transfer — JAX + diffrax.

Integrates the invariant-imbedding Riccati ODE using diffrax's Kvaerno5
(L-stable ESDIRK, order 5, adaptive) with PIDController step-size control.

    dR/dσ = α·R + R·α + R·β·R + β           [N×N, nonlinear Riccati]
    dT/dσ = (α + R·β)·T                      [N×N, linear in T]
    ds/dσ = (α + R·β)·s + R·q₁ + q₂         [N, linear in s]

State is a PyTree {'R': (N,N), 'T': (N,N), 's': (N,)} — no flattening needed.
Legendre polynomial products are pre-computed at setup time using scipy.special,
making the ODE vector field fully JAX-traceable (no scipy calls during integration).

All terms have positive sign — no growing exponentials (satisfies the
no-positive-exponents invariant).
"""

import os
import warnings
from typing import NamedTuple
import jax
# Float32 by default; opt into float64 via PYDISORT_RICCATI_JAX_X64=1. This MUST
# match pydisort_riccati_jax.py so the dtype is import-order-independent (this
# module is imported both directly by tests and indirectly via the public API).
# See the float32 rationale in pydisort_riccati_jax.py.
jax.config.update(
    "jax_enable_x64",
    os.environ.get("PYDISORT_RICCATI_JAX_X64", "0") == "1",
)

import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import diffrax


# ---------------------------------------------------------------------------
# Pre-computation (scipy, at setup time — NOT inside JAX trace)
# (cf. _solve_for_gen_and_part_sols: asso_leg_term_pos, asso_leg_term_neg,
#  fac_asso_leg_term_posT, scalar_fac_asso_leg_term_mu0)
# ---------------------------------------------------------------------------

def _precompute_legendre(m, NLeg, mu_arr_pos):
    """Pre-compute Legendre polynomial products at the quadrature points.

    Uses scipy.special at *setup* time (not JAX-traceable). Everything here is a
    function of (m, NLeg, mu_arr_pos) only — all **mu0-independent** — so it is
    built once per Fourier mode in :func:`riccati_setup`. The mu0-dependent term
    ``P_l^m(-mu0)`` is likewise precomputed host-side with scipy at the **static**
    mu0 (``_padded_legendre_modes`` in ``pydisort_riccati_jax``) — there is no
    in-trace recurrence (mu0 is no longer a traced input; OUTSTANDING §H).

    Parameters
    ----------
    m : int
        Azimuthal Fourier mode index.
    NLeg : int
        Number of Legendre terms.
    mu_arr_pos : (N,) array
        Positive quadrature cosines.

    Returns
    -------
    dict with JAX arrays:
        poch              : (n_ells,) — (l-m)!/(l+m)! via Pochhammer
        weighted_poch     : (n_ells,) — (2l+1) * (l-m)!/(l+m)!
        asso_leg_term_pos : (n_ells, N) — P_l^m(+mu_i)
        asso_leg_term_neg : (n_ells, N) — P_l^m(-mu_i)
    """
    mu_np = np.asarray(mu_arr_pos, dtype=np.float64)  # scipy.special needs float64
    N = len(mu_np)
    ells = np.arange(m, NLeg)
    n_ells = len(ells)

    if n_ells == 0:
        return {
            'poch': jnp.zeros(0),
            'weighted_poch': jnp.zeros(0),
            'asso_leg_term_pos': jnp.zeros((0, N)),
            'asso_leg_term_neg': jnp.zeros((0, N)),
        }

    poch = sp.poch(ells + m + 1, -2.0 * m)
    weighted_poch = (2 * ells + 1) * poch  # (2l+1) * (l-m)!/(l+m)!

    asso_leg_term_pos = np.zeros((n_ells, N))
    asso_leg_term_neg = np.zeros((n_ells, N))
    for idx, l_val in enumerate(ells):
        asso_leg_term_pos[idx] = sp.lpmv(m, l_val, mu_np)
        asso_leg_term_neg[idx] = sp.lpmv(m, l_val, -mu_np)

    return {
        'poch': jnp.array(poch),
        'weighted_poch': jnp.array(weighted_poch),
        'asso_leg_term_pos': jnp.array(asso_leg_term_pos),
        'asso_leg_term_neg': jnp.array(asso_leg_term_neg),
    }


# ---------------------------------------------------------------------------
# JAX-traceable coefficient functions
# (cf. _solve_for_gen_and_part_sols: D_pos, D_neg, alpha, beta construction)
# ---------------------------------------------------------------------------

def _make_alpha_beta_funcs_jax(omega_func, Leg_coeffs_func,
                                weighted_poch_m, asso_leg_pos_m, asso_leg_neg_m,
                                W, M_inv, N, NLeg=None, delta_M=False):
    """Build JAX-traceable alpha(tau) and beta(tau) for the Riccati ODE.

    In pydisort, alpha and beta are constant per layer and computed inline.
    Here they are closures evaluated at each tau during Kvaerno5 integration,
    since omega(tau) and g_l(tau) vary continuously.

    The D^m kernel (without omega) is:
        D^m_ij = (1/2) sum_l (2l+1) * poch_l * g_l * P_l^m(mu_i) * P_l^m(±mu_j)
    omega is applied separately so that omega(tau) and the phase function
    can vary independently (cf. Remark in report section 1.2).

    **Mode-map form (OUTSTANDING §H).** The per-mode Legendre tensors are passed
    as *padded* ``(NLeg,)`` / ``(NLeg, N)`` arrays indexed by absolute l, with the
    ``l < m`` rows pre-zeroed (``weighted_poch_m`` carries the zeros). There is no
    static ``m`` and no ``c[m:]`` slice, so this body runs unchanged for every
    Fourier mode under ``lax.scan`` (the mode index is the scanned axis, not a
    Python int). The l<m terms contribute zero, so the einsums equal the original
    ragged ``(NLeg-m, N)`` contraction.

    Delta-M scaling (``delta_M=True``), physical-tau form (see
    docs/DESIGN_DECISIONS.md): the truncation fraction f(tau) = g_{NLeg}(tau)
    is the first dropped Legendre moment.  The phase-function moments become the
    *effective* moments c_l = g_l - f, and the identity term -I in alpha is
    replaced by -scale_tau * I with scale_tau(tau) = 1 - omega(tau) * f(tau).
    f=0 reproduces the un-scaled kernel exactly.

    Parameters
    ----------
    omega_func      : tau -> scalar
    Leg_coeffs_func : tau -> (NLeg_all,) array of Legendre coefficients
    weighted_poch_m : (NLeg,) padded (2l+1)(l-m)!/(l+m)!, zero for l < m
    asso_leg_pos_m  : (NLeg, N) padded P_l^m(+mu_i), zero for l < m
    asso_leg_neg_m  : (NLeg, N) padded P_l^m(-mu_i), zero for l < m
    W               : (N,) JAX array, quadrature weights
    M_inv           : (N,) JAX array, 1/mu_arr_pos
    N               : int, half-stream count
    NLeg            : int, number of Legendre terms used (f = Leg_coeffs[NLeg])
    delta_M         : bool, enable delta-M scaling

    Returns
    -------
    (alpha_func, beta_func) : each  tau -> (N, N) JAX array
    """
    I_N = jnp.eye(N)

    def _effective_moments(tau, omega):
        """Return (weighted_poch_m * c, scale_tau): delta-M effective moments.

        c_l = g_l - f over all l in [0, NLeg); the padded weighted_poch_m zeroes
        the l < m terms. f=0 (delta_M=False) gives scale_tau=1 and the un-scaled
        moments, reproducing the original kernel bit-for-bit.
        """
        Leg_coeffs = Leg_coeffs_func(tau)
        f = Leg_coeffs[NLeg] if delta_M else 0.0
        c = Leg_coeffs[:NLeg] - f          # effective moments c_l = g_l - f
        scale_tau = 1.0 - omega * f         # = 1 when f = 0
        return weighted_poch_m * c, scale_tau

    def alpha_func(tau):
        omega = omega_func(tau)
        weighted_Leg_coeffs, scale_tau = _effective_moments(tau, omega)
        # D_pos = (1/2) sum_l weighted_Leg_coeffs_l * P_l^m(mu_i) * P_l^m(mu_j)
        D_pos = 0.5 * jnp.einsum('l,li,lj->ij', weighted_Leg_coeffs,
                                  asso_leg_pos_m, asso_leg_pos_m)
        return M_inv[:, None] * (omega * D_pos * W[None, :] - scale_tau * I_N)

    def beta_func(tau):
        omega = omega_func(tau)
        weighted_Leg_coeffs, _ = _effective_moments(tau, omega)
        # D_neg = (1/2) sum_l weighted_Leg_coeffs_l * P_l^m(mu_i) * P_l^m(-mu_j)
        D_neg = 0.5 * jnp.einsum('l,li,lj->ij', weighted_Leg_coeffs,
                                  asso_leg_pos_m, asso_leg_neg_m)
        return M_inv[:, None] * (omega * D_neg * W[None, :])

    return alpha_func, beta_func


def _make_q_funcs_jax(omega_func, Leg_coeffs_func,
                       weighted_poch_m, asso_leg_pos_m, asso_leg_neg_m,
                       asso_leg_mu0_m, M_inv, mu0, I0_div_4pi, m_is_zero, N,
                       NLeg=None, delta_M=False, tau_star_eval=None):
    """Build JAX-traceable beam-source q functions.

    Computes the beam-source vectors Q^+(tau) and Q^-(tau), scaled by 1/mu_i.
    (cf. _solve_for_gen_and_part_sols: X_pos, X_neg computation, and
     section 3.6.1 of the PythonicDISORT docs, pythonic-disort.readthedocs.io)

    **Mode-map form (OUTSTANDING §H), static mu0.** The Legendre tensors — including
    ``asso_leg_mu0_m = P_l^m(-mu0)`` — are *padded* ``(NLeg,)`` / ``(NLeg, N)`` arrays
    indexed by absolute l (zero for l < m), so the body is mode-index-free and runs
    under ``lax.scan``. ``m_is_zero`` (1.0 for mode 0, else 0.0) replaces
    ``int(m == 0)``. mu0 is now **static** (baked into ``setup``): ``asso_leg_mu0_m``
    is precomputed host-side with scipy in :func:`riccati_setup`, so the in-trace
    associated-Legendre recurrence is gone.

    Delta-M scaling (``delta_M=True``): the moments become c_l = g_l - f (f = g_{NLeg})
    and the beam attenuation exp(-tau/mu0) becomes exp(-tau*(tau)/mu0) with the scaled
    cumulative depth ``tau_star_eval``. f=0 + identity tau_star_eval reproduce the
    original source bit-for-bit.

    Parameters
    ----------
    weighted_poch_m : (NLeg,) padded weighted Pochhammer, zero for l < m
    asso_leg_pos_m  : (NLeg, N) padded P_l^m(+mu_i), zero for l < m
    asso_leg_neg_m  : (NLeg, N) padded P_l^m(-mu_i), zero for l < m
    asso_leg_mu0_m  : (NLeg,) padded P_l^m(-mu0), zero for l < m  (static mu0)
    M_inv           : (N,) JAX array
    mu0             : float (static), beam cosine
    I0_div_4pi      : float, I0 / (4 pi) after rescaling
    m_is_zero       : scalar, 1.0 for mode 0 else 0.0
    N, NLeg         : ints
    delta_M         : bool
    tau_star_eval   : callable tau -> tau*(tau); identity if None (un-scaled).

    Returns
    -------
    (q_up_func, q_down_func) : each  tau -> (N,) JAX array
    """
    if tau_star_eval is None:
        tau_star_eval = lambda tau: tau

    # scalar_fac = I0_div_4pi * (2 - delta_m0)
    scalar_fac = I0_div_4pi * (2.0 - m_is_zero)

    def _weighted_eff_moments(tau):
        """weighted_poch_m * c with c_l = g_l - f (padding zeroes l < m)."""
        Leg_coeffs = Leg_coeffs_func(tau)
        f = Leg_coeffs[NLeg] if delta_M else 0.0
        c = Leg_coeffs[:NLeg] - f
        return weighted_poch_m * c

    def q_up_func(tau):
        omega = omega_func(tau)
        # X_pos: beam source for upward direction. No factor 1/2 (unlike D^m).
        X_pos = jnp.einsum('l,li,l->i', _weighted_eff_moments(tau),
                           asso_leg_pos_m, asso_leg_mu0_m)
        return (M_inv * scalar_fac * omega
                * jnp.exp(-tau_star_eval(tau) / mu0) * X_pos)

    def q_down_func(tau):
        omega = omega_func(tau)
        # X_neg: beam source for downward direction.
        X_neg = jnp.einsum('l,li,l->i', _weighted_eff_moments(tau),
                           asso_leg_neg_m, asso_leg_mu0_m)
        return (M_inv * scalar_fac * omega
                * jnp.exp(-tau_star_eval(tau) / mu0) * X_neg)

    return q_up_func, q_down_func


# ---------------------------------------------------------------------------
# Delta-M / Nakajima-Tanaka (TMS) support
# ---------------------------------------------------------------------------

def _compute_tau_star(omega_func, f_of_tau, tau_bot, n_grid=1025):
    """Scaled cumulative optical depth tau*(tau) = int_0^tau (1 - omega f) dtau'.

    The only place delta-M needs a *cumulative* (non-pointwise) quantity: the
    beam attenuation exp(-tau*/mu0).  It is azimuth-mode independent, so it is
    built once before the Fourier loop.  Implementation: sample the smooth
    integrand 1 - omega(tau) f(tau) on a fixed grid, cumulative-trapezoid, and
    linearly interpolate -- fully JAX-traceable/differentiable, no nested ODE
    solve, and (only ever exp(-tau*/.)) preserves the NO-POSITIVE-EXPONENTS
    invariant since tau* >= 0.

    Parameters
    ----------
    omega_func : tau -> scalar single-scattering albedo
    f_of_tau   : tau -> scalar truncation fraction f = g_{NLeg}(tau)
    tau_bot    : float, bottom optical depth
    n_grid     : int, number of fixed sample points (smooth integrand -> coarse OK)

    Returns
    -------
    tau_star_eval : callable tau -> tau*(tau)  (JAX-traceable)
    tau_star_bot  : JAX scalar, tau*(tau_bot)
    """
    # tau_bot may be a JAX tracer (jit/grad path): jnp.linspace accepts a traced
    # `stop` with a static `num`, so no float() concretisation here.
    tau_nodes = jnp.linspace(0.0, tau_bot, n_grid)
    omega_nodes = jax.vmap(omega_func)(tau_nodes)
    f_nodes = jax.vmap(f_of_tau)(tau_nodes)
    scale_tau_nodes = 1.0 - omega_nodes * f_nodes          # integrand
    dx = tau_nodes[1] - tau_nodes[0]
    seg = 0.5 * (scale_tau_nodes[1:] + scale_tau_nodes[:-1]) * dx
    tau_star_nodes = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    tau_star_bot = tau_star_nodes[-1]

    def tau_star_eval(tau):
        return jnp.interp(tau, tau_nodes, tau_star_nodes)

    return tau_star_eval, tau_star_bot


def _legendre_weighted_sum_jax(x, weighted_coeffs):
    """Sum_l weighted_coeffs[l] * P_l(x) via the Bonnet recurrence (lax.scan).

    P_l are ordinary Legendre polynomials; ``weighted_coeffs[l]`` already
    includes any (2l+1) and coefficient factors.  Accumulates on the fly so no
    (L, ...) stack of polynomials is materialised.  JAX-traceable and
    differentiable; the term count L = len(weighted_coeffs) is static.

    Parameters
    ----------
    x               : JAX array of arbitrary shape (evaluation points, e.g. cos Theta)
    weighted_coeffs : (L,) JAX array

    Returns
    -------
    JAX array, same shape as x.
    """
    L = weighted_coeffs.shape[0]
    P0 = jnp.ones_like(x)
    acc = weighted_coeffs[0] * P0
    if L == 1:
        return acc
    P1 = x * jnp.ones_like(x)
    acc = acc + weighted_coeffs[1] * P1
    if L == 2:
        return acc

    ks = jnp.arange(2, L)

    def body(carry, inp):
        P_pp, P_p, acc = carry          # P_{k-2}, P_{k-1}, accumulator
        k, ck = inp
        k_f = k.astype(x.dtype)
        # Bonnet: k P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}
        P_k = ((2.0 * k_f - 1.0) * x * P_p - (k_f - 1.0) * P_pp) / k_f
        return (P_p, P_k, acc + ck * P_k), None

    (_, _, acc), _ = jax.lax.scan(body, (P0, P1, acc), (ks, weighted_coeffs[2:]))
    return acc


class _TMSData(NamedTuple):
    """Precomputed Nakajima-Tanaka TMS state (a JAX pytree of arrays).

    All fields are traceable, so this rides inside a ``jit``-ed solve and is the
    ``tms_data`` carried by ``SolveResult``; :func:`_apply_tms` evaluates the
    correction at requested ``(mu_out, phi)``.  ``phi0`` and ``I0_div_4pi`` are
    concrete scalars (constant-folded under jit); ``mu0`` may be a tracer.
    """
    weighted_b_k: jnp.ndarray   # (Q, NLeg_all) missing-forward-peak coefficients
    fac_k: jnp.ndarray          # (Q,) t_weights * omega_k
    tstar_k: jnp.ndarray        # (Q,) scaled cumulative depth at the nodes
    mu0: jnp.ndarray            # beam cosine
    sqrt_1m_mu0sq: jnp.ndarray  # sqrt(1 - mu0^2)
    phi0: float
    I0_div_4pi: float


def _precompute_tms(omega_func, Leg_coeffs_func, tau_star_eval, tau_bot,
                    mu0, phi0, I0_div_4pi, NLeg, NLeg_all, quad_order=128):
    """Precompute the Nakajima-Tanaka TMS single-scattering correction state.

    Returns a :class:`_TMSData` pytree (traceable; ``jit``-able and a valid
    ``SolveResult`` field).  Pair with :func:`_apply_tms`.  The correction itself
    is the additive upwelling radiance

        Du+_i(phi) = (I0/(4 pi mu_i)) * int_0^{tau_bot}
                     omega(tau) * [p_full - (1-f) p_trunc](cos Theta_i(phi))
                     * exp(-tau*(tau) * (1/mu0 + 1/mu_i)) dtau

    derived (see docs/DESIGN_DECISIONS.md) by mapping PythonicDISORT's per-layer
    TMS coefficient to continuous, physical-tau form.  The angular bracket
    collapses to the *missing forward-peak detail*

        p_full - (1-f) p_trunc = sum_{l<NLeg} (2l+1) f P_l(cosTheta)
                               + sum_{l>=NLeg} (2l+1) g_l P_l(cosTheta),

    i.e. effective coefficients b_l = f for l < NLeg and b_l = g_l for
    l >= NLeg.  ``I0_div_4pi`` must be the *un-rescaled* I0/(4 pi) (the correction
    is added to un-rescaled output).  ``tau_bot`` and ``mu0`` may be tracers.

    A fixed Gauss-Legendre tau-quadrature of order ``quad_order`` integrates the
    smooth integrand (angular sharpness lives in cos Theta, evaluated exactly in
    :func:`_apply_tms`).  Node optics are evaluated once here and reused across
    all ``(mu_out, phi)``.
    """
    # Fixed GL abscissae/weights on [-1, 1] (static; numpy). tau_bot may be a
    # tracer, so the affine map onto [0, tau_bot] is done with jnp.
    x_gl, w_gl = np.polynomial.legendre.leggauss(int(quad_order))
    x_gl = jnp.asarray(x_gl)
    w_gl = jnp.asarray(w_gl)
    t_nodes = 0.5 * tau_bot * (x_gl + 1.0)                   # (Q,)
    t_weights = 0.5 * tau_bot * w_gl                         # (Q,)

    omega_k = jax.vmap(omega_func)(t_nodes)                  # (Q,)
    Leg_k = jax.vmap(Leg_coeffs_func)(t_nodes)               # (Q, NLeg_all)
    tstar_k = jax.vmap(tau_star_eval)(t_nodes)               # (Q,)

    f_k = Leg_k[:, NLeg]                                     # (Q,)
    ell = jnp.arange(NLeg_all)
    low = ell < NLeg
    # b_l = f for l < NLeg, else g_l  -> the missing forward-peak coefficients
    b_k = jnp.where(low[None, :], f_k[:, None], Leg_k)       # (Q, NLeg_all)
    weighted_b_k = (2.0 * ell + 1.0)[None, :] * b_k          # (Q, NLeg_all)
    fac_k = t_weights * omega_k                              # (Q,)

    mu0 = jnp.asarray(mu0, dtype=jnp.result_type(float))
    sqrt_1m_mu0sq = jnp.sqrt(jnp.maximum(1.0 - mu0 ** 2, 0.0))

    return _TMSData(
        weighted_b_k=weighted_b_k, fac_k=fac_k, tstar_k=tstar_k,
        mu0=mu0, sqrt_1m_mu0sq=sqrt_1m_mu0sq,
        phi0=phi0, I0_div_4pi=I0_div_4pi,
    )


def _apply_tms(tms_data, mu_out, phi):
    """Evaluate the TMS upwelling-radiance correction at ``(mu_out, phi)``.

    Parameters
    ----------
    tms_data : _TMSData from :func:`_precompute_tms`.
    mu_out   : scalar or (M,) positive cosines.
    phi      : scalar or (P,) azimuthal angles.

    Returns
    -------
    (M, P) JAX array — the additive correction Du+(mu_out, phi).
    """
    dtype = jnp.result_type(float)
    mu_out = jnp.atleast_1d(jnp.asarray(mu_out, dtype=dtype))            # (M,)
    phi = jnp.atleast_1d(jnp.asarray(phi, dtype=dtype))                  # (P,)
    mu0 = tms_data.mu0
    # cos Theta between (-mu0, phi0) incident and (+mu_out, phi) outgoing
    nu = (mu_out[:, None] * (-mu0)
          + jnp.sqrt(jnp.maximum(1.0 - mu_out[:, None] ** 2, 0.0))
          * tms_data.sqrt_1m_mu0sq
          * jnp.cos(tms_data.phi0 - phi)[None, :])                       # (M, P)
    # bracket(tau_k, cosTheta) for every quadrature node
    brackets = jax.vmap(
        lambda wc: _legendre_weighted_sum_jax(nu, wc)
    )(tms_data.weighted_b_k)                                             # (Q, M, P)
    attn = jnp.exp(
        -tms_data.tstar_k[:, None] * (1.0 / mu0 + 1.0 / mu_out)[None, :]
    )                                                                    # (Q, M)
    contrib = tms_data.fac_k[:, None, None] * attn[:, :, None] * brackets  # (Q,M,P)
    integral = contrib.sum(axis=0)                                       # (M, P)
    return (tms_data.I0_div_4pi / mu_out)[:, None] * integral            # (M, P)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _riccati_rhs_jax(R, alpha, beta):
    """F(R) = alpha R + R alpha + R beta R + beta  (Riccati RHS)."""
    return alpha @ R + R @ alpha + R @ beta @ R + beta


# ---------------------------------------------------------------------------
# Tolerance flooring
# ---------------------------------------------------------------------------

# Per-dtype floor on the relative tolerance `rtol`. The PIDController error test
# is ||error / (atol + rtol*|y|)|| ~ 1; if the target is below what the dtype can
# actually achieve, the estimate can never satisfy it and the controller shrinks
# dt until it hits max_steps (an EquinoxRuntimeError) — see the float32 note in
# pydisort_riccati_jax.py.
#
# The achievable floor is NOT raw machine epsilon: in float32 (eps ~ 1.2e-7),
# roundoff is *amplified* by the nonlinear Riccati matrix products (R@beta@R, ...)
# to an effective accuracy of ~1e-3 on thick atmospheres. Empirically (omega up to
# ~1, tau up to 64) rtol=1e-3 and 5e-4 converge comfortably while 1e-4 max-steps
# out; we floor at the production value 1e-3, which is verified safe across all
# test configurations. In float64 the only limit is raw roundoff, so the floor is
# a few*eps — low enough that legitimate tight work (e.g. resolving the transmission
# operator T ~ 1e-13 in thick atmospheres) is never impeded.
#
# We keep the established `atol = tol*1e-3` coupling (which makes rtol dominate for
# the O(1) reflection state, per the SciPy/SUNDIALS convention) and floor atol in
# step with rtol. A too-tight `tol` therefore *caps accuracy* at the dtype floor
# instead of crashing.
_RTOL_FLOOR_F32 = 1e-3


def _floored_tolerances(tol):
    """Return (rtol, atol) for the adaptive controller in the active dtype.

    rtol = max(tol, floor); atol = max(tol*1e-3, floor*1e-3). At the float32
    production tol=1e-3 the floor does not bind, so behaviour is unchanged. A
    too-tight `tol` is clamped (a one-time warning is emitted) so the solve
    returns at the dtype's achievable accuracy instead of raising max_steps.
    """
    if jnp.result_type(float) == jnp.float32:
        floor = _RTOL_FLOOR_F32
    else:
        floor = 8.0 * float(np.finfo(np.float64).eps)   # ~1.8e-15, never binds in practice
    tol = float(tol)
    if tol < floor:
        warnings.warn(
            f"`tol`={tol:g} is below the {jnp.result_type(float).name} accuracy "
            f"floor {floor:g}; clamping rtol to {floor:g}. Tighter tolerances "
            f"are unreachable in this dtype (use PYDISORT_RICCATI_JAX_X64=1 for "
            f"float64).",
            stacklevel=2,
        )
    rtol = max(tol, floor)
    atol = max(tol * 1e-3, floor * 1e-3)
    return rtol, atol


# ---------------------------------------------------------------------------
# Kvaerno5 integration (core)
# ---------------------------------------------------------------------------

def _kvaerno5_integrate(alpha_func, beta_func, sigma_end, N, tol,
                        q1_func=None, q2_func=None, max_steps=4096,
                        save_grid=True, adjoint=None):
    """Adaptive Kvaerno5 (L-stable ESDIRK, order 5) integration.

    Integrates the coupled Riccati system from sigma=0 to sigma=sigma_end.
    State: PyTree {'R': (N,N), 'T': (N,N), 's': (N,)}.
    IC: R(0)=0, T(0)=I, s(0)=0.

    Parameters
    ----------
    alpha_func, beta_func : sigma -> (N, N) JAX arrays
    sigma_end : float > 0 (may be a JAX tracer when ``save_grid=False``)
    N         : int
    tol       : float > 0
    q1_func, q2_func : sigma -> (N,) JAX arrays, or None
    max_steps : int
    save_grid : bool
        If True (default), save every adaptive step (``SaveAt(steps=True)``) and
        return the realised σ-grid — this needs a host-side ``np.asarray(sol.ts)``
        sync, so it is **not** ``jit``-able. If False, save only the final state
        (``SaveAt(t1=True)``): no host sync, ``sigma_end`` may be traced, and the
        whole integration is ``jit``/``grad``-able. ``sigma_grid`` is then None.
        Only the ToA observable needs the final state, so the jit/retrieval path
        uses ``save_grid=False``; the offline σ-grid (retrieval-grid candidate
        pool, docs/DESIGN_DECISIONS.md §3) is recovered with ``save_grid=True``.
    adjoint : diffrax.AbstractAdjoint or None
        Differentiation strategy passed to ``diffeqsolve``. ``None`` (default)
        uses diffrax's default ``RecursiveCheckpointAdjoint`` — reverse-mode
        (``jax.grad``), the verified discrete adjoint (docs/DESIGN_DECISIONS.md
        §5). For forward-mode AD (``jax.jacfwd``/``jvp``, preferred for
        small-DOF retrieval) pass ``diffrax.ForwardMode()``; the default's
        ``custom_vjp`` cannot be forward-differentiated.

    Returns
    -------
    R, T : (N, N) JAX arrays at sigma_end
    s    : (N,) JAX array (zeros if no beam source)
    sigma_grid : 1-D numpy array [0, sigma_1, ..., sigma_end], or None if
                 ``save_grid=False``.
    """
    has_source = q1_func is not None

    def vector_field(sigma, state, args):
        R = state['R']
        T = state['T']
        s = state['s']

        alpha = alpha_func(sigma)
        beta = beta_func(sigma)

        # Riccati: dR/dsigma = alpha R + R alpha + R beta R + beta
        dR = alpha @ R + R @ alpha + R @ beta @ R + beta

        # Transmission: dT/dsigma = (alpha + R beta) T
        dT = (alpha + R @ beta) @ T

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
    _rtol, _atol = _floored_tolerances(tol)
    controller = diffrax.PIDController(rtol=_rtol, atol=_atol)
    # Only pass `adjoint` when explicitly requested, so the default path is
    # exactly diffrax's default (RecursiveCheckpointAdjoint) — unchanged.
    adjoint_kw = {} if adjoint is None else {'adjoint': adjoint}

    if not save_grid:
        # jit/grad path: only the final state is needed (ToA observable). No
        # host sync, sigma_end may be a tracer -> fully jit-able.
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=sigma_end,
            dt0=None,
            y0=y0,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
            **adjoint_kw,
        )
        R = sol.ys['R'][-1]
        T = sol.ys['T'][-1]
        s = sol.ys['s'][-1]
        return R, T, s, None

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=float(sigma_end),
        dt0=None,
        y0=y0,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(steps=True),
        max_steps=max_steps,
        **adjoint_kw,
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
                          q_up_func=None, q_down_func=None, max_steps=4096,
                          save_grid=True, adjoint=None):
    """Forward Riccati: build slab from bottom (tau_bot) upward to top (0).

    Integration variable sigma in [0, tau_bot].
    Coefficient evaluation: alpha(tau_bot - sigma), beta(tau_bot - sigma).
    Source mapping: q1(sigma) = q_down(tau_bot - sigma),
                    q2(sigma) = q_up(tau_bot - sigma).

    ``tau_bot`` may be a JAX tracer when ``save_grid=False`` (jit/grad path).

    Returns (R_up, T_up, s_up, tau_grid); ``tau_grid`` is None if
    ``save_grid=False``.
    """
    tb = tau_bot
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
        save_grid=save_grid,
        adjoint=adjoint,
    )
    tau_grid = None if sigma_grid is None else tb - sigma_grid[::-1]
    return R, T, s, tau_grid


def _riccati_backward_jax(alpha_func, beta_func, tau_bot, N, tol,
                           q_up_func=None, q_down_func=None, max_steps=4096,
                           save_grid=True, adjoint=None):
    """Backward Riccati: build slab from top (0) downward to bottom (tau_bot).

    Integration variable sigma in [0, tau_bot].
    Coefficient evaluation: alpha(sigma), beta(sigma).
    Source mapping: q1(sigma) = q_up(sigma), q2(sigma) = q_down(sigma).

    ``tau_bot`` may be a JAX tracer when ``save_grid=False`` (jit/grad path).

    Returns (R_down, T_down, s_down, tau_grid); ``tau_grid`` is None if
    ``save_grid=False``.
    """
    tb = tau_bot
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
        save_grid=save_grid,
        adjoint=adjoint,
    )
    return R, T, s, sigma_grid
