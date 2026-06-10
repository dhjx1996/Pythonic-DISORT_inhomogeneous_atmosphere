"""
pydisort_riccati_jax — Riccati forward solver for radiative transfer (JAX + diffrax).

Solves the 1-D RTE for atmospheres with continuously tau-varying
single-scattering albedo omega(tau) and phase function g_l(tau),
using invariant-imbedding Riccati ODE integrated via diffrax Kvaerno5
(L-stable ESDIRK, order 5, adaptive PIDController step-size).

See CLAUDE.md for design rationale and deferred features.

Two ways to call it
-------------------
1. **One-shot** ``pydisort_riccati_jax(...)`` — the original public entry; runs
   host-side setup + the full Fourier solve and returns the documented 5-tuple
   (including the offline ODE ``tau_grid``).  Differentiable but **not** jit-able
   as a whole (it interleaves SciPy setup with the traceable solve).

2. **Composable seam** (for the retrieval loop; docs/DESIGN_DECISIONS.md §7) —
   split the host-side setup from a traceable, jit-able solve::

       setup = riccati_setup(NQuad, I0, phi0, mu0, ...)   # host-side, run once

       def forward(theta, tau_bot):
           of, lf = optics_from(theta)
           res = riccati_solve(setup, of, lf, tau_bot)    # all NFourier modes
           return eval_radiance(setup, res, mu_obs, phi_obs)

       f   = jax.jit(forward)              # compile once (scan over modes), cached
       g   = jax.jit(jax.grad(forward))    # reverse-mode (discrete adjoint, default)

   The K Fourier modes are mapped with ``lax.scan`` (the mode body is compiled
   **once**, O(1) in mode count — OUTSTANDING §H), which removes the
   forward/jacrev compile-memory OOM of the old K-fold unroll. ``setup`` defaults
   to the reverse-mode discrete adjoint (``jax.grad``; §5). For **forward-mode**
   retrieval (``jax.jacfwd``, preferred at small DOF) build the setup with a
   forward-capable adjoint, because diffrax's reverse-mode default is a
   ``custom_vjp`` that cannot be forward-differentiated::

       import diffrax
       setup_fwd = riccati_setup(..., adjoint=diffrax.ForwardMode())
       Jac = jax.jit(jax.jacfwd(lambda th, tb:
                 eval_radiance(setup_fwd, riccati_solve(setup_fwd, *optics_from(th),
                                                         tb), mu_obs, phi_obs)))

   ``tau_bot`` (and the optics closures) are **traced**; grid sizes, ``I0``,
   ``phi0``, ``mu0``, the boundary conditions, the BDRF, and the
   ``delta_M``/``NT_cor`` flags are **static** (baked into ``setup``).  ``mu0`` is
   static (re-build ``setup`` to change solar geometry — one cheap compile per
   mu0). Close ``setup`` over the jitted function (as above); it is a host-side
   object and must **not** be passed as a traced argument.
"""

import os
import jax
# Float32 by default: the Riccati ODE stays O(1) and retrieval precision is
# limited by measurement noise (~10-20%), not floating-point resolution.
# Float32 also keeps the adaptive step count low (the primary cost target) and
# is the only viable mode for tight tolerances on thick atmospheres: float32
# machine epsilon is ~1.2e-7, so any `tol` whose `atol = tol*1e-3` lands at or
# below that asks the step controller to resolve below roundoff -> it shrinks
# dt forever and hits max_steps (see the stringent test partition below).
#
# Opt into float64 by setting PYDISORT_RICCATI_JAX_X64=1 *before* importing this
# module. That is used only by the float64 test partition (`pytest -m float64`),
# which keeps the original tight-tolerance convergence/precision checks where
# the integrator can legitimately reach ~1e-8 (e.g. reference solutions).
_USE_X64 = os.environ.get("PYDISORT_RICCATI_JAX_X64", "0") == "1"
jax.config.update("jax_enable_x64", _USE_X64)

import warnings
from typing import NamedTuple, Any
import jax.numpy as jnp
import numpy as np
import scipy.special as sp
from jax import lax
from math import pi

from PythonicDISORT import subroutines
from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
    _riccati_forward_jax,
    _riccati_backward_jax,
    _compute_tau_star,
    _precompute_tms,
    _apply_tms,
)
from _solve_bc_riccati_jax import _solve_bc_riccati_jax


# ======================================================================
# Composable seam — data containers
# ======================================================================

class SetupData(NamedTuple):
    """Host-side, run-once setup for the Riccati solve (the static contract).

    Built by :func:`riccati_setup`. Holds everything that does **not** depend on
    the traced solve inputs (``tau_bot`` and the optics closures): grid sizes,
    quadrature, the **padded per-mode** Legendre tensors (including the static-mu0
    ``P_l^m(-mu0)`` table and the surface-BDRF arrays), the per-mode
    boundary-condition vectors, the rescale bookkeeping, and the barycentric
    weights. ``mu0`` is now **static** (baked here), so the whole mode loop runs as
    a ``lax.scan`` over the leading axis of these stacks (OUTSTANDING §H).

    This is a host-side object: **close it over** the jitted forward (it becomes
    a compile-time constant); do **not** pass it as a traced ``jax.jit`` argument.

    Fields
    ------
    NQuad, N, NLeg, NFourier, NLeg_all : int
        Stream / Legendre / Fourier counts (``N = NQuad // 2``).
    there_is_beam_source : bool
    I0_div_4pi : float        — rescaled I0/(4 pi) used inside the solve.
    I0_orig_div_4pi : float   — un-rescaled I0/(4 pi) used by the TMS correction.
    rescale_factor : float
    phi0 : float
    mu0 : float               — beam cosine, **static** (in (0, 1]).
    tol : float               — relative tol for the adaptive Kvaerno5 integration.
    mu_arr_pos : (N,) ndarray — positive quadrature cosines (numpy).
    mu_arr_pos_jax, W_jax, M_inv : (N,) JAX arrays
    mu_nodes, bary_weights : (N,) JAX arrays — for :func:`eval_radiance`.
    weighted_poch_modes : (NFourier, NLeg) JAX array
        Padded (2l+1)(l-m)!/(l+m)!, rows l<m zeroed (per Fourier mode m).
    asso_leg_pos_modes, asso_leg_neg_modes : (NFourier, NLeg, N) JAX arrays
        Padded P_l^m(±mu_i), rows l<m zeroed.
    asso_leg_mu0_modes : (NFourier, NLeg) JAX array
        Padded P_l^m(-mu0) (static mu0; scipy host-side), rows l<m zeroed.
    b_pos_modes, b_neg_modes : (NFourier, N) JAX arrays
        Per-mode boundary-condition vectors (already rescaled).
    bdrf_R_modes : (NFourier, N, N) JAX array
        Per-mode raw surface BDRF reflectance (static mu0; zeros for no-surface).
    bdrf_beam_modes : (NFourier, N) JAX array
        Per-mode raw direct-beam surface reflectance ``BDRF(mu_i, mu0)`` (static mu0).
    m_is_zero : (NFourier,) JAX array
        1.0 for mode 0 else 0.0 (the (2 - delta_{m0}) / (1 + delta_{m0}) factors).
    delta_M_scaling, NT_cor : bool
    NT_quad_order : int
    adjoint : diffrax.AbstractAdjoint or None
        Differentiation strategy for the ODE solve. ``None`` (default) =
        diffrax's reverse-mode ``RecursiveCheckpointAdjoint`` (the verified
        discrete adjoint; use with ``jax.grad``). For forward-mode retrieval
        (``jax.jacfwd``, small DOF) build the setup with
        ``adjoint=diffrax.ForwardMode()``.
    """
    NQuad: int
    N: int
    NLeg: int
    NFourier: int
    NLeg_all: int
    there_is_beam_source: bool
    I0_div_4pi: float
    I0_orig_div_4pi: float
    rescale_factor: float
    phi0: float
    mu0: float
    tol: float
    mu_arr_pos: Any
    mu_arr_pos_jax: Any
    W_jax: Any
    M_inv: Any
    mu_nodes: Any
    bary_weights: Any
    weighted_poch_modes: Any
    asso_leg_pos_modes: Any
    asso_leg_neg_modes: Any
    asso_leg_mu0_modes: Any
    b_pos_modes: Any
    b_neg_modes: Any
    bdrf_R_modes: Any
    bdrf_beam_modes: Any
    m_is_zero: Any
    delta_M_scaling: bool
    NT_cor: bool
    NT_quad_order: int
    adjoint: Any


class SolveResult(NamedTuple):
    """Output of :func:`riccati_solve` / :func:`_fourier_solve` — a JAX pytree.

    Fields
    ------
    u_modes : (K, N) JAX array
        The K computed Fourier modes of the ToA upwelling intensity (rescaled to
        physical units). ``u_modes[0]`` is ``u0_ToA``.
    tms_data : _TMSData or None
        Precomputed Nakajima-Tanaka TMS state (``None`` unless ``NT_cor``).
        Consumed by :func:`eval_radiance` to add the analytic single-scattering
        correction at the requested mu.
    tau_grid : ndarray or None
        The m=0 ODE tau-grid (the retrieval-grid candidate pool,
        docs/DESIGN_DECISIONS.md §3) — only populated on the offline
        ``return_grid=True`` path; ``None`` on the jit-able path.
    """
    u_modes: Any
    tms_data: Any
    tau_grid: Any


# ======================================================================
# Composable seam — host-side setup
# ======================================================================

def _bc_mode_arrays(b, N, NFourier, rescale_factor, which):
    """Expand a boundary condition into a stacked ``(NFourier, N)`` array (host-side).

    Mirrors pydisort: a scalar or (N,) vector contributes only to the zeroth
    Fourier mode (higher modes get zeros); a full (N, NFourier) array gives
    column m to mode m. Each mode is rescaled by ``1/rescale_factor`` (if > 0).
    The stacked layout (vs the old tuple) lets the mode loop ``lax.scan`` over it.

    ``which`` is "bottom" (b_pos) or "top" (b_neg), for the error message.
    """
    b_arr = np.atleast_1d(b)
    is_scalar = len(b_arr) == 1
    is_vector = (not is_scalar) and len(b) == N
    if not (is_scalar or is_vector) and np.shape(b) != (N, NFourier):
        raise ValueError(f"The shape of the {which} boundary condition is incorrect.")

    modes = np.zeros((NFourier, N))
    for m in range(NFourier):
        if is_scalar:
            bm = np.full(N, float(b)) if m == 0 else np.zeros(N)
        elif is_vector:
            bm = np.asarray(b, dtype=float) if m == 0 else np.zeros(N)
        else:
            bm = np.asarray(b)[:, m]
        if rescale_factor > 0:
            bm = bm / rescale_factor
        modes[m] = bm
    return jnp.asarray(modes)


def _padded_legendre_modes(NFourier, NLeg, N, mu_arr_pos, mu0):
    """Build the padded per-mode Legendre stacks + static-mu0 table (host-side).

    For each Fourier mode m, :func:`_precompute_legendre` returns ragged
    ``(NLeg-m, .)`` tensors for l = m..NLeg-1; we place them at absolute index l
    (rows l<m left zero) so the stacks are uniform ``(NFourier, NLeg[, N])`` and the
    mode loop can ``lax.scan`` over axis 0. ``asso_leg_mu0`` is computed here with
    scipy at the **static** mu0 (the in-trace recurrence is gone).

    Returns
    -------
    (weighted_poch, asso_leg_pos, asso_leg_neg, asso_leg_mu0) : JAX arrays of
        shapes (NFourier, NLeg), (NFourier, NLeg, N), (NFourier, NLeg, N),
        (NFourier, NLeg).
    """
    mu_np = np.asarray(mu_arr_pos, dtype=np.float64)
    weighted_poch = np.zeros((NFourier, NLeg))
    asso_leg_pos = np.zeros((NFourier, NLeg, N))
    asso_leg_neg = np.zeros((NFourier, NLeg, N))
    asso_leg_mu0 = np.zeros((NFourier, NLeg))
    for m in range(NFourier):
        ld = _precompute_legendre(m, NLeg, mu_arr_pos)
        weighted_poch[m, m:] = np.asarray(ld['weighted_poch'])
        asso_leg_pos[m, m:, :] = np.asarray(ld['asso_leg_term_pos'])
        asso_leg_neg[m, m:, :] = np.asarray(ld['asso_leg_term_neg'])
        for l in range(m, NLeg):
            asso_leg_mu0[m, l] = sp.lpmv(m, l, -float(mu0))
    return (jnp.asarray(weighted_poch), jnp.asarray(asso_leg_pos),
            jnp.asarray(asso_leg_neg), jnp.asarray(asso_leg_mu0))


def _bdrf_mode_arrays(BDRF_Fourier_modes, NFourier, N, mu_arr_pos, mu0):
    """Padded surface-BDRF stacks (static mu0; host-side).

    Mirrors the BDRF branches of the (old) BC solve, but evaluated once host-side
    at the static mu0 and stored per Fourier mode so the BC runs uniformly under
    ``lax.scan``. Mode m gets the BDRF Fourier coefficient ``BDRF_Fourier_modes[m]``
    (scalar or callable; matrix supported for the reflectance but not the beam
    term — no test/​use needs a true (N,N) bidirectional surface). Modes m >= NBDRF
    (and the no-surface case) get zeros, which makes the BC's surface terms vanish.

    Returns
    -------
    (bdrf_R_modes, bdrf_beam_modes) : (NFourier, N, N) and (NFourier, N) JAX arrays.
        ``R`` is the raw reflectance ``BDRF(mu_i, mu_j)``; ``beam`` is the raw
        direct-beam reflectance ``BDRF(mu_i, mu0)``.
    """
    mu_pos = np.asarray(mu_arr_pos, dtype=float)
    NBDRF = len(BDRF_Fourier_modes)
    R_modes = np.zeros((NFourier, N, N))
    beam_modes = np.zeros((NFourier, N))
    for m in range(NFourier):
        if m >= NBDRF:
            continue
        bdrf = BDRF_Fourier_modes[m]
        if callable(bdrf):
            R_val = np.asarray(bdrf(mu_pos, mu_pos), dtype=float)
            beam_val = np.asarray(bdrf(mu_pos, float(mu0)), dtype=float).ravel()
        else:
            R_val = np.asarray(bdrf, dtype=float)        # scalar or (N, N)
            beam_val = R_val * np.ones(N)                # scalar -> (N,)
        R_modes[m] = np.broadcast_to(R_val, (N, N))
        beam_modes[m] = np.broadcast_to(beam_val, (N,))
    return jnp.asarray(R_modes), jnp.asarray(beam_modes)


def riccati_setup(
    NQuad,
    I0,
    phi0,
    mu0,
    NLeg=None,
    NFourier=None,
    NLeg_all=None,
    b_pos=0,
    b_neg=0,
    BDRF_Fourier_modes=(),
    delta_M_scaling=False,
    NT_cor=False,
    NT_quad_order=128,
    tol=1e-3,
    adjoint=None,
):
    """Build the host-side :class:`SetupData` for a Riccati solve (run once).

    Performs all the SciPy-based, **static** work — input validation, double-Gauss
    quadrature, the padded per-mode Legendre tensors (incl. the static-``mu0``
    ``P_l^m(-mu0)`` table and the surface-BDRF arrays), per-mode boundary
    conditions, source rescaling, and barycentric weights — so that the subsequent
    :func:`riccati_solve` is a pure, jit-able function of the traced inputs
    (``tau_bot`` and the optics closures). ``mu0`` is static (baked in here), which
    is what lets the whole mode loop run as a ``lax.scan`` (OUTSTANDING §H). See the
    module docstring for the full recipe.

    Parameters
    ----------
    NQuad : int
        Number of quadrature streams (even, >= 2; NQuad >= 6 recommended).
    I0 : float
        Beam intensity (>= 0).
    phi0 : float
        Beam azimuthal angle, in [0, 2 pi).
    mu0 : float
        Cosine of the beam zenith angle, in (0, 1]. **Static** — re-build the setup
        to change geometry (one compile per mu0; cheap relative to the solve).
    NLeg, NFourier, NLeg_all : int or None
        Legendre / Fourier / total-Legendre counts (default: NQuad, NQuad, NLeg).
    b_pos, b_neg : float / (N,) / (N, NFourier)
        Bottom / top diffuse boundary conditions.
    BDRF_Fourier_modes : list
        Bidirectional reflectance Fourier-mode coefficients (scalars or callables;
        a true (N,N) matrix is supported for the reflectance but not the beam term).
        Evaluated host-side at the static ``mu0`` (so a callable BDRF is now fully
        fine — no traced-mu0 jit hazard).
    delta_M_scaling, NT_cor : bool
        Delta-M scaling and the Nakajima-Tanaka TMS correction (see
        ``pydisort_riccati_jax`` for the guards and docs/DESIGN_DECISIONS.md §6).
    NT_quad_order : int
        Gauss-Legendre order for the TMS tau-quadrature.
    tol : float
        Relative tolerance for the adaptive Kvaerno5 ODE integration.
    adjoint : diffrax.AbstractAdjoint or None
        ODE differentiation strategy. ``None`` (default) = reverse-mode
        ``RecursiveCheckpointAdjoint`` (the verified discrete adjoint; use with
        ``jax.grad``). For forward-mode retrieval (``jax.jacfwd``, small DOF)
        pass ``diffrax.ForwardMode()`` — the reverse-mode default's ``custom_vjp``
        cannot be forward-differentiated.

    Returns
    -------
    SetupData
    """
    if NLeg is None:
        NLeg = NQuad
    if NFourier is None:
        NFourier = NQuad
    if NLeg_all is None:
        NLeg_all = NLeg

    N = NQuad // 2
    there_is_beam_source = I0 > 0

    # ---- Input checks (mirror pydisort; the tau_bot / mu0 solve-param checks
    #      live in the caller, since those are traced in the jit path) --------
    if not NQuad >= 2:
        raise ValueError("There must be at least two streams.")
    if not NQuad % 2 == 0:
        raise ValueError("The number of streams must be even.")
    if not NLeg > 0:
        raise ValueError(
            "The number of phase function Legendre coefficients must be positive."
        )
    if not NFourier > 0:
        raise ValueError(
            "The number of Fourier modes to use in the solution must be positive."
        )
    if not NFourier <= NLeg:
        raise ValueError(
            "The number of Fourier modes to use in the solution must be "
            "less than or equal to the number of phase function Legendre "
            "coefficients used."
        )
    if not NLeg <= NQuad:
        raise ValueError(
            "There should be more streams than the number of phase function "
            "Legendre coefficients used."
        )
    if NFourier > 64:
        warnings.warn(
            "`NFourier` is large and may cause errors, consider decreasing "
            "`NFourier` to 64 and it probably should be even less. "
            "By default `NFourier` equals `NQuad`."
        )
    if I0 < 0:
        raise ValueError("The intensity of the incident beam cannot be negative.")
    if there_is_beam_source:
        if not (0 <= phi0 and phi0 < 2 * pi):
            raise ValueError(
                "Provide the principal azimuthal angle for the incident beam "
                "(must be between 0 and 2pi, excluding 2pi)."
            )
    # mu0 is now static and always required: the beam-source exp(-tau*/mu0) is
    # evaluated for every mode (zeroed by I0_div_4pi=0 when there is no beam, but
    # mu0 must still be a valid positive cosine to keep that term finite).
    if not (0 < mu0 <= 1):
        raise ValueError(
            "The cosine of the polar angle of the incident beam (mu0) must be "
            "in (0, 1]."
        )
    if tol <= 0:
        raise ValueError("`tol` must be positive.")
    if not NLeg_all >= NLeg:
        raise ValueError(
            "`NLeg_all` (number of Legendre coefficients returned by "
            "`Leg_coeffs_func`) must be >= `NLeg`."
        )
    if delta_M_scaling and NLeg_all < NLeg + 1:
        raise ValueError(
            "Delta-M scaling needs the first dropped Legendre moment "
            "f = g_{NLeg}, so `NLeg_all` must be >= NLeg + 1. Pass a "
            "`Leg_coeffs_func` returning more coefficients (and set `NLeg_all`)."
        )
    if NT_cor:
        if not delta_M_scaling:
            raise ValueError(
                "`NT_cor` (Nakajima-Tanaka TMS correction) requires "
                "`delta_M_scaling=True`."
            )
        if NLeg_all <= NLeg:
            raise ValueError(
                "`NT_cor` needs additional (untruncated) Legendre moments: "
                "`NLeg_all` must exceed `NLeg`."
            )
        if not there_is_beam_source:
            raise ValueError(
                "`NT_cor` corrects the beam single-scattering and requires a "
                "beam source (I0 > 0)."
            )

    # ---- Double-Gauss quadrature -----------------------------------------
    mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
    mu_arr_pos_jax = jnp.array(mu_arr_pos)
    W_jax = jnp.array(W)
    M_inv = 1.0 / mu_arr_pos_jax

    # ---- Source rescaling (concrete; static inputs only) ------------------
    I0_orig_div_4pi = float(I0) / (4 * pi)
    rescale_factor = float(np.max((I0, np.max(b_pos), np.max(b_neg))))
    if rescale_factor > 0:
        I0_rescaled = I0 / rescale_factor
    else:
        I0_rescaled = 0.0
    I0_div_4pi = I0_rescaled / (4 * pi)

    # ---- Per-mode boundary conditions (rescaled) --------------------------
    b_pos_modes = _bc_mode_arrays(b_pos, N, NFourier, rescale_factor, "bottom")
    b_neg_modes = _bc_mode_arrays(b_neg, N, NFourier, rescale_factor, "top")

    # ---- Padded per-mode Legendre stacks + static-mu0 P_l^m(-mu0) table ----
    (weighted_poch_modes, asso_leg_pos_modes,
     asso_leg_neg_modes, asso_leg_mu0_modes) = _padded_legendre_modes(
        NFourier, NLeg, N, mu_arr_pos, mu0)

    # ---- Padded surface-BDRF stacks (static mu0) --------------------------
    bdrf_R_modes, bdrf_beam_modes = _bdrf_mode_arrays(
        BDRF_Fourier_modes, NFourier, N, mu_arr_pos, mu0)

    m_is_zero = jnp.asarray(
        [1.0 if m == 0 else 0.0 for m in range(NFourier)])

    # ---- Barycentric weights for mu-interpolation -------------------------
    bary_weights = jnp.asarray(_compute_bary_weights(np.asarray(mu_arr_pos)))

    return SetupData(
        NQuad=NQuad, N=N, NLeg=NLeg, NFourier=NFourier, NLeg_all=NLeg_all,
        there_is_beam_source=there_is_beam_source,
        I0_div_4pi=I0_div_4pi, I0_orig_div_4pi=I0_orig_div_4pi,
        rescale_factor=rescale_factor, phi0=float(phi0), mu0=float(mu0),
        tol=float(tol),
        mu_arr_pos=mu_arr_pos, mu_arr_pos_jax=mu_arr_pos_jax,
        W_jax=W_jax, M_inv=M_inv,
        mu_nodes=mu_arr_pos_jax, bary_weights=bary_weights,
        weighted_poch_modes=weighted_poch_modes,
        asso_leg_pos_modes=asso_leg_pos_modes,
        asso_leg_neg_modes=asso_leg_neg_modes,
        asso_leg_mu0_modes=asso_leg_mu0_modes,
        b_pos_modes=b_pos_modes, b_neg_modes=b_neg_modes,
        bdrf_R_modes=bdrf_R_modes, bdrf_beam_modes=bdrf_beam_modes,
        m_is_zero=m_is_zero,
        delta_M_scaling=delta_M_scaling, NT_cor=NT_cor,
        NT_quad_order=NT_quad_order, adjoint=adjoint,
    )


# ======================================================================
# Composable seam — the shared, traceable Fourier solve
# ======================================================================

def _fourier_solve(setup, omega_func, Leg_coeffs_func, tau_bot,
                   *, num_modes, return_grid):
    """Run the Fourier-mode solve for ``num_modes`` modes (the shared core).

    Both the one-shot ``pydisort_riccati_jax`` (``return_grid=True``) and the
    jit-able :func:`riccati_solve` (``return_grid=False``) delegate here, so they
    are numerically identical. ``tau_bot`` and the optics closures are traced;
    ``mu0`` is static (in ``setup``); ``num_modes`` is a static Python int.

    The modes are mapped with **``lax.scan``** over the padded per-mode stacks
    (OUTSTANDING §H): the Kvaerno5 mode body is compiled **once** (O(1) in mode
    count) instead of unrolled K times, which removes the forward/jacrev
    compile-memory OOM while preserving each mode's *independent* adaptive
    stepping. Each mode's solve is identical to the old unrolled body (validated to
    ~1e-15, scan-vs-unrolled, fwd + jacrev + jacfwd; see tests/supplementary).

    With ``return_grid=False`` nothing forces a host sync, so the whole call is
    ``jit`` / ``grad`` / ``jacfwd``-able; with ``return_grid=True`` the m=0 ODE
    tau-grid is recovered in a small **un-scanned** branch (offline retrieval-grid
    pool, not jit-able).

    Returns a :class:`SolveResult`.
    """
    N = setup.N
    NLeg = setup.NLeg
    K = num_modes

    # Delta-M scaled cumulative depth tau*(tau) (azimuth-mode independent: built
    # once, before the mode map). tau_bot may be traced.
    if setup.delta_M_scaling:
        f_of_tau = lambda tau: Leg_coeffs_func(tau)[NLeg]
        tau_star_eval, tau_star_bot = _compute_tau_star(
            omega_func, f_of_tau, tau_bot
        )
    else:
        tau_star_eval = None
        tau_star_bot = tau_bot

    def one_mode(wp_m, ap_pos_m, ap_neg_m, amu0_m, bpos_m, bneg_m,
                 R_raw_m, beam_m, mz, *, save_grid):
        """One Fourier mode's ToA upwelling vector u_m (N,). The scan body core."""
        alpha_func, beta_func = _make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m,
            setup.W_jax, setup.M_inv, N, NLeg, setup.delta_M_scaling,
        )
        q_up, q_down = _make_q_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m, amu0_m,
            setup.M_inv, setup.mu0, setup.I0_div_4pi, mz, N, NLeg,
            setup.delta_M_scaling, tau_star_eval,
        )
        R_up, T_up, s_up, tau_grid_m = _riccati_forward_jax(
            alpha_func, beta_func, tau_bot, N, setup.tol,
            q_up_func=q_up, q_down_func=q_down, save_grid=save_grid,
            adjoint=setup.adjoint,
        )
        R_down, T_down, s_down, _ = _riccati_backward_jax(
            alpha_func, beta_func, tau_bot, N, setup.tol,
            q_up_func=q_up, q_down_func=q_down, save_grid=False,
            adjoint=setup.adjoint,
        )
        u_m = _solve_bc_riccati_jax(
            R_up, T_up, T_down, R_down, s_up, s_down, N,
            bpos_m, bneg_m, R_raw_m, beam_m, mz,
            setup.mu_arr_pos_jax, setup.W_jax, setup.I0_div_4pi,
            setup.mu0, tau_star_bot,
        )
        return u_m, tau_grid_m

    # Padded per-mode stacks, sliced to the K computed modes.
    stacks = (setup.weighted_poch_modes[:K], setup.asso_leg_pos_modes[:K],
              setup.asso_leg_neg_modes[:K], setup.asso_leg_mu0_modes[:K],
              setup.b_pos_modes[:K], setup.b_neg_modes[:K],
              setup.bdrf_R_modes[:K], setup.bdrf_beam_modes[:K],
              setup.m_is_zero[:K])

    def _scan_body(carry, x):
        u_m, _ = one_mode(*x, save_grid=False)
        return carry, u_m

    tau_grid_m0 = None
    if return_grid:
        # Recover the m=0 ODE grid in a small un-scanned branch (offline; needs a
        # host sync, so it cannot live inside the scan). Scan the remaining modes.
        u0, tau_grid_m0 = one_mode(*(s[0] for s in stacks), save_grid=True)
        if K > 1:
            _, u_rest = lax.scan(_scan_body, (), tuple(s[1:] for s in stacks))
            u_modes_arr = jnp.concatenate([u0[None, :], u_rest], axis=0)
        else:
            u_modes_arr = u0[None, :]
    else:
        _, u_modes_arr = lax.scan(_scan_body, (), stacks)        # (K, N)

    if setup.rescale_factor > 0:
        u_modes_arr = u_modes_arr * setup.rescale_factor

    tms_data = None
    if setup.NT_cor:
        tms_data = _precompute_tms(
            omega_func, Leg_coeffs_func, tau_star_eval, tau_bot,
            setup.mu0, setup.phi0, setup.I0_orig_div_4pi, NLeg, setup.NLeg_all,
            setup.NT_quad_order,
        )

    return SolveResult(u_modes=u_modes_arr, tms_data=tms_data, tau_grid=tau_grid_m0)


def riccati_solve(setup, omega_func, Leg_coeffs_func, tau_bot, num_modes=None):
    """Traceable, jit-able Riccati solve (the retrieval forward model).

    A pure function of the traced inputs (``tau_bot`` and the ``omega_func`` /
    ``Leg_coeffs_func`` optics closures) given a host-side ``setup`` (close it over
    the jitted function — see the module docstring). ``mu0`` is now static (in
    ``setup``). Wrap with ``jax.jit`` / ``jax.grad`` / ``jax.jacfwd`` freely.

    Parameters
    ----------
    setup : SetupData
        From :func:`riccati_setup` (carries the static ``mu0``).
    omega_func : callable
        ``tau -> omega`` (scalar in [0, 1)).
    Leg_coeffs_func : callable
        ``tau -> (NLeg_all,)`` Legendre coefficients g_l(tau).
    tau_bot : float or JAX scalar
        Bottom optical depth (> 0). Traced.
    num_modes : int or None
        Number of Fourier modes to compute (static). Default ``setup.NFourier``.
        For a truncated solve pick the count offline (e.g. the S_ε mode selector in
        ``retrieval_oe``) and pass it here; post-scan, fewer modes is a *runtime*
        saving, no longer a compile-memory necessity.

    Returns
    -------
    SolveResult
    """
    if num_modes is None:
        num_modes = setup.NFourier
    num_modes = int(num_modes)
    if not (1 <= num_modes <= setup.NFourier):
        raise ValueError(
            f"`num_modes` must be in [1, NFourier={setup.NFourier}], got {num_modes}."
        )
    return _fourier_solve(
        setup, omega_func, Leg_coeffs_func, tau_bot,
        num_modes=num_modes, return_grid=False,
    )


def eval_radiance(setup, result, mu, phi):
    """ToA upwelling radiance at arbitrary ``(mu, phi)`` from a SolveResult.

    The retrieval observable. Barycentrically interpolates the *smooth*
    multiple-scattering Fourier field in mu, then adds the analytic
    Nakajima-Tanaka TMS single-scattering correction (if ``NT_cor``) **directly
    at the requested mu** (never interpolating the sharp single-scatter peak).
    Fully JAX-traceable.

    Parameters
    ----------
    setup : SetupData
    result : SolveResult
    mu : scalar or (M,) positive cosines in (0, 1].
    phi : scalar or (P,) azimuthal angles.

    Returns
    -------
    JAX array, shape per broadcasting: scalar mu & phi -> scalar; (M,) & scalar
    -> (M,); scalar & (P,) -> (P,); (M,) & (P,) -> (M, P).
    """
    dtype = jnp.result_type(float)
    mu_q = jnp.atleast_1d(jnp.asarray(mu, dtype=dtype))     # (M,)
    phi_a = jnp.atleast_1d(jnp.asarray(phi, dtype=dtype))   # (P,)

    u_modes = result.u_modes                                # (K, N)
    K = u_modes.shape[0]
    m_arr = jnp.arange(K)
    cos_phases = jnp.cos(jnp.outer(m_arr, setup.phi0 - phi_a))   # (K, P)
    u_smooth_nodes = u_modes.T @ cos_phases                      # (N, P)

    out = _barycentric_interpolate(
        mu_q, setup.mu_nodes, u_smooth_nodes, setup.bary_weights
    )  # (M, P)

    if result.tms_data is not None:
        out = out + _apply_tms(result.tms_data, mu_q, phi_a)    # (M, P)

    mu_scalar = jnp.ndim(mu) == 0
    phi_scalar = jnp.ndim(phi) == 0
    if mu_scalar and phi_scalar:
        return out[0, 0]
    if mu_scalar:
        return out[0, :]
    if phi_scalar:
        return out[:, 0]
    return out


# ======================================================================
# One-shot public entry (delegates to the composable seam)
# ======================================================================

def pydisort_riccati_jax(
    tau_bot,
    omega_func,
    Leg_coeffs_func,
    NQuad,
    mu0,
    I0,
    phi0,
    NLeg=None,
    NFourier=None,
    tol=1e-3,
    b_pos=0,
    b_neg=0,
    BDRF_Fourier_modes=[],
    delta_M_scaling=False,
    NLeg_all=None,
    NT_cor=False,
    NT_quad_order=128,
):
    """
    Riccati forward solver for a single atmospheric column with
    continuously tau-varying single-scattering albedo and phase function.

    The original one-shot entry point: runs :func:`riccati_setup` (passing ``mu0``,
    now static) then the full Fourier solve (all ``NFourier`` modes) and returns the
    documented 5-tuple. For the jit-able retrieval path use the composable seam
    instead (module docstring; :func:`riccati_setup` / :func:`riccati_solve` /
    :func:`eval_radiance`).

    Parameters
    ----------
    tau_bot : float
        Optical depth of the bottom boundary (> 0).
    omega_func : callable
        tau -> omega (float in [0, 1)).
    Leg_coeffs_func : callable
        tau -> (NLeg,) array of Legendre coefficients g_l(tau).
    NQuad : int
        Number of quadrature streams (even, >= 2).
    mu0 : float
        Cosine of the beam zenith angle, in (0, 1].
    I0 : float
        Beam intensity (>= 0).
    phi0 : float
        Beam azimuthal angle, in [0, 2pi).
    NLeg : int or None, optional
        Number of Legendre terms (default: NQuad).
    NFourier : int or None, optional
        Number of azimuthal Fourier modes (default: NQuad).
    tol : float, optional
        Relative tolerance for adaptive Kvaerno5 (default 1e-3).
    b_pos : float or (N,) or (N, NFourier), optional
        Upward diffuse intensity at the bottom boundary.
    b_neg : float or (N,) or (N, NFourier), optional
        Downward diffuse intensity at the top boundary.
    BDRF_Fourier_modes : list, optional
        Bidirectional reflectance Fourier mode coefficients.
    delta_M_scaling : bool, optional
        Enable delta-M scaling (Wiscombe 1977) to remove the forward-scattering
        peak from the truncated phase function, which otherwise makes the
        finite-stream radiance ring (and go negative) for forward-peaked clouds.
        The truncation fraction is derived internally as f(tau) =
        Leg_coeffs_func(tau)[NLeg] (the first dropped Legendre moment); it is
        never passed separately.  Default False (behaviour unchanged).
    NLeg_all : int or None, optional
        Number of Legendre coefficients ``Leg_coeffs_func(tau)`` returns. For
        delta-M this must be >= NLeg + 1 (so g_{NLeg} = f is available); for a
        good TMS correction supply as many as available (the full Mie/HG
        expansion). Defaults to NLeg (no extra moments).
    NT_cor : bool, optional
        Add the Nakajima-Tanaka TMS single-scattering correction to
        ``u_ToA_func`` (and to ``interpolate``). Requires ``delta_M_scaling``,
        ``NLeg_all > NLeg``, and a beam source. Default False. (IMS is omitted:
        it corrects only the downward field, irrelevant to a ToA-upwelling
        observable -- see docs/DESIGN_DECISIONS.md.)
    NT_quad_order : int, optional
        Gauss-Legendre order for the 1-D tau-quadrature of the TMS integral
        (default 128); the integrand is smooth so this is generous.

    Returns
    -------
    mu_arr_pos : (N,) ndarray  — positive quadrature cosines (upwelling)
    flux_up_ToA : JAX scalar  — upward diffuse flux at ToA (traceable; float() it
        outside any jax transform if a Python number is needed)
    u0_ToA : (N,) ndarray  — upwelling intensity at ToA (zeroth Fourier mode)
    u_ToA_func : callable  phi -> (N,) or (N, len(phi))
    tau_grid : ndarray  [0, ..., tau_bot]
    """
    # Solve-parameter validation (tau_bot is traced on the seam path, so this
    # concrete check lives here; mu0 is now validated in riccati_setup).
    if tau_bot <= 0:
        raise ValueError("tau values cannot be non-positive.")

    # Host-side setup (mu0 is static, baked into setup).
    setup = riccati_setup(
        NQuad, I0, phi0, mu0, NLeg=NLeg, NFourier=NFourier, NLeg_all=NLeg_all,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
        delta_M_scaling=delta_M_scaling, NT_cor=NT_cor,
        NT_quad_order=NT_quad_order, tol=tol,
    )

    # Full solve with the offline m=0 ODE grid retained.
    result = _fourier_solve(
        setup, omega_func, Leg_coeffs_func, tau_bot,
        num_modes=setup.NFourier, return_grid=True,
    )

    u_modes_arr = result.u_modes           # (NFourier, N), rescaled
    u0_ToA = u_modes_arr[0]

    # Upward diffuse flux at ToA: 2pi * sum_i w_i mu_i u+(0)_i. Kept as a JAX
    # scalar (NOT float()) so the whole function stays traceable: an eager
    # float() here concretizes and breaks `jax.grad` through the solve — i.e. the
    # discrete-adjoint retrieval Jacobian.
    flux_up_ToA = 2 * pi * jnp.dot(setup.mu_arr_pos_jax * setup.W_jax, u0_ToA)

    # Upwelling intensity function at tau=0 (delta-M-scaled multiple-scattering
    # field; smooth in mu -- the part safe to mu-interpolate).
    def u_smooth_func(phi):
        phi = jnp.atleast_1d(jnp.asarray(phi, dtype=float))
        m_arr = jnp.arange(u_modes_arr.shape[0])
        cos_phases = jnp.cos(jnp.outer(m_arr, setup.phi0 - phi))  # (NFourier, P)
        res = u_modes_arr.T @ cos_phases                          # (N, P)
        if res.shape[1] == 1:
            return res[:, 0]
        return res

    if setup.NT_cor:
        tms_data = result.tms_data

        def u_ToA_func(phi):
            base = u_smooth_func(phi)                            # (N,) or (N, P)
            corr = _apply_tms(tms_data, setup.mu_arr_pos_jax, phi)  # (N, P)
            if base.ndim == 1:
                corr = corr[:, 0]
            return base + corr

        # Expose the smooth field and the analytic correction so that
        # `interpolate` can add TMS *directly at the requested mu*.
        u_ToA_func.u_smooth = u_smooth_func
        u_ToA_func.tms = lambda mu, phi: _apply_tms(tms_data, mu, phi)
    else:
        u_ToA_func = u_smooth_func

    return setup.mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, result.tau_grid


# ======================================================================
# Barycentric Lagrange interpolation in mu (JAX-traceable)
# ======================================================================

def _compute_bary_weights(nodes):
    """Barycentric weights for Lagrange interpolation.

    Parameters
    ----------
    nodes : (N,) numpy array of distinct interpolation nodes.

    Returns
    -------
    weights : (N,) numpy array, w_j = 1 / prod_{k != j} (x_j - x_k).
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    weights = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                weights[j] /= (nodes[j] - nodes[k])
    return weights


def _barycentric_interpolate(mu_query, mu_nodes, values, bary_weights):
    """Barycentric Lagrange interpolation — JAX-traceable.

    Parameters
    ----------
    mu_query : (M,) JAX array of query points.
    mu_nodes : (N,) JAX array of interpolation nodes.
    values : (N,) or (N, K) JAX array of function values at nodes.
    bary_weights : (N,) JAX array of precomputed barycentric weights.

    Returns
    -------
    result : (M,) or (M, K) interpolated values.
    """
    # diff[i, j] = mu_query[i] - mu_nodes[j], shape (M, N)
    diff = mu_query[:, None] - mu_nodes[None, :]

    # Detect exact node matches (avoid division by zero).
    # Both branches of jnp.where must be NaN-free for clean JAX gradients.
    is_node = jnp.abs(diff) < 1e-14
    safe_diff = jnp.where(is_node, 1.0, diff)

    # Kernel: w_j / (mu - mu_j), zeroed at exact matches
    kernel = jnp.where(is_node, 0.0, bary_weights[None, :] / safe_diff)  # (M, N)

    if values.ndim == 1:
        numer = kernel @ values                 # (M,)
        denom = kernel.sum(axis=1)              # (M,)
        interp = numer / denom                  # (M,)
        # At exact nodes: pick the node value directly
        node_val = is_node @ values             # (M,) — at most one True per row
    else:
        numer = kernel @ values                 # (M, K)
        denom = kernel.sum(axis=1, keepdims=True)  # (M, 1)
        interp = numer / denom                  # (M, K)
        node_val = is_node.astype(values.dtype) @ values  # (M, K)

    any_exact = is_node.any(axis=1)  # (M,)
    if values.ndim == 1:
        return jnp.where(any_exact, node_val, interp)
    else:
        return jnp.where(any_exact[:, None], node_val, interp)


def interpolate(u_ToA_func, mu_arr_pos):
    """Barycentric interpolation in mu for ToA upwelling intensity.

    Analog of ``PythonicDISORT.subroutines.interpolate``, restricted to
    tau=0 (ToA) and positive mu (upwelling hemisphere). JAX-traceable for
    autodiff through the entire forward model chain.

    Parameters
    ----------
    u_ToA_func : callable
        ``phi -> (N,)`` or ``phi -> (N, len(phi))``, as returned by
        ``pydisort_riccati_jax``.
    mu_arr_pos : (N,) ndarray
        Positive Gauss-Legendre quadrature cosines, as returned by
        ``pydisort_riccati_jax``.

    Returns
    -------
    u_interp : callable
        ``(mu, phi) -> intensity`` where *mu* is a scalar or 1-D array
        of positive cosines in (0, 1] and *phi* is a scalar or 1-D array
        of azimuthal angles.  Return shape follows broadcasting:
        scalar mu & scalar phi -> scalar; array mu & scalar phi -> (M,);
        scalar mu & array phi -> (K,); array mu & array phi -> (M, K).
    """
    bary_weights = jnp.asarray(_compute_bary_weights(np.asarray(mu_arr_pos)))
    mu_nodes = jnp.asarray(mu_arr_pos)

    # If a Nakajima-Tanaka TMS correction is attached (NT_cor=True), interpolate
    # the *smooth* multiple-scattering field and add the TMS single-scattering
    # correction analytically at the requested mu -- far better than
    # interpolating the sharp correction through the barycentric nodes.
    tms_func = getattr(u_ToA_func, "tms", None)
    smooth_func = getattr(u_ToA_func, "u_smooth", u_ToA_func)

    def u_interp(mu, phi):
        mu_q = jnp.atleast_1d(jnp.asarray(mu, dtype=float))
        vals = smooth_func(phi)  # (N,) or (N, K)
        result = _barycentric_interpolate(mu_q, mu_nodes, vals, bary_weights)
        if tms_func is not None:
            corr = tms_func(mu_q, phi)            # (M, P)
            if result.ndim == 1:                  # scalar/1-D phi -> (M,)
                corr = corr[:, 0]
            result = result + corr
        # Squeeze singleton dimensions to match scalar inputs
        if jnp.ndim(mu) == 0 and result.ndim >= 1:
            result = result[0]
        return result

    return u_interp
