import numpy as np
import scipy.linalg

# Threshold below which a singular value is treated as numerically zero,
# signalling entry into the diffusion domain (see _solve_diffusion_domain).
# Diagnostics show the step-by-step SVD begins diverging from the true
# singular values of exp(A*tau) well before this point for non-normal A
# (DISORT's A is always non-normal).  The first sigma to cross this threshold
# typically does so at tau_transition ~ log(1/DIFFUSION_THRESH) / lambda_max.
# For thinner thresholds (e.g. 1e-8) the handoff is earlier and the Magnus
# state at tau_transition is more accurate; the current value matches
# machine epsilon so thin-tau tests (tau <= 2) are entirely unaffected.
DIFFUSION_THRESH = np.finfo(float).eps   # ~ 2.2e-16


def _solve_diffusion_domain(
    A_func,
    S_func,
    tau_bot,
    NQuad,
    tau_transition,
    U_magnus,
    Sigma_magnus,
    Vt_magnus,
    q_scaled_magnus,
):
    """
    [PLACEHOLDER] Solver for the thick / diffusion domain.

    Called by _compute_magnus_propagator when the accumulated propagator's
    smallest singular value drops below DIFFUSION_THRESH, indicating that
    the step-by-step SVD can no longer accurately track the full solution.
    This first occurs at approximately

        tau_transition  ~  log(1 / DIFFUSION_THRESH) / lambda_max(A)

    At tau_transition the Magnus propagator is accurately known:

        Phi_hom(0 → tau_transition) = U_magnus @ diag(Sigma_magnus) @ Vt_magnus
        phi_part(0 → tau_transition) = U_magnus @ diag(Sigma_magnus) @ q_scaled_magnus

    where all entries of Sigma_magnus are > DIFFUSION_THRESH.  The descending
    ordering is: first N entries are growing modes (Sigma > 1), last N are
    decaying modes (Sigma < 1).

    This function must compute the *total* propagator from tau=0 to tau_bot
    by composing the Magnus part with whatever approximation is used for
    tau_transition → tau_bot, and return it in the same format that
    _solve_bc_magnus.py expects from _compute_magnus_propagator.

    Composition identity
    --------------------
    If the diffusion-domain propagator for tau_transition → tau_bot has the
    homogeneous factor  Phi_diff  and particular increment  phi_diff, then:

        Phi_total_hom  = Phi_diff @ Phi_hom_magnus
        phi_total_part = Phi_diff @ phi_part_magnus + phi_diff

    In SVD form this becomes a new SVD factorisation that must be expressed
    in the return format below.

    Parameters
    ----------
    A_func : callable, tau (float) -> (NQuad, NQuad) ndarray
        Coefficient matrix of the ODE, valid for all tau in [0, tau_bot].
    S_func : callable, tau (float) -> (NQuad,) ndarray
        Source vector (zero everywhere if there is no beam source).
    tau_bot : float
        Optical depth of the bottom boundary.
    NQuad : int
        Number of quadrature streams (2N, always even).
    tau_transition : float
        Optical depth at which the Magnus solver hands off.
        The remaining thickness is  tau_bot - tau_transition.
    U_magnus : (NQuad, NQuad) ndarray
        Left singular vectors of the Magnus propagator at tau_transition.
        All NQuad modes are present (no freezing has occurred yet).
    Sigma_magnus : (NQuad,) ndarray
        Singular values of the Magnus propagator at tau_transition.
        Descending order; all entries > DIFFUSION_THRESH.
        First N entries: growing modes (Sigma > 1).
        Last  N entries: decaying modes (0 < Sigma < 1).
    Vt_magnus : (NQuad, NQuad) ndarray
        Right singular vectors of the Magnus propagator at tau_transition.
        Rows ordered descending by Sigma_magnus.  Orthonormal matrix.
    q_scaled_magnus : (NQuad,) ndarray
        Scaled particular-solution coordinates at tau_transition.
        Entry j:  q_scaled_j = (U_magnus.T @ phi_part)[j] / Sigma_magnus[j],
        so that phi_part(tau_transition) = U_magnus @ diag(Sigma_magnus) @ q_scaled_magnus.

    Returns
    -------
    U : (NQuad, r_live) ndarray
        Left singular vectors of the *total* (0 → tau_bot) propagator.
        Growing modes in columns 0 .. N-1, any surviving decaying modes after.
        Must satisfy the NO-POSITIVE-EXPONENTS invariant: no column of
        U @ diag(Sigma) may contain a factor exp(+lambda*tau) with lambda > 0.
    Sigma : (r_live,) ndarray
        Singular values of the total propagator, descending.
        Entries with Sigma ~ 0 should be omitted (just leave them out of
        r_live); they will be treated as frozen with q_scaled = 0 by the
        BC solver.
    Vt : (NQuad, NQuad) ndarray
        All NQuad right-singular-vector rows of the total propagator.
        Row ordering must match _solve_bc_magnus expectations:
          rows 0 .. r_live-1      : live modes (growing then decaying live)
          rows r_live .. NQuad-1  : frozen decaying modes (sigma ~ 0)
        Must be an orthonormal (NQuad, NQuad) matrix.
    q_scaled : (NQuad,) ndarray
        Scaled particular-solution coordinates for the total propagator.
          entries 0 .. r_live-1      : live modes (O(1) values)
          entries r_live .. NQuad-1  : frozen modes (set to 0; cancels in BC)

    Notes
    -----
    - All intermediate quantities must satisfy the NO-POSITIVE-EXPONENTS
      invariant: no factor of the form exp(+lambda*tau) with lambda > 0
      may appear anywhere.  See CLAUDE.md for details.
    - The eigenvalues of A_func(tau) come in ±lambda pairs (DISORT structure).
      lambda_max can be estimated as Sigma_magnus[0] ** (1/tau_transition)
      (approximately, for the dominant mode).
    - A_func and S_func are provided in case the implementation needs to
      evaluate them in the remaining domain (tau_transition, tau_bot).
    """
    raise NotImplementedError(
        "_solve_diffusion_domain: placeholder -- implement the diffusion-domain solver.\n"
        f"  tau_transition = {tau_transition:.6f},  tau_bot = {tau_bot:.6f},  "
        f"tau_remaining = {tau_bot - tau_transition:.6f}\n"
        f"  Sigma_magnus (at transition): {Sigma_magnus}"
    )


def _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, NQuad):
    """
    Accumulates the full-domain Magnus propagator (U, Sigma, Vt, q_scaled) for the ODE

        dI/dtau = A(tau) I + S(tau),    I in R^{NQuad},    tau in [0, tau_bot].

    Uses first-order Magnus (midpoint rule) with step-by-step SVD to maintain
    the NO-POSITIVE-EXPONENTS invariant.

    Thin-atmosphere path (no singular value reaches DIFFUSION_THRESH)
    -----------------------------------------------------------------
    All NQuad modes stay live.  Returns the full SVD of the accumulated
    propagator: U (NQuad, NQuad), Sigma (NQuad,), Vt (NQuad, NQuad),
    q_scaled (NQuad,).

    Thick-atmosphere path (diffusion domain)
    ----------------------------------------
    When the smallest surviving singular value drops below DIFFUSION_THRESH
    the step-by-step SVD can no longer accurately track the full solution
    (non-normal transient growth in DISORT's A makes the step-by-step sigma
    diverge from the true sigma of exp(A*tau) for thick atmospheres).

    At that point the accumulated state at tau_transition is passed to
    _solve_diffusion_domain, which must be implemented by the user.  The
    Magnus loop terminates early and returns whatever _solve_diffusion_domain
    returns.

    NO-POSITIVE-EXPONENTS invariant
    --------------------------------
    No intermediate quantity contains exp(+lambda*tau) with lambda > 0.
    Growing modes appear only via D_gro_inv = 1/sigma_gro < 1 in the BC solver.

    Arguments
    ---------
    A_func   : callable, tau (float) -> (NQuad, NQuad) ndarray
    S_func   : callable, tau (float) -> (NQuad,) ndarray  (zeros if no beam)
    tau_bot  : float > 0, optical depth of the bottom boundary
    N_steps  : int >= 1, number of equidistant Magnus steps
    NQuad    : int, 2N (total number of quadrature streams)

    Returns
    -------
    U        : (NQuad, r_live) ndarray
        Left singular vectors of live modes.  Columns ordered descending by sigma.
        Growing modes occupy the first r_gro = N columns, live decaying the rest.
    Sigma    : (r_live,) ndarray
        Singular values of live modes, descending.  All > DIFFUSION_THRESH.
    Vt       : (NQuad, NQuad) ndarray
        All NQuad right-singular-vector rows.  Row ordering:
          rows 0 .. r_live-1  : live modes (growing then live decaying), desc.
          rows r_live .. NQuad-1 : frozen decaying modes (sigma ~ 0).
        Decaying subspace = rows N .. NQuad-1 (N rows total).
    q_scaled : (NQuad,) ndarray
        Scaled particular-solution coordinates.
          entries 0 .. r_live-1  : live modes (O(1))
          entries r_live .. NQuad-1 : frozen modes (0; cancels algebraically in BC)
    """
    h = tau_bot / N_steps

    # Initial state: Phi_hom = I_{NQuad}
    U_live        = np.eye(NQuad)       # (NQuad, NQuad) -> may shrink on diffusion handoff
    Sigma_live    = np.ones(NQuad)      # (NQuad,); descending; all = 1 at tau=0
    Vt_live       = np.eye(NQuad)       # (NQuad, NQuad)
    q_scaled_live = np.zeros(NQuad)     # (NQuad,); all zero at tau=0

    ext   = NQuad + 1
    M_ext = np.zeros((ext, ext))  # reused each step; last row stays zero

    for k in range(N_steps):
        tau_mid = (k + 0.5) * h

        A_k = A_func(tau_mid)   # (NQuad, NQuad)
        S_k = S_func(tau_mid)   # (NQuad,)

        M_ext[:NQuad, :NQuad] = h * A_k
        M_ext[:NQuad, NQuad]  = h * S_k
        # M_ext[NQuad, :] remains all zeros throughout

        Phi_ext  = scipy.linalg.expm(M_ext)        # (NQuad+1, NQuad+1)
        Phi_step = Phi_ext[:NQuad, :NQuad]          # homogeneous step propagator
        delta_p  = Phi_ext[:NQuad, NQuad]           # particular-solution step increment

        # ---------------------------------------------------------------
        # Update live SVD (no positive exponents):
        #   Phi_hom_new = Phi_step @ U_live @ diag(Sigma_live) @ Vt_live
        #   G = (Phi_step @ U_live) @ diag(Sigma_live)
        #   G = U_new @ diag(Sigma_new) @ Wt   (thin SVD)
        # => Phi_hom_new = U_new @ diag(Sigma_new) @ (Wt @ Vt_live)
        # ---------------------------------------------------------------
        B = Phi_step @ U_live                            # (NQuad, r_live)
        G = B * Sigma_live[None, :]                      # (NQuad, r_live)
        U_new, Sigma_new, Wt = np.linalg.svd(G, full_matrices=False)
        Vt_live_new    = Wt @ Vt_live                    # (r_live, NQuad)

        # ---------------------------------------------------------------
        # Update q_scaled for live modes (always O(1)):
        #   q_scaled_new = Wt @ q_scaled + U_new.T @ delta_p / Sigma_new
        # ---------------------------------------------------------------
        q_scaled_new = Wt @ q_scaled_live + U_new.T @ delta_p / Sigma_new

        # ---------------------------------------------------------------
        # Check for diffusion-domain entry.
        #
        # When the smallest singular value drops below DIFFUSION_THRESH the
        # step-by-step SVD diverges from the true singular values of
        # exp(A*tau) (non-normal transient growth).  Hand off to the
        # diffusion-domain solver rather than continuing to accumulate a
        # corrupted propagator.
        #
        # tau_transition is the end of the current step (k+1 steps of h).
        # The state (U_new, Sigma_new, Vt_live_new, q_scaled_new) is the
        # last accurately-known Magnus propagator.
        # ---------------------------------------------------------------
        if Sigma_new[-1] < DIFFUSION_THRESH:
            tau_transition = (k + 1) * h
            return _solve_diffusion_domain(
                A_func, S_func, tau_bot, NQuad,
                tau_transition,
                U_new, Sigma_new, Vt_live_new, q_scaled_new,
            )

        U_live        = U_new
        Sigma_live    = Sigma_new
        Vt_live       = Vt_live_new
        q_scaled_live = q_scaled_new

    # -----------------------------------------------------------------------
    # Thin-atmosphere exit: no diffusion handoff triggered.
    # All NQuad modes are live; Vt_live is already (NQuad, NQuad).
    # -----------------------------------------------------------------------
    return U_live, Sigma_live, Vt_live, q_scaled_live
