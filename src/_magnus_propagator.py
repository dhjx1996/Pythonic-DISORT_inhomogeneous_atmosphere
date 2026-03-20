import numpy as np
import scipy.linalg


def _extract_slab_operators(Phi_step, delta_p, N):
    """
    Extract the 6 N×N scattering-matrix operators from one Magnus step.

    From the 2N×2N propagator Phi_step = [[a,b],[c,d]] and particular increment
    delta_p = [δ⁺, δ⁻], computes:

        t_up = a⁻¹,            r_up = −a⁻¹b,          s_up = −a⁻¹δ⁺
        r_down = ca⁻¹,         t_down = d − ca⁻¹b,    s_down = δ⁻ − ca⁻¹δ⁺

    Uses a single batched solve a @ X = [I_N | b | δ⁺] for efficiency.
    cond(a) ≈ 1 + O(h·λ_max), so this is numerically stable for small steps.
    """
    a = Phi_step[:N, :N]
    b = Phi_step[:N, N:]
    c = Phi_step[N:, :N]
    d = Phi_step[N:, N:]
    dp = delta_p[:N]
    dm = delta_p[N:]

    # Single (N, 2N+1) solve: a @ X = [I_N, b, dp]
    RHS = np.empty((N, 2 * N + 1))
    RHS[:, :N] = np.eye(N)
    RHS[:, N:2*N] = b
    RHS[:, 2*N] = dp
    X = np.linalg.solve(a, RHS)

    t_up     = X[:, :N]           # a⁻¹
    a_inv_b  = X[:, N:2*N]        # a⁻¹b
    a_inv_dp = X[:, 2*N]          # a⁻¹δ⁺

    r_up   = -a_inv_b
    s_up   = -a_inv_dp
    r_down = c @ t_up              # ca⁻¹
    t_down = d - c @ a_inv_b       # d − ca⁻¹b
    s_down = dm - c @ a_inv_dp     # δ⁻ − ca⁻¹δ⁺

    return r_up, t_up, t_down, r_down, s_up, s_down


def _star_product(R_up, T_up, T_down, R_down, s_up, s_down,
                  r_up_k, t_up_k, t_down_k, r_down_k, s_up_k, s_down_k, N):
    """
    Redheffer star product: combine accumulated slab (top) with step slab (bottom).

    All intermediates are O(1) — unconditionally stable for any optical depth.
    Uses a single batched solve for the resolvent E = (I − r_up_k @ R_down)⁻¹.
    """
    LHS = np.eye(N) - r_up_k @ R_down

    # Precompute RHS columns: [r_up_k @ T_down | t_up_k | r_up_k @ s_down + s_up_k]
    temp_s = r_up_k @ s_down + s_up_k
    RHS = np.empty((N, 2 * N + 1))
    RHS[:, :N]    = r_up_k @ T_down
    RHS[:, N:2*N] = t_up_k
    RHS[:, 2*N]   = temp_s
    E_rhs = np.linalg.solve(LHS, RHS)

    E_rT = E_rhs[:, :N]          # E @ r_up_k @ T_down
    E_t  = E_rhs[:, N:2*N]       # E @ t_up_k
    E_s  = E_rhs[:, 2*N]         # E @ (r_up_k @ s_down + s_up_k)

    R_up_new   = R_up + T_up @ E_rT
    T_up_new   = T_up @ E_t
    T_down_new = t_down_k @ (T_down + R_down @ E_rT)
    R_down_new = r_down_k + t_down_k @ R_down @ E_t
    s_up_new   = s_up + T_up @ E_s
    s_down_new = s_down_k + t_down_k @ (s_down + R_down @ E_s)

    return R_up_new, T_up_new, T_down_new, R_down_new, s_up_new, s_down_new


def _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, NQuad):
    """
    Accumulates the full-domain scattering operators via Redheffer star product.

    Uses first-order Magnus (midpoint rule) for each step, then combines steps
    via the star product of N×N reflection/transmission/source operators.
    All intermediates are O(1) — unconditionally stable for any tau_bot.
    Accuracy controlled by N_steps; need h · λ_max ≲ 1.

    Arguments
    ---------
    A_func   : callable, tau (float) -> (NQuad, NQuad) ndarray
    S_func   : callable, tau (float) -> (NQuad,) ndarray  (zeros if no beam)
    tau_bot  : float > 0, optical depth of the bottom boundary
    N_steps  : int >= 1, number of equidistant Magnus steps
    NQuad    : int, 2N (total number of quadrature streams)

    Returns
    -------
    R_up, T_up, T_down, R_down : (N, N) ndarrays
        Reflection and transmission operators for the full slab.
    s_up, s_down : (N,) ndarrays
        Source vectors for the full slab.
    """
    N = NQuad // 2
    h = tau_bot / N_steps

    # Initialize identity slab (no scattering, no source)
    R_up   = np.zeros((N, N))
    T_up   = np.eye(N)
    T_down = np.eye(N)
    R_down = np.zeros((N, N))
    s_up   = np.zeros(N)
    s_down = np.zeros(N)

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

        # Extract per-step scattering operators
        r_up_k, t_up_k, t_down_k, r_down_k, s_up_k, s_down_k = \
            _extract_slab_operators(Phi_step, delta_p, N)

        # Combine via Redheffer star product
        R_up, T_up, T_down, R_down, s_up, s_down = _star_product(
            R_up, T_up, T_down, R_down, s_up, s_down,
            r_up_k, t_up_k, t_down_k, r_down_k, s_up_k, s_down_k, N,
        )

    return R_up, T_up, T_down, R_down, s_up, s_down
