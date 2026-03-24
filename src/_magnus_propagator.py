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


_GL_C1 = 0.5 - np.sqrt(3.0) / 6.0   # ≈ 0.2113
_GL_C2 = 0.5 + np.sqrt(3.0) / 6.0   # ≈ 0.7887
_COMM_COEFF = np.sqrt(3.0) / 12.0    # ≈ 0.1443


def _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, NQuad):
    """
    Accumulates the full-domain scattering operators via Redheffer star product.

    Uses 4th-order Magnus integration (2-point Gauss-Legendre quadrature with
    commutator correction) for each step, then combines steps via the star
    product of N×N reflection/transmission/source operators.

    Per step [τ_k, τ_k + h]:
        Ω_A = (h/2)(A₁ + A₂) + (√3/12)h²[A₂, A₁]
        Ω_S = (h/2)(S₁ + S₂) + (√3/12)h²(A₂S₁ − A₁S₂)
    where A₁, A₂ are evaluated at the two GL nodes within the step.

    For constant A: [A, A] = 0, so this reduces to midpoint (exact for constant).
    For τ-varying A: achieves O(h⁴) convergence.

    All intermediates are O(1) — unconditionally stable for any tau_bot.

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
    h2_comm = _COMM_COEFF * h * h

    for k in range(N_steps):
        tau_k = k * h

        # Evaluate A and S at the two Gauss-Legendre nodes
        A1 = A_func(tau_k + _GL_C1 * h)
        A2 = A_func(tau_k + _GL_C2 * h)
        S1 = S_func(tau_k + _GL_C1 * h)
        S2 = S_func(tau_k + _GL_C2 * h)

        # 4th-order Magnus exponent: Ω = (h/2)(A₁+A₂) + (√3/12)h²[A₂, A₁]
        Omega_A = 0.5 * h * (A1 + A2)
        comm = A2 @ A1 - A1 @ A2       # [A₂, A₁]
        Omega_A += h2_comm * comm

        # Source correction: (h/2)(S₁+S₂) + (√3/12)h²(A₂S₁ − A₁S₂)
        Omega_S = 0.5 * h * (S1 + S2) + h2_comm * (A2 @ S1 - A1 @ S2)

        M_ext[:NQuad, :NQuad] = Omega_A
        M_ext[:NQuad, NQuad]  = Omega_S
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


_C_STAB = 1.5  # stability ceiling: max h·λ_max before cond(a) degrades


def _compute_magnus_propagator_adaptive(A_func, S_func, tau_bot, NQuad, tol):
    """
    Adaptive 4th-order Magnus propagator with star-product accumulation.

    Uses the commutator norm ‖[A₂, A₁]‖_F · h² · (√3/12) as a cheap error
    indicator (no second expm).  For constant A the commutator vanishes exactly,
    so the step size is limited only by the stability ceiling h ≤ c_stab/λ_max.

    Arguments
    ---------
    A_func   : callable, tau -> (NQuad, NQuad) ndarray
    S_func   : callable, tau -> (NQuad,) ndarray
    tau_bot  : float > 0
    NQuad    : int, 2N
    tol      : float > 0, relative error tolerance for the commutator indicator

    Returns
    -------
    R_up, T_up, T_down, R_down : (N, N) ndarrays
    s_up, s_down : (N,) ndarrays
    tau_grid : (K+1,) ndarray — step boundary points [0, τ₁, …, τ_bot]
    """
    N = NQuad // 2

    # Estimate λ_max from A at midpoint (one eigenvalue computation)
    A_mid = A_func(tau_bot / 2)
    lambda_max = float(np.max(np.abs(np.linalg.eigvals(A_mid).real)))
    lambda_max = max(lambda_max, 1e-10)

    h_max = _C_STAB / lambda_max  # stability ceiling
    h = min(tau_bot, h_max)

    # Initialize identity slab
    R_up   = np.zeros((N, N))
    T_up   = np.eye(N)
    T_down = np.eye(N)
    R_down = np.zeros((N, N))
    s_up   = np.zeros(N)
    s_down = np.zeros(N)

    tau_grid = [0.0]
    tau_current = 0.0
    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))

    while tau_current < tau_bot - 1e-14:
        # Clamp to boundary
        final_step = (tau_current + h >= tau_bot - 1e-14)
        if final_step:
            h = tau_bot - tau_current

        # Evaluate at the two Gauss-Legendre nodes
        A1 = A_func(tau_current + _GL_C1 * h)
        A2 = A_func(tau_current + _GL_C2 * h)

        # Commutator and error indicator
        comm = A2 @ A1 - A1 @ A2
        h2_comm = _COMM_COEFF * h * h
        comm_norm = np.linalg.norm(comm, 'fro') * h2_comm
        err_rel = comm_norm / max(1.0, h * lambda_max)

        if err_rel > tol and not final_step:
            # Reject step, shrink
            factor = max(0.2, 0.9 * (tol / err_rel) ** 0.25)
            h = min(h * factor, h_max)
            continue

        # Accept step — evaluate source at GL nodes
        S1 = S_func(tau_current + _GL_C1 * h)
        S2 = S_func(tau_current + _GL_C2 * h)

        # 4th-order Magnus exponent
        Omega_A = 0.5 * h * (A1 + A2) + h2_comm * comm
        Omega_S = 0.5 * h * (S1 + S2) + h2_comm * (A2 @ S1 - A1 @ S2)

        M_ext[:NQuad, :NQuad] = Omega_A
        M_ext[:NQuad, NQuad]  = Omega_S

        Phi_ext  = scipy.linalg.expm(M_ext)
        Phi_step = Phi_ext[:NQuad, :NQuad]
        delta_p  = Phi_ext[:NQuad, NQuad]

        r_up_k, t_up_k, t_down_k, r_down_k, s_up_k, s_down_k = \
            _extract_slab_operators(Phi_step, delta_p, N)

        R_up, T_up, T_down, R_down, s_up, s_down = _star_product(
            R_up, T_up, T_down, R_down, s_up, s_down,
            r_up_k, t_up_k, t_down_k, r_down_k, s_up_k, s_down_k, N,
        )

        tau_current += h
        tau_grid.append(tau_current)

        # Predict next step size
        if err_rel > 0:
            factor = min(2.0, max(0.2, 0.9 * (tol / err_rel) ** 0.25))
        else:
            factor = 2.0
        h = min(h * factor, h_max)

    return R_up, T_up, T_down, R_down, s_up, s_down, np.array(tau_grid)
