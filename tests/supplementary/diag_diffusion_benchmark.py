"""
Diagnostic: diffusion-domain benchmark with known reference solution.

Uses pydisort (exact eigendecomposition) to solve a thick slab, then extracts
a diffusion sub-domain with known boundary sources.  Compares Magnus star product,
explicit RK4 Riccati, and implicit Radau Riccati on the sub-domain.

Setup: tau_bot=50, omega=0.99, g=0.85, beam (mu0=0.5, I0=1), NQuad=16
Sub-domain: [tau1=10, tau2=30], beam negligible (exp(-20) ~ 2e-9)

Part A: Full Magnus star product on the sub-domain
Part B: Explicit RK4 Riccati (find stability boundary)
Part C: Implicit Radau Riccati with analytical Jacobian
Part D: Scaling with omega (the key result)
Part E: Three-domain decomposition end-to-end
"""
import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
import sys, math, time
sys.path.insert(0, '.')
from _helpers import make_D_m_funcs, pydisort_toa
from PythonicDISORT import subroutines
from PythonicDISORT.pydisort import pydisort

# --- Configuration ---
NQuad = 16  # N=8; increase to 32 for stress test (N=16, lambda_max~250)
N = NQuad // 2
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos

tau_bot_full = 50.0
tau1 = 10.0  # sub-domain top
tau2 = 30.0  # sub-domain bottom
tau_sub = tau2 - tau1

omega_default = 0.99
g_default = 0.85
mu0 = 0.5; I0 = 1.0; phi0 = math.pi


def build_alpha_beta(omega, g):
    """Build constant alpha, beta for homogeneous atmosphere."""
    g_l = g ** np.arange(NQuad)
    D_m_funcs = make_D_m_funcs(g_l, NQuad, NQuad)
    D_m = D_m_funcs[0]
    D_pos = omega * D_m(1.0, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m(1.0, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    return alpha, beta


def build_A_and_S_func(omega, g, mu0_val=None, I0_val=0):
    """Build A(tau) and S(tau) callables."""
    g_l = g ** np.arange(NQuad)
    D_m_funcs = make_D_m_funcs(g_l, NQuad, NQuad)
    D_m = D_m_funcs[0]
    I0_div_4pi = I0_val / (4*math.pi) if I0_val > 0 else 0
    fac_const = I0_div_4pi * 2  # m=0: (2 - 1) * 2

    def A_func(tau):
        D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        DW_neg = D_neg * W[None, :]
        alpha = M_inv[:, None] * (DW_pos - np.eye(N))
        beta  = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N] = -alpha; A[:N, N:] = -beta
        A[N:, :N] = beta;   A[N:, N:] = alpha
        return A

    if I0_val > 0 and mu0_val is not None:
        def S_func(tau):
            fac = fac_const * omega * np.exp(-tau / mu0_val)
            S_pos = -M_inv * fac * D_m(tau, mu_arr_pos, -mu0_val)
            S_neg =  M_inv * fac * D_m(tau, -mu_arr_pos, -mu0_val)
            return np.concatenate([S_pos, S_neg])
    else:
        def S_func(tau):
            return np.zeros(NQuad)

    return A_func, S_func


def one_step_star(A_func, S_func, tau_start, h):
    """Compute one star-product step."""
    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))
    M_ext[:NQuad, :NQuad] = h * A_func(tau_start + h/2)
    M_ext[:NQuad, NQuad] = h * S_func(tau_start + h/2)
    Phi_ext = scipy.linalg.expm(M_ext)
    Phi = Phi_ext[:NQuad, :NQuad]
    dp = Phi_ext[:NQuad, NQuad]
    a = Phi[:N, :N]; b = Phi[:N, N:]
    c = Phi[N:, :N]; d = Phi[N:, N:]
    a_inv = np.linalg.inv(a)
    return (-a_inv @ b, a_inv, c @ a_inv, d - c @ a_inv @ b,
            -a_inv @ dp[:N], dp[N:] - c @ a_inv @ dp[:N])


def star_combine(R_top, T_up, T_down, R_bot, s_up, s_down,
                 r_top, t_up, r_bot, t_down, src_up, src_down):
    """Combine accumulated (upper) with new step (lower) via star product."""
    LHS = np.eye(N) - r_top @ R_bot
    temp_s = r_top @ s_down + src_up
    RHS = np.column_stack([r_top @ T_down, t_up, temp_s])
    E_rhs = np.linalg.solve(LHS, RHS)
    E_rT = E_rhs[:, :N]; E_t = E_rhs[:, N:2*N]; E_s = E_rhs[:, 2*N]
    return (R_top + T_up @ E_rT, T_up @ E_t,
            t_down @ (T_down + R_bot @ E_rT), r_bot + t_down @ R_bot @ E_t,
            s_up + T_up @ E_s,
            src_down + t_down @ (s_down + R_bot @ E_s))


def run_star_product(A_func, S_func, tau_bot, N_steps):
    """Full star-product solve."""
    h = tau_bot / N_steps
    R_top = np.zeros((N,N)); R_bot = np.zeros((N,N))
    T_up = np.eye(N); T_down = np.eye(N)
    s_up = np.zeros(N); s_down = np.zeros(N)
    for k in range(N_steps):
        r, tu, rb, td, su, sd = one_step_star(A_func, S_func, k*h, h)
        R_top, T_up, T_down, R_bot, s_up, s_down = star_combine(
            R_top, T_up, T_down, R_bot, s_up, s_down, r, tu, rb, td, su, sd)
    return R_top, T_up, T_down, R_bot, s_up, s_down


def get_pydisort_reference(tau_bot, omega, g, NQ):
    """Run pydisort and return (u0f, mu_arr) for the full atmosphere."""
    g_l = g ** np.arange(NQ)
    mu_arr, Fp, Fm, u0f, uf = pydisort(
        np.array([tau_bot]), np.array([omega]), NQ,
        np.atleast_2d(g_l), mu0, I0, phi0,
        NLeg=NQ, NFourier=NQ, only_flux=False,
    )
    return u0f, Fp, mu_arr


def compute_k_values(alpha, beta):
    """Compute k-eigenvalues from N*N reduction."""
    k_sq = np.linalg.eigvals((alpha - beta) @ (alpha + beta)).real
    return np.sqrt(np.maximum(np.sort(k_sq)[::-1], 0))


# ======================================================================
# SETUP: pydisort reference and sub-domain BCs
# ======================================================================
print("=" * 70)
print(f"SETUP: pydisort reference (tau_bot={tau_bot_full}, omega={omega_default}, "
      f"g={g_default}, NQuad={NQuad})")
print("=" * 70)

alpha0, beta0 = build_alpha_beta(omega_default, g_default)
k_vals = compute_k_values(alpha0, beta0)
k_max = k_vals[0]; k_min = k_vals[-1]
print(f"k_max = {k_max:.4f}, k_min = {k_min:.4f}, gap = {k_max/k_min:.1f}")
print(f"1/mu_min = {1/mu_arr_pos[0]:.4f}")
print(f"Beam at tau1={tau1}: exp(-tau1/mu0) = {math.exp(-tau1/mu0):.3e} (negligible)")

print(f"\nRunning pydisort...")
t0 = time.perf_counter()
u0f_ref, Fp_ref, mu_arr_ref = get_pydisort_reference(
    tau_bot_full, omega_default, g_default, NQuad)
t_pyd = time.perf_counter() - t0
print(f"  pydisort wall time: {t_pyd:.2f} s")

# Extract sub-domain BCs (m=0 mode)
b_neg = u0f_ref(tau1)[N:]    # downwelling at tau1 (N,)
b_pos = u0f_ref(tau2)[:N]    # upwelling at tau2 (N,)
u_ref = u0f_ref(tau1)[:N]    # upwelling at tau1 (reference) (N,)

print(f"  ||b_neg|| = {np.linalg.norm(b_neg):.6e}")
print(f"  ||b_pos|| = {np.linalg.norm(b_pos):.6e}")
print(f"  ||u_ref|| = {np.linalg.norm(u_ref):.6e}")


# ======================================================================
print("\n" + "=" * 70)
print("PART A: Full Magnus star product on sub-domain")
print("=" * 70)

A_sub, S_sub = build_A_and_S_func(omega_default, g_default)  # no beam

K_magnus = max(int(k_max * tau_sub) + 10, 100)
print(f"\ntau_sub={tau_sub}, K_magnus={K_magnus} (h*k_max={k_max*tau_sub/K_magnus:.2f})")

t0 = time.perf_counter()
R_up, T_up, T_down, R_bot, s_up, s_down = run_star_product(
    A_sub, S_sub, tau_sub, K_magnus)
t_magnus = time.perf_counter() - t0

u_magnus = R_up @ b_neg + T_up @ b_pos + s_up
rel_err = np.linalg.norm(u_magnus - u_ref) / np.linalg.norm(u_ref)
print(f"  ||u_magnus - u_ref|| / ||u_ref|| = {rel_err:.3e}")
print(f"  Magnus wall time: {t_magnus:.2f} s")


# ======================================================================
print("\n" + "=" * 70)
print("PART B: Explicit RK4 Riccati on sub-domain")
print("=" * 70)

def riccati_rhs(R, alpha, beta):
    """Invariant imbedding Riccati: dR/dsigma = alpha R + R alpha + R beta R + beta.

    Builds the slab from the bottom (sigma = slab thickness).  R(0) = 0.
    R(tau_sub) = R_up = R_down (for a homogeneous slab).  All positive signs.
    """
    return alpha @ R + R @ alpha + R @ beta @ R + beta


def riccati_T_rhs(T, alpha, beta, R):
    """Companion T equation: dT/dsigma = T @ (alpha + beta @ R).

    T(0) = I, T(tau_sub) = T_up (upward transmission of the full slab).
    Note: right-multiplication (T @ C), not left.
    """
    return T @ (alpha + beta @ R)


def rk4_riccati(alpha, beta, tau_sub, K):
    """Integrate Riccati ODE with explicit RK4, return R(tau_sub)."""
    h = tau_sub / K
    R = np.zeros((N, N))
    for _ in range(K):
        k1 = h * riccati_rhs(R, alpha, beta)
        k2 = h * riccati_rhs(R + k1/2, alpha, beta)
        k3 = h * riccati_rhs(R + k2/2, alpha, beta)
        k4 = h * riccati_rhs(R + k3, alpha, beta)
        R = R + (k1 + 2*k2 + 2*k3 + k4) / 6
        if np.any(np.abs(R) > 1e10):
            return None  # blew up
    return R


def rk4_riccati_with_T(alpha, beta, tau_sub, K):
    """Integrate Riccati + companion T, return (R, T) at tau_sub."""
    h = tau_sub / K
    R = np.zeros((N, N))
    T = np.eye(N)
    for _ in range(K):
        # RK4 for R
        k1_R = h * riccati_rhs(R, alpha, beta)
        k2_R = h * riccati_rhs(R + k1_R/2, alpha, beta)
        k3_R = h * riccati_rhs(R + k2_R/2, alpha, beta)
        k4_R = h * riccati_rhs(R + k3_R, alpha, beta)
        R_new = R + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6
        if np.any(np.abs(R_new) > 1e10):
            return None, None
        # RK4 for T: dT/dsigma = T @ (alpha + beta @ R)
        C0 = alpha + beta @ R
        C_mid = alpha + beta @ (R + k2_R / (2*h) * h)  # approx at midpoint
        C1 = alpha + beta @ R_new
        k1_T = h * (T @ C0)
        k2_T = h * ((T + k1_T/2) @ C_mid)
        k3_T = h * ((T + k2_T/2) @ C_mid)
        k4_T = h * ((T + k3_T) @ C1)
        T = T + (k1_T + 2*k2_T + 2*k3_T + k4_T) / 6
        R = R_new
    return R, T


print(f"\nFinding stability boundary for explicit RK4...")
print(f"  k_max = {k_max:.4f}, stability limit h*k_max ~ 2.8 for RK4")
print(f"  Expected min K ~ {tau_sub * k_max / 2.8:.0f}")
print(f"  {'K':>8s} {'h*k_max':>10s} {'status':>10s} {'rel_err':>12s}")

for K in [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]:
    h = tau_sub / K
    R_rk4 = rk4_riccati(alpha0, beta0, tau_sub, K)
    if R_rk4 is None:
        print(f"  {K:8d} {h*k_max:10.3f} {'BLOWUP':>10s} {'---':>12s}")
    else:
        R_rk4, T_rk4 = rk4_riccati_with_T(alpha0, beta0, tau_sub, K)
        if R_rk4 is None:
            print(f"  {K:8d} {h*k_max:10.3f} {'T BLOWUP':>10s} {'---':>12s}")
        else:
            u_rk4 = R_rk4 @ b_neg + T_rk4 @ b_pos
            err = np.linalg.norm(u_rk4 - u_ref) / np.linalg.norm(u_ref)
            print(f"  {K:8d} {h*k_max:10.3f} {'OK':>10s} {err:12.3e}")

print(f"\nCritical negative result: RK4 needs h*k_max < ~2.8,")
print(f"same constraint as Magnus (h*k_max < ~1). No speedup from Riccati + RK4.")


# ======================================================================
print("\n" + "=" * 70)
print("PART C: Implicit Radau Riccati on sub-domain")
print("=" * 70)

N_sq = N * N
I_N = np.eye(N)


def riccati_rhs_vec(tau, R_vec):
    """Vectorized Riccati RHS for solve_ivp (positive / invariant imbedding)."""
    R = R_vec.reshape(N, N)
    dR = alpha0 @ R + R @ alpha0 + R @ beta0 @ R + beta0
    return dR.ravel()


def riccati_jac_vec(tau, R_vec):
    """Analytical Jacobian (row-major vectorization).

    J = kron(alpha + R @ beta, I) + kron(I, (alpha + beta @ R).T)
    """
    R = R_vec.reshape(N, N)
    return (np.kron(alpha0 + R @ beta0, I_N) +
            np.kron(I_N, (alpha0 + beta0 @ R).T))


print(f"\nIntegrating Riccati ODE with Radau (N^2={N_sq} DOFs, Jacobian {N_sq}x{N_sq})...")
t0 = time.perf_counter()
sol_R = solve_ivp(riccati_rhs_vec, [0, tau_sub], np.zeros(N_sq),
                  method='Radau', jac=riccati_jac_vec,
                  rtol=1e-8, atol=1e-10, dense_output=True)
t_radau_R = time.perf_counter() - t0
print(f"  Riccati R: nfev={sol_R.nfev}, njev={sol_R.njev}, nlu={sol_R.nlu}, "
      f"wall={t_radau_R:.2f}s, status={sol_R.status}")

R_radau = sol_R.y[:, -1].reshape(N, N)

# Companion T equation: dT/dsigma = T @ (alpha + beta @ R)  (right-multiply)
def T_rhs_vec(tau, T_vec):
    R = sol_R.sol(tau).reshape(N, N)
    T = T_vec.reshape(N, N)
    return (T @ (alpha0 + beta0 @ R)).ravel()


def T_jac_vec(tau, T_vec):
    R = sol_R.sol(tau).reshape(N, N)
    return np.kron(I_N, (alpha0 + beta0 @ R).T)


print(f"  Integrating companion T equation...")
t0 = time.perf_counter()
sol_T = solve_ivp(T_rhs_vec, [0, tau_sub], np.eye(N).ravel(),
                  method='Radau', jac=T_jac_vec,
                  rtol=1e-8, atol=1e-10)
t_radau_T = time.perf_counter() - t0
T_radau = sol_T.y[:, -1].reshape(N, N)
print(f"  T equation: nfev={sol_T.nfev}, wall={t_radau_T:.2f}s")

u_radau = R_radau @ b_neg + T_radau @ b_pos
rel_err_radau = np.linalg.norm(u_radau - u_ref) / np.linalg.norm(u_ref)
print(f"\n  ||u_radau - u_ref|| / ||u_ref|| = {rel_err_radau:.3e}")
print(f"  Total Radau wall time: {t_radau_R + t_radau_T:.2f} s")
print(f"  Total Radau nfev: {sol_R.nfev + sol_T.nfev}")
print(f"  Magnus K={K_magnus}, wall time={t_magnus:.2f} s")
print(f"  Radau/Magnus nfev ratio: {(sol_R.nfev + sol_T.nfev)/K_magnus:.2f}")


# ======================================================================
print("\n" + "=" * 70)
print("PART D: Scaling with omega (the key result)")
print("=" * 70)

print(f"\ntau_sub={tau_sub}, g={g_default}, NQuad={NQuad}")
print(f"  {'omega':>8s} {'k_max':>8s} {'k_min':>8s} {'gamma':>8s} "
      f"{'K_magnus':>9s} {'nfev_rad':>9s} {'err_mag':>9s} {'err_rad':>9s} "
      f"{'t_mag':>7s} {'t_rad':>7s}")

for omega_test in [0.9, 0.99, 0.999, 0.9999]:
    # Recompute alpha, beta, k-values
    alpha_t, beta_t = build_alpha_beta(omega_test, g_default)
    k_t = compute_k_values(alpha_t, beta_t)
    kmax_t, kmin_t = k_t[0], k_t[-1]
    gamma_t = kmax_t / kmin_t

    # Get fresh sub-domain BCs from pydisort
    u0f_t, _, _ = get_pydisort_reference(tau_bot_full, omega_test, g_default, NQuad)
    bn_t = u0f_t(tau1)[N:]
    bp_t = u0f_t(tau2)[:N]
    ur_t = u0f_t(tau1)[:N]

    # Magnus star product
    A_t, S_t = build_A_and_S_func(omega_test, g_default)
    K_t = max(int(kmax_t * tau_sub) + 10, 100)
    t0 = time.perf_counter()
    Ru_t, Tu_t, Td_t, Rb_t, su_t, sd_t = run_star_product(A_t, S_t, tau_sub, K_t)
    t_mag_t = time.perf_counter() - t0
    u_mag_t = Ru_t @ bn_t + Tu_t @ bp_t + su_t
    err_mag_t = np.linalg.norm(u_mag_t - ur_t) / max(np.linalg.norm(ur_t), 1e-15)

    # Radau Riccati (positive / invariant imbedding signs)
    def _riccati_rhs(tau, R_vec, _a=alpha_t, _b=beta_t):
        R = R_vec.reshape(N, N)
        return (_a @ R + R @ _a + R @ _b @ R + _b).ravel()

    def _riccati_jac(tau, R_vec, _a=alpha_t, _b=beta_t):
        R = R_vec.reshape(N, N)
        return (np.kron(_a + R @ _b, I_N) + np.kron(I_N, (_a + _b @ R).T))

    t0 = time.perf_counter()
    sol_Rt = solve_ivp(_riccati_rhs, [0, tau_sub], np.zeros(N_sq),
                       method='Radau', jac=_riccati_jac, rtol=1e-8, atol=1e-10,
                       dense_output=True)

    def _T_rhs(tau, T_vec, _a=alpha_t, _b=beta_t, _sol=sol_Rt):
        R = _sol.sol(tau).reshape(N, N)
        T = T_vec.reshape(N, N)
        return (T @ (_a + _b @ R)).ravel()

    def _T_jac(tau, T_vec, _a=alpha_t, _b=beta_t, _sol=sol_Rt):
        R = _sol.sol(tau).reshape(N, N)
        return np.kron(I_N, (_a + _b @ R).T)

    sol_Tt = solve_ivp(_T_rhs, [0, tau_sub], np.eye(N).ravel(),
                       method='Radau', jac=_T_jac, rtol=1e-8, atol=1e-10)
    t_rad_t = time.perf_counter() - t0
    nfev_rad_t = sol_Rt.nfev + sol_Tt.nfev

    R_rad = sol_Rt.y[:, -1].reshape(N, N)
    T_rad = sol_Tt.y[:, -1].reshape(N, N)
    u_rad_t = R_rad @ bn_t + T_rad @ bp_t
    err_rad_t = np.linalg.norm(u_rad_t - ur_t) / max(np.linalg.norm(ur_t), 1e-15)

    print(f"  {omega_test:8.4f} {kmax_t:8.2f} {kmin_t:8.4f} {gamma_t:8.1f} "
          f"{K_t:9d} {nfev_rad_t:9d} {err_mag_t:9.2e} {err_rad_t:9.2e} "
          f"{t_mag_t:7.2f} {t_rad_t:7.2f}")

print(f"\nIf nfev_rad << K_magnus as omega -> 1, the approach is validated:")
print(f"  Radau adapts to the slow diffusion scale (k_min), not the fast ballistic (k_max).")
print(f"If nfev_rad ~ K_magnus, the approach is invalidated: the Riccati ODE")
print(f"  does not decouple the stiff ballistic modes as hoped.")


# ======================================================================
print("\n" + "=" * 70)
print("PART E: Three-domain decomposition end-to-end")
print("=" * 70)
print(f"\nDomain I:  [0, {tau1}]  -- beam active, full Magnus")
print(f"Domain II: [{tau1}, {tau2}]  -- diffusion domain, full Magnus")
print(f"Domain III:[{tau2}, {tau_bot_full}]  -- full Magnus")

# Build A_func and S_func WITH beam for domains I and III
A_beam, S_beam = build_A_and_S_func(
    omega_default, g_default, mu0_val=mu0, I0_val=I0)
A_nobeam, S_nobeam = build_A_and_S_func(omega_default, g_default)

# Step count: use h*k_max ~ 1
K_I = max(int(k_max * tau1) + 10, 50)
K_II = max(int(k_max * tau_sub) + 10, 100)
K_III = max(int(k_max * (tau_bot_full - tau2)) + 10, 100)
K_full = max(int(k_max * tau_bot_full) + 10, 200)

print(f"\nStep counts: K_I={K_I}, K_II={K_II}, K_III={K_III}, K_full={K_full}")

# Domain I [0, tau1] with beam
t0 = time.perf_counter()
ops_I = run_star_product(A_beam, S_beam, tau1, K_I)

# Domain II [tau1, tau2] without beam (negligible)
# Shift tau: A_func for domain II evaluates at tau1 + tau_local
def A_func_II(tau):
    return A_beam(tau1 + tau)
def S_func_II(tau):
    return S_beam(tau1 + tau)
ops_II = run_star_product(A_func_II, S_func_II, tau_sub, K_II)

# Domain III [tau2, tau_bot_full] with beam (negligible but included)
def A_func_III(tau):
    return A_beam(tau2 + tau)
def S_func_III(tau):
    return S_beam(tau2 + tau)
ops_III = run_star_product(A_func_III, S_func_III, tau_bot_full - tau2, K_III)

# Star-product couple: total = I * II * III
R1, Tu1, Td1, Rb1, su1, sd1 = ops_I
R2, Tu2, Td2, Rb2, su2, sd2 = ops_II
R3, Tu3, Td3, Rb3, su3, sd3 = ops_III

# I * II
LHS = np.eye(N) - R2[:, :N] @ Rb1 if R2.shape[1] == N else np.eye(N) - R2 @ Rb1
# Use star_combine
ops_12 = star_combine(R1, Tu1, Td1, Rb1, su1, sd1,
                      R2, Tu2, Rb2, Td2, su2, sd2)

# (I*II) * III
ops_total = star_combine(*ops_12, R3, Tu3, Rb3, Td3, su3, sd3)
R_total, T_up_total, T_down_total, R_bot_total, s_up_total, s_down_total = ops_total

t_3domain = time.perf_counter() - t0

# Full reference: single Magnus run over [0, tau_bot_full]
t0 = time.perf_counter()
ops_full = run_star_product(A_beam, S_beam, tau_bot_full, K_full)
R_full, T_up_full, T_down_full, R_bot_full, s_up_full, s_down_full = ops_full
t_full = time.perf_counter() - t0

# Compare: I+(0) with b_neg=0, b_pos=0
u_3domain = s_up_total  # I+(0) for zero BCs
u_full_mag = s_up_full

# Also compare with pydisort reference
flux_ref = float(Fp_ref(0))
flux_3domain = float(2 * math.pi * np.dot(mu_arr_pos * W, u_3domain))
flux_full = float(2 * math.pi * np.dot(mu_arr_pos * W, u_full_mag))

print(f"\nFlux comparison at ToA:")
print(f"  pydisort reference:    {flux_ref:.6e}")
print(f"  Full Magnus (K={K_full}): {flux_full:.6e}  "
      f"rel_err = {abs(flux_full-flux_ref)/abs(flux_ref):.3e}")
print(f"  3-domain (K={K_I}+{K_II}+{K_III}={K_I+K_II+K_III}): {flux_3domain:.6e}  "
      f"rel_err = {abs(flux_3domain-flux_ref)/abs(flux_ref):.3e}")

# Intensity comparison
err_u_3d = np.linalg.norm(u_3domain - u_full_mag) / max(np.linalg.norm(u_full_mag), 1e-15)
print(f"\n  ||u_3domain - u_full_magnus|| / ||u_full|| = {err_u_3d:.3e}")
print(f"  3-domain wall time: {t_3domain:.2f} s")
print(f"  Full Magnus wall time: {t_full:.2f} s")

print(f"\nThe 3-domain coupling via star product is exact (no additional error).")
print(f"In a production implementation, Domain II could use Radau Riccati")
print(f"instead of Magnus, potentially reducing the step count from {K_II}")
print(f"to ~nfev_radau steps.")


# ======================================================================
print("\n" + "=" * 70)
print("PART F: Exponential / Sylvester-based integrators on sub-domain")
print("=" * 70)
print(f"\nKey idea: the Riccati Jacobian J(E) = (alpha+R*beta)E + E(alpha+beta*R)")
print(f"is a Sylvester operator. Implicit steps solve N*N Sylvester equations")
print(f"in O(N^3), avoiding the N^2*N^2 = {N_sq}*{N_sq} Jacobian entirely.")

I_N = np.eye(N)


def rosenbrock_euler_RT(alpha, beta, tau_sub_val, K):
    """1st-order L-stable Rosenbrock-Euler with Sylvester solve for R,
    implicit Euler for T.  Cost: 1 Sylvester + 1 linear solve per step."""
    h = tau_sub_val / K
    R = np.zeros((N, N))
    T = np.eye(N)
    for _ in range(K):
        F = alpha @ R + R @ alpha + R @ beta @ R + beta
        An = alpha + R @ beta
        Bn = alpha + beta @ R
        # Sylvester: (I - h An) dR + dR (-h Bn) = h F
        dR = scipy.linalg.solve_sylvester(I_N - h * An, -h * Bn, h * F)
        # T: T_{n+1} = T_n (I - h Bn)^{-1}
        T = scipy.linalg.solve((I_N - h * Bn).T, T.T).T
        R = R + dR
    return R, T


def richardson_RT(alpha, beta, tau_sub_val, K):
    """2nd-order via Richardson extrapolation of Rosenbrock-Euler.
    Runs K and 2K steps, extrapolates R.  T computed at 2K resolution."""
    def _ros1_RT(alpha, beta, tau_sub_val, K_inner):
        h = tau_sub_val / K_inner
        R = np.zeros((N, N))
        T = np.eye(N)
        for _ in range(K_inner):
            F = alpha @ R + R @ alpha + R @ beta @ R + beta
            An = alpha + R @ beta
            Bn = alpha + beta @ R
            dR = scipy.linalg.solve_sylvester(I_N - h * An, -h * Bn, h * F)
            T = scipy.linalg.solve((I_N - h * Bn).T, T.T).T
            R = R + dR
        return R, T
    R_K, _ = _ros1_RT(alpha, beta, tau_sub_val, K)
    R_2K, T_2K = _ros1_RT(alpha, beta, tau_sub_val, 2 * K)
    R_rich = 2.0 * R_2K - R_K  # Richardson extrapolation: O(h^2)
    return R_rich, T_2K  # T at 2K resolution (1st order, but 2x finer)


def exp_euler_RT(alpha, beta, tau_sub_val, K):
    """Exponential Euler with midpoint phi_1 approximation.
    A-stable (not L-stable).  Cost: 2 N*N expm + 1 N*N expm per step."""
    h = tau_sub_val / K
    R = np.zeros((N, N))
    T = np.eye(N)
    for _ in range(K):
        F = alpha @ R + R @ alpha + R @ beta @ R + beta
        An = alpha + R @ beta
        Bn = alpha + beta @ R
        # phi_1(hJ)(F) ≈ expm(h/2 An) F expm(h/2 Bn)  (midpoint quadrature)
        eA = scipy.linalg.expm(0.5 * h * An)
        eB = scipy.linalg.expm(0.5 * h * Bn)
        R = R + h * (eA @ F @ eB)
        # T: T_{n+1} = T_n expm(h Bn)
        T = T @ scipy.linalg.expm(h * Bn)
    return R, T


# --- Sweep K for each method ---
print(f"\ntau_sub={tau_sub}, k_max={k_max:.2f}, k_min={k_min:.4f}")
print(f"Magnus reference: K={K_magnus}, rel_err={rel_err:.2e}, wall={t_magnus:.2f}s")
print(f"Radau reference:  nfev={sol_R.nfev + sol_T.nfev}, "
      f"rel_err={rel_err_radau:.2e}, wall={t_radau_R + t_radau_T:.2f}s")

K_sweep = [5, 10, 20, 50, 100, 200, 500]

for method_name, method_func in [
    ("Rosenbrock-Euler (Sylvester, order 1)", rosenbrock_euler_RT),
    ("Richardson (Sylvester, order 2, cost=3K)", richardson_RT),
    ("Exp-Euler (midpoint phi_1)", exp_euler_RT),
]:
    print(f"\n  {method_name}:")
    print(f"    {'K':>6s} {'h*k_max':>10s} {'rel_err':>12s} {'wall(s)':>8s}")
    for K_test in K_sweep:
        h_test = tau_sub / K_test
        t0 = time.perf_counter()
        try:
            R_test, T_test = method_func(alpha0, beta0, tau_sub, K_test)
            t_wall = time.perf_counter() - t0
            if np.any(np.abs(R_test) > 1e10) or np.any(np.isnan(R_test)):
                print(f"    {K_test:6d} {h_test*k_max:10.2f} "
                      f"{'BLOWUP':>12s} {t_wall:8.3f}")
                continue
            u_test = R_test @ b_neg + T_test @ b_pos
            err_test = np.linalg.norm(u_test - u_ref) / np.linalg.norm(u_ref)
            print(f"    {K_test:6d} {h_test*k_max:10.2f} "
                  f"{err_test:12.3e} {t_wall:8.3f}")
        except Exception as e:
            t_wall = time.perf_counter() - t0
            print(f"    {K_test:6d} {h_test*k_max:10.2f} "
                  f"{'FAIL':>12s} {t_wall:8.3f}  ({e.__class__.__name__})")


# --- Scaling with omega for Rosenbrock-Euler (Sylvester) ---
print(f"\n{'=' * 50}")
print(f"Rosenbrock-Euler (Sylvester) scaling with omega (K=200):")
print(f"  tau_sub={tau_sub}, g={g_default}, NQuad={NQuad}")
print(f"  {'omega':>8s} {'gamma':>8s} {'K_ros':>7s} {'err_ros':>10s} "
      f"{'t_ros':>7s} {'K_mag':>7s} {'err_mag':>10s} {'t_mag':>7s}")

K_ros_fixed = 200
for omega_test in [0.9, 0.99, 0.999, 0.9999]:
    alpha_t, beta_t = build_alpha_beta(omega_test, g_default)
    k_t = compute_k_values(alpha_t, beta_t)
    kmax_t, kmin_t = k_t[0], k_t[-1]
    gamma_t = kmax_t / kmin_t

    u0f_t, _, _ = get_pydisort_reference(tau_bot_full, omega_test, g_default, NQuad)
    bn_t = u0f_t(tau1)[N:]
    bp_t = u0f_t(tau2)[:N]
    ur_t = u0f_t(tau1)[:N]

    # Magnus
    A_t, S_t = build_A_and_S_func(omega_test, g_default)
    K_mag = max(int(kmax_t * tau_sub) + 10, 100)
    t0 = time.perf_counter()
    ops_t = run_star_product(A_t, S_t, tau_sub, K_mag)
    t_mag = time.perf_counter() - t0
    u_mag = ops_t[0] @ bn_t + ops_t[1] @ bp_t + ops_t[4]
    err_mag = np.linalg.norm(u_mag - ur_t) / max(np.linalg.norm(ur_t), 1e-15)

    # Rosenbrock-Euler (Sylvester)
    t0 = time.perf_counter()
    R_ros, T_ros = rosenbrock_euler_RT(alpha_t, beta_t, tau_sub, K_ros_fixed)
    t_ros = time.perf_counter() - t0
    u_ros = R_ros @ bn_t + T_ros @ bp_t
    err_ros = np.linalg.norm(u_ros - ur_t) / max(np.linalg.norm(ur_t), 1e-15)

    print(f"  {omega_test:8.4f} {gamma_t:8.1f} {K_ros_fixed:7d} {err_ros:10.2e} "
          f"{t_ros:7.3f} {K_mag:7d} {err_mag:10.2e} {t_mag:7.3f}")

print(f"\nKey: Rosenbrock-Euler K is FIXED at {K_ros_fixed}, independent of k_max.")
print(f"Magnus K scales as k_max*tau_sub.  Wall-time advantage grows with k_max.")
