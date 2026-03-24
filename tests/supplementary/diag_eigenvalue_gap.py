"""
Diagnostic: eigenvalue gap in the DISORT coefficient matrix.

Quantifies the separation between the fastest (ballistic) and slowest (diffusion)
modes across parameter space and cloud profiles.  The gap ratio γ = k_max/k_min
determines the potential speedup from diffusion-domain methods.

Part A: Eigenvalues vs ω and g (homogeneous atmospheres)
Part B: Eigenvalue gap along an adiabatic cloud profile
Part C: NQuad dependence of k_max and k_min
"""
import numpy as np
import sys, math
sys.path.insert(0, '.')
from _helpers import make_D_m_funcs, make_cloud_profile
from PythonicDISORT import subroutines


def compute_alpha_beta(omega, D_m, tau, mu_arr_pos, W, N):
    """Build α and β matrices at a given optical depth."""
    M_inv = 1.0 / mu_arr_pos
    D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    return alpha, beta


def compute_k_values(alpha, beta):
    """
    Compute k-eigenvalues via the N×N reduction.

    The 2N×2N matrix A = [[-α, -β], [β, α]] has eigenvalues ±k_j where
    k_j² = eigenvalues of (α-β)(α+β).  This follows from defining
    sum = u⁺ + u⁻ and diff = u⁺ - u⁻, which decouples the eigenvalue
    problem into two N×N problems.
    """
    k_sq = np.linalg.eigvals((alpha - beta) @ (alpha + beta)).real
    k_sq = np.sort(k_sq)[::-1]
    return np.sqrt(np.maximum(k_sq, 0))


# ======================================================================
print("=" * 70)
print("PART A: Eigenvalues vs omega and g (homogeneous, NQuad=8)")
print("=" * 70)

NQuad = 8; N = NQuad // 2
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)

for g in [0.0, 0.5, 0.85]:
    g_l = g ** np.arange(NQuad)
    D_m_funcs = make_D_m_funcs(g_l, NQuad, NQuad)
    D_m = D_m_funcs[0]

    print(f"\ng = {g}")
    print(f"  {'omega':>8s} {'k_max':>10s} {'k_min':>10s} {'gamma':>10s} "
          f"{'k_2s':>10s} {'k_min/k_2s':>12s}")
    print(f"  {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} "
          f"{'-'*10:>10s} {'-'*12:>12s}")

    for omega in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]:
        alpha, beta = compute_alpha_beta(omega, D_m, 1.0, mu_arr_pos, W, N)
        k_vals = compute_k_values(alpha, beta)
        k_max = k_vals[0]
        k_min = k_vals[-1]
        gamma = k_max / k_min if k_min > 1e-15 else float('inf')
        k_2s = math.sqrt(3 * (1 - omega) * (1 - omega * g))
        ratio = k_min / k_2s if k_2s > 1e-15 else float('inf')
        print(f"  {omega:8.4f} {k_max:10.4f} {k_min:10.4f} {gamma:10.1f} "
              f"{k_2s:10.4f} {ratio:12.4f}")


# ======================================================================
print("\n" + "=" * 70)
print("PART B: Eigenvalue gap along adiabatic cloud profile")
print("=" * 70)

tau_bot_cloud = 30.0
omega_func, g_l_func, D_m_funcs_cloud = make_cloud_profile(
    tau_bot=tau_bot_cloud, omega_top=0.85, omega_bot=0.96,
    g_top=0.865, g_bot=0.820, NLeg=NQuad, NQuad=NQuad,
)
D_m_cloud = D_m_funcs_cloud[0]

print(f"\nCloud: tau_bot={tau_bot_cloud}, omega=[0.85 -> 0.96], g=[0.865 -> 0.820]")
print(f"  {'tau':>8s} {'omega':>8s} {'g':>8s} {'k_max':>10s} {'k_min':>10s} {'gamma':>10s}")
print(f"  {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")

for tau in np.linspace(0.5, tau_bot_cloud - 0.5, 20):
    omega = omega_func(tau)
    g_l = g_l_func(tau)
    g_val = g_l[1] if len(g_l) > 1 else 0.0
    alpha, beta = compute_alpha_beta(omega, D_m_cloud, tau, mu_arr_pos, W, N)
    k_vals = compute_k_values(alpha, beta)
    k_max, k_min = k_vals[0], k_vals[-1]
    gamma = k_max / k_min if k_min > 1e-15 else float('inf')
    print(f"  {tau:8.2f} {omega:8.4f} {g_val:8.4f} {k_max:10.4f} {k_min:10.4f} {gamma:10.1f}")


# ======================================================================
print("\n" + "=" * 70)
print("PART C: NQuad dependence (omega=0.99, g=0.85)")
print("=" * 70)

omega_c = 0.99; g_c = 0.85
print(f"\nomega={omega_c}, g={g_c}")
print(f"  {'NQuad':>6s} {'N':>4s} {'k_max':>10s} {'k_min':>10s} "
      f"{'gamma':>10s} {'1/mu_min':>10s}")
print(f"  {'-'*6:>6s} {'-'*4:>4s} {'-'*10:>10s} {'-'*10:>10s} "
      f"{'-'*10:>10s} {'-'*10:>10s}")

for NQ in [4, 8, 16]:
    N_loc = NQ // 2
    mu_pos, W_loc = subroutines.Gauss_Legendre_quad(N_loc)
    g_l_loc = g_c ** np.arange(NQ)
    D_fns = make_D_m_funcs(g_l_loc, NQ, NQ)
    alpha, beta = compute_alpha_beta(omega_c, D_fns[0], 1.0, mu_pos, W_loc, N_loc)
    k_vals = compute_k_values(alpha, beta)
    k_max, k_min = k_vals[0], k_vals[-1]
    gamma = k_max / k_min if k_min > 1e-15 else float('inf')
    print(f"  {NQ:6d} {N_loc:4d} {k_max:10.4f} {k_min:10.4f} "
          f"{gamma:10.1f} {1/mu_pos[0]:10.4f}")

print("\nNote: k_max ~ 1/mu_min grows with NQuad (ballistic mode).")
print("      k_min stays approximately constant (physical diffusion mode).")
print("      The gap gamma grows as ~NQuad^2.")
