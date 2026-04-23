"""Compute Jacobians du0_ToA/d_omega(tau) and du0_ToA/d_g1(tau) via JAX AD.

Usage: python _jacobian_ad.py <outfile.npz>

Forward-mode AD (jax.jacfwd) through the full Riccati ODE (forward + backward
sweeps) and BC solve for the m=0 Fourier mode.  Perturbations are piecewise-
linear on a uniform tau grid of K=61 nodes.

Output: J_omega (N, K), J_g1 (N, K), plus sensitivity norms and metadata.
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from math import pi
import diffrax

from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
)
from _solve_bc_riccati_jax import _solve_bc_riccati_jax
from PythonicDISORT import subroutines
from pydisort_riccati_jax import pydisort_riccati_jax

outfile = sys.argv[1]

# ----------------------------------------------------------------
# Problem parameters (adiabatic cloud with drizzle)
# ----------------------------------------------------------------
tau_bot = 30.0
omega_top, omega_bot = 0.85, 0.96
g_top, g_bot = 0.865, 0.820
NQuad = 16; NLeg = NQuad
N = NQuad // 2
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tau_spike = 15.0; sigma_w = 0.5
d_omega_spike = -0.15; d_g_spike = 0.04
tol = 1e-3

def omega_nominal(tau):
    return (omega_top + (omega_bot - omega_top) * tau / tau_bot
            + d_omega_spike * jnp.exp(-0.5 * ((tau - tau_spike) / sigma_w)**2))

def g_nominal(tau):
    return (g_top + (g_bot - g_top) * tau / tau_bot
            + d_g_spike * jnp.exp(-0.5 * ((tau - tau_spike) / sigma_w)**2))

def Leg_coeffs_nominal(tau):
    return g_nominal(tau) ** jnp.arange(NLeg)

# ----------------------------------------------------------------
# Step 1: Nominal solve (for tau_grid and baseline u0)
# ----------------------------------------------------------------
print("Step 1: Nominal solve...", flush=True)
t0 = time.time()
mu_arr, flux_up, u0_nom, u_func_nom, tau_grid_nom = pydisort_riccati_jax(
    tau_bot, omega_nominal, Leg_coeffs_nominal, NQuad, mu0, I0, phi0, tol=tol)
print(f"  {time.time()-t0:.1f}s, {len(tau_grid_nom)} grid points", flush=True)
print(f"  u0_nominal = {u0_nom}", flush=True)

# ----------------------------------------------------------------
# Step 2: Quadrature and Legendre setup
# ----------------------------------------------------------------
mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos = jnp.array(mu_arr_pos_np)
W_jax = jnp.array(W_np)
M_inv = 1.0 / mu_arr_pos
I0_div_4pi = I0 / (4 * pi)

K = 61
tau_nodes = jnp.linspace(0.0, tau_bot, K)

leg_data_m0 = _precompute_legendre(0, NLeg, mu_arr_pos, mu0)

# ----------------------------------------------------------------
# Step 3: AD-compatible Riccati integration (SaveAt(t1=True))
# ----------------------------------------------------------------

def _integrate_final(alpha_func, beta_func, sigma_end,
                     q1_func, q2_func):
    """Kvaerno5 -> final (R, T, s). Fully JAX-traceable."""
    def vf(sigma, state, args):
        R, T, s = state['R'], state['T'], state['s']
        alpha = alpha_func(sigma)
        beta  = beta_func(sigma)
        A = alpha + R @ beta
        dR = alpha @ R + R @ alpha + R @ beta @ R + beta
        dT = A @ T
        ds = A @ s + R @ q1_func(sigma) + q2_func(sigma)
        return {'R': dR, 'T': dT, 's': ds}

    y0 = {'R': jnp.zeros((N, N)), 'T': jnp.eye(N), 's': jnp.zeros(N)}
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(),
        t0=0.0, t1=float(sigma_end), dt0=None, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol*1e-3),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=4096)
    # SaveAt(t1=True) adds a leading time dim of 1; remove it
    return sol.ys['R'][0], sol.ys['T'][0], sol.ys['s'][0]


def solve_m0(delta_omega_vec, delta_g1_vec):
    """m=0: (delta_omega, delta_g1) each (K,) -> u0_ToA (N,)."""
    def omega_func(tau):
        return omega_nominal(tau) + jnp.interp(tau, tau_nodes, delta_omega_vec)
    def Leg_coeffs_func(tau):
        return Leg_coeffs_nominal(tau).at[1].add(
            jnp.interp(tau, tau_nodes, delta_g1_vec))

    alpha_f, beta_f = _make_alpha_beta_funcs_jax(
        omega_func, Leg_coeffs_func, 0, leg_data_m0,
        mu_arr_pos, W_jax, M_inv, N)
    q_up, q_down = _make_q_funcs_jax(
        omega_func, Leg_coeffs_func, 0, leg_data_m0,
        mu_arr_pos, M_inv, mu0, I0_div_4pi, True, N)

    tb = float(tau_bot)

    # Forward sweep (BoA -> ToA): sigma in [0, tau_bot],
    #   coefficients evaluated at tau = tau_bot - sigma
    R_up, T_up, s_up = _integrate_final(
        lambda s: alpha_f(tb - s), lambda s: beta_f(tb - s), tb,
        lambda s: q_down(tb - s), lambda s: q_up(tb - s))

    # Backward sweep (ToA -> BoA): sigma in [0, tau_bot],
    #   coefficients evaluated at tau = sigma
    R_dn, T_dn, s_dn = _integrate_final(
        alpha_f, beta_f, tb, q_up, q_down)

    # BC solve (no BDRF, no diffuse BCs)
    return _solve_bc_riccati_jax(
        R_up, T_up, T_dn, R_dn, s_up, s_dn,
        N, jnp.zeros(N), jnp.zeros(N),
        None, mu_arr_pos, W_jax, 0, mu0, I0_div_4pi, tau_bot, True)

# ----------------------------------------------------------------
# Step 4: Compute Jacobians via jax.jacfwd
# ----------------------------------------------------------------
zero_K = jnp.zeros(K)

print("\nStep 4a: J_omega = jacrev(solve_m0, argnums=0) ...", flush=True)
t0 = time.time()
J_omega = jax.jacrev(solve_m0, argnums=0)(zero_K, zero_K)
J_omega = jnp.asarray(J_omega)                # ensure concrete
print(f"  {time.time()-t0:.1f}s, shape {J_omega.shape}", flush=True)

print("Step 4b: J_g1 = jacrev(solve_m0, argnums=1) ...", flush=True)
t0 = time.time()
J_g1 = jax.jacrev(solve_m0, argnums=1)(zero_K, zero_K)
J_g1 = jnp.asarray(J_g1)
print(f"  {time.time()-t0:.1f}s, shape {J_g1.shape}", flush=True)

# ----------------------------------------------------------------
# Step 5: Finite-difference validation (central, eps=1e-5)
# ----------------------------------------------------------------
print("\nStep 5: FD validation ...", flush=True)
solve_jit = jax.jit(solve_m0)
# Warm up JIT
_ = solve_jit(zero_K, zero_K)

eps = 1e-5
test_cols = [0, K//4, K//2, 3*K//4, K-1]
for j in test_cols:
    e_j = jnp.zeros(K).at[j].set(1.0)
    u_p = solve_jit(eps * e_j, zero_K)
    u_m = solve_jit(-eps * e_j, zero_K)
    fd = (u_p - u_m) / (2 * eps)
    ad = J_omega[:, j]
    denom = jnp.max(jnp.abs(ad))
    rel = jnp.max(jnp.abs(fd - ad)) / jnp.where(denom > 1e-30, denom, 1.0)
    print(f"  omega j={j:2d} (tau={float(tau_nodes[j]):5.1f}): "
          f"max|FD-AD|/max|AD| = {float(rel):.2e}", flush=True)

for j in test_cols:
    e_j = jnp.zeros(K).at[j].set(1.0)
    u_p = solve_jit(zero_K, eps * e_j)
    u_m = solve_jit(zero_K, -eps * e_j)
    fd = (u_p - u_m) / (2 * eps)
    ad = J_g1[:, j]
    denom = jnp.max(jnp.abs(ad))
    rel = jnp.max(jnp.abs(fd - ad)) / jnp.where(denom > 1e-30, denom, 1.0)
    print(f"  g1    j={j:2d} (tau={float(tau_nodes[j]):5.1f}): "
          f"max|FD-AD|/max|AD| = {float(rel):.2e}", flush=True)

# ----------------------------------------------------------------
# Step 6: Analysis
# ----------------------------------------------------------------
print("\nStep 6: Sensitivity analysis", flush=True)
J_omega_np = np.asarray(J_omega)
J_g1_np    = np.asarray(J_g1)
tau_np     = np.asarray(tau_nodes)

sens_omega = np.sqrt(np.sum(J_omega_np**2, axis=0))  # (K,)
sens_g1    = np.sqrt(np.sum(J_g1_np**2, axis=0))     # (K,)

regions = [
    ("ToA  (tau<5)",      tau_np < 5),
    ("Mid  (5<tau<13)",   (tau_np >= 5) & (tau_np < 13)),
    ("Spike(13<tau<17)",  (tau_np >= 13) & (tau_np < 17)),
    ("Deep (17<tau<25)",  (tau_np >= 17) & (tau_np < 25)),
    ("BoA  (tau>=25)",    tau_np >= 25),
]

print("  ||dU0/d_omega||_2 by region:")
for name, mask in regions:
    v = sens_omega[mask]
    print(f"    {name}: mean={v.mean():.4e}  max={v.max():.4e}  min={v.min():.4e}")

print("  ||dU0/d_g1||_2 by region:")
for name, mask in regions:
    v = sens_g1[mask]
    print(f"    {name}: mean={v.mean():.4e}  max={v.max():.4e}  min={v.min():.4e}")

# Attenuation ratio BoA / ToA
boa_mask = tau_np >= 25
toa_mask = tau_np < 5
for label, sens in [("omega", sens_omega), ("g1", sens_g1)]:
    ratio = sens[boa_mask].mean() / sens[toa_mask].mean()
    print(f"  {label}: BoA/ToA mean sensitivity ratio = {ratio:.2e}")

# Per-stream breakdown at BoA vs ToA
print("\n  Per-stream |J_omega| at BoA (tau=30) vs ToA (tau=0):")
for i in range(N):
    boa_val = abs(J_omega_np[i, -1])
    toa_val = abs(J_omega_np[i,  0])
    ratio = boa_val / toa_val if toa_val > 1e-30 else float('inf')
    print(f"    stream {i+1} (mu={float(mu_arr_pos[i]):.3f}): "
          f"|J_BoA|={boa_val:.3e}  |J_ToA|={toa_val:.3e}  ratio={ratio:.2e}")

# ----------------------------------------------------------------
# Step 7: Save
# ----------------------------------------------------------------
np.savez(outfile,
         tau_nodes=tau_np,
         J_omega=J_omega_np,
         J_g1=J_g1_np,
         sensitivity_omega=sens_omega,
         sensitivity_g1=sens_g1,
         u0_nominal=np.asarray(u0_nom),
         tau_grid_nominal=np.asarray(tau_grid_nom),
         mu_arr=np.asarray(mu_arr))
print(f"\nSaved to {outfile}", flush=True)
