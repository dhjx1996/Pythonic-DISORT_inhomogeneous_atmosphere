"""Jacobian of u0_ToA w.r.t. g1(tau) in a conservative adiabatic cloud.

Usage: python _jacobian_conservative.py <outfile.npz>

Scenario: omega = 1 - 1e-6 (conservative), g(tau) linear adiabatic profile
(no drizzle spike).  Isolates the diffusion-domain decay from absorption.
"""
import sys, os, time
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
# Problem parameters: conservative adiabatic cloud
# ----------------------------------------------------------------
tau_bot = 30.0
omega_val = 1.0 - 1e-6               # nearly conservative
g_top, g_bot = 0.865, 0.820          # adiabatic cloud (no spike)
NQuad = 16; NLeg = NQuad
N = NQuad // 2
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tol = 1e-3

def omega_nominal(tau):
    return omega_val + 0.0 * tau      # constant, but JAX-traceable

def g_nominal(tau):
    return g_top + (g_bot - g_top) * tau / tau_bot

def Leg_coeffs_nominal(tau):
    return g_nominal(tau) ** jnp.arange(NLeg)

# ----------------------------------------------------------------
# Step 1: Nominal solve
# ----------------------------------------------------------------
print("Step 1: Nominal solve (conservative cloud)...", flush=True)
t0 = time.time()
mu_arr, flux_up, u0_nom, u_func_nom, tau_grid_nom = pydisort_riccati_jax(
    tau_bot, omega_nominal, Leg_coeffs_nominal, NQuad, mu0, I0, phi0, tol=tol)
print(f"  {time.time()-t0:.1f}s, {len(tau_grid_nom)} grid points", flush=True)
print(f"  u0_nominal = {u0_nom}", flush=True)
print(f"  flux_up = {flux_up:.6f}", flush=True)

# Grid distribution
n_boa = np.sum(np.asarray(tau_grid_nom) > 25)
n_spike = np.sum((np.asarray(tau_grid_nom) > 13) & (np.asarray(tau_grid_nom) < 17))
n_toa = np.sum(np.asarray(tau_grid_nom) < 5)
print(f"  Grid: {n_toa} ToA (<5), {n_spike} mid (13-17), {n_boa} BoA (>25)", flush=True)

# ----------------------------------------------------------------
# Step 2: Setup for Jacobian
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
# Step 3: AD-compatible integration
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
    R_up, T_up, s_up = _integrate_final(
        lambda s: alpha_f(tb - s), lambda s: beta_f(tb - s), tb,
        lambda s: q_down(tb - s), lambda s: q_up(tb - s))
    R_dn, T_dn, s_dn = _integrate_final(
        alpha_f, beta_f, tb, q_up, q_down)
    return _solve_bc_riccati_jax(
        R_up, T_up, T_dn, R_dn, s_up, s_dn,
        N, jnp.zeros(N), jnp.zeros(N),
        None, mu_arr_pos, W_jax, 0, mu0, I0_div_4pi, tau_bot, True)

# ----------------------------------------------------------------
# Step 4: Compute Jacobians
# ----------------------------------------------------------------
zero_K = jnp.zeros(K)

print("\nStep 4a: J_g1 = jacrev(solve_m0, argnums=1) ...", flush=True)
t0 = time.time()
J_g1 = jnp.asarray(jax.jacrev(solve_m0, argnums=1)(zero_K, zero_K))
print(f"  {time.time()-t0:.1f}s, shape {J_g1.shape}", flush=True)

print("Step 4b: J_omega = jacrev(solve_m0, argnums=0) ...", flush=True)
t0 = time.time()
J_omega = jnp.asarray(jax.jacrev(solve_m0, argnums=0)(zero_K, zero_K))
print(f"  {time.time()-t0:.1f}s, shape {J_omega.shape}", flush=True)

# ----------------------------------------------------------------
# Step 5: Analysis
# ----------------------------------------------------------------
print("\nStep 5: Sensitivity analysis (conservative cloud)", flush=True)
J_g1_np    = np.asarray(J_g1)
J_omega_np = np.asarray(J_omega)
tau_np     = np.asarray(tau_nodes)

sens_g1    = np.sqrt(np.sum(J_g1_np**2, axis=0))
sens_omega = np.sqrt(np.sum(J_omega_np**2, axis=0))

regions = [
    ("ToA  (tau<5)",      tau_np < 5),
    ("Mid  (5<tau<13)",   (tau_np >= 5) & (tau_np < 13)),
    ("Mid2 (13<tau<17)",  (tau_np >= 13) & (tau_np < 17)),
    ("Deep (17<tau<25)",  (tau_np >= 17) & (tau_np < 25)),
    ("BoA  (tau>=25)",    tau_np >= 25),
]

print("  ||dU0/d_g1||_2 by region:")
for name, mask in regions:
    v = sens_g1[mask]
    print(f"    {name}: mean={v.mean():.4e}  max={v.max():.4e}  min={v.min():.4e}")

print("  ||dU0/d_omega||_2 by region:")
for name, mask in regions:
    v = sens_omega[mask]
    print(f"    {name}: mean={v.mean():.4e}  max={v.max():.4e}  min={v.min():.4e}")

# Ratios
for label, sens in [("g1", sens_g1), ("omega", sens_omega)]:
    boa_mean = sens[tau_np >= 25].mean()
    toa_mean = sens[tau_np < 5].mean()
    deep_mean = sens[(tau_np >= 17) & (tau_np < 25)].mean()
    print(f"  {label}: BoA/ToA = {boa_mean/toa_mean:.2e}, "
          f"Deep/ToA = {deep_mean/toa_mean:.2e}")

# Per-stream at BoA vs ToA for g1
print("\n  Per-stream |J_g1| at BoA (tau=30) vs ToA (tau=0):")
for i in range(N):
    boa_val = abs(J_g1_np[i, -1])
    toa_val = abs(J_g1_np[i,  0])
    ratio = boa_val / toa_val if toa_val > 1e-30 else float('inf')
    print(f"    stream {i+1} (mu={float(mu_arr_pos[i]):.3f}): "
          f"|J_BoA|={boa_val:.3e}  |J_ToA|={toa_val:.3e}  ratio={ratio:.2e}")

# Compare with absorbing case if available
prev = "/tmp/jacobian_results.npz"
if os.path.exists(prev):
    d = np.load(prev)
    prev_sens_g1 = d['sensitivity_g1']
    prev_tau = d['tau_nodes']
    # BoA/ToA ratio in absorbing case
    prev_boa = prev_sens_g1[prev_tau >= 25].mean()
    prev_toa = prev_sens_g1[prev_tau < 5].mean()
    print(f"\n  Comparison with absorbing case:")
    print(f"    Absorbing: BoA/ToA = {prev_boa/prev_toa:.2e}")
    print(f"    Conservative: BoA/ToA = {sens_g1[tau_np>=25].mean()/sens_g1[tau_np<5].mean():.2e}")

# ----------------------------------------------------------------
# Step 6: Save
# ----------------------------------------------------------------
np.savez(outfile,
         tau_nodes=tau_np,
         J_g1=J_g1_np,
         J_omega=J_omega_np,
         sensitivity_g1=sens_g1,
         sensitivity_omega=sens_omega,
         u0_nominal=np.asarray(u0_nom),
         tau_grid_nominal=np.asarray(tau_grid_nom),
         mu_arr=np.asarray(mu_arr),
         omega_val=omega_val,
         g_top=g_top, g_bot=g_bot)
print(f"\nSaved to {outfile}", flush=True)
