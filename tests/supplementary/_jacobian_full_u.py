"""Full-radiance-field Jacobian analysis of J_g = du_ToA/dg(tau).

Usage: python _jacobian_full_u.py <omega> <tau_spike|none> <NQuad> <max_m> <outfile>

  omega     : uniform single-scattering albedo (constant with tau)
  tau_spike : spike centre in g(tau), or "none" for smooth linear g profile
  NQuad     : number of quadrature streams (e.g. 16 or 32)
  max_m     : number of Fourier modes (1 = m=0 only; NQuad = all modes)
  outfile   : .npz output file

Computes J_g_m = du_m_ToA/dg(tau) for each Fourier mode m=0,...,max_m-1
via jax.jacrev, then stacks into J_g_full (N*max_m, K) for SVD rank
analysis.  Supports variable NQuad for tiered u-tests:
  Tier (a): all Fourier modes, NQuad=16
  Tier (b): m=0 only, NQuad=32
  Tier (a+b): all Fourier modes, NQuad=32
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

# ================================================================
# Parse arguments
# ================================================================
omega_val = float(sys.argv[1])
tau_spike_arg = sys.argv[2]
NQuad = int(sys.argv[3])
max_m = int(sys.argv[4])
outfile = sys.argv[5]

use_spike = (tau_spike_arg.lower() != "none")
tau_spike_center = float(tau_spike_arg) if use_spike else None

NLeg = NQuad
N = NQuad // 2

print(f"=== Full-u Jacobian analysis ===", flush=True)
print(f"  omega={omega_val}, NQuad={NQuad}, max_m={max_m}", flush=True)
print(f"  spike={'none' if not use_spike else f'tau={tau_spike_center}'}", flush=True)

# ================================================================
# Problem parameters
# ================================================================
tau_bot = 30.0
g_top, g_bot = 0.865, 0.820
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tol = 1e-3
K = 61
sigma_spike = 0.5
d_g_spike = 0.04

def omega_nominal(tau):
    return omega_val + 0.0 * tau

def g_nominal(tau):
    g_lin = g_top + (g_bot - g_top) * tau / tau_bot
    if use_spike:
        g_lin = g_lin + d_g_spike * jnp.exp(
            -0.5 * ((tau - tau_spike_center) / sigma_spike)**2)
    return g_lin

def Leg_coeffs_nominal(tau):
    return g_nominal(tau) ** jnp.arange(NLeg)

# ================================================================
# Quadrature and tau grid
# ================================================================
mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos = jnp.array(mu_arr_pos_np)
W_jax = jnp.array(W_np)
M_inv = 1.0 / mu_arr_pos
I0_div_4pi = I0 / (4 * pi)
tau_nodes = jnp.linspace(0.0, tau_bot, K)
tau_np = np.asarray(tau_nodes)

# ================================================================
# AD-traceable Riccati integrator (SaveAt(t1=True) for AD)
# ================================================================
def _integrate_final(alpha_func, beta_func, sigma_end, q1_func, q2_func):
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
        saveat=diffrax.SaveAt(t1=True), max_steps=4096)
    return sol.ys['R'][0], sol.ys['T'][0], sol.ys['s'][0]


# ================================================================
# Per-mode Jacobian computation
# ================================================================
zero_K = jnp.zeros(K)
J_g_modes = []  # list of (N, K) arrays, one per Fourier mode
t_total_start = time.time()

for m in range(max_m):
    print(f"\n--- Fourier mode m={m} ---", flush=True)
    t0 = time.time()

    # Precompute Legendre data for this mode
    leg_data_m = _precompute_legendre(m, NLeg, mu_arr_pos, mu0)

    # Check if this mode has any Legendre terms
    n_ells = NLeg - m
    if n_ells <= 0:
        print(f"  No Legendre terms (NLeg={NLeg}, m={m}), skipping", flush=True)
        J_g_modes.append(np.zeros((N, K)))
        continue

    def solve_m(delta_g_vec, _m=m, _leg_data=leg_data_m):
        """Solve for u_m_ToA given perturbation to g(tau)."""
        def omega_func(tau):
            return omega_nominal(tau)
        def Leg_coeffs_func(tau):
            g_pert = g_nominal(tau) + jnp.interp(tau, tau_nodes, delta_g_vec)
            return g_pert ** jnp.arange(NLeg)

        alpha_f, beta_f = _make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, _m, _leg_data,
            mu_arr_pos, W_jax, M_inv, N)
        q_up, q_down = _make_q_funcs_jax(
            omega_func, Leg_coeffs_func, _m, _leg_data,
            mu_arr_pos, M_inv, mu0, I0_div_4pi, _m == 0, N)
        tb = float(tau_bot)

        # Forward sweep (BoA -> ToA)
        R_up, T_up, s_up = _integrate_final(
            lambda s: alpha_f(tb - s), lambda s: beta_f(tb - s), tb,
            lambda s: q_down(tb - s), lambda s: q_up(tb - s))

        # Backward sweep (ToA -> BoA)
        R_dn, T_dn, s_dn = _integrate_final(
            alpha_f, beta_f, tb, q_up, q_down)

        return _solve_bc_riccati_jax(
            R_up, T_up, T_dn, R_dn, s_up, s_dn,
            N, jnp.zeros(N), jnp.zeros(N),
            None, mu_arr_pos, W_jax, _m, mu0, I0_div_4pi, tau_bot, True)

    # Compute Jacobian via jacrev
    J_g_m = jnp.asarray(jax.jacrev(solve_m)(zero_K))
    J_g_m_np = np.asarray(J_g_m)
    del J_g_m
    J_g_modes.append(J_g_m_np)

    # Report magnitude
    sens_m = np.sqrt(np.sum(J_g_m_np**2, axis=0))
    elapsed = time.time() - t0
    print(f"  {elapsed:.1f}s, shape {J_g_m_np.shape}", flush=True)
    print(f"  ||J_g_m={m}|| range: [{sens_m.min():.4e}, {sens_m.max():.4e}]", flush=True)

print(f"\nTotal Jacobian time: {time.time()-t_total_start:.1f}s", flush=True)

# ================================================================
# Stack into J_g_full and analyse
# ================================================================
J_g_full = np.vstack(J_g_modes)  # (N*max_m, K)
print(f"\nJ_g_full shape: {J_g_full.shape}", flush=True)

# Also extract m=0 block for comparison
J_g_m0 = J_g_modes[0]  # (N, K)

# L2 sensitivity (full and m=0)
sens_full = np.sqrt(np.sum(J_g_full**2, axis=0))
sens_m0 = np.sqrt(np.sum(J_g_m0**2, axis=0))

regions = [
    ("ToA  (tau<5)",     tau_np < 5),
    ("Mid  (5<tau<13)",  (tau_np >= 5) & (tau_np < 13)),
    ("Mid2 (13<tau<17)", (tau_np >= 13) & (tau_np < 17)),
    ("Deep (17<tau<25)", (tau_np >= 17) & (tau_np < 25)),
    ("BoA  (tau>=25)",   tau_np >= 25),
]

# ================================================================
# Per-mode ||J_g_m|| summary
# ================================================================
print("\n--- Per-mode sensitivity summary ---", flush=True)
print(f"  {'mode':<6s}  {'max ||J||':<12s}  {'min ||J||':<12s}  {'BoA ||J||':<12s}", flush=True)
for m_idx, J_m in enumerate(J_g_modes):
    sens = np.sqrt(np.sum(J_m**2, axis=0))
    print(f"  m={m_idx:<3d}  {sens.max():.4e}    {sens.min():.4e}    {sens[-1]:.4e}", flush=True)

# ================================================================
# Regional SVD analysis
# ================================================================
print("\n--- Regional SVD analysis ---", flush=True)

regions_svd = [
    ("ToA (tau<5)",      tau_np < 5),
    ("Mid (5<=tau<25)",  (tau_np >= 5) & (tau_np < 25)),
    ("BoA (tau>=25)",    tau_np >= 25),
]

svd_results = {}
for label, source, J_matrix in [("m=0 only", "m0", J_g_m0),
                                  ("full u", "full", J_g_full)]:
    print(f"\n  [{label}] (rows={J_matrix.shape[0]}):", flush=True)
    for name, mask in regions_svd:
        J_block = J_matrix[:, mask]
        n_cols = J_block.shape[1]
        S = np.linalg.svd(J_block, compute_uv=False)
        key = f"{source}_{name}"
        svd_results[key] = S
        print(f"    {name} ({n_cols} cols):", flush=True)
        print(f"      SV: {', '.join(f'{s:.3e}' for s in S[:min(10, len(S))])}", flush=True)
        for thresh_pct in [10, 1, 0.1]:
            rank = int(np.sum(S > (thresh_pct / 100) * S[0]))
            print(f"      Rank (>{thresh_pct}% of sigma_1): {rank}", flush=True)

# ================================================================
# Rank comparison table
# ================================================================
print("\n--- Rank comparison: m=0 vs full u ---", flush=True)
print(f"  {'Region':<20s}  {'m=0 (1%)':<10s}  {'full (1%)':<10s}  "
      f"{'m=0 (0.1%)':<10s}  {'full (0.1%)':<10s}", flush=True)
for name, mask in regions_svd:
    S_m0 = svd_results[f"m0_{name}"]
    S_full = svd_results[f"full_{name}"]
    r_m0_1 = int(np.sum(S_m0 > 0.01 * S_m0[0]))
    r_full_1 = int(np.sum(S_full > 0.01 * S_full[0]))
    r_m0_01 = int(np.sum(S_m0 > 0.001 * S_m0[0]))
    r_full_01 = int(np.sum(S_full > 0.001 * S_full[0]))
    print(f"  {name:<20s}  {r_m0_1:<10d}  {r_full_1:<10d}  "
          f"{r_m0_01:<10d}  {r_full_01:<10d}", flush=True)

# ================================================================
# Cosine similarity with BoA column
# ================================================================
print("\n--- Cosine similarity with BoA ---", flush=True)
for label, J_matrix in [("m=0", J_g_m0), ("full u", J_g_full)]:
    boa_col = J_matrix[:, -1]
    boa_norm = np.linalg.norm(boa_col)
    cos_sim = np.zeros(K)
    for j in range(K):
        col = J_matrix[:, j]
        cos_sim[j] = abs(np.dot(col, boa_col)) / (np.linalg.norm(col) * boa_norm + 1e-30)
    print(f"  [{label}]:", flush=True)
    for rname, mask in regions:
        v = cos_sim[mask]
        print(f"    {rname}: mean={v.mean():.4f}  min={v.min():.4f}", flush=True)

# ================================================================
# Sliding-window rank
# ================================================================
print("\n--- Sliding-window rank ---", flush=True)
w = 5
for label, J_matrix in [("m=0", J_g_m0), ("full u", J_g_full)]:
    sliding_rank = np.full(K, np.nan)
    for j in range(K):
        lo = max(0, j - w)
        hi = min(K, j + w + 1)
        if hi - lo < 2:
            continue
        S = np.linalg.svd(J_matrix[:, lo:hi], compute_uv=False)
        sliding_rank[j] = np.sum(S > 0.01 * S[0])
    print(f"  [{label}] (1% threshold):", flush=True)
    for rname, mask in regions:
        v = sliding_rank[mask]
        v = v[~np.isnan(v)]
        if len(v) > 0:
            print(f"    {rname}: mean={v.mean():.1f}  min={v.min():.0f}  max={v.max():.0f}",
                  flush=True)

# ================================================================
# Save
# ================================================================
print(f"\nSaving to {outfile} ...", flush=True)
save_dict = dict(
    omega=omega_val,
    NQuad=NQuad,
    max_m=max_m,
    tau_nodes=tau_np,
    J_g_full=J_g_full,
    J_g_m0=J_g_m0,
    sensitivity_full=sens_full,
    sensitivity_m0=sens_m0,
    mu_arr=mu_arr_pos_np,
    W=W_np,
)
# Save per-mode Jacobians
for m_idx, J_m in enumerate(J_g_modes):
    save_dict[f'J_g_m{m_idx}'] = J_m
if use_spike:
    save_dict['tau_spike_center'] = tau_spike_center
np.savez(outfile, **save_dict)
print("Done.", flush=True)
