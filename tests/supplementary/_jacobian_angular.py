"""Angular mode analysis of J_g = du_ToA/dg(tau).

Usage: python _jacobian_angular.py <smooth|spike|absorbing> <outfile.npz> [tau_spike]

  tau_spike: spike centre (default 15.0); only used when case == "spike".

Perturbs the HG asymmetry parameter g (not g1 in isolation), coupling all
Legendre orders via g_l = g^l.  Analyses (some skipped for absorbing case):
  - Shifted-Legendre mode decomposition
  - BoA-anchored greedy column selection (Definition A)
  - Regional SVD / sliding-window rank (Definition B)
  - Local resolving length (Definition C)
  - Cosine similarity with BoA column
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

case = sys.argv[1]
outfile = sys.argv[2]
assert case in ("smooth", "spike", "absorbing"), f"Unknown case: {case}"

# ================================================================
# Problem parameters
# ================================================================
tau_bot = 30.0
g_top, g_bot = 0.865, 0.820
NQuad = 16; NLeg = NQuad; N = NQuad // 2
mu0 = 0.5; I0 = 1.0; phi0 = 0.0
tol = 1e-3
K = 61

use_spike = (case == "spike")
tau_spike_center = float(sys.argv[3]) if (use_spike and len(sys.argv) > 3) else 15.0
sigma_spike = 0.5
d_g_spike = 0.04

if case == "absorbing":
    # Absorbing adiabatic cloud with drizzle (from _jacobian_ad.py)
    omega_top, omega_bot = 0.85, 0.96
    d_omega_spike = -0.15
    tau_spike_center = 15.0  # drizzle spike location (fixed for absorbing)

    def omega_nominal(tau):
        return (omega_top + (omega_bot - omega_top) * tau / tau_bot
                + d_omega_spike * jnp.exp(-0.5 * ((tau - tau_spike_center) / sigma_spike)**2))

    def g_nominal(tau):
        return (g_top + (g_bot - g_top) * tau / tau_bot
                + d_g_spike * jnp.exp(-0.5 * ((tau - tau_spike_center) / sigma_spike)**2))

    print(f"=== Angular Jacobian analysis: absorbing ===", flush=True)
    print(f"  omega=[{omega_top},{omega_bot}], g=[{g_top},{g_bot}], "
          f"d_omega_spike={d_omega_spike}, d_g_spike={d_g_spike}", flush=True)
else:
    # Conservative cases (smooth / spike)
    omega_val = 1.0 - 1e-6

    def omega_nominal(tau):
        return omega_val + 0.0 * tau

    def g_nominal(tau):
        g_lin = g_top + (g_bot - g_top) * tau / tau_bot
        if use_spike:
            g_lin = g_lin + d_g_spike * jnp.exp(
                -0.5 * ((tau - tau_spike_center) / sigma_spike)**2)
        return g_lin

    print(f"=== Angular Jacobian analysis: {case} ===", flush=True)
    print(f"  omega={omega_val}, g=[{g_top},{g_bot}], "
          f"spike={use_spike}" + (f", tau_spike={tau_spike_center}" if use_spike else ""),
          flush=True)

def Leg_coeffs_nominal(tau):
    return g_nominal(tau) ** jnp.arange(NLeg)

# ================================================================
# Step 1: Nominal solve
# ================================================================
print("\nStep 1: Nominal solve ...", flush=True)
t0 = time.time()
mu_arr, flux_up, u0_nom, u_func_nom, tau_grid_nom = pydisort_riccati_jax(
    tau_bot, omega_nominal, Leg_coeffs_nominal, NQuad, mu0, I0, phi0, tol=tol)
tau_grid_np = np.asarray(tau_grid_nom)
print(f"  {time.time()-t0:.1f}s, {len(tau_grid_np)} grid points", flush=True)

n_boa = np.sum(tau_grid_np > 25)
n_toa = np.sum(tau_grid_np < 5)
print(f"  Grid: {n_toa} ToA (<5), {n_boa} BoA (>25)", flush=True)

# ================================================================
# Step 2: Setup
# ================================================================
mu_arr_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos = jnp.array(mu_arr_pos_np)
W_jax = jnp.array(W_np)
M_inv = 1.0 / mu_arr_pos
I0_div_4pi = I0 / (4 * pi)
tau_nodes = jnp.linspace(0.0, tau_bot, K)
tau_np = np.asarray(tau_nodes)
leg_data_m0 = _precompute_legendre(0, NLeg, mu_arr_pos, mu0)

# ================================================================
# Step 3: AD-traceable forward model
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


def solve_m0(delta_omega_vec, delta_g_vec):
    """m=0: (delta_omega, delta_g) each (K,) -> u0_ToA (N,)."""
    def omega_func(tau):
        return omega_nominal(tau) + jnp.interp(tau, tau_nodes, delta_omega_vec)
    def Leg_coeffs_func(tau):
        g_pert = g_nominal(tau) + jnp.interp(tau, tau_nodes, delta_g_vec)
        return g_pert ** jnp.arange(NLeg)

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

# ================================================================
# Step 4: Compute J_g via jacrev
# ================================================================
zero_K = jnp.zeros(K)

print("\nStep 4: J_g = jacrev(solve_m0, argnums=1) ...", flush=True)
t0 = time.time()
J_g = jnp.asarray(jax.jacrev(solve_m0, argnums=1)(zero_K, zero_K))
J_g_np = np.asarray(J_g)
del J_g  # free GPU memory
print(f"  {time.time()-t0:.1f}s, shape {J_g_np.shape}", flush=True)

# L2 sensitivity
sens_g = np.sqrt(np.sum(J_g_np**2, axis=0))
print(f"  ||J_g|| range: [{sens_g.min():.4e}, {sens_g.max():.4e}]", flush=True)

# ================================================================
# Step 5: FD validation at a few tau-nodes (skip for absorbing)
# ================================================================
if case != "absorbing":
    print("\nStep 5: Finite-difference validation ...", flush=True)
    eps_fd = 1e-5
    fd_indices = [0, 10, 30]  # tau=0, tau~5, tau=15
    for idx in fd_indices:
        pert = np.zeros(K); pert[idx] = eps_fd
        u_plus  = np.asarray(solve_m0(zero_K, jnp.array(pert)))
        pert[idx] = -eps_fd
        u_minus = np.asarray(solve_m0(zero_K, jnp.array(pert)))
        J_fd_col = (u_plus - u_minus) / (2 * eps_fd)
        J_ad_col = J_g_np[:, idx]
        rel_err = np.linalg.norm(J_fd_col - J_ad_col) / (np.linalg.norm(J_ad_col) + 1e-30)
        print(f"  tau={tau_np[idx]:.1f}: ||FD-AD||/||AD|| = {rel_err:.2e}", flush=True)
else:
    print("\nStep 5: FD validation — skipped (absorbing)", flush=True)

# ================================================================
# Step 6: Angular mode decomposition (shifted Legendre)
# ================================================================
print("\nStep 6: Angular mode decomposition ...", flush=True)

# Build shifted Legendre matrix: P_shifted[l, i] = P_l(2*mu_i - 1)
mu_shifted = 2.0 * mu_arr_pos_np - 1.0  # map [0,1] -> [-1,1]
P_shifted = np.zeros((N, N))
for l in range(N):
    coeffs = np.zeros(l + 1)
    coeffs[l] = 1.0
    P_shifted[l, :] = np.polynomial.legendre.legval(mu_shifted, coeffs)

# Verify orthogonality: P @ diag(W) @ P^T should be diag(1/(2l+1))
ortho_check = P_shifted @ np.diag(W_np) @ P_shifted.T
ortho_expected = np.diag(1.0 / (2 * np.arange(N) + 1))
ortho_err = np.max(np.abs(ortho_check - ortho_expected))
print(f"  Orthogonality error: {ortho_err:.2e}", flush=True)

# Weighted projection: c_l(j) = (2l+1) * sum_i J[i,j] * P_shifted[l,i] * W[i]
# Vectorized: modes[l, j] = (2l+1) * (P_shifted[l,:] * W) @ J_g_np
PW = P_shifted * W_np[None, :]  # (N, N)
modes = np.zeros((N, K))
for l in range(N):
    modes[l, :] = (2 * l + 1) * (PW[l, :] @ J_g_np)

# Print mode amplitudes by region
regions = [
    ("ToA  (tau<5)",     tau_np < 5),
    ("Mid  (5<tau<13)",  (tau_np >= 5) & (tau_np < 13)),
    ("Mid2 (13<tau<17)", (tau_np >= 13) & (tau_np < 17)),
    ("Deep (17<tau<25)", (tau_np >= 17) & (tau_np < 25)),
    ("BoA  (tau>=25)",   tau_np >= 25),
]

print("\n  Mean |c_l| by region:")
print(f"  {'Region':<20s}", end="")
for l in range(min(6, N)):
    print(f"  l={l:<8d}", end="")
print()
for name, mask in regions:
    print(f"  {name:<20s}", end="")
    for l in range(min(6, N)):
        print(f"  {np.abs(modes[l, mask]).mean():.3e}", end="")
    print()

# Normalized: |c_l/c_1| by region
print("\n  Mean |c_l/c_1| by region:")
print(f"  {'Region':<20s}", end="")
for l in range(min(6, N)):
    print(f"  l={l:<8d}", end="")
print()
for name, mask in regions:
    c1_mean = np.abs(modes[1, mask]).mean()
    print(f"  {name:<20s}", end="")
    for l in range(min(6, N)):
        ratio = np.abs(modes[l, mask]).mean() / (c1_mean + 1e-30)
        print(f"  {ratio:.3e}", end="")
    print()

# ================================================================
# Step 7: Definition A — BoA-anchored greedy column selection (skip for absorbing)
# ================================================================
if case != "absorbing":
    print("\nStep 7: Definition A — BoA-anchored greedy selection ...", flush=True)

    def greedy_column_selection(J, anchor_idx):
        """Greedy column selection starting from anchor.
        Returns (selected_indices, max_residual_at_each_step)."""
        Nn, Kk = J.shape
        max_sel = min(Nn, Kk)
        selected = [anchor_idx]
        # Orthonormal basis
        q0 = J[:, anchor_idx].copy()
        q0 = q0 / np.linalg.norm(q0)
        Q = q0[:, None]  # (N, 1)

        step_residuals = []
        for step in range(max_sel - 1):
            proj = Q @ (Q.T @ J)  # (N, K)
            resid_norms = np.sqrt(np.sum((J - proj)**2, axis=0))
            resid_norms[selected] = 0.0
            best = np.argmax(resid_norms)
            step_residuals.append(resid_norms[best])
            if resid_norms[best] < 1e-12:
                break
            selected.append(int(best))
            # Gram-Schmidt
            new_col = J[:, best] - proj[:, best]
            new_col = new_col / np.linalg.norm(new_col)
            Q = np.column_stack([Q, new_col])
        return selected, step_residuals

    sel_order, sel_resid = greedy_column_selection(J_g_np, K - 1)
    print(f"  Selected {len(sel_order)} columns (anchor=BoA, idx={K-1})")
    print(f"  Selection order (tau values):")
    for rank, idx in enumerate(sel_order[:15]):
        resid = sel_resid[rank - 1] if rank > 0 else np.linalg.norm(J_g_np[:, idx])
        print(f"    rank {rank}: idx={idx:2d}, tau={tau_np[idx]:5.1f}, residual={resid:.3e}")
else:
    sel_order, sel_resid = [], []
    print("\nStep 7: Definition A — skipped (absorbing)", flush=True)

# Cosine similarity with BoA column (all cases)
print("\n  Cosine similarity with BoA:", flush=True)
boa_col = J_g_np[:, -1]
boa_norm = np.linalg.norm(boa_col)
cos_sim = np.zeros(K)
for j in range(K):
    col = J_g_np[:, j]
    cos_sim[j] = abs(np.dot(col, boa_col)) / (np.linalg.norm(col) * boa_norm + 1e-30)

for name, mask in regions:
    v = cos_sim[mask]
    print(f"    {name}: mean={v.mean():.4f}  min={v.min():.4f}  max={v.max():.4f}")

# ================================================================
# Step 8: Definition B — Regional SVD / intrinsic dimensionality
# ================================================================
print("\nStep 8: Definition B — Regional SVD ...", flush=True)

regions_svd = [
    ("ToA (tau<5)",      tau_np < 5),
    ("Mid (5<=tau<25)",  (tau_np >= 5) & (tau_np < 25)),
    ("BoA (tau>=25)",    tau_np >= 25),
]
for name, mask in regions_svd:
    J_block = J_g_np[:, mask]
    n_cols = J_block.shape[1]
    S = np.linalg.svd(J_block, compute_uv=False)
    print(f"  {name} ({n_cols} cols):")
    print(f"    Singular values: {', '.join(f'{s:.3e}' for s in S[:min(8,len(S))])}")
    for thresh_pct in [10, 1, 0.1]:
        rank = int(np.sum(S > (thresh_pct / 100) * S[0]))
        print(f"    Rank (>{thresh_pct}% of sigma_1): {rank}")

# Sliding-window rank
w = 5  # half-window
sliding_rank_1pct = np.full(K, np.nan)
sliding_rank_10pct = np.full(K, np.nan)
for j in range(K):
    lo = max(0, j - w)
    hi = min(K, j + w + 1)
    if hi - lo < 2:
        continue
    S = np.linalg.svd(J_g_np[:, lo:hi], compute_uv=False)
    sliding_rank_1pct[j] = np.sum(S > 0.01 * S[0])
    sliding_rank_10pct[j] = np.sum(S > 0.10 * S[0])

print(f"\n  Sliding-window rank (w={w}, 1% threshold):")
for name, mask in regions:
    v = sliding_rank_1pct[mask]
    v = v[~np.isnan(v)]
    if len(v) > 0:
        print(f"    {name}: mean={v.mean():.1f}  min={v.min():.0f}  max={v.max():.0f}")

# ================================================================
# Step 9: Definition C — Local resolving length (skip for absorbing)
# ================================================================
if case != "absorbing":
    print("\nStep 9: Definition C — Local resolving length ...", flush=True)

    dtau = float(tau_np[1] - tau_np[0])
    dJ = np.diff(J_g_np, axis=1) / dtau  # (N, K-1)
    J_mid = 0.5 * (J_g_np[:, :-1] + J_g_np[:, 1:])  # midpoints
    J_mid_norm = np.sqrt(np.sum(J_mid**2, axis=0))
    dJ_norm = np.sqrt(np.sum(dJ**2, axis=0))
    L_res = J_mid_norm / (dJ_norm + 1e-30)
    tau_mid = 0.5 * (tau_np[:-1] + tau_np[1:])

    print(f"  L_res range: [{L_res.min():.2f}, {L_res.max():.2f}]")
    print(f"  Implied grid density (1/L_res) range: [{1/L_res.max():.4f}, {1/L_res.min():.4f}]")
    print(f"\n  L_res by region:")
    for name, mask_full in regions:
        mask_mid = mask_full[:-1] & mask_full[1:]
        if mask_mid.sum() == 0:
            mask_mid = mask_full[:-1] | mask_full[1:]
        v = L_res[mask_mid]
        if len(v) > 0:
            print(f"    {name}: mean={v.mean():.2f}  min={v.min():.2f}  max={v.max():.2f}")

    print(f"\n  Implied grid points by region (integral of 1/L_res * dtau):")
    for name, mask_full in regions:
        mask_mid = mask_full[:-1] & mask_full[1:]
        if mask_mid.sum() == 0:
            mask_mid = mask_full[:-1] | mask_full[1:]
        n_implied = np.sum(dtau / L_res[mask_mid])
        tau_range = tau_np[mask_full]
        print(f"    {name}: {n_implied:.1f} points over [{tau_range.min():.0f}, {tau_range.max():.0f}]")
else:
    L_res = np.array([])
    tau_mid = np.array([])
    print("\nStep 9: Definition C — skipped (absorbing)", flush=True)

# ================================================================
# Step 10: Comparison with solver's adaptive tau_grid (skip for absorbing)
# ================================================================
if case != "absorbing":
    print("\nStep 10: Solver tau_grid comparison ...", flush=True)

    solver_diffs = np.diff(tau_grid_np)
    solver_density_tau = 0.5 * (tau_grid_np[:-1] + tau_grid_np[1:])
    solver_density = 1.0 / solver_diffs

    L_res_at_solver = np.interp(solver_density_tau, tau_mid, L_res)
    retrieval_density = 1.0 / L_res_at_solver

    print(f"  Solver grid: {len(tau_grid_np)} points")
    print(f"  Regions where solver is denser than retrieval needs:")
    over_resolved = solver_density > 2 * retrieval_density
    if np.any(over_resolved):
        tau_over = solver_density_tau[over_resolved]
        print(f"    tau in [{tau_over.min():.1f}, {tau_over.max():.1f}] "
              f"({np.sum(over_resolved)} intervals)")
    else:
        print(f"    None")

    print(f"  Regions where retrieval needs more than solver provides:")
    under_resolved = retrieval_density > 2 * solver_density
    if np.any(under_resolved):
        tau_under = solver_density_tau[under_resolved]
        print(f"    tau in [{tau_under.min():.1f}, {tau_under.max():.1f}] "
              f"({np.sum(under_resolved)} intervals)")
    else:
        print(f"    None")
else:
    print("\nStep 10: Solver tau_grid comparison — skipped (absorbing)", flush=True)

# ================================================================
# Step 11: Save
# ================================================================
print(f"\nSaving to {outfile} ...", flush=True)
save_dict = dict(
    case=case,
    tau_nodes=tau_np,
    J_g=J_g_np,
    sensitivity_g=sens_g,
    modes=modes,
    cos_sim_boa=cos_sim,
    selection_order=np.array(sel_order),
    selection_residuals=np.array(sel_resid),
    sliding_rank_1pct=sliding_rank_1pct,
    sliding_rank_10pct=sliding_rank_10pct,
    L_res=L_res,
    tau_mid=tau_mid,
    tau_grid_nominal=tau_grid_np,
    mu_arr=mu_arr_pos_np,
    W=W_np,
    P_shifted=P_shifted,
    u0_nominal=np.asarray(u0_nom),
)
if use_spike:
    save_dict['tau_spike_center'] = tau_spike_center
np.savez(outfile, **save_dict)
print("Done.", flush=True)
