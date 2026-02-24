"""Diagnostic: eigenvalue structure of Phi_hom for various tau_bot."""
import sys
sys.path.insert(0, '../src')
import numpy as np
from _helpers import make_D_m_funcs
from PythonicDISORT._magnus_propagator import _compute_magnus_propagator
from PythonicDISORT import subroutines

NQuad = 8; N = 4; NLeg = 8
g_l = np.zeros(NLeg); g_l[0] = 1.0
D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos
print('mu_arr_pos:', mu_arr_pos)
print('W:', W, 'sum(W)=', sum(W))

# Build A for isotropic omega=0.2 (constant)
omega = 0.2
D_m = D_m_funcs[0]
tau_mid = 0.5
D_pos = omega * D_m(tau_mid, mu_arr_pos[:, None], mu_arr_pos[None, :])
D_neg = omega * D_m(tau_mid, mu_arr_pos[:, None], -mu_arr_pos[None, :])
DW_pos = D_pos * W[None, :]
DW_neg = D_neg * W[None, :]
alpha = M_inv[:, None] * (DW_pos - np.eye(N))
beta  = M_inv[:, None] * DW_neg
A = np.empty((2*N, 2*N))
A[:N, :N] = -alpha; A[:N, N:] = -beta
A[N:, :N] =  beta;  A[N:, N:] =  alpha

A_eigvals = np.linalg.eigvals(A)
print('A eigenvalues (real):', np.sort(A_eigvals.real))
print('lambda_max:', np.max(np.abs(A_eigvals.real)))

# Check condition of Phi_hom for various tau_bot
for tau_bot, N_steps in [(2.0, 200), (5.0, 500), (32.0, 2000)]:
    def A_func(tau): return A  # constant for this check
    def S_func(tau): return np.zeros(2*N)
    Phi_hom, _ = _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, 2*N)
    eigs = np.linalg.eigvals(Phi_hom)
    eigs_abs = np.sort(np.abs(eigs))
    cond = eigs_abs[-1] / eigs_abs[0] if eigs_abs[0] > 1e-300 else np.inf
    print(f'tau_bot={tau_bot}: cond(Phi_hom)={cond:.3e}, |eigs|={eigs_abs}')

    # Try eigendecomposition
    d_vals, V = np.linalg.eig(Phi_hom)
    idx = np.argsort(np.abs(d_vals))
    d_vals_sorted = d_vals[idx]
    print(f'  Sorted |d|: {np.abs(d_vals_sorted)}')
    print(f'  d_dec (small): {d_vals_sorted[:N]}')
    print(f'  d_gro (large): {d_vals_sorted[N:]}')
    print(f'  product d_dec*d_gro should be ~1: {np.abs(d_vals_sorted[:N]) * np.abs(d_vals_sorted[N:])}')
