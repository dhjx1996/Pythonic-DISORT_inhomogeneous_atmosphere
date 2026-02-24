"""Diagnostic: check Phi_hom conditioning for test 6c (HG g=0.5, variable omega, tau_bot=pi)."""
import sys
sys.path.insert(0, '../src')
import numpy as np
from math import pi
from _helpers import make_D_m_funcs
from PythonicDISORT._magnus_propagator import _compute_magnus_propagator
from PythonicDISORT import subroutines

NQuad = 8; N = 4; NLeg = 8
g = 0.5
g_l = g ** np.arange(NLeg)
D_m_funcs = make_D_m_funcs(g_l, NLeg, NQuad)
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos
omega_func = lambda tau: 0.70 + 0.20 * np.sin(2 * tau)

D_m = D_m_funcs[0]  # m=0 mode
tau_bot = pi

def A_func(tau):
    omega = omega_func(tau)
    D_pos = omega * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((2*N, 2*N))
    A[:N, :N] = -alpha; A[:N, N:] = -beta
    A[N:, :N] =  beta;  A[N:, N:] =  alpha
    return A

# Eigenvalues of A at midpoint
A_mid = A_func(tau_bot / 2)
A_eigvals = np.linalg.eigvals(A_mid)
lambda_max = np.max(np.abs(A_eigvals.real))
print(f'A eigenvalues at tau_mid={tau_bot/2:.3f}: {np.sort(A_eigvals.real)}')
print(f'lambda_max = {lambda_max:.4f}')
print(f'Expected smallest |d| at tau_bot={tau_bot:.4f}: exp(-{lambda_max:.2f}*{tau_bot:.4f}) = {np.exp(-lambda_max*tau_bot):.3e}')
print(f'Expected cond(Phi_hom) = exp(2*{lambda_max:.2f}*{tau_bot:.4f}) = {np.exp(2*lambda_max*tau_bot):.3e}')
print()

# Check actual Phi_hom for various N_steps
def S_func(tau): return np.zeros(2*N)

for N_steps in [200, 2000]:
    Phi_hom, _ = _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, 2*N)
    eigs = np.linalg.eigvals(Phi_hom)
    eigs_abs = np.sort(np.abs(eigs))
    cond = eigs_abs[-1] / eigs_abs[0] if eigs_abs[0] > 1e-300 else np.inf
    print(f'N_steps={N_steps}: cond(Phi_hom)={cond:.3e}')
    print(f'  |eigs|: {eigs_abs}')
    idx = np.argsort(np.abs(eigs))
    eigs_sorted = eigs[idx]
    print(f'  d_dec (should be small): {eigs_sorted[:N]}')
    print(f'  d_gro (should be large): {eigs_sorted[N:]}')
    print()

# Also check isotropic omega=0.5 (worst case for this test's range)
omega_const = 0.5
def A_func_const(tau):
    return A_func.__code__.co_consts  # not used
# Build A for omega=0.5 directly
def A_omega(tau, om=0.5):
    D_pos = om * D_m(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = om * D_m(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]
    DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((2*N, 2*N))
    A[:N, :N] = -alpha; A[:N, N:] = -beta
    A[N:, :N] =  beta;  A[N:, N:] =  alpha
    return A

A_const = A_omega(0.0, om=0.5)
A_eigs_const = np.linalg.eigvals(A_const)
lmax_const = np.max(np.abs(A_eigs_const.real))
print(f'For constant omega=0.5, HG g=0.5:')
print(f'  lambda_max = {lmax_const:.4f}')
print(f'  exp(-lambda_max * pi) = {np.exp(-lmax_const * pi):.3e}')
print(f'  cond at tau_bot=pi: {np.exp(2*lmax_const*pi):.3e}')
