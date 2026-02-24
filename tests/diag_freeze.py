"""Diagnostic: check sigma_min and freeze thresholds for each test tau."""
import sys
sys.path.insert(0, '../src')
import numpy as np
from PythonicDISORT import subroutines

NQuad = 8
N = NQuad // 2
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos

print("Quadrature points:", mu_arr_pos)

for label, omega in [("iso omega=0.2", 0.2), ("iso omega=0.5", 0.5), ("iso omega~1", 0.9999)]:
    def D0_iso(tau, mu_i, mu_j):
        return 0.5 * np.ones(np.broadcast_shapes(np.shape(mu_i), np.shape(mu_j)))

    def A_func(tau):
        D_pos = omega * D0_iso(tau, mu_arr_pos[:, None], mu_arr_pos[None, :])
        D_neg = omega * D0_iso(tau, mu_arr_pos[:, None], -mu_arr_pos[None, :])
        DW_pos = D_pos * W[None, :]
        DW_neg = D_neg * W[None, :]
        alpha = M_inv[:, None] * (DW_pos - np.eye(N))
        beta  = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N]  = -alpha
        A[:N, N:]  = -beta
        A[N:, :N]  =  beta
        A[N:, N:]  =  alpha
        return A

    A = A_func(0.0)
    lam = np.linalg.eigvals(A)
    lambda_max = max(lam.real)
    print(f"\n{label}: lambda_max = {lambda_max:.4f}")
    for tau_bot, N_steps in [(0.5, 100), (1.0, 100), (2.0, 300), (5.0, 100), (32.0, 100)]:
        sigma_min = np.exp(-lambda_max * tau_bot)
        eps_mach = np.finfo(float).eps
        freeze_1e8 = sigma_min < 1e-8
        freeze_eps = sigma_min < eps_mach
        tau_freeze_1e8 = -np.log(1e-8) / lambda_max if lambda_max > 0 else np.inf
        tau_freeze_eps = -np.log(eps_mach) / lambda_max if lambda_max > 0 else np.inf
        print(f"  tau={tau_bot:5.1f} N={N_steps}: sigma_min={sigma_min:.2e}  "
              f"freeze@1e-8: {'YES tau_f='+f'{tau_freeze_1e8:.2f}' if freeze_1e8 else 'no'}  "
              f"freeze@eps: {'YES tau_f='+f'{tau_freeze_eps:.2f}' if freeze_eps else 'no'}")
