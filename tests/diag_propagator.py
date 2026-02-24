"""Diagnostic: trace the propagator step-by-step for failing test cases."""
import sys
sys.path.insert(0, '../src')
import numpy as np
import scipy.linalg
from PythonicDISORT import subroutines

NQuad = 8; N = 4
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos

def make_iso_A(omega):
    D_pos = 0.5 * omega * np.ones((N, N))
    D_neg = 0.5 * omega * np.ones((N, N))
    DW_pos = D_pos * W[None, :]; DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((NQuad, NQuad))
    A[:N, :N] = -alpha; A[:N, N:] = -beta; A[N:, :N] = beta; A[N:, N:] = alpha
    return A

# Check Sigma evolution for tests 2c (tau=5, omega=0.5) and 1d (tau=32, omega=0.2)
for omega, tau_bot, N_steps, label in [(0.5, 5.0, 100, "2c: tau=5, omega=0.5"),
                                        (0.2, 32.0, 100, "1d: tau=32, omega=0.2")]:
    A = make_iso_A(omega)
    lam_A = np.linalg.eigvals(A)
    lam_max = max(lam_A.real)
    lam_sorted = sorted(lam_A.real)
    print(f"\n=== {label} ===")
    print(f"  Eigenvalues: {np.round(lam_sorted, 4)}")
    print(f"  lambda_max = {lam_max:.4f}, lambda_min_pos = {min(l for l in lam_sorted if l>0):.4f}")
    tau_freeze = -np.log(np.finfo(float).eps) / lam_max if lam_max > 0 else np.inf
    print(f"  tau_freeze (eps_mach) = {tau_freeze:.3f}")

    h = tau_bot / N_steps
    Phi_step = scipy.linalg.expm(A * h)  # constant for isotropic

    U_live = np.eye(NQuad)
    Sigma_live = np.ones(NQuad)
    Vt_live = np.eye(NQuad)
    r_f = 0
    Vt_frozen = np.zeros((0, NQuad))

    freeze_steps = []
    for k in range(N_steps):
        B = Phi_step @ U_live
        G = B * Sigma_live[None, :]
        U_new, Sigma_new, Wt = np.linalg.svd(G, full_matrices=False)
        Vt_live_new = Wt @ Vt_live

        new_frozen_mask = Sigma_new < np.finfo(float).eps
        if np.any(new_frozen_mask):
            n_new = int(new_frozen_mask.sum())
            r_f += n_new
            freeze_steps.append((k, n_new, r_f))
            keep = ~new_frozen_mask
            U_live = U_new[:, keep]; Sigma_live = Sigma_new[keep]
            Vt_live = Vt_live_new[keep, :]
        else:
            U_live = U_new; Sigma_live = Sigma_new; Vt_live = Vt_live_new

    print(f"  Freeze events (step, n_new, total_r_f): {freeze_steps}")
    print(f"  Final: r_live={len(Sigma_live)}, r_f={r_f}, r_dec_live={len(Sigma_live)-N}")
    print(f"  Final Sigma_live: {np.round(Sigma_live, 4)}")

    # Compare final Vt_live (gro rows) vs exact SVD
    Phi_full = scipy.linalg.expm(A * tau_bot)
    U_exact, Sig_exact, Vt_exact = np.linalg.svd(Phi_full)
    Vt_gro_exact = Vt_exact[:N, :]

    # Compare subspaces
    r_gro = int(np.sum(Sigma_live > 1.0))
    Vt_live_gro = Vt_live[:r_gro, :]  # growing live rows
    _, s, _ = np.linalg.svd(Vt_live_gro @ Vt_gro_exact.T)
    print(f"  Vt_live_gro vs exact Vt_gro: canonical cosines = {np.round(s, 8)}")
    print(f"  (should all be ~1.0 if converged)")

    # What does the code's Vt_frozen look like vs exact Vt_dec?
    if r_f > 0:
        # Rerun to get Vt_frozen (re-compute at last step)
        lam, E_left = np.linalg.eig(A.T)  # current code uses A.T (left eigvecs)
        sort_idx = np.argsort(lam.real)
        E_dec = E_left[:, sort_idx[:r_f]].real
        Vt_frozen_code = np.zeros((r_f, NQuad))
        Vt_gro_final = Vt_live[:N, :]  # assuming r_gro = N
        for j in range(r_f):
            v = E_dec[:, j].copy()
            for row in Vt_live:
                v -= np.dot(v, row) * row
            for kk in range(j):
                v -= np.dot(v, Vt_frozen_code[kk]) * Vt_frozen_code[kk]
            nv = np.linalg.norm(v)
            Vt_frozen_code[j] = v / nv if nv > 1e-14 else v

        Vt_dec_exact = Vt_exact[N:, :]  # exact decaying rows
        Vt_full = np.vstack([Vt_live, Vt_frozen_code])
        print(f"  Code Vt_frozen vs exact Vt_dec (canonical cosines of subspaces):")
        _, s_frozen, _ = np.linalg.svd(Vt_frozen_code @ Vt_dec_exact.T)
        print(f"    {np.round(s_frozen, 8)}")
        print(f"    Full Vt orthogonality error: {np.round(np.linalg.norm(Vt_full @ Vt_full.T - np.eye(NQuad)), 2)}")
