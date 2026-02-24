"""Diagnostic: verify which eigenvectors span the decaying input subspace."""
import sys
sys.path.insert(0, '../src')
import numpy as np
import scipy.linalg
from PythonicDISORT import subroutines

NQuad = 8; N = 4
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos

def make_A(omega):
    def D0_iso(tau, mu_i, mu_j):
        return 0.5 * np.ones(np.broadcast_shapes(np.shape(mu_i), np.shape(mu_j)))
    D_pos = omega * D0_iso(0, mu_arr_pos[:, None], mu_arr_pos[None, :])
    D_neg = omega * D0_iso(0, mu_arr_pos[:, None], -mu_arr_pos[None, :])
    DW_pos = D_pos * W[None, :]; DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((NQuad, NQuad))
    A[:N, :N] = -alpha; A[:N, N:] = -beta; A[N:, :N] = beta; A[N:, N:] = alpha
    return A

for omega, tau in [(0.2, 5.0), (0.2, 32.0)]:
    A = make_A(omega)

    # Right eigenvectors: A V = V diag(lam)
    lam_r, V_r = np.linalg.eig(A)
    idx_r = np.argsort(lam_r.real)
    neg_idx_r = idx_r[:N]  # most-negative eigenvalues
    pos_idx_r = idx_r[N:]
    V_neg = V_r[:, neg_idx_r].real  # right eigvecs for negative lam
    V_pos = V_r[:, pos_idx_r].real  # right eigvecs for positive lam

    # Left eigenvectors: A^T E = E diag(lam)
    lam_l, V_l = np.linalg.eig(A.T)
    idx_l = np.argsort(lam_l.real)
    neg_idx_l = idx_l[:N]
    V_l_neg = V_l[:, neg_idx_l].real  # left eigvecs for negative lam (A^T right eigvecs)

    # Exact SVD of exp(A*tau)
    Phi = scipy.linalg.expm(A * tau)
    U_phi, Sig_phi, Vt_phi = np.linalg.svd(Phi)
    print(f"\nomega={omega}, tau={tau}")
    print(f"  Singular values: {np.round(Sig_phi, 4)}")

    # Decaying right singular vectors (last N rows of Vt)
    Vt_dec_exact = Vt_phi[N:, :]  # shape (N, NQuad)
    Vt_gro_exact = Vt_phi[:N, :]  # shape (N, NQuad)

    # Test 1: Does span{Vt_dec_exact} == span{right eigvecs for neg lam}?
    _, s_r, _ = np.linalg.svd(Vt_dec_exact @ V_neg)
    print(f"  Canonical sines (right eigvecs): {np.round(1-s_r**2, 6)}")
    print(f"  -> RIGHT eigvecs span = Vt_dec? {np.allclose(s_r, 1.0, atol=1e-6)}")

    # Test 2: Does span{Vt_dec_exact} == span{left eigvecs for neg lam}?
    _, s_l, _ = np.linalg.svd(Vt_dec_exact @ V_l_neg)
    print(f"  Canonical sines (left  eigvecs): {np.round(1-s_l**2, 6)}")
    print(f"  -> LEFT  eigvecs span = Vt_dec? {np.allclose(s_l, 1.0, atol=1e-6)}")

    # Test 3: Gram-Schmidt of right eigvecs against Vt_gro_exact
    E_dec = V_neg.copy()  # (NQuad, N) right eigvecs for neg lam
    Vt_frozen_r = np.zeros((N, NQuad))
    for j in range(N):
        v = E_dec[:, j].copy()
        for row in Vt_gro_exact:
            v -= np.dot(v, row) * row
        for kk in range(j):
            v -= np.dot(v, Vt_frozen_r[kk]) * Vt_frozen_r[kk]
        nv = np.linalg.norm(v)
        Vt_frozen_r[j] = v / nv if nv > 1e-14 else v

    _, s_rgs, _ = np.linalg.svd(Vt_dec_exact @ Vt_frozen_r.T)
    print(f"  GS(right) vs Vt_dec_exact: {np.round(1-s_rgs**2, 6)}")
    print(f"  -> RIGHT GS correct? {np.allclose(s_rgs, 1.0, atol=1e-6)}")

    # Test 4: Gram-Schmidt of left eigvecs against Vt_gro_exact (current code)
    E_dec_l = V_l_neg.copy()  # (NQuad, N) left eigvecs for neg lam
    Vt_frozen_l = np.zeros((N, NQuad))
    for j in range(N):
        v = E_dec_l[:, j].copy()
        for row in Vt_gro_exact:
            v -= np.dot(v, row) * row
        for kk in range(j):
            v -= np.dot(v, Vt_frozen_l[kk]) * Vt_frozen_l[kk]
        nv = np.linalg.norm(v)
        Vt_frozen_l[j] = v / nv if nv > 1e-14 else v

    _, s_lgs, _ = np.linalg.svd(Vt_dec_exact @ Vt_frozen_l.T)
    print(f"  GS(left)  vs Vt_dec_exact: {np.round(1-s_lgs**2, 6)}")
    print(f"  -> LEFT  GS correct? {np.allclose(s_lgs, 1.0, atol=1e-6)}")
