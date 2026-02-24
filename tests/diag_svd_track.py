"""Track step-by-step SVD vs exact SVD of accumulated propagator, and check test 3c."""
import sys
sys.path.insert(0, '../src')
import numpy as np
import scipy.linalg
from PythonicDISORT import subroutines

NQuad = 8; N = 4
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
M_inv = 1.0 / mu_arr_pos

def make_A_iso(omega):
    D_pos = 0.5 * omega * np.ones((N, N))
    D_neg = 0.5 * omega * np.ones((N, N))
    DW_pos = D_pos * W[None, :]; DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((NQuad, NQuad))
    A[:N, :N] = -alpha; A[:N, N:] = -beta; A[N:, :N] = beta; A[N:, N:] = alpha
    return A

# Test: exact SVD of exp(A*tau) for test 2c and check singular values
print("=== Exact SVD singular values ===")
for label, omega, tau, g_l_use in [
    ("2c Rayleigh om=0.5 tau=5", 0.5, 5.0, None),
    ("2d Rayleigh om~1  tau=5", 1-1e-6, 5.0, None),
    ("1d iso    om=0.2 tau=32", 0.2, 32.0, None),
    ("3c HG g=0.5 om=0.8 tau=5", 0.8, 5.0, 0.5),
]:
    if g_l_use is None:
        # Rayleigh-like: g_l[0]=1, g_l[2]=0.1
        g_l = np.zeros(8); g_l[0]=1.0; g_l[2]=0.1
    else:
        g_l = g_l_use ** np.arange(8)

    # Build A for m=0
    from numpy.polynomial.legendre import legval
    def make_A_from_gl(omega_v, g_l_v):
        D_pos = np.zeros((N, N)); D_neg = np.zeros((N, N))
        for l in range(8):
            c = np.zeros(l+1); c[l] = 1
            Pl_pos = np.array([legval(mu, c) for mu in mu_arr_pos])
            Pl_neg = np.array([legval(-mu, c) for mu in mu_arr_pos])
            D_pos += (2*l+1) * g_l_v[l] * np.outer(Pl_pos, Pl_pos) * 0.5
            D_neg += (2*l+1) * g_l_v[l] * np.outer(Pl_pos, Pl_neg) * 0.5
        D_pos *= omega_v; D_neg *= omega_v
        DW_pos = D_pos * W[None, :]; DW_neg = D_neg * W[None, :]
        alpha = M_inv[:, None] * (DW_pos - np.eye(N))
        beta  = M_inv[:, None] * DW_neg
        A = np.empty((NQuad, NQuad))
        A[:N, :N] = -alpha; A[:N, N:] = -beta; A[N:, :N] = beta; A[N:, N:] = alpha
        return A

    A = make_A_from_gl(omega, g_l)
    lam = np.linalg.eigvals(A)
    lam_max = max(lam.real)
    tau_freeze = -np.log(np.finfo(float).eps) / lam_max if lam_max > 0 else np.inf
    Phi = scipy.linalg.expm(A * tau)
    _, sig, _ = np.linalg.svd(Phi)
    print(f"\n  {label}")
    print(f"    lambda_max={lam_max:.4f}, tau_freeze={tau_freeze:.3f}")
    print(f"    Exact SVD sigmas: {sig}")
    print(f"    All >1? {np.all(sig > 1.0)}")

# Track step-by-step vs exact for test 2c
print("\n\n=== Step-by-step vs exact SVD tracking for test 2c ===")
g_l = np.zeros(8); g_l[0]=1.0; g_l[2]=0.1
from numpy.polynomial.legendre import legval
def make_A_rayleigh(omega_v):
    D_pos = np.zeros((N, N)); D_neg = np.zeros((N, N))
    for l in range(8):
        c = np.zeros(l+1); c[l] = 1
        Pl_pos = np.array([legval(mu, c) for mu in mu_arr_pos])
        Pl_neg = np.array([legval(-mu, c) for mu in mu_arr_pos])
        D_pos += (2*l+1) * g_l[l] * np.outer(Pl_pos, Pl_pos) * 0.5
        D_neg += (2*l+1) * g_l[l] * np.outer(Pl_pos, Pl_neg) * 0.5
    D_pos *= omega_v; D_neg *= omega_v
    DW_pos = D_pos * W[None, :]; DW_neg = D_neg * W[None, :]
    alpha = M_inv[:, None] * (DW_pos - np.eye(N))
    beta  = M_inv[:, None] * DW_neg
    A = np.empty((NQuad, NQuad))
    A[:N, :N] = -alpha; A[:N, N:] = -beta; A[N:, :N] = beta; A[N:, N:] = alpha
    return A

A = make_A_rayleigh(0.5)
Phi_step = scipy.linalg.expm(A * 0.05)  # h=0.05 for tau=5, N_steps=100
U_live = np.eye(NQuad); Sigma_live = np.ones(NQuad); Vt_live = np.eye(NQuad)
print(f"{'step':>5} {'sigma_min_step':>16} {'sigma_min_exact':>16} {'ratio':>10}")
Phi_acc = np.eye(NQuad)
for k in range(53):  # track through the first freeze
    # Step-by-step update
    G = Phi_step @ U_live * Sigma_live[None, :]
    U_new, Sigma_new, Wt = np.linalg.svd(G, full_matrices=False)
    Vt_live_new = Wt @ Vt_live
    # Exact update
    Phi_acc = Phi_step @ Phi_acc
    _, sig_exact, _ = np.linalg.svd(Phi_acc)
    sigma_min_step = Sigma_new.min()
    sigma_min_exact = sig_exact.min()
    if k < 5 or k > 45 or sigma_min_step < 1e-10:
        print(f"{k:>5} {sigma_min_step:>16.6e} {sigma_min_exact:>16.6e} {sigma_min_step/sigma_min_exact:>10.4f}")
    U_live = U_new; Sigma_live = Sigma_new; Vt_live = Vt_live_new
