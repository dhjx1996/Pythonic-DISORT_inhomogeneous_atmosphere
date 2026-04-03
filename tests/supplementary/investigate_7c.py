"""
Deep investigation of test_7c: tau-varying omega+g with BDRF.

Questions to answer:
1. Does Riccati converge as tol decreases? To what?
2. Does multilayer pydisort converge as NLayers increases? To what?
3. Do they converge to the SAME answer?
4. Where (which mu, which phi, which Fourier mode) is the error concentrated?
"""
import sys
from pathlib import Path
from math import pi
import numpy as np

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

from _helpers import multilayer_pydisort_toa_full_phi, PHI_VALUES
from pydisort_riccati_jax import pydisort_riccati_jax

NQuad = 8
NLeg = NQuad
N = NQuad // 2

tau_bot = 1.0
mu0, I0, phi0 = 0.5, 1.0, 0.0
rho = 0.3
BDRF = [rho / pi]
omega_func = lambda tau: 0.90 + (0.60 - 0.90) * tau / tau_bot
g_func = lambda tau: 0.70 + (0.30 - 0.70) * tau / tau_bot

def Leg_coeffs_func(tau):
    g = g_func(tau)
    return g ** np.arange(NLeg)

def u_phi_array(func, *args):
    return np.column_stack([func(*args, phi)[:N] for phi in PHI_VALUES])

# ---- Riccati at multiple tolerances ----
print("=" * 60)
print("RICCATI CONVERGENCE (varying tol)")
print("=" * 60)
ric_results = {}
for tol in [1e-3, 1e-5, 1e-8, 1e-10]:
    _, _, _, u_func, tg = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
        tol=tol, BDRF_Fourier_modes=BDRF,
    )
    u = u_phi_array(u_func)
    ric_results[tol] = u
    print(f"  tol={tol:.0e}: steps={len(tg)-1}")

# Pairwise Riccati differences
print("\nRiccati self-convergence (max rel diff between successive tols):")
tols = sorted(ric_results.keys())
ref_scale = max(float(np.max(np.abs(ric_results[tols[-1]]))), 1e-8)
for i in range(1, len(tols)):
    diff = float(np.max(np.abs(ric_results[tols[i]] - ric_results[tols[i-1]]))) / ref_scale
    print(f"  tol={tols[i-1]:.0e} vs {tols[i]:.0e}: {diff:.4e}")

# ---- Multilayer pydisort at multiple layer counts ----
print("\n" + "=" * 60)
print("MULTILAYER PYDISORT CONVERGENCE (varying NLayers)")
print("=" * 60)
ml_results = {}
for NL in [50, 200, 500, 2000, 5000]:
    _, _, uf = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, NL, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )
    u = u_phi_array(uf, 0)
    ml_results[NL] = u
    print(f"  NLayers={NL}")

print("\nMultilayer self-convergence (max rel diff between successive NLayers):")
nls = sorted(ml_results.keys())
for i in range(1, len(nls)):
    diff = float(np.max(np.abs(ml_results[nls[i]] - ml_results[nls[i-1]]))) / ref_scale
    print(f"  NL={nls[i-1]} vs {nls[i]}: {diff:.4e}")

# ---- Cross-comparison: converged Riccati vs converged pydisort ----
print("\n" + "=" * 60)
print("CROSS-METHOD COMPARISON")
print("=" * 60)
u_ric = ric_results[1e-10]
u_ml = ml_results[5000]

cross_diff = float(np.max(np.abs(u_ric - u_ml))) / ref_scale
print(f"Riccati(tol=1e-10) vs Multilayer(NL=5000): max rel diff = {cross_diff:.4e}")

# Per-phi breakdown
print("\nPer-phi breakdown (Riccati tol=1e-10 vs Multilayer NL=5000):")
for j, phi in enumerate(PHI_VALUES):
    col_ref = u_ml[:, j]
    col_ric = u_ric[:, j]
    col_scale = max(float(np.max(np.abs(col_ref))), 1e-8)
    col_diff = float(np.max(np.abs(col_ric - col_ref))) / col_scale
    # Per-mu breakdown
    for i in range(N):
        mu_diff = abs(float(col_ric[i] - col_ref[i]))
        mu_rel = mu_diff / max(abs(float(col_ref[i])), 1e-8)
        marker = " <---" if mu_rel > 1e-3 else ""
        print(f"  phi={phi:.4f}, mu[{i}]: ric={col_ric[i]:.8e}  ml={col_ref[i]:.8e}  rel_diff={mu_rel:.4e}{marker}")

# ---- Same problem WITHOUT BDRF ----
print("\n" + "=" * 60)
print("CONTROL: SAME PROBLEM WITHOUT BDRF")
print("=" * 60)
_, _, _, u_func_nb, _ = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-10,
)
u_ric_nb = u_phi_array(u_func_nb)

_, _, uf_nb = multilayer_pydisort_toa_full_phi(
    tau_bot, omega_func, Leg_coeffs_func, 5000, NQuad, NLeg,
    mu0, I0, phi0,
)
u_ml_nb = u_phi_array(uf_nb, 0)

nb_scale = max(float(np.max(np.abs(u_ml_nb))), 1e-8)
nb_diff = float(np.max(np.abs(u_ric_nb - u_ml_nb))) / nb_scale
print(f"Without BDRF: Riccati(1e-12) vs Multilayer(5000): max rel diff = {nb_diff:.4e}")
