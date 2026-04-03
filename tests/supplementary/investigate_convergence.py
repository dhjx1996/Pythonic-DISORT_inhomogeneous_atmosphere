"""
Investigate O(h^2) convergence of multilayer pydisort toward Riccati reference.

Uses test_10a scenario: tau=10, adiabatic cloud, BDRF rho=0.05.
Riccati reference at tol=1e-8 to minimize reference contamination.
"""
import sys
from pathlib import Path
from math import pi
import numpy as np

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

from _helpers import (
    make_cloud_profile, multilayer_pydisort_toa_full_phi, PHI_VALUES,
)
from pydisort_riccati_jax import pydisort_riccati_jax

NQuad = 8
NLeg = NQuad
N = NQuad // 2

tau_bot = 10.0
mu0, I0, phi0 = 0.5, 1.0, 0.0
rho = 0.05
BDRF = [rho / pi]

omega_func, Leg_coeffs_func = make_cloud_profile(
    tau_bot, omega_top=0.85, omega_bot=0.96,
    g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
)

# ----- Riccati references at two tolerances -----
print("Riccati reference at tol=1e-5...")
_, _, _, u_func_5, _ = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-5, BDRF_Fourier_modes=BDRF,
)
u_ref_5 = np.column_stack([u_func_5(phi)[:N] for phi in PHI_VALUES])

print("Riccati reference at tol=1e-8...")
_, _, _, u_func_8, _ = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-8, BDRF_Fourier_modes=BDRF,
)
u_ref_8 = np.column_stack([u_func_8(phi)[:N] for phi in PHI_VALUES])

scale = max(float(np.max(np.abs(u_ref_8))), 1e-8)
ref_diff = float(np.max(np.abs(u_ref_5 - u_ref_8))) / scale
print(f"\nRiccati tol=1e-5 vs tol=1e-8 difference: {ref_diff:.4e}")
print("(This is the reference error contaminating the current tests)\n")

# ----- Multilayer convergence study -----
layer_counts = [20, 50, 100, 200, 500, 1000, 2000, 5000]
errors_vs_5 = []
errors_vs_8 = []

for NL in layer_counts:
    print(f"Multilayer pydisort NLayers={NL}...")
    _, _, uf = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, NL, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )
    u_ml = np.column_stack([uf(0, phi)[:N] for phi in PHI_VALUES])

    err_5 = float(np.max(np.abs(u_ml - u_ref_5))) / scale
    err_8 = float(np.max(np.abs(u_ml - u_ref_8))) / scale
    errors_vs_5.append(err_5)
    errors_vs_8.append(err_8)

# ----- Print convergence table -----
print(f"\n{'NLayers':>8s}  {'err vs tol=1e-5':>16s}  {'ratio':>8s}  {'err vs tol=1e-8':>16s}  {'ratio':>8s}")
print("-" * 68)
for i, NL in enumerate(layer_counts):
    r5 = f"{errors_vs_5[i-1]/errors_vs_5[i]:.1f}" if i > 0 and errors_vs_5[i] > 1e-15 else ""
    r8 = f"{errors_vs_8[i-1]/errors_vs_8[i]:.1f}" if i > 0 and errors_vs_8[i] > 1e-15 else ""
    print(f"{NL:8d}  {errors_vs_5[i]:16.4e}  {r5:>8s}  {errors_vs_8[i]:16.4e}  {r8:>8s}")

print(f"\nTheoretical O(h^2) ratio for 2.5x refinement: {2.5**2:.1f}")
print(f"Theoretical O(h^2) ratio for 2.0x refinement: {2.0**2:.1f}")
print(f"Theoretical O(h^2) ratio for 5.0x refinement: {5.0**2:.1f}")
