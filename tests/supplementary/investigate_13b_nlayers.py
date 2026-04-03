"""
Investigate test_13b / test_15a failure: does increasing NLayers_ref reduce
the 6.9e-3 error between Riccati (tol=1e-3) and multilayer pydisort?
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

tau_bot = 30.0
mu0, I0, phi0 = 0.5, 1.0, 0.0
rho = 0.05
BDRF = [rho / pi]

omega_func, Leg_coeffs_func = make_cloud_profile(
    tau_bot, omega_top=0.85, omega_bot=0.96,
    g_top=0.865, g_bot=0.820, NLeg=NLeg, NQuad=NQuad,
)

# Riccati solve
print("Running Riccati (tol=1e-3)...")
_, _, _, u_ToA_func, tau_grid = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-3, BDRF_Fourier_modes=BDRF,
)
print(f"  Riccati steps: {len(tau_grid) - 1}")

u_ric = np.column_stack([u_ToA_func(phi)[:N] for phi in PHI_VALUES])

# Multilayer references at increasing NLayers
for NLayers_ref in [6000, 10000, 15000]:
    print(f"\nRunning multilayer pydisort (NLayers={NLayers_ref})...")
    _, _, uf_ref = multilayer_pydisort_toa_full_phi(
        tau_bot, omega_func, Leg_coeffs_func, NLayers_ref, NQuad, NLeg,
        mu0, I0, phi0, BDRF_Fourier_modes=BDRF,
    )
    u_ref = np.column_stack([uf_ref(0, phi)[:N] for phi in PHI_VALUES])

    scale = max(float(np.max(np.abs(u_ref))), 1e-8)
    rel_err = float(np.max(np.abs(u_ric - u_ref))) / scale
    print(f"  max rel_err vs Riccati: {rel_err:.4e}")

    # Per-phi breakdown
    for i, phi in enumerate(PHI_VALUES):
        col_scale = max(float(np.max(np.abs(u_ref[:, i]))), 1e-8)
        col_err = float(np.max(np.abs(u_ric[:, i] - u_ref[:, i]))) / col_scale
        print(f"    phi={phi:.4f}: rel_err={col_err:.4e}")
