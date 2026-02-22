"""
generate_reference.py — Pre-compute pydisort reference results for tests 1-5.

Run once (from the tests/ directory) to populate reference_results/*.npz:

    cd tests
    python generate_reference.py

The .npz files are used as a fallback when PythonicDISORT.pydisort is not
importable at test time.  Each file stores:
    flux_up_ToA  : float scalar
    u0_ToA       : (NQuad,) array of zeroth-mode intensities at tau=0
"""
import sys
from pathlib import Path
from math import pi

import numpy as np

# Ensure the package is findable whether or not it is installed.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import pydisort_toa

REF = Path(__file__).parent / "reference_results"
REF.mkdir(exist_ok=True)


def save(name, tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
         b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    flux_up, u0 = pydisort_toa(
        tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    np.savez(REF / f"{name}.npz", flux_up_ToA=flux_up, u0_ToA=u0)
    print(f"  {name}: flux_up_ToA={flux_up:.6e}")


NQuad = 8

print("=== Test 1: Isotropic scattering ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0
mu0_1, I0_1, phi0_1 = 0.1, pi / 0.1, pi
for lbl, tau_bot, omega in [
    ("1a", 0.03125, 0.2),
    ("1b", 0.03125, 1 - 1e-6),
    ("1c", 0.03125, 0.99),
    ("1d", 1.5,     0.2),
    ("1e", 1.5,     1 - 1e-6),
    ("1f", 1.5,     0.99),
]:
    save(lbl, tau_bot, omega, NQuad, g_l_iso, mu0_1, I0_1, phi0_1)

print("\n=== Test 2: Rayleigh-like scattering ===")
g_l_ray = np.zeros(NQuad); g_l_ray[0] = 1.0; g_l_ray[2] = 0.1
mu0_2, I0_2, phi0_2 = 0.080442, pi, pi
for lbl, tau_bot, omega in [
    ("2a", 0.2, 0.5),
    ("2b", 0.2, 1 - 1e-6),
    ("2c", 1.5, 0.5),
    ("2d", 1.5, 1 - 1e-6),
]:
    save(lbl, tau_bot, omega, NQuad, g_l_ray, mu0_2, I0_2, phi0_2)

print("\n=== Test 3: HG scattering ===")
for lbl, tau_bot, omega, g, mu0, I0, phi0 in [
    ("3a", 1.0, 1 - 1e-6, 0.75, 1.0,     pi,       pi),
    ("3b", 1.0, 0.9,      0.75, 0.5,     1.0,      0.0),
    ("3c", 2.0, 0.8,      0.5,  0.6, pi / 0.6, 0.9 * pi),
]:
    g_l = g ** np.arange(NQuad)
    save(lbl, tau_bot, omega, NQuad, g_l, mu0, I0, phi0)

print("\n=== Test 4: Non-zero BCs ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0

# 4a: no beam, isotropic, b_neg only
save("4a", 0.5, 0.5, NQuad, g_l_iso, 0.5, 0.0, 0.0, b_neg=1.0 / pi)

# 4b: beam + b_pos, HG g=0.75
g_l_4b = 0.75 ** np.arange(NQuad)
save("4b", 1.0, 0.8, NQuad, g_l_4b, 0.5, 1.0, 0.0, b_pos=0.5)

# 4c: beam + b_pos + b_neg, isotropic
save("4c", 2.0, 0.5, NQuad, g_l_iso, 0.6, pi / 0.6, 0.5 * pi,
     b_pos=0.3, b_neg=0.1)

print("\n=== Test 5: BDRF (Lambertian) ===")
# 5a: isotropic, scalar BDRF rho=0.1
save("5a", 0.5, 0.5, NQuad, g_l_iso, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.1 / pi])

# 5b: HG g=0.75, scalar BDRF rho=0.5
g_l_5b = 0.75 ** np.arange(NQuad)
save("5b", 1.0, 0.8, NQuad, g_l_5b, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.5 / pi])

# 5c: HG g=0.5, scalar BDRF rho=0.3 (callable test uses same physics)
g_l_5c = 0.5 ** np.arange(NQuad)
save("5c", 2.0, 0.7, NQuad, g_l_5c, 0.6, pi / 0.6, 0.0,
     BDRF_Fourier_modes=[0.3 / pi])

print("\nDone. All reference files written to:", REF)
