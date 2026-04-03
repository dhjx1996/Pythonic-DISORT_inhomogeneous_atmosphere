"""
generate_reference.py — Pre-compute pydisort reference results.

Run once (from the tests/ directory) to populate reference_results/*.npz:

    cd tests
    python supplementary/generate_reference.py

The .npz files are used as a fallback when PythonicDISORT.pydisort is not
importable at test time.  Each file stores:
    u_phi_ToA  : (N, n_phi) array of upwelling intensities at tau=0,
                 evaluated at PHI_VALUES azimuthal angles
"""
import sys
from pathlib import Path
from math import pi

import numpy as np

# Ensure the package is findable whether or not it is installed.
_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

from _helpers import pydisort_toa_full_phi, PHI_VALUES

REF = _tests_dir / "reference_results"
REF.mkdir(exist_ok=True)


def save(name, tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
         b_pos=0, b_neg=0, BDRF_Fourier_modes=()):
    N = NQuad // 2
    _, _, uf = pydisort_toa_full_phi(
        tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )
    # Evaluate u(tau=0, phi) at standard azimuthal angles, upwelling only
    u_phi = np.column_stack([uf(0, phi)[:N] for phi in PHI_VALUES])  # (N, n_phi)
    np.savez(REF / f"{name}.npz", u_phi_ToA=u_phi)
    print(f"  {name}: u_phi_ToA shape={u_phi.shape}")


NQuad = 8

print("=== Test 1: Isotropic scattering ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0
mu0_1, I0_1, phi0_1 = 0.1, pi / 0.1, pi
for lbl, tau_bot, omega in [
    ("1a", 0.03125, 0.2),
    ("1b", 0.03125, 1 - 1e-6),
    ("1c", 0.03125, 0.99),
    ("1d", 32.0,    0.2),
    ("1e", 32.0,    1 - 1e-6),
    ("1f", 32.0,    0.99),
]:
    save(lbl, tau_bot, omega, NQuad, g_l_iso, mu0_1, I0_1, phi0_1)

print("\n=== Test 2: Rayleigh-like scattering ===")
g_l_ray = np.zeros(NQuad); g_l_ray[0] = 1.0; g_l_ray[2] = 0.1
mu0_2, I0_2, phi0_2 = 0.080442, pi, pi
for lbl, tau_bot, omega in [
    ("2a", 0.2, 0.5),
    ("2b", 0.2, 1 - 1e-6),
    ("2c", 5.0, 0.5),
    ("2d", 5.0, 1 - 1e-6),
]:
    save(lbl, tau_bot, omega, NQuad, g_l_ray, mu0_2, I0_2, phi0_2)

print("\n=== Test 3: HG scattering ===")
for lbl, tau_bot, omega, g, mu0, I0, phi0 in [
    ("3a", 1.0, 1 - 1e-6, 0.75, 1.0,     pi,       pi),
    ("3b", 1.0, 0.9,      0.75, 0.5,     1.0,      0.0),
    ("3c", 5.0, 0.8,      0.5,  0.6, pi / 0.6, 0.9 * pi),
]:
    g_l = g ** np.arange(NQuad)
    save(lbl, tau_bot, omega, NQuad, g_l, mu0, I0, phi0)

print("\n=== Test 3d: Forward-peaked HG ===")
g_l_3d = 0.85 ** np.arange(NQuad)
save("3d", 1.5, 0.95, NQuad, g_l_3d, 0.5, 1.0, 0.0)

print("\n=== Test 4: Non-zero BCs ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0
save("4a", 0.5, 0.5, NQuad, g_l_iso, 0.5, 0.0, 0.0, b_neg=1.0 / pi)

g_l_4b = 0.75 ** np.arange(NQuad)
save("4b", 1.0, 0.8, NQuad, g_l_4b, 0.5, 1.0, 0.0, b_pos=0.5)

save("4c", 2.0, 0.5, NQuad, g_l_iso, 0.6, pi / 0.6, 0.5 * pi,
     b_pos=0.3, b_neg=0.1)

print("\n=== Test 4d: Purely absorbing (Beer's law) ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0
save("4d", 1.0, 1e-10, NQuad, g_l_iso, 0.5, 0.0, 0.0, b_neg=1.0 / pi)

print("\n=== Test 5: BDRF (Lambertian) ===")
save("5a", 0.5, 0.5, NQuad, g_l_iso, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.1 / pi])

g_l_5b = 0.75 ** np.arange(NQuad)
save("5b", 1.0, 0.8, NQuad, g_l_5b, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.5 / pi])

g_l_5c = 0.5 ** np.arange(NQuad)
save("5c", 2.0, 0.7, NQuad, g_l_5c, 0.6, pi / 0.6, 0.0,
     BDRF_Fourier_modes=[0.3 / pi])

print("\n=== Test 5d-5e: Combined BDRF + BCs / High albedo ===")
g_l_5d = 0.5 ** np.arange(NQuad)
save("5d", 1.0, 0.7, NQuad, g_l_5d, 0.5, 1.0, 0.0,
     b_pos=0.2, b_neg=0.1, BDRF_Fourier_modes=[0.3 / pi])

g_l_5e = 0.75 ** np.arange(NQuad)
save("5e", 0.5, 0.9, NQuad, g_l_5e, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.9 / pi])

print("\n=== Test 8: Thick + BCs ===")
g_l_iso = np.zeros(NQuad); g_l_iso[0] = 1.0

save("8a", 32.0, 0.99, NQuad, g_l_iso, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.05 / pi])
save("8b", 32.0, 0.99, NQuad, g_l_iso, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.3 / pi])

g_l_8c = 0.75 ** np.arange(NQuad)
save("8c", 10.0, 0.9, NQuad, g_l_8c, 0.5, 1.0, 0.0,
     BDRF_Fourier_modes=[0.85 / pi])

save("8d", 32.0, 0.5, NQuad, g_l_iso, 0.1, pi / 0.1, pi,
     b_pos=0.5)

g_l_8e = 0.5 ** np.arange(NQuad)
save("8e", 5.0, 0.8, NQuad, g_l_8e, 0.6, pi / 0.6, 0.9 * pi,
     b_pos=0.2, BDRF_Fourier_modes=[0.3 / pi])

g_l_8f = np.zeros(NQuad); g_l_8f[0] = 1.0; g_l_8f[2] = 0.1
save("8f", 5.0, 1 - 1e-6, NQuad, g_l_8f, 0.080442, pi, pi,
     BDRF_Fourier_modes=[0.1 / pi])

print("\n=== Test 11: NQuad variation + azimuthal ===")
NQuad_4 = 4
g_l_iso4 = np.zeros(NQuad_4); g_l_iso4[0] = 1.0
save("11a", 3.0, 0.5, NQuad_4, g_l_iso4, 0.5, 1.0, 0.0)

NQuad_16 = 16
g_l_hg16 = 0.75 ** np.arange(NQuad_16)
save("11b", 3.0, 0.9, NQuad_16, g_l_hg16, 0.5, 1.0, 0.0)

# 11c: azimuthal validation
NQuad_8 = 8
g_l_11c = 0.75 ** np.arange(NQuad_8)
save("11c", 3.0, 0.8, NQuad_8, g_l_11c, 0.5, 1.0, 0.0)

print("\nDone. All reference files written to:", REF)
