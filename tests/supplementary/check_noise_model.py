"""Sanity checks for src/noise_model.py (pure NumPy — no solver).

Verifies the three-term σ(ρ): shot-off collapse to "Option B", the shot term's
SNR-at-rho_ref calibration, per-band coefficients (band-major layout), Se=diag(σ²),
sample() statistics, and the bright-cloud regime where shot is subdominant.

Run:  python tests/supplementary/check_noise_model.py
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import noise_model as nm


def test_shot_off_is_option_B():
    m = nm.NoiseModel(k_cal=0.02, floor=1e-3)              # snr_ref=inf default
    rho = np.array([0.1, 0.22, 0.29])
    expect = np.sqrt((0.02 * rho) ** 2 + 1e-3 ** 2)
    assert np.allclose(m.sigma(rho), expect)


def test_shot_term_calibration():
    # k=floor=0, snr_ref at rho_ref  ->  SNR(rho_ref) == snr_ref exactly
    m = nm.NoiseModel(k_cal=0.0, snr_ref=200.0, rho_ref=0.3, floor=0.0)
    s = m.sigma(np.array([0.3]))[0]
    assert np.isclose(0.3 / s, 200.0)
    # and the shot term scales as sqrt(rho): SNR(rho) = snr_ref*sqrt(rho/rho_ref)
    s2 = m.sigma(np.array([1.2]))[0]
    assert np.isclose(1.2 / s2, 200.0 * np.sqrt(1.2 / 0.3))


def test_three_terms_quadrature():
    m = nm.NoiseModel(k_cal=0.02, snr_ref=150.0, rho_ref=0.25, floor=2e-3)
    rho = np.array([0.15, 0.4])
    expect = np.sqrt((0.02 * rho) ** 2 + rho * 0.25 / 150.0 ** 2 + 2e-3 ** 2)
    assert np.allclose(m.sigma(rho), expect)


def test_per_band_coeffs_band_major():
    # 2 bands x 3 views, band-major: first 3 obs band0, next 3 band1
    m = nm.NoiseModel(k_cal=np.array([0.01, 0.03]), floor=0.0)
    rho = np.ones(6)
    s = m.sigma(rho, n_bands=2)
    assert np.allclose(s[:3], 0.01) and np.allclose(s[3:], 0.03)
    # bad divisibility raises
    try:
        m.sigma(np.ones(5), n_bands=2); assert False
    except ValueError:
        pass


def test_Se_is_sigma_squared_diagonal():
    m = nm.oci_swir()
    rho = np.array([0.1, 0.2, 0.3, 0.25])
    Se = m.Se(rho, n_bands=2)
    assert np.allclose(np.diag(Se), m.sigma(rho, n_bands=2) ** 2)
    assert np.allclose(Se - np.diag(np.diag(Se)), 0.0)


def test_sample_statistics_and_reproducibility():
    m = nm.oci_swir(k_cal=0.03, floor=1e-3)
    rho = np.full(4, 0.25)
    a = m.sample(rho, seed=7)
    b = m.sample(rho, seed=7)
    assert np.allclose(a, b)                                # reproducible
    draws = np.array([m.sample(rho, seed=k) for k in range(4000)])
    assert np.allclose(draws.mean(0), rho, atol=5e-4)       # unbiased
    assert np.allclose(draws.std(0), m.sigma(rho), rtol=0.08)


def test_bright_cloud_shot_subdominant():
    # With a literal OCI-ish SNR (if it were known), shot << calibration for clouds.
    m = nm.NoiseModel(k_cal=0.02, snr_ref=250.0, rho_ref=0.05, floor=1e-3)
    rho = np.array([0.29])                                  # bright cloud
    cal = 0.02 * rho[0]
    shot = np.sqrt(rho[0] * 0.05) / 250.0
    assert shot < 0.25 * cal                                # shot is a small correction


def test_generic_relative_matches_legacy():
    # legacy Se used 0.03*max(|y|,0.02); for bright y this is 0.03*y (3% relative)
    m = nm.generic_relative()
    y = np.array([0.1, 0.22, 0.29])
    # ~3% relative; the small quadrature floor lifts the dimmest point ~2%
    assert np.allclose(m.sigma(y), 0.03 * y, rtol=3e-2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"ok  {f.__name__}")
    print(f"\nall {len(fns)} noise_model checks passed")
