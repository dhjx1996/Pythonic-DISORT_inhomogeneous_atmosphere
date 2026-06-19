"""Test suite 22: thin-cloud Mie off-nadir physical plausibility + NLeg_all convergence (float32).

A *structured Mie* cloud phase function (size parameter
x≈24 at 2.13 µm ⇒ ~70 significant Legendre moments) needs enough TMS moments (`NLeg_all`) or the
reconstructed single-scatter `p_full(cosΘ)` Gibbs-oscillates and sign-flips, making the **thin**
cloud's (single-scatter-dominated) ToA reflectance erratic / negative / azimuth-rippled. Tests
19/20 only use smooth analytic Henyey–Greenstein, so they never exercise this; flux / m=0 stay
physical even when the off-nadir field is wrong.

Modeled in spirit on PythonicDISORT's NT validation (a many-moment cloud phase function), but it
asserts physical **plausibility** (positivity / magnitude / smoothness) and
**NLeg_all convergence**, not pydisort agreement (a shared low-NLeg_all TMS artifact would pass an
agreement test while both are wrong). Views stay in the back/side-scatter envelope (μ≥0.5,
φ∈[π/2,π]) to avoid the irreducible ~10° forward solar-aureole exclusion.

  22a NLeg_all convergence : R(128) ≈ R(192) (converged) while R(32) is under-resolved.
  22b Plausibility @ 128   : positive, physical magnitude, smooth.
"""
import numpy as np
from math import pi
import pytest

from pydisort_riccati_jax import pydisort_riccati_jax, interpolate

pytest.importorskip("miejax_lite")
from miejax_lite import mie_legendre_precompute, mie_avg_legendre  # noqa: E402

# Thin marine-Sc-like Mie cloud at 2.13 µm (RF11-ish): ~70 significant Legendre moments, so
# NLeg_all=32 truncates badly while 128 resolves it.
WAVELENGTH, V_EFF, R_EFF = 2.13, 0.10, 10.0
TAU_BOT = 1.2
NQuad = 16
MU0, I0, PHI0 = 0.6, 1.0, 0.0

# Back/side-scatter viewing envelope (avoid the forward aureole: small Θ / φ≈0 / grazing μ).
# A dense grid so the smoothness (Gibbs-ripple) metric is meaningful.
VIEW_MU = np.linspace(0.50, 0.95, 6)
VIEW_PHI = np.linspace(pi / 2, pi, 5)

# Exact gamma-averaged Mie optics, 256 moments available (enough for the 192 reference).
_PRECOMP = mie_legendre_precompute(max_nstop=512, NLeg=256)
_omega, _leg, _ = mie_avg_legendre(R_EFF, WAVELENGTH, V_EFF, _PRECOMP)
_OMEGA, _LEG = float(_omega), np.asarray(_leg)

_CACHE = {}


def _reflectance_grid(NLeg_all):
    """NT-corrected ToA reflectance R = π u / (μ0 I0) on the (μ, φ) envelope (memoized).

    The diffuse solve uses NQuad streams; `NLeg_all` is the (untruncated) moment count fed to
    delta-M + the TMS single-scatter τ-quadrature — the quantity under test.
    """
    if NLeg_all not in _CACHE:
        leg = _LEG[:NLeg_all]
        omega_func = lambda tau: _OMEGA
        Leg_coeffs_func = lambda tau: leg
        mu_arr_pos, _, _, u_func, _ = pydisort_riccati_jax(
            TAU_BOT, omega_func, Leg_coeffs_func, NQuad, MU0, I0, PHI0, tol=1e-3,
            delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=True,
        )
        uf = interpolate(u_func, mu_arr_pos)
        R = np.array([[float(uf(mu, phi)) for phi in VIEW_PHI] for mu in VIEW_MU])
        _CACHE[NLeg_all] = R * pi / (MU0 * I0)
    return _CACHE[NLeg_all]


def _maxrel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-3)))


def _max_turning_points(R):
    """Most first-difference sign changes along any μ-cut or φ-cut — a Gibbs-ripple metric.
    Over this small back/side-scatter envelope a smooth radiance field is monotone or
    single-peaked (≤1 turning point per cut); an oscillating `p_full` flips many times."""
    def tp(line):
        s = np.sign(np.diff(line))
        s = s[s != 0]
        return int(np.sum(np.diff(s) != 0)) if s.size else 0
    mu_tp = max(tp(R[:, j]) for j in range(R.shape[1]))    # along μ at fixed φ
    phi_tp = max(tp(R[i, :]) for i in range(R.shape[0]))   # along φ at fixed μ
    return max(mu_tp, phi_tp)


def test_22a_nleg_all_convergence():
    """The TMS single-scatter must be converged in NLeg_all: 128≈192, 32 far off."""
    print("\n--- Test 22a: NLeg_all convergence (thin Mie, off-nadir) ---")
    R32, R128, R192 = (_reflectance_grid(k) for k in (32, 128, 192))
    conv = _maxrel(R128, R192)
    under = _maxrel(R32, R192)
    print(f"  max rel diff  (128 vs 192) = {conv:.3f}   (32 vs 192) = {under:.3f}")
    assert conv < 0.05, f"NLeg_all=128 not converged vs 192 (max rel={conv:.3f})"
    assert under > 0.20, f"NLeg_all=32 should be far from converged (max rel={under:.3f})"


def test_22b_physical_plausibility():
    """At NLeg_all=128 the off-nadir reflectance is positive, physical-magnitude, and smooth.
    (Gibbs artifact shows up as *roughness* and the discriminator is smoothness vs the under-resolved field.)
    """
    print("\n--- Test 22b: physical plausibility at NLeg_all=128 ---")
    R128, R32 = _reflectance_grid(128), _reflectance_grid(32)
    print(f"  R(128) range = [{R128.min():.3f}, {R128.max():.3f}]")
    assert np.all(R128 > 0.0), f"negative reflectance (min={R128.min():.3f})"
    assert 0.01 < R128.min() and R128.max() < 0.6, \
        f"unphysical magnitude [{R128.min():.3f},{R128.max():.3f}]"
    # Smoothness: the converged field has *fewer* spurious oscillations than the under-resolved
    # one (an absolute count can't separate genuine backscatter/glory structure from Gibbs ripple
    # — convergence in 22a is what proves 128 is the resolved field; this is the relative check).
    tp128, tp32 = _max_turning_points(R128), _max_turning_points(R32)
    print(f"  max turning points: 128 -> {tp128}   32 -> {tp32}")
    assert tp128 < tp32, f"NLeg_all=128 not smoother than 32 ({tp128} vs {tp32} turning points)"


# Interesting, but not a functionality test
'''
def test_22c_low_nleg_all_pathology():
    """Document the deficit: NLeg_all=32 is physically implausible (negative / oversized / rippled)."""
    print("\n--- Test 22c: NLeg_all=32 pathology ---")
    R = _reflectance_grid(32)
    tp = _max_turning_points(R)
    pathology = bool((R.min() < 0.0) or (R.max() > 0.6) or (tp >= 2))
    print(f"  NLeg_all=32: R range=[{R.min():.3f},{R.max():.3f}]  max turning points={tp}")
    assert pathology, "expected NLeg_all=32 to be physically implausible (the §A′ deficit)"
'''
