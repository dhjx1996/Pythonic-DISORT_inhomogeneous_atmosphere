"""
Shared utilities for the pydisort_riccati test suite.

Reference values are obtained by running pydisort (exact eigendecomposition
solver) on-the-fly.  If PythonicDISORT.pydisort is unavailable (e.g. in an
environment that only has the Riccati module installed), the tests fall back to
pre-computed .npz files stored in reference_results/.
"""
from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
REFERENCE_DIR = Path(__file__).parent / "reference_results"

# Standard azimuthal angles for full-field comparison against pydisort.
PHI_VALUES = (0.0, pi / 4, pi / 2, pi, 3 * pi / 2)

def get_reference(
    test_name, tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Return u_func from the reference solver.

    On-the-fly path: u_func = pydisort's u(tau, phi) -> (2N,).
    Fallback path:   u_func wraps stored u_phi_ToA -> (N,),
                     only valid at tau=0 and PHI_VALUES.
    Both paths are compatible with assert_close_to_reference_phi (which
    calls u_func(0, phi)[:N]).
    """
    try:
        _, _, uf = pydisort_toa_full_phi(
            tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
            b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
        )
        return uf
    except ImportError:
        data = np.load(REFERENCE_DIR / f"{test_name}.npz")
        u_phi_ToA = data["u_phi_ToA"]  # (N, n_phi)
        phi_to_col = {phi: i for i, phi in enumerate(PHI_VALUES)}
        def _u_func(tau, phi):
            col = phi_to_col.get(phi)
            if col is not None:
                return u_phi_ToA[:, col]
            raise ValueError(f"phi={phi} not in stored PHI_VALUES")
        return _u_func


# ---------------------------------------------------------------------------
# Multi-layer pydisort reference (for convergence tests)
# ---------------------------------------------------------------------------

def multilayer_pydisort_toa_full_phi(
    tau_bot, omega_func, Leg_coeffs_func, NLayers, NQuad, NLeg, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Approximate tau-varying (omega, g_l) with NLayers piecewise-constant layers
    (midpoint rule) and return (flux_up_ToA, u0_ToA, u_func) from pydisort.
    u_func = u(tau, phi) is the full azimuthally-resolved intensity.
    """
    from PythonicDISORT.pydisort import pydisort

    N = NQuad // 2
    edges = np.linspace(0, tau_bot, NLayers + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    tau_arr = edges[1:]

    omega_arr = np.array([omega_func(t) for t in mids])
    Leg_arr = np.array([Leg_coeffs_func(t) for t in mids])  # (NLayers, NLeg)

    mu_arr, Fp, Fm, u0f, uf = pydisort(
        tau_arr, omega_arr, NQuad,
        Leg_arr, float(mu0), float(I0), float(phi0),
        NLeg=NLeg, NFourier=NQuad,
        only_flux=False,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=list(BDRF_Fourier_modes),
    )
    return float(Fp(0)), u0f(0)[:N], uf


def make_cloud_profile(tau_bot, omega_top, omega_bot, g_top, g_bot, NLeg, NQuad):
    """
    Build (omega_func, Leg_coeffs_func) for a linearly-interpolated cloud.

    omega and g vary linearly from top (tau=0) to bottom (tau=tau_bot).
    Phase function is Henyey-Greenstein: g_l(tau) = g(tau)^l.
    """
    omega_func = lambda tau: omega_top + (omega_bot - omega_top) * tau / tau_bot
    g_func     = lambda tau: g_top + (g_bot - g_top) * tau / tau_bot
    Leg_coeffs_func   = lambda tau: g_func(tau) ** np.arange(NLeg)
    return omega_func, Leg_coeffs_func


# ---------------------------------------------------------------------------
# pydisort reference wrapper returning full phi-dependent intensity
# ---------------------------------------------------------------------------

def pydisort_toa_full_phi(
    tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Run pydisort for a single homogeneous layer and return
    (flux_up_ToA, u0_ToA, u_func) where u_func = u(tau, phi).
    Callers typically discard flux and u0 and use only u_func.
    """
    from PythonicDISORT.pydisort import pydisort

    N = NQuad // 2
    mu_arr, Fp, Fm, u0f, uf = pydisort(
        np.array([float(tau_bot)]),
        np.array([float(omega)]),
        int(NQuad),
        np.atleast_2d(np.asarray(g_l, dtype=float)),
        float(mu0), float(I0), float(phi0),
        NLeg=NQuad, NFourier=NQuad,
        only_flux=False,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=list(BDRF_Fourier_modes),
    )
    return float(Fp(0)), u0f(0)[:N], uf


# ---------------------------------------------------------------------------
# Azimuthal intensity assertion helper
# ---------------------------------------------------------------------------

def assert_close_to_reference_phi(u_func_ric, u_func_ref, phi_values, N, rel_tol=5e-3):
    """
    Compare Riccati u_ToA_func(phi) vs pydisort u(0, phi) at several azimuthal angles.

    Only upwelling intensities (first N elements) are compared.
    u_func_ric: phi -> (N,) from pydisort_riccati_jax
    u_func_ref: u(tau, phi) from pydisort (called at tau=0)
    N: half the number of quadrature streams (upwelling hemisphere size)
    """
    for phi in phi_values:
        u_ric = u_func_ric(phi)[:N]
        u_ref = u_func_ref(0, phi)[:N]
        scale = max(float(np.max(np.abs(u_ref))), 1e-8)
        rel_err = float(np.max(np.abs(u_ric - u_ref))) / scale
        assert rel_err < rel_tol, (
            f"phi={phi:.4f}: u_ToA rel_err={rel_err:.3e} >= tol={rel_tol}"
        )


def assert_convergence_phi(u_ref_phi, u_coarse_phi, u_fine_phi,
                           min_ratio=8.0, abs_tol=1e-2):
    """
    Assert that multilayer pydisort u(phi) converges toward Riccati u(phi).

    All inputs are (N, n_phi) arrays of upwelling intensities at ToA,
    evaluated at the same set of azimuthal angles.

    min_ratio : coarse_err / fine_err must exceed this threshold.
    abs_tol   : fine_err must be below this.
    """
    scale = max(float(np.max(np.abs(u_ref_phi))), 1e-8)
    err_coarse = float(np.max(np.abs(u_coarse_phi - u_ref_phi))) / scale
    err_fine   = float(np.max(np.abs(u_fine_phi   - u_ref_phi))) / scale

    assert err_fine < err_coarse, (
        f"Fine grid ({err_fine:.3e}) not more accurate than coarse ({err_coarse:.3e})"
    )
    ratio = err_coarse / max(err_fine, 1e-15)
    assert ratio >= min_ratio, (
        f"Convergence ratio {ratio:.1f} < required {min_ratio:.1f} "
        f"(coarse err={err_coarse:.3e}, fine err={err_fine:.3e})"
    )
    assert err_fine < abs_tol, (
        f"Fine-grid u(phi) rel_err={err_fine:.3e} >= abs_tol={abs_tol}"
    )
