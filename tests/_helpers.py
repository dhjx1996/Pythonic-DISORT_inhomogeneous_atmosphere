"""
Shared utilities for the pydisort_magnus test suite.

Reference values are obtained by running pydisort (exact eigendecomposition
solver) on-the-fly.  If PythonicDISORT.pydisort is unavailable (e.g. in an
environment that only has the Magnus module installed), the tests fall back to
pre-computed .npz files stored in reference_results/.
"""
from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import scipy.special as sp

REFERENCE_DIR = Path(__file__).parent / "reference_results"

# ---------------------------------------------------------------------------
# Phase-function kernel factory
# ---------------------------------------------------------------------------

def make_D_m_funcs(g_l_or_func, NLeg: int, NFourier: int) -> list:
    """
    Build the list of D_m_funcs[m](tau, mu_i, mu_j) required by pydisort_magnus.

    Parameters
    ----------
    g_l_or_func : array-like of shape (NLeg,) **or** callable tau -> array(NLeg,)
        Legendre coefficients g_l (NOT weighted by 2l+1).
        When a callable is supplied the phase function is τ-varying.
    NLeg : int
        Number of Legendre terms considered.
    NFourier : int
        Number of Fourier azimuthal modes to build.

    Returns
    -------
    list of NFourier callables, each with signature
        D_m(tau, mu_i, mu_j) -> ndarray  (same shape as broadcast(mu_i, mu_j))
    where
        D^m_pure(mu_i, mu_j; tau)
            = 0.5 * sum_{l=m}^{NLeg-1} (2l+1) * poch(l+m+1,-2m) * g_l(tau)
              * P_l^m(mu_i) * P_l^m(mu_j)
    """
    varying = callable(g_l_or_func)
    if not varying:
        g_l_base = np.asarray(g_l_or_func, dtype=float)

    funcs: list = []
    for m in range(NFourier):
        ells = np.arange(m, NLeg)
        if len(ells) == 0:
            funcs.append(lambda tau, mu_i, mu_j:
                         np.zeros(np.broadcast_shapes(np.shape(mu_i), np.shape(mu_j))))
            continue

        poch_arr = sp.poch(ells + m + 1, -2.0 * m)

        def _D_m(tau, mu_i, mu_j, *, _m=m, _ells=ells, _poch=poch_arr):
            g_l = g_l_or_func(tau) if varying else g_l_base
            wt = (2 * _ells + 1) * _poch * g_l[_m:]
            result = sum(
                float(w) * sp.lpmv(_m, int(l), mu_i) * sp.lpmv(_m, int(l), mu_j)
                for l, w in zip(_ells, wt)
            )
            return 0.5 * np.asarray(result, dtype=float)

        funcs.append(_D_m)

    return funcs


# ---------------------------------------------------------------------------
# pydisort reference wrapper (single layer, ToA evaluation)
# ---------------------------------------------------------------------------

def pydisort_toa(
    tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Run pydisort for a single homogeneous layer and return
    (flux_up_ToA, u0_ToA) at tau = 0.
    """
    from PythonicDISORT.pydisort import pydisort

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
    return float(Fp(0)), u0f(0)


def get_reference(
    test_name, tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Return (flux_up_ToA, u0_ToA) from the reference solver.

    Tries pydisort on-the-fly first; falls back to a stored .npz file if
    pydisort is not importable.
    """
    try:
        return pydisort_toa(
            tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
            b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
        )
    except ImportError:
        data = np.load(REFERENCE_DIR / f"{test_name}.npz")
        return float(data["flux_up_ToA"]), data["u0_ToA"]


# ---------------------------------------------------------------------------
# Multi-layer pydisort reference (for convergence tests)
# ---------------------------------------------------------------------------

def multilayer_pydisort_toa(
    tau_bot, omega_func, g_l_func, NLayers, NQuad, NLeg, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """
    Approximate tau-varying (omega, g_l) with NLayers piecewise-constant layers
    (midpoint rule) and return (flux_up_ToA, u0_ToA) from pydisort.
    """
    from PythonicDISORT.pydisort import pydisort

    edges = np.linspace(0, tau_bot, NLayers + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    tau_arr = edges[1:]

    omega_arr = np.array([omega_func(t) for t in mids])
    Leg_arr = np.array([g_l_func(t) for t in mids])  # (NLayers, NLeg)

    mu_arr, Fp, Fm, u0f, uf = pydisort(
        tau_arr, omega_arr, NQuad,
        Leg_arr, float(mu0), float(I0), float(phi0),
        NLeg=NLeg, NFourier=NQuad,
        only_flux=False,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=list(BDRF_Fourier_modes),
    )
    return float(Fp(0)), u0f(0)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_close_to_reference(flux_mag, u0_mag, flux_ref, u0_ref, rel_tol=5e-3):
    """Assert Magnus matches pydisort reference within relative tolerance."""
    flux_scale = max(abs(flux_ref), 1e-8)
    rel_err_flux = abs(flux_mag - flux_ref) / flux_scale
    assert rel_err_flux < rel_tol, (
        f"flux_up_ToA: Magnus={flux_mag:.6e}, ref={flux_ref:.6e}, "
        f"rel_err={rel_err_flux:.3e} >= tol={rel_tol}"
    )
    u0_scale = max(float(np.max(np.abs(u0_ref))), 1e-8)
    rel_err_u0 = float(np.max(np.abs(u0_mag - u0_ref))) / u0_scale
    assert rel_err_u0 < rel_tol, (
        f"u0_ToA max rel_err={rel_err_u0:.3e} >= tol={rel_tol}"
    )


def make_cloud_profile(tau_bot, omega_top, omega_bot, g_top, g_bot, NLeg, NQuad):
    """
    Build (omega_func, g_l_func, D_m_funcs) for a linearly-interpolated cloud.

    omega and g vary linearly from top (tau=0) to bottom (tau=tau_bot).
    Phase function is Henyey-Greenstein: g_l(tau) = g(tau)^l.
    """
    omega_func = lambda tau: omega_top + (omega_bot - omega_top) * tau / tau_bot
    g_func     = lambda tau: g_top + (g_bot - g_top) * tau / tau_bot
    g_l_func   = lambda tau: g_func(tau) ** np.arange(NLeg)
    D_m_funcs  = make_D_m_funcs(g_l_func, NLeg, NQuad)
    return omega_func, g_l_func, D_m_funcs


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
    """
    from PythonicDISORT.pydisort import pydisort

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
    return float(Fp(0)), u0f(0), uf


# ---------------------------------------------------------------------------
# Azimuthal intensity assertion helper
# ---------------------------------------------------------------------------

def assert_close_to_reference_phi(u_func_mag, u_func_ref, phi_values, N, rel_tol=5e-3):
    """
    Compare Magnus u_ToA_func(phi) vs pydisort u(0, phi) at several azimuthal angles.

    Only upward-hemisphere intensities (first N elements) are compared.
    u_func_mag: phi -> (NQuad,) from pydisort_magnus
    u_func_ref: u(tau, phi) from pydisort (called at tau=0)
    N: half the number of quadrature streams (upward hemisphere size)
    """
    for phi in phi_values:
        u_mag = u_func_mag(phi)[:N]
        u_ref = u_func_ref(0, phi)[:N]
        scale = max(float(np.max(np.abs(u_ref))), 1e-8)
        rel_err = float(np.max(np.abs(u_mag - u_ref))) / scale
        assert rel_err < rel_tol, (
            f"phi={phi:.4f}: u_ToA rel_err={rel_err:.3e} >= tol={rel_tol}"
        )


def assert_convergence(
    flux_ref, flux_coarse, flux_fine,
    u0_ref, u0_coarse, u0_fine,
    min_ratio=8.0, abs_tol=1e-2,
):
    """
    Assert that the multi-layer pydisort solution converges toward the
    Magnus reference as the layer count increases.

    min_ratio : error at coarse must be at least this many times larger than
                error at fine (checks O(h^2) convergence direction).
    abs_tol   : error at fine must be below this relative to Magnus reference.
    """
    scale_flux = max(abs(flux_ref), 1e-8)
    err_coarse = abs(flux_coarse - flux_ref) / scale_flux
    err_fine   = abs(flux_fine   - flux_ref) / scale_flux

    # Fine must be more accurate than coarse
    assert err_fine < err_coarse, (
        f"Fine grid ({err_fine:.3e}) not more accurate than coarse ({err_coarse:.3e})"
    )
    # Convergence ratio
    ratio = err_coarse / max(err_fine, 1e-15)
    assert ratio >= min_ratio, (
        f"Convergence ratio {ratio:.1f} < required {min_ratio:.1f} "
        f"(coarse err={err_coarse:.3e}, fine err={err_fine:.3e})"
    )
    # Absolute accuracy of fine grid
    assert err_fine < abs_tol, (
        f"Fine-grid flux rel_err={err_fine:.3e} >= abs_tol={abs_tol}"
    )

    # Same checks for u0
    scale_u0 = max(float(np.max(np.abs(u0_ref))), 1e-8)
    err_u0_fine = float(np.max(np.abs(u0_fine - u0_ref))) / scale_u0
    assert err_u0_fine < abs_tol, (
        f"Fine-grid u0 rel_err={err_u0_fine:.3e} >= abs_tol={abs_tol}"
    )
