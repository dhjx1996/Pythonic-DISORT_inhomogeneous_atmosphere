"""
Shared utilities for the pydisort_riccati test suite.

Reference values come from running pydisort (exact eigendecomposition solver)
on-the-fly via :mod:`pydisort_riccati_jax.reference`, which also serves the
results notebook. PythonicDISORT is a hard dependency of the solver itself, so
it is always importable when the tests run.
"""
from __future__ import annotations

import numpy as np

from pydisort_riccati_jax.reference import (  # noqa: F401  (re-exported to the tests)
    PHI_VALUES,
    make_cloud_profile,
    multilayer_pydisort_toa_full_phi,
    pydisort_toa_full_phi,
)


def get_reference(
    tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
):
    """u(tau, phi) from the pydisort reference solver."""
    return pydisort_toa_full_phi(
        tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
        b_pos=b_pos, b_neg=b_neg, BDRF_Fourier_modes=BDRF_Fourier_modes,
    )[2]


# ---------------------------------------------------------------------------
# Azimuthal intensity assertion helpers
# ---------------------------------------------------------------------------

def assert_close_to_reference_phi(u_func_ric, u_func_ref, phi_values, N, rel_tol=1e-2):
    """
    Compare Riccati u_ToA_func(phi) vs pydisort u(0, phi) at several azimuthal angles.

    Only upwelling intensities (first N elements) are compared.

    Default rel_tol=1e-2 is the float32 production tolerance: the solver runs at
    tol=1e-3 in float32, whose accuracy floor vs exact pydisort is ~2e-3 over the
    full test range (thick + conservative + high-BDRF), so 1e-2 gives ~5x margin.
    The stringent float64 partition uses its own tight comparisons.
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


def assert_nonnegative_phi(u_func, phi_values, N, atol=1e-8):
    """Assert upwelling ToA radiance u_func(phi)[:N] is >= -atol at all phi.

    The headline delta-M/TMS fix: a forward-peaked phase function makes the
    finite-stream radiance ring negative (docs/OUTSTANDING.md A). A small atol
    absorbs float roundoff near zero.

    u_func : phi -> (N,) (Riccati u_ToA_func) or (tau, phi) -> (2N,) (pydisort).
    """
    for phi in phi_values:
        try:
            u = np.asarray(u_func(phi))[:N]          # Riccati signature
        except TypeError:
            u = np.asarray(u_func(0, phi))[:N]       # pydisort u(tau, phi)
        min_val = float(np.min(u))
        assert min_val >= -atol, (
            f"phi={phi:.4f}: min upwelling radiance {min_val:.3e} < -{atol:g} "
            f"(negative radiance not removed)"
        )


def find_min_radiance_phi(u_func, phi_values, N):
    """Return the minimum upwelling radiance over phi_values (diagnostic)."""
    mins = []
    for phi in phi_values:
        try:
            u = np.asarray(u_func(phi))[:N]
        except TypeError:
            u = np.asarray(u_func(0, phi))[:N]
        mins.append(float(np.min(u)))
    return min(mins)
