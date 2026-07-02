"""PythonicDISORT reference wrappers, for validation.

pydisort (exact eigendecomposition, piecewise-constant layers) is the trusted
reference the Riccati solver is verified against — by the test suite and by the
results notebook's convergence checks. The wrappers return
``(flux_up_ToA, u0_ToA, u_func)`` in the Riccati solver's conventions; the
multilayer variant approximates a continuously τ-varying atmosphere with
``NLayers`` piecewise-constant layers (midpoint rule), so refining ``NLayers``
gives an O(h²) convergence check against the Riccati solution.
"""
from math import pi

import numpy as np

# Standard azimuthal angles for full-field comparison against pydisort.
PHI_VALUES = (0.0, pi / 4, pi / 2, pi, 3 * pi / 2)


def pydisort_toa_full_phi(
    tau_bot, omega, NQuad, g_l, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
    NLeg=None, delta_M_scaling=False, NT_cor=False,
):
    """
    Run pydisort for a single homogeneous layer and return
    (flux_up_ToA, u0_ToA, u_func) where u_func = u(tau, phi).
    Callers typically discard flux and u0 and use only u_func.

    For delta-M / NT apples-to-apples: ``g_l`` holds NLeg_all moments, ``NLeg``
    is the number used in the solve, and f = g_l[NLeg] (the first dropped
    moment) matches the Riccati solver's internal convention.
    """
    from PythonicDISORT.pydisort import pydisort

    N = NQuad // 2
    # np.array (not asarray) forces a writable copy: pydisort normalizes the
    # 0th Legendre moment in-place (Leg_coeffs_all[:, 0] = 1), and a jnp->numpy
    # input (e.g. Mie coeffs) would otherwise be a read-only buffer.
    g_l = np.atleast_2d(np.array(g_l, dtype=float))     # (1, NLeg_all)
    if NLeg is None:
        NLeg = NQuad
    if delta_M_scaling:
        f_arr = g_l[:, NLeg]                            # (1,) -> f = g_{NLeg}
    else:
        f_arr = 0

    mu_arr, Fp, Fm, u0f, uf = pydisort(
        np.array([float(tau_bot)]),
        np.array([float(omega)]),
        int(NQuad),
        g_l,
        float(mu0), float(I0), float(phi0),
        NLeg=NLeg, NFourier=NQuad,
        only_flux=False,
        f_arr=f_arr, NT_cor=NT_cor,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=list(BDRF_Fourier_modes),
    )
    return float(Fp(0)), u0f(0)[:N], uf


def multilayer_pydisort_toa_full_phi(
    tau_bot, omega_func, Leg_coeffs_func, NLayers, NQuad, NLeg, mu0, I0, phi0,
    b_pos=0, b_neg=0, BDRF_Fourier_modes=(),
    delta_M_scaling=False, NT_cor=False,
):
    """
    Approximate tau-varying (omega, g_l) with NLayers piecewise-constant layers
    (midpoint rule) and return (flux_up_ToA, u0_ToA, u_func) from pydisort.
    u_func = u(tau, phi) is the full azimuthally-resolved intensity.

    delta_M_scaling / NT_cor mirror pydisort_riccati_jax: the per-layer
    truncation fraction is f = Leg_coeffs_all[:, NLeg] (the first dropped
    moment), matching the Riccati solver's internal convention, so this is an
    apples-to-apples reference.  Leg_coeffs_func must then return >= NLeg+1
    coefficients (use make_cloud_profile(..., NLeg_all=...)).
    """
    from PythonicDISORT.pydisort import pydisort

    N = NQuad // 2
    edges = np.linspace(0, tau_bot, NLayers + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    tau_arr = edges[1:]

    omega_arr = np.array([omega_func(t) for t in mids])
    Leg_arr = np.array([Leg_coeffs_func(t) for t in mids])  # (NLayers, NLeg_all)

    if delta_M_scaling:
        f_arr = Leg_arr[:, NLeg]                            # f = g_{NLeg} per layer
    else:
        f_arr = 0

    mu_arr, Fp, Fm, u0f, uf = pydisort(
        tau_arr, omega_arr, NQuad,
        Leg_arr, float(mu0), float(I0), float(phi0),
        NLeg=NLeg, NFourier=NQuad,
        only_flux=False,
        f_arr=f_arr, NT_cor=NT_cor,
        b_pos=b_pos, b_neg=b_neg,
        BDRF_Fourier_modes=list(BDRF_Fourier_modes),
    )
    return float(Fp(0)), u0f(0)[:N], uf


def make_cloud_profile(tau_bot, omega_top, omega_bot, g_top, g_bot, NLeg, NQuad,
                       NLeg_all=None):
    """
    Build (omega_func, Leg_coeffs_func) for a linearly-interpolated cloud.

    omega and g vary linearly from top (tau=0) to bottom (tau=tau_bot).
    Phase function is Henyey-Greenstein: g_l(tau) = g(tau)^l.

    Leg_coeffs_func returns ``NLeg_all`` moments (default NLeg).  For delta-M /
    NT tests pass NLeg_all > NLeg so that f = g_{NLeg} and the extra untruncated
    moments are available; the solver still uses only NLeg of them in the
    discrete-ordinate solve.
    """
    if NLeg_all is None:
        NLeg_all = NLeg
    omega_func = lambda tau: omega_top + (omega_bot - omega_top) * tau / tau_bot
    g_func     = lambda tau: g_top + (g_bot - g_top) * tau / tau_bot
    Leg_coeffs_func   = lambda tau: g_func(tau) ** np.arange(NLeg_all)
    return omega_func, Leg_coeffs_func
