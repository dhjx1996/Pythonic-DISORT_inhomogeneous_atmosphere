"""
demo_deltaM_tms.py — visual "intended-effect" demonstration of delta-M + TMS.

Analogue of the PythonicDISORT documentation demonstration: plot the ToA
upwelling radiance vs polar angle for a forward-peaked cloud phase function,
comparing

    raw  (NQuad=16, no correction)   <- rings, goes negative
    dM   (NQuad=16, delta-M only)    <- non-negative, peak still coarse
    dM+TMS (NQuad=16, delta-M + TMS) <- single-scattering peak restored
    truth (NQuad=64, full Legendre)  <- reference

against the high-stream truth, printing the before/after max-rel-error table and
saving a PNG (for the report figure / manual verification).

Run from tests/ (float64 recommended for a clean reference):

    PYDISORT_RICCATI_JAX_X64=1 python supplementary/demo_deltaM_tms.py
"""
import sys
from pathlib import Path
from math import pi

import numpy as np

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

import jax.numpy as jnp
from PythonicDISORT import subroutines

from pydisort_riccati_jax import pydisort_riccati_jax, interpolate
from _helpers import pydisort_toa_full_phi


def _build_interp(u_func, mu_pos):
    return interpolate(u_func, mu_pos)


def run_demo(tau_bot=8.0, omega=1 - 1e-6, g=0.75, NLeg_all=64,
             mu0=0.6, I0=100.0, phi0=0.0, phi=0.0, savepath=None):
    NQuad = 16
    N = NQuad // 2

    Leg_full = g ** np.arange(NLeg_all)
    omega_func = lambda tau: omega
    Leg_coeffs_func = lambda tau: g ** jnp.arange(NLeg_all)

    # Truth: NQuad=64, full Legendre.
    NQuad_truth = 64
    N_truth = NQuad_truth // 2
    _, _, uf_truth = pydisort_toa_full_phi(
        tau_bot, omega, NQuad_truth, Leg_full, mu0, I0, phi0, NLeg=NLeg_all,
    )
    mu_truth = subroutines.Gauss_Legendre_quad(N_truth)[0]
    truth_interp = _build_interp(lambda p: np.asarray(uf_truth(0, p))[:N_truth],
                                 mu_truth)

    # Raw / delta-M / delta-M+TMS, all NQuad=16.
    mu_pos, _, _, u_raw, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=False, NLeg_all=NLeg_all)
    _, _, _, u_dM, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=False)
    _, _, _, u_dMTMS, _ = pydisort_riccati_jax(
        tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=1e-8,
        delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=True)

    raw_i = _build_interp(u_raw, mu_pos)
    dM_i = _build_interp(u_dM, mu_pos)
    tms_i = _build_interp(u_dMTMS, mu_pos)

    mu_eval = np.linspace(0.05, 1.0, 60)
    t = np.asarray(truth_interp(mu_eval, phi))
    r = np.asarray(raw_i(mu_eval, phi))
    d = np.asarray(dM_i(mu_eval, phi))
    c = np.asarray(tms_i(mu_eval, phi))

    scale = max(float(np.max(np.abs(t))), 1e-12)
    print(f"\nDemo: tau={tau_bot}, omega={omega}, g={g}, mu0={mu0}, phi={phi}")
    print(f"  {'case':<14}{'max-rel-err':>14}{'min radiance':>16}")
    for name, arr in [("raw", r), ("delta-M", d), ("delta-M+TMS", c)]:
        err = float(np.max(np.abs(arr - t))) / scale
        print(f"  {name:<14}{err:>14.3e}{float(np.min(arr)):>16.3e}")
    print(f"  {'truth':<14}{0.0:>14.3e}{float(np.min(t)):>16.3e}")

    if savepath is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            theta = np.degrees(np.arccos(mu_eval))
            plt.figure(figsize=(7, 5))
            plt.plot(theta, t, "k-", lw=2, label="truth (NQuad=64)")
            plt.plot(theta, r, "r--", label="raw (NQuad=16)")
            plt.plot(theta, d, "b-.", label="delta-M")
            plt.plot(theta, c, "g-", label="delta-M + TMS")
            plt.axhline(0, color="gray", lw=0.6)
            plt.xlabel("polar angle (deg)")
            plt.ylabel("ToA upwelling radiance")
            plt.title(f"Delta-M / TMS: tau={tau_bot}, g={g}, omega={omega}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(savepath, dpi=120)
            print(f"  saved figure -> {savepath}")
        except Exception as exc:  # plotting is optional
            print(f"  (figure not saved: {exc})")


if __name__ == "__main__":
    out = _tests_dir / "supplementary" / "demo_deltaM_tms.png"
    run_demo(savepath=str(out))
    # A cloud-like forward peak as a second demonstration.
    run_demo(tau_bot=10.0, omega=0.99, g=0.85, I0=1.0,
             savepath=str(_tests_dir / "supplementary" / "demo_deltaM_tms_cloud.png"))
