"""Quick post-refactor smoke: one-shot + seam run, finite, and mutually consistent.

Not a correctness check vs pydisort (that is the suite) — just catches shape /
signature / scan bugs from the §H refactor before running the full suite.
"""
import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "src"))

import numpy as np
import jax
import jax.numpy as jnp

import pydisort_riccati_jax as pdj
from pydisort_riccati_jax import riccati_setup, riccati_solve, eval_radiance

NQuad, mu0, I0, phi0, tau_bot = 8, 0.6, 1.0, 0.0, 8.0
g, omega = 0.8, 0.95
NLeg_all = 16
omega_func = lambda t: omega
Leg = lambda t: g ** jnp.arange(NLeg_all)
BDRF = [0.05 / pi]            # scalar Lambertian (the suite's form)

print(f"dtype={jnp.result_type(float)}")

# ---- one-shot (delta-M + TMS + Lambertian surface) ----
mu_pos, flux, u0, uf, grid = pdj.pydisort_riccati_jax(
    tau_bot, omega_func, Leg, NQuad, mu0, I0, phi0,
    BDRF_Fourier_modes=BDRF, delta_M_scaling=True, NLeg_all=NLeg_all, NT_cor=True)
u0 = np.asarray(u0)
print(f"[one-shot] u0_ToA={np.round(u0,5)}")
print(f"           flux={float(flux):.5f}  finite={np.all(np.isfinite(u0))}  "
      f"grid:[{grid[0]:.3f}..{grid[-1]:.3f}] n={len(grid)}")
assert np.all(np.isfinite(u0)) and np.all(u0 > 0), "one-shot u0 not finite/positive"

# ---- seam (mu0 now in setup; no calibrate) ----
setup = riccati_setup(NQuad, I0, phi0, mu0, NLeg_all=NLeg_all,
                      BDRF_Fourier_modes=BDRF, delta_M_scaling=True, NT_cor=True)
res = riccati_solve(setup, omega_func, Leg, tau_bot)
u0_seam = np.asarray(res.u_modes[0])
print(f"[seam]     u_modes[0]={np.round(u0_seam,5)}")
d = float(np.max(np.abs(u0_seam - u0)))
print(f"           max|seam u0 - one-shot u0| = {d:.2e}")
assert d < 1e-4, f"seam vs one-shot u0 mismatch {d:.2e}"

# ---- jit forward + jacfwd through the seam (the retrieval path) ----
import diffrax
setup_fwd = riccati_setup(NQuad, I0, phi0, mu0, NLeg_all=NLeg_all,
                          BDRF_Fourier_modes=BDRF, delta_M_scaling=True,
                          NT_cor=True, adjoint=diffrax.ForwardMode())
mu_obs = jnp.array([0.35, 0.6, 0.85]); phi_obs = jnp.array([0.0, pi / 3, pi])

def fwd(tb):
    r = riccati_solve(setup_fwd, omega_func, Leg, tb)
    return eval_radiance(setup_fwd, r, mu_obs, phi_obs)

y = np.asarray(jax.jit(fwd)(tau_bot))
J = np.asarray(jax.jit(jax.jacfwd(lambda tb: jnp.sum(fwd(tb))))(tau_bot))
print(f"[seam jit] radiance shape={y.shape} finite={np.all(np.isfinite(y))}  "
      f"d(sum)/d tau_bot={float(J):.4e} finite={np.isfinite(J)}")
assert np.all(np.isfinite(y)) and np.isfinite(J), "seam jit/jacfwd not finite"

print("\nSMOKE PASS — one-shot + seam run, finite, consistent; jit forward + jacfwd OK.")
