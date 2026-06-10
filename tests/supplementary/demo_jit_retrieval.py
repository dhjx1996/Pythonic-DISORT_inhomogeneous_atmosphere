"""
demo_jit_retrieval.py — the jit-able retrieval forward recipe (OUTSTANDING §C).

Demonstrates the composable seam that resolves the jit blocker: split the
host-side SciPy setup (run once) from a traceable, jit-able solve of the traced
inputs (tau_bot and the optics closures), so the retrieval amortises one compile
across hundreds of forward / gradient evaluations. ``mu0`` is **static** (baked
into ``setup``); the Fourier modes are mapped with ``lax.scan`` (compiled once,
O(1) in mode count — OUTSTANDING §H), which removes the old forward/jacrev
compile-memory ceiling.

Run (from tests/):
    python supplementary/demo_jit_retrieval.py
    JAX_PLATFORMS=cpu python supplementary/demo_jit_retrieval.py

Shows:
  [A] host-side setup (static mu0; all NFourier modes — the noise-aware mode
      truncation now lives in retrieval_oe.select_num_modes, an S_ε selector)
  [B] jit(forward): cold compile once, then cached at ~ms across varying
      tau_bot (no recompile) — the §C win
  [C] reverse-mode jax.grad through the jitted forward (the discrete adjoint)
  [D] forward-mode jax.jacfwd via an adjoint=ForwardMode() setup (small-DOF
      retrieval; the reverse default's custom_vjp can't be forward-differentiated)
"""
import sys
import time
from pathlib import Path
from math import pi

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

from pydisort_riccati_jax import (
    riccati_setup, riccati_solve, eval_radiance,
)

print(f"BACKEND={jax.default_backend()}  x64={jax.config.jax_enable_x64}")

NQuad = 8
mu0, I0, phi0 = 0.6, 1.0, 0.0
tau_bot = 10.0
omega_func = lambda tau: 0.99
g = 0.83
Leg_coeffs_func = lambda tau: g ** jnp.arange(NQuad)

# Retrieval observation angles (where the radiance is compared to data).
mu_obs = jnp.array([0.35, 0.6, 0.85])
phi_obs = jnp.array([0.0, pi / 3, pi])

# --- [A] host-side setup (static mu0) ---------------------------------------
setup = riccati_setup(NQuad, I0, phi0, mu0)
K = setup.NFourier      # all modes; trim offline with retrieval_oe.select_num_modes
print(f"\n[A] setup built; mu0={mu0} (static); running all NFourier = {K} modes")


def forward(tau_bot):
    """The retrieval forward model: traced tau_bot; mu0 static; K modes."""
    res = riccati_solve(setup, omega_func, Leg_coeffs_func, tau_bot, num_modes=K)
    return eval_radiance(setup, res, mu_obs, phi_obs)


# --- [B] jit: compile once, cache across tau_bot ----------------------------
f = jax.jit(forward)
t0 = time.perf_counter()
f(tau_bot).block_until_ready()
cold = time.perf_counter() - t0
warms = []
for tb in (9.0, 11.0, 12.0, 8.0):
    t0 = time.perf_counter()
    f(tb).block_until_ready()
    warms.append(time.perf_counter() - t0)
print(f"[B] jit(forward): cold(compile)={cold:6.1f}s   "
      f"warm(cached)={np.mean(warms) * 1e3:6.1f} ms   "
      f"(speedup {cold / max(np.mean(warms), 1e-9):.0f}x, no recompile across tau_bot)")

# --- [C] reverse-mode gradient (discrete adjoint, default) ------------------
def scalar(tau_bot):
    return jnp.sum(forward(tau_bot))

grad_tau = jax.jit(jax.grad(scalar))
t0 = time.perf_counter()
g_tau = float(grad_tau(tau_bot))
print(f"[C] reverse jax.grad d(sum radiance)/d tau_bot = {g_tau:.4e}  "
      f"(compile {time.perf_counter() - t0:.1f}s, then cached)")

# --- [D] forward-mode jacobian (small-DOF retrieval) ------------------------
setup_fwd = riccati_setup(NQuad, I0, phi0, mu0, adjoint=diffrax.ForwardMode())
def scalar_fwd(tau_bot):
    res = riccati_solve(setup_fwd, omega_func, Leg_coeffs_func, tau_bot,
                        num_modes=K)
    return jnp.sum(eval_radiance(setup_fwd, res, mu_obs, phi_obs))

jf = float(jax.jit(jax.jacfwd(scalar_fwd))(tau_bot))
print(f"[D] forward jax.jacfwd d(sum radiance)/d tau_bot = {jf:.4e}  "
      f"(agrees with reverse: {abs(jf - g_tau) / max(abs(g_tau), 1e-9) < 1e-2})")

print("\nRecipe: setup once (static mu0) -> jit(forward) -> cached forward + "
      "grad/jacfwd. The forward model is now jit-amortised for the retrieval loop.")
