"""
profile_solver.py — empirical test of the cost claims in docs/OUTSTANDING.md C & D.

Run from tests/ (default GPU/whatever backend), and again forced onto CPU:

    python supplementary/profile_solver.py
    JAX_PLATFORMS=cpu python supplementary/profile_solver.py

What it measures (NQuad=8):
  [1]  unjitted full solver, 3 identical calls   -> §C "every call recompiles"
  [1b] NFourier / tol levers (unjitted)          -> §D "levers are NFourier/tol"
  [2]  jit-ABLE single-mode forward R-solve       -> §C "fix works" + §D cached/latency
       (the §C fix prototype: SaveAt(t1=True) + numpy GL nodes -> no host sync, no scipy in trace)
  [3]  cProfile of one unjitted call             -> shows host-side trace/compile dominates

Measured 2026-06-08 (Tesla T4 vs CPU, jax 0.10.1, float32):
  [1]  unjitted x3:  GPU 60.2/57.9/59.7 s   CPU 50.9/49.4/50.3 s   (recompiles every call;
                     GPU ~18% slower — host-side compile, GPU idle)
  [1b] NFourier 8->2: GPU 62->14 s, CPU 51->14 s ;  tol 1e-3->1e-2 (unjitted): ~no change
  [2]  jit single-mode: cold ~2 s -> warm  CPU 2.1 ms vs GPU 28.9 ms  (GPU 14x SLOWER cached:
                     kernel-launch-latency-bound on 4x4 matmuls).  tol 1e-2 warm: 1.6/22.2 ms.
  [3]  ~54 s of the ~60 s call is jax trace_to_jaxpr + pjit._trace_for_jit (16x via diffeqsolve)
                     = host-side tracing/XLA lowering, not device execution.
  Verdict: §C and §D confirmed; the GPU does not help (slightly hurts) this solver.
"""
import sys, time, cProfile, pstats, io
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

import numpy as np
import jax, jax.numpy as jnp
import diffrax
from PythonicDISORT import subroutines
from pydisort_riccati_jax import pydisort_riccati_jax
from _riccati_solver_jax import _precompute_legendre, _make_alpha_beta_funcs_jax

print(f"BACKEND={jax.default_backend()}  DEVICE={jax.devices()[0]}  x64={jax.config.jax_enable_x64}")

NQuad, NLa = 8, 16
N = NQuad // 2
tau, mu0, I0, phi0 = 8.0, 0.6, 1.0, 0.0
omega_func = lambda t: 0.95
Leg_func = lambda t: 0.85 ** jnp.arange(NLa)


def full_call(NFourier=NQuad, tol=1e-3):
    out = pydisort_riccati_jax(tau, omega_func, Leg_func, NQuad, mu0, I0, phi0,
                               NFourier=NFourier, tol=tol)
    return float(out[1])  # concretize flux -> forces device sync


print("\n[1] UNJITTED full solver, 3 identical calls (NFourier=8, tol=1e-3):")
for i in range(3):
    t0 = time.perf_counter(); v = full_call(); dt = time.perf_counter() - t0
    print(f"    call {i}: {dt:7.1f}s   flux={v:.6f}")

print("\n[1b] UNJITTED levers (each a fresh call):")
for label, kw in [("NFourier=8 tol=1e-3", {}), ("NFourier=2 tol=1e-3", {"NFourier": 2}),
                  ("NFourier=8 tol=1e-2", {"tol": 1e-2})]:
    t0 = time.perf_counter(); full_call(**kw); dt = time.perf_counter() - t0
    print(f"    {label:22s}: {dt:7.1f}s")

mu_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_pos = jnp.asarray(mu_pos_np); W = jnp.asarray(W_np); M_inv = 1.0 / mu_pos
leg0 = _precompute_legendre(0, NQuad, mu_pos)


def make_mode_solve(tol):
    def solve(omega_val, g_val):
        of = lambda t: omega_val
        Lf = lambda t: g_val ** jnp.arange(NQuad)
        alpha_f, beta_f = _make_alpha_beta_funcs_jax(of, Lf, 0, leg0, mu_pos, W, M_inv, N)

        def vf(sigma, R, args):
            a = alpha_f(tau - sigma); b = beta_f(tau - sigma)
            return a @ R + R @ a + R @ b @ R + b

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Kvaerno5(), 0.0, tau, None,
            jnp.zeros((N, N)),
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol * 1e-3),
            saveat=diffrax.SaveAt(t1=True), max_steps=4096)
        return sol.ys[0]
    return solve


print("\n[2] JIT-ABLE single-mode forward R-solve (SaveAt(t1=True)):")
for tol in (1e-3, 1e-2):
    solve_jit = jax.jit(make_mode_solve(tol))
    t0 = time.perf_counter(); R = solve_jit(0.95, 0.85); R.block_until_ready()
    cold = time.perf_counter() - t0
    warms = []
    for om, g in [(0.96, 0.84), (0.97, 0.83), (0.94, 0.86)]:
        t0 = time.perf_counter(); R = solve_jit(om, g); R.block_until_ready()
        warms.append(time.perf_counter() - t0)
    print(f"    tol={tol:.0e}: cold(compile)={cold:7.2f}s   warm(cached)={np.mean(warms)*1e3:7.1f}ms"
          f"  (each {[round(w*1e3,1) for w in warms]} ms)")

print("\n[3] cProfile of ONE unjitted full call (top cumulative):")
pr = cProfile.Profile(); pr.enable(); full_call(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
for line in s.getvalue().splitlines():
    low = line.lower()
    if any(k in low for k in ("compile", "lower", "trace", "xla", "dispatch", "backend",
                              "diffeqsolve", "pydisort_riccati", "cumtime", "function calls")):
        print("   ", line.strip())
