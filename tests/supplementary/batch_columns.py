"""
Does batching across columns flip the GPU-vs-CPU verdict? (companion to profile_solver.py)

The per-column solve is latency-bound (sequential tiny matmuls). But the retrieval
runs MANY columns, which is embarrassingly parallel: vmap turns the tiny matmuls
into batched matmuls that fill the GPU. Measure warm (cached) per-column time vs
batch size B, GPU vs CPU.

Run:
    python supplementary/batch_columns.py
    JAX_PLATFORMS=cpu python supplementary/batch_columns.py

Measured 2026-06-08 (Tesla T4 vs CPU, jax 0.10.1, float32, NQuad=8) — warm us/column:

    B       GPU (us/col)   CPU (us/col)  GPU total(ms)   CPU total(ms)
    1          30592.4        1908.4          30.59           1.91
    16          2233.3        1021.1          35.73          16.34
    64           555.5         959.2          35.55          61.39      <- crossover
    256          155.2         854.0          39.73         218.63
    1024          50.4         829.0          51.58         848.91
    4096          16.0         846.1          65.45        3465.79

  Findings:
    - CPU per-column ~flat (~850-1900 us); CPU total ~linear in B (limited parallelism).
    - GPU per-column collapses ~1900x (30592 -> 16 us); GPU total barely grows (idle at B=1,
      filled by batching -> latency hidden by occupancy).
    - Crossover ~B=64; at B=4096 the GPU is ~53x faster per column than CPU.
  Implication: the §D "GPU not a lever" result is the SINGLE-COLUMN regime. The retrieval is
  parallel across columns -> jit (item C) + vmap a batch onto the GPU is the right architecture.
  Per-op launch latency (the B=1 bottleneck) is set off-device, so this is GPU-agnostic: the
  conclusion is about batch regime, not device generation. (Measured on a Tesla T4 for the record.)
"""
import sys, time
sys.path.insert(0, "/home/jovyan/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/src")
sys.path.insert(0, "/home/jovyan/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/tests")
import numpy as np
import jax, jax.numpy as jnp
import diffrax
from PythonicDISORT import subroutines
from _riccati_solver_jax import _precompute_legendre, _make_alpha_beta_funcs_jax

print(f"BACKEND={jax.default_backend()}  DEVICE={jax.devices()[0]}")

NQuad = 8
N = NQuad // 2
tau, mu0 = 8.0, 0.6
mu_pos_np, W_np = subroutines.Gauss_Legendre_quad(N)
mu_pos = jnp.asarray(mu_pos_np); W = jnp.asarray(W_np); M_inv = 1.0 / mu_pos
leg0 = _precompute_legendre(0, NQuad, mu_pos)


def solve(omega_val, g_val):
    of = lambda t: omega_val
    Lf = lambda t: g_val ** jnp.arange(NQuad)
    a_f, b_f = _make_alpha_beta_funcs_jax(of, Lf, 0, leg0, mu_pos, W, M_inv, N)

    def vf(s, R, args):
        a = a_f(tau - s); b = b_f(tau - s)
        return a @ R + R @ a + R @ b @ R + b

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(), 0.0, tau, None, jnp.zeros((N, N)),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(t1=True), max_steps=4096)
    return sol.ys[0]


vsolve = jax.jit(jax.vmap(solve))
print(f"{'B':>6} {'warm_total_ms':>14} {'us_per_column':>14}")
for B in [1, 16, 64, 256, 1024, 4096]:
    oms = jnp.linspace(0.90, 0.99, B)
    gs = jnp.linspace(0.80, 0.88, B)
    R = vsolve(oms, gs); R.block_until_ready()          # cold compile for this shape
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); R = vsolve(oms, gs); R.block_until_ready()
        ts.append(time.perf_counter() - t0)
    tw = np.median(ts)
    print(f"{B:>6} {tw * 1e3:>14.2f} {tw / B * 1e6:>14.1f}")
