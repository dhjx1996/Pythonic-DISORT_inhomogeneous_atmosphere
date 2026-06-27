"""Runtime/thread setup — IMPORT (and call ``setup()``) BEFORE jax is imported.

Two levers, both proven on jovyan (DESIGN: threading study):

1. **CPU-affinity pool cap.** XLA's CPU (Eigen) intra-op thread pool sizes to the number
   of *visible* CPUs. On crew1, ``srun --cpu-bind=cores`` did NOT constrain affinity, so a
   1-CPU task saw 32 cores -> a 32-wide pool thrashing on 1 CPU (the oversubscription bug).
   Pinning affinity to the allocated core count caps the pool (jovyan demo: 16 visible -> 73
   threads, 4 -> 25, 2 -> 17). Must happen before jax spins up its pools.

2. **Persistent compilation cache.** With the per-profile-shape axes collapsed (precomputed
   radiances + per-band NFourier + fixed k), the whole ensemble reduces to a handful of
   distinct XLA graphs keyed by the retrieval node-count k. A shared on-disk cache turns
   "recompile per task" into "compile once, load thereafter". A cache HIT returns the
   bit-identical executable, so results are unchanged (verify once: cached vs uncached).

Env:
  FR_PIN_CORES        explicit core count to pin (overrides SLURM_CPUS_PER_TASK).
  SLURM_CPUS_PER_TASK used if FR_PIN_CORES unset.
  JAX_COMPILE_CACHE_DIR  if set, enable the persistent compilation cache there.
"""
import os


def setup():
    _pin_affinity()
    _enable_compile_cache()


def _pin_affinity():
    n = os.environ.get("FR_PIN_CORES") or os.environ.get("SLURM_CPUS_PER_TASK")
    if not n:
        return
    try:
        n = int(n)
        allowed = sorted(os.sched_getaffinity(0))
        # SLURM_LOCALID offset so co-located tasks pin to DISJOINT cores when the cgroup
        # didn't already isolate them (the crew1 case); harmless when it did.
        local = int(os.environ.get("SLURM_LOCALID", "0"))
        start = (local * n) % max(len(allowed), 1)
        pick = [allowed[(start + i) % len(allowed)] for i in range(min(n, len(allowed)))]
        os.sched_setaffinity(0, set(pick))
        print(f"[runtime] pinned affinity to {len(pick)} cores "
              f"(localid={local}, allowed={len(allowed)})", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"[runtime] affinity pin skipped: {e}", flush=True)


def _enable_compile_cache():
    cache = os.environ.get("JAX_COMPILE_CACHE_DIR")
    if not cache:
        return
    try:
        import jax
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        print(f"[runtime] persistent JAX compile cache: {cache}", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"[runtime] compile cache not enabled: {e}", flush=True)
