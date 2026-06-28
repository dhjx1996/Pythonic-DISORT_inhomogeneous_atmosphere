"""Runtime/thread setup — IMPORT (and call ``setup()``) BEFORE jax is imported.

**CPU-affinity pool cap.** XLA's CPU (Eigen) intra-op thread pool sizes to the number
of *visible* CPUs. On crew1, ``srun --cpu-bind=cores`` did NOT constrain affinity, so a
1-CPU task saw 32 cores -> a 32-wide pool thrashing on 1 CPU (the oversubscription bug).
Pinning affinity to the allocated core count caps the pool (jovyan demo: 16 visible -> 73
threads, 4 -> 25, 2 -> 17). Must happen before jax spins up its pools.

(A persistent JAX compilation cache was evaluated and **dropped**. Its reachable compile
surface is small and largely un-cacheable: the dominant per-profile compile is the
``select_retrieval_grid`` pool Jacobian, keyed on the *state-dependent* ODE-grid width, so
it rarely cache-hits; net value was ~1-2 % of a solve-bound task against real
staleness/concurrency risk. See docs/OUTSTANDING.md.)

Env:
  FR_PIN_CORES        explicit core count to pin (overrides SLURM_CPUS_PER_TASK).
  SLURM_CPUS_PER_TASK used if FR_PIN_CORES unset.
"""
import os


def setup():
    _pin_affinity()


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
