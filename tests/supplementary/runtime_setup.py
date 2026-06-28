"""Runtime/thread setup — IMPORT (and call ``setup()``) BEFORE jax is imported.

**CPU-affinity pool cap.** XLA's CPU (Eigen) intra-op thread pool sizes to the number
of *visible* CPUs. On crew1, ``srun --cpu-bind=cores`` did NOT constrain affinity, so a
1-CPU task saw 32 cores -> a 32-wide pool thrashing on 1 CPU (the oversubscription bug).
Pinning affinity to the allocated core count caps the pool (jovyan demo: 16 visible -> 73
threads, 4 -> 25, 2 -> 17). Must happen before jax spins up its pools.

**Disjoint-range claim (the co-location fix).** When the cgroup does NOT isolate the cpuset
(the crew1/ginsburg case: every task sees all 32 cores), N co-located array tasks must pin
to DIFFERENT core ranges or they oversubscribe one range and thrash the cache / memory bus
while the rest of the node sits idle. ``SLURM_LOCALID`` cannot tell them apart — it is 0 for
every independent array element (each is its own job step), so the old ``localid*n`` offset
put ALL of them on cores[0:n]. Instead each task atomically claims the lowest free per-node
slot via a shared-FS registry keyed by (array-job, hostname) and pins to cores
[slot*n : slot*n+n]; dead owners' slots are reclaimed, and a task releases its slot at exit.

(A persistent JAX compilation cache was evaluated and **dropped**. Its reachable compile
surface is small and largely un-cacheable: the dominant per-profile compile is the
``select_retrieval_grid`` pool Jacobian, keyed on the *state-dependent* ODE-grid width, so
it rarely cache-hits; net value was ~1-2 % of a solve-bound task against real
staleness/concurrency risk. See docs/OUTSTANDING.md.)

Env:
  FR_PIN_CORES        explicit core count to pin (overrides SLURM_CPUS_PER_TASK).
  SLURM_CPUS_PER_TASK used if FR_PIN_CORES unset.
  FR_SLOT_DIR         base dir for the per-node slot registry (default a shared-FS path).
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
        if not allowed:
            return
        # If the cgroup already isolated us to <= n cores, there is exactly one range and
        # nothing to coordinate; otherwise claim a DISJOINT per-node slot so co-located
        # tasks land on different cores (SLURM_LOCALID can't distinguish array elements).
        n_slots = max(1, len(allowed) // n)
        slot = _claim_node_slot(n_slots) if n_slots > 1 else 0
        start = (slot * n) % len(allowed)
        pick = [allowed[(start + i) % len(allowed)] for i in range(min(n, len(allowed)))]
        os.sched_setaffinity(0, set(pick))
        print(f"[runtime] pinned affinity to {len(pick)} cores "
              f"(slot={slot}/{n_slots}, cores={pick[0]}..{pick[-1]}, allowed={len(allowed)})",
              flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"[runtime] affinity pin skipped: {e}", flush=True)


def _claim_node_slot(n_slots):
    """Atomically claim the lowest free per-node core-range slot in [0, n_slots). Co-located
    array tasks share the registry dir (same array-job + same hostname) and so end up on
    disjoint ranges. Reclaims slots whose owner PID is dead; releases ours at exit. Returns
    0 on any failure or if every slot is held (degrades to the old single-range behaviour)."""
    import socket
    import atexit
    jobid = (os.environ.get("SLURM_ARRAY_JOB_ID")
             or os.environ.get("SLURM_JOB_ID") or "x")
    node = socket.gethostname()
    base = os.environ.get("FR_SLOT_DIR",
                          "/burg-archive/home/dh3065/.rad_core_slots")
    reg = os.path.join(base, f"{jobid}_{node}")
    try:
        os.makedirs(reg, exist_ok=True)
    except Exception:                                              # noqa: BLE001
        return 0
    mypid = os.getpid()
    for slot in range(n_slots):
        p = os.path.join(reg, str(slot))
        if _try_take(p, mypid):
            atexit.register(_release, p, mypid)
            return slot
        if _owner_dead(p):                                         # reclaim a stale slot
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            if _try_take(p, mypid):
                atexit.register(_release, p, mypid)
                return slot
    return 0


def _try_take(path, pid):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(pid).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:                                              # noqa: BLE001
        return False


def _owner_dead(path):
    try:
        with open(path) as f:
            owner = int(f.read().strip())
    except (ValueError, FileNotFoundError, OSError):
        return True                                               # unreadable/empty -> stale
    try:
        os.kill(owner, 0)                                         # signal 0 = liveness probe
        return False                                             # alive
    except ProcessLookupError:
        return True                                              # gone -> reclaimable
    except PermissionError:
        return False                                             # alive (someone else's)
    except OSError:
        return False


def _release(path, pid):
    # only remove if WE still own it, so a reclaimer's entry is never clobbered
    try:
        with open(path) as f:
            if int(f.read().strip()) == pid:
                os.remove(path)
    except Exception:                                              # noqa: BLE001
        pass
