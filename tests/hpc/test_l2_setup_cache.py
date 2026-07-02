"""Layer-2 setup-cache equivalence gate (production scale — HPC only).

``build_forward_and_obs`` with a cache HIT (load + skip the deterministic setup)
must reproduce the COMPUTE path bit-exactly on the same platform: same K_list,
s_grid, tau_bot_pre, AND the same forward/Jacobian evaluations at a test state.
A FAIL blocks enabling ``FR_SETUP_CACHE`` in a production sweep.

Standardized from the ad-hoc ``hpc/gates/_fr_l2_test.py`` (2026-07); the
in-suite small-config analog is tests/23 (L1) — this one runs the REAL worker
setup, hours of compute, so it is opt-in:

    PYDISORT_HPC_GATES=1 PYDISORT_RICCATI_JAX_X64=1 \
        python -m pytest tests/hpc -m hpc -k l2 -v

Env: OPTICS_CACHE / RADIANCE_CACHE / VOCALS_DATA as in the production sbatch;
DIAG_IDX selects the profile (default 95, a thin one).
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.hpc


def _gate():
    if not os.environ.get("PYDISORT_HPC_GATES"):
        pytest.skip("HPC gate: opt in with PYDISORT_HPC_GATES=1 "
                    "(hours of compute; needs the production caches)")


def _assert_equiv(name, a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b):
        return
    worst = float(np.max(np.abs(a - b)))
    assert worst < tol, (f"L2 equivalence: {name} differs (max|d|={worst:.3e} "
                         f">= {tol:g}; bit-exact expected on one platform)")


def test_l2_setup_cache_equivalence(tmp_path):
    _gate()
    import retrieval_worker as rw
    from pydisort_riccati_jax import retrieval_oe as roe
    from pydisort_riccati_jax import vocals_io as vio

    idx = int(os.environ.get("DIAG_IDX", "95"))
    cache = tmp_path / f"{idx}.setup.npz"
    profiles = vio.load_all_profiles(rw.DATA)
    truth = profiles[idx]
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)

    def run():
        fwd, y, Se, s_grid, pb_phys, pb_log, tt, tbp = rw.build_forward_and_obs(
            truth, clim, idx, setup_cache_path=str(cache))
        x = fwd._encode_state(roe.make_climatology_prior(s_grid, clim)[0])
        F = np.asarray(fwd.forward(x, s_grid), float)
        K = np.asarray(fwd.jacobian(x, s_grid), float)
        return list(map(int, fwd.K_list)), np.asarray(s_grid, float), float(tbp), F, K

    k1, g1, t1, F1, K1 = run()                     # PASS 1: compute + WRITE
    # The gate's premise: PASS 1 actually wrote the cache, so PASS 2 exercises
    # the LOAD path. Without this it silently degrades to compute-vs-compute —
    # exactly the vacuous PASS the 2026-07-01 np.savez write bug (F7) produced.
    assert cache.exists(), ("PASS 1 did not write the setup cache; "
                            "LOAD path untestable (vacuous gate)")
    k2, g2, t2, F2, K2 = run()                     # PASS 2: LOAD

    assert k1 == k2, f"K_list differs: {k1} vs {k2}"
    _assert_equiv("s_grid", g1, g2)
    _assert_equiv("tau_bot_pre", t1, t2)
    _assert_equiv("forward", F1, F2)
    _assert_equiv("jacobian", K1, K2)
