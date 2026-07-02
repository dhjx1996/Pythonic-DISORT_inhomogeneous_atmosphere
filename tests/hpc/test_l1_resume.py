"""Layer-1 resume-equivalence gate (production scale — HPC only).

An uninterrupted GN solve must equal an interrupted-then-resumed one on the
same platform (FR_CHECKPOINT_RESUME_PLAN verification #1). Uses the REAL worker
setup on one profile; the in-suite small-config analog is tests/23h.

Standardized from the ad-hoc ``hpc/gates/_fr_resume_test.py`` (2026-07). Opt-in:

    PYDISORT_HPC_GATES=1 PYDISORT_RICCATI_JAX_X64=1 \
        python -m pytest tests/hpc -m hpc -k l1 -v

Env: OPTICS_CACHE / RADIANCE_CACHE / VOCALS_DATA; DIAG_IDX (default 95).
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.hpc


def _gate():
    if not os.environ.get("PYDISORT_HPC_GATES"):
        pytest.skip("HPC gate: opt in with PYDISORT_HPC_GATES=1 "
                    "(hours of compute; needs the production caches)")


def test_l1_resume_equivalence(tmp_path):
    _gate()
    import retrieval_worker as rw
    from pydisort_riccati_jax import retrieval_oe as roe
    from pydisort_riccati_jax import vocals_io as vio

    idx = int(os.environ.get("DIAG_IDX", "95"))
    ck = str(tmp_path / "resume.ckpt.npz")
    profiles = vio.load_all_profiles(rw.DATA)
    truth = profiles[idx]
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    fwd, y, Se, s_grid, pb_phys, pb_log, tt, tbp = rw.build_forward_and_obs(
        truth, clim, idx)
    x_a, Sa = roe.make_climatology_prior(s_grid, clim, log=True)

    def run(ckpt, n_iter):
        return roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, x0=x_a,
                                   n_iter=n_iter, lm=1e-2, xtol=2e-3,
                                   cost_rtol=None, chi2_floor=None,
                                   max_n_outer=1, prior_builder=pb_log,
                                   checkpoint_path=ckpt)

    ref = run(None, 8)                             # uninterrupted
    run(ck, 3)                                     # interrupt after 3 (ckpt written)
    assert os.path.exists(ck), "interrupted run wrote no checkpoint"
    res = run(ck, 8)                               # resume -> complete

    if not np.array_equal(ref.x, res.x):
        dmax = float(np.max(np.abs(ref.x - res.x)))
        assert dmax < 1e-6, (f"resume-equivalence: max|x_ref - x_resumed|="
                             f"{dmax:.3e} (bit-exact expected on one platform)")
