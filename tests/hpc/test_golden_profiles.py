"""Golden cross-check: fresh FR retrievals vs the golden GPU-probe bundle.

THE standardized post-refactor validation (fable assessment: "re-gate on a
3-profile golden-diff before the next production sweep"): retrieve 2–3 trusted
profiles with the CURRENT code, then compare against the golden bundle on the
platform-INVARIANT quantities — the dense retrieved profile ``re_ours_dense``
(grid-independent, so CPU results compare against GPU goldens even when the
QRCP node grids differ slightly) and the retrieved scalars.

Standardized from the ad-hoc ``hpc/gates/_fr_golden_compare.py`` (2026-07).
Workflow: run ``scripts/retrieval_worker.py <idx> runs/_fr_parts/<idx>`` for the
golden indices, then

    PYDISORT_HPC_GATES=1 python -m pytest tests/hpc -m hpc -k golden -v

Env: FR_GOLD_DIR (default runs/precision_probe_out), FR_PARTS_DIR (default
runs/_fr_parts), FR_GOLDEN_IDXS (default "20,47,49" — the trusted probe set).
Thresholds mirror the original verdict scale: <1e-4 µm = tight (same-platform),
<5e-2 µm = correct with expected CPU/GPU or version drift, beyond = FAIL.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.hpc

GOLD_DIR = Path(os.environ.get("FR_GOLD_DIR", "runs/precision_probe_out"))
PARTS_DIR = Path(os.environ.get("FR_PARTS_DIR", "runs/_fr_parts"))
IDXS = [int(v) for v in os.environ.get("FR_GOLDEN_IDXS", "20,47,49").split(",")]


def _gate():
    if not os.environ.get("PYDISORT_HPC_GATES"):
        pytest.skip("HPC gate: opt in with PYDISORT_HPC_GATES=1")


@pytest.mark.parametrize("idx", IDXS)
def test_golden_profile(idx):
    _gate()
    ours_p = PARTS_DIR / f"{idx}_A.npz"
    gold_p = GOLD_DIR / f"probe_{idx}_tol1e-4_A.npz"
    if not ours_p.exists():
        pytest.skip(f"no fresh result {ours_p} — run the worker for idx {idx} first")
    if not gold_p.exists():
        pytest.skip(f"no golden {gold_p} — set FR_GOLD_DIR")
    o = np.load(ours_p, allow_pickle=True)
    g = np.load(gold_p, allow_pickle=True)

    # identical truth (same profile) — a sanity check on the comparison itself
    rt_o = np.asarray(o["re_truth_dense"], float)
    rt_g = np.asarray(g["re_truth_dense"], float)
    np.testing.assert_allclose(rt_o, rt_g, atol=1e-9,
                               err_msg="truth-dense differs: comparing different "
                                       "profiles or a changed truth pipeline")

    ro = np.asarray(o["re_ours_dense"], float)
    rg = np.asarray(g["re_ours_dense"], float)
    m = float(np.max(np.abs(ro - rg)))
    scalars = {k: (float(o[k]), float(g[k]))
               for k in ("tau_bot_ret", "r_base_ret", "dofs", "sic")}
    detail = "  ".join(f"{k}: {a:.4f} vs {b:.4f}" for k, (a, b) in scalars.items())
    assert m < 5e-2, (f"idx {idx}: max|re_dense diff|={m:.3e} um exceeds the "
                      f"5e-2 correctness bound — INVESTIGATE. [{detail}]")
    # loose scalar coherence (grid-independent quantities)
    assert abs(scalars["tau_bot_ret"][0] - scalars["tau_bot_ret"][1]) < 0.5, detail
    assert abs(scalars["dofs"][0] - scalars["dofs"][1]) < 0.3, detail
