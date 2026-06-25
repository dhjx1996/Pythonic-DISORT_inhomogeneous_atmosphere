# STREAMLINE.md — Stale / Obsolete File Audit

Flags candidates for deletion. **Nothing has been deleted except the superseded IC result JSONs
(§7, 2026-06-25); scripts are flagged, never deleted here.** Confirm each group with the repo
owner before removing. Files are grouped by certainty; within each group, most-clearly-stale first.

Cross-references: `DESIGN_DECISIONS.md` §N = **DD§N**; `OUTSTANDING.md` item = **OD§X**.

---

## 1 · Unambiguously stale — superseded and explicitly named in the docs

These are safe to delete without further investigation.

### `tests/supplementary/`

| File | Why stale |
|---|---|
| `tune_filter_threshold.py` | DD§10f: "superseded 3 %-noise sweeps". Its own docstring labels it "historical 3 %-noise comparison only". Superseded by `sweep_threshold_2pct.py`. |
| `thick_sweep2.py` | DD§10f: same verdict. Uses `Se = 0.03*…`. The JSON it writes (`docs/cached_results/thick_sweep2.json`) is also stale (see §2). |
| `check_jac.py` | Investigated the NFourier-OOM compile blocker (OD§H "does capping NFourier let jacrev compile?"). OD§H is RESOLVED → DD§7; the behaviour is now covered by `21_jit_test.py`. |
| `smoke_refactor.py` | "Quick post-refactor smoke for the §H scan-over-modes refactor." The refactor is settled (DD§7) and the seam is tested in `21_jit_test.py`. |
| `characterize_geom.py` | Mapped the negative-radiance / sign envelope before delta-M existed (OD§A). OD§A is RESOLVED → DD§6; delta-M/TMS is now tested in `19_deltaM_test.py` and `20_deltaM_benchmark_test.py`. |
| `check_atnode.py` | Isolated barycentric-interpolation error at quadrature nodes to diagnose erratic off-nadir radiances. The TMS correction (DD§6) resolved the erratic-radiance root cause; this diagnostic is no longer needed and is unmentioned in any active doc. |
| `check_forward.py` | "Forward-only positivity + S_ε mode-selection check" written during the §A/§H debugging phase. Not referenced in any active doc; the production check is now `22_thin_mie_test.py`. |

### `docs/cached_results/`

| File | Why stale |
|---|---|
| `thick_sweep2.json` | Written by `thick_sweep2.py` (see above); the 3 %-noise thick filter-threshold sweep explicitly superseded by `filter_threshold_sweep.json` (2 % OCI noise, DD§10f). |
| `.ipynb_checkpoints/` (entire subdirectory, 5 files) | Auto-generated Jupyter checkpoints of JSON / notebook files that are already present one directory up. Pure noise: `filter_threshold_sweep-checkpoint.json`, `jacobian_decomposition-checkpoint.ipynb`, `joint_dofs_results-checkpoint.json`, `joint_osse_results-checkpoint.json`, `retrieval_baseline_linear_class-checkpoint.json`, `retrieval_thick_RF03_tau23-checkpoint.json`. |

---

## 2 · Orphaned compiled/binary artifacts — sources deleted, stubs remain

All safe to delete (they will regenerate automatically on next import / run).

### `tests/supplementary/__pycache__/` — `.pyc` files whose source no longer exists

| Orphaned `.pyc` | Deleted source was for |
|---|---|
| `build_mc_table.cpython-312.pyc` | `build_mc_table.py` — generated `mc_cloudbow_table.npz` (ve_retrieval cloudbow work, OD§I) |
| `build_mc_table_3d.cpython-312.pyc` | `build_mc_table_3d.py` — 3-D cloudbow table |
| `cloudbow_3d_demo.cpython-312.pyc` | `cloudbow_3d_demo.py` — cloudbow demonstration (ve_retrieval) |
| `mc_ve_signal.cpython-312.pyc` | `mc_ve_signal.py` — v_e MC signal study (ve_retrieval) |
| `mc_vs_ss_retrieval.cpython-312.pyc` | `mc_vs_ss_retrieval.py` — MC vs single-scatter comparison |
| `proto_scan_vmap.cpython-312.pyc` | `proto_scan_vmap.py` — early `lax.scan`/`vmap` prototype (pre-DD§7) |
| `test_T_equation.cpython-312-pytest-9.0.3.pyc` | `test_T_equation.py` — one-off test, source removed |
| `validate_mc.cpython-312.pyc` | `validate_mc.py` — MC table validation (ve_retrieval) |

### `tests/supplementary/__pycache__/` — `.pyc` for scripts flagged stale above

| Orphaned `.pyc` |
|---|
| `tune_filter_threshold.cpython-312.pyc` |
| `thick_sweep2.cpython-312.pyc` |

_(Will also orphan after the source deletions in §1.)_

### `tests/supplementary/.ipynb_checkpoints/` — 5 Jupyter checkpoint copies

| File |
|---|
| `.ipynb_checkpoints/batch_columns-checkpoint.py` |
| `.ipynb_checkpoints/generate_reference-checkpoint.py` |
| `.ipynb_checkpoints/info_content_linearity_probe-checkpoint.py` |
| `.ipynb_checkpoints/info_content_stage1-checkpoint.py` |
| `.ipynb_checkpoints/summarize_joint_results-checkpoint.py` |

### `docs/ve_retrieval/__pycache__/` — 4 `.pyc` files from the ve_retrieval branch

`cloudbow_joint_demo.cpython-312.pyc`, `cloudbow_ve_demo.cpython-312.pyc`,
`joint_retrieval_demo.cpython-312.pyc`, `ve_retrieval_demo.cpython-312.pyc` — their Python
sources live in the `ve_retrieval` branch (OD§I: "Set aside until further notice"). The
`.pyc` files here are leftover from a past merge/import.

---

## 3 · ve_retrieval binary data — out-of-scope until OD§I resumes

The cloudbow work is set aside (OD§I). The lookup tables it uses are still checked in:

| File | Size | Notes |
|---|---|---|
| `tests/supplementary/mc_cloudbow_table.npz` | ~780 KB | 2-D Monte Carlo cloudbow table; built by the now-deleted `build_mc_table.py`. No current Python source in this branch references it. |
| `tests/supplementary/mc_cloudbow_table_3d.npz` | large | 3-D (r_e, v_e, τ) cloudbow table; same situation. |

**Recommendation:** move to the `ve_retrieval` branch only, or delete and regenerate there when needed. Keeping them in `main` is confusing — there is no code in this branch that uses them.

---

## 4 · Possibly stale — run at the old 3 % noise model, not explicitly superseded

These scripts and their output JSONs pre-date the noise-model overhaul in commit `55e51f9`
("`Make_Se` / `oci_swir`; raise `filter_threshold` to 0.5"). All use the superseded
`Se = diag((0.03 · max(|y|, 0.02))²)` formula. DD§12 describes the migration; the *decision*
results referenced in DESIGN_DECISIONS.md were run at 3 %. Whether these need re-running at
2 % OCI noise is a judgement call — flagged here so the gap is visible.

| Script | Output JSON | Status in docs |
|---|---|---|
| `joint_osse_retrieval.py` | `docs/cached_results/joint_osse_results.json` | Definitive results, cited in DD§10c. 3 % noise. |
| `subadiabatic_thin_retrieval.py` | `docs/cached_results/subadiabatic_thin_results.json` | Cited in DD§10c (RF14/RF05 sub-adiabatic test). 3 % noise. |
| `sic_dofs_vs_k_fullgrid.py` | `docs/cached_results/sic_dofs_vs_k_fullgrid.json` | SIC vs k shape study. 3 % noise. Not cited by name in DESIGN. |
| `approx_sic_sweep.py` | `docs/cached_results/approx_sic_sweep.json` | Approximate SIC(k) sweep. 3 % noise. Not cited by name in DESIGN. |
| `fg_sic_dofs_sweep.py` | `docs/fg_sic_dofs_sweep.json` *(file does not exist in `cached_results`!)* | Writes to the wrong path (`docs/` root, not `docs/cached_results/`); result was never preserved. |
| `thick_retrieval.py` | `docs/cached_results/retrieval_baseline_linear_class.json` | 1 % noise baseline for the re5-linear vs linear comparison (OD§B′ / DD§10g, settled second-order). |
| `smoke_retrieval.py` | none | ~1 % noise smoke; plumbing-level, noise barely matters. Low priority. |
| `smoke_joint_retrieval.py` | none | 3 % noise; plumbing-level smoke test. Low priority. |

**Special note on `joint_dofs_experiment.py` / `joint_dofs_results.json`:** this script uses
`Se` built via `posterior_diagnostics`, which respects whatever `Se` is passed in — check
whether the run that produced the current JSON used 3 % or 2 %. The DD§10c table was produced
before the noise overhaul and may be at 3 %.

---

## 5 · Possibly stale — no longer referenced but not obviously harmful

| File | Notes |
|---|---|
| `tests/supplementary/bp2025_fig2_check.py` / `docs/cached_results/our_bp2025_fig2.png` | Fact-check of BP2025 Fig. 2 median VOCALS profile. Not referenced in DESIGN_DECISIONS or OUTSTANDING. Keep if used in the paper/report; otherwise marginal. |
| `docs/cached_results/retrieval_thick_RF03_tau23.json` | Raw thick RF03 re5-linear retrieval baseline (3 % noise, 2026-06-10). Referenced by `thick_retrieval.py` as a "companion" artifact; not cited in DESIGN. |
| `docs/cached_results/nquad_saturation.json` + `tests/supplementary/nquad_saturation.py` + `nquad_saturation_thick48.py` | NQuad saturation study ("Q2"). Not cited in DESIGN_DECISIONS or OUTSTANDING by name. If the verdict (DOFS plateaus by NQuad=32–48) has been incorporated into the operating-point choice, these may be archivable. |

---

## 6 · Files to keep — DO NOT delete

For clarity, the following are actively referenced and should **not** be touched:

- `tests/supplementary/generate_reference.py` — CLAUDE.md says "run once after changing tau values"; generates reference `.npz` files for the test suite.
- `tests/supplementary/demo_jit_retrieval.py` — explicitly cited in DD§7 as the recipe demo.
- `tests/supplementary/demo_deltaM_tms.py` — explicitly cited in DD§6 as the radiance-vs-angle figure.
- `tests/supplementary/check_noise_model.py` — cited in DD§12 as verification.
- `tests/supplementary/_ic_parallel.py` — cited in DD§13 (process-parallel IC sweeps); active infrastructure for Stage-2 IC runs.
- `tests/supplementary/ic_worker_*.py` (4 files) — active Stage-2b workers.
- `tests/supplementary/info_content_stage1.py` — Stage-1 IC **pilot, SUPERSEDED** by the definitive run
  (`ic_worker_profile.py` + `ic_analysis_definitive.py`); truth-linearization retired (`_linearity_probe.py`
  deleted; `info_content_linearity_probe_clim.py` kept as the climatology-linearization probe).
- `tests/supplementary/per_mode_grid_investigation.py` — cited by name in OD§G and DD§3a.
- `tests/supplementary/joint_dofs_experiment.py` / `joint_osse_retrieval.py` — produce definitive DD§10 results.
- `tests/supplementary/prior_investigation.py` — DD§11 grounding.
- `tests/supplementary/sweep_threshold_2pct.py` — the *current* (2 % noise) filter-threshold decision evidence (DD§10f).
- `tests/supplementary/summarize_joint_results.py` — reads active JSONs.
- `tests/supplementary/thin_top_resolution.py` / `stream_view_thickthin.py` — DD§11b angular-sampling finding.
- `tests/supplementary/subadiabatic_thin_retrieval.py` — DD§10c RF14/RF05 result (even though noise level is §4 above).
- `tests/supplementary/sic_dofs_vs_k_fullgrid.py` / `fg_sic_dofs_sweep.py` / `approx_sic_sweep.py` — §4 flagged, not unambiguously stale.
- `docs/cached_results/jacobian_decomposition.ipynb` — cited in DD§8 as the Jacobian-decomposition analysis notebook.
- `docs/cached_results/joint_dofs_results.json` / `joint_osse_results.json` / `prior_investigation_results.json` / `filter_threshold_sweep.json` / `info_content_*.json` — all active cited results.
- `tests/supplementary/results/_nb_cache_nospike.npz` / `_nb_cache_spike.npz` — ODE-solve cache for `adiabatic_cloud_with_drizzle.ipynb`; deliberately committed for notebook speed.

---

## 7 · IC definitive run (10-band, 2026-06-25) — superseded pilots [results REMOVED, scripts FLAGGED]

The 10-band Shapley definitive run (`ic_worker_profile.py` + `ic_worker_mechanism.py` →
`ic_analysis_definitive.py` → notebook §15; **DD§14**) supersedes the 9-band pilot, the NQuad-convergence
study, the ensemble pilot, and the truth / linearization-probe runs. **Their result JSONs were removed**
(2026-06-25); the scripts are **flagged here, not deleted** (they document the methodology trail).

### Result JSONs REMOVED (unreferenced — the notebook loads only `info_content_{definitive,mechanism}.json`)

`docs/cached_results/info_content_{stage1, ensemble, robust_priormean, robust_draw, linearity_probe,
linearity_probe_clim}.json` (the two `robust_*` were ~92 k lines each).

### PRESERVED (per repo owner, 2026-06-25) — the NQuad-convergence study

Justifies the **NQuad=48 operating point** (DOFS converges by ~NQuad 32–48; the NQuad=64 check moves it
≲0.05 DOFS): `tests/supplementary/ic_worker_nquad.py` + `docs/cached_results/info_content_nquad_profiles.json`
(NQuad 16/24/32/48 sweep over 5 profiles, τ=1.2–23.3) + `info_content_nquad64_check.json` (NQuad=64 on the
thin/thick extremes). **Keep.**

### Scripts SUPERSEDED — flag only, keep for now

| File | Superseded by |
|---|---|
| `info_content_stage1.py` | `ic_worker_profile.py` + `ic_analysis_definitive.py` (Stage-1 pilot) |
| `info_content_linearity_probe_clim.py` | truth-linearization **retired** (DD§14); IC shown insensitive to the linearization state |
| `ic_worker_ensemble.py` | folded into `ic_worker_profile.py` (the per-profile worker) |
| `nquad_saturation.py` + `nquad_saturation_thick48.py` | older NQuad-saturation study (Q2, §5) — distinct from the **preserved** NQuad-convergence study above |

**Corrects §6:** only **`ic_worker_profile.py`** and **`ic_worker_mechanism.py`** are the active IC workers
(not 4); `_ic_parallel.py` is local process-parallel infra — the definitive run uses HPC SLURM, so review
whether it is still needed. Active/keep: `ic_analysis_definitive.py`, `ic_tau_bot_check.py` (DD§14 τ_bot
sensitivity), `validate_optics_table.py`.
