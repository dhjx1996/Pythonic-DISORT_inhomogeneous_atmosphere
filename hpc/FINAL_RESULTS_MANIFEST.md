# Batch-3 final results manifest (2026-07-06)

The consolidated FR + IC deliverable after the efficiency refactor, the sign-bug corrections, and
the post-hoc re-runs. This file is the **standalone canonical record** (the run-by-run postmortem
was diffused into `STRATEGY_hpc_retrieval_runs.md` + the `AGENT_all125_*.md` banners +
`docs/DESIGN_DECISIONS.md` §16 on 2026-07-07; full history in git). Sanity-swept over every
result: **0 genuine failures / bugs.**

**The sign-bug, for the record.** `_gn_inner`'s cost-stagnation stop compared the *signed*
relative data-misfit change (`rel < cost_rtol`, no `abs()`); an accepted LM step can make `rel`
negative (the prior term absorbs the difference), so one large data-fit regression satisfied the
test and stopped a still-converging solve. Fixed 2026-07-04 (`abs(rel) < cost_rtol`); a
full-population audit found 29/250 configs technically affected, of which only 26A/42B/68B were
interrupted far from the χ² floor — every affected config was re-run/continued and resolves to an
unbugged artifact under the canonical rule below (single exception: 26A, rule-4 caveat).

## FR — 125 profiles × 2 configs (A, B) = 250

**Base:** `runs/_fr_parts/{idx}_{A,B}.{npz,json}` (all 125 profiles, both configs). All result parts
dirs named below live under `runs/` (git-ignored worker outputs). The notebook's cached inputs
(`osse_radiances.npz`, `info_content_{definitive,mechanism}.json`) stay in `docs/cached_results/`.

**CANONICAL-SET RULE (user, 2026-07-07 — supersedes the earlier priority ladder).** The analysis
population is exactly **250 configs (125 × {A,B}) + 1**; when multiple versions of a config exist,
the canonical one is **the earliest result an *unbugged* run would have produced**. All other
versions (re-meshes, later re-runs) are *past/obsolete — instructive only*. Mechanically
(implemented in notebook §16's loader; per-config provenance recorded in
`docs/cached_results/retrieval_summary.json`):

1. **Fresh new-code re-runs of the UNCACHED sign-bug victims** —
   `_fr_negrel_posthoc/`: **9A, 34B, 45B, 78A, 80A**; `_fr_continuation_posthoc/`: **5A, 119A**.
   Sibling-config byproducts in these dirs do **NOT** supersede (78B/80B etc. keep base).
   **119A is the twice-re-run case**: its 07-04 continuation was itself produced by bugged code,
   so the post-hoc run is the earliest *unbugged* one.
2. **Same-grid continuations under the fixed criterion** (= the completion the unbugged run would
   have reached): the 20-config batch + 119B (`_fr_parts_continuation/`), the 62-config verify
   set incl. **28A** (`_fr_parts_continuation_verify/`; for never-fired configs this differs from
   the original by ≤1e-3 χ²ᵣ), and 75B/100A/110B (`_fr_parts_negrel_verify/`).
3. **42B, 68B**: bug-caused stops with no persisted continuation — their `_fr_parts_remesh/`
   re-runs are **grid-identical to base and the re-mesh gate never fired**, i.e. plain unbugged
   fresh runs on the original grid → canonical.
4. **Base (`_fr_parts/`) for everything else** — including the flagged-but-**legitimate** stops
   **11A (χ²ᵣ 14.8), 52B (44.4), 12B (4.6), 22B (4.2)** (final |rel| < 1 %: the stagnation
   criterion fires with or without the sign fix, so the original *is* the unbugged outcome) and
   **124B, the catastrophic-but-VALID draw** (τ_bot drawn 0.54 vs truth 5.6 → zero accepted
   steps, χ²ᵣ = 1036 — a legitimate config-B outcome). Their remesh versions are obsolete.
   *26A caveat:* bug-caused, but no same-grid unbugged artifact exists (its continuation test was
   not persisted; its remesh MOVED the grid) → base (χ²ᵣ 3.04) stands as the nearest
   earliest-unbugged approximation (the tested continuation stayed elevated ≈ base).

**The “+1”:** the **124B re-draw** (`_fr_parts_remesh/124_B.npz`, `draw_info` present; truth-free
climatology-filtered draw, seed 3000+idx) is carried **alongside, outside the 250** — the
fortune-adjusted version of the one catastrophic draw (χ²ᵣ 1036 → 0.007).

Resolution count: 94 of 250 configs resolve to a non-base artifact (2 posthoc-continuation +
5 posthoc-negrel + 20 continuation + 62 verify + 3 negrel-verify + 2 same-grid remesh).
`conv=False` on 20B is the iter-cap-at-floor technicality, not a failure.

## Low-confidence flags (25 configs, worst-decile d_W1 ≤ −0.335 τ on the canonical set)

Metric note: the primary error is now the **1-Wasserstein distance W1** (RMSE retired 2026-07-08 —
it penalized the in-situ truth's jaggedness and its verdict flipped under τ_bot bookkeeping);
low-confidence = the worst decile of d_W1 = W1_adia − W1_ours (self-scaling, no magic constant).
Good data-fit (χ² at floor) but the retrieved r_e(τ) underperforms the oracle adiabatic baseline in
W1 — inherent OE ill-posedness (worst deep in optically-thick cloud; 8 × A, 17 × B; median τ_bot ≈ 26
vs population ≈ 10), **NOT a bug, NOT a re-run trigger** — flag, don't drop. Most severe (d_W1 τ):
28B (−4.63), 100A (−4.33), 110B (−4.21), 56B (−3.21), 42B (−3.09), 78B (−2.91), 13B (−2.35),
124B (−2.32). Auto-flagged by `scripts/retrieval_analysis.py`; full list (members + threshold) in
`docs/cached_results/retrieval_summary.json` (`summary.flags.low_confidence`).

## IC — information content (canonical = the refactor re-run on the fixed forward)

`runs/_ic_{A,B,C}_parts/` (the refactor re-run), A=priormean, B=draw, C=mechanism; 125 real + idx-0
skip per array. Raw Jacobian K-sidecars (`_ic_{A,B}` `.npz`) are the product — all non-null, finite,
fully populated (the historical physical-vs-log null-Jacobian bug is absent). DOFS/SIC re-derived
downstream by `scripts/ic_analysis_definitive.py`. The superseded pre-refactor IC has been removed.

## Notes

- **18A / 92A (very minor caveat, user-confirmed 2026-07-04):** uniquely among the
  unknown-sign-bug-status set these two could not be *empirically* continuation-verified (their
  L2 setup grid post-dates and differs from the sidecar grid — ordinary cross-version
  grid-select sensitivity, not a defect); they are verified **analytically by fit quality**
  (deeply converged, χ²_red 0.0021 / 0.0121, n_gn 12 / 10, no re-mesh) and kept as final.
- **Near-threshold reproducibility:** `structural_misfit` at χ²_red ≈ the 2.0 gate is not
  perfectly hardware-reproducible (GPU float noise can displace a marginal stagnation stop —
  observed on 11A); a precision caveat at the margin only, not on the flagged set's validity.
- Result **data** (`.npz`/`.json`) is git-ignored and moves as a bundle (zip), never via `git pull`.
- Provenance for each corrected result is in its own JSON (`remesh_provenance` / `continuation_provenance` / `draw_info`).
- Validation of the code that produced these: `CHANGELOG.md` + the `tests/` suites (float32 68/68,
  float64 26/26 vs PythonicDISORT) + `tests/hpc/` L1/L2 equivalence gates. The old golden-probe gate
  is retired (stale/different-grid reference).
