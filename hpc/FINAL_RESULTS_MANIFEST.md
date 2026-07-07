# Batch-3 final results manifest (2026-07-06)

The consolidated FR + IC deliverable after the efficiency refactor, the sign-bug corrections, and
the post-hoc re-runs. Companion to `AGENT_batch3_postmortem.md` (full history + FINAL STATUS).
Sanity-swept over every result: **0 genuine failures / bugs.**

## FR — 125 profiles × 2 configs (A, B) = 250

**Base:** `docs/cached_results/_fr_parts/{idx}_{A,B}.{npz,json}` (all 125 profiles, both configs).

**Supersession (a corrected result replaces the bugged/misfit original).** Priority, highest first:

1. **Post-hoc, new code** (uncached sign-bug victims re-run on the reconciled refactor):
   - `runs/_fr_negrel_posthoc/`: **9A, 34B, 45B, 78A, 80A**
   - `runs/_fr_continuation_posthoc/`: **5A, 119A**
   - (These worker runs also produced the sibling config as a byproduct — the **sibling is NOT
     superseded**; it keeps its `_fr_parts` result, e.g. 78B/80B.)
2. **Continuation / continuation-verify** (cached sign-bug victims, continued on the *original*
   grid — the clean victim test): `_fr_parts_continuation/` + `_fr_parts_continuation_verify/` +
   `_fr_parts_negrel_verify/` (e.g. **28A → 0.0054**, 100A, 110B, 75B, and the 20-config batch).
3. **Remesh** (genuine grid-inadequacy, moved grid, `max_n_outer=2`): `_fr_parts_remesh/` —
   **11A, 26A, 52B, 68B, 42B, 12B, 22B**, and **124B** (non-pathological *re-draw*, replacing the
   τ_bot=0.54 pathological draw). 12B/22B report `conv=False` = hit the iter-cap **at** the χ²
   floor (fine). Adjudication rule: continuation supersedes if it descends on the original grid
   (28A); otherwise the remesh (moved grid) supersedes (26A, 11A).

Every superseded original is a bugged premature-stop (χ² 3–1036) whose correction sits at the χ²
floor with τ_bot recovered. `conv=False` on thin 20A/20B is the iter-cap technicality, not a failure.

## Low-confidence flags (~22 configs, d_rmse < −1 µm on the canonical result)

Good data-fit (χ² at floor) but the retrieved r_e(τ) underperforms the naive adiabatic baseline —
inherent OE ill-posedness (worst deep in optically-thick cloud), **present identically pre- and
post-refactor; NOT a bug, NOT a re-run trigger.** Flag these in downstream analysis. Most severe:
**110B (−5.6), 28B (−5.2), 49B (−5.0), 29B (−3.2), 56B (−3.1), 100A (−2.8), 119B (−2.8), 78B (−2.6)**,
then 66A, 62B, 42B, 32B, 102A, 80B, 13B, 101B, 57B, 109B, 119A, 85A, 123A/B (−1.0 … −2.3).

## IC — information content (canonical = the refactor re-run on the fixed forward)

`runs/_ic_{A,B,C}_parts/` (refactor tree), A=priormean, B=draw, C=mechanism; 125 real + idx-0 skip
per array. Raw Jacobian K-sidecars (`_ic_{A,B}` `.npz`) are the product — all non-null, finite,
fully populated (the historical physical-vs-log null-Jacobian bug is absent). DOFS/SIC re-derived
downstream by `scripts/ic_analysis_definitive.py`. The pre-refactor `_ic_*` are superseded.

## Notes

- Result **data** (`.npz`/`.json`) is git-ignored and moves as a bundle (zip), never via `git pull`.
- Provenance for each corrected result is in its own JSON (`remesh_provenance` / `continuation_provenance` / `draw_info`).
- Validation of the code that produced these: `CHANGELOG.md` + the `tests/` suites (float32 68/68,
  float64 26/26 vs PythonicDISORT) + `tests/hpc/` L1/L2 equivalence gates. The old golden-probe gate
  is retired (stale/different-grid reference).
