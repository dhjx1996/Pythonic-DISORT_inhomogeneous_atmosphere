# Results manifest — provenance of every OSSE / probe result set

*Single source of truth for "what is this file, at what precision/tol, on what hardware, is it
canonical or superseded." Rule: **rather re-run than mix corrupted results.** Data files (.npz)
are delivered by zip/bundle and are NOT committed; this manifest (committed) tracks them. Updated
2026-06-28.*

## Signature scheme (the anti-mixing gate)

`osse_config.signature()` fingerprints the observing system + forward settings; `load_radiance`
asserts a match before use. **`tol` is now IN the signature** (2026-06-28) so radiances at different
ODE tolerances can never share a hash:

| setting | signature hash | notes |
|---|---|---|
| f64, tol=1e-3 | `f37f3314846eb928` | current scheme |
| f64, tol=1e-4 | `4edc7c26ebebc6a9` | Part-B canary radiances |
| f64, tol=1e-5 | *(compute on use)* | gold / truth tier |
| f64, tol=1e-3, **pre-tol-scheme** | `543eee296e1022f7` | the rad-batch cache (tol was NOT yet in the payload) |

The `543…` cache therefore no longer validates under the current scheme — **intended**: it is the
provisional under-converged batch, to be replaced by the tol\* re-gen.

## Radiance caches (truth tier — `y = F(truth)`)

| file | source | n | precision/tol | signature | status |
|---|---|---|---|---|---|
| `rad_bundle/osse_radiances.npz` | HPC (job 8612305) | 124 | f64 / tol=1e-3 | `543…` | superseded |
| `rad_bundle/osse_radiances_125.npz` | HPC 124 + **idx-20 jovyan** | 125 | f64 / tol=1e-3 | `543…` | **PROVISIONAL** — thick profiles ~1–2 % under-converged; orphaned by the tol-in-signature change; use only for tol=1e-3 work |
| *(future)* tol\* re-gen | HPC GPU (dual pipeline) | 125 | f64 / **tol\*≈3e-5** | new (tol in payload) | **CANONICAL once produced** |

- idx-20 (RF03, τ=1.5, 21 nodes) was computed on jovyan; cross-verified class (it is in the
  bit-identical regime, so it matches what the HPC forward would give to ~1e-6).
- idx-0 is a degenerate auto-skip (τ≈1585); the productive set is 1..125.

## tol-convergence (why a re-gen is needed)

jovyan study, idx-105 (worst cross-checked, 64-node) band-3: tol=1e-3 is **1.13 % / 5.8e-3** off the
tol=1e-5 reference — **over the 1 %/1e-3 bar**. Ladder: 3e-4→7e-4, 1e-4→4e-4, **3e-5→4.4e-5**.
→ **tol\* = 3e-5** (≈23× margin, covers the un-tested 74–112-node profiles; cost ~3.3× tol=1e-3,
absorbed by the GPU). *This is the GENERATION (native) tol; the retrieval-forward (coarse) tol is a
separate, looser question — probe #3 Part A.*

## Cross-verification (jovyan vs HPC, the 28 overlapping profiles 95–125 + 20)

17/28 bit-identical (<1e-5); 7 deviate >1e-4, **all high node-count** (33–88), deviation ∝ stiffness
(idx-105 worst at 1.26e-2). Confirms: tables identical, precision identical → the spread is purely
adaptive-step-sequence divergence on hard profiles (the tol issue, not a bug).

## GPU probes (in `docs/cached_results/results.md`)

| probe | where | finding |
|---|---|---|
| #1 modes-vmap (1 band) | results.md §A | GPU-vmap jacfwd 28.3 s vs CPU-scan 197.3 s → **~7×** |
| #2 bands×modes 240-way (10 band) | results.md §A2 | `vmap_both` jacfwd 127.7 s, **2.3×** over modes-only, **~17×** vs CPU, 13.9/40 GB, bit-faithful |
| #3 precision/tol + GPU-suitability | results.md §A3 *(pending)* | float32 viability + tol=1e-4 sufficiency (Part A) + V100S/RTX8000/A40 silent-FP64 canary (Part B). Data → `precision_probe_out.zip` |

## Solver / code state

- `mode_map='scan'|'vmap'` in `SetupData`/`riccati_setup`/`_fourier_solve`/`RetrievalForward`
  (709dec0 modes, e362fad bands×modes). CPU-validated bit-identical (jacfwd rel ~1e-12).
- `osse_config.build_forward` + `generate_osse_radiances` now take `tol` (via `SOLVER_TOL`) +
  `mode_map` (via `MODE_MAP`); tol is in `signature()`.

## OPEN flags (must clear before batch-3)

- **`retrieval_worker.py` is STALE**: `NLeg_all=128` (pre-TMS-fix) + wrong default table → would
  corrupt short bands. MUST refactor onto `osse_config` (task #31) before any real retrieval batch.
  (`retrieval_precision_probe.py` is osse_config-based and unaffected.)
- **GPU pool unverified beyond A100**: V100S (likely OK), RTX 8000 / A40 (crippled FP64, diffrax
  compatibility doubtful) — probe #3 Part B settles it.
- **Accuracy tiers**: radiances + IC at high accuracy (f64, tol\*); retrievals at operational
  precision/tol if probe #3 shows it stable (a known, accepted bias).
