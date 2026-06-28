# Delegated task — A100/GPU probe #3: float32 viability + tol sufficiency + GPU-suitability

*(Hand-off for a Sonnet agent. Read-only experiment on results we already trust — **no commits
of result data**; report numbers + push only `docs/cached_results/results.md`. The primary
decides. 2026-06-28.)*

## Why (accuracy tiers — keep these straight)

The pipeline has TWO accuracy tiers:
- **Truth tier** (radiances + information-content profiling) → **high accuracy** (float64, tol\*≈3e-5);
  must sit below the practical-significance bar (1 % / 1e-3). The tol study already set tol\*.
- **Operational tier** (the retrieval itself) → may run at **lower precision/tol** (cheaper; a known,
  accepted bias) because that is what an operational retrieval actually does. **Stability is
  non-negotiable** (NaN / `max_steps` / non-convergence = fail); higher *error* is fine.

This probe measures, on the GPU, whether the operational tier can be **float32** and/or **loose tol**
without de-stabilizing, and whether the candidate **tol=1e-4** is as accurate as tol=1e-5 — by
inverting a **fixed gold observation** (`y=F(truth)` at f64/tol=1e-5) with operational-precision
forwards. It also (lower priority) settles whether the crippled-FP64 cards run f64 *correctly*.

If float32 is viable it unlocks the RTX 8000 (36) + A40 (18) cards at full FP32 throughput — turning
the 24-GPU real-FP64 pool into up to 78. That is the prize.

## Profiles (chosen across the hard axes + a control — global metrics, not a subset artifact)

| role | idx | why |
|---|---|---|
| stiffest (most ODE steps) | **49** | longest HPC runtime 250 min, τ=36.5 |
| thickest | **40** | τ=51.5 — stresses the no-positive-exponent Riccati invariant + f32 overflow headroom |
| jaggedest | **47** | 112 native nodes |
| easy control | **20** | 21 nodes, τ=1.5 — should show ZERO precision/tol sensitivity (null anchor) |

GPU-suitability canary profiles (Part B) = the 3 longest-runtime profiles we **already
cross-verified** (jovyan==HPC to <1e-4), so the correct answer is known: **115, 120, 121**.

## Setup

```bash
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT && git fetch origin && git reset --hard origin/main      # picks up probe #3 scripts + osse_config tol/mode_map
PY=$ROOT/...                                                       # the GPU-overlay python from probe #1/#2 (jaxlib+0.10.2 plugin)
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz
export VOCALS_DATA=<the VOCALS_REx_data path on the cluster>
export GOLD_DIR=$ROOT/tests/supplementary/precision_probe_gold
export PROBE_OUT=$ROOT/tests/supplementary/precision_probe_out
```
Precision is the `PYDISORT_RICCATI_JAX_X64` env (1=float64, 0=float32). `SOLVER_TOL` sets the ODE
tol. `MODE_MAP=vmap` uses the 240-way bands×modes GPU path (probe #2: ~17×). **Run everything on a
GPU** (`JAX_PLATFORMS=cuda`). Part A on **A100 only** (the confirmed-safe card — do NOT spend f64
reference runs on unverified cards). Part B is the card test.

## Step 0 — SMOKE TEST first (one cheap end-to-end run; do not skip)

The probe script (`retrieval_precision_probe.py`) was wiring-checked but NOT run to completion
locally. Confirm it runs end-to-end on the easy control before the matrix:

```bash
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py gold 20
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py retrieve 20
```
Expect a `gold_20.npz` then a `[probe 20 float64_tol1e-5_vmap] {ran:true, finite:true,
converged:true, rmse_ours:..., n_gn:...}`. If it errors, fix the script (it reuses tested
`gauss_newton_oe` / `osse_config.build_forward`; the likely culprits are env paths or a kwarg) and
note the fix in your report. **Only proceed once the smoke test is clean.**

## Part A — float32 viability + tol sufficiency (A100; 4 profiles × {gold + 3 retrieves})

For each profile `i` in **20, 40, 47, 49** (control first):

```bash
# (1) gold observation: f64 / tol=1e-5 (ONCE per profile)
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py gold $i
# (2) retrieve at the three operational settings (invert that same gold):
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=0 SOLVER_TOL=1e-3 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py retrieve $i        # float32 / tol=1e-3
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py retrieve $i        # float64 / tol=1e-4
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
  tests/supplementary/retrieval_precision_probe.py retrieve $i        # float64 / tol=1e-5  (reference)
```
Outputs land in `$PROBE_OUT/probe_<i>_<precision>_tol<..>_vmap.{npz,json}` — provenance in every
file + name. The 16 runs are independent; fan them across the A100 pool.

**What it answers (per profile, all vs the f64/tol=1e-5 reference retrieval):**
- **float32 viable?** Does `float32_tol1e-3` give `ran=true, finite=true, converged=true`? If so,
  how big is `|rmse_ours(f32) − rmse_ours(f64/1e-5)|`? (Bias is OK; de-stabilization is not.)
- **tol=1e-4 sufficient?** Is `float64_tol1e-4` ≈ `float64_tol1e-5` (rmse + x̂)? If yes, 1e-4 suffices
  for the retrieval tier.
- **control idx-20** should show all three nearly identical (null) — if it doesn't, suspect a bug.

## Part B — GPU-suitability canary (LOWER priority; do after Part A)

Silent-FP64-failure test: run a **known-answer** radiance forward at **f64/tol=1e-4** on each suspect
card and diff against the **A100** result at the *same* setting (gross diff ⇒ the card silently
corrupts f64 ⇒ unusable). One profile per card:

```bash
# A100 references (trusted) for the 3 canary profiles:
for i in 115 120 121; do
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap $PY \
    tests/supplementary/generate_osse_radiances.py $i $PROBE_OUT/canary_ref_A100 ; done
# suspect cards (one profile each), same setting:
#   idx115 -> V100S , idx120 -> RTX 8000 , idx121 -> A40   (use the matching --constraint/--gres)
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap $PY \
  tests/supplementary/generate_osse_radiances.py 115 $PROBE_OUT/canary_V100S
# ... idx120 on RTX 8000, idx121 on A40, into canary_<card> dirs.
```
For each suspect card report: did it **run** (or crash/OOM)? `max|y_card − y_A100|` (same tol=1e-4) —
expect ≲1e-6 if FP64 is honest, **flag any gross diff or NaN as silent failure**. Also note the
wall-time (crippled FP64 may run but be slow). V100S is expected fine (Volta full-rate FP64);
RTX 8000 is the doubtful one (Turing + diffrax implicit FP64). The `osse_*.npz` sidecars these
write carry signature hash for **tol=1e-4** (`4edc7c26ebebc6a9`) — distinct from the tol=1e-3 cache,
so they cannot be confused with the real radiances.

## Report back (append a §A3 to docs/cached_results/results.md; push only that file)

- **Part A table**: per (profile, setting) → ran / finite / converged / n_gn / rmse_ours / d_rmse /
  dofs / runtime_s. Then the two verdicts: **float32 viable? tol=1e-4 sufficient?** with the bias
  magnitudes, and confirm idx-20 is the flat null.
- **Part B table**: per suspect card → ran? `max|y−y_A100|` / NaN? wall-time. Verdict per card:
  **usable for f64 / silently-wrong / crashes**.
- Note any script fix you made in Step 0.

## Result-tracking discipline (the primary is consolidating many disparate sets — do not blur them)

- Everything you produce is **provenance-tagged** already (precision, tol, mode_map, node in each
  JSON; tol in the radiance signature). Keep the `$PROBE_OUT` tree intact; **zip it back** as
  `precision_probe_out.zip` alongside `results.md` (do NOT commit the npz/json data — bundle only).
- Do **not** touch / overwrite the radiance cache (`osse_radiances*.npz`) — Part B writes to
  separate `canary_*` dirs.
- **Standing rule reminder:** no job cancellations / no dynamic GPU migration without the primary's
  express permission. Run the static fan-out; the primary handles any reassignment manually.

## Caveat the primary already knows (don't act on it, just don't trip over it)

`tests/supplementary/retrieval_worker.py` (the eventual batch-3 script) still has `NLeg_all=128`
(pre-TMS-fix) + the wrong default table — it is **stale** and must be refactored onto `osse_config`
before batch-3. This probe deliberately uses `retrieval_precision_probe.py` (osse_config-based,
NLeg_all=1024), so it is unaffected. Do not use `retrieval_worker.py` here.
