# Delegated task — GPU probe #3: float32 viability + tol sufficiency + GPU-suitability

*(Hand-off for a Sonnet agent. Experiment on results we already trust. **Push only
`docs/cached_results/results.md`** (the §A3 narrative); **bundle the result data** as
`precision_probe_out.zip` — do NOT commit npz/json data. The primary decides + consolidates.
2026-06-28.)*

> **Update (monitoring agent, 2026-06-28) — two operational corrections from the primary's env:**
>
> 1. **V100S is fully f64-viable, alongside A100** (user ruling). Part A's "A100 only" is
>    relaxed: run the f64 gold/reference/retrieval tier on **a100 OR v100s** (`-C "a100|v100s"`).
>    Both are full-rate (1:2) FP64; RTX 8000 + A40 are 1:32 (correct but ~32× slow) → still f32-only
>    candidates, which is what this probe decides.
> 2. **Account/partition map (verified via `sacctmgr`/`sinfo`, not folklore).** The `crew` and
>    `apam` accounts are identical in QOS/priority/fairshare — the only difference is the **private
>    partitions** they unlock. **GPU jobs → `--account=crew`**, which opens two *less-contended* GPU
>    partitions the shared `short` pool doesn't: **`crew1`** (dedicated: 2×v100s, 2×rtx8000, 1×a40)
>    and **`ocp_gpu`** (shared w/ ~11 labs: 2×v100s, 2×rtx8000) — both **MaxTime 7 d** vs short's
>    12 h. **A100 lives ONLY in shared `short`/`burst`** (none in crew1/ocp_gpu). `apam` only unlocks
>    `apam1` = **CPU-only** (mem192). Recommended GPU submit:
>    `--account=crew --partition=crew1,ocp_gpu,short -C "<feature>"` (scheduler picks soonest-free;
>    keep `--time ≤ 12h` to keep `short` eligible, or drop `short` for v100s-only runs needing >12 h).
>    → Route v100s/rtx8000/a40 work to **crew1+ocp_gpu** (dedicated, 7-day wall); use `short` only
>    when you specifically need an **a100**.

## Why (accuracy tiers — keep straight)

- **Truth tier** (radiances + IC) = high accuracy (float64, tol\*≈3e-5). Already set.
- **Operational tier** (the retrieval) MAY run at lower precision/tol — cheaper, a known accepted
  bias. **Stability is non-negotiable** (NaN / `max_steps` / non-convergence = fail); higher
  *error* is fine.

This probe runs the **production retrieval worker** (`retrieval_worker.py`, now osse_config-based)
exactly as batch-3 will — it **loads a high-accuracy gold radiance cache** (the fixed truth) and
inverts it with an **operational-precision forward**. So the probe IS the operational path, not a
parallel script. It measures, on GPU: is **float32** viable? is **tol=1e-4** as accurate as 1e-5?
which **GPUs** run float64 correctly? If float32 is viable it unlocks RTX 8000 (36) + A40 (18) at
full FP32 → the GPU pool goes 24 → up to 78.

## Profiles (global hard axes + a control)

| role | idx | why |
|---|---|---|
| stiffest | **49** | longest runtime 250 min, τ=36.5 |
| thickest | **40** | τ=51.5 — Riccati-invariant + f32 overflow stress |
| jaggedest | **47** | 112 nodes |
| control | **20** | 21 nodes, τ=1.5 — should be precision/tol-insensitive (null) |

Part-B canary = the 3 longest **cross-verified** profiles (known-good answer): **115, 120, 121**.

## Setup

```bash
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT && git fetch origin && git reset --hard origin/main
PY=<GPU-overlay python from probe #1/#2>                            # jaxlib + 0.10.2 cuda plugin
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz
export VOCALS_DATA=<VOCALS_REx_data path on the cluster>
GOLDDIR=$ROOT/tests/supplementary/probe_gold_parts        # per-index gold sidecars
GOLD=$ROOT/tests/supplementary/osse_radiances_gold.npz    # consolidated gold cache (tol=1e-5)
OUT=$ROOT/tests/supplementary/precision_probe_out         # worker outputs (the deliverable)
mkdir -p $GOLDDIR $OUT
```
Knobs: `PYDISORT_RICCATI_JAX_X64` (1=f64, 0=f32), `SOLVER_TOL` (ODE tol of the *forward*),
`MODE_MAP=vmap` (240-way GPU path), `RADIANCE_CACHE` (which truth cache to invert), `RADIANCE_TOL`
(the expected truth tol — the worker refuses a wrong-accuracy cache). Run on GPU
(`JAX_PLATFORMS=cuda`). **Part A on A100 only** (confirmed-safe; do not spend f64 truth/reference
runs on unverified cards). Part B is the card test.

## Step 0 — SMOKE TEST (one cheap end-to-end run; do not skip)

`retrieval_worker.py` was de-staled + wiring-checked but NOT run to completion locally. Validate it
end-to-end on the control before the matrix:

```bash
# (a) gold radiance for idx-20 at f64/tol=1e-5, then consolidate to a 1-profile gold cache
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
  tests/supplementary/generate_osse_radiances.py 20 $GOLDDIR
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 $PY \
  tests/supplementary/generate_osse_radiances.py consolidate $GOLDDIR ${GOLD%.npz}_smoke.npz
# (b) retrieve idx-20 at the gold setting (inverting that cache)
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap \
  RADIANCE_CACHE=${GOLD%.npz}_smoke.npz RADIANCE_TOL=1e-5 $PY \
  tests/supplementary/retrieval_worker.py 20 $OUT/smoke_20
```
Expect a `[20] RF03 … DONE … A: dRMSE=… conv=True | B: … conv=True`. If it errors, fix it (the
worker reuses tested pieces; likely env paths or a kwarg) and note the fix. **Proceed only when
clean.**

## Part A — float32 viability + tol sufficiency (A100)

```bash
# 1) gold radiances (f64/tol=1e-5) for the 4 profiles -> one gold cache
for i in 20 40 47 49; do
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap $PY \
    tests/supplementary/generate_osse_radiances.py $i $GOLDDIR ; done
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 $PY \
  tests/supplementary/generate_osse_radiances.py consolidate $GOLDDIR $GOLD

# 2) retrieve each profile at the 3 operational settings (invert the SAME gold cache)
for i in 20 40 47 49; do
  # float32 / tol=1e-3
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=0 SOLVER_TOL=1e-3 MODE_MAP=vmap \
    RADIANCE_CACHE=$GOLD RADIANCE_TOL=1e-5 $PY \
    tests/supplementary/retrieval_worker.py $i $OUT/probe_${i}_f32_tol1e-3
  # float64 / tol=1e-4
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap \
    RADIANCE_CACHE=$GOLD RADIANCE_TOL=1e-5 $PY \
    tests/supplementary/retrieval_worker.py $i $OUT/probe_${i}_f64_tol1e-4
  # float64 / tol=1e-5  (reference)
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-5 MODE_MAP=vmap \
    RADIANCE_CACHE=$GOLD RADIANCE_TOL=1e-5 $PY \
    tests/supplementary/retrieval_worker.py $i $OUT/probe_${i}_f64_tol1e-5
done
```
Each worker run writes `<prefix>_A.{npz,json}` + `_B.{npz,json}` (two prior configs; **compare
config A** across settings — B is a reproducible draw, secondary). Every JSON carries
`precision / tol / mode_map / radiance_tol` provenance. The 12 runs (×2 configs) are independent →
fan across the A100 pool.

**Verdicts (per profile, vs the f64/tol=1e-5 reference, config A):**
- **float32 viable?** Does `f32_tol1e-3` give `converged=true` (mon.A) with finite x̂? Bias =
  `|rmse_ours(f32) − rmse_ours(f64/1e-5)|` — report it (bias OK, de-stabilization not).
- **tol=1e-4 sufficient?** Is `f64_tol1e-4` ≈ `f64_tol1e-5` (rmse_ours, d_rmse, tau_bot_ret)?
- **control idx-20** should show all three ~identical — if not, suspect a bug.

## Part B — GPU-suitability canary (LOWER priority; after Part A)

Silent-FP64 test: a **known-answer** radiance forward at f64/tol=1e-4 on each suspect card vs the
**A100** at the same setting (gross diff ⇒ card silently corrupts f64 ⇒ unusable). One per card:

```bash
for i in 115 120 121; do                                   # A100 references
  JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap $PY \
    tests/supplementary/generate_osse_radiances.py $i $OUT/canary_ref_A100 ; done
# suspect cards (matching --constraint/--gres), same setting:
#   idx115 -> V100S , idx120 -> RTX 8000 , idx121 -> A40   into canary_<card>/
JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 SOLVER_TOL=1e-4 MODE_MAP=vmap $PY \
  tests/supplementary/generate_osse_radiances.py 115 $OUT/canary_V100S    # ...120 RTX8000, 121 A40
```
Per card report: ran (or crash/OOM)? `max|y_card − y_A100|` (same tol) — ≲1e-6 if FP64 honest,
**flag any gross diff / NaN as silent failure**; plus wall-time. V100S expected fine (Volta full
FP64); RTX 8000 the doubtful one (Turing + diffrax implicit FP64). These sidecars carry the
**tol=1e-4** tag, distinct from the truth cache.

## Report (append §A3 to docs/cached_results/results.md; push ONLY that file)

- **Part A table**: (profile, setting) → ran/finite/converged/n_gn/rmse_ours/d_rmse/dofs/runtime
  (config A). Then the two verdicts with bias magnitudes; confirm idx-20 is the flat null.
- **Part B table**: per card → ran? `max|y−y_A100|`/NaN? wall-time → usable / silently-wrong /
  crashes.
- Any Step-0 fix you made.

## Result handoff + tracking (the primary consolidates many sets — do not blur them)

- **`git push` only `results.md`.** Everything in `$OUT` (+ the gold cache) → **`zip -r
  precision_probe_out.zip $OUT $GOLD`** and leave it for the primary to retrieve manually. Do NOT
  commit npz/json.
- Provenance is already in every JSON (precision/tol/mode_map/radiance_tol) — keep `$OUT` intact.
- Do not touch the real radiance cache (`rad_bundle/…`). The gold cache is separate (`$GOLD`).
- **Standing rule:** no job cancellations / no dynamic GPU migration without the primary's express
  permission. Static fan-out only.

*(`retrieval_worker.py` was just de-staled onto osse_config — NLeg_all=1024, the irregular 24-view
fan, the radiance cache. The old NLeg_all=128 is gone. This probe both validates that worker and
answers the precision/tol/GPU questions for batch-3.)*
