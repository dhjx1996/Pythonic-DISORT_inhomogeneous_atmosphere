# IC batch-2 — PRELIMINARY report (delegated Sonnet agent → main agent)

**Date:** 2026-06-30 · **Status:** ~95 % complete, ~40 tasks still finishing on GPU (ETA ~1–3 h).
All 126 VOCALS profiles are being re-run on the **fixed** forward (sig `d71a8559bbe457e8`:
`NLeg_all=1536`, `NFourier=24`, `tol=1e-4`), three modes (A=priormean, B=draw, C=mechanism).

This is preliminary so the main agent can start reviewing the approach; the final bundle + sanity
check follow at completion.

## Current coverage (target = 125 ok / 1 skip per array)

| Array | mode | ok | skip | note |
|---|---|---|---|---|
| A | priormean | 124 | 1 | 1 OOM re-run (idx 88/93) still finishing |
| B | draw | 106 | 1 | thin + OOM re-runs finishing |
| C | mechanism | 103 | 1 | tail + thin + OOM re-runs finishing |

idx-0 (RF01, τ≈1585) is the lone degenerate skip — same 125 profiles as the batch-1 radiance
cache. **No science anomalies**; all failures were infrastructure (below), all fixed.

## Deliverable (manual move, NOT git — unchanged pattern)

Raw K sidecars are the product → bundled into `ic_bundle.zip` at completion and moved to
`/burg-archive/home/dh3065/cloud_profile_retrieval/`. On disk now:
`_ic_{A,B,C}_parts/<idx>.json` (+ `<idx>.npz` for A/B profile workers; C mechanism worker writes
JSON only — there is **no** `_ic_C_parts/*.npz`).

## Bugs found & fixed (all infrastructure/handoff — none in the physics)

1. **Out-path `IsADirectoryError`** — `ic_worker_profile.py` / `ic_worker_mechanism.py` use
   `sys.argv[2]` *directly* as the output file (unlike the rad worker, which appends `osse_{idx}`).
   The handoff passed the bare dir `_ic_A_parts` → crash **after** the full Jacobian. **Fix:**
   per-index `_ic_X_parts/$SLURM_ARRAY_TASK_ID.json` (handoff srun lines corrected + pushed).
2. **Missing `.json`** — the first path fix dropped the extension → files named `21` not
   `21.json` (valid data, but breaks `cp *.json` bundling). **Fix:** `.json` in the path;
   priormean's 124 already-written files renamed on disk (no re-run needed).
3. **OOM at 32 G** — the heavy/thick tail SIGABRTs (`Aborted (core dumped)`) in the XLA **CPU
   host** threadpool even at 32 G (≈9/126 per mode). **Fix:** re-ran the heavy tail at **64 G**;
   handoff mem guidance updated.
4. *(infra)* CPU-offload attempts first wrote to node-local `/local` (`$SCRATCH`, invisible on
   compute nodes); `apam1` is low-priority `burst` QOS and won't backfill long walls. **Fix:**
   outputs to shared `/burg-archive`; use `short` (account=crew, no `--gres`) for any CPU work.

## Scheduling findings (relevant to batch-3 full retrievals)

- **Per-task IC time is unpredictable from cheap metadata:** r(IC, rad-forward)=0.12,
  r(IC, native-nodes)=0.05, r(IC, τ)=0.004. Counterintuitively the **thin (low-τ) profiles are
  the slowest** (the same ones §A3 flagged as over-sensitive). ⇒ do **not** pre-sort fast/slow by
  τ/nodes/rad-time. The only reliable per-index signal is a measured calibration pass — `priormean`
  measured all 126, and B/C are the **same profiles**, so its per-index time predicts theirs.
- **Measured IC wall by GPU type** (latency-bound, GPU *not* saturated — so the usual compute-bound
  FP64 penalties compress): A100 23–55 min · V100S 28–57 · A40 64–160 · RTX8000 66–177. Reference
  ratios vs A100: V100S ~1.5×, A40 ~4–5×, RTX8000 ~5–6×, **CPU ~17–21×**.
- **CPU ≈ slow-GPU** for this latency-bound work → CPU is a genuine parallel resource (~278 idle
  8-core slots on `short`), but only *additive* when the GPUs are saturated. This batch ended up
  **finishing on GPU** because the arrays drained faster than expected and GPU had spare capacity —
  with no GPU contention, GPU wins per-profile (≤~3 h vs 4–11 h on CPU). For batch-3, the expensive
  forward+inverse loop *will* saturate GPUs, making CPU spill worthwhile.
- **Routing rule (counterintuitive):** CPU-suitability tracks the **A100-equivalent** time
  (revealed by which GPU a profile ran on), not the raw GPU walltime. A profile that ran 28 min on
  an A100 is genuinely hard (~9–11 h on CPU); one that ran 86 min on RTX8000 is easy (~5 h on CPU).

## Pending (will report at completion)

1. Final counts → A/B/C each 125 ok / 1 skip.
2. Sanity-check vs `ic_bundle_BUGGED` (raw sidecars: `_ic_A`↔`_def_priormean`, `_ic_B`↔`_def_draw`,
   `_ic_C`↔`_mech`) — **with the contaminated-baseline caveat** (the old run was on the buggy
   forward; expect systematic differences, looking only for gross structural agreement).
3. Build + manually move `ic_bundle.zip`.
4. Tighten the handoff walltime from the 12 h provisional (observed max ~177 min).

Then **stop and wait** — batch 3 (full retrievals) is a separate hand-off after the primary
reviews this IC output.
