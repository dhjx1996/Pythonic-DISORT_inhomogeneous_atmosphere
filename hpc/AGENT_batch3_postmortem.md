# Batch-3 post-mortem — run-manager agent → main agent

> **⚠️ PRELIMINARY — pushed 2026-07-01 pre-FR-completion** (12/125 FR profiles complete, run in
> flight). Final version lands when the FR run drains: FR results roll-up, per-profile anomalies,
> final cache disposition. Cost/caching/scheduling findings below are stable; FR coverage numbers
> are not. See also `fable_assessment_2026-07-01.md` (cache verdicts + FR code review + the E1–E6
> efficiency refactor targets) and `STRATEGY_hpc_retrieval_runs.md`.

Feedback on the IC batch and the FR (full-retrieval) run for the main agent + user. Covers what was
produced, the bugs found and fixed, the measured cost structure, and the caching / scheduling lessons.

---

## 1. IC batch (information content)

**Configuration.** All 125 real VOCALS profiles re-run on the fixed forward
(sig `d71a8559bbe457e8`: `NLeg_all=1536`, `NFourier=24`, `tol=1e-4`), three modes:
**A = priormean**, **B = draw**, **C = mechanism**. idx-0 (RF01, τ≈1585) is the lone degenerate skip —
the same 125 profiles as the batch-1 radiance cache.

**Coverage.** 125 ok + 1 degenerate skip per array. No science anomalies — every failure was
infrastructure (see below), all fixed.

**Deliverable (manual move, NOT git — unchanged pattern).** Raw K sidecars are the product →
`ic_bundle.zip`, moved to `/burg-archive/home/dh3065/cloud_profile_retrieval/`. On disk:
`_ic_{A,B,C}_parts/<idx>.json` (+ `<idx>.npz` for the A/B **profile** workers; the C **mechanism**
worker writes JSON only — there is no `_ic_C_parts/*.npz`, so a "0 npz" in that dir is correct, not a
gap). Bundle: `ic_bundle.zip` (632 entries, ~16 MB) at
`/burg-archive/home/dh3065/cloud_profile_retrieval/`.

**Sanity vs `ic_bundle_BUGGED`** (`_ic_A`↔`_def_priormean`, `_ic_B`↔`_def_draw`, `_ic_C`↔`_mech`), with
the contaminated-baseline caveat (old run on the buggy forward → systematic offset expected, only
structure checked): **PASS.** 0 null Jacobians across all 125×2 A/B (the batch-2 state-space null-Jac fix
held), 0 non-finite K, all DOFS/SIC finite & positive. Structure tracks the baseline — DOFS_fullview corr
A=0.57 / B=0.76, C single-view corr=0.77 — with a consistent systematic shift med(new/bug)=0.80 / 0.83 /
0.77 (the fixed forward lowers the bug-inflated DOFS/SIC while preserving profile-to-profile structure).

**Bugs found & fixed (all infrastructure/handoff — none in the physics).**
1. **Out-path `IsADirectoryError`** — `ic_worker_profile.py` / `ic_worker_mechanism.py` use `sys.argv[2]`
   directly as the output file (unlike the rad worker, which appends `osse_{idx}`). The handoff passed
   the bare dir → crash *after* the full Jacobian. Fix: per-index `_ic_X_parts/$SLURM_ARRAY_TASK_ID.json`.
2. **Missing `.json` extension** — the first path fix produced files named `21` not `21.json` (valid
   data, but breaks `cp *.json` bundling). Fix: `.json` in the path; the 124 priormean files renamed on
   disk (no re-run).
3. **`ptxas` PTX-compile aborts** (≈9/126 per mode) — SIGABRT (`Aborted (core dumped)`) inside the XLA
   GPU compiler (`CompileGpuAsmUsingPtxAs` / `NVPTXCompiler::CompileTargetBinary`), **MaxRSS only ~5 MB–5 GB
   (NOT memory)**. Transient/load-dependent `ptxas` subprocess contention (likely pip-CUDA-12.9 `ptxas`
   vs node driver), 1–2 per node across ~7 nodes. **The fix that works is simply re-run** — retries land
   on a healthy node. (An earlier mis-read as 32 G host-RAM OOM led to a 64 G bump that was incidental,
   not causal.) The persistent JAX compile cache (`JAX_COMPILATION_CACHE_DIR`, shared FS) removes the
   repeat `ptxas` exposure by replaying stored cubins — this is the one place the compile cache earns
   its keep (contrast §4c, where it's useless for FR).
4. **CPU-offload infra** — first attempts wrote to node-local `/local` (`$SCRATCH`, invisible on compute
   nodes); `apam1` is low-priority `burst` QOS that won't backfill long walls. Fix: outputs to shared
   `/burg-archive`; use `short` (account=crew, no `--gres`) for any CPU work.

**Scheduling findings.**
- **Per-task IC time is unpredictable from cheap metadata:** r(IC, rad-forward)=0.12,
  r(IC, native-nodes)=0.05, r(IC, τ)=0.004. Counterintuitively the **thin (low-τ) profiles are the
  slowest** (the same over-sensitive ones). Do **not** pre-sort fast/slow by τ/nodes/rad-time; the only
  reliable per-index signal is a measured calibration pass (priormean measured all 126; B/C are the same
  profiles, so its per-index time predicts theirs).
- **Measured IC wall by GPU type** (latency-bound, GPU not saturated, so FP64 penalties compress):
  A100 23–55 min · V100S 28–57 · A40 64–160 · RTX8000 66–177. Ratios vs A100: V100S ~1.5×, A40 ~4–5×,
  RTX8000 ~5–6×, **CPU ~17–21×**. Observed max ~177 min → the 12 h handoff wall can be tightened.
- **CPU ≈ slow-GPU** for this latency-bound work → CPU is a genuine parallel resource (~278 idle 8-core
  `short` slots), but only *additive* when GPUs are saturated. IC finished on GPU because arrays drained
  faster than expected. For FR (§2) the expensive forward+inverse loop *does* saturate GPUs, so CPU spill
  is worthwhile there.
- **Routing rule (counterintuitive):** CPU-suitability tracks the **A100-equivalent** time (revealed by
  which GPU a profile happened to run on), not raw GPU walltime. A profile that ran 28 min on an A100 is
  genuinely hard (~9–11 h on CPU); one that ran 86 min on RTX8000 is easy (~5 h on CPU).

---

## 2. FR build/setup cost & the L1-resume tax — MEASURED

The FR build/setup (mode selection + grid selection + τ_bot pre-retrieval, before the first GN iteration)
is large and strongly GPU-dependent:

| GPU | build/setup | of which τ_bot pre-retrieval |
|---|---|---|
| A100 | ~1.9 h (6759 s) | ~80 min |
| RTX8000 | ~5.5 h (19817 s) | ~3.8 h |
| A40 | ~6.0 h (20535–21576 s) | ~4.5 h |

**L1 checkpointing works** (verified in production — `gpu_49.out`: "resumed from 49_A.ckpt.npz at
iter 4"), but it checkpoints only the GN state, **not the build**, so every resume **re-pays the full
setup above**. Consequence: requeuing/displacing an FR job costs 1.9–6 h of re-paid setup (GPU-dependent)
— which is why FR jobs were never bulk-cancelled to free GPUs.

---

## 3. MAJOR INEFFICIENCY — τ_bot pre-retrieval (future work)

The setup cost in §2 is dominated by the **τ_bot pre-retrieval** (`retrieval_worker.build_forward_and_obs`
→ `retrieval_oe.retrieve_tau_bot`, `src/retrieval_oe.py:1440`). Despite the docstring calling it "cheap",
it is a **full `gauss_newton_oe(n_iter=8, xtol=5e-3)` mini-retrieval on the full 240-element obs vector**,
paying a full-cost 1536-moment Jacobian every iteration → ~5 Jacobians.

- **(a) Tol too tight.** The pre-retrieved τ_bot is only an *informed prior anchor* (τ_bot is refined free
  in the retrieval proper) → `n_iter=8`/`xtol=5e-3` is overkill; loosening saves ~1–2 Jacobians/profile.
- **(b) All 10 bands, only ~3 contribute — the major one.** With r_e pinned, only the conservative /
  near-VIS bands (~3 of 10) carry the residual τ_bot signal; the other 7 contribute little. **Keep the
  forward's compile shape** (don't change declared shape), but **simplify the EVAL** — mask/down-weight
  those 7 in the pre-retrieval's cost/Jacobian. User + main agent to implement; not touched here.

---

## 4. Caching layers

**L1 — per-GN-iteration checkpoint: works, high value.** Proven in production (§2). Restarts at the
checkpointed iteration; saves the completed GN iterations but re-pays the build.

**L2 — setup cache (`FR_SETUP_CACHE`): implemented, high value, deferred.** Caches
`K_list / s_grid / tau_bot_pre / sigma_tau_pre` → eliminates exactly the §2 build tax (1.9–6 h/resume).
Code is in `tests/supplementary/retrieval_worker.py` (uncommitted); config-keyed
(`prec|tol|NQ`, NOT mode_map). It plays no role in this batch — FR runs without it, and **FR > L2**. The
rigor gate is a bit-exact equivalence test (compute+write vs load must match K_list/s_grid/tau_bot_pre +
forward/Jacobian evals); the CPU version of that test is impractical because it must itself run the §3
τ_bot pre-retrieval (~5 CPU Jacobians). The definitive verdict belongs on spare GPU after FR drains. On
PASS: commit the L2 code, flip the docs (this file, `AGENT_all125_fr.md`, `FR_CHECKPOINT_RESUME_PLAN.md`),
wire it seamless guarded by `max_n_outer < 2`, and fold `_fr_parts_l2/` into `_fr_parts/`. On FAIL: revert
the L2 edits.

**L3 — JAX persistent compile cache: ineffective for FR, flagged for deletion.** FR is execution-bound; the
expensive forward/Jacobian compiles are **not** cached — `_jax_cache_fr` is 26,556 files / 107 MB (avg
~4 KB = only small helpers). That file *count* is a Lustre small-file storm (CUIT concern) with no FR
upside → net liability. Plan at end-of-run: **delete the cache DATA** — `_jax_cache_fr` (+ the tiny
`_jax_cache_fr_cpu`) get `rm`'d once FR has drained (keep `_jax_cache` — that's IC's genuine `ptxas`
mitigation, §1 bug 3) — and **disable L3 for future FR** (drop/repoint `JAX_COMPILATION_CACHE_DIR` in
`_fr_*.sbatch`). The L3 **code/feature** itself is only *flagged* for removal — that deletion is left to
the main agent + user. Never set thresholds to −1/0 (Lustre).

---

## 5. Scheduling / ops lessons

- **Watch output FILES, not `squeue`.** Transient `squeue` hiccups false-reported several running jobs as
  "left queue"/"walled" (idx-49, the L2 test). Poll the result/log file for a completion marker instead.
- **FR: `cpus-per-task=1`.** The algorithm is sequential and XLA runs single-thread; a 16-core canary gave
  no speedup (2-vs-1 was only +11% BLAS). IC uses cpt=4. The FR worker must `runtime_setup.setup()` before
  importing JAX (XLA-oversubscription fix — otherwise XLA sizes its Eigen pool to all 32 node cores).
- **Goldens are sanity, not exact validation.** `probe_{40,47,49}` are valid (1024 moments, no Gibbs
  ringing) but differ from the current 1536-moment forward via retrieval sensitivity (a different QRCP
  grid). idx-49's headline τ_bot matched its golden to 0.2 %; the dense-profile "MISMATCH" flag is just
  the crude bar. Never substitute a golden for a fresh result.
- **GPU contention.** During FR the cluster GPUs were fully subscribed (mix nodes at gpu:2 of gpu:2); IC
  got ahead purely via normal-priority submission while FR-pending was nice'd down — no FR was cancelled.
