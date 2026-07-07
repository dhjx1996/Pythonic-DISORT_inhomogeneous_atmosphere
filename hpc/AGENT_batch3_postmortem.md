# Batch-3 post-mortem — run-manager agent → main agent

> **✅ FINALIZED 2026-07-06** (FR + IC + efficiency refactor all complete). The §1–§7 record below
> is the run-by-run history as it unfolded (2026-07-01 → 07-04); the **FINAL STATUS** immediately
> below supersedes any "in-flight"/"preliminary" wording in it. See also `CHANGELOG.md` (refactor +
> HPC validation), `fable_assessment_2026-07-01.md`, and `STRATEGY_hpc_retrieval_runs.md`.

## FINAL STATUS (2026-07-06)

**FR — all 125 profiles × 2 configs complete; 0 genuine failures after supersession.** A final
sanity sweep over every result (converged, χ² at floor, τ_bot recovered, physical) found no bugs.
Bugged/misfit originals are superseded by valid corrections: **remesh** (11A, 26A, 52B, 68B, 124B
re-draw, 42B, and 12B/22B which hit the iter-cap *at* the χ² floor) and **continuation-verify**
(28A → 0.0054 on the original grid, the clean sign-bug-victim test). The uncached sign-bug victims
were re-run on the new code (§7): negrel {9A,34B,45B,78A,80A}, continuation {5A,119A}; non-victim
siblings keep their original result. `conv=False` on a few thin profiles (20A/20B) is the iter-cap
technicality, not a failure. **~30 deep-cloud configs are LOW-CONFIDENCE** (high τ, strongly
negative d_rmse — e.g. 28B, 110B, 78B, 80B): good data-fit but under-constrained r_e(τ) deep in
optically-thick cloud. This is inherent OE ill-posedness (present identically in pre- and
post-refactor code), **not** a bug and **not** a re-run trigger — flag it in downstream analysis.

**Sign-bug (§7) attribution, reconciled.** The `_gn_inner` cost-stagnation criterion was
sign-asymmetric (`rel < cost_rtol` vs `abs(rel) < cost_rtol`); it only changed an outcome when the
terminal `rel ≤ −0.01`. Victim-vs-genuine was settled by the continuation-descent test (does a
sign-bug-fixed continuation descend on the *original* grid?): **28A = sign-bug victim** (descends to
0.0054 → its result, not the moved-grid remesh, enters the bundle); **26A, 11A = genuine**
grid-inadequacy (continuation stays elevated → the remesh supersedes).

**Efficiency refactor (E1/E2a/E3/E5/E6a) — reconciled, vetted immaculate, merged to `main`.** The
three post-branch fixes (sign-bug R3, the `retrieve_one` grid-staleness/`max_n_outer` fix, and the
`structural_misfit` redefinition to `chi2_red > thr`) were reconciled into the refactor. Validation:
float32 suite **68/68** (CPU) and float64 **26/26** (GPU, production precision) vs PythonicDISORT
ground truth; L1 checkpoint-resume + L2 setup-cache equivalence gates PASS; E4 farm-setup smoke PASS.
Merged to `origin/main` at `3f51839`. The old golden-probe gate is **RETIRED** (stale/different-grid
reference — pre-refactor production also failed it; the retrieval is inherently grid-sensitive, so it
was never a valid cross-version check; validation is the pytest suites + L1/L2 gates + per-retrieval
sanity checks instead).

**IC re-run — redone on the fixed forward + new code (canonical), all 125+skip per array A/B/C.**
Jacobians (`K_full`) all non-null, finite, fully populated (the historical physical-vs-log
null-Jacobian bug is absent); DOFS/SIC re-derived downstream. The IC deliverable is the refactor
re-run, not the pre-refactor `_ic_*`.

**Live tree migrated to the new code** non-destructively (all git-ignored caches/results intact).

---

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

**Post-hoc erratum bound (2026-07-02, refactor CHANGELOG §2).** The IC workers' azimuthal mode
trim used a flat `(0.005)²·I` Se instead of the OCI model (dark-SWIR σ sits ~5× lower), so the
definitive bundle could in principle understate dark-band information. An 8-profile dark-weighted
probe on the fixed selection (CPU, post-refactor code) measured the actual impact:
**|ΔDOFS| ≤ 0.7 % and |ΔSIC| ≤ 0.21 % (profile worker, 7 profiles incl. the darkest-SWIR thin
set); single-view mechanism DOFS max |Δ| ≤ 1.4 %, median ~0.2 % (5 profiles).** Negligible at the
level of any §14/§15 conclusion → **the definitive bundle stands; no IC re-run.** (Deltas also
fold in the refactor's E5/E6a tol-level path noise, so they are upper bounds on the fix alone.)

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

**L2 — setup cache (`FR_SETUP_CACHE`): ✅ VERIFIED (bit-exact) + WIRED, 2026-07-01.** Caches
`K_list / s_grid / tau_bot_pre / sigma_tau_pre` → eliminates exactly the §2 build tax (1.9–11 h/resume;
the thin-profile RTX8000 tail runs setup ≈ a full wall, so for that class L2 is the difference between
zero and full progress per wall). The GPU equivalence gate **PASSED bit-exact** (job 8683416, idx-95,
A100: K_list match, s_grid/tau_bot_pre/forward/jacobian dmax all 0.0 on cache HIT). Two earlier traps,
both fixed pre-verdict: the first gate run was **vacuous** (the cache write silently failed —
`np.savez` appends `.npz` to string paths, breaking the atomic tmp+`os.replace`; fixed via file-object
write, and the gate now *asserts* the cache file exists after PASS 1), and the cfg key was widened to
`prec|tol|NQ|signature|index` so a future observing-system change can't silently load a stale setup.
Wired into `_fr_gpu_realloc.sbatch` (`FR_SETUP_CACHE=1`) mid-run — numerically safe by the bit-exact
gate; caches land as `_fr_parts/<idx>.setup.npz` (no `_fr_parts_l2/` flagging needed given the PASS).
Committed. Design caveat stands: valid for `max_n_outer ≤ 1` (fixed grid).

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

---

## 6. `max_n_outer` semantics + the 7 structural-misfit configs (2026-07-03)

> **NOTE TO FINALIZING AGENT:** the block below is a verbatim paste of the Fable-5 investigation
> report answering two user questions (the `n_outer` re-mesh tier logic vs. the user's intended
> design, and the 7 flagged configs). **Please CONDENSE it** to at least ~half its length for the final
> post-mortem — keep the two verdicts (worker's `max_n_outer=1` is deliberate; the 7 flags are
> premature-convergence, not a machinery bug), the idx-124-B mechanism, and the corrective-action
> decision (§6a below records what the user actually chose); drop the repeated table/setup framing
> that duplicates §3/§4. Provenance: session 2026-07-03, Opus-4.8 main / Fable-5 investigation.

### 6a. Corrective action TAKEN (user-directed, 2026-07-03)

The user did **not** pick the "no mid-run changes / post-drain side-study" option Fable recommended.
Instead: **re-run the 7 flagged configs with `max_n_outer=2`** ("the equivalent of a next n_outer
iteration" — placement re-mesh enabled), results stored **separately + flagged** in
`docs/cached_results/_fr_parts_remesh/`, and for **idx-124 B specifically a fresh non-tail
climatology re-draw** (the original draw's τ_bot=0.54 was optically-thin → first-GN-step no-descent;
the re-draw uses seed base `3000+index` with a truth-free filter `τ_bot∈[max(2.0, μ−σ), μ+σ]` on the
RF14-LOO climatology `μ=10.0, σ=10.1`, landing τ_bot=3.54 on the first draw). **These 7 re-runs COUNT
as part of the FR task** — they go into the bundle (flagged), and they block the downstream tasks
(§3 L3 deletion, §7 refactor migration, etc.). Mechanism: a backwards-compatible `max_n_outer=1`
kwarg was added to `retrieval_worker.retrieve_one` (default preserves production exactly); a driver
`_fr_remesh_rerun.py` + `_fr_remesh_rerun.sbatch` reuse the production build/retrieve path verbatim,
reusing the on-disk L2 setup caches. idx-11 lacks a setup.npz (recomputes ~2 h); idx-28 only needs
config A (its flagged one). Verification gate on completion: each re-run must take real GN steps
(n_gn>1) and land at a sane χ²; idx-124-B re-draw is checked to escape the zero-step failure.

**LATENT BUG found + fixed while wiring this (RECONCILE AT REFACTOR MIGRATION).** `retrieve_one` did
all its post-GN processing (physical decode, dense `re_ours = fwd.profile(res.x, s_grid, …)`,
RMSE/LWP/d_rmse, sidecar `s_grid`, `k`, `dofs_by_component`) on the **input** grid — but a re-mesh
(`max_n_outer≥2`) returns the result on a **re-selected** grid `res.tau_nodes`, so those metrics would
be silently corrupted the moment re-meshing fires. Latent because production `max_n_outer=1` never
moves the grid (`res.tau_nodes` ≡ input). Fixed: use `grid = res.tau_nodes` for all post-GN work
(no-op at `max_n_outer=1`, verified). **The refactor's E1/E2a/E3 rewrite this worker — when the
refactor is pulled (task #7), re-apply/verify BOTH this fix and the `max_n_outer` kwarg**, else
re-mesh runs produce corrupted metrics with no error.

### 6b. Fable investigation (VERBATIM — condense per note above)

The initial select-once `_gn_inner` call (lines 1188-1191) happens **unconditionally, before the loop, and isn't counted by `n_outer` at all** — it's not "outer iteration 0" in the code's own bookkeeping, it's just the baseline pass that always runs regardless of `max_n_outer`.

The `n_outer` variable is initialized to **2**, not 0 or 1 (line 1202: `n_outer = 2`, with the comment confirming "`n_outer=2` → fixed-count re-mesh, `n_outer=3` → changed-count"). So the loop's very first check is `n_outer(2) > max_n_outer`. With `max_n_outer=1`, that's `2 > 1` — **true on the first evaluation**, before any re-meshing logic runs. So:

- There's no state in this code where "n_outer=1" means "re-place nodes" — that tier is what the code calls `n_outer=2`.
- `max_n_outer=1` blocks that tier entirely (never even attempts a placement-only re-mesh), landing directly on the "disabled" branch and the warning check.

This matches the docstring's explicit claim (`max_n_outer=1 → no re-meshing (select-once)`) exactly — it's not just a description, the code enforces it structurally: the loop counter's starting value (2) was chosen specifically so that `max_n_outer=1` fails the very first `n_outer > max_n_outer` check, and `max_n_outer=2` fails only the second one (tier 3, count change). The offset is confusing (I'd have expected 0/1/2 too), but it's intentional — `n_outer` here numbers *escalation tiers reached*, not *loop passes executed*, and the initial select-once pass isn't a tier at all.

So my earlier answer holds: with FR's `max_n_outer=1`, zero re-meshing (of either kind) ever executes — the 7 flagged profiles are cases where even the *placement-only* re-mesh never got attempted, despite chi2 warranting it.

**The `n_outer` logic — the user's mental model is off-by-one from the code, and the worker's setting disables re-meshing entirely**

The implemented ladder (`src/retrieval_oe.py::gauss_newton_oe`, implemented 2026-06-19 per DESIGN_DECISIONS.md §10) is:

| Setting | Behavior |
|---|---|
| `max_n_outer=1` | **No re-meshing at all** (select-once); χ²>2 only warns |
| `max_n_outer=2` | Placement-only re-mesh allowed; count-change warns *(library default)* |
| `max_n_outer=3` | Count-change also allowed (hard ceiling) |

The behavior the user wanted — "re-place nodes allowed, count-change warn-only" — **is the library default, `max_n_outer=2`**, not 1. There is no tier numbered 0; the initial select-once solve isn't counted at all, and the counter literally starts at 2. The 0/1/2 numbering is a reasonable reading the code doesn't implement.

The production worker's `max_n_outer=1` was **not an accident** — deliberate and documented in two places (worker docstring `retrieval_worker.py:38`, and DESIGN_DECISIONS.md:1088): *"Grid fixed per profile (max_n_outer=1 — a clean A-vs-B comparison; a structural-misfit χ²>thr still warns, flagged in the sidecar)."* Rationale: configs A and B share one grid and one compiled forward, so letting either re-mesh would break the paired comparison and pay extra recompiles. Cost-model correction: since the 2026-06-19 change, **even placement-only re-mesh triggers a recompile** (the recompile-free node-pool padding was dropped once the χ²-gate made re-mesh rare), so tier 2 isn't cheap either — tier 3 is just a *larger* recompile.

**The 7 flagged configs — not a machinery bug; premature-convergence failures from bad starting points, correctly flagged**

Rescanned all 213 completed config-sidecars: exactly 7 flagged (11A, 26A, 28A, 42B, 52B, 68B, 124B). The unflagged population sits at median chi2_red = 0.0066, max 1.91 — a clean bimodal separation.

**Decisive observation: every flagged config's sibling converged to chi2_red ≈ 0.003–0.03 on the exact same grid.** The grid, forward model, Jacobian, and state-space encoding are all fine — what differs between siblings is only the starting point/prior center. These are genuine sensitivity-to-first-guess optimization failures, and the `structural_misfit` machinery caught and persisted every one.

Two distinct mechanisms from the GN traces:

- **Six of seven** stopped via the tuned `cost_rtol=0.01` single-iteration stagnation criterion, fired during an LM damping transient — one iteration with rel < 0.01 (several *negative*: data misfit ticked up slightly while the posterior cost still decreased). Direct evidence it can fire prematurely: idx-75's healthy run had a near-stall (rel = 0.088 after 7 backtracks at chi2_red = 33) and *recovered* to rel = 0.72 two iterations later, converging to 0.004. Had its stall dipped below 0.01 it would have joined the flagged set.
- **idx-124 B** is the severe outlier and a different failure: the climatology draw gave tau_bot = 0.54 vs truth 5.58 (a legitimate draw, `tries=1`), and the *very first* LM step failed all 10 backtracks (19,000 s of forward evals) → declared `converged=True` with **zero accepted steps** — `x_hat_log == x_a_log` bit-for-bit; the "retrieval" is the unmodified draw with chi2_red = 1036. The backtrack J values aren't logged, so the exact no-descent mechanism isn't pinned (likely: huge log-space first steps overshoot, and by high damping the step's ΔJ falls under the tol=1e-4 solver noise so strict `J_new < J` keeps failing).

**Rigor assessment**: nothing here contaminates the clean 206 [user comment: at the time there were 213 completed retrievals]. The 7 are flagged, quantified, and segregable in analysis. Config B *exists* to test robustness to prior draws — an operational retrieval handed 124-B's prior would fail too, so these are arguably legitimate OSSE outcomes. Two genuine design warts: `converged=True` on a zero-step run is semantically misleading, and the single-iteration stagnation stop is fragile during LM transients.

**Corrective options as presented (the user chose the re-mesh re-run, see §6a):**

- (a) No mid-run changes; post-drain side-study re-running the 7 with `cost_rtol=None` (LM to no-descent/step-norm/12-iter), labeled separately, to quantify premature-stop vs true-local-minimum. ~7 config-runs, setup caches on disk — cheap. [Fable's recommendation]
- (b) Re-running the 7 with unchanged settings is pointless — seeds are deterministic (`rng=2000+index`), they'd reproduce exactly.
- (c) Fixing the criterion itself (e.g. require two consecutive stagnant iterations, or don't fire while `lm_cur` is elevated) means re-running all 250 configs for homogeneity — not warranted for a 3.3% flagged rate.

**n_outer indexing — DECIDED (user, 2026-07-03): leave as-is, no renumber.** The confusing 2/3 tier numbering stays; the user's only requirement is that the *default* allow re-meshing at the same node count, and the library default `max_n_outer=2` (= fixed-node-count placement re-mesh, warn on count-change) already satisfies that. The production worker's `max_n_outer=1` remains a separate, deliberate A-vs-B-comparison choice, unchanged.

**idx-11 A remesh result.** `grid_moved=False` — the re-mesh path never fired (χ² dropped below
`thr=2.0` before the outer loop needed to act) — yet the re-run resolved to `chi2_red=0.0031` (from
production's flagged `14.8`). Setup and the GN trajectory reproduced production bit-for-bit through
`init`/`iter 0`; iter 1 diverged (`rel=3.05e-02` here vs. production's `4.78e-03`, opposite sides of
the `cost_rtol=0.01` stagnation gate), purely from GPU floating-point differences, and the run
continued to a clean convergence instead of stopping early.

This is unsurprising, not a new risk: the retrieval's sensitivity to restart/noise perturbation is
expected, and `max_n_outer` only ever raises the *probability* of a good retrieval — it never
guarantees one (why the ladder escalates as far as `n_outer=3`). There is no functional difference
between a perturbation from GPU floating-point noise, an injected jitter, a different RNG seed, or a
re-drawn prior — all displace the trajectory enough to potentially escape a spurious stagnation
stop, and a fix via any of them is equally legitimate. The only axis that actually differs is
**controllability**: GPU noise is incidental (outcome depends on which node the scheduler happens to
pick); a deliberate seed change or jitter (e.g. idx-124B's re-draw, seed `3000+idx`, logged in
`draw_info`) is a designed, repeatable retry we can invoke on purpose.

**Watch list:** 26A (χ²_red=3.04) and 28A (χ²_red=3.59) sit closest to `thr=2.0` among the flagged
set — the likeliest to resolve via this same incidental-noise pathway rather than an actual re-mesh.
More broadly, this means `structural_misfit` near the threshold isn't perfectly hardware-reproducible
— a precision caveat at the margin, not on the flagged set's overall validity (the severe cases —
52B, 68B, 124B — are nowhere near this ambiguity).

**χ²_red (structural misfit) as a reportable metric.** Recommend reporting it alongside
RMSE/DOFS/SIC/LWP as a headline metric, not just an internal QC gate. It captures an axis those
don't: whether the retrieval explains the *observed radiances* to within instrument noise,
independent of how the recovered profile compares to truth. A retrieval can have deceptively good
RMSE with a poor data-fit, or a clean data-fit with imperfect RMSE — χ²_red is the orthogonal signal
that catches both failure modes RMSE alone would miss.

**Bug fixed: `_gn_inner`'s cost-stagnation criterion was sign-asymmetric (`src/retrieval_oe.py`).**
The LM accept/reject test only guarantees the full posterior cost `J` (data+prior) is monotone; the
stagnation check instead compares `rel`, computed from `phi=sqrt(dchi2)` — the data-only term — which
an accepted step can still increase slightly (the prior term absorbs the rest), so `rel` can go
negative even though `J` doesn't. The check was `rel < cost_rtol` (no `abs()`), so **any** negative
`rel`, however large, satisfied it — treating a big single-step data-fit *regression* the same as a
genuine tiny non-improvement. Fixed to `abs(rel) < cost_rtol`.

Exact final-iteration `rel` for the 8 determinable flagged configs (124B stopped via an unrelated
no-descent path; 28A's log was overwritten by a later resubmission reusing the same path):

| Config | final `rel` | \|rel\| vs 1% threshold | Bug-caused premature stop? |
|---|---|---|---|
| 26 A | −2.67e-02 | 2.7% | **Yes** |
| 42 B | −1.77e-02 | 1.8% | **Yes** |
| 68 B | −2.30e-02 | 2.3% | **Yes** |
| 11 A | +4.78e-03 | 0.5% | No — stops either way |
| 52 B | −4.60e-03 | 0.5% | No — stops either way |
| 22 B | +5.18e-03 | 0.5% | No — stops either way |
| 12 B | +7.38e-03 | 0.7% | No — stops either way |

3 of 8 (26A, 42B, 68B) were flagged purely because of this bug. It's not confined to the flagged
set: idx-75 (unflagged) hit the same asymmetry at its own final iteration (`rel=−6.57e-02`) with no
visible effect only because it was already excellently converged by then — so the bug likely fires
across more of the 250-config population than the 9 flagged cases show, just silently.

**Full-population audit (2026-07-04).** Checked all 247 completed configs' final logged `rel`, not
just the 9 flagged ones. Coverage: 179/247 (72.5%) determinable (145 with a usable final `rel` + 34
that stopped via the unrelated no-descent path); 68 (27.5%) unknown — their `gpu_<idx>.out` log was
silently overwritten by a later resubmission reusing the same `%a`-indexed output path (same failure
mode as 28A, §6). Of the 145 determinable: **29 were bug-affected** (not just the 3 already flagged),
**62 stop either way** (|rel|<1% regardless of sign — genuine tiny stagnation), **54 stopped via an
unrelated mechanism** (step-size or iteration cap). The 26 newly-found bug-affected-but-unflagged
configs mostly show excellent χ²_red already (0.005–0.35) — i.e. the bug fired only *after* they'd
essentially reached the representation-error floor, a materially different situation from 26A/42B/68B
(where it interrupted a still-actively-converging retrieval far from any floor).

**Corrective action on the newly-found 26 (+1 found later, see below) (2026-07-04).** All 27 are
being **continuation re-run**: seed `x0` from the config's own persisted `x_hat_log` (no checkpoint
needed — deleted on completion, but the finished sidecar already has
`x_hat_log`/`x_a_log`/`Sa_log`/`s_grid`), then continue the GN solve under the now-fixed criterion
via a single `retrieve_one(..., max_n_outer=1)` call (no re-mesh machinery engaged — chi2_red is
already under the 2.0 threshold at every one of these x0's). Driver: `_fr_continuation_rerun.py` +
`.sbatch`; results flagged in `_fr_parts_continuation/` with a `continuation_provenance` field
(original vs. continued chi2_red/n_gn). 16 had an L2 setup cache (near-free rebuild) and ran
immediately (tasks 0-15); idx-119 B, idx-10 B, 57 A, and 90 A lack one (full setup repay) but were
run anyway (tasks 16-19) — the latter 3 because their final-`rel` magnitude (−8.2%, −4.0%, −6.5%)
was the largest among the uncached group, flagged by the user for a closer look. Of the remaining 6
uncached configs, 5 (tasks 20-24) are deferred and 1 (119A, task 25) is running now — see below.

**The remaining 5 (9A, 34B, 45B, 78A, 80A) — uncached, deferred to the IC re-run window:**

| Config | chi2_red | final `rel` | d_rmse | note |
|---|---|---|---|---|
| 9 A | 0.0044 | −1.4% | +0.086 | at floor, negligible |
| 34 B | 0.0062 | −1.5% | −0.248 | at floor |
| 45 B | 0.0084 | −1.9% | −0.081 | hit the n_iter=12 cap, but plateaued from iter 5 on |
| 78 A | 0.0187 | −1.8% | −0.313 | at floor |
| 80 A | 0.034 | −1.4% | −0.458 | at floor |

All 5 show the same signature as the 20 already re-run: chi2_red was already near its floor when the
bug fired, and the trajectory was flat for several iterations beforehand — no evidence of a genuinely
unconverged retrieval cut short. **34B, 78A, and 80A carry a moderately negative d_rmse** (worse RMSE
than the naive adiabatic floor) — this looks like a real profile-shape retrieval limitation (good
data-fit, imperfect r_e(τ) recovery for that particular truth) rather than a symptom of premature
stopping, since more iterations on an already-flat trajectory are unlikely to move the state.

**Disposition (2026-07-04): these 5 will be re-run, deferred.** Rather than pay the full setup
rebuild now under FR-completion time pressure, they're queued in `_fr_continuation_rerun.sbatch`
(tasks 20-24) to run alongside the user-approved IC re-run (task #9) — that re-run was always
scoped as a post-hoc validation pass, and these uncached bug-affected continuations fit the same
bucket. Submit with `--array=20-24 --constraint=a100|v100s` (f64-efficient GPUs) when task #9's
window opens; not part of the FR-completion hourly gate.

**idx-119 A — found this pass, pulled out and re-run immediately.** Only 119B was tracked through
the original 26-config breakdown (20 continuation-batch + 5 caveat-only = 25); 119A — same profile,
sibling config, also uncached and bug-affected — was missed. It closes the count: 3 (remesh) + 20
(continuation) + 5 (deferred) + 1 (119A) = 29, matching the full-population audit total exactly.

| Config | chi2_red | final `rel` | dofs | d_rmse | rmse_ours / adia | note |
|---|---|---|---|---|---|---|
| 119 A | 0.0115 | −4.85% | 4.433 | **−1.497** | 2.190 / 0.693 | pulled out of the deferred batch |

Unlike the other 5, 119A's `d_rmse=−1.497` is far more negative than any deferred config (next-worst
was 80A at −0.458) — retrieved RMSE is >3x the naive-adiabatic floor, materially worse than "at floor,
small profile-shape limitation." Rather than let it wait for the IC re-run window, re-run it now: job
8823182 (task 25, `--array=25 --constraint=a100|v100s`), submitted 2026-07-04.

**Full-population audit re-verification (2026-07-04).** Rebuilt the audit script independently
(rather than trusting the earlier run's saved numbers) and reproduced the original breakdown exactly:
247 completed configs, 145 determinable + 34 no-descent-path + 68 unknown (overwritten log) = 247;
29 bug-affected among the 145 determinable. Cross-checked all 29 bug-affected configs against every
tracked bucket (3 remesh + 20 continuation + 5 deferred + 1 immediate) programmatically: **zero
configs unaccounted for, zero duplicates across buckets.**

**Why the 68 logs got overwritten — traced via `sacct`, not an accidental duplicate-submission
bug.** Each `fr_gpu` array task hits its 11:55:00 wall and TIMEOUTs regularly on hard profiles —
e.g. idx-1 needed 6 submissions (5 TIMEOUT, 1 COMPLETED) before both configs finished, a normal
part of the checkpoint-chained resume design. Config A and B run sequentially inside one job; if A
finishes but the wall hits during B, the *next* resubmission only needs to resume B (A is already
persisted, so the worker prints `config A already persisted — resume-skip` and moves on). Since
`--output=logs/gpu_%a.out` has no `--open-mode=append`, that later job's shorter stdout truncates
the file — erasing A's original completion trace, even though A's result itself is fully valid.
Every "no_block" case checked shows this exact shape (verified for idx-1, idx-5, idx-6 directly).

**L2 setup-cache coverage of the 68 (2026-07-04): 67 unique profile indices, 64 cached, 3
missing (idx 5, 34, 47).** One of those 3 resolved for free: idx-47 runs through a separate golden-
gate driver (`_fr_gate.sbatch`), which logs to `_fr_gate/frgate_47.out` — not the standard
`logs/gpu_47.out` — so its trace survived untouched. Checked it directly: 47A final `rel=+5.4e-02`
(converged via an unrelated mechanism, not cost_rtol), 47B final `rel=+7.6e-03` (small positive,
stops either way) — **neither is bug-affected.** That drops the true unknown count to **66 configs
/ 66 unique indices, 64 cached / 2 uncached (idx 5, 34)**. All 66 are config A — a direct
consequence of the mechanism above (A always retrieves first per profile, so it's always A's trace
a later B-only resume overwrites).

**Disposition (2026-07-04): all 66 empirically verified via continuation, on CPU.** Rather than
leave these as a permanent epistemic gap, resolve them directly: continue each from its persisted
`x_hat_log` under the fixed criterion and check whether `chi2_red` moves — if `delta_chi2_red≈0`,
the original stopping point was fine regardless of whether the bug technically fired. Run on CPU
(`short` partition, `cpus-per-task=1` + `--xla_cpu_multi_thread_eigen=false`, the same pattern as
the original all-CPU FR launch) rather than GPU: expected work per task is a single quick
iteration, which is latency- not compute-bound, so CPU should track slow-GPU pace rather than
paying the ~20x compute-bound penalty — and it fully sidesteps the already-busy a100/v100s queue.
Driver: `_fr_continuation_verify.sbatch` (66 tasks, reuses `_fr_continuation_rerun.py` unmodified),
results in `_fr_parts_continuation_verify/`. Job 8823208, submitted 2026-07-04, all 66 tasks
started running immediately (ample idle capacity on `short`). Any task showing genuine
multi-iteration progress (i.e. not a 1-shot convergence) gets re-pinned to `a100|v100s` instead of
left on CPU.

**64 of the 66 run now on CPU; the 2 uncached (5A, 34A) were pulled out and deferred instead
(2026-07-04).** Both lack an L2 setup cache, and the user judged paying that full rebuild on CPU
disproportionate to the payoff — tasks 3 (5A) and 21 (34A) in job 8823208 were cancelled and moved
to `_fr_continuation_rerun.sbatch` tasks 26-27, to run alongside task #9's IC re-run on
`a100|v100s` instead. Baseline numbers going in:

| Config | chi2_red | dofs | d_rmse | rmse_ours / adia | n_gn | note |
|---|---|---|---|---|---|---|
| 5 A | 0.0041 | 4.366 | +0.049 | 0.232 / 0.281 | 6 | already better than adiabatic floor |
| 34 A | 0.0063 | 4.368 | −0.131 | 0.487 / 0.355 | 5 | mild, well inside the range seen elsewhere |

Both unremarkable — squarely in the same "at floor" range as the 29 confirmed bug-affected
configs, nothing here suggests either is a hidden severe case.

**Standing GPU-allocation policy (2026-07-04):** several batches now pin `a100|v100s` at once
(remesh tail, continuation tasks 0-19+25, and eventually 20-24/26-27). User granted standing
permission to cancel-and-reallocate RTX8000/A40 jobs if needed to keep that fast-FP64 pool from
becoming a bottleneck — i.e. run overflow on RTX8000/A40 first, then move it to a100/v100s as slots
free up. **A100/V100S jobs still require explicit per-instance permission to cancel.**

**`remesh_if_chi2_red_gt` is now user-configurable.** Added `REMESH_CHI2_THR` (env, default `2.0` —
preserves existing behavior) and a `remesh_chi2_thr` parameter on `retrieve_one`
(`tests/supplementary/retrieval_worker.py`), mirroring how `cost_rtol`/`COST_RTOL` was already
exposed. The threshold actually used is now persisted into both the npz sidecar and the `mon` JSON
(`remesh_chi2_thr` field) for traceability, since it's no longer a fixed constant across runs.

## Grid-mismatch disposition (18A/92A) — verify batch tasks 11/47 (2026-07-04, Opus)

Of the 64 CPU verify tasks, exactly two — task 11 (idx-18A) and task 47 (idx-92A) — completed with
NO output in ~86 s. Not a crash: both hit the explicit grid-mismatch guard at
`_fr_continuation_rerun.py:118`, which refuses to continue when the freshly-built forward's τ-grid
disagrees (len or `atol=1e-3`) with the grid stored in the persisted result the continuation seeds
x0 from. Isolated and explained:

- **Only these two.** Zero grid-mismatches across the entire confirmed-bug-affected continuation
  batch (`_fr_parts_continuation/logs`), and the other 62 verify tasks passed the guard (they ran
  real GN, hence still running long after 11/47 exited). Not a broad-corruption signal.
- **Direction (file mtimes).** For both, the L2 setup cache was written *after* the persisted result
  — idx-18 result 07-02 04:12 vs cache 07-03 01:25 (**+21 h**); idx-92 result 07-01 23:06 vs cache
  07-02 12:22 (**+13 h**). Clean control idx-75: cache *predates* result (−7 h) → grids agree, guard
  passes. So the cache was regenerated by newer grid-select code (the 2026-07-03 grid-staleness-fix
  window and/or τ_bot-pre jitter) whose mesh lands differently than the old-code grid baked into the
  sidecar. The guard is doing exactly its job.
- **No continuation can ever verify these two.** The continuation seeds x0 from the OLD-code sidecar
  grid; any current-code forward (cache HIT or fresh rebuild — a rebuild reproduces the new grid, not
  the sidecar's) is on the new grid. Irreconcilable by construction. A from-scratch re-run would
  *supersede* (not verify) the result, on a different-version grid, and opens the question of mixed
  code-versions across all 125 — out of scope for a sign-bug spot-check.
- **No re-run needed anyway.** Both persisted results are deeply converged at the fit floor
  (18A chi2_red=0.0021 n_gn=12; 92A chi2_red=0.0121 n_gn=10) and did NOT re-mesh (chi2_red << 2.0),
  so the 07-03 grid-staleness metrics bug — which only corrupts re-mesh runs — cannot apply, and the
  sign bug's only harm (a ~1-iter-early stop) is negligible at this depth. **Verdict: verified by
  fit-quality; not re-run.** Removed from the deferred bundle; do not resubmit verify tasks 11/47.

**VERY MINOR CAVEAT — otherwise FINALIZED (user-confirmed 2026-07-04).** idx-18A and idx-92A are
treated as finalized results. The only caveat is that, uniquely among the "unknown sign-bug status"
set, they could not be *empirically* re-verified by continuation (their cached setup grid no longer
matches the old-code sidecar grid the continuation would seed from). They are instead verified
analytically by fit-quality, which fully covers the concern for two deeply-converged, floor-fit,
non-re-meshed configs. Full metrics for the record:

| Config | chi2_red | dofs | d_rmse | rmse ours/adia | tau_bot ret/truth | LWP ours/adia/truth | n_gn | conv |
|---|---|---|---|---|---|---|---|---|
| 18 A | 0.0021 | 4.334 | −0.039 | 0.187 / 0.148 | 1.691 / 1.691 | 6.93 / 6.93 / 6.93 | 12 | ✓ |
| 92 A | 0.0121 | 4.347 | +0.107 | 0.832 / 0.938 | 2.167 / 2.162 | 9.87 / 9.72 / 9.66 | 10 | ✓ |

(The grid-select difference that triggered the guard is just ordinary retrieval sensitivity across
code versions — expected, not a defect; each result is internally self-consistent.)

## Verify batch: CPU stall → GPU re-allocation (2026-07-04, Opus) — L2 cache ≠ XLA compile

The CPU verify batch (`_fr_continuation_verify.sbatch`, job 8823208) STALLED: all 62 tasks sat 4h+
with zero GN progress. Root cause found: **the L2 setup cache saves the setup work (grid-select,
τ_bot pre-retrieval, `select_num_modes`) but does NOT save the XLA compilation of the forward +
jacobian.** Worse, the continuation driver calls `retrieve_one(max_n_outer=1)`, whose jaxpr differs
from the main worker's default call, so its **jacrev is a cache MISS even against the warm GPU cache**
and must compile fresh. On CPU that float64 jacrev compile is effectively unbounded (never finished
inside the 11:55 wall; `_jax_cache_fr_cpu` was cold — 16 files vs the GPU's 368). The original "CPU
is fine, it's latency-bound 1-shot" premise was wrong twice over — the cost is a one-time COMPILE,
not the retrieval itself, and it is compile-bound not latency-bound.

**Fix:** re-allocated to GPU (`_fr_continuation_verify_gpu.sbatch`, job 8825599, `nice=3000`, generic
partition, `%16`). On GPU the same jacrev compile takes ~25–40 min per task (still a miss, but
bounded) then runs; parts accrue at ~16/wave, ~2–3 h total on idle GPU, off the critical path. The
identical-driver continuation batch (which also pays this compile) had already completed 21/21,
proving the path finishes. First 6 verify parts confirmed **Δchi2_red ≤ 0.0006** (converging in 1–2
GN iters from the persisted state) — the intended "sign bug was harmless" result.

**Lesson:** for any continuation/verify driver whose call signature differs from the producer's, the
jacobian is a fresh compile regardless of the setup cache — budget it as ~25–40 min GPU / unbounded
CPU, keep it on GPU, nice'd, off the critical path. Do not assume a "warm cache" covers it.
