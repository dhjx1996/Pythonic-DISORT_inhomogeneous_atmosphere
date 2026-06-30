# Hand-off — checkpoint/resume + resume-scoped compile cache for the `fr` full-retrieval worker

*(Spec for the HPC agent — well-placed to implement AND test on the cluster. This is the original
plan, preserved on Git for posterity; the implementing agent may revise it as it sees fit. **No
timing/walltime numbers are prescribed — the HPC agent owns scheduling.**)*

## ⚠️ Operational prerequisite — checkpoints MUST live on persistent, shared storage

Checkpoints (and the Layer-3 compile cache) MUST be written to a **shared, persistent filesystem**
(e.g. `/burg-archive/...`) — **never** node-local `/local` or `$SCRATCH`. Node-local storage is
wiped when the node dies / the job walls, which is the *exact* event checkpointing exists to survive
— so a node-local checkpoint gives false resilience and then vanishes with the node, leaving nothing
to resume from. The IC batch-2 run already hit this trap (CPU-offload outputs first went to
node-local `/local` ($SCRATCH), invisible on compute nodes). Mirror the existing `_ic_*_parts/` /
`_rad_parts/` convention under `/burg-archive`.

## Context — why

Batch-3 (`fr`) full retrievals are expensive and per-profile cost is **unpredictable from cheap
metadata**: the IC batch-2 report measured `r(time, τ)=0.004`, `r(time, native-nodes)=0.05`, and
found **thin (low-τ) profiles are the slowest** (idx-20, τ=1.5, was the priciest in the §A3 GPU
probe — walled at 12 h still iterating). So you **cannot right-size per-profile walls** by sorting on
τ/nodes/rad-time. Today a task that walls **mid-solve loses all work**; the worker persists each
*config* on convergence (`retrieval_worker.py` `_persist`, commit 454cad5) but a config that walls
*during* its GN solve keeps nothing.

**Goal:** make an `fr` task **resumable** so (1) a wall loses ≤1 GN iteration, and (2) one long
retrieval can be **chunked into several `short`-partition jobs** (e.g. 2×12 h instead of 1×24 h —
`short` caps walltime, so a 1-day job can't use it at all). This unlocks aggressive allocations and
lets the IC report's ~278 idle 8-core `short` slots (CPU spill) participate without losing work.

**Non-goals:** no change to retrieval *physics* or convergence; default (no checkpoint path) must be
**bit-for-bit current behaviour** so the notebook and tests are untouched. Scope is `fr` only — the
IC workers are single-shot (one Jacobian, no GN loop to checkpoint).

## ⚠️ Design caveat — fixed grid only (works for `max_n_outer` 0 or 1)

The checkpointed state is the iterate `x`, which is defined **on a specific grid `s_grid`**. This
design therefore assumes the grid is **fixed for the duration of the GN solve being checkpointed** —
i.e. `max_n_outer=1` (fr's setting: select-once, no re-mesh) or `0`. If re-meshing
(`max_n_outer ≥ 2`) fires and **changes the grid** mid-retrieval, a checkpointed `x` from the old
grid cannot be resumed onto the new grid without also versioning the grid and handling the
transition — out of scope here. fr uses `max_n_outer=1`, so this is not a practical limitation; just
do not combine re-meshing with resume without extending the design.

## Design — three layers, all small state, keyed by `(index, config)`

### Layer 1 — GN-iteration checkpoint (the core, opt-in)
`_gn_inner` (`src/retrieval_oe.py:993`) is a plain Python `for _it in range(n_iter)` loop (`:1053`);
with the grid fixed (above) the resumable state is essentially the iterate `x`. Add an **opt-in
`checkpoint_path=None`** param threaded `gauss_newton_oe` (`:1099`, already exposes `x0`) → `_gn_inner`:
- **On entry:** if `checkpoint_path` exists, load `(x, lm_cur, history, it_done)` and resume the loop
  from `it_done` with that `x`/`lm_cur` (recompute `Fx, K = fwd.forward/jacobian(x)` — one iteration's
  cost; keep the file tiny, do NOT persist K/Fx).
- **Each accepted iteration** (after `history.append(J)` at `:1079`): atomically dump
  `(x, lm_cur, history, _it+1)` via **temp-file + `os.replace`** (a wall mid-write can't corrupt it).
- `checkpoint_path=None` ⇒ no I/O, behaviour identical to today.

Granularity = 1 GN iter (≈ one `fwd.jacobian`). (Rejected: a worker-side "burst loop" calling
`gauss_newton_oe` with small `n_iter` — it re-inits `Fx/K` at `x0` every burst, continuous overhead.)

### Layer 2 — setup checkpoint (the bigger compute saver)
`build_forward_and_obs` (`tests/supplementary/retrieval_worker.py:102`) runs `select_num_modes` +
two-phase `select_retrieval_grid` + the **τ_bot pre-retrieval** (`retrieve_tau_bot`, ~8 GN iters) +
the **pool-Jacobian compile**. Save its small outputs `(s_grid, K_list, tau_pre, sigma_pre, Se)` to
a per-index setup checkpoint; on resume, load and **skip the whole setup phase**. (The pre-retrieval
is itself a `gauss_newton_oe` call at `retrieval_oe.py:1395`, so Layer 1's `checkpoint_path` can
cover it too but there is no need, it's ~8 cheap pinned iters.)

**Why skip the pool-Jac rather than cache it (Layer 2 vs Layer 3 division of labour).** The
`select_retrieval_grid` pool Jacobian is the per-profile compile the dropped cross-profile cache
could never reliably hit, because its XLA *shape* is keyed on the **state-dependent ODE-grid width**
(the adaptive pool size, which varies with profile/first-guess) — so a persistent cache recompiles
it most of the time (`runtime_setup.py:18-22`). A compile cache (Layer 3) could, at best, save the
*recompile* of this op on a resume, and only if its shape happened to recur. Layer 2 instead saves
the setup **output** and skips the setup phase entirely, so the pool Jacobian (and the τ_bot
pre-retrieval) is **neither recomputed NOR recompiled** — avoiding both the COMPILE *and* the
underlying COMPUTE. Caching at best removes the compile; skipping removes compile + compute → for the
setup phase, skipping is strictly better. Hence: **Layer 2 owns the setup** (skip the un-cacheable
pool-Jac), **Layer 3 owns only the forward/jacobian compiles**, which DO recur reliably on resume
(keyed on the small fixed node count `k`; positions/τ_bot are *traced*) and so cache-hit cleanly.

### Layer 3 — resume-scoped persistent compile cache (nearly free)
Enable JAX's persistent cache at worker startup, **before any compile**, guarded by an env var (e.g.
`FR_COMPILE_CACHE_DIR`): `jax_compilation_cache_dir` + `jax_persistent_cache_min_entry_size_bytes=-1`
+ `jax_persistent_cache_min_compile_time_secs=0`, on **shared persistent FS** (see prerequisite). On
resume (same profile → same `k` → same forward/jacobian shapes) the compiles are disk hits.
- This is the **resume-scoped** use, NOT the cross-profile solve-time cache that was dropped
  (`runtime_setup.py:18-22`, low hit-rate ~1-2 %); same-profile resume is a guaranteed hit. Update
  that comment to record the distinction.
- **Caveat (HPC agent's call):** the cache keys on the XLA *backend* — a resume on a *different* GPU
  type (or CPU↔GPU) misses → recompiles (still correct, just slower). Pin the card (`-C`) for
  guaranteed hits, or accept the recompile. Within-card resumes hit.
- **This is the same cache as the IC-batch `ptxas` mitigation — widened, not separate.** The IC
  `ptxas` / `NVPTXCompiler` aborts (≈9/126 SIGABRT, GPU-only, load-dependent, **not** memory) are
  repeated *compiles* hitting a flaky `ptxas` subprocess. A cache **hit replays a stored cubin and
  runs no `ptxas` at all**, so it strictly removes abort opportunities, not just slowness. With one
  shared `FR_COMPILE_CACHE_DIR` the hits are (i) **same-profile resume** — guaranteed (same `k`,
  `K_list`, card); and (ii) **cross-task** — opportunistic, whenever another task matches the
  production forward/jacobian's static key `(k, per-band mode count K_list, XLA backend/card)`. Total
  compiles drop from "once per task" toward "once per distinct shape" — *fewer*, not one (the shape
  varies with the profile's selected `k`/`K_list`). This is **not** the ~1–2 % cross-profile figure
  that retired the solve-time cache: that was dominated by the **un-cacheable** `select_retrieval_grid`
  pool Jacobian (state-dependent grid width), which **Layer 2 removes** from the resume path — what
  the cache keeps is exactly the well-keyed forward/jacobian.
- **What the cache can't catch, Layer 1 does.** The cache never preempts a *cold* compile — the first
  task to build a given shape, and every un-cacheable pool-Jac compile in setup, still invokes `ptxas`
  and can still abort. In the IC batch that meant re-running the whole task; here **Layer 1 resumes
  from the last GN checkpoint (≤1 iteration lost)**, and a resumed task also skips the setup pool-Jac
  via Layer 2. Net composition: **Layer 3 cuts how OFTEN `ptxas` runs; Layers 1–2 make the residual
  aborts cheap.** All `ptxas`-specific — a CPU venue has no `ptxas`, so there the cache is pure
  compile-overhead savings and aborts are a non-issue.

## Are checkpoints compatible across GPU↔CPU (and across GPU types)?

**Yes — the checkpoint DATA is fully portable; only the compile cache is backend-specific.**
- Layers 1–2 store plain **numpy** state (`x, lm, history, it`; `s_grid, K_list, tau_pre, sigma, Se`)
  — no device handles, no XLA artifacts — so a checkpoint written on GPU loads on CPU and vice-versa,
  and across GPU types. This is a *feature*: a profile can **start on GPU and resume on CPU** (the IC
  report's CPU-spill idea) without losing iteration state.
- The **Layer-3 compile cache** keys on the XLA *backend*, so a cross-platform / cross-GPU-type
  resume **misses** it and recompiles (correct, just slower). Within-same-card resumes hit.
- **Numerical note:** iterations *past* the resume point run in the resuming platform's arithmetic,
  so a cross-platform resume continues the descent from the same `x` but its post-resume trajectory
  matches the original only **within solver tol, not bit-for-bit** (the retrieval is deterministic
  only up to platform rounding — the same caveat the GPU resume-equivalence test carries).

## Files to modify
- `src/retrieval_oe.py` — `_gn_inner` + `gauss_newton_oe`: opt-in `checkpoint_path`, atomic
  dump/restore (Layer 1). The ONLY core-code change; default `None` preserves current behaviour.
- `tests/supplementary/retrieval_worker.py` — setup-ckpt save/skip (Layer 2); per-config GN
  checkpoint paths + resume/skip orchestration; reuse the existing `_persist` as the done-marker.
- `tests/supplementary/runtime_setup.py` (or worker preamble) — enable the persistent compile cache
  behind `FR_COMPILE_CACHE_DIR` (Layer 3); amend the `:18-22` dropped-cache note with the
  resume-scoped re-enable.
- `AGENT_all125_fr.md` — document the resume **mechanism** (persistent checkpoint dir,
  resubmit-to-resume flow, card-pin caveat). **No timing/walltime estimates.** Remove the stale
  `~1.5–3 h` / `8 h ample margin` figures (already flagged as nonsense).

## Reuse (don't write new)
- `gauss_newton_oe(..., x0=…)` (`retrieval_oe.py:1099`) already takes the first guess — the resume hook.
- `_gn_inner` LM state is just `x` + `lm_cur` + `history` (`:1041-1092`) — nothing else to serialize.
- `_persist(tag, sc, mon)` (`retrieval_worker.py` main) — the existing per-config done-marker.
- `build_forward_and_obs` returns `(fwd, y, Se, s_grid, pb_phys, pb_log, truth_tol, tau_bot_pre)`
  (`:185`) — exactly the Layer-2 setup state to cache (minus the re-derivable closures).

## Verification (run these)
1. **Resume-equivalence (the rigor gate):** run one profile uninterrupted → record `x_hat`. Re-run,
   kill after ~iter 3, resume → confirm it lands at the **same `x_hat`** (bit-exact CPU; within
   solver tol GPU). A resumed run must equal an uninterrupted one.
2. **Setup-ckpt:** confirm skip-on-resume reproduces `s_grid` and `Se` exactly (hash/`allclose`).
3. **Compile cache:** same-card resume shows JAX cache **hits** + no recompile; different card
   recompiles and is still correct.
4. **Cross-platform resume:** start on GPU, kill, resume on CPU (and vice-versa) → continues from the
   checkpoint, final result within solver tol of the uninterrupted run (closes the CPU-spill story).
5. **Chunk end-to-end:** short artificial wall, resubmit N times, final result matches uninterrupted.
6. **Default-unchanged:** `checkpoint_path=None` is byte-identical to current output (guards
   notebook/tests).
