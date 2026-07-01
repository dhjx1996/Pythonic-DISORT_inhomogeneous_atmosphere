# Strategy guide — minimizing wall-clock for the OSSE retrieval batches (`rad` / `ic` / `fr`)

*(Written 2026-07-01 by the run-manager agent, mid-batch-3, from measured production numbers —
batch-1 `rad`, batch-2 `ic`, batch-3 `fr` on Ginsburg. Companions: `AGENT_all125_{rad,ic,fr}.md`
(the task hand-offs), `FR_CHECKPOINT_RESUME_PLAN.md` (resume architecture + field status),
`AGENT_batch3_postmortem.md` (what actually happened). §1–2 and §5–7 generalize beyond this
cluster; §3–4 carry the cluster- and pipeline-specific numbers.)*

---

## 1. Cost anatomy — where the time actually goes

The pipeline's natural unit of account is **one full-forward Jacobian** at the production config
(NQuad=48, 10 bands, 1536 moments, float64, tol=1e-4): **~600–700 s to execute on an A100**
(post-compile; one-time compile ~850 s). Everything else prices in this unit:

| batch | work per profile | Jacobian-equivalents | measured wall/profile |
|---|---|---|---|
| `rad` | 1 forward eval | ~0.05 (eval ~36 s A100) + 2–5 min compile | minutes |
| `ic`  | 1–2 Jacobians | 1–2 | 23–55 min A100 · 66–177 min RTX8000 |
| `fr`  | setup (≈5 Jac: mode-select + 2×grid-select + **τ_bot pre-retrieval**) + 5–18 GN iterations × 1 Jac, × 2 configs sharing one setup | ~10–25 | 4.8–10.6 h on fast cards (11 completed, all converged); thin profiles: **multiple 12 h walls on slow cards** |

So `fr` costs ~10–25× `ic` and ~300× `rad` per profile. Every optimization is one of three moves:
**(a)** cut the *number* of Jacobian-equivalents, **(b)** cut the *unit price*, **(c)** never pay
the *same* one twice.

---

## 2. The levers, ranked (these transfer to any cluster)

1. **Never recompute a pipeline-stage product.** One optics table, one radiance cache
   (`y = F(truth)` computed once in batch-1), loaded by every downstream worker. **Signature-gate
   every cache** (hash the full numerical config; refuse mismatches): the same mechanism is both
   the rigor guard and what makes caching safe to rely on.
2. **Right-size the numerics to the science, once, with a probe.** tol=1e-4 was shown
   indistinguishable from 1e-5 for this regime (§A3); NQuad=48 is the convergence point; float64
   only because the 10-band dense-truth forward genuinely needs it. Each of these was settled by a
   *cheap targeted probe*, not by running the batch twice. A tolerance you didn't need is pure
   multiplicative waste on every Jacobian.
3. **Checkpoint at the natural iteration boundary** (L1: the GN iterate `x`, per `(index,config)`,
   atomic write, plain numpy → portable CPU↔GPU and across GPU types). A wall or preemption then
   loses ≤1 iteration. This converts "avoid walls at all costs" into "walls are cheap", which
   unlocks §2.9.
4. **Cache the *setup output*, not the compile** (L2: `K_list / s_grid / τ_bot_pre / σ_τ`,
   config-keyed). Skipping the setup phase removes its *compute and* its *compile*; a compile
   cache at best removes the compile. Persistent **compile caches (L3) only pay when the workload
   is compile-bound and shapes recur exactly** — measured: useless for `fr` (execution-bound; the
   heavy executables never even landed in the cache), genuinely useful for `ic` only as a
   `ptxas`-crash mitigation (a cache hit replays a stored cubin and runs no `ptxas` at all).
5. **Measure, then schedule.** Per-task cost is *not* predictable from cheap metadata — measured
   r ≤ 0.12 against τ, node count, and rad-forward time; the intuitive "thin = cheap" is
   *backwards* here (thin profiles are convergence-over-sensitive and have the most expensive
   τ_bot pre-retrievals). The only reliable per-index predictor is a **calibration pass**: `ic`
   config-A times every profile's Jacobian ≈ `fr`'s per-*iteration* cost (it cannot predict GN
   iteration *count*, so it bounds but does not settle total cost).
6. **Match hardware to the arithmetic.** FP64-fast cards (A100, V100S) vs FP64-slow gamer cards
   (RTX8000, A40 — nominal 1/32 rate, but **measured only 4–6× slower when latency-bound**,
   because neither saturates). CPU measured ~17–21× A100 → **CPU ≈ another slow GPU** for
   latency-bound work, i.e. a real spill resource once GPUs are saturated (route to CPU by
   *A100-equivalent* time, not by observed wall on whatever card the task happened to get).
7. **Route by progress-per-wall, not speed alone** (the batch-3 lesson). A slow card is fine when
   its unit of progress (1 GN iteration ≈ 50–70 min on RTX8000, checkpointed) fits the wall many
   times — steady progress. It is *catastrophic* when a monolithic uncheckpointed phase ≈ the
   wall: thin-profile setup ran 9–11+ h on RTX8000 (idx-22: 10.6 h; idx-20 walled *inside* setup
   with zero state written) → the wall produces **zero net progress, indefinitely**. Pin exactly
   those tasks to fast cards; that is where the fast card's marginal value is highest. Corollary:
   L2 shrinks the monolithic phase to ~0 and dissolves the whole problem class.
8. **Maximize scheduler surface.** List *every* eligible partition on every job; add constraints
   (`-C a100`) only when §2.7 demands it; submit one wide array with a `%` throttle rather than
   many small jobs; use `nice` to order your own queue (bulk at high nice, latency-critical at 0)
   — intra-user priority is free and reversible (`scontrol update jobid=… nice=…`).
9. **Make interruption cheap instead of walls precise.** Per-task totals are unpredictable (§2.5),
   so don't tune per-task walltimes — standardize on a chunk length that fits the *most permissive
   pool* (11:55 h fits `short`, which is where the A100s live) and let the resume chain
   (L1+L2 + a resubmit driver) absorb the tail. One conservative wall + cheap resume strictly
   dominates per-task wall tuning.
10. **Watch outputs, not the scheduler.** Poll result files for completion markers; `squeue`
    hiccups false-report running jobs. The whole run reduces to one idempotent driver loop:
    `stalled = {1..N} − complete(files) − queued(squeue)` → resubmit `stalled`. Hourly is plenty.

---

## 3. Per-batch playbooks (this pipeline, this cluster)

### `rad` — already near-optimal
GPU array over profiles, all partitions, `MODE_MAP=vmap`, 16 G, 2 h wall; consolidate +
signature-check. Compile dominates (unique XLA shape per native profile resolution); eval is
~36 s. Only worth revisiting if profile count grows 10×+ — then bucket profiles by padded node
count to share compiles.

### `ic` — the calibration pass that pays for itself
All three arrays (A/B/C) simultaneously; 32 G; compile cache ON (default thresholds — this is the
`ptxas` mitigation); walls at the 12 h standing default (observed max 177 min, so the risk is
nil). Treat config-A's measured times as the per-index cost table for `fr` scheduling. Re-run
transient `ptxas` SIGABRTs (~7 %) — they land on a healthy node.

### `fr` — the capstone; everything above composes here
- **Venue:** GPU, float64, `cpus-per-task=1` (16-core canary: no speedup; the algorithm is
  sequential), `MODE_MAP=vmap` on GPU (`scan` on CPU). 32 G for big profiles.
- **Architecture:** L1 (GN checkpoint) + L2 (setup cache, once the equivalence gate passes) +
  11:55 walls + hourly resubmit driver = a self-healing chain toward all-125. L3 compile cache
  OFF for `fr` (measured no-op + a 26k-small-file Lustre liability).
- **Launch:** one array idx 1–125, `--partition=crew1,ocp_gpu,short`, `%`-throttled wide. An
  **A100 "gate" job on one representative profile first** as bug-catcher — in batch-3 it caught
  nothing late (the state-space bug was batch-2's) and its output was folded in as a free
  completed profile (idx-47).
- **Straggler routing (the batch-3 rule):** tasks that walled with **no checkpoint** (died in
  setup — thin profiles on slow cards) → resubmit `--constraint=a100 --partition=short`; tasks
  with banked GN checkpoints → resubmit generic. Once L2 is live the first class mostly vanishes
  (setup is paid once per profile, ever).
- **CPU spill** only when GPUs are saturated, `short` partition, outputs on shared FS, and only
  for profiles whose A100-equivalent time is small (§2.6).
- **The standing MAJOR INEFFICIENCY** (user-flagged, future work): the τ_bot pre-retrieval is a
  full 8-iteration GN mini-retrieval over **all 10 bands** when only ~3 conservative/near-VIS
  bands carry τ_bot signal, at a tolerance far tighter than an informed prior needs. Masking the
  7 dead bands in the *eval* (keeping the compile shape) + loosening `n_iter`/`xtol` should cut
  ~30–50 % of setup (~1–3 Jacobian-equivalents/profile) — multiplicative with L2, which removes
  the *re-pay* of whatever remains.

---

## 4. What will change between runs — and what it does to cost

Ordered by likelihood (new profiles are certain; geometry/bands/streams plausible; the rest rare):

| change | cost impact | action |
|---|---|---|
| **New profiles** | linear in N; per-profile compile shapes (`rad`); unknown per-profile tail | re-run calibration pass (§3-ic); screen degenerates up front (the idx-0 pattern); expect the thin/over-sensitive class to reappear |
| **Solar geometry μ₀** (fixed 0.9 in OSSE; **varies per scene operationally**) | `μ₀` is a *static* arg of `riccati_setup` → **every distinct μ₀ is a fresh compile** of the whole forward | quantize/bin μ₀ (e.g. 32 bins), batch scenes by bin so compiles are shared; a compile cache keyed on the bin actually earns its keep here |
| **Bands** | ~linear in forward cost (vmap over bands); NFourier is per-band | rebuild optics table (one-time, minutes); **regenerate the radiance cache** (signature changes — non-negotiable); re-tune per-band `NFourier` and re-run `select_num_modes` (it already trims per-band K adaptively) |
| **Viewing angles** | nearly free in the forward (barycentric μ-interpolation at eval); ~linear in obs-vector length m for Jacobian assembly + GN linear algebra (small) | more views may *improve* GN conditioning → fewer iterations; don't fear adding views |
| **NQuad (streams)** | integrator **step count is ~NQuad-independent** (design invariant — ~35 steps on a τ=30 cloud); per-step cost grows as the N×N Riccati matrix ops, N = NQuad | re-probe convergence before accepting a higher NQuad; remember views = NQuad/2 convention couples this to the observing system |
| **Tolerance / precision** | tol drives adaptive step count (weakly, ~log); f32 would unlock the slow cards' full rate **but is measured-invalid** for the 10-band dense-truth forward at NQuad=48 (Kvaerno5 `max_steps` blowup) | revisit f32 only for few-band configs; re-run the §A3-style tol probe whenever bands/NQuad change |
| **Noise realizations** (OSSE noiseless → real/ensemble noise) | K realizations per profile share truth, optics, **and setup** | this multiplies L2's value by K: setup once, retrieve K times |
| **NLeg / optics grid** | one-time optics rebuild; NLeg only matters through TMS fidelity | keep 1536 unless the phase-function content changes (new particle model) |

---

## 5. Next-run checklist (condensed)

1. Bands/optics/NLeg changed? → rebuild the optics table once, shared path.
2. Anything in the forward config changed? → regenerate the radiance cache (`rad` batch); the
   signature gate will refuse stale caches — let it.
3. One-time env sanity on a GPU node: plugin versions, `LD_LIBRARY_PATH` nvidia prepend,
   `runtime_setup.setup()` before JAX, affinity print.
4. Calibration pass = `ic`-A (or a 1-Jacobian probe) timed over all profiles on one card class.
5. `fr`: gate job on one representative profile (A100) → then the wide array, L1+L2 on, 11:55
   walls, all partitions, hourly coverage/resubmit driver, straggler routing per §3.
6. End-of-run: zip raw sidecars (never git), structural sanity vs the previous run (correlation,
   no-null-Jacobian check), delete Lustre-hostile cache data.

---

## 6. Ginsburg crib sheet

- **GPUs:** 16× A100 + 8× V100S (FP64-fast) · 18× A40 + 36× RTX8000 (FP64-slow). A100s live in
  `short` and advertise `Features=a100`. Measured latency-bound ratios vs A100: V100S ~1.5×,
  A40 ~4–5×, RTX8000 ~5–6×, CPU ~17–21×.
- **Submit:** `--account=crew --partition=crew1,ocp_gpu,short` (crew1/ocp_gpu ≤7 d walls; `short`
  ≤12 h — hence the 11:55 standard chunk). `apam1` is CPU-only, low-priority burst QOS — avoid
  for anything urgent. ~278 idle 8-core `short` CPU slots exist for spill.
- **Env traps:** login profile's `~/cuda-12.6` shadows pip CUDA libs (prepend `$NVLIB`);
  `runtime_setup.setup()` must run before JAX imports or XLA grabs all 32 node cores;
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` for cpt=1 CPU work.
- **Lustre:** `/burg-archive` hates small-file storms — JAX cache at *default* thresholds only
  (never −1/0); keep file counts low; checkpoints/outputs on `/burg-archive`, never `$SCRATCH`
  (node-local, dies with the node).
- **Login node:** file ops and quick single-core python only; all compute via sbatch/srun.
- **Intra-user queue control:** `scontrol update jobid=… nice=N` (higher = lower priority) —
  reversible, takes effect on pending jobs immediately; equal-priority ties go to older jobs.

---

## 7. Beyond this cluster

The load-bearing ideas map cleanly:

- **Walls ⇄ spot/preemptible instances.** The L1+L2+driver chain is exactly a spot-instance
  strategy: standardize the chunk, checkpoint the iterate, cache the setup, resubmit
  idempotently. Cost-optimal capacity (spot, `short`, backfill) is only usable when interruption
  is cheap — make it cheap *first*, then buy the cheap capacity.
- **Signature-gated caches ⇄ content-addressed artifact stores.** Hash the config into the
  artifact key; a mismatch is a refusal, not a warning. Object storage replaces `/burg-archive`.
- **Progress-per-wall routing ⇄ heterogeneous fleets.** Classify tasks by their largest
  uncheckpointable phase; anything whose phase ≈ the preemption horizon goes on the reliable/fast
  tier, everything else on the cheap tier.
- **Calibration pass ⇄ canary batch.** Time a 1-unit proxy of the real work over the full input
  set before committing the fleet; never schedule from metadata intuition (measured r ≤ 0.12
  here — intuition was not just weak but wrong-signed).
- **Single big server (no scheduler):** `xargs -P k` with per-process core pinning + the same
  L1/L2 checkpoints (the run becomes freely stoppable/resumable, which is the same property walls
  forced on us here).

*The single highest-leverage future change for this pipeline remains algorithmic, not
infrastructural: the τ_bot pre-retrieval band-mask + tolerance fix (§3), which attacks the
dominant setup term at its source; L2 then makes whatever survives a one-time cost per profile.*
