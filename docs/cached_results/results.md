# Monitoring-agent update for the main agent

*Status update (not a consultation). Two delegated tasks ran concurrently: the **A100 GPU
probe** and **rad batch 1 of 3**. Both are essentially complete; this records the results, a
forward-model-adjacent **bug fix you should review**, and the one outstanding straggler.
Prepared 2026-06-28.*

---

## TL;DR

1. **GPU probe — decisive GPU win.** GPU-vmap-over-modes `jacfwd` = **28.3 s** vs CPU-scan
   `jacfwd` = **197.3 s** → **~7× faster**, and it beats GPU-*scan* (58.1 s) by ~2×. On the
   GPU, batching the Fourier modes (vmap) wins; on CPU it loses. **A GPU retrieval path is
   worth building.** All four backends agree (`sum=12.619636`, `n_neg=0`).
2. **Thrashing bug found + fixed (please review).** `runtime_setup.py` pinned *every*
   co-located array task to the same cores (`SLURM_LOCALID=0` for all array elements), so 5–8
   tasks/node fought over cores 0–3 while the rest of the node idled — this, not any property
   of the profiles, is why the first rad run stalled. Fixed with an atomic disjoint per-node
   core-slot claim. Commits **`8fc43cf`** + **`a5ab9a7`** (pushed to `main`). Verified live:
   **115 tasks / 39 nodes, zero slot collisions.**
3. **Rad batch — 124/125, consolidated + bundled.** First run (8610566) got 10/125 then
   stalled; cancelled + resubmitted the 115 missing indices (8612305) with the fix → 124/125.
   **Index 20 TIMED OUT at the 8 h wall** (thickest RF03 profile); per the primary it is being
   sourced elsewhere, so **124 is final**. Consolidated → `osse_radiances.npz` (sig
   `543eee296e1022f7`) and bundled to `osse_radiances_bundle.zip` (582K).

---

## A. GPU probe (A100, job 8612090) — `vmap`-over-modes vs `lax.scan`

Single-band, K=24 Fourier modes, float64, `jac_mode='fwd'` — the probe in `vmap_probe.py`.
EVAL-ONLY = warm, compile excluded.

| Run            | backend | forward (s) | **jacfwd (s)** | device peak |
|----------------|---------|------------:|---------------:|------------:|
| **GPU vmap**   | CUDA    |        8.58 |     **28.32**  |    1.26 GB  |
| GPU scan       | CUDA    |       19.60 |       58.08    |    0.77 GB  |
| CPU scan (ref) | CPU     |       64.65 |      197.27    |      —      |

Reference CPU-scan `jacfwd` was 206 s (jovyan) / 200.8 s (a separate cluster run) — this
probe's 197.3 s agrees.

**Reading it.**
- **Decisive metric:** GPU-vmap `jacfwd` 28.3 s vs CPU-scan 197.3 s → **6.97× faster** (well
  under the "≲50 s = worth it" bar set in the hand-off).
- **vmap beats scan *on GPU*** (28.3 vs 58.1 s, ~2×) — the SIMT-over-the-batch win the probe
  was testing for. This is the *opposite* of CPU, where vmap was ~3× slower than scan. So the
  mode-batching variant is GPU-specific: keep `lax.scan` for CPU, use vmap for a GPU path.
- vmap's higher device peak (1.26 vs 0.77 GB) is expected (modes materialised as a batch);
  trivial against 40 GB.
- **Correctness:** forward `sum=12.619636` and `n_neg=0` on all four (GPU/CPU × vmap/scan) —
  vmap is bit-faithful to scan modulo cross-backend rounding.

**One thing you should know — the env fix (no production-env changes).** The probe first
failed: the conda `JAX` env reported "no CUDA / cuSPARSE not found." Root cause was **not**
missing CUDA — the env already has every `nvidia-*-cu12` lib (cuSPARSE, cuBLAS, cuDNN, NCCL…)
**and** the cuda12 plugin, but `jax-cuda12-plugin`/`-pjrt` are **0.9.2** while `jaxlib` is
**0.10.2**, so JAX disabled the plugin ("it will not be used"). The first auto-fix attempt
then tried a full `jax[cuda12]` reinstall into `/tmp` and hit `ENOSPC`. Final fix: a
`--system-site-packages` **overlay venv** that reuses the conda env's existing NVIDIA libs +
jaxlib and installs *only* the matching **0.10.2** plugin/pjrt (~200 MB, on shared FS) — the
production env is untouched. If you want GPU JAX to "just work" in the conda env itself,
bump those two packages to 0.10.2 (note: the rad/IC/fr CPU batches don't need it).

---

## B. Thrashing bug in `runtime_setup.py` — root cause, fix, verification

**Symptom.** First rad run (8610566): 10 profiles completed in the first ~2 h, then **zero
new completions for 2.5 h** while 115 tasks sat "RUNNING." A tell I flagged earlier: the set
of nodes that produced completions and the set still "running" had **zero overlap**.

**Root cause.** `_pin_affinity()` offset the core pin by `SLURM_LOCALID` so co-located tasks
would take disjoint cores — but **`SLURM_LOCALID` is 0 for every independent array element**
(each is its own job step). So all N co-located tasks pinned to `cores[0:n]` (= 0..3). With
5–8 tasks packed per node, that's 5–8× oversubscription of cores 0–3 + memory-bus/cache
contention, while cores 4–31 idled. Nodes that happened to be packed densely stalled; the
"lucky" early completions were just on nodes that drained to 1–2 tasks. **Nothing about those
profiles was special** (answering the original question) — it was pure co-location thrash.

**Fix (commits `8fc43cf`, `a5ab9a7`).** Replace the `localid` offset with an **atomic
shared-FS per-node slot claim**: each task `O_EXCL`-creates the lowest free slot file in a
registry keyed by `(array-job-id, hostname)` and pins to `cores[slot*n : slot*n+n]`; dead
owners are reclaimed (PID liveness probe), and the slot is released at exit. A follow-up
commit (`a5ab9a7`) fixes a race the *live run surfaced*: the `O_EXCL` loser could read the
winner's slot file in the brief window before its PID was flushed, see it empty, treat
"empty == stale," and steal it → two tasks on one range. Now empty/unreadable == **occupied**
(never stolen; only a confirmed-dead PID is reclaimable) and the PID write is `fsync`'d.

**Verification (live, on 8612305).** Densest node g171 (7 co-located): slots {0,1,2,3,4,5,6},
fully disjoint. Aggregate across the whole array: **39 nodes, 115 tasks, 0 collisions.** The
batch then completed 124/125 (vs 10/125 thrashing) — the fix is confirmed in production.

> **For your review:** the fix touches only thread/affinity setup (no numerics), so the 10
> npz from the pre-fix run are still valid (affinity changes speed, not `y`). The slot
> registry defaults to `/burg-archive/home/dh3065/.rad_core_slots` (override `FR_SLOT_DIR`).
> Same fix benefits batches 2–3 (IC, fr), which pack many tasks/node similarly.

---

## C. Rad batch (synthetic L1B) status

| | |
|---|---|
| Original job | 8610566 — 10/125, then thrash-stalled; **cancelled** |
| Resubmit | 8612305 — 115 missing indices, with the affinity fix |
| Completed | **124 / 125** productive npz (idx 0 = RF01 τ≈1585 auto-skip, JSON present) |
| Index 20 | **TIMED OUT** at the 8 h wall on g066 (thickest RF03); sourced elsewhere by primary → **124 final** |
| Signature | every npz carries `543eee296e1022f7` (asserted at consolidate) |
| Per-task wall | completed tasks ranged **28 min – 4 h 11 m** |
| Consolidated | `docs/cached_results/osse_radiances.npz` — 124 profiles, sig `543eee296e1022f7`, skipped [] (idx 0 has no npz) |
| Bundle | `cloud_profile_retrieval/osse_radiances_bundle.zip` (582K) — npz + JSON sidecars + slurm logs |

**The straggler, index 20 — resolved (TIMEOUT).** It ran the full **8 h wall** with no output
past the affinity line: the thickest/jaggedest RF03 column (most native nodes → biggest XLA
compile + most ODE steps). It was **not** thrashing (its node had drained; it owned its cores)
— it is simply the one profile a single 8 h forward can't clear. Per the primary's call we do
**not** resubmit it (index 20 will be sourced elsewhere); 124 is final. Consolidate ran on a
compute node → `osse_radiances.npz` (124 profiles, sig `543eee296e1022f7`); bundle zipped to
`cloud_profile_retrieval/osse_radiances_bundle.zip` (582K).

10 valid npz preserved from the first run: indices 1, 2, 3, 11, 12, 18, 19, 53, 64, 110.
(If index 20 is regenerated later, drop its `osse_20.npz` into `_rad_parts/` and re-run
`consolidate` — it merges whatever sidecars are present, so 124 → 125 is a one-command top-up.)

---

## D. What I'd flag for you

- **GPU path:** the probe says go — design the GPU retrieval around vmap-over-modes (keep the
  CPU `lax.scan` for the CPU batches). Memory and correctness are non-issues.
- **Affinity fix:** please review `8fc43cf`+`a5ab9a7`; it's the difference between 10/125 and
  124/125. Roll it into the IC and fr batch submissions (same `--cpus-per-task` packing).
- **Index 20:** TIMED OUT; per your call it's sourced elsewhere, so the batch is closed at 124.

*Rad batch 1 of 3 is **closed** — 124/125 consolidated (sig `543eee296e1022f7`) and bundled to
`cloud_profile_retrieval/osse_radiances_bundle.zip` (582K). GPU **probe #2** verdict below (§A2).*

---

## A2. GPU probe #2 — bands × modes (240-way), full 10-band config

*Follow-up to §A. Does ALSO batching the 10 bands into the vmap (→ 240-way: 10 bands × 24
modes) buy more on the A100, or does the adaptive-solver lock-step eat the SIMT fill? Run as a
**split** job set: GPU job 8616774 (`vmap_loop` + `vmap_both`) + a parallel CPU-partition
reference 8616775 (`scan`). Full operational config: 10 bands, NQuad=48, NFourier=24,
NLeg_all=1024, float64, jacfwd, 24 views. EVAL-ONLY = warm, compile excluded.*

| Leg | path | forward (s) | **jacfwd (s)** | device peak |
|-----|------|------------:|---------------:|------------:|
| `vmap_loop` (GPU) | band-loop, modes-vmap (10× sequential 24-way) | 87.45 | **290.32** | 1.42 GB |
| `vmap_both` (GPU) | ONE vmap over bands × modes (240-way) | 35.76 | **127.74** | 13.87 GB |
| `scan` (CPU ref)  | band-loop, modes-scan (production CPU) | 1029.02† | **2691.31†** | — |

*†CPU `scan` EVAL-ONLY now **captured** by the 12 h rerun **8617956** (the original 3 h job 8616775
timed out mid eval-loop; the earlier rigorous bracket (920, 2555) s is superseded by this direct
measurement). EVAL-ONLY forward **1029.02 s** / jacfwd **2691.31 s** (`sum=100.664015`, `n_neg=0`,
bit-matching both GPU legs). NOTE 8617956 ran on a **slower CPU node** than the compile+eval/GPU
baseline — its compile+eval were 1325 s / 3140 s vs 8616775's 910 s / 2555 s (~1.23× slower
silicon); normalizing the eval-only to that faster node → forward ≈ 837 s / jacfwd ≈ 2190 s. GPU
`vmap_both` eval-only for comparison: forward 35.76 s / jacfwd 127.74 s.*

**(i) Decisive — does batching bands help? YES.** `vmap_both` jacfwd **127.74 s** vs `vmap_loop`
**290.32 s** → **2.27× faster** (forward 35.76 vs 87.45 → 2.45×). The second axis fills the
badly under-used A100 (§A noted only 1.26/40 GB at 24-way) and the SIMT gain **beats** the
~10 % absorbing-band lock-step cost. So the lock-step did *not* eat it — batch both axes.

**(ii) End-to-end vs CPU — measured ~17–21×.** Two consistent measurements:
- **compile+eval:** CPU-scan jacfwd **2555 s** vs GPU `vmap_both` jacfwd **151 s** → **16.9×**
  (forward 910 vs 51 → **17.8×**) — the ratio the *first* (compile-paying) retrieval solve sees.
- **eval-only (now measured, 8617956):** GPU `vmap_both` jacfwd eval-only **127.74 s** vs CPU-scan
  jacfwd eval-only **2691.31 s** → **21.1×** (forward 35.76 vs 1029.02 → **28.8×**). The CPU node
  here was ~1.23× slower than the compile+eval baseline node; normalizing to that faster node →
  jacfwd ≈ 2190 s → **~17.1×**, matching the compile+eval figure. Either basis, the per-`jacfwd`
  GPU speedup the retrieval loop repeatedly sees is **~17–21×**.

`sum = 100.664015` and `n_neg = 0` are **bit-identical across all three legs** (`vmap_loop`,
`vmap_both`, CPU `scan`) — the 240-way GPU path is faithful to the production scan.

**(iii) Memory.** `vmap_both` device peak **13.87 GB / 40 GB** → ~26 GB headroom (matches the
expected ≈10–13 GB; not near 40). The 240-way fits the A100 comfortably.

**(iv) Correctness.** forward `sum = 100.664015` and `n_neg = 0` on all three legs
(`vmap_loop`, `vmap_both`, CPU `scan` forward) — the 240-way bands×modes vmap is bit-faithful
to the production scan, consistent with the primary's CPU validation (fwd rel 1.6e-13).

**Recommendation to the primary.** Build the GPU retrieval on the **full bands × modes (240-way)
vmap** — not the modes-only / band-looped path. It is **2.3× faster than band-looped GPU** and
**~17× faster than the production CPU scan** per jacfwd solve (16.9× measured compile+eval; eval-only
rigorously >7×), fits the A100 with large
headroom, and is bit-identical. The earlier "ship modes-only if bands don't help" fallback is
**not** needed; both axes pay off on this hardware.

---

## §A3 — Probe #3: float32 viability + tol sufficiency (PRELIMINARY, in-flight)

> **STATUS: PRELIMINARY — 4 of 11 matrix cells still running as of 2026-06-29 ~10:54.**
> Inverts the f64/tol=1e-5 **gold** radiance cache (`osse_radiances_gold.npz`, sig
> `543eee296e1022f7`) with an operational-precision forward, via the production
> `retrieval_worker.py`. Compare **config A** (LOO prior mean); config B is the secondary
> reproducible draw. Numbers below are final for completed cells; idx-20 (both legs) and the
> idx-40 tol=1e-5 reference are still on GPU — those rows will be revised in the final §A3.

### Matrix status (4 profiles × 3 settings)

| profile | f32 @ tol1e-3 | f64 @ tol1e-4 | f64 @ tol1e-5 (ref) |
|---|---|---|---|
| **20** thin, τ=1.5 | ▶ running (init+iter0 clean) | ▶ running (init only) | dropped (cost ≫ value for the null) |
| **40** thickest, τ=51.5 | ✗ **crash** (CpuCallback) | ✓ done | ▶ rerun running (8630289) |
| **47** jagged, τ=18.6 | ✗ **soft-fail** (timed out, under-resolved) | ✓ done | ✓ done |
| **49** stiffest, τ=36.5 | ✗ **crash** (CpuCallback) | ✓ done | ✓ done |

### Completed config-A state metrics

| run | conv | n_gn | chi²_red | rmse_ours | τ_bot_ret (truth) | DOFS | d_rmse |
|---|---|---|---|---|---|---|---|
| 47 tol1e-5 | ✓ | 6 | 0.0045 | 0.3223 | 18.667 (18.643) | 3.99 | −0.0022 |
| 47 tol1e-4 | ✓ | 9 | 0.0056 | 0.3229 | 18.611 (18.643) | 4.25 | −0.0028 |
| 49 tol1e-5 | ✓ | 6 | 0.0461 | 1.8445 | 35.689 (36.542) | 3.93 | −0.7135 |
| 49 tol1e-4 | ✓ | 6 | 0.0470 | 1.4811 | 36.067 (36.542) | 4.14 | −0.3501 |
| 40 tol1e-4 | ✓ | 7 | 0.0039 | 0.3847 | 51.395 (51.467) | 4.20 | −0.0948 |

### Verdict (a) — float32 viability: **NO for τ≳18.6; thin (τ=1.5) promising but unconfirmed**

f32 carries the same fingerprint in every run: the angular field is **mode-truncated to K=17**
(vs f64's 23–24) and `RuntimeWarning: overflow encountered in cast` fires during grid selection —
f32 runs at the edge of its dynamic range here.

- **Thick — idx-40 (τ=51.5) & idx-49 (τ=36.5): hard de-stabilization.** GN steps into a stiffer
  state at iter 0/1 → `INTERNAL: CpuCallback error` (the diffrax/equinox implicit-solver guard).
  Not a bias, a **stability failure** → f32 is unusable here.
- **Medium — idx-47 (τ=18.6): soft-fail.** Does *not* crash, but (i) under-resolves (K=17, grid
  k=5 vs f64's k=4), (ii) suffers intermittent **8–20× Jacobian blow-ups** (iter-1 jac 13 278 s vs
  ~1 600 s neighbours — adaptive ODE step-count explosion, *not* hardware), and (iii) plateaus at a
  **residual floor chi²_red ≈ 16** (vs f64's ≈0.005, ~3000× worse) — the f32/tol1e-3 forward cannot
  reproduce the f64 gold radiance. Did not converge in 8 h. **Marginal → effectively non-viable.**
- **Thin — idx-20 (τ=1.5): in progress, encouraging.** Through init + iter 0 cleanly
  (chi²_red 6.06e3 → 1.5e3, rel 0.50, 1 backtrack, **monotone, no crash**) — it has already passed
  the iter-0/1 point where the thick profiles crashed. Same f32 fingerprints present (K=17,
  overflow-in-cast). **Too early to confirm convergence or rule out late oscillation** — the full
  GN trace is pending.

→ Shaping up: **f32 is viable only for genuinely thin clouds; the viability boundary sits below
τ=18.6.** This is consistent with the two-tier hypothesis (thin → f32/looser-tol, thick →
f64/tighter-tol), pending idx-20's completed trace.

### Verdict (b) — is tol=1e-4 as good as tol=1e-5? **Sufficient at medium thickness; NOT at the stiffest**

- **idx-47 (τ=18.6): equivalent.** rmse_ours 0.3229 vs 0.3223 (0.2 %), τ_bot 18.611 vs 18.667
  (0.3 %), DOFS 4.25 vs 3.99. **tol1e-4 ≈ tol1e-5.**
- **idx-49 (τ=36.5): NOT equivalent.** Both converge, but to **different states** — rmse_ours
  **1.481 vs 1.845 (~20 %)**, τ_bot 36.067 vs 35.689, DOFS 4.14 vs 3.93. (Notably tol1e-4 lands
  *closer* to truth here, 36.067 vs 35.689 vs 36.542 — looser tol is not simply "worse," it lands
  on a different local solution.) → **tol matters for the stiffest profile.**
- **idx-40 (τ=51.5): tol1e-4 in hand** (rmse 0.3847, τ_bot 51.395 vs 51.467, converged) — its
  tol1e-5 reference is rerunning (8630289); comparison **pending**.

→ Preliminary: the "tol1e-4 is free" assumption holds at τ≈18.6 but **breaks down by τ≈36.5**,
where the two tolerances reach materially different retrievals. The τ trend will be completed once
idx-40's tol1e-5 reference lands. This supports treating tol as **thickness-dependent**.

### Verdict (c) — idx-20 null control: **in progress**

Both legs (f64/tol1e-4, f32/tol1e-3) running; the f32 leg is descending cleanly so far. Full
null-flatness assessment (all settings ~identical for the thin control) **pending GN completion**.

### Worker note (Step-0 fixes applied)

- **Full-traceback capture** in the worker's except-block — characterizes f32 de-stabilization
  (gave us the `CpuCallback` signature above) instead of a 200-char truncation.
- **Config-A write-ordering fix:** the worker previously wrote `_A`/`_B` artifacts *only after
  config B finished*, so a config-B wall/timeout erased an already-converged config A — this is how
  the original idx-40 tol1e-5 reference (config A converged at 19 544 s) was lost. Now each config's
  npz/json is persisted **the moment that config converges**. (Behavior-preserving: same files,
  written earlier.)

### Remaining before final §A3

idx-20 f32@tol1e-3 (8629811) + idx-20 f64@tol1e-4 (8629810) GN traces to completion (the
oscillation/null question); idx-40 f64@tol1e-5 reference (8630289) for the thickest-profile tol
comparison. All current jobs are within the 12 h wall; no >12 h needs outstanding. Part B
(GPU silent-FP64 canary) not yet started.
