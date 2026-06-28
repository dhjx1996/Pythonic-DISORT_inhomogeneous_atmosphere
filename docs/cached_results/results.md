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
3. **Rad batch — 124/125 done.** First run (8610566) got 10/125 then stalled; cancelled +
   resubmitted the 115 missing indices (8612305) with the fix → 124/125 complete. **One
   straggler remains: index 20** (the thickest RF03 profile, ~7.5 h and counting).
   Consolidation/bundle is deferred until it lands (or is re-run).

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
| Outstanding | **index 20** (RF03) still RUNNING on g066 |
| Signature | every npz carries `543eee296e1022f7` (asserted at consolidate) |
| Per-task wall | completed tasks ranged **28 min – 4 h 11 m** |

**The straggler, index 20.** Running ~7.5 h with no output past the affinity line — i.e. still
in the long compile/integration of the thickest/jaggedest RF03 column (most native nodes →
biggest XLA compile + most ODE steps). It is **not** thrashing (its node drained; it owns its
cores). It has <40 min of the 8 h wall left, so it will either just finish or TIMEOUT. Per the
standing rule I have **not** cancelled it. If it TIMEOUTs I will resubmit **index 20 alone**
with a longer wall (e.g. 16 h) — a single thick forward is the one case the 8 h ceiling can
miss. **Consolidate + signature-check + bundle are deferred until 125/125.**

10 valid npz preserved from the first run: indices 1, 2, 3, 11, 12, 18, 19, 53, 64, 110.

---

## D. What I'd flag for you

- **GPU path:** the probe says go — design the GPU retrieval around vmap-over-modes (keep the
  CPU `lax.scan` for the CPU batches). Memory and correctness are non-issues.
- **Affinity fix:** please review `8fc43cf`+`a5ab9a7`; it's the difference between 10/125 and
  124/125. Roll it into the IC and fr batch submissions (same `--cpus-per-task` packing).
- **Index 20:** the lone open item on this batch; being monitored to completion/TIMEOUT.

*Next: monitor index 20; on success run `consolidate` → assert `543eee296e1022f7` → zip to
`cloud_profile_retrieval/osse_radiances_bundle.zip`; on TIMEOUT resubmit index 20 with a
longer wall. Will report when the batch is fully closed.*
