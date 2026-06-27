# Delegated task — precompute the OSSE radiances (synthetic L1B), all VOCALS profiles

*(Hand-off for a Sonnet agent. Code moves by git (sync first); the **radiance cache moves two ways**:
it STAYS on the cluster for the downstream IC/retrieval batches to load, AND a copy is zipped to the
workspace root for the primary to download. Preserve this file as a handoff template — don't delete it.
2026-06-27. This is **batch 1 of 3**: `rad` (this) → `ic` re-run → `fr` retrievals.)*

## What this is — and why it exists

We are re-running the whole pipeline after fixing a **forward-model bug** and re-tuning two parameters —
all captured in `tests/supplementary/osse_config.py`, the single source of truth (do **not** edit it):
- **TMS fix (`NLeg_all=1024`):** the Nakajima–Tanaka single-scatter correction reconstructed the phase
  function from only 128 Legendre moments, which Gibbs-rings for the sharp short-λ Mie peaks → it injected
  **unphysical negative radiances** for every cloud at the short bands (contaminated the original capstone
  AND the §15 IC). Fixed by carrying 1024 moments for the TMS (the solve still uses NLeg=NQuad=48).
- **NFourier=24** (was 8): re-tuned on the *fixed* forward to the practically-significant threshold
  (rel<1% / abs<1e-3, PythonicDISORT's tolerance) — the old 8 left ~10 % truncation in the short bands.
- **r_e clamp = [2,20]** (was 25): the ensemble truth max is 18.1 µm, so 20 is +2 margin (table is finer).
The optics table is rebuilt at `NLeg=1024, n_gl=3072` (Step 0 handles this).

This batch computes the **synthetic measurement** `y = F(truth)` for every VOCALS profile, **once**, at the
truth's native in-situ resolution (exact), and caches it. Downstream the IC and retrieval workers **load**
`y` instead of recomputing it — which removes the per-profile-shape forward (14..111 nodes → 62 distinct
XLA compiles) from their critical path entirely. In a real retrieval the radiances come from the
instrument; only the OSSE manufactures them, so this is the natural place to do it. The cache embeds an
**observing-system signature**; the downstream workers assert it matches before use.

- **Worker:** `tests/supplementary/generate_osse_radiances.py <index> <out_dir>` — writes
  `<out_dir>/osse_<index>.npz` (the measurement `y` + truth metadata + signature) and a slim `.json`.
- **Consolidate:** `generate_osse_radiances.py consolidate <out_dir> <out.npz>` → one signed
  `osse_radiances.npz` keyed by profile index (what the downstream workers load).
- Each task is **one forward solve** (no Jacobian, no retrieval). Unique per-profile shape ⇒ the compile
  cache does NOT help here (it does for the later batches), so each task compiles its own forward once.

## Repo & paths

- Repo: `/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere`.
  **Sync first** (may be force-pushed): `git fetch origin && git reset --hard origin/main`.
- `PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python`
- `ROOT=<repo path above>`
- `VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data`
- `OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz`
- Radiance cache (output, **stays here** for batches 2–3): `$ROOT/docs/cached_results/osse_radiances.npz`
- Deliverable copy = ONE zip in the workspace root `cloud_profile_retrieval/osse_radiances_bundle.zip`
  (downloaded manually; **not** via git — do not commit/push any npz/json results).

## Step 0 — optics table cache ONCE (shared, ~3–4 min)

```bash
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz
$PY - <<EOF
import sys; sys.path.insert(0,'$ROOT/src'); sys.path.insert(0,'$ROOT/tests/supplementary')
import osse_config as oc
opt = oc.load_optics('$OPTICS_CACHE'); print("optics OK; sig", oc.signature()[1])
EOF
```

## Step 1 — env check

```bash
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
$PY - <<EOF
import sys; sys.path.insert(0,'$ROOT/src'); sys.path.insert(0,'$ROOT/tests/supplementary')
import jax, retrieval_oe, vocals_io, optics_table, osse_config, runtime_setup
print("ENV OK", jax.__version__, "| signature", osse_config.signature()[1])
print("per-band NFourier:", osse_config.NFOURIER)
N=len(vocals_io.load_all_profiles('$VOCALS_DATA')); print("N profiles =", N)   # expect 126
EOF
```

## Step 2 — run the array (one forward per profile)

**Threading (the crew1 oversubscription fix, now built into the worker):** the worker imports
`runtime_setup` and **pins CPU affinity to `SLURM_CPUS_PER_TASK`** *before* JAX starts (offset by
`SLURM_LOCALID` so co-located tasks take disjoint cores) — this caps XLA's Eigen pool to the allocation
instead of the whole node, which is what `--cpu-bind=cores` failed to do. So we **do NOT** need the
single-thread XLA flag. Give each task a few real cores so the compile is multithreaded.

Use `--cpus-per-task=4` (≈500 cores at full 125-wide concurrency) **more cores per
task barely helps** — the forward's azimuthal modes run under a *sequential* `lax.scan`, so the per-task
cost doesn't parallelize beyond the ~10 independent bands. Spend the budget on **array width** (all 125
in parallel), not per-task width. Each task is **one forward** at NFourier=24/NLeg_all=1024/NQuad=48 ×10
bands. This batch is **one forward per profile** (no Jacobian, no GN loop). Measured locally on a thin
native profile (RF11, 20 nodes) at **~38 min including compile** (on 14 cores). The thickest jagged
profiles (up to ~111 native nodes → more ODE steps + a bigger compile, and the 4-core allocation compiles
slower) run several× longer, so budget **`--time=08:00:00` (8 h)** — a conservative ceiling with large
buffer (≈10× the thin datum) that still clears the 12-hour soft limit. With 125 tasks in parallel the
*wall* time is ~one profile's time (a few hours at most); the 8 h just protects the thickest jagged
profiles from being killed.

```bash
cd $ROOT && git fetch origin && git reset --hard origin/main
N=126
mkdir -p docs/cached_results/_rad_parts
cat > /tmp/rad.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=osse_rad
#SBATCH --array=0-$((N-1))%250
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=$ROOT/docs/cached_results/_rad_logs/rad_%a.out
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
srun $PY tests/supplementary/generate_osse_radiances.py \$SLURM_ARRAY_TASK_ID \
   docs/cached_results/_rad_parts
EOF
mkdir -p docs/cached_results/_rad_logs
sbatch /tmp/rad.sbatch
```

Logs go to a **shared FS** path (`_rad_logs/rad_%a.out`) so you can watch progress live: a healthy task
prints `[idx] FLIGHT tau=..: y=240 (native N nodes) in <S>s | sig=<hash>`. (Each task pins affinity and
prints `[runtime] pinned affinity to 4 cores …` — if you instead see a task silent for many minutes with
`affinity ≫ 4`, the pin failed; see Troubleshooting.)

## Step 3 — consolidate + sanity-check

```bash
cd $ROOT
$PY tests/supplementary/generate_osse_radiances.py consolidate \
   docs/cached_results/_rad_parts docs/cached_results/osse_radiances.npz
# expect: "consolidated 125 profiles -> ... (sig <hash>); skipped [0]"  (index 0 = RF01 τ≈1585)
```

The signature in the consolidated file MUST equal `osse_config.signature()[1]` — `consolidate` asserts it.
Leave `osse_radiances.npz` in place at `$ROOT/docs/cached_results/` — **batches 2–3 load it from there.**

## Step 4 — bundle a copy for the primary (record; NOT git)

```bash
cd $ROOT
rm -rf /tmp/rad_bundle && mkdir -p /tmp/rad_bundle/slurm_logs
cp docs/cached_results/osse_radiances.npz /tmp/rad_bundle/
cp docs/cached_results/_rad_parts/*.json /tmp/rad_bundle/ 2>/dev/null
cp docs/cached_results/_rad_logs/*.out /tmp/rad_bundle/slurm_logs/ 2>/dev/null
( cd /tmp && zip -rq osse_radiances_bundle.zip rad_bundle )
mv /tmp/osse_radiances_bundle.zip /burg-archive/home/dh3065/cloud_profile_retrieval/
ls -lh /burg-archive/home/dh3065/cloud_profile_retrieval/osse_radiances_bundle.zip
```

## Troubleshooting

- **Task silent / very slow + `affinity ≫ cpus-per-task`** → the affinity pin didn't take. Check the
  worker printed `[runtime] pinned affinity to N cores`. As a fallback add to the sbatch exports:
  `export FR_PIN_CORES=\${SLURM_CPUS_PER_TASK}` (forces the pin) and, if still thrashing,
  `export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"` (single-thread XLA — slower compile but no
  thrash; last resort).
- **Signature mismatch on consolidate** → some sidecars were generated against a different
  `osse_config` (stale code). Re-sync the repo and re-run the offending indices.
- **Degenerate profiles** auto-write `{"skipped": ...}` — **expect exactly 1** (index 0, RF01, τ≈1585).
- **`optics_table_10band.npz` missing** → re-run Step 0 (don't let 125 tasks race to build it).

## Report back

After the array + consolidate finish, report: (1) profiles consolidated vs skipped (expect 125 / 1);
(2) the **signature hash** (must match `osse_config.signature()[1]`); (3) per-task wall-time range and
cpus-per-task used (a real datum for sizing batches 2–3); (4) the bundle path + size; (5) any errors.
Then **stop and wait** — batch 2 (the IC re-run) is a separate hand-off once the primary confirms the
radiances.
