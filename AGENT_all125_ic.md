# Delegated task — all-VOCALS information-content **DEFINITIVE** run (NQuad=48)

*(Hand-off for a Sonnet agent. The primary server is a **separate filesystem**; **code** moves by git
(sync first), but the **raw results move as a single zip the primary downloads manually** — NOT via git
(see Step 3). Preserve this file as a handoff template — don't delete it. 2026-06-25 **DEFINITIVE** run
(10-band + noise-sweep caching); supersedes the robustness re-run.)*

## What this is

The **definitive** cloud-droplet effective-radius information-content study over **every VOCALS-REx
profile** (126 records; 1 non-physical τ≈1585 skipped → 125 valid), linearized at the converged
**NQuad=48**, μ₀=0.9. It replaces the pilot (`info_content_stage1.py`) and the robustness re-run.
Changes vs the pilot (DESIGN §13/§14):

- **10-band superset** `[0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]` µm — HARP2/OCI/NK1990
  + the **4.05 µm VIIRS M13** band (ω≈0.87, operational MWIR, slightly more absorbing than 3.7; tests
  spectral headroom). No band order baked in — value-/data-greedy ordering is applied post-hoc to the cached K.
- **IC state includes the base node** (`include_base=True`; r_base is an r_e value); τ_bot held known.
- **OCI 2 % noise baseline** (`noise_model.oci_swir`), radiance + flux — the headline level (matches the
  retrievals); the post-hoc **noise sweep** (1/2/3/5 %) is rebuilt on jovyan from the cached `y` (the
  worker's cached σ is the 2 % baseline only).
- **miepython optics table** (`src/optics_table.py`; JAX-Mie / `miejax_lite` retired) — built once,
  **disk-cached**, shared by all tasks.
- **Truth-linearization retired** — modes are `priormean` (headline) and `draw`.
- **Raw Jacobians cached** — each task writes a `.npz` sidecar (K_full, K_flux, s_int, σ, the reflectance
  `y` for the post-hoc **noise sweep**, and the prior covariances). All figures are assembled post-hoc
  from these by `ic_analysis_definitive.py` (on jovyan).

**One profile worker** (`ic_worker_profile.py`), parametrized by `IC_MODE`, run as **2 SLURM arrays**:

| `IC_MODE` | linearize at | prior block(s) cached in the .npz | result set |
|---|---|---|---|
| `priormean` | LOO prior mean | `loo` · `weak` (σ≈10 µm flat, KV2012) · `loo2x` (LOO, ℓ=1.0) | iv (HEADLINE) · i · vi |
| `draw` | a LOO adiabatic realization¹ | `loo` | v |

¹ `draw` = a physical **3-param adiabatic** realization (`roe.draw_climatology_realization`): r_top,
r_base ~ LOO marginals (independent), τ_bot fixed at truth, rejection-sampled so 25 ≥ r_top > r_base ≥
2, then r_e(s)=(r_base⁵+(r_top⁵−r_base⁵)(1−s))^{1/5} on the grid (seeded per index).

Plus a **3rd array** — `ic_worker_mechanism.py` (direct-vs-prior depth mechanism; same definitive
config, linearized at the LOO prior mean) → fig 5.

Each task writes `<idx>.json` (slim headline scalars, for monitoring) **and `<idx>.npz` (the raw K
sidecar — the actual product)**. No truth array; **no analysis on the cluster** — the raw sidecars are
zipped (Step 3) and the primary does all analysis on jovyan.

## Repo & output

- Repo: `/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere`.
  **Sync first** (may be force-pushed): `git fetch origin && git reset --hard origin/main`.
- **Deliverable = ONE zip of the raw sidecars** (Step 3), placed in the workspace root
  `cloud_profile_retrieval/ic_definitive_bundle.zip`, **downloaded manually** (NOT via Git — do not
  commit/push any result JSON or npz). It contains the `_def_*_parts` (npz K-cache + slim json),
  `_mech_parts` (json), and the SLURM logs.
- Code (present after sync, **don't modify**): `ic_worker_profile.py` (`IC_MODE`, `ENSEMBLE_NQUAD`
  default 48, `IC_SIGMA_WEAK` default 10, `OPTICS_CACHE`), `ic_worker_mechanism.py`,
  `ic_analysis_definitive.py`, `src/optics_table.py`, `tests/supplementary/_ic_parallel.py`.

## Step 0 — build the optics table cache ONCE (shared, ~3–4 min)

The miepython optics table is **profile-independent**; build it once to a shared path so the array tasks
**load** it (don't let 125 tasks race to build it):
```bash
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
$PY - <<EOF
import sys; sys.path.insert(0,'$ROOT/src')
import optics_table as ot
t = ot.build_or_load_table([0.55,0.67,0.86,1.038,1.24,1.64,2.13,2.26,3.7,4.05],
                           2.0,25.0,32,0.10, cache_path='$OPTICS_CACHE', NLeg=128, n_radii=600)
print("optics cache OK:", t['omega'].shape, "->", '$OPTICS_CACHE')
EOF
```
**Export `OPTICS_CACHE` in every sbatch** (already wired below). `miepython` and `numba` must be in the
env (`$PY -m pip install miepython` if missing).

## Step 1 — JAX (CPU) env

```bash
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
$PY - <<'EOF'
import sys
sys.path.insert(0,'/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/src')
import jax, numpy, scipy, diffrax, netCDF4, miepython, numba
import retrieval_oe, info_content, vocals_io, optics_table, noise_model
import PythonicDISORT
print("ENV OK", jax.__version__)   # NB: optics_table (miepython), NOT miejax_lite
EOF
```
If it fails: install into the env with `$PY -m pip install <pkg>` (`miepython`, `numba`, `netCDF4` were
the snags). `miejax_lite` is **no longer needed** by the production path.

Profile count (expect **126**):
```bash
N=$($PY -c "import sys;sys.path.insert(0,'/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/src');import vocals_io;print(len(vocals_io.load_all_profiles('$VOCALS_DATA')))")
echo "N=$N profiles"
```

## Step 2 — run it (Venue A: SLURM arrays)

Per-task ≈ **1 h (thin) → ~2.5 h (thick)** at `--cpus-per-task=1`; the workers print `optics ready…`
then `radiance Jacobian done in <S>s` (a task silent for *hours* is the thread-oversubscription bug —
Troubleshooting). **cpt=1**, **--mem=12G**, **--time=08:00:00**, **%250** concurrency. Don't change
NQuad (48), bands, views, `IC_SIGMA_WEAK` (10), or worker physics.

**Profile worker — the `priormean` (HEADLINE) and `draw` arrays:**
```bash
cd /burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
for MODE in priormean draw; do
  mkdir -p docs/cached_results/_def_${MODE}_parts
  cat > /tmp/ic_${MODE}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ic_${MODE}
#SBATCH --array=0-$((N-1))%250
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/ic_${MODE}_%a.out
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 IC_MODE=${MODE}
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
srun --cpu-bind=cores $PY tests/supplementary/ic_worker_profile.py \$SLURM_ARRAY_TASK_ID \
   docs/cached_results/_def_${MODE}_parts/\$SLURM_ARRAY_TASK_ID.json
EOF
  sbatch /tmp/ic_${MODE}.sbatch
done
```

**Mechanism worker:**
```bash
mkdir -p docs/cached_results/_mech_parts
cat > /tmp/ic_mech.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ic_mech
#SBATCH --array=0-$((N-1))%250
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/ic_mech_%a.out
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
srun --cpu-bind=cores $PY tests/supplementary/ic_worker_mechanism.py \$SLURM_ARRAY_TASK_ID \
   docs/cached_results/_mech_parts/\$SLURM_ARRAY_TASK_ID.json
EOF
sbatch /tmp/ic_mech.sbatch
```

### Venue B — single multi-core server (only if no SLURM)
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_WAIT_POLICY=passive
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
for MODE in priormean draw; do
  seq 0 $((N-1)) | ENSEMBLE_NQUAD=48 IC_MODE=$MODE $PY tests/supplementary/_ic_parallel.py \
    tests/supplementary/ic_worker_profile.py docs/cached_results/info_content_robust_$MODE.json 4 4
done
```
(The launcher writes parts to `/tmp/_ic_parts_<stem>/`; the `.npz` sidecars land beside each part
`.json`. For the definitive analysis the **`.npz` parts dir** is what matters, not the merged json.)

## Step 3 — bundle the raw data (NO analysis on the cluster)

**All analysis happens back on the primary (jovyan), on the raw per-profile sidecars** — do **not** run
`ic_analysis_definitive.py` here, and do **not** merge or commit any result JSONs. Instead, zip
**everything raw** (the `_def_*_parts` npz+json, the `_mech_parts` json, and the SLURM logs) into a
**single file placed in `cloud_profile_retrieval/`** — the workspace root, one level above this repo —
for the primary to download manually (results are **not** delivered via Git this time):
```bash
cd /burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
rm -rf /tmp/ic_bundle && mkdir -p /tmp/ic_bundle/slurm_logs
cp -r docs/cached_results/_def_priormean_parts docs/cached_results/_def_draw_parts docs/cached_results/_mech_parts /tmp/ic_bundle/ 2>/dev/null
cp /tmp/ic_*_*.out /tmp/ic_bundle/slurm_logs/ 2>/dev/null
( cd /tmp && zip -rq ic_definitive_bundle.zip ic_bundle )         # or: tar czf ic_definitive_bundle.tgz ic_bundle
mv /tmp/ic_definitive_bundle.zip /burg-archive/home/dh3065/cloud_profile_retrieval/   # workspace root
echo "npz in bundle:"; find /tmp/ic_bundle -name '*.npz' | wc -l   # expect ~248 (≈124/mode × {priormean,draw})
ls -lh /burg-archive/home/dh3065/cloud_profile_retrieval/ic_definitive_bundle.zip
```

## Failure handling (fix yourself)

- **Hang on the first Jacobian** → Troubleshooting (thread oversubscription — the main risk).
- **OOM** (exit 137; unlikely at `--mem=12G`): raise `--mem`, or on Venue B change `4 4` → `2 8`.
- **Degenerate profiles** auto-write `{"skipped": ...}` — **expect exactly 1** per array (index 0, RF01,
  τ≈1585). Any other skip is worth a glance but not a failure.
- **`optics_table_10band.npz` missing / signature mismatch** → re-run Step 0 (a task will otherwise build
  it itself, wasting ~4 min and risking a write race — pre-build it).
- Do **not** change NQuad (48), bands, views, `IC_SIGMA_WEAK`, or the worker physics.

## Troubleshooting: a single Jacobian hangs for hours (thread oversubscription)

**Symptom:** a task sits on the *first* Jacobian for hours — no `radiance Jacobian done` print.
**Cause:** with no CPU-binding, XLA's Eigen pool *and* OpenBLAS size to the **whole node**, then the
`--cpus-per-task` cgroup throttles 64–128 threads onto a few CPUs → thrash. The fix needs **both** the
env caps **and** `srun --cpu-bind=cores` (XLA's pool sizes to *affinity*, not OMP). Both are in the
sbatch above. **Smoking-gun check (2 s):**
```bash
srun --cpus-per-task=1 --mem=12G python -c "import os; print('affinity', len(os.sched_getaffinity(0)), 'cpu_count', os.cpu_count())"
```
`affinity ≫ 4` ⇒ SLURM isn't binding ⇒ this is the cause. `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`
forces single-threaded XLA as a localizing test.

## Report back

Results are delivered **as the zip (Step 3), NOT via Git** — do **not** commit or push any JSON/npz.
After the arrays finish and the bundle is built, report: (1) records vs `skipped` per array (expect 1
skip — RF01 τ≈1585); (2) the **`npz in bundle` count** (≈248) and the bundle path + size; (3) wall time
/ venue; (4) any errors. The primary downloads `cloud_profile_retrieval/ic_definitive_bundle.zip` and
runs all analysis (`ic_analysis_definitive.py`) on jovyan.
