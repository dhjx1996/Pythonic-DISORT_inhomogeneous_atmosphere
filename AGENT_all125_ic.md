# Delegated task — re-run the information-content profiling, all VOCALS profiles (FIXED forward)

*(Hand-off for a Sonnet agent. Code moves by git (sync first); the **raw K sidecars move as ONE
zip the primary downloads manually** — NOT via git. Preserve this file as a handoff template —
don't delete it. This is **batch 2 of 3**: `rad` (done) → `ic` (this) → `fr` retrievals.
**Do not start until the primary confirms the batch-1 radiance cache is consolidated and in
place at `docs/cached_results/osse_radiances.npz`.***)*

## What this is — and why we are re-running it

The §15 information-content study was **contaminated by a forward-model bug** and must be re-run
on the **fixed** forward. All config is now in `tests/supplementary/osse_config.py`, the single
source of truth (do **not** edit it):

- **TMS fix (`NLeg_all=1536`, `n_gl=4096`):** the Nakajima–Tanaka single-scatter correction
  reconstructed the phase function from only 128 Legendre moments, which Gibbs-rings the short-λ
  Mie forward peaks NEGATIVE → unphysical (down to −2.5) radiances at short bands contaminated
  the original §15 IC. `osse_config` now carries `NLEG_ALL=1536`.
- **NFourier=24** (was adaptive ~8): fixed per-band, re-tuned on the corrected forward.
- **tol=1e-4**: §A3 probe showed tol=1e-4 sufficient (indistinguishable from 1e-5 at τ≲20). Set
  via `SOLVER_TOL=1e-4` in the sbatch.
- **r_e clamp = [2,20]** (was 25); **irregular golden-ratio views** (32-view superset for IC; the
  24-subset goes to the retrieval) — replaces the regular μ-fan that aliased at 0.55 µm.

The optics table is the **same** one batch 1 built (`optics_table_10band_nleg1536_re20.npz`).
The radiance cache (`osse_radiances.npz`) built by batch 1 is **REQUIRED** — both IC workers
assert its signature before loading. The signature must match `osse_config.signature()[1]`.

**Do not pre-judge the §14 / §15 findings** — just produce the clean sidecars. The primary will
re-derive.

## What each task does

One profile-worker (`ic_worker_profile.py`) parametrized by `IC_MODE` runs as **2 SLURM arrays**
(`priormean` / `draw`). One mechanism-worker (`ic_worker_mechanism.py`) runs as a **3rd array**.
All three use the same optics table and radiance cache.

| Array | Script | `IC_MODE` | Purpose |
|---|---|---|---|
| Array A | `ic_worker_profile.py` | `priormean` | Jacobian at LOO prior mean |
| Array B | `ic_worker_profile.py` | `draw` | Jacobian at adiabatic-draw IC |
| Array C | `ic_worker_mechanism.py` | (hardcoded) | Reach/DOFS by view-type |

Array B and C are independent of A — all three can be submitted simultaneously after batch 1
is confirmed.

## Repo & paths

```bash
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz
```

## Step 0 — GPU env (one-time; skip if already done in batch 1)

```bash
$PY -m pip install --quiet jax-cuda12-plugin==0.10.2 jax-cuda12-pjrt==0.10.2
# FIX (batch-1 lesson): prepend the pip nvidia CUDA lib dirs — the login profile's
# ~/cuda-12.6/lib64 on LD_LIBRARY_PATH shadows them and breaks cuSPARSE at 0.10.2, so the
# 'cuda' backend silently disappears (['cpu','tpu'] only). Required in EVERY GPU job below too.
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
JAX_PLATFORMS=cuda $PY -c "import jax; print('GPU OK, devices:', jax.devices())"
```

## Step 1 — confirm batch-1 prerequisites

```bash
cd $ROOT && git fetch origin && git reset --hard origin/main
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
$PY - <<'EOF'
import os, sys; sys.path.insert(0,'tests/supplementary'); sys.path.insert(0,'src')
import numpy as np, osse_config as oc, vocals_io as vio
_, sig = oc.signature()
d = np.load(os.environ['RADIANCE_CACHE'], allow_pickle=True)
got = str(d['signature_hash'])
print(f"sig match: {got == sig}  (want={sig} got={got})")
N = len(vio.load_all_profiles(os.environ['VOCALS_DATA'])); print(f"N profiles={N}")
EOF
# Expect: "sig match: True" — if False, batch 1 needs a re-run.
```

## Step 2 — submit all three arrays simultaneously

**Timing (from §A2/A3 probes):** GPU jacfwd = ~128 s on A100 (two Jacobian calls → ~4 min A100;
~17 min RTX8000). Compile adds ~5–10 min per unique profile (retrieval grid = 5 s_ref nodes →
fixed shape, so most compilation is shared across profiles after the first). **Wall set to 12 h**
(conservative standing default — provisional, to be tightened post-run from observed times: batch-1
forwards already hit 40 min, so IC Jacobians can exceed 1 h on RTX8000/A40). **Mem 32 G** (batch-1
saw SIGABRT in the XLA CPU threadpool at 16 G on the cheaper forward). All partitions listed.

**Affinity (already in both workers):** `runtime_setup.setup()` claims an atomic per-node core
slot before JAX (commits 8fc43cf/a5ab9a7 — verified 39 nodes, 115 tasks, 0 collisions).

```bash
cd $ROOT && git fetch origin && git reset --hard origin/main
mkdir -p docs/cached_results/_ic_{A,B,C}_parts docs/cached_results/_ic_{A,B,C}_logs

# ------- Array A: profile-worker, priormean -------
cat > /tmp/icA.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=ic_profile_pm
#SBATCH --account=crew
#SBATCH --array=0-125%250
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=__LOGA__/icA_%a.out

export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 RADIANCE_TOL=1e-4
export IC_MODE=priormean
export OPTICS_CACHE=__ROOT__/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=__ROOT__/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=__VOCALS__
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=__PY__
# FIX (batch-1 lesson): prepend pip nvidia CUDA libs so cuSPARSE loads (login ~/cuda-12.6
# shadows them otherwise -> 'cuda' backend disappears).
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun __PY__ tests/supplementary/ic_worker_profile.py \
     $SLURM_ARRAY_TASK_ID __ROOT__/docs/cached_results/_ic_A_parts
SBATCH

# ------- Array B: profile-worker, draw -------
cat > /tmp/icB.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=ic_profile_draw
#SBATCH --account=crew
#SBATCH --array=0-125%250
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=__LOGB__/icB_%a.out

export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 RADIANCE_TOL=1e-4
export IC_MODE=draw
export OPTICS_CACHE=__ROOT__/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=__ROOT__/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=__VOCALS__
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=__PY__
# FIX (batch-1 lesson): prepend pip nvidia CUDA libs so cuSPARSE loads (login ~/cuda-12.6
# shadows them otherwise -> 'cuda' backend disappears).
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun __PY__ tests/supplementary/ic_worker_profile.py \
     $SLURM_ARRAY_TASK_ID __ROOT__/docs/cached_results/_ic_B_parts
SBATCH

# ------- Array C: mechanism-worker -------
cat > /tmp/icC.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=ic_mechanism
#SBATCH --account=crew
#SBATCH --array=0-125%250
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=__LOGC__/icC_%a.out

export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 RADIANCE_TOL=1e-4
export OPTICS_CACHE=__ROOT__/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=__ROOT__/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=__VOCALS__
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=__PY__
# FIX (batch-1 lesson): prepend pip nvidia CUDA libs so cuSPARSE loads (login ~/cuda-12.6
# shadows them otherwise -> 'cuda' backend disappears).
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun __PY__ tests/supplementary/ic_worker_mechanism.py \
     $SLURM_ARRAY_TASK_ID __ROOT__/docs/cached_results/_ic_C_parts
SBATCH

for F in /tmp/icA.sbatch /tmp/icB.sbatch /tmp/icC.sbatch; do
  sed -i \
    "s|__ROOT__|$ROOT|g; s|__PY__|$PY|g; s|__VOCALS__|$VOCALS_DATA|g; \
     s|__LOGA__|$ROOT/docs/cached_results/_ic_A_logs|g; \
     s|__LOGB__|$ROOT/docs/cached_results/_ic_B_logs|g; \
     s|__LOGC__|$ROOT/docs/cached_results/_ic_C_logs|g" $F
done
sbatch /tmp/icA.sbatch
sbatch /tmp/icB.sbatch
sbatch /tmp/icC.sbatch
```

A healthy task prints `[idx] FLIGHT tau=..: radiance Jacobian done in <S>s (n_int=<N>)`.
Profile indices 0 (RF01, τ≈1585) will auto-write `{"skipped": ...}` — expected.

## Step 3 — sanity-check completeness (after arrays finish)

```bash
cd $ROOT
for ARR in A B C; do
  OK=$(ls docs/cached_results/_ic_${ARR}_parts/*.json 2>/dev/null \
       | xargs -I{} python -c "import json; d=json.load(open('{}'));\
         print('ok' if 'index' in d and 'skipped' not in d else '')" \
       | grep -c ok)
  SK=$(ls docs/cached_results/_ic_${ARR}_parts/*.json 2>/dev/null \
       | xargs -I{} python -c "import json; d=json.load(open('{}'));\
         print('skip' if 'skipped' in d else '')" 2>/dev/null | grep -c skip)
  echo "Array $ARR: ok=$OK skipped=$SK"
done
# Expect each: ok=125 skipped=1 (126 indices; idx-0 RF01 τ≈1585 is the lone skip — same 125
# profiles as the batch-1 radiance cache and the prior IC run)
```

## Step 4 — bundle everything for the primary (NOT git)

```bash
cd $ROOT
rm -rf /tmp/ic_bundle && mkdir -p /tmp/ic_bundle/logs_{A,B,C}
# JSON sidecars from all three arrays
cp docs/cached_results/_ic_A_parts/*.json /tmp/ic_bundle/
cp docs/cached_results/_ic_B_parts/*.json /tmp/ic_bundle/ 2>/dev/null
cp docs/cached_results/_ic_C_parts/*.json /tmp/ic_bundle/ 2>/dev/null
# NPZ K-sidecars (from profile worker; large but required)
cp docs/cached_results/_ic_A_parts/*.npz /tmp/ic_bundle/ 2>/dev/null
cp docs/cached_results/_ic_B_parts/*.npz /tmp/ic_bundle/ 2>/dev/null
# SLURM logs
cp docs/cached_results/_ic_A_logs/*.out /tmp/ic_bundle/logs_A/ 2>/dev/null
cp docs/cached_results/_ic_B_logs/*.out /tmp/ic_bundle/logs_B/ 2>/dev/null
cp docs/cached_results/_ic_C_logs/*.out /tmp/ic_bundle/logs_C/ 2>/dev/null
( cd /tmp && zip -rq ic_bundle.zip ic_bundle )
mv /tmp/ic_bundle.zip /burg-archive/home/dh3065/cloud_profile_retrieval/
ls -lh /burg-archive/home/dh3065/cloud_profile_retrieval/ic_bundle.zip
```

## Troubleshooting

- **Signature mismatch in worker** → `RADIANCE_CACHE` points to the wrong file (stale tol=1e-3
  cache at `543eee...`). Confirm `$RADIANCE_CACHE` resolves to `osse_radiances.npz` from batch 1
  (not `osse_radiances_v543.npz`).
- **"no CUDA device"** → see Step 0; force-reinstall if `jax-cuda12-plugin` shows 0.9.2.
- **Missing `oc.VIEW_MU_FULL` / `oc.RETRIEVAL_VIEW_IDX`** → repo out of date. Re-sync with
  `git reset --hard origin/main`.
- **Array B (draw) degenerate beyond idx-0** → the LOO climatology exclude_flight step fails if
  `flight` is empty; check the log for the skipped index's `flight` field.

## Report back

After all three arrays + sanity check finish: (1) ok/skip counts for each array (expect 125/1);
(2) per-task wall-time range by GPU type; (3) bundle path + size; (4) any repeated failures. Then
**stop and wait** — batch 3 (full retrievals) is a separate hand-off once the primary reviews the IC.
