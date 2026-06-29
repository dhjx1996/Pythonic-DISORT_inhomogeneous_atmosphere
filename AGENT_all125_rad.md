# Delegated task — precompute the OSSE radiances (synthetic L1B), all VOCALS profiles

*(Hand-off for a Sonnet agent. Code moves by git (sync first); the **radiance cache moves two
ways**: it STAYS on the cluster for the downstream IC/retrieval batches, AND a copy is zipped
to the workspace root for the primary to download. Preserve this file as a handoff template —
don't delete it. This is **batch 1 of 3**: `rad` (this) → `ic` re-run → `fr` retrievals.)*

## What this is — and why it exists

We compute `y = F(truth)` for every VOCALS profile **once** at the correct forward config, cache
it, and all downstream workers load it instead of regenerating. Three bugs fixed vs the prior
run (all captured in `tests/supplementary/osse_config.py`, the **single source of truth**):

- **TMS fix (`NLeg_all=1536`, `n_gl=4096`):** the NT single-scatter correction needed 1536
  Legendre moments (not 128 or 1024) to avoid Gibbs-ringing the short-λ Mie peaks NEGATIVE.
- **NFourier=24** (was 8): re-tuned on the fixed forward; the old 8 left ~10 % truncation error
  at the short bands.
- **tol=1e-4**: §A3 probe showed tol=1e-4 is sufficient (indistinguishable from 1e-5 at τ≲20;
  both converge at τ≈36). Set via `SOLVER_TOL=1e-4` in the sbatch.

The optics table is rebuilt at `NLeg=1536, n_gl=4096` (Step 0). One forward per profile at
native in-situ resolution (14–111 ODE nodes → unique XLA compile per task); the cache removes
this from the IC and retrieval critical paths. **Run on GPU** (240-way bands×modes vmap; §A2
probe: eval-only forward ~36 s on A100 vs ~1000 s CPU — even RTX8000 at 4.1× fits in 2 h).

## Repo & paths

```bash
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
```

Output stays at `$ROOT/docs/cached_results/osse_radiances.npz`; zip copy at
`/burg-archive/home/dh3065/cloud_profile_retrieval/osse_radiances_bundle.zip`.

## Step 0 — GPU env (one-time; skip if already done)

The conda JAX env has `jaxlib=0.10.2` but `jax-cuda12-plugin/pjrt=0.9.2` (version mismatch
found in probe #1 — JAX disables the GPU plugin). Upgrade to match:

```bash
$PY -m pip install --quiet jax-cuda12-plugin==0.10.2 jax-cuda12-pjrt==0.10.2
# Verify:
JAX_PLATFORMS=cuda $PY -c "import jax; print('GPU OK, devices:', jax.devices())"
# Expect: [CudaDevice(id=0)] (or similar). If "no CUDA device found", the upgrade didn't take.
```

## Step 1 — optics table (build once; ~4 min)

```bash
cd $ROOT && git fetch origin && git reset --hard origin/main
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
$PY - <<'EOF'
import sys; sys.path.insert(0,'tests/supplementary'); sys.path.insert(0,'src')
import osse_config as oc
opt = oc.load_optics('tests/supplementary/optics_table_10band_nleg1536_re20.npz')
print("optics OK; sig", oc.signature()[1])
EOF
```

## Step 2 — env check

```bash
$PY - <<'EOF'
import os, sys
sys.path.insert(0,'src'); sys.path.insert(0,'tests/supplementary')
import jax, osse_config as oc, vocals_io as vio
print("ENV OK", jax.__version__, "| sig", oc.signature()[1])
print("NLeg_all:", oc.NLEG_ALL, "| NFourier:", oc.NFOURIER[:3], "| tol (env):", oc.SOLVER_TOL)
N = len(vio.load_all_profiles(os.environ['VOCALS_DATA'])); print("N profiles =", N)   # expect 126
EOF
```

## Step 3 — run the array (one forward per profile, GPU)

**Affinity (already in the worker):** `runtime_setup.setup()` claims an atomic per-node core
slot BEFORE JAX starts (commits 8fc43cf/a5ab9a7 — verified live: 39 nodes, 115 tasks, 0
collisions). No `--cpu-bind=cores` or XLA single-thread flag needed.

**GPU note:** `--partition=crew1,ocp_gpu,short` lists ALL eligible partitions so the scheduler
picks the soonest-free node across dedicated + shared pools. A100 is fastest (1×); V100S ≈1.5×;
A40 ≈3×; RTX8000 ≈4.1× A100 time — all pass the f64 canary and beat CPU by >5×. No `-C`
constraint: let the scheduler allocate freely.

```bash
cd $ROOT && git fetch origin && git reset --hard origin/main
N=126
mkdir -p docs/cached_results/_rad_parts docs/cached_results/_rad_logs
cat > /tmp/rad.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=osse_rad
#SBATCH --account=crew
#SBATCH --array=0-125%250
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=__RADLOGS__/rad_%a.out

export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4
export OPTICS_CACHE=__ROOT__/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export VOCALS_DATA=__VOCALS__
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
srun __PY__ tests/supplementary/generate_osse_radiances.py $SLURM_ARRAY_TASK_ID \
     docs/cached_results/_rad_parts
SBATCH

# Substitute real paths:
sed -i \
  "s|__ROOT__|$ROOT|g; s|__PY__|$PY|g; \
   s|__VOCALS__|$VOCALS_DATA|g; \
   s|__RADLOGS__|$ROOT/docs/cached_results/_rad_logs|g" \
  /tmp/rad.sbatch
sbatch /tmp/rad.sbatch
```

A healthy task prints `[idx] FLIGHT tau=..: y=320 (native N nodes) in <S>s | sig=<hash>`.
GPU forward eval is ~36 s on A100 (eval-only, 10 bands); compile adds ~2–5 min per task
(unique per native-profile shape, but GPU compile is ~10× faster than CPU). Budget 2 h covers
RTX8000 on the stiffest profiles with comfortable margin. Index 20 (thickest RF03) previously
timed out at 8 h on CPU; GPU handles it.

## Step 4 — consolidate + sanity-check

```bash
cd $ROOT
$PY tests/supplementary/generate_osse_radiances.py consolidate \
   docs/cached_results/_rad_parts docs/cached_results/osse_radiances.npz
# Expect: "consolidated 125 profiles -> ... (sig <hash>); skipped [0]"  (index 0 = RF01 τ≈1585)
```

The signature in the consolidated file MUST equal `osse_config.signature()[1]`. Leave
`osse_radiances.npz` in place — **batches 2–3 load it from there.**

## Step 5 — bundle a copy for the primary (record; NOT git)

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

- **"no CUDA device" / GPU not visible** → Step 0 plugin upgrade didn't apply. Check
  `$PY -m pip show jax-cuda12-plugin` — version must be 0.10.2. If `pip install` silently
  skipped (already-satisfied), force: `$PY -m pip install --upgrade --force-reinstall jax-cuda12-plugin==0.10.2 jax-cuda12-pjrt==0.10.2`.
- **Task silent or very slow** → affinity pin should already print `[runtime] pinned N cores`.
  Fallback: add `export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"` to the sbatch.
- **Signature mismatch on consolidate** → some sidecars were generated against a stale
  `osse_config` (different NLeg_all or NFourier). Re-sync the repo (`git reset --hard
  origin/main`) and re-run the offending indices.
- **Degenerate profiles** auto-write `{"skipped": ...}` — **expect exactly 1** (index 0,
  RF01, τ≈1585).

## Report back

After the array + consolidate finish: (1) profiles consolidated vs skipped (expect 125 / 1);
(2) the **signature hash** (must match `osse_config.signature()[1]`); (3) per-task wall-time
range (A100 vs slower cards) and which GPU types were used; (4) bundle path + size; (5) any
errors. Then **stop and wait** — batch 2 (IC re-run) is a separate hand-off once the primary
confirms the radiances.
