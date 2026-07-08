#!/bin/bash
# ve_rerun campaign (hpc/AGENT_ve_rerun.md) — Step 3: full FR array, config A only, CPU primary
# venue (mirrors batch-3's actual launch: CPU short partition, 12h walls, L1-checkpoint resumable
# across successive submissions; GPU used for the gate + straggler realloc, not here).
# MAJOR DEVIATIONS vs the spec: OSSE_VEFF=0.046, OSSE_RE_MAX=22 (raised r_e clamp), MAX_N_OUTER=2
# (every profile opted into placement re-meshing, not just flagged structural-misfit configs).
# Indices 1..125 are the real profiles (0 = degenerate header, excluded); idx-47 excluded here —
# it is covered by the separate A100 gate job (avoid a write race on the shared _47 sidecar).
#SBATCH --account=crew
#SBATCH --array=1-46,48-125%125
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --time=11:55:00
#SBATCH --job-name=ve046_fr
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_rerun_logs/fr_%A_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01 SOLVER_TOL=1e-4
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A MAX_N_OUTER=2
export FR_SETUP_CACHE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"   # REQUIRED: single-thread XLA at cpus-per-task=1
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046.npz RADIANCE_TOL=1e-4
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
echo "[ve046_fr] idx=$IDX wall-start $(date '+%F %T') host=$(hostname)"
srun --cpu-bind=cores $PY scripts/retrieval_worker.py $IDX \
   $ROOT/runs/_ve_rerun_fr_parts/$IDX
echo "[ve046_fr] idx=$IDX wall-end $(date '+%F %T') rc=$?"
