#!/bin/bash
# Batch-3 FR full retrieval, ALL 125 profiles on CPU (short partition, 12h walls, L1-checkpoint
# resumable across successive submissions). Indices 1..125 are the real profiles (0 = degenerate
# header, excluded). Config: float64 (X64=1), SOLVER_TOL=1e-4, NQuad=48, COST_RTOL=0.01 — exactly
# the IC/FR operating point. cpus-per-task=1 + single-thread XLA (avoids the oversubscription bug).
# Resubmit unfinished indices (sbatch --array=<list> _fr_all125_cpu.sbatch) to resume from checkpoint.
#SBATCH --account=crew
#SBATCH --array=1-125%125
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --time=11:55:00
#SBATCH --job-name=fr_cpu
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_parts/logs/fr_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01 SOLVER_TOL=1e-4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"   # REQUIRED: single-thread XLA at cpus-per-task=1
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr_cpu   # L3 persistent compile cache (DEFAULT thresholds — keep file count low on Lustre)
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz RADIANCE_TOL=1e-4   # truth cache (sig d71a8559, tol=1e-4, idx 1..125)
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
echo "[fr_cpu] idx=$IDX wall-start $(date '+%F %T') host=$(hostname)"
srun --cpu-bind=cores $PY tests/supplementary/retrieval_worker.py $IDX \
   $ROOT/docs/cached_results/_fr_parts/$IDX
echo "[fr_cpu] idx=$IDX wall-end $(date '+%F %T') rc=$?"
