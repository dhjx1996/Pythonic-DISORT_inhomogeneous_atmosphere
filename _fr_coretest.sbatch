#!/bin/bash
#SBATCH --account=crew
#SBATCH --array=3,30
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --job-name=fr_ctest
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_ctest/logs/ct_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01 SOLVER_TOL=1e-4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr_cpu
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz RADIANCE_TOL=1e-4
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
echo "[ctest] idx=$IDX cpt=$SLURM_CPUS_PER_TASK wall-start $(date '+%T')"
srun --cpu-bind=cores $PY tests/supplementary/retrieval_worker.py $IDX $ROOT/docs/cached_results/_fr_ctest/$IDX
