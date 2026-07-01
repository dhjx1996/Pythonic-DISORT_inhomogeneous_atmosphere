#!/bin/bash
#SBATCH --account=crew
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --job-name=fr_l2test
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_ctest/logs/l2test.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01 SOLVER_TOL=1e-4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr_cpu
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz RADIANCE_TOL=1e-4
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=passive
export DIAG_IDX=95 L2_CKPT=$ROOT/docs/cached_results/_fr_ctest/95.setup.npz
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
srun --cpu-bind=cores $PY _fr_l2_test.py
