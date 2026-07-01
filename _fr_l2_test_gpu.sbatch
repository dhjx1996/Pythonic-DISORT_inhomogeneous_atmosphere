#!/bin/bash
# GPU L2-equivalence test (A100-targeted for a fast verdict ~2.5h vs ~5-6h on slow cards / never on CPU).
# Runs build_forward_and_obs twice (compute+WRITE cache, then LOAD) and asserts bit-exact
# K_list/s_grid/tau_bot_pre + forward/Jacobian evals. Distinct L2_CKPT + log from the CPU test.
#SBATCH --account=crew
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=fr_l2gpu
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_ctest/logs/l2test_gpu.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01 SOLVER_TOL=1e-4 MODE_MAP=vmap
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz RADIANCE_TOL=1e-4
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=passive
export DIAG_IDX=95 L2_CKPT=$ROOT/docs/cached_results/_fr_ctest/95_gpu.setup.npz
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun $PY _fr_l2_test.py
