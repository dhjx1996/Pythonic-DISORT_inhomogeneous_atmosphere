#!/bin/bash
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=frgate47
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_gate/frgate_47.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4
export JAX_COMPILATION_CACHE_DIR=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_jax_cache_fr
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun $PY tests/supplementary/retrieval_worker.py 47 /burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_gate/47
