#!/bin/bash
# Golden-index FR validation = REAL retrieval on idx-49 (golden, != gate's 47), A100, real worker
# config. Output to the canonical _fr_parts/49 so it IS the kept FR result (incl. IC-at-x_hat) and
# the full run skips 49. Checkpointed -> driver does a deliberate kill+resume, then cross-checks
# _fr_parts/49 vs golden probe_49_tol1e-4_A.npz (correctness + production resume in one keepable run).
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=frgold49
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_parts/logs/gold49_%j.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr   # GPU compile cache (shared w/ gate, warm)
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[frgold49] start $(date '+%F %T') host=$(hostname)"
srun $PY tests/supplementary/retrieval_worker.py 49 $ROOT/docs/cached_results/_fr_parts/49
echo "[frgold49] end $(date '+%F %T') rc=$?"
