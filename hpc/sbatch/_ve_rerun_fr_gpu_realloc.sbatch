#!/bin/bash
# ve_rerun campaign — re-allocate stragglers (walled with no checkpoint = died in setup) from CPU
# to GPU. Submit with an explicit index list, e.g. sbatch --array=12,40 _ve_rerun_fr_gpu_realloc.sbatch
# PRECONDITION: scancel the index's CPU array task first (avoid a write race on _ve_rerun_fr_parts/<idx>).
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:55:00
#SBATCH --job-name=ve046_gpu
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_rerun_logs/gpu_%A_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A MAX_N_OUTER=2
export FR_SETUP_CACHE=1
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[ve046_gpu] idx=$IDX wall-start $(date '+%F %T') host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY scripts/retrieval_worker.py $IDX $ROOT/runs/_ve_rerun_fr_parts/$IDX
echo "[ve046_gpu] idx=$IDX wall-end $(date '+%F %T') rc=$?"
