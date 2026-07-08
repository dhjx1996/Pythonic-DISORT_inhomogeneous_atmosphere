#!/bin/bash
# ve_rerun campaign — bug-catcher gate on one representative profile (idx-47, same as the
# batch-3 gate) before committing to the full array. GPU/A100, config A only, MAX_N_OUTER=2,
# OSSE_VEFF=0.046, OSSE_RE_MAX=22 (MAJOR DEVIATIONS). Its output folds into the full array as a
# free completed profile (writes into the same runs/_ve_rerun_fr_parts/47 prefix).
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=ve046_gate47
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_rerun_logs/gate_47.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A MAX_N_OUTER=2
export FR_SETUP_CACHE=1
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[ve046_gate] wall-start $(date '+%F %T') host=$(hostname)"
srun $PY scripts/retrieval_worker.py 47 $ROOT/runs/_ve_rerun_fr_parts/47
echo "[ve046_gate] wall-end $(date '+%F %T') rc=$?"
