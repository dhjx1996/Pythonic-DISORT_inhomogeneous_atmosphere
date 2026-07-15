#!/bin/bash
# idx32/49/57 pathology re-attempt (2026-07-13, user-directed): the REMESH_CHI2_THR=0.1
# fixed-count re-mesh re-run fixed idx32 but made idx49/57 worse (see
# runs/_ve046_tik_fr_parts/_idx{N}_pathology_orig). This run escalates per the code's own
# RemeshWarning advice and removes the τ_bot-anchor luck factor:
#   * MAX_N_OUTER=3 (node-COUNT re-mesh allowed) + REMESH_CHI2_THR=0.1
#   * CLIM_TAU_BOT=<truth τ_bot> per index — the climatological τ_bot prior mean is
#     manually fed the TRUTH (τ_bot still fully retrieved; σ unchanged)
#   * standard climatological prior / starting point otherwise (config A, λ_curv=1.0,
#     same caches as the ve046 campaign — comparable apples-to-apples)
# Results go to a DEDICATED parts dir (never the campaign dir): at consolidation the
# user picks the canonical record per index among orig / remesh / this run.
# Submit: sbatch --array=32,49,57 _ve046_tik_pathology3_gpu.sbatch
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100s
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --job-name=ve046path3
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_tik_logs/ve046_path3_%A_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
case $IDX in                       # truth τ_bot (from the campaign _A.npz truth_tau_bot)
  32) export CLIM_TAU_BOT=5.004702213875641 ;;
  49) export CLIM_TAU_BOT=36.54197688698907 ;;
  54) export CLIM_TAU_BOT=36.187758094014576 ;;   # 2026-07-13: same fingerprint as idx49
  57) export CLIM_TAU_BOT=3.2450559493249638 ;;
  *)  echo "unexpected index $IDX (no truth tau_bot wired)"; exit 1 ;;
esac
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A
export MAX_N_OUTER=3 REMESH_CHI2_THR=0.1
export CURVATURE_LAMBDA=1.0                              # match the ve046 campaign
export FR_SETUP_CACHE=1
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
OUT=$ROOT/runs/_ve046_tik_pathology3_parts
mkdir -p $OUT $ROOT/runs/_ve_tik_logs
echo "[ve046path3] idx=$IDX CLIM_TAU_BOT=$CLIM_TAU_BOT max_n_outer=3 thr=0.1 wall-start $(date '+%F %T') host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY scripts/retrieval_worker.py $IDX $OUT/$IDX
echo "[ve046path3] idx=$IDX wall-end $(date '+%F %T') rc=$?"
