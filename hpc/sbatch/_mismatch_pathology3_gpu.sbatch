#!/bin/bash
# 2026-07-14 mismatch-run pathology treatment (user-directed; the now-STANDARD recipe):
# re-run mismatch pathologies with the escalation + inverted-acceptance logic
# (MAX_N_OUTER=3, REMESH_CHI2_THR=0.1, remesh_accept_rtol=0.5 in-code) AND the TRUE
# tau_bot fed as the climatological prior mean (CLIM_TAU_BOT; tau_bot still fully
# retrieved). Justification (user): tau_bot retrieval is standard practice and can be
# improved externally — it is not the innovation under test, so de-lucking it is fair.
# NB (2026-07-14 user correction): the matched-campaign THR=0.1 is WRONG here — the
# mismatch model-error chi2 floor is ~0.3-1.9, so 0.1 would fire escalation on every
# profile forever. THR=2 for idx39/78 "for lack of a better option" (fires iff the fit is
# ABOVE the floor's ceiling); tune the mismatch pathology threshold as good mismatch
# retrievals accumulate. Campaign/code default stays remesh_if_chi2_red_gt=2.0.
# Targets (pilot sweep 2026-07-14): idx39 (d_w1=+2.11 vs baseline — the watch case) and
# idx78 (borderline, +0.68). Everything else inherits the pilot env EXACTLY
# (mismatch _mismatch_pilot.sbatch, incl. the corruption-fix RETRIEVAL_OPTICS_CACHE).
# Output: DEDICATED parts dir; user picks canonical at consolidation.
# Submit: sbatch --array=39,78 _mismatch_pathology3_gpu.sbatch
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100s
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --job-name=mmPath3
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_fq_logs/mmpath3_%A_%a.out

ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
case $IDX in                       # truth tau_bot (campaign _A.npz truth_tau_bot)
  39) export CLIM_TAU_BOT=27.586242757183513 ;;
  78) export CLIM_TAU_BOT=32.87579508961966 ;;
  *)  echo "unexpected index $IDX (no truth tau_bot wired)"; exit 1 ;;
esac
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export OSSE_QUADRATURE=fixed OSSE_RE_GRID_N=181 OSSE_N_RADII=4096
export FR_CONFIGS=A CURVATURE_LAMBDA=1.0
export MAX_N_OUTER=3 REMESH_CHI2_THR=2
export FR_SETUP_CACHE=1
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
# TRUTH world: v_e=0.046 (env governs oc.signature() -> gates RADIANCE_CACHE)
export OSSE_VEFF=0.046 OSSE_RE_MAX=22
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22_fq.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046_fq.npz
# RETRIEVAL assumption: v_e=0.10, LOAD-ONLY dedicated table (corruption fix, pilot header)
export RETRIEVAL_OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re20_fq_mismatch.npz
OUT=$ROOT/runs/_mismatch_pathology3_parts
mkdir -p $OUT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[mmPath3] idx=$IDX CLIM_TAU_BOT=$CLIM_TAU_BOT max_n_outer=3 thr=2 delta=0.5 wall-start $(date '+%F %T') gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY scripts/retrieval_worker.py $IDX $OUT/$IDX
echo "[mmPath3] idx=$IDX wall-end $(date '+%F %T') rc=$?"
