#!/bin/bash
# 2026-07-13 fq/ve046 IC bundle recompute (user-directed; feeds the notebook §16 refresh).
# One parametrized driver for all three arrays of AGENT_all125_ic.md, under the CORRECTED
# paradigm: v_e=0.046, re_max=22, fixed-quadrature optics (the moving-quadrature table's
# resonance noise corrupted d(optics)/dr_e 200-600x -- exactly the Jacobians IC is made of).
# Fresh parts dirs -- the canonical 2026-07-06 v_e=0.10 bundle (runs/_ic_{A,B,C}_parts)
# is never touched; the aggregation targets the new dirs explicitly.
# Submit (after ../data/osse_radiances_ve046_fq.npz exists -- full 124-profile fq cache):
#   sbatch --export=ALL,ARR=A --job-name=icAfq46 --array=0-124 <this>
#   sbatch --export=ALL,ARR=B --job-name=icBfq46 --array=0-124 <this>
#   sbatch --export=ALL,ARR=C --job-name=icCfq46 --array=0-124 <this>
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --nice=2000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ic_fq046_logs/%x_%A_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export OSSE_VEFF=0.046 OSSE_RE_MAX=22
export OSSE_QUADRATURE=fixed OSSE_RE_GRID_N=181 OSSE_N_RADII=4096
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22_fq.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046_fq.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
case $ARR in
  A) export IC_MODE=priormean; SCRIPT=ic_worker_profile.py ;;
  B) export IC_MODE=draw;      SCRIPT=ic_worker_profile.py ;;
  C)                           SCRIPT=ic_worker_mechanism.py ;;
  *) echo "ARR must be A|B|C"; exit 1 ;;
esac
OUT=$ROOT/runs/_ic_${ARR}_fq046_parts
mkdir -p $OUT $ROOT/runs/_ic_fq046_logs
echo "[ic${ARR}fq46] idx=$IDX wall-start $(date '+%F %T') host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY scripts/$SCRIPT $IDX $OUT/$IDX.json
echo "[ic${ARR}fq46] idx=$IDX wall-end $(date '+%F %T') rc=$?"
