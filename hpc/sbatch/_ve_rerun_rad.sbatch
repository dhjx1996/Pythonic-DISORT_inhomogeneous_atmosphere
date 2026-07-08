#!/bin/bash
# ve_rerun campaign (hpc/AGENT_ve_rerun.md) — Step 2: regenerate the truth-radiance cache at the
# new signature (OSSE_VEFF=0.046, OSSE_RE_MAX=22 — MAJOR DEVIATION vs the spec's re20 table).
# One forward per profile (GPU). Consolidate separately once the array finishes.
#SBATCH --job-name=ve046_rad
#SBATCH --account=crew
#SBATCH --array=0-125%250
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_rerun_logs/rad_%A_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
# cuSPARSE fix: login profile's ~/cuda-12.6 shadows the pip-bundled CUDA libs on GPU nodes.
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
srun $PY scripts/generate_osse_radiances.py $SLURM_ARRAY_TASK_ID \
     runs/_ve_rerun_rad_parts
