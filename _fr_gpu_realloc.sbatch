#!/bin/bash
# Re-allocate expensive FR profiles from CPU to GPU. Submit with an explicit index list, e.g.
#   sbatch --array=12,40,88 _fr_gpu_realloc.sbatch
# Writes the canonical _fr_parts/<idx> and RESUMES from the CPU-written L1 checkpoint (platform-agnostic
# numpy), so no CPU progress is lost. nice'd below IC so it only uses leftover GPU space (IC keeps
# priority). Any GPU beats CPU's ~20x penalty for the expensive tail. PRECONDITION: scancel the index's
# CPU array task first (avoid a write race on _fr_parts/<idx>). Permission gated on golden 47/49 passing.
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --nice=5000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:55:00
#SBATCH --job-name=fr_gpu
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/docs/cached_results/_fr_parts/logs/gpu_%a.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export JAX_COMPILATION_CACHE_DIR=$ROOT/docs/cached_results/_jax_cache_fr   # GPU compile cache
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band_nleg1536_re20.npz
export RADIANCE_CACHE=$ROOT/docs/cached_results/osse_radiances.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[fr_gpu] idx=$IDX wall-start $(date '+%F %T') host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
srun $PY tests/supplementary/retrieval_worker.py $IDX $ROOT/docs/cached_results/_fr_parts/$IDX
echo "[fr_gpu] idx=$IDX wall-end $(date '+%F %T') rc=$?"
