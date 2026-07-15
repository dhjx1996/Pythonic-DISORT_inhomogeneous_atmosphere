#!/bin/bash
# 2026-07-12 mismatched-v_eff experiment (user request): does the retrieval assuming the
# WRONG drop-size-distribution width hurt results? TRUTH = v_e=0.046 (fixed-quadrature,
# already-regenerated radiances); RETRIEVAL assumes v_e=0.10 (fixed-quadrature table).
# Env stays at the TRUTH's v_eff/re_bounds (so oc.signature() still matches the truth
# radiance cache -- the safety gate is never bypassed); RETRIEVAL_OPTICS_CACHE decouples
# only the retrieval's own forward model onto the v_e=0.10 table. Output to a DEDICATED
# out_prefix -- never the campaign _fr_parts dirs -- so mismatch results can't be mistaken
# for self-consistent ones or collide with them.
# Baseline for comparison: runs/_ve046_tik_fqcompare_parts/<idx>_A.json (self-consistent
# v_e=0.046, same truth, same fq table family) -- the delta IS "does mismatch hurt."
#
# 2026-07-13 CORRUPTION FIX: RETRIEVAL_OPTICS_CACHE previously pointed at
# optics_table_10band_nleg1536_re20_fq.npz, which every earlier pilot task SILENTLY
# REBUILT AND OVERWROTE with v_e=0.046 content -- oc.load_optics's build-or-load gate
# computes its expected signature from the env (still at the TRUTH's v_eff=0.046 here by
# design), not from the requested path, so pointing it at a v_e=0.10 file while the env
# said 0.046 caused a silent rebuild-to-0.046. retrieval_worker.py's
# build_forward_and_obs now takes a dedicated retrieval_optics_override path that LOADS
# ONLY (optics_table.load_table, never rebuilds/writes) -- safe by construction. This now
# points at a FRESH, DEDICATED, never-shared file
# (optics_table_10band_nleg1536_re20_fq_mismatch.npz, verified v_eff=0.10/re_max=20.0)
# that no other campaign sbatch references, closing the corruption vector for good.
# All 3 earlier "landed" pilot results (idx 9/36/70) were invalid (self-consistent, not
# mismatched) and were moved to _corrupted_orig/ -- discard, do not compare against them.
#SBATCH --account=crew
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=mismatchPilot
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_fq_logs/mismatch_%A_%a.out

ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
IDX=$SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 MODE_MAP=vmap SOLVER_TOL=1e-4 COST_RTOL=0.01 RADIANCE_TOL=1e-4 ENSEMBLE_NQUAD=48
export OSSE_QUADRATURE=fixed OSSE_RE_GRID_N=181 OSSE_N_RADII=4096
export FR_CONFIGS=A MAX_N_OUTER=2 CURVATURE_LAMBDA=1.0
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
# TRUTH world: v_e=0.046 (env governs oc.signature() -> gates RADIANCE_CACHE)
export OSSE_VEFF=0.046 OSSE_RE_MAX=22
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22_fq.npz
export RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046_fq_sentinel.npz
# RETRIEVAL assumption: v_e=0.10 (decoupled -- does NOT affect the truth/radiance gate)
# dedicated, never-shared file (2026-07-13 corruption fix -- see header note above)
export RETRIEVAL_OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re20_fq_mismatch.npz
OUT=$ROOT/runs/_mismatch_ve046truth_ve100assumed_parts
mkdir -p $OUT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
NVLIB=$($PY -c "import nvidia,os;b=os.path.dirname(nvidia.__file__);print(':'.join(os.path.join(b,d,'lib') for d in sorted(os.listdir(b)) if os.path.isdir(os.path.join(b,d,'lib'))))")
export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH}"
echo "[mismatchPilot] idx=$IDX TRUTH v_e=0.046 / RETRIEVAL assumes v_e=0.10 (DEDICATED out_prefix) wall-start $(date '+%F %T')"
srun $PY scripts/retrieval_worker.py $IDX $OUT/$IDX
echo "[mismatchPilot] idx=$IDX wall-end $(date '+%F %T') rc=$?"
