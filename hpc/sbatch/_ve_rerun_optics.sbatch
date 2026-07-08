#!/bin/bash
# ve_rerun campaign (hpc/AGENT_ve_rerun.md) — Step 1: build the corrected-v_e / raised-r_e-clamp
# optics table ONCE (shared, ~4 min). MAJOR DEVIATIONS vs the spec: OSSE_VEFF=0.046 (min-RMS
# constant, as recommended) AND OSSE_RE_MAX=22 (raised from the published-campaign 20 clamp).
# NLEG_ALL=1536 comfortably covers r_e up to ~25 (osse_config.py note), so 22 needs no other change.
#SBATCH --account=crew
#SBATCH --partition=crew1,ocp_gpu,short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ve046_optics
#SBATCH --output=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/runs/_ve_rerun_logs/optics.out
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
cd $ROOT
export OSSE_VEFF=0.046 OSSE_RE_MAX=22 FR_CONFIGS=A
export OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re22.npz
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
srun $PY - <<'EOF'
import sys, os
sys.path.insert(0, 'src')
from pydisort_riccati_jax import osse_config as oc
print("v_eff =", oc.V_EFF, "| re_bounds =", oc.RE_BOUNDS)
assert oc.V_EFF == 0.046, oc.V_EFF
assert oc.RE_BOUNDS == (2.0, 22.0), oc.RE_BOUNDS
oc.load_optics(os.environ["OPTICS_CACHE"])
_, sig = oc.signature()
print("optics OK ->", os.environ["OPTICS_CACHE"], "| sig", sig)
EOF
