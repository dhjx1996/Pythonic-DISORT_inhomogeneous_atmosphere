# Delegated task — all-VOCALS **FULL r_e(τ) RETRIEVALS** (the capstone; NQuad=48)

*(Hand-off for a Sonnet agent. The primary server is a **separate filesystem**; **code** moves by git
(sync first), but the **raw results move as a single zip the primary downloads manually** — NOT via git
(see Step 3). Preserve this file as a handoff template — don't delete it. 2026-06-26 **full-retrievals**
run; repurposed from `AGENT_all125_ic.md` — the information-content run is DONE.)*

## What this is

The **capstone**: full joint optimal-estimation retrievals of the effective-radius profile over **every
VOCALS-REx profile** (126 records; 1 non-physical τ≈1585 auto-skipped → 125 valid), at the converged
**NQuad=48**, μ₀=0.9. For each in-situ truth profile the worker retrieves the joint state
`x = [r_e(s_nodes), r_base, τ_bot]` by Gauss–Newton OE, in **two prior configurations** that share ONE
compiled forward:

| config | prior mean `x_a` & first guess `x0` | prior cov `Sa` | result |
|---|---|---|---|
| **A "LOO prior"** | leave-one-flight-out climatology median adiabatic anchor | LOO climatology cov | **HEADLINE** |
| **B "LOO realization"** | one `draw_climatology_realization` (τ_bot **sampled** from climatology) | same LOO cov | robustness |

Truth = the real VOCALS profile in both. The observation is a **NOISELESS OSSE** (`y = F(x_truth)`, no
noise realization added); `Se` (PACE-OCI 2 % calibration-relative, `noise_model.oci_swir`) enters only as
the assumed weighting / posterior covariance. Observing system = the **§15 multi-angle × 10-band
superset** (principal-plane fan, 24 views = NQuad//2, μ₀=0.9).

Three upgrades vs the pre-§15 retrievals, all in `src/retrieval_oe.py` (already on `main` after sync):

- **log-space state** (`state_space='log'`) + a log-space climatology prior (`to_log_prior`) — BP2026 §2.4
  (positivity free; better GN convergence);
- **BP2026 cost-stagnation convergence** (`cost_rtol`, **tuned = 0.01**); the noise-floor stop
  (`chi2_floor`) is implemented but **INACTIVE** (Se magnitude not reliably profiled);
- the **oracle best-fit-adiabatic floor** (`best_fit_adiabatic`) — the RMSE lower bound under the
  adiabatic constraint — computed post-hoc by the worker for monitoring (the headline ΔRMSE).

**One worker** (`tests/supplementary/retrieval_worker.py`), run as **a SINGLE SLURM array** (both configs
A+B per task; no per-mode arrays). Each task writes, for index `i`:

- `<i>_A.npz` / `<i>_B.npz` — the **raw sidecars** (the product): the log-space Jacobian `K`, prior `Sa`,
  posterior `S_hat`/`A`/DOFS/SIC, the retrieved state, the noise σ, `y`/`Fx`, the truth arrays (incl.
  `lwc` + `altitude` for the z-resolved `LWP_truth`), and dense convenience profiles;
- `<i>_A.json` / `<i>_B.json` — slim per-config monitoring scalars;
- `<i>.json` — a combined monitoring record (both configs + grid + runtime).

**No analysis on the cluster** — every metric (RMSE/ΔRMSE, LWP bias, Mahalanobis, posterior IC) is a
post-hoc computation the primary runs on jovyan (`retrieval_analysis.py`) from the raw sidecars.

## Repo & output

- Repo: `/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere`.
  **Sync first** (may be force-pushed): `git fetch origin && git reset --hard origin/main`.
- **Deliverable = ONE zip of the raw sidecars** (Step 3), placed in the workspace root
  `cloud_profile_retrieval/fr_bundle.zip`, **downloaded manually** (NOT via Git — do not commit/push any
  result JSON or npz). It contains `_fr_parts/` (all `<i>_{A,B}.npz` + `<i>_{A,B}.json` + `<i>.json`) and
  the SLURM logs.
- Code (present after sync, **don't modify**): `tests/supplementary/retrieval_worker.py`
  (`ENSEMBLE_NQUAD` default 48, `OPTICS_CACHE`, `COST_RTOL` default 0.01), `src/retrieval_oe.py`,
  `src/optics_table.py`, `src/vocals_io.py`, `src/noise_model.py`.

## Step 0 — build the optics table cache ONCE (shared, ~3–4 min)

The miepython optics table is **profile-independent** and is the **same 10-band table as the IC run**
(identical signature `re=[2,25]/32 veff=0.1 NLeg=128`) — if `optics_table_10band.npz` from the IC run is
still present it is **reused** (signature-checked); otherwise build it once to a shared path so the array
tasks **load** it (don't let 125 tasks race to build it):
```bash
PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
$PY - <<EOF
import sys; sys.path.insert(0,'$ROOT/src')
import optics_table as ot
t = ot.build_or_load_table([0.55,0.67,0.86,1.038,1.24,1.64,2.13,2.26,3.7,4.05],
                           2.0,25.0,32,0.10, cache_path='$OPTICS_CACHE', NLeg=128, n_radii=600)
print("optics cache OK:", t['omega'].shape, "->", '$OPTICS_CACHE')
EOF
```
**Export `OPTICS_CACHE` in every sbatch** (already wired below). `miepython` and `numba` must be in the
env (`$PY -m pip install miepython` if missing).

## Step 1 — JAX (CPU) env

The retrieval runs in **float64** (`export PYDISORT_RICCATI_JAX_X64=1`), exactly as the IC run did.
**This is required, not optional for the 10-band forward:** at float32 the steep dense in-situ truth
forward drives the adaptive Kvaerno5 integrator past `max_steps` on the NQuad=48 forward (the
`overflow…cast` → `maximum number of solver steps` failure) — the notebook's float32 retrievals only ever
used the 2-band bispectral pair, where this does not bite; the 10-band superset needs float64 (which the
IC run proved on all 125 profiles). It also keeps the posterior IC directly comparable to the float64 §14.
One env check:
```bash
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
$PY - <<'EOF'
import sys
sys.path.insert(0,'/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/src')
import jax, numpy, scipy, diffrax, netCDF4, miepython, numba
import retrieval_oe, vocals_io, optics_table, noise_model
print("ENV OK", jax.__version__)   # NB: optics_table (miepython), NOT miejax_lite
EOF
```
If it fails: install into the env with `$PY -m pip install <pkg>` (`miepython`, `numba`, `netCDF4` were
the snags). `miejax_lite` is **not** needed by the production path.

Profile count (expect **126**):
```bash
N=$($PY -c "import sys;sys.path.insert(0,'/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere/src');import vocals_io;print(len(vocals_io.load_all_profiles('$VOCALS_DATA')))")
echo "N=$N profiles"
```

## Step 2 — run it (Venue A: a SINGLE SLURM array)

Per-task ≈ **1.5–3 h** at `--cpus-per-task=1` (both configs A+B reuse the one JAX compile) — the dense
in-situ truth OSSE forward and the float64 grid-selection Jacobian *compiles* dominate (build ≈ 80 min at
NQ=48 float64; the GN retrievals on the smooth ~6-node grid are fast), exactly as in the IC run; thick
profiles run longer. The 8 h `--time` has ample margin. The worker prints `built fwd + selected grid…`
then `… DONE`. A task silent for *hours* is the
thread-oversubscription bug (Troubleshooting). **cpt=1**, **--mem=12G**, **--time=08:00:00**, **%250**
concurrency. Don't change NQuad (48), bands, views, `COST_RTOL`, or worker physics.

```bash
cd /burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
mkdir -p docs/cached_results/_fr_parts
cat > /tmp/fr.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fr_retr
#SBATCH --array=0-$((N-1))%250
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/fr_%a.out
export JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"   # REQUIRED: single-thread XLA (cpt=1) — see Troubleshooting
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} OMP_WAIT_POLICY=passive
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
srun --cpu-bind=cores $PY tests/supplementary/retrieval_worker.py \$SLURM_ARRAY_TASK_ID \
   docs/cached_results/_fr_parts/\$SLURM_ARRAY_TASK_ID
EOF
sbatch /tmp/fr.sbatch
```
(The worker appends `_A`/`_B` to the prefix and writes `<prefix>.json`, so the second arg is the **index
stem** `…/_fr_parts/$SLURM_ARRAY_TASK_ID` — no extension.)

### Venue B — single multi-core server (only if no SLURM)
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_WAIT_POLICY=passive
export VOCALS_DATA=/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data
export OPTICS_CACHE=$ROOT/tests/supplementary/optics_table_10band.npz
export PYDISORT_RICCATI_JAX_X64=1 ENSEMBLE_NQUAD=48 COST_RTOL=0.01
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"   # single-thread XLA (see Troubleshooting)
mkdir -p docs/cached_results/_fr_parts
# simple bounded-parallel loop (4 at a time); raise/lower -P to fit the box's cores & RAM
seq 0 $((N-1)) | xargs -P 4 -I {} $PY tests/supplementary/retrieval_worker.py {} docs/cached_results/_fr_parts/{}
```

## Step 3 — bundle the raw data (NO analysis on the cluster)

**All analysis happens back on the primary (jovyan)** on the raw per-profile sidecars — do **not** merge or
commit any result JSON/npz. Zip **everything raw** (`_fr_parts/` + the SLURM logs) into a **single file in
`cloud_profile_retrieval/`** (the workspace root, one level above this repo) for manual download:
```bash
cd /burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
rm -rf /tmp/fr_bundle && mkdir -p /tmp/fr_bundle/slurm_logs
cp -r docs/cached_results/_fr_parts /tmp/fr_bundle/
cp /tmp/fr_*.out /tmp/fr_bundle/slurm_logs/ 2>/dev/null
( cd /tmp && zip -rq fr_bundle.zip fr_bundle )
mv /tmp/fr_bundle.zip /burg-archive/home/dh3065/cloud_profile_retrieval/   # workspace root
echo "npz in bundle:"; find /tmp/fr_bundle -name '*.npz' | wc -l   # expect ~250 (125 valid × {A,B})
ls -lh /burg-archive/home/dh3065/cloud_profile_retrieval/fr_bundle.zip
```

## Failure handling (fix yourself)

- **Hang on the first forward/Jacobian** → Troubleshooting (thread oversubscription — the main risk).
- **OOM** (exit 137; unlikely at `--mem=12G`): raise `--mem`, or on Venue B lower `-P`.
- **Degenerate profiles** auto-write `{"skipped": ...}` in `<i>.json` (no `_A`/`_B` sidecars) — **expect
  exactly 1** (index 0, RF01, τ≈1585). Any other skip is worth a glance but not a failure.
- **Non-converged retrievals** are NOT failures — the worker flags `converged:false` / `structural_misfit`
  in the sidecar (the primary analyses these); let the task finish and write its sidecar.
- **`optics_table_10band.npz` missing / signature mismatch** → re-run Step 0 (a task will otherwise build
  it itself, wasting ~4 min and risking a write race — pre-build it).
- Do **not** change NQuad (48), bands, views, `COST_RTOL`, the `X64` setting, or the worker physics.

## Troubleshooting: a single forward/Jacobian hangs for hours (thread oversubscription)

**Symptom:** a task sits on the *first* forward/Jacobian for hours — no `DONE` print (e.g. the
2026-06-26 first submission ran 3h45m with **zero** completions).
**Cause:** XLA's Eigen thread pool sizes to the task's CPU **affinity**, not to `OMP_NUM_THREADS`; if the
affinity isn't constrained to 1, the cgroup then throttles ~32 XLA threads onto the 1 allocated CPU →
thrash → indefinite hang. `OMP_NUM_THREADS=1` caps OpenBLAS but does **nothing** to XLA's pool.
**Fix (now baked into the sbatch): `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"`** — forces
single-threaded XLA *regardless of affinity*, which is the correct config for `cpus-per-task=1` anyway. We
formerly relied on `srun --cpu-bind=cores` to constrain affinity (it worked for the IC run), but on the
current **crew1** nodes it does **not** bind — confirmed by the **smoking-gun check (2 s):**
```bash
srun --cpus-per-task=1 --cpu-bind=cores python -c "import os; print('affinity', len(os.sched_getaffinity(0)), 'cpu_count', os.cpu_count())"
```
`affinity ≫ 1` ⇒ binding isn't working ⇒ the `XLA_FLAGS` default above is what saves you (keep
`--cpu-bind=cores` too — harmless, and it helps on nodes where it *does* bind).

## Report back

Results are delivered **as the zip (Step 3), NOT via Git** — do **not** commit or push any JSON/npz. After
the array finishes and the bundle is built, report: (1) records vs `skipped` (expect 1 skip — RF01
τ≈1585); (2) the **`npz in bundle` count** (≈250 = 125×{A,B}) and the bundle path + size; (3) how many
retrievals flagged `converged:false` or `structural_misfit:true` (per config); (4) wall time / venue; (5)
any errors. The primary downloads `cloud_profile_retrieval/fr_bundle.zip` and runs all analysis
(`retrieval_analysis.py`) on jovyan.
