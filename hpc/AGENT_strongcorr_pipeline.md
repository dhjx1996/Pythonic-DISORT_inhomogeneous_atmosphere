# Delegated task — the STRONG-CORRELATION pipeline (4 campaigns, STRICT ORDER)

*(Branch `ve_rerun`. Results move by zip, never git. 2026-07-16. User directive: run the four in
strict order — each must 100 % drain AND have its deliverable bundle ready before the next
starts.)*

## The single lever these campaigns share

`CORR_LENGTH` (new env, `scripts/retrieval_worker.py`): the exp-kernel length ℓ (normalized depth)
in `make_adiabatic_prior`, so prior `corr(top s=0, base s=1) = exp(−1/ℓ)`. Threaded ONLY into the
main-retrieval prior (grid selection uses `diag(S_a)` only, and `retrieve_tau_bot` keeps its own
fixed-tight prior), so the L2 setup (grid/K/τ_bot_pre) is **corr-invariant** and weak↔strong
isolates exactly the prior correlation in the joint GN. Default unset → ℓ=0.5 → corr **0.135**
(byte-identical to before). **Strong = `CORR_LENGTH=5.7355` → corr 0.84**, the empirical VOCALS
in-situ `corr(ln r_top, ln r_base)` (hyperparameter_audit §2.4b; OUTSTANDING §L). Verified: knob
parses, default None reproduces 0.135, strong gives 0.840, τ_bot stays independent; S_a stays PD /
well-conditioned (cond ≤ 2.2e3) at the strong value.

## The four campaigns (STRICT ORDER — do not overlap)

All share: `OSSE_VEFF=0.046 OSSE_RE_MAX=22`, fq optics (`OSSE_QUADRATURE=fixed/181/4096`),
fq truth `osse_radiances_ve046_fq.npz` (sig `4069d4bba73f0bab`, 1..125), config A,
`MAX_N_OUTER=2`, `CURVATURE_LAMBDA=1.0`, float64/tol1e-4/NQuad48, ve_rerun HEAD (Improved GN),
`--array=1-125%60`, `nice=2000`, 11:59 walls, L1+L2 on.

| # | campaign | sbatch | delta vs its baseline | output dir | baseline to compare |
|---|---|---|---|---|---|
| 1 | adiabatic, WEAK corr | `_ve046_tik_adia_wide.sbatch` (**RUNNING**, job 9058600) | FORCE_K_ACTIVE=1, weak corr | `_ve046_tik_adia_parts` | canonical `_ve046_tik_fr_parts` |
| 2 | FR (free-node), STRONG corr | `_ve046_strongcorr_fr_wide.sbatch` | +CORR_LENGTH=5.7355 | `_ve046_strongcorr_fr_parts` | canonical `_ve046_tik_fr_parts` (weak, free-node) |
| 3 | adiabatic, STRONG corr | `_ve046_strongcorr_adia_wide.sbatch` | +CORR_LENGTH=5.7355, FORCE_K_ACTIVE=1 | `_ve046_strongcorr_adia_parts` | campaign 1 (weak adiabatic) |
| 4 | v_e-MISMATCH, STRONG corr | `_ve046_strongcorr_mismatch_wide.sbatch` | +CORR_LENGTH=5.7355, RETRIEVAL_OPTICS_CACHE=v_e0.10 mismatch table | `_ve046_strongcorr_mismatch_parts` | weak-corr mismatch `_mismatch_ve046truth_ve100assumed_parts` |

**Why this order / what each answers:**
1. (running) 2-point adiabatic under the weak prior — the DOF-ablation vs canonical free-node.
2. Does a data-grounded strong top-base prior change the free-node retrieval? (expect: base/LWP
   accuracy up on the ~118 near-adiabatic profiles; the 7 base-decoupled ones biased toward
   adiabatic; upper-cloud data-held detections survive.) Compare to canonical.
3. Strong corr under the adiabatic constraint — expected to track the climatological regression,
   pushing toward the oracle adiabat. Compare to campaign 1 (isolates corr within the 2-point class).
4. The v_e-mismatch previously gave CATASTROPHIC deep-cloud (base) retrievals; test whether tying
   the shielded base to the data-rich top rescues the base under the wrong-v_e forward. Compare to
   the weak-corr mismatch run.

## ⚠️ Campaign-4 precondition — rebuild the mismatch optics table (deleted 2026-07-16)

The retrieval's assumed-v_e table `optics_table_10band_nleg1536_re20_fq_mismatch.npz`
(v_e=0.10, re=[2,20], fixed/181/4096) was **pruned from `../data/` by the workspace consolidation**
mid-session. Campaign 4's worker path is LOAD-ONLY (never rebuilds), so **pre-build it just before
campaign 4** (deterministic, ~4 min):
```bash
OSSE_VEFF=0.10 OSSE_RE_MAX=20 OSSE_QUADRATURE=fixed OSSE_RE_GRID_N=181 OSSE_N_RADII=4096 \
PYDISORT_RICCATI_JAX_X64=1 JAX_PLATFORMS=cpu python -c "import sys;sys.path.insert(0,'src'); \
from pydisort_riccati_jax import osse_config as oc; \
oc.load_optics('$ROOT/../data/optics_table_10band_nleg1536_re20_fq_mismatch.npz')"
```
Verify its stamped signature reads `veff=0.1 re=[2,20]/181 quad=fixed/4096` before launch (the
filename has been an overwrite trap before — never trust it by name; DESIGN/worker docstring).

## Deliverables (one bundle per campaign, at each strict-order handoff)

For campaign C with parts dir `P` and logs `L`:
`zip -rq <name>_bundle.zip P L` → move to workspace root
`cloud_profile_retrieval/`. Bundle names: `ve046_adia_bundle.zip` (1),
`ve046_strongcorr_fr_bundle.zip` (2), `ve046_strongcorr_adia_bundle.zip` (3),
`ve046_strongcorr_mismatch_bundle.zip` (4). A campaign is "done" only when all 125 `_A.npz` exist
(resubmit missing indices `sbatch --array=<list> <sbatch>`; straggler routing per STRATEGY §3) and
its bundle is at the workspace root.

## Provenance

Each sidecar's combined `<i>.json` now stamps `corr_length` (and `force_k_active`) — the one-glance
filter for which campaign produced a result.
