# CHANGELOG — 2026-07-02 repository refactor

*Audience: primarily the HPC agent (this is your validation brief), secondarily
anyone returning to the repo. One coordinated refactor, commits `1bd56e2` (pre-refactor
checkpoint = the revert point) through this one. Nothing here changes what the
science means; several things change what re-runs compute at/below the noise level —
each such change is called out with the gate that signs it off.*

## 1. What changed

**Layout** (everything moved as git renames):
`src/pydisort_riccati_jax/` is now a proper package (lazy `__init__`; importing it or
`runtime_setup` touches no JAX — the affinity-pin contract). Worker entry points:
`tests/supplementary/*` → **`scripts/`**. HPC specs/strategy/sbatch: repo root →
**`hpc/`**. Run outputs: `docs/cached_results/_*_parts` → untracked **`runs/`**. Large
caches: **`../data/`** (workspace level): `optics_table_10band_nleg1536_re20.npz`,
`osse_radiances.npz`. 41 superseded scripts + 18 stale cached results deleted (git
history keeps everything). Two-level documentation added (`docs/user_guide.md`,
`docs/technical_documentation.md` — the latter replaces `report_riccati_solver.tex`,
retired). Per-knob audit: `docs/hyperparameter_audit_2026-07.md`.

**Numerics-relevant code changes** (the fable-assessment E-list, all landed):

| Change | Effect | Sign-off |
|---|---|---|
| **E1** uniform-K pad on `mode_map='vmap'` + bands×modes batch extended to the pool Jacobian and the mode census | restores the ~2–5×/Jacobian GPU batch FR silently lost; padded modes are sub-noise by construction | golden gate |
| **E2a** `retrieve_tau_bot`: `n_iter` 8→4, `xtol` 5e-3→2e-2 | ~30–50 % of setup removed (the flagged MAJOR INEFFICIENCY); anchor precision unchanged at the σ that matters | golden gate |
| **E3** τ_bot pre-retrieval runs on fixed `S_COARSE`; the initial QRCP grid selection is deleted | ~800 s A100/profile removed; compile reuse improves | golden gate |
| **E5** optics tables are traced jit arguments (not closure constants) | leaner, cache-stable compiles; no numeric change beyond tol-level path noise | golden gate |
| **E6a** fused forward+Jacobian (`has_aux`) at GN init/resume | one forward saved per solve; primal now rides the augmented adaptive solve (tol-level) | golden gate |
| **E4** `FR_SETUP_ONLY=1` worker mode (with `FR_SETUP_CACHE=1`) | CPU "setup farm": build + cache the L2 setup, exit before GN, never write the combined `<i>.json` | smoke: one farmed idx, then a GPU run must print "Layer-2 setup cache HIT" |
| **L2 key → `v2\|…`** | pre-refactor `.setup.npz` caches config-mismatch and recompute (correct, wasted) — **seed the farm only with post-refactor code** | inspect one cache miss log line |
| **L3 removed for FR** | `JAX_COMPILATION_CACHE_DIR` lines stripped from `_fr_*` sbatch; delete `_jax_cache_fr*` data on the cluster; IC keeps `_jax_cache` (ptxas mitigation) | none (measured no-op) |
| stale-jit invalidation fix | `_jac_flux_grid_jit`/`_fwd_jac_jit` are now invalidated on `K_list` changes (flux Jacobians could previously use a stale mode count) | covered by tests/23 |

## 2. ⚠️ HIGHLIGHT — IC mode-selection noise fix (re-run decision needed)

`scripts/ic_worker_profile.py` and `ic_worker_mechanism.py` previously selected the
azimuthal mode count against a **flat `(0.005)²·I` Se**; everything downstream
(DOFS/SIC, weighting) uses the **OCI model** (2 %·ρ + 1e-3 floor). For dark SWIR
scenes the OCI σ sits near the floor — ~5× below 0.005 — so the old trim threshold
could drop modes that are *not* sub-noise for the darkest observations. **Fixed:**
both workers now load the radiance record first and select against
`NOISE.Se(y_measured)` (each worker's own view set: 32-view superset / 24-view fan).

Consequences for a re-run of `AGENT_all125_ic.md`:
- K selection changes (same or MORE modes kept — the accuracy-safe direction);
  combined with the E1 uniform pad, re-run Jacobians/DOFS/SIC will **not** bit-match
  the definitive bundle, with the largest movement possible on the dark
  3.7/4.05 µm bands' contributions.
- The truth-radiance tier never mode-trims → `osse_radiances.npz` (signature
  `d71a8559…`) remains valid; **no `rad` re-run needed**.
- The published notebook figures were built from the definitive bundle (old
  selection). Whether to re-run IC is the **user's + HPC agent's call**; if re-run,
  `scripts/ic_analysis_definitive.py` regenerates both notebook JSONs (including the
  previously ad-hoc `info_content_mechanism.json` — now a scripted step).

## 3. HPC validation checklist (in order)

0. **Migrate the clone** (per the preamble in each `hpc/AGENT_*.md`): `mkdir -p runs
   ../data`; move `optics_table*.npz` and `osse_radiances.npz` to `../data/`; move
   `_*_parts` dirs into `runs/`; `rm -rf docs/cached_results/_jax_cache_fr*`. If an
   FR array is still in flight, finish or checkpoint it BEFORE pulling.
1. **CI-equivalent locally** (fast sanity): `cd tests && python -m pytest . -v`
   — 66 tests; and the float64 partition if time permits.
2. **L2 gate** (per platform): `PYDISORT_HPC_GATES=1 PYDISORT_RICCATI_JAX_X64=1
   python -m pytest tests/hpc -m hpc -k l2 -v` (sbatch: `hpc/sbatch/_fr_l2_test*.sbatch`).
   Note: the pre-refactor GPU gate PASSED 2026-07-02; this re-run validates the
   post-refactor (v2) setup path.
3. **L1 gate**: `... -k l1 -v` (or `hpc/sbatch/_fr_resume_test.sbatch`).
4. **THE golden gate** (signs off E1/E2a/E3/E5/E6a together): run
   `scripts/retrieval_worker.py <idx> runs/_fr_parts/<idx>` fresh for idx ∈
   {20, 47, 49} (the trusted probe set), then `... -k golden -v` with
   `FR_GOLD_DIR=<golden probe bundle>`. PASS bound: max|re_dense diff| < 5e-2 µm
   (expected: ≪, plus CPU/GPU drift). Also compare wall time against the batch-3
   numbers — E1×(E2a+E3) should take a thick profile from ~7 h toward ~2–2.5 h (A100).
5. **E4 smoke**: `FR_SETUP_ONLY=1 FR_SETUP_CACHE=1 scripts/retrieval_worker.py <idx>
   runs/_fr_parts/<idx>` on a CPU slot (thick/mid profile only — thin setups can
   exceed a 12 h wall), then the GPU run of the same idx must log the L2 HIT.
6. **Decide the IC re-run** (§2) with the user; if yes, follow `hpc/AGENT_all125_ic.md`
   (arrays A/B/C) → download → `scripts/ic_analysis_definitive.py`.

Report wall-time deltas and any gate FAIL against `1bd56e2` (the full pre-refactor
state) — `git diff 1bd56e2 -- <file>` shows every change to any file you suspect.
