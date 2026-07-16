# CLAUDE.md

Guidance for Claude Code working in this repo: the differentiable RT solver
`pydisort_riccati_jax` and the VOCALS r_e(τ) optimal-estimation retrieval built on it.

## Sources of truth (read these first)

- **`docs/user_guide.ipynb`** — how to run everything (executed examples);
  **`docs/technical_documentation.md`** — the math + methodology (successor to the retired
  LaTeX report, which lives on in git history); **`docs/hyperparameter_audit_2026-07.md`**
  — per-knob evidence and flags.
- **`docs/DESIGN_DECISIONS.md`** — settled decisions and *why* (solver lineage, the
  no-positive-exponents invariant, delta-M/TMS, precision policy, priors, IC findings,
  full-retrieval design; §16 = the FR capstone + 2026-07 refactor record).
  **`docs/OUTSTANDING.md`** — open problems (currently §K noise items and §L post-refactor
  actions). Check OUTSTANDING before assuming a feature exists or is missing. Both revised
  2026-07-02; `CHANGELOG.md` is the refactor record + HPC validation brief.
- **`hpc/`** — everything about the HPC production runs: the re-runnable AGENT task specs
  (`AGENT_all125_{rad,ic,fr}.md`), `STRATEGY_hpc_retrieval_runs.md` (compute-minimization
  playbook + the batch-3 lessons), `FINAL_RESULTS_MANIFEST.md` (the canonical batch-3 result
  set + supersession rules), `sbatch/` (Slurm drivers). The L1/L2 checkpoint-resume design
  lives in `scripts/retrieval_worker.py` + `CHANGELOG.md`. The production-scale rigor gates
  are the standardized `tests/hpc/` suite (opt-in: `PYDISORT_HPC_GATES=1 pytest -m hpc`).
- **`docs/riccati_solver_VOCALS_retrieval.ipynb`** — the results notebook (presented figures;
  its cached inputs live in `docs/cached_results/` — §16's are the retrieval-grid IC set:
  `ic_retrieval_grid.json`, `ic_kforce_demo.json`, `kernel_probe_ve046_*.npz`,
  `ic_pump_mechanism_ve046.npz`).

## Layout

```
src/pydisort_riccati_jax/   the package (all importable code)
scripts/                    the 6 worker/analysis entry points the AGENT specs run
tests/                      pytest suite (float32 default / float64 / hpc partitions)
hpc/                        run specs, strategy, sbatch
docs/                       results notebook + its cached inputs, figures, report, design docs
runs/                       (untracked) worker outputs: parts dirs, logs, checkpoints
../data/                    (untracked, workspace-level) large caches: optics table,
                            osse_radiances.npz; VOCALS netCDFs are in
                            ../multispectral-retrieval-using-MODIS/VOCALS_REx_data
```

Package modules (lazy `__init__` — importing the package or `runtime_setup` touches no JAX, so
workers can pin CPU affinity BEFORE JAX loads; solver API names re-export from `solver`):

| Module | Purpose |
|---|---|
| `solver` | one-shot `pydisort_riccati_jax` + the jit-able seam (`riccati_setup`/`riccati_solve`/`eval_radiance`), `lax.scan`-over-modes Fourier solve, `interpolate` |
| `_riccati_solver_jax` | Kvaerno5 Riccati kernels: invariant-imbedding R, companion T, beam source s; delta-M/TMS (`_precompute_tms`/`_apply_tms`) |
| `_solve_bc_riccati_jax` | N×N boundary-condition solve |
| `retrieval_oe` | Gauss–Newton OE retrieval: `RetrievalForward`, priors, QRCP grid + mode selection, `gauss_newton_oe` (+L1 checkpoint), `retrieve_tau_bot`, `posterior_diagnostics` |
| `optics_table` | miepython-grounded r_e → (ω, Legendre) table; differentiable `table_lookup`. (`miejax_lite` is **retired** — legacy validation only.) |
| `info_content` | IC profiling on the full ODE grid (Jacobians → DOFS/SIC via `retrieval_oe`) |
| `noise_model` | OCI-SWIR σ(ρ) measurement noise (2 % calibration-relative) |
| `vocals_io` | VOCALS-REx netCDF profile loader + climatology |
| `osse_config` | **single source of truth** for the observing system (bands, views, NQuad=48, NLEG_ALL=1536, NFOURIER, `signature()`, data-path defaults) |
| `runtime_setup` | HPC per-node core-slot affinity pinning — import + `setup()` before JAX |
| `reference` | PythonicDISORT reference wrappers (tests + notebook convergence checks) |

**PythonicDISORT** is a hard dependency (solver uses its `subroutines`; `pydisort()` is the test
reference). Data/caches are env-overridable (`OPTICS_CACHE`, `RADIANCE_CACHE`, `VOCALS_DATA`)
with `../data` defaults resolved in `osse_config`.

## Commands

```bash
# float32 suite (default; pytest.ini excludes the float64 partition)
cd tests && python -m pytest . -v

# float64 partition (tight tolerances + FD gradient checks; slow)
cd tests && PYDISORT_RICCATI_JAX_X64=1 python -m pytest -m float64 -v

# quick representative subset (~5 min)
cd tests && python -m pytest 13_key_test.py 14_key_test.py -v
```

No suitable local env? Build one: `python3 -m venv /tmp/jaxve && /tmp/jaxve/bin/pip install
numpy scipy jax diffrax pytest PythonicDISORT netCDF4 miepython` (cluster: the `JAX` conda env).

Production runs are driven by the `hpc/AGENT_*.md` specs (Slurm arrays over 125 VOCALS
profiles; `scripts/*.py` entry points; float64, `SOLVER_TOL=1e-4`, NQuad=48).

## Hard invariant — NO POSITIVE EXPONENTS

No intermediate quantity may contain `exp(+λ·τ)` with `λ>0`, `τ>0` (thick-atmosphere overflow).
The Riccati state stays O(1) by construction; any algorithm change must preserve this
(`docs/DESIGN_DECISIONS.md` §2).

## The solver

Forward solver for a single column with continuously τ-varying ω(τ) and phase function gₗ(τ),
returning the upwelling field at ToA. Invariant-imbedding Riccati ODE, diffrax Kvaerno5
(L-stable ESDIRK, adaptive); state PyTree `{'R':(N,N),'T':(N,N),'s':(N,)}`;
`dR/dσ = αR + Rα + RβR + β`, `dT/dσ = (α+Rβ)T`, `ds/dσ = (α+Rβ)s + Rq₁ + q₂`.
Design priority: **minimise integration step count** (the forward runs inside the retrieval
loop; step count is nearly NQuad-independent). Return: 5-tuple
`(mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`, all JAX-traceable — never `float()`
a traced output inside the solver. Delta-M + Nakajima–Tanaka TMS are implemented (opt-in
`delta_M_scaling=True, NT_cor=True`; production ON) — they resolved the negative-radiance issue;
IMS omitted by design. The discrete adjoint is free reverse-mode AD, not a separate feature.

### jit-able seam (DESIGN §7)

```python
setup = riccati_setup(NQuad, I0, phi0, mu0, ...)   # host-side, once; mu0 STATIC
res   = riccati_solve(setup, omega_func, leg_func, tau_bot)   # traceable
obs   = eval_radiance(setup, res, mu_obs, phi_obs)
```
Traced: `tau_bot`, optics closures. Static: grid sizes, geometry, BCs, delta-M/TMS flags —
close `setup` over the jitted fn; rebuild it to change μ0. Fourier modes run under `lax.scan`
(mode body compiles once). `jax.jacfwd` needs `riccati_setup(..., adjoint=diffrax.ForwardMode())`.
Tests: `tests/21_jit_test.py`.

## Precision policy

Solver default float32 (`tol≈1e-3`), opt-in float64 via `PYDISORT_RICCATI_JAX_X64=1` (DESIGN §4).
**Production science (10-band, NQuad=48) requires float64 + `tol=1e-4`** — float32 hits the
Kvaerno5 max_steps blowup there (DESIGN §15); the notebook's 2-band cases run float32 fine.

## Style

PEP 8 readability, no strict formatter. Variable names mirror the math (`mu_arr_pos`,
`weighted_Leg_coeffs`); `_`-prefixed names are internal. Any change to numerical behaviour needs
a verification test and an explanation.
