# CATALOG — src/pydisort_riccati_jax

Details: `CLAUDE.md` (table), `docs/DESIGN_DECISIONS.md` (why).

## Scripts

| File | What it is |
|---|---|
| `__init__.py` | Lazy package root: importing it touches no JAX/SciPy (so `runtime_setup` can pin CPU affinity first) and re-exports the solver API names from `solver`. |
| `solver.py` | The forward solver's public face: one-shot `pydisort_riccati_jax`, the jit-able seam (`riccati_setup`/`riccati_solve`/`eval_radiance`), the `lax.scan`/`vmap` Fourier-mode solve, and barycentric `interpolate`. |
| `_riccati_solver_jax.py` | The numerical core: Kvaerno5 invariant-imbedding Riccati kernels (R/T/s sweeps), delta-M scaling and the Nakajima–Tanaka TMS correction, and the gradient-frozen PID step controller. |
| `_solve_bc_riccati_jax.py` | The N×N boundary-condition solve that closes the Riccati sweep at the surface. |
| `retrieval_oe.py` | The Gauss–Newton optimal-estimation retrieval: `RetrievalForward`, the prior builders, QRCP retrieval-grid + Fourier-mode selection, `gauss_newton_oe` with L1 checkpointing, `retrieve_tau_bot`, and the Rodgers posterior diagnostics. |
| `optics_table.py` | miepython-grounded r_e → (Q_ext, ω, Legendre) lookup table (gamma-DSD averaged) with the differentiable `table_lookup` the retrieval autodiffs through. |
| `info_content.py` | Information-content profiling on the full ODE grid: Jacobians → DOFS/SIC/averaging-kernel via the single Rodgers implementation in `retrieval_oe`. |
| `noise_model.py` | OCI-SWIR σ(ρ) measurement-noise model (2 % calibration-relative + floor) supplying Se and optional noise realizations. |
| `vocals_io.py` | VOCALS-REx netCDF flight loader: in-cloud profile extraction, climatology, and LWP/v_eff convenience diagnostics. |
| `osse_config.py` | Single source of truth for the observing system (bands, views, NQuad, NFourier, data paths) with the `signature()` hash that gates every cache against a stale forward model. |
| `runtime_setup.py` | HPC per-node core-slot affinity pinning — import and call `setup()` BEFORE JAX loads its thread pools. |
| `reference.py` | PythonicDISORT reference wrappers (single- and multi-layer) that the test suite and notebook convergence checks compare the Riccati solver against. |

## The priors (`retrieval_oe.py` §3, lines 822–1053)

There is really **one** prior — an adiabatic mean with a Bayesian-Tikhonov (exponentially
correlated) covariance. Everything else is a wrapper that supplies its numbers or transforms
its output. Read this family bottom-up; each row calls the one above it.

| Function | Role | Calls | Used operationally? |
|---|---|---|---|
| `make_adiabatic_prior` | **The primitive.** Mean = adiabatic r_e⁵-linear over the r_e nodes| — | via the composer |
| `make_joint_prior` | **The composer.** Runs the primitive in *normalized depth* (unit τ_bot), folds `r_base` in as the deepest node (s=1) of the correlated block, and appends `τ_bot` as an independent broad scalar (block-diagonal). | `make_adiabatic_prior` | via the two named priors |
| `make_climatology_prior` | **Option 1 — the operational prior.** Leave-one-flight-out VOCALS-REx. Means/σ's come from the `clim` dict (robust **median/MAD**, despite `*_mean`/`*_std` key names — see `vocals_io.vocals_climatology`). | `make_joint_prior` | **YES — this produced the results** |
| `make_marine_sc_prior` | **Option 2 — generic fallback.** Literature/data-grounded marine-Sc; this is the *only* place `r_base_ratio=0.65` lives. | `make_joint_prior` | **No** — never called in `scripts/`/`hpc/` |
| `draw_climatology_realization` | **Not a Gaussian prior.** Draws a *physical* 3-param adiabatic profile from the `clim` marginals (rejection-sampled `r_top>r_base`, `τ_bot>0`) — for synthetic truths / full-retrieval config-B. Use this, not a covariance draw of `Sa` (which is unphysically non-monotonic). | — | YES (config-B truths) |
| `to_log_prior` | **A transformer, not a prior.** Delta-method map `(x_a, Sa) → (ln x_a, D·Sa·Dᵀ)`, `D=diag(1/x_a)`, for a `state_space='log'` forward. Invoked via `log=True`. | — | YES (`log=True` in prod) |

### Default prior arguments

Every default below is a *fallback for interactive/generic use*. In the operational
climatology path the σ's are replaced by VOCALS statistics (from `clim`) and `ℓ` is passed
explicitly by the worker.

| Default | Value(s) | Provenance | Operational fate |
|---|---|---|---|
| `sigma_top` | **2.5** (all three builders) | VOCALS cloud-top MAD ≈2.3 µm; value barely matters (r_top observable, A_top≈1). Harmonized 2026-07-20 (was 3.0/5.0/2.5) — an inert change: every live caller passes σ explicitly. | **Overridden** by `clim["r_top_std"]` |
| `sigma_base` | **1.5** (all three) | VOCALS r_base robust-core MAD ≈1.4 µm; kept *tight* because the base is radiatively shielded (A_base≈0.06, ~80 % prior-dominated). Harmonized 2026-07-20 (was 1.5/2.0). | **Overridden** by `clim["r_base_std"]` |
| `r_base_ratio` | 0.65 | adiabatic ratio; VOCALS median 0.60, King/Vukićević AMT-2025 ≈0.70 → split. Clipped to `r_top−0.5`. | **Unused** (marine-only) |
| `sigma_tau_bot` | None→0.5·τ_bot (joint) / 1.0·τ_bot (marine) | deliberately uninformative — τ_bot is data-determined (A≈1) | **Overridden** by `clim["tau_bot_std"]` |
| `corr_length` | None→τ_bot/2 (adiab) / 0.5 normalized (joint) | Bayesian-Tikhonov smoothness length | **Overridden** — worker passes `CORR_LENGTH` (e.g. 5.7355 → corr 0.84, VOCALS in-situ) |
| `strength` | 1.0 | single global σ-scale knob (the one genuine free lever) | kept |
| σ floors | 0.5/0.5/1.0 µm/τ | keep `Sa` usefully SPD | kept |
