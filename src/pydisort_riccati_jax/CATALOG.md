# CATALOG — src/pydisort_riccati_jax

Details: `CLAUDE.md` (table), `docs/DESIGN_DECISIONS.md` (why).

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
