"""pydisort_riccati_jax — differentiable invariant-imbedding Riccati RT solver
plus the VOCALS r_e(τ) retrieval stack built on it.

All submodules load lazily: importing this package touches neither JAX nor SciPy,
so ``runtime_setup`` can pin CPU affinity BEFORE JAX spins up its thread pools.

  solver                 forward solver: one-shot ``pydisort_riccati_jax`` and the
                         jit-able seam ``riccati_setup``/``riccati_solve``/``eval_radiance``
  _riccati_solver_jax    Kvaerno5 Riccati kernels (R/T/s sweeps, delta-M + TMS)
  _solve_bc_riccati_jax  boundary-condition solve
  retrieval_oe           Gauss–Newton optimal-estimation retrieval (priors, DOFS/SIC, grids)
  optics_table           miepython-grounded r_e → (ω, Q_ext, Legendre) lookup table
  info_content           information-content profiling on the full ODE grid
  noise_model            ToA measurement-noise models (OCI-SWIR)
  vocals_io              VOCALS-REx in-situ profile loader
  osse_config            canonical VOCALS OSSE observing system (single source of truth)
  runtime_setup          HPC affinity pinning — import and call ``setup()`` before JAX
  reference              PythonicDISORT reference wrappers (validation / tests)

Solver API names resolve lazily from ``solver``, so the historical
``from pydisort_riccati_jax import pydisort_riccati_jax, riccati_setup, ...``
continues to work unchanged.
"""
import importlib

_SUBMODULES = {
    "solver", "_riccati_solver_jax", "_solve_bc_riccati_jax", "retrieval_oe",
    "optics_table", "info_content", "noise_model", "vocals_io", "osse_config",
    "runtime_setup", "reference",
}


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    solver = importlib.import_module(".solver", __name__)
    try:
        return getattr(solver, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(_SUBMODULES | set(globals()))
