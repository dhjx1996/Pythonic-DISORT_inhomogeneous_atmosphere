# Contributing

This is a research codebase for `pydisort_riccati_jax`, a differentiable RT solver. Improvements
to correctness, numerical robustness, performance (especially **reducing the adaptive step
count**), docs, and examples are welcome. Contact: dh3065@columbia.edu.

## Getting set up

Python ≥ 3.11 with `numpy`, `scipy`, `jax`, `diffrax`, and **PythonicDISORT** (external
reference for tests). A conda env is the easiest route (the project is developed in a `JAX`
conda env). Optionally `pip install -e .` to expose the `src/` modules; the test suite also adds
them to `sys.path` via `tests/conftest.py`, so installation is not strictly required to run it.

## Running tests

```bash
# float32 production suite (default; auto-excludes the float64 partition)
cd tests && python -m pytest . -v

# float64 partition (tight tolerances + finite-difference gradient checks; slow)
cd tests && PYDISORT_RICCATI_JAX_X64=1 python -m pytest -m float64 -v
```

See `docs/DESIGN_DECISIONS.md` §4 for why the suite is split into float32 / float64 partitions.

## Ground rules

- Keep PRs small and focused; branch from `main`.
- **Any change to numerical behaviour must include a verification test and an explanation of
  why.** Compare the full azimuthally-resolved `u_ToA_func(φ)` against a `pydisort` reference,
  not just the flux or the zeroth Fourier mode.
- Respect the hard invariant: **no positive exponents** anywhere in the solver (see
  `docs/DESIGN_DECISIONS.md` §2). Thick-atmosphere overflow lives here.
- Before assuming a feature exists or is missing, check `docs/OUTSTANDING.md` — it tracks open
  problems and decisions (this is what prevents "is the adjoint implemented?"-type confusion).

## Style

No strict formatter; aim for PEP 8 readability. Variable names mirror the math (e.g.
`mu_arr_pos`, `weighted_Leg_coeffs`). Internal helpers are prefixed `_` and are not public API.
