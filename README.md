# pydisort_riccati_jax — differentiable RT solver for inhomogeneous atmospheres

A JAX, fully differentiable forward solver for the 1-D radiative transfer equation in a
plane-parallel atmosphere with **continuously τ-varying** single-scattering albedo ω(τ) and
phase function p(τ; μ, φ). It returns the upwelling radiance field at the top of atmosphere
u⁺(τ=0; μ, φ) and is built to sit inside an iterative retrieval of the cloud effective-radius
profile rₑ(τ).

The solver integrates the **invariant-imbedding matrix Riccati equation** with diffrax's
adaptive **Kvaerno5/4** (L-stable ESDIRK). Differentiating the
forward model is free reverse-mode autodiff — no hand-derived adjoint.

> This project began as a fork of **PythonicDISORT** but is now its own solver. PythonicDISORT
> is used only as an external dependency (for `pydisort()` references and `subroutines`); its
> current home is https://github.com/LDEO-CREW/Pythonic-DISORT.

## The retrieval chain

```
rₑ(τ)  ──optics_table──▶  (ω(τ), p(τ; μ, φ))  ──pydisort_riccati_jax──▶  u⁺(τ=0, μ, φ)
(Mie table, differentiable)      (this solver)             (retrieval observable at ToA)
```

`optics_table` builds a [`miepython`](https://miepython.readthedocs.io/en/latest/)-grounded,
gamma-averaged rₑ → (ω, Q_ext, Legendre) lookup table whose `table_lookup` is differentiable
(the earlier JAX-Mie front-end `miejax_lite` is retired; kept as a sibling package for legacy
validation only).

### When to use which solver

| | `PythonicDISORT.pydisort` | `pydisort_riccati_jax` |
|---|---|---|
| Atmosphere | piecewise-constant layers | **continuous** ω(τ), gₗ(τ) |
| Method | exact eigendecomposition (Stamnes–Conklin) | invariant-imbedding Riccati ODE (Kvaerno5) |
| Differentiable | via `autograd` (output funcs only) | **yes** — `jax.grad` through the whole solve |
| Output depths | any τ | ToA (τ=0) only |
| Best for | constant-property columns | τ-varying retrieval forward model |

For **constant** ω / phase function, prefer `pydisort` — it is exact and faster.

## VOCALS-REx retrieval demo

Effective-radius profiles rₑ(τ) per [VOCALS-REx](https://doi.org/10.5194/acp-11-627-2011)
in-situ observations of marine stratocumulus (C-130 CDP probe), retrieved using multi-band (1.24 / 1.64 / 2.13 µm) multi-angle
satellite radiances with Gauss–Newton optimal estimation and autodiff Jacobians. Grey: in-situ
truth; blue: retrieved ±1σ; dashed orange: adiabatic prior; red dot: (assumed) known cloud base.

<p align="center">
<img src="docs/figures/idealized_retrieval_thin.png" width="380" alt="Thin cloud retrieval (RF11, τ≈1.2)"/>
&nbsp;&nbsp;
<img src="docs/figures/idealized_retrieval_thick.png" width="380" alt="Thick cloud retrieval (RF03, τ≈23)"/>
</p>

**Left:** thin, near-adiabatic cloud (RF11, τ ≈ 1.2).
**Right:** thick, non-adiabatic cloud (RF03, τ ≈ 23).
See [`docs/riccati_solver_VOCALS_retrieval.ipynb`](docs/riccati_solver_VOCALS_retrieval.ipynb).

## Documentation

Two levels, mirroring PythonicDISORT's docs:

- **[User guide](docs/user_guide.md)** — install and run: the forward solver,
  gradients/jit, Mie optics, an OE retrieval, information-content profiling, the
  HPC pipeline, troubleshooting. All code blocks executed.
- **[Comprehensive technical documentation](docs/technical_documentation.md)** —
  for scientists/mathematicians: governing equations, the invariant-imbedding
  Riccati formulation and its stability guarantee, delta-M/TMS, the integrator,
  differentiability, Mie optics, the full OE retrieval + information-content
  methodology, the OSSE rigor discipline, and references.

## Layout

| Path | What |
|---|---|
| `src/pydisort_riccati_jax/` | the package: solver (`solver.py` + kernels) and the retrieval stack (`retrieval_oe`, `optics_table`, `info_content`, `noise_model`, `vocals_io`, `osse_config`, `runtime_setup`, `reference`) |
| `scripts/` | worker/analysis entry points for the all-125 VOCALS OSSE runs |
| `tests/` | PyTest suite (float32 default + a `float64` opt-in partition) |
| `hpc/` | re-runnable HPC task specs, run strategy, rigor gates, Slurm drivers |
| `docs/riccati_solver_VOCALS_retrieval.ipynb` | VOCALS-retrieval notebook (solver tour, validation, and full retrieval) |
| `docs/DESIGN_DECISIONS.md` | **settled** design decisions and their rationale |
| `docs/OUTSTANDING.md` | **open** problems and decisions (read this before assuming a feature exists) |
| `docs/technical_documentation.md` | the comprehensive technical documentation (math + methodology) |

## Install & test

Requires Python ≥ 3.11 with `numpy`, `scipy`, `jax`, `diffrax`, and **PythonicDISORT** (a core
dependency: the solver uses its `subroutines`, and the tests use `pydisort()` as reference).
`pip install -e .` installs the package (`.[retrieval]` adds `netCDF4` + `miepython` for the
VOCALS workers); the test suite also adds `src/` to `sys.path` via `tests/conftest.py`.

```bash
# float32 production suite (default)
cd tests && python -m pytest . -v

# float64 partition (tight tolerances / FD gradient checks; slow)
cd tests && PYDISORT_RICCATI_JAX_X64=1 python -m pytest -m float64 -v
```

## Status

Forward solver and retrieval loop work end-to-end: differentiable Mie optics → Riccati RT →
Gauss–Newton optimal estimation with autodiff Jacobians, validated on VOCALS-REx profiles
(see demo above). Delta-M scaling and Nakajima–Tanaka TMS correction are implemented
(production ON). The repository was reviewed, reorganized, and efficiency-upgraded on
2026-07-02 — `CHANGELOG.md` records what changed and how it is being validated; open items
are tracked in `docs/OUTSTANDING.md` (§K, §L).

Contact: Dion Ho, dh3065@columbia.edu.

License: MIT (see `LICENSE.md`).

Claude Opus 4.6 & 4.8 and Fable 5 have been heavily used in this project.