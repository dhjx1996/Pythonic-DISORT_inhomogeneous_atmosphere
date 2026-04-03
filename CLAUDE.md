# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests** (from the `tests/` directory; use the `JAX` conda env on Linux):
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest . -v
```

**Run a single test file:**
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest 1_test.py -v
```

**Run a single test function:**
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest 1_test.py::test_1a -v
```

**Regenerate .npz fallback reference files** (run once after changing tau values):
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python supplementary/generate_reference.py
```

Note: Full test suite takes ~28 min due to per-test JIT recompilation.
Representative subset for quick verification: `10_test.py 13_test.py 14_test.py`.

## Architecture

**PythonicDISORT** (the original Discrete Ordinates Solver) is an **external dependency**
installed in the `JAX` conda env. It provides `pydisort()` and `subroutines`.
See https://pythonic-disort.readthedocs.io for its documentation.

The Riccati forward solver code lives in `src/` (3 files — not a package, just modules
added to `sys.path` by `tests/conftest.py`).

### Tests

`tests/` contains the PyTest suite for `pydisort_riccati_jax` (56 tests across 14 files):

| File | What it covers |
|---|---|
| `1_test.py` – `3_test.py` | Constant-ω Riccati vs pydisort reference: u(φ) (isotropic, Rayleigh-like, HG) |
| `4_test.py` | Non-zero diffuse BCs: u(φ) (b_pos, b_neg, purely absorbing) |
| `5_test.py` | Lambertian BDRF surface: u(φ) (scalar, callable, combined with BCs, high albedo) |
| `6_test.py` | τ-varying ω convergence: multi-layer pydisort u(φ) → Riccati reference |
| `7_test.py` | τ-varying ω and g, including BDRF: u(φ) convergence |
| `8_test.py` | Thick atmospheres + BCs: u(φ) (constant ω, BDRF, b_pos) |
| `9_test.py` | Thick atmospheres + τ-varying properties: u(φ) convergence |
| `10_test.py` | Adiabatic cloud profiles: u(φ) convergence |
| `11_test.py` | NQuad variation (4, 16): u(φ) |
| `13_test.py` | Adaptive Riccati solver: u(φ) (thin, cloud, constant-ω) |
| `14_test.py` | Kvaerno5 Riccati solver standalone (R_up, tol-sweep, T, symmetry, beam source) |
| `15_test.py` | Full-domain Riccati integration: u(φ) (cloud, thin, reproducibility) |

`tests/_helpers.py` provides `get_reference`, `pydisort_toa_full_phi`, `multilayer_pydisort_toa_full_phi`, `make_cloud_profile`, `assert_close_to_reference_phi`, `assert_convergence_phi`, and `PHI_VALUES`.
`tests/supplementary/generate_reference.py` pre-computes `.npz` fallback files (run once when tau values change).

### Documentation

The primary reference for the mathematics is `docs/Pythonic-DISORT.ipynb`, especially section 3 (derivation of the DISORT algorithm). Labels in the source code (e.g., "see section 3.7.2") refer to this notebook. Online docs are at https://pythonic-disort.readthedocs.io.

### Style notes

- No strict formatter; follow PEP 8 readability
- Variable naming mirrors the mathematical notation from the notebook (e.g., `mu_arr_pos`, `weighted_scaled_Leg_coeffs`)
- Internal functions are prefixed with `_` and are not part of the public API
- Changes to numerical behavior must include a verification test and explanation

### Numerical stability invariant — NO POSITIVE EXPONENTS

**This is a hard design invariant for both `pydisort` and `pydisort_riccati_jax`.**

No intermediate quantity in the solver may contain a factor of the form `exp(+λ · τ)` with
`λ > 0` and `τ > 0`.  Violating this causes catastrophic floating-point overflow for thick
atmospheres.  The original PythonicDISORT enforces this via the Stamnes-Conklin substitution
(growing-mode coefficients are always parametrised from `τ_bot`, making them ≤ 1).  Any new
algorithm (Magnus, doubling/adding, SVD-based propagation, etc.) must satisfy the same
invariant — no positive exponents anywhere in the code.

---

## Riccati forward solver (`pydisort_riccati_jax`)

**Ultimate goal**: retrieve effective radius profile r_e(τ) given a lookup table r_e(τ) → (τ-dependent phase function, τ-dependent ω). The Riccati forward solver is the first building block.

**Retrieval observable**: the full upwelling radiance field u⁺(τ=0, μ, φ) at ToA — not just the flux.  Tests must compare the full azimuthally-resolved `u_ToA_func(φ)` against pydisort, not only the zeroth Fourier mode `u0` or scalar `flux_up`.

**Purpose**: `pydisort_riccati_jax` is a forward solver for a single atmospheric column with continuously τ-varying single-scattering albedo ω(τ) and phase function g_l(τ), yielding the upward field at ToA (τ=0). Uses the invariant-imbedding Riccati ODE integrated via diffrax's Kvaerno5 solver (L-stable ESDIRK, order 5, adaptive PIDController step-size).

### Design priority — minimise step count

**Minimising the number of integration steps is the primary optimisation target**, more important
than overall computational complexity or wall time (though the latter matters too).  The forward
solver sits inside an iterative retrieval loop: each call to `pydisort_riccati_jax` is one evaluation
of the forward model, and the retrieval may require hundreds of evaluations.  Reducing the step
count from K=900 to K=200 (even at a slightly higher per-step cost) compounds across the
retrieval and dominates total run time.

### Design scope and restrictions

**Primary use case — τ-dependent ω and/or phase function.** This solver exists specifically
for atmospheres where ω(τ) and/or g_l(τ) vary continuously with optical depth.  The
τ-independent case is handled more efficiently and exactly by the original `pydisort` solver;
`pydisort_riccati_jax` will not be optimised for that case.  Constant-ω / constant-phase problems
are retained only as sanity-check test cases (tests 1–5), not as a target use case.

**Bottom boundary condition: unrestricted.**  Arbitrary `b_pos` and BDRF are supported.
The Redheffer star product naturally handles both forward and backward propagation through
the N×N BC system — no separate backward pass is needed for `b_pos`.

### Source files

| File | Purpose |
|---|---|
| `src/pydisort_riccati_jax.py` | Public entry point: validation, quadrature, Fourier loop, output assembly |
| `src/_riccati_solver_jax.py` | Kvaerno5 Riccati solver: invariant-imbedding R, companion T, beam source s |
| `src/_solve_bc_riccati_jax.py` | `_solve_bc_riccati_jax`: N×N BC system from scattering operators |

### Phase-function interface

The phase function is specified via `Leg_coeffs_func(τ) → (NLeg,)` returning Legendre coefficients,
plus explicit `NLeg` and `NFourier` parameters. Legendre polynomial products at quadrature
points are pre-computed at setup time using `scipy.special` (via `_precompute_legendre`),
then contracted via `jnp.einsum` in the JAX-traceable ODE vector field. This replaces the
old `D_m_funcs` callable interface and makes the entire integration autodiff-compatible.

### Numerical stability

The Riccati ODE formulation has all positive signs — no growing exponentials anywhere.
The BC solver is a simple N×N system (half the size of the old 2N×2N Stamnes–Conklin system).

### Riccati solver

Full-domain invariant-imbedding Riccati ODE via diffrax Kvaerno5 (5-stage ESDIRK,
L-stable, order 5, adaptive PIDController). Two passes: forward sweep for (R_up, T_up, s_up),
backward sweep for (R_down, T_down, s_down). Tolerance controlled via `tol` parameter
(default 1e-3).

State is a PyTree `{'R': (N,N), 'T': (N,N), 's': (N,)}` — no flattening needed.

Riccati ODE system:
- dR/dσ = α·R + R·α + R·β·R + β       (N×N, nonlinear Riccati)
- dT/dσ = T·(α + β·R)                  (N×N, linear in T)
- ds/dσ = (α + R·β)·s + R·q₁ + q₂     (N, linear in s, beam source)

Step count is nearly NQuad-independent (~35 steps on τ=30 cloud for NQuad=8/16/32).
NQuad ≥ 6 required: Riccati ARE is ill-conditioned for NQuad=4 (‖R_stab‖ ≈ 10).

### Return value

Always a 5-tuple: `(mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`.
All intensity outputs are upwelling-only (size N = NQuad // 2):
`mu_arr_pos` is `(N,)`, `u0_ToA` is `(N,)`, `u_ToA_func(φ)` returns `(N,)` or `(N, len(φ))`.
`tau_grid` is an ndarray of step boundary points from the forward Riccati sweep.

### Deferred features (not yet implemented — do not forget)

- **Delta-M scaling**: not applied in `pydisort_riccati_jax`
- **Nakajima-Tanaka (NT) corrections**: not applied; may be added in the future
- **Isotropic internal source**: only the collimated beam source is handled
- **Non-ToA depth evaluation**: currently only τ=0 (ToA) is returned
- **A priori τ-grid utility**: cheap grid from profile, usable with SCIATRAN (see memory)
- **Discrete adjoint**: diffrax supports `RecursiveCheckpointAdjoint`, `BacksolveAdjoint`, `ImplicitAdjoint`, and `DirectAdjoint` for differentiating through the ODE solve — needed for the r_e(τ) retrieval
