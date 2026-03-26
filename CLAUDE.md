# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests** (from the `tests/` directory; use the `twostream` conda env):
```bash
cd tests && "C:\Users\dionh\miniconda3\envs\twostream\python.exe" -m pytest . -v
```

**Run a single test file:**
```bash
cd tests && "C:\Users\dionh\miniconda3\envs\twostream\python.exe" -m pytest 1_test.py -v
```

**Run a single test function:**
```bash
cd tests && "C:\Users\dionh\miniconda3\envs\twostream\python.exe" -m pytest 1_test.py::test_1a -v
```

**Regenerate .npz fallback reference files** (run once after changing tau values):
```bash
cd tests && "C:\Users\dionh\miniconda3\envs\twostream\python.exe" supplementary/generate_reference.py
```

Note: Do NOT use `conda run` on Windows — it fails with UnicodeEncodeError (CP1252).

## Architecture

**PythonicDISORT** (the original Discrete Ordinates Solver) is an **external dependency**
installed in the `twostream` conda env. It provides `pydisort()` and `subroutines`.
See https://pythonic-disort.readthedocs.io for its documentation.

The Riccati forward solver code lives in `src/` (3 files — not a package, just modules
added to `sys.path` by `tests/conftest.py`).

### Tests

`tests/` contains the PyTest suite for `pydisort_magnus` (56 tests across 14 files):

| File | What it covers |
|---|---|
| `1_test.py` – `3_test.py` | Constant-ω Riccati vs pydisort reference (isotropic, Rayleigh-like, HG) |
| `4_test.py` | Non-zero diffuse BCs (b_pos, b_neg, purely absorbing) |
| `5_test.py` | Lambertian BDRF surface (scalar, callable, combined with BCs, high albedo) |
| `6_test.py` | τ-varying ω convergence: multi-layer pydisort → Riccati reference |
| `7_test.py` | τ-varying ω and g, including BDRF |
| `8_test.py` | Thick atmospheres + BCs (constant ω, BDRF, b_pos) |
| `9_test.py` | Thick atmospheres + τ-varying properties (convergence) |
| `10_test.py` | Adiabatic cloud profiles (convergence) |
| `11_test.py` | NQuad variation (4, 16) + azimuthal u_ToA_func validation |
| `13_test.py` | Adaptive Riccati solver (thin, cloud, constant-ω) |
| `14_test.py` | Radau Riccati solver standalone (R_up, tol-sweep, T, symmetry, beam source) |
| `15_test.py` | Full-domain Riccati integration (cloud, thin) |

`tests/_helpers.py` provides `make_D_m_funcs`, `make_cloud_profile`, `pydisort_toa`, `pydisort_toa_full_phi`, `get_reference`, `multilayer_pydisort_toa`, `assert_close_to_reference`, `assert_close_to_reference_phi`, `assert_convergence`, and `assert_convergence_and_accuracy`.
`tests/supplementary/generate_reference.py` pre-computes `.npz` fallback files (run once when tau values change).
`tests/supplementary/` also contains star-product diagnostic/exploration scripts.

### Documentation

The primary reference for the mathematics is `docs/Pythonic-DISORT.ipynb`, especially section 3 (derivation of the DISORT algorithm). Labels in the source code (e.g., "see section 3.7.2") refer to this notebook. Online docs are at https://pythonic-disort.readthedocs.io.

### Style notes

- No strict formatter; follow PEP 8 readability
- Variable naming mirrors the mathematical notation from the notebook (e.g., `mu_arr_pos`, `weighted_scaled_Leg_coeffs`)
- Internal functions are prefixed with `_` and are not part of the public API
- Changes to numerical behavior must include a verification test and explanation

### Numerical stability invariant — NO POSITIVE EXPONENTS

**This is a hard design invariant for both `pydisort` and `pydisort_magnus`.**

No intermediate quantity in the solver may contain a factor of the form `exp(+λ · τ)` with
`λ > 0` and `τ > 0`.  Violating this causes catastrophic floating-point overflow for thick
atmospheres.  The original PythonicDISORT enforces this via the Stamnes-Conklin substitution
(growing-mode coefficients are always parametrised from `τ_bot`, making them ≤ 1).  Any new
algorithm (Magnus, doubling/adding, SVD-based propagation, etc.) must satisfy the same
invariant — no positive exponents anywhere in the code.

---

## Riccati forward solver (`pydisort_magnus`)

**Ultimate goal**: retrieve effective radius profile r_e(τ) given a lookup table r_e(τ) → (τ-dependent phase function, τ-dependent ω). The Riccati forward solver is the first building block.

**Purpose**: `pydisort_magnus` is a forward solver for a single atmospheric column with continuously τ-varying single-scattering albedo ω(τ) and phase function D^m(τ), yielding the upward field at ToA (τ=0). Uses the invariant-imbedding Riccati ODE integrated via scipy's adaptive Radau IIA solver (L-stable, order 5).

### Design priority — minimise step count

**Minimising the number of integration steps is the primary optimisation target**, more important
than overall computational complexity or wall time (though the latter matters too).  The forward
solver sits inside an iterative retrieval loop: each call to `pydisort_magnus` is one evaluation
of the forward model, and the retrieval may require hundreds of evaluations.  Reducing the step
count from K=900 to K=200 (even at a slightly higher per-step cost) compounds across the
retrieval and dominates total run time.  When evaluating candidate algorithms (e.g., implicit
Riccati, Rosenbrock–Sylvester, adaptive stepping), prefer the one that achieves a given accuracy
with the fewest steps, not the one with the cheapest individual step.

### Design scope and restrictions

**Primary use case — τ-dependent ω and/or phase function.** This solver exists specifically
for atmospheres where ω(τ) and/or D^m(τ, ·, ·) vary continuously with optical depth.  The
τ-independent case is handled more efficiently and exactly by the original `pydisort` solver;
`pydisort_magnus` will not be optimised for that case.  Constant-ω / constant-phase problems
are retained only as sanity-check test cases (tests 1–5), not as a target use case.

**Bottom boundary condition: unrestricted.**  Arbitrary `b_pos` and BDRF are supported.
The Redheffer star product naturally handles both forward and backward propagation through
the N×N BC system — no separate backward pass is needed for `b_pos`.

### Source files

| File | Purpose |
|---|---|
| `src/pydisort_magnus.py` | Public entry point: validation, quadrature, Fourier loop, output assembly |
| `src/_riccati_solver.py` | Radau IIA Riccati solver: invariant-imbedding R, companion T, beam source s |
| `src/_solve_bc_magnus.py` | `_solve_bc_magnus`: N×N BC system from scattering operators |

**NFourier** = `len(D_m_funcs)`. The m=0 callable handles isotropic/azimuth-symmetric scattering.

**D_m_funcs interface**: `D_m_funcs[m](τ, mu_i, mu_j)` returns the phase-function kernel WITHOUT the ω factor:
`D^m_pure(μ_i, μ_j; τ) = (1/2) Σ_l (2l+1) * poch_l * g_l^m(τ) * P_l^m(μ_i) * P_l^m(μ_j)`.
Handles arbitrary signs of μ_i, μ_j with broadcasting support. ω is handled internally via `omega_func`.

### Numerical stability

The Riccati ODE formulation has all positive signs — no growing exponentials anywhere.
The BC solver is a simple N×N system (half the size of the old 2N×2N Stamnes–Conklin system).

### Riccati solver

Full-domain invariant-imbedding Riccati ODE via scipy's Radau IIA solver (3-stage implicit
RK, L-stable, order 5, adaptive). Two passes: forward sweep for (R_up, T_up, s_up),
backward sweep for (R_down, T_down, s_down). Tolerance controlled via `tol` parameter
(default 1e-3).

Riccati ODE system (vectorized as y = [vec(R), vec(T), s], size 2N² + N):
- dR/dσ = α·R + R·α + R·β·R + β       (N×N, nonlinear Riccati)
- dT/dσ = T·(α + β·R)                  (N×N, linear in T)
- ds/dσ = (α + R·β)·s + R·q₁ + q₂     (N, linear in s, beam source)

Step count is nearly NQuad-independent (~35 steps on τ=30 cloud for NQuad=8/16/32).
NQuad ≥ 6 required: Riccati ARE is ill-conditioned for NQuad=4 (‖R_stab‖ ≈ 10).

### Return value

Always a 5-tuple: `(mu_arr, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`.
`tau_grid` is an ndarray of step boundary points from the forward Riccati sweep.

### Deferred features (not yet implemented — do not forget)

- **Delta-M scaling**: not applied in `pydisort_magnus`
- **Nakajima-Tanaka (NT) corrections**: not applied; may be added in the future
- **Isotropic internal source**: only the collimated beam source is handled
- **Non-ToA depth evaluation**: currently only τ=0 (ToA) is returned
- **A priori τ-grid utility**: cheap grid from profile, usable with SCIATRAN (see memory)
- **JAX + diffrax migration**: port forward solver to JAX + diffrax (Kvaerno5, L-stable ESDIRK) on Linux/GPU to enable discrete adjoint computation via autodiff for the ill-conditioned r_e(τ) retrieval. The current scipy Radau implementation serves as the reference. This migration will be prompted after the project moves to a Linux architecture.
