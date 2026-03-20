# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable mode with test deps):**
```bash
pip install -e ".[pytest]"
```

**Run all tests** (from the `tests/` directory; use the `DISORT` conda env):
```bash
cd tests && conda run -n DISORT python -m pytest .
```

**Run a single test file:**
```bash
cd tests && conda run -n DISORT python -m pytest 1_test.py
```

**Run a single test function:**
```bash
cd tests && conda run -n DISORT python -m pytest 1_test.py::test_1a
```

**Regenerate .npz fallback reference files** (run once after changing tau values in generate_reference.py):
```bash
cd tests && conda run -n DISORT python generate_reference.py
```

**Install notebook/example dependencies:**
```bash
pip install -e ".[notebook_dependencies]"
```

## Architecture

The package lives in `src/PythonicDISORT/` and is a pure-Python reimplementation of Stamnes' FORTRAN DISORT — a Discrete Ordinates Solver for the 1D Radiative Transfer Equation (RTE) in a plane-parallel atmosphere.

### Call chain

The user calls only `pydisort()` (re-exported from `__init__.py`). Internally:

1. **`pydisort.py` → `pydisort()`**: Validates inputs, applies delta-M scaling (controlled by `f_arr`), rescales sources for numerical stability, generates double-Gauss quadrature nodes, and orchestrates the solve. If NT corrections are requested (`NT_cor=True`), it computes TMS and IMS post-hoc intensity corrections here before returning.

2. **`_assemble_intensity_and_fluxes.py` → `_assemble_intensity_and_fluxes()`**: Calls the two solvers below, then assembles closures `flux_up(tau)`, `flux_down(tau)`, `u0(tau)`, and `u(tau, phi)` that are returned to the user.

3. **`_solve_for_gen_and_part_sols.py` → `_solve_for_gen_and_part_sols()`**: Diagonalizes the ODE coefficient matrix for each Fourier mode (eigendecomposition), producing the general solution eigenpairs and particular solution coefficients.

4. **`_solve_for_coeffs.py` → `_solve_for_coeffs()`**: Applies boundary conditions (Dirichlet BCs, BDRF surface reflection) to solve for the unknown coefficients. Uses `scipy.linalg.solve_banded` for multi-layer atmospheres (≥ `use_banded_solver_NLayers` layers) for efficiency.

5. **`subroutines.py`**: Utility functions shared across all modules — quadrature generation, phase function evaluation, affine transforms, actinic flux helpers, and `_compare` (used in tests to compare against Stamnes' DISORT).

### Key parameters

| Parameter | Meaning |
|---|---|
| `tau_arr` | Optical depth of the lower boundary of each layer |
| `omega_arr` | Single-scattering albedo per layer (must be in [0, 1)) |
| `NQuad` | Number of streams (must be even, ≥ 2) |
| `Leg_coeffs_all` | Phase function Legendre coefficients (NLayers × NLeg_all) |
| `NLeg` | Number of Legendre coefficients used in solver (≤ NQuad) |
| `NFourier` | Number of Fourier modes for intensity reconstruction (≤ NLeg) |
| `f_arr` | Fractional forward-scattering peak for delta-M scaling |
| `NT_cor` | Enable Nakajima-Tanaka intensity corrections |
| `BDRF_Fourier_modes` | Surface BDRF as list of Fourier mode functions |

### Output functions

`pydisort()` returns `(mu_arr, Fp, Fm, u0[, u])`:
- `Fp(tau)` — upward diffuse flux
- `Fm(tau)` — tuple of (downward diffuse, downward direct) fluxes
- `u0(tau)` — zeroth Fourier mode of intensity (useful for actinic flux)
- `u(tau, phi)` — full intensity (only when `only_flux=False`)

All output functions accept `is_antiderivative_wrt_tau=True` to switch to their τ-antiderivative (for layer-integrated quantities).

### Tests

`tests/` contains the PyTest suite for `pydisort_magnus` (46 tests across 11 files):

| File | What it covers |
|---|---|
| `1_test.py` – `3_test.py` | Constant-ω Magnus vs pydisort reference (isotropic, Rayleigh-like, HG) |
| `4_test.py` | Non-zero diffuse BCs (b_pos, b_neg, purely absorbing) |
| `5_test.py` | Lambertian BDRF surface (scalar, callable, combined with BCs, high albedo) |
| `6_test.py` | τ-varying ω convergence: multi-layer pydisort → Magnus reference |
| `7_test.py` | τ-varying ω and g, including BDRF |
| `8_test.py` | Thick atmospheres + BCs (constant ω, BDRF, b_pos) |
| `9_test.py` | Thick atmospheres + τ-varying properties (convergence) |
| `10_test.py` | Adiabatic cloud profiles (convergence) |
| `11_test.py` | NQuad variation (4, 16) + azimuthal u_ToA_func validation |

`tests/_helpers.py` provides `make_D_m_funcs`, `make_cloud_profile`, `pydisort_toa`, `pydisort_toa_full_phi`, `get_reference`, `multilayer_pydisort_toa`, `assert_close_to_reference`, `assert_close_to_reference_phi`, and `assert_convergence`.
`tests/generate_reference.py` pre-computes `.npz` fallback files (run once when tau values change).

The old `pydisotest/` (Stamnes DISORT test problems comparing against wrapped FORTRAN DISORT) has been removed.

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

## Magnus forward solver (`pydisort_magnus`)

**Ultimate goal**: retrieve effective radius profile r_e(τ) given a lookup table r_e(τ) → (τ-dependent phase function, τ-dependent ω). The Magnus forward solver is the first building block.

**Purpose**: `pydisort_magnus` is a forward solver for a single atmospheric column with continuously τ-varying single-scattering albedo ω(τ) and phase function D^m(τ), yielding the upward field at ToA (τ=0). It uses first-order Magnus integration (midpoint-rule matrix exponential) rather than per-layer eigendecomposition.

### Design scope and restrictions

**Primary use case — τ-dependent ω and/or phase function.** This solver exists specifically
for atmospheres where ω(τ) and/or D^m(τ, ·, ·) vary continuously with optical depth.  The
τ-independent case is handled more efficiently and exactly by the original `pydisort` solver;
`pydisort_magnus` will not be optimised for that case.  Constant-ω / constant-phase problems
are retained only as sanity-check test cases (tests 1–5), not as a target use case.

**Bottom boundary condition: unrestricted.**  Arbitrary `b_pos` and BDRF are supported.
The Redheffer star product naturally handles both forward and backward propagation through
the N×N BC system — no separate backward pass is needed for `b_pos`.

### New files

| File | Purpose |
|---|---|
| `src/PythonicDISORT/pydisort_magnus.py` | Public entry point: validation, quadrature, Fourier loop, output assembly |
| `src/PythonicDISORT/_magnus_propagator.py` | `_compute_magnus_propagator`: accumulates R/T/s operators via Redheffer star product |
| `src/PythonicDISORT/_solve_bc_magnus.py` | `_solve_bc_magnus`: N×N BC system from star-product scattering operators |

**NFourier** = `len(D_m_funcs)`. The m=0 callable handles isotropic/azimuth-symmetric scattering.

**D_m_funcs interface**: `D_m_funcs[m](τ, mu_i, mu_j)` returns the phase-function kernel WITHOUT the ω factor:
`D^m_pure(μ_i, μ_j; τ) = (1/2) Σ_l (2l+1) * poch_l * g_l^m(τ) * P_l^m(μ_i) * P_l^m(μ_j)`.
Handles arbitrary signs of μ_i, μ_j with broadcasting support. ω is handled internally via `omega_func`.

### Numerical stability — Redheffer star product

The Magnus propagator uses **Redheffer star-product accumulation** of N×N reflection /
transmission / source operators.  All intermediates are O(1) — unconditionally stable for
any τ_bot.  The per-step Magnus expm is unchanged; only how steps are combined changed
(from 2N×2N propagator accumulation to N×N star product).

Accuracy is controlled by `N_magnus_steps`: the first-order Magnus approximation requires
h · λ_max ≲ 1, where h = τ_bot / N_steps and λ_max ≈ 14 for NQuad=8.  Convergence is O(h²).

The BC solver is a simple N×N system (half the size of the old 2N×2N Stamnes–Conklin system).

### Deferred features (not yet implemented — do not forget)

- **Adaptive Magnus step control**: currently equidistant steps (`N_magnus_steps`)
- **Delta-M scaling**: not applied in `pydisort_magnus`
- **Nakajima-Tanaka (NT) corrections**: not applied; may be added in the future
- **Isotropic internal source**: only the collimated beam source is handled
- **Non-ToA depth evaluation**: currently only τ=0 (ToA) is returned
