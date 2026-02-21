# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable mode with test deps):**
```bash
pip install -e ".[pytest]"
```

**Run all tests** (from the `pydisotest/` directory):
```bash
cd pydisotest && pytest
```

**Run a single test file:**
```bash
cd pydisotest && pytest 1_test.py
```

**Run a single test function:**
```bash
cd pydisotest && pytest 1_test.py::test_1a
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

`pydisotest/` contains `N_test.py` files (N = 1–9, 11, I) corresponding to Stamnes' DISORT test problems, plus matching `N_test.ipynb` notebooks that show physical setup and plots. Tests use `subroutines._compare()` to verify PythonicDISORT results against Stamnes' wrapped FORTRAN DISORT. Section 6 of `docs/Pythonic-DISORT.ipynb` requires an F2PY-wrapped DISORT to be installed locally.

### Documentation

The primary reference for the mathematics is `docs/Pythonic-DISORT.ipynb`, especially section 3 (derivation of the DISORT algorithm). Labels in the source code (e.g., "see section 3.7.2") refer to this notebook. Online docs are at https://pythonic-disort.readthedocs.io.

### Style notes

- No strict formatter; follow PEP 8 readability
- Variable naming mirrors the mathematical notation from the notebook (e.g., `mu_arr_pos`, `weighted_scaled_Leg_coeffs`)
- Internal functions are prefixed with `_` and are not part of the public API
- Changes to numerical behavior must include a verification test and explanation
