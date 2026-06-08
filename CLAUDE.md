# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The suite is split into **two precision partitions** (see "Test precision partitions" below).

**Default run — float32 production accuracy** (from the `tests/` directory; `JAX` conda env):
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest . -v
```
This runs the 47 float32 tests and auto-excludes the float64 partition (via
`tests/pytest.ini` `addopts = -m "not float64"`).

**float64 partition — stringent convergence/precision** (must set the env var):
```bash
cd tests && PYDISORT_RICCATI_JAX_X64=1 /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest -m float64 -v
```
The 19 float64 tests (convergence-ratio 6/7/9/10 + tight-tolerance standalone
14a/b/c/e + the finite-difference adjoint check 18a) — run on demand, not every
time. Takes ~1 hour (50/500-layer references). Finite-difference gradient checks
belong here because FD roundoff (~eps/h) makes them meaningless in float32.

**Run a single test file / function:**
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest 1_test.py -v
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python -m pytest 1_test.py::test_1a -v
```

**Regenerate .npz fallback reference files** (run once after changing tau values):
```bash
cd tests && /burg/home/dh3065/miniconda3/envs/JAX/bin/python supplementary/generate_reference.py
```

Representative quick subset (float32 default): `13_key_test.py 14_key_test.py`
(adaptive solver + standalone Riccati mechanics; ~5 min).

### Test precision partitions

The solver runs in **float32 by default** (retrieval precision is set by ~10-20%
measurement noise, not float resolution, and float32 keeps the adaptive step count
low). float32 machine epsilon is ~1.2e-7, so the adaptive Kvaerno5 controller cannot
honor a tolerance whose `atol = tol*1e-3` falls at/below it: it shrinks `dt` forever
and raises `max_steps` (an `EquinoxRuntimeError`). Practical consequences:

- **float32 production `tol` ≈ 1e-3** (`atol = 1e-6`, safely above eps). This reaches
  ~1e-3 accuracy vs exact DISORT even at τ=64, in ~30 adaptive steps. `tol ≲ 1e-4`
  is unsafe on thick atmospheres (τ ≳ 20) and `tol ≲ 1e-5` is meaningless in float32.
- **The crashes are controller failures, not instability** — the Riccati state stays
  O(1) (no positive exponents). Set `PYDISORT_RICCATI_JAX_X64=1` to integrate in
  float64, where tight `tol` (1e-8 … 1e-10) is reachable.

Therefore:

| Partition | Marker | dtype | Tests | When |
|---|---|---|---|---|
| Default | (none) | float32 | 1–5, 8, 11, 13, 14d/f, 15, 16, 17, 18 (flux-type) (47) | every run |
| Convergence/precision | `@pytest.mark.float64` | float64 | 6, 7, 9, 10, 14a/b/c/e, 18a (FD adjoint) (19) | on demand |

The float64 partition keeps the original tight tolerances and high convergence ratios
(`tol=1e-8`, `min_ratio=50`) because an O(h²) ratio is only meaningful when the
reference is float-exact. The float32 tests use production tolerances
(`tol=1e-3`, `assert_close_to_reference_phi` default `rel_tol=1e-2`). The float64 opt-in
is read from `PYDISORT_RICCATI_JAX_X64` by **both** `src/pydisort_riccati_jax.py` and
`src/_riccati_solver_jax.py` (so it is import-order-independent); `conftest.py` skips
`float64`-marked tests if x64 is not actually enabled.

## Architecture

**PythonicDISORT** (the original Discrete Ordinates Solver) is an **external dependency**
installed in the `JAX` conda env. It provides `pydisort()` and `subroutines`.
See https://pythonic-disort.readthedocs.io for its documentation.

The Riccati forward solver code lives in `src/` (3 files — not a package, just modules
added to `sys.path` by `tests/conftest.py`).

### Tests

`tests/` contains the PyTest suite for `pydisort_riccati_jax` (57 tests across 14 files):

| File | What it covers |
|---|---|
| `1_test.py` – `3_test.py` | Constant-ω Riccati vs pydisort reference: u(φ) (isotropic, Rayleigh-like, HG) |
| `4_test.py` | Non-zero diffuse BCs: u(φ) (b_pos, b_neg, purely absorbing) |
| `5_test.py` | Lambertian BDRF surface: u(φ) (scalar, callable, combined with BCs, high albedo) |
| `6_test.py` | τ-varying ω convergence: 50/500-layer pydisort u(φ) → Riccati (tol=1e-8), min_ratio=50 |
| `7_test.py` | τ-varying ω and g, including BDRF: u(φ) convergence |
| `8_test.py` | Thick atmospheres + BCs: u(φ) (constant ω, BDRF, b_pos) |
| `9_test.py` | Thick atmospheres + τ-varying properties: u(φ) convergence, 50/500 layers, min_ratio=50 |
| `10_test.py` | Adiabatic cloud profiles: u(φ) convergence, 50/500 layers, min_ratio=50 |
| `11_test.py` | NQuad variation (4, 16): u(φ) |
| `13_test.py` | Adaptive Riccati solver: u(φ) (thin, cloud, constant-ω) |
| `14_test.py` | Kvaerno5 Riccati solver standalone (R_up, tol-sweep, T, symmetry, beam source, T_up tau-varying vs pydisort) |
| `15_test.py` | Full-domain Riccati integration: u(φ) (cloud, thin, reproducibility) |
| `16_interpolate_test.py` | Barycentric μ-interpolation: identity at nodes, cross-check vs pydisort, autodiff |

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
- dT/dσ = (α + R·β)·T                  (N×N, linear in T)
- ds/dσ = (α + R·β)·s + R·q₁ + q₂     (N, linear in s, beam source)

Step count is nearly NQuad-independent (~35 steps on τ=30 cloud for NQuad=8/16/32).
NQuad ≥ 6 required: Riccati ARE is ill-conditioned for NQuad=4 (‖R_stab‖ ≈ 10).

### Return value

Always a 5-tuple: `(mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`.
All intensity outputs are upwelling-only (size N = NQuad // 2):
`mu_arr_pos` is `(N,)`, `u0_ToA` is `(N,)`, `u_ToA_func(φ)` returns `(N,)` or `(N, len(φ))`.
`tau_grid` is an ndarray of step boundary points from the forward Riccati sweep.

`u0_ToA`, `u_ToA_func`, **and `flux_up_ToA`** are JAX arrays / JAX-traceable closures —
the autodiff chain from solver inputs through Fourier reconstruction is unbroken.
`flux_up_ToA` is a JAX scalar (do **not** wrap it in `float()` inside the solver: an
eager `float()` concretizes and breaks `jax.grad` through the whole solve — the
retrieval Jacobian; call `float(flux_up_ToA)` outside any jax transform if you need a
Python number).

### Interpolation to arbitrary observation angles (`interpolate`)

`interpolate(u_ToA_func, mu_arr_pos)` returns `u_interp(mu, phi)` — barycentric Lagrange
interpolation in μ at ToA, enabling evaluation at arbitrary polar angles in (0, 1].
JAX-traceable: `jax.grad(lambda mu: u_interp(mu, phi))` works end-to-end.

Usage:
```python
mu_arr_pos, flux_up, u0, u_ToA_func, tau_grid = pydisort_riccati_jax(...)
u_interp = interpolate(u_ToA_func, mu_arr_pos)
intensity = u_interp(0.35, pi/4)          # scalar mu, scalar phi -> scalar
intensities = u_interp(mu_obs_array, phi) # (M,) mu, scalar phi -> (M,)
```

### Deferred features (not yet implemented — do not forget)

- **Delta-M scaling**: not applied in `pydisort_riccati_jax`
- **Nakajima-Tanaka (NT) corrections**: not applied; may be added in the future
- **Isotropic internal source**: only the collimated beam source is handled
- **Non-ToA depth evaluation**: currently only τ=0 (ToA) is returned
- **A priori τ-grid utility**: cheap grid from profile, usable with SCIATRAN (see memory)
- **Retrieval loop**: the cost function, Gauss–Newton/LM iteration, and r_e(τ) profile
  parameterisation (report §"Toward Retrieval") are not yet implemented.

### Discrete adjoint — works (verified), NOT a separate feature

Differentiating through the solve is **free reverse-mode AD** via diffrax's default
`RecursiveCheckpointAdjoint` (a *discrete* adjoint; the report recommends it over the
hand-derived continuous adjoint of LIDORT/SCIATRAN). No separate adjoint code is needed
or planned. Verified: `jax.grad` of the **upwelling flux output `flux_up_ToA`** through
`pydisort_riccati_jax` w.r.t. an optical property agrees with finite differences to
~2e-10 (`18_adjoint_test.py::test_grad_through_solve_matches_finite_difference`, in the
**float64 partition** — FD is meaningless in float32, where its own roundoff floor is
~1e-5).
Caveats: the reverse pass compiles ~3–4× slower than the forward (amortized under `jit`
in a retrieval loop); the report flags backward-pass conditioning to monitor, but Sandu
(2006) guarantees the discrete adjoint of L-stable Kvaerno5 inherits stiff stability.
**A prerequisite was removing the eager `float(flux_up_ToA)`** which used to concretize
and break all grad-through-solve.

---

## Differentiable Mie front-end (`miejax_lite`)

The lookup table `r_e(τ) → (ω, phase function)` that this solver was built to
consume is provided by **`miejax_lite`**, a standalone JAX package living as a
sibling directory at the workspace root (`../miejax_lite`, not inside this repo).
It is the differentiable link in the retrieval chain
`r_e(τ) → Mie → (ω(τ), Leg_coeffs(τ)) → pydisort_riccati_jax → u_ToA`, so that
retrieval Jacobians come from autodiff.

Install: `pip install -e ../miejax_lite`

### Public API (`from miejax_lite import ...`)

| Function | Returns | Notes |
|---|---|---|
| `water_refractive_index(wavelength)` | `(m_real, m_imag)` | Segelstein (1981) water table, `m_imag < 0` |
| `mie_avg(r_eff, wavelength, v_eff, ...)` | `(omega, g, Q_ext)` | scalar-g (HG) interface |
| `mie_legendre_precompute(max_nstop, NLeg, n_gl=1024)` | `precomp` dict | call once at setup |
| `mie_avg_legendre(r_eff, wavelength, v_eff, precomp, ...)` | `(omega, Leg_coeffs, Q_ext)` | **primary** — `Leg_coeffs` (NLeg,) feed `Leg_coeffs_func(τ)` |

All gamma-distribution-averaged (Hansen-Travis r²-weighting) and differentiable
through `r_eff`, `v_eff`, and `wavelength` (via the size parameter; pass
`m_real`/`m_imag` explicitly for autodiff w.r.t. wavelength). `Leg_coeffs` are
*exact* Mie Legendre coefficients (GL projection of the phase function), not the
HG approximation `g^l`. No `@jax.jit` on public functions — JIT at the caller's
optimal boundary (e.g. the whole retrieval loop), as with this solver.

### Scope / limitations

- **Water droplets only** (upward `D_n` recurrence; valid wherever
  `x·κ < 3.9 − 10.8 n + 13.78 n²`). Verified that 0% of VOCALS-REx profiles
  (r_eff ≤ 17 µm at 0.65/1.64/2.13 µm → x·κ ≤ 0.02) reach the downward branch.
- Dtype follows `jax_enable_x64` (complex64 by default; float64 when enabled).
- Fixed `max_nstop=512` (carry-clamped past the Wiscombe cutoff) covers the
  VOCALS regime with margin. Size parameter `x > 1` (no small-sphere shortcut).
- **Deferred**: downward / Lentz `D_n` recurrence (strongly absorbing / high
  index — not needed for liquid water).

### Tests (`../miejax_lite/tests/`)

`test_miejax_lite.py` — fidelity vs miepython (single-particle over the VOCALS
size range and all retrieval bands, Bohren & Huffman, large/extreme x, exact
Legendre coefficients) and VOCALS distribution-averaged coverage. Default config
is float32. `test_gradients.py` — opt-in float64 finite-difference checks of the
autodiff gradients (`pytest --run-gradients`); finite differences are not
meaningful in float32. Run: `cd ../miejax_lite && python -m pytest tests/`.
