# CLAUDE.md

Guidance for Claude Code working in this repo: the differentiable RT solver
`pydisort_riccati_jax`.

## Sources of truth (read these first)

- **`docs/DESIGN_DECISIONS.md`** — settled decisions and *why* (solver lineage, the
  no-positive-exponents invariant, retrieval-grid findings, float32/float64, discrete adjoint).
- **`docs/OUTSTANDING.md`** — open problems and decisions still to make. **Check here before
  assuming a feature exists or is missing** (this is what prevents "is the adjoint
  implemented?"-type confusion).
- **`report_riccati_solver.tex`** — the formal report (math + design justification). The old
  PythonicDISORT documentation notebook has been removed; for DISORT background see
  PythonicDISORT's online docs (https://pythonic-disort.readthedocs.io).

## Commands

Run from `tests/`. Use the project's `JAX` conda env (on the cluster:
`/burg/home/dh3065/miniconda3/envs/JAX/bin/python`; elsewhere just `python` in that env).

```bash
# float32 production suite (default; auto-excludes the float64 partition via tests/pytest.ini)
cd tests && python -m pytest . -v

# float64 partition (tight tolerances + FD gradient checks; slow, ~1h)
cd tests && PYDISORT_RICCATI_JAX_X64=1 python -m pytest -m float64 -v

# single file / function
cd tests && python -m pytest 1_test.py::test_1a -v

# regenerate .npz reference fallbacks (run once after changing tau values)
cd tests && python supplementary/generate_reference.py
```

Quick representative subset: `13_key_test.py 14_key_test.py` (~5 min). The solver runs in
**float32 by default** (`tol≈1e-3`); float64 (opt-in via `PYDISORT_RICCATI_JAX_X64=1`) is for
tight tolerances and finite-difference gradient checks. Rationale and the partition table are in
`docs/DESIGN_DECISIONS.md` §4.

## Architecture

**PythonicDISORT** is an **external dependency** (provides `pydisort()` and `subroutines`), used
only as a test reference — see https://github.com/LDEO-CREW/Pythonic-DISORT.

The solver is three flat modules in `src/` (added to `sys.path` by `tests/conftest.py`):

| File | Purpose |
|---|---|
| `src/pydisort_riccati_jax.py` | one-shot `pydisort_riccati_jax` **and** the jit-able composable seam (`riccati_setup`/`riccati_solve`/`calibrate_num_modes`/`eval_radiance`); Fourier loop, output assembly, `interpolate` |
| `src/_riccati_solver_jax.py` | Kvaerno5 Riccati solver: invariant-imbedding R, companion T, beam source s; `_assoc_legendre_neg_mu0_jax` (traced-mu0), `_precompute_tms`/`_apply_tms` |
| `src/_solve_bc_riccati_jax.py` | N×N boundary-condition solve from the scattering operators |

`tests/` holds the PyTest suite (constant-ω sanity checks → τ-varying convergence → thick
atmospheres → adaptive solver → μ-interpolation → adjoint), split into float32 / float64
partitions; `tests/_helpers.py` has the reference/assertion helpers;
`tests/supplementary/generate_reference.py` precomputes `.npz` fallbacks.

### Hard invariant — NO POSITIVE EXPONENTS

No intermediate quantity may contain `exp(+λ·τ)` with `λ>0`, `τ>0` (thick-atmosphere overflow).
The Riccati state stays O(1) by construction; any algorithm change must preserve this. See
`docs/DESIGN_DECISIONS.md` §2.

## The solver

**Purpose.** Forward solver for a single column with continuously τ-varying ω(τ) and phase
function gₗ(τ), returning the upwelling field at ToA (τ=0). Invariant-imbedding Riccati ODE
integrated with diffrax Kvaerno5 (L-stable ESDIRK, adaptive). Two sweeps: forward
(R_up, T_up, s_up), backward (R_down, T_down, s_down); state is a PyTree
`{'R':(N,N),'T':(N,N),'s':(N,)}`.

Riccati system: `dR/dσ = αR + Rα + RβR + β`, `dT/dσ = (α+Rβ)T`, `ds/dσ = (α+Rβ)s + Rq₁ + q₂`.

**Design priority — minimise the integration step count.** The forward model runs inside the
retrieval loop, so step count (more than per-step cost or wall time) dominates total cost. Step
count is nearly NQuad-independent (~35 steps on a τ=30 cloud); NQuad ≥ 6 required.

**Scope.** Built for τ-varying ω and/or phase function (the τ-independent case is handled
better by plain `pydisort`; constant-ω cases are kept only as sanity tests). Arbitrary `b_pos`
and BDRF bottom boundary supported.

**Phase-function interface.** `Leg_coeffs_func(τ) → (NLeg,)` Legendre coefficients, with
explicit `NLeg`/`NFourier`. Legendre products at quadrature points are precomputed with
`scipy.special` (`_precompute_legendre`) and contracted via `jnp.einsum` in the JAX-traceable
vector field.

**Return value.** A 5-tuple `(mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`, all
upwelling-only (size N = NQuad//2). `u0_ToA`, `u_ToA_func`, and `flux_up_ToA` are
JAX-traceable; **do not** wrap `flux_up_ToA` in `float()` inside the solver (it concretises and
breaks `jax.grad`). `interpolate(u_ToA_func, mu_arr_pos)` gives barycentric μ-interpolation to
arbitrary polar angles (JAX-traceable).

**Retrieval observable.** The full azimuthally-resolved `u_ToA_func(φ)` at ToA — tests compare
that against pydisort, not just `u0` or `flux_up`.

### jit-able retrieval forward — the composable seam  (`docs/DESIGN_DECISIONS.md` §7)

The forward model **is jit-able** (OUTSTANDING §C **resolved**) via a host-side setup / traceable
solve split — the one-shot `pydisort_riccati_jax` is the same core, so its 5-tuple is unchanged
(bit-for-bit). Recipe:

```python
setup = riccati_setup(NQuad, I0, phi0, ...)               # host-side, run once
K     = calibrate_num_modes(setup, of, lf, tau_bot, mu0,  # exact DISORT Cauchy stop (p.89)
                            mu_obs, phi_obs)               # -> static int K <= NFourier
def forward(theta, tau_bot, mu0):
    of, lf = optics_from(theta)
    res = riccati_solve(setup, of, lf, tau_bot, mu0, num_modes=K)   # traceable
    return eval_radiance(setup, res, mu_obs, phi_obs)               # observable
f = jax.jit(forward)                 # compile once (K mode-blocks), cached
g = jax.jit(jax.grad(forward))       # reverse-mode discrete adjoint (default)
```

- **Traced:** `tau_bot`, `mu0`, optics closures. **Static (in `setup`):** grid sizes, `I0`, `phi0`,
  BCs, BDRF, `delta_M`/`NT_cor` flags. Close `setup` over the jitted fn; don't pass it as a traced arg.
- **Forward-mode** (`jax.jacfwd`, small-DOF retrieval) needs `riccati_setup(..., adjoint=diffrax.ForwardMode())`
  — the reverse-mode default is a `custom_vjp` that can't be forward-differentiated.
- **`tol_azim`** sets the Cauchy ε (default 1e-3; `0` ⇒ all NFourier modes, which the one-shot entry uses).
- Demo: `tests/supplementary/demo_jit_retrieval.py`. Tests: `tests/21_jit_test.py`.

### Open / deferred

**Delta-M scaling and the Nakajima–Tanaka TMS correction are implemented** (opt-in
`delta_M_scaling=True, NT_cor=True`; see `docs/DESIGN_DECISIONS.md` §6) — this resolved the
**negative-radiance** issue for forward-peaked phase functions (`docs/OUTSTANDING.md` §A). IMS is
omitted by design (it corrects only the downward field). **jit-ability is resolved** (the
composable seam above; `docs/OUTSTANDING.md` §C → `docs/DESIGN_DECISIONS.md` §7). Still **not yet
implemented**: isotropic internal source, non-ToA depth, the retrieval loop itself, and the τ-grid
utility — all tracked in `docs/OUTSTANDING.md`. The discrete adjoint is **not** a separate feature
— it is free reverse-mode AD (verified); see `docs/DESIGN_DECISIONS.md` §5.

## Differentiable Mie front-end (`miejax_lite`)

The lookup `rₑ(τ) → (ω, phase function)` is supplied by **`miejax_lite`**, a sibling package
(`../miejax_lite`, `pip install -e ../miejax_lite`). It closes the differentiable chain
`rₑ(τ) → Mie → (ω, Leg_coeffs) → pydisort_riccati_jax → u_ToA`. Primary API:
`mie_avg_legendre(r_eff, wavelength, v_eff, precomp, ...) → (ω, Leg_coeffs, Q_ext)` (gamma-
averaged, differentiable, exact Mie Legendre coefficients). Water droplets only. See
`../miejax_lite/README.md`.

## Style

PEP 8 readability, no strict formatter. Variable names mirror the math (`mu_arr_pos`,
`weighted_Leg_coeffs`); `_`-prefixed functions are internal. Any change to numerical behaviour
needs a verification test and an explanation.
