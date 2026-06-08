# pydisort_riccati_jax

A **differentiable** forward solver for the (1-D) Radiative Transfer Equation in a
plane-parallel atmosphere whose single-scattering albedo $\omega(\tau)$ and phase
function $g_l(\tau)$ vary **continuously** with optical depth. Written in
[JAX](https://docs.jax.dev) with the [diffrax](https://docs.kidger.site/diffrax/)
ODE suite, it is the radiative-transfer link in the effective-radius retrieval chain

$$ r_e(\tau)\;\to\;\text{Mie}\;\to\;\big(\omega(\tau),\,\text{Leg\_coeffs}(\tau)\big)\;\to\;\textbf{pydisort\_riccati\_jax}\;\to\;u^{+}(\tau{=}0,\mu,\phi). $$

The retrieval **observable** is the full azimuthally-resolved upwelling radiance
field at the top of the atmosphere (ToA), $u^{+}(\tau{=}0,\mu,\phi)$ — not just the
flux. Because the whole map is autodifferentiable, retrieval Jacobians come from
`jax.grad`/`jax.jacobian` rather than finite differences.

This solver is a companion to [**PythonicDISORT**](https://pythonic-disort.readthedocs.io/en/latest/)
(the eigendecomposition Discrete Ordinates Solver, an external dependency). It reuses
PythonicDISORT's conventions, quadrature (`subroutines.Gauss_Legendre_quad`), and
input-checking, and is validated against it (`pydisort`) throughout the test suite.

## When to use which solver

| | `PythonicDISORT.pydisort` | `pydisort_riccati_jax` |
|---|---|---|
| Atmosphere | piecewise-constant layers | **continuous** $\omega(\tau)$, $g_l(\tau)$ |
| Method | exact eigendecomposition (Stamnes–Conklin) | invariant-imbedding Riccati ODE (Kvaerno5) |
| Differentiable | via `autograd` (output funcs only) | **yes** — `jax.grad` through the whole solve |
| Output depths | any $\tau$ | ToA ($\tau{=}0$) only |
| Best for | constant-property columns | $\tau$-varying retrieval forward model |

For **constant** $\omega$ / phase function, prefer `pydisort` — it is exact and
faster. `pydisort_riccati_jax` exists specifically for the $\tau$-varying case and is
not optimised for constant properties (those are retained only as sanity-check tests).

## Installation / environment

`pydisort_riccati_jax` is a set of source modules (not yet a package) in `src/`. It
requires `jax`, `diffrax`, `numpy`, `scipy`, and `PythonicDISORT`, installed in the
`JAX` conda environment. Add `src/` (and `tests/` for the shared helpers) to the path:

```python
import sys
from pathlib import Path
repo_root = Path.cwd()                       # repo root
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "tests")) # for _helpers (optional)

from pydisort_riccati_jax import pydisort_riccati_jax, interpolate
```

## Quick start

A $\tau$-varying ("adiabatic") cloud: $\omega$ and $g$ ramp from cloud top
($\tau{=}0$) to base, with a Henyey–Greenstein phase function $g_l(\tau)=g(\tau)^l$.

```python
import jax.numpy as jnp

tau_bot = 10.0                                # cloud optical thickness
omega_func = lambda tau: 0.85 + (0.96 - 0.85) * tau / tau_bot
g_func     = lambda tau: 0.865 + (0.820 - 0.865) * tau / tau_bot

NQuad = 16                                    # quadrature streams (even, >= 6)
NLeg  = NQuad
Leg_coeffs_func = lambda tau: g_func(tau) ** jnp.arange(NLeg)

mu0, I0, phi0 = 0.5, 1.0, 0.0                 # solar geometry / beam intensity

mu_arr_pos, flux_up, u0_ToA, u_ToA_func, tau_grid = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-3,                                 # adaptive Kvaerno5 tolerance (float32-safe)
    BDRF_Fourier_modes=[0.05 / jnp.pi],       # Lambertian surface, albedo 0.05
)

u_ToA_func(phi0)        # (N,) upwelling radiance at ToA, azimuth phi0
flux_up                 # upward diffuse flux at ToA (Python float)
```

`u_ToA_func(phi)` accepts a scalar or 1-D array of azimuths and returns `(N,)` or
`(N, len(phi))`. To evaluate at an arbitrary **observation polar angle** $\mu\in(0,1]$
(e.g. a satellite viewing geometry), interpolate in $\mu$:

```python
u_interp = interpolate(u_ToA_func, mu_arr_pos)
u_interp(0.35, jnp.pi / 4)    # scalar mu, scalar phi -> scalar radiance
```

## Public API

### `pydisort_riccati_jax(tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, NLeg=None, NFourier=None, tol=1e-3, b_pos=0, b_neg=0, BDRF_Fourier_modes=[])`

| Argument | Meaning |
|---|---|
| `tau_bot` | optical depth of the bottom boundary ($>0$) |
| `omega_func` | `tau -> omega` in $[0,1)$ — **JAX-traceable** |
| `Leg_coeffs_func` | `tau -> (NLeg,)` Legendre coefficients $g_l(\tau)$ — **JAX-traceable** |
| `NQuad` | number of quadrature streams (even, $\ge 6$; ARE is ill-conditioned at 4) |
| `mu0`, `I0`, `phi0` | beam cosine $\in(0,1]$, intensity $\ge0$, azimuth $\in[0,2\pi)$ |
| `NLeg`, `NFourier` | Legendre / Fourier term counts (default `NQuad`; `NFourier ≤ NLeg ≤ NQuad`) |
| `tol` | relative tolerance of the adaptive Kvaerno5 step controller |
| `b_pos`, `b_neg` | diffuse boundary sources (bottom-up, top-down); scalar, `(N,)`, or `(N, NFourier)` |
| `BDRF_Fourier_modes` | list of surface bidirectional-reflectance Fourier modes |

**Returns** a 5-tuple, all upwelling-only ($N = $ `NQuad // 2`):

```
mu_arr_pos   (N,)      positive Gauss-Legendre quadrature cosines
flux_up_ToA  JAX scalar  upward diffuse flux at ToA (traceable)
u0_ToA       (N,)      JAX array — zeroth Fourier mode of u+ at ToA
u_ToA_func   callable  phi -> (N,) or (N, len(phi)), JAX-traceable
tau_grid     ndarray   step boundaries [0, ..., tau_bot] from the forward sweep
```

`flux_up_ToA`, `u0_ToA`, and `u_ToA_func` are all JAX arrays / traceable closures:
the autodiff chain from `omega_func`/`Leg_coeffs_func` through the Fourier
reconstruction is unbroken, so `jax.grad` **through the solve** (the retrieval
Jacobian) works — verified against finite differences. `float(flux_up_ToA)` outside a
jax transform if you need a Python number.

### `interpolate(u_ToA_func, mu_arr_pos) -> u_interp(mu, phi)`

Barycentric Lagrange interpolation in $\mu$ at ToA (the analog of
`PythonicDISORT.subroutines.interpolate`, restricted to $\tau{=}0$ and the upwelling
hemisphere). JAX-traceable, so `jax.grad(lambda mu: u_interp(mu, phi))` works
end-to-end. Output shape follows broadcasting of `mu` and `phi`.

## Conventions

Same as PythonicDISORT (see its [comprehensive notebook](https://github.com/LDEO-CREW/Pythonic-DISORT/blob/main/docs/Pythonic-DISORT.ipynb),
section 1):

- **$\tau$** is dimensionless optical depth, increasing downward; the top of the
  atmosphere is $\tau{=}0$ and the surface is $\tau=$ `tau_bot`.
- **$\mu>0$** is the cosine of the polar angle in the **upwelling** hemisphere;
  `mu_arr_pos` are the positive double-Gauss quadrature nodes.
- **Phase function** is given as Legendre coefficients with
  $P(\mu)=\sum_l (2l+1)\,g_l\,P_l(\mu)$, $g_0=1$, $g_1=g$ (the asymmetry parameter).
  For Henyey–Greenstein, $g_l=g^l$.
- **BDRF** is supplied as Fourier modes; a Lambertian surface of albedo $\rho$ is the
  single mode $\rho/\pi$.

Additional to this solver:

- **dtype is float32 by default.** The Riccati state stays $O(1)$ and retrieval
  precision is set by measurement noise ($\sim$10–20 %), not float resolution. Set the
  environment variable `PYDISORT_RICCATI_JAX_X64=1` *before* importing the solver to
  integrate in float64 (used by the stringent `pytest -m float64` partition).
- **Keep `tol` ≈ 1e-3 in float32.** Float32 machine epsilon is $\sim$1.2e-7, and the
  controller uses `atol = tol·1e-3`; a `tol ≲ 1e-4` drives `atol` to/below eps so the
  adaptive step controller cannot converge and raises `max_steps` on thick atmospheres
  ($\tau \gtrsim 20$). `tol=1e-3` reaches $\sim$0.1 % accuracy vs exact DISORT (≈30 steps
  at $\tau{=}30$ — well below the retrieval noise floor) and is the right production
  setting. Tighter references require float64 (`PYDISORT_RICCATI_JAX_X64=1`).
- **Step count is the primary cost.** The solver sits in an iterative retrieval loop;
  `tol` trades accuracy for steps, but in float32 the useful range is narrow
  (1e-3 … 1e-4) — loosen toward 1e-3 in the loop; do not tighten below 1e-4.

### Numerical-stability invariant — no positive exponents

Like PythonicDISORT, the solver contains **no factor of the form
$e^{+\lambda\tau}$ with $\lambda,\tau>0$** anywhere. The invariant-imbedding Riccati
formulation has all-positive signs, so thick atmospheres ($\tau\gg1$) never overflow.
This is a hard design invariant — any change to the numerics must preserve it.

## Working with VOCALS-REx data

VOCALS-REx marine stratocumulus has effective radii of roughly **4–17 µm**, observed
in the MODIS bands **0.645, 1.64, 2.13 µm** (2.13 µm being the most absorbing, where
$\omega<1$ carries the size signal), over optical depths up to $\tau\sim30$ above a
low-albedo ocean ($\rho\approx0.05$).

The phase function and single-scattering albedo come from
[`miejax_lite`](../../miejax_lite/README.md), whose Legendre/$\omega$ output wires
straight into the `*_func(tau)` interfaces. Given an effective-radius profile
$r_e(\tau)$:

```python
from miejax_lite import water_refractive_index, mie_legendre_precompute, mie_avg_legendre

wavelength, v_eff = 2.13, 0.10                       # MODIS band 7, gamma width
m_real, m_imag = water_refractive_index(wavelength)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg)   # once, at setup

r_e_of_tau = lambda tau: 5.0 + 7.0 * (1 - tau / tau_bot) ** (1/3) # adiabatic: r_e grows toward cloud top (tau=0)

omega_func      = lambda tau: mie_avg_legendre(r_e_of_tau(tau), wavelength, v_eff,
                                               precomp, m_real, m_imag)[0]
Leg_coeffs_func = lambda tau: mie_avg_legendre(r_e_of_tau(tau), wavelength, v_eff,
                                               precomp, m_real, m_imag)[1]

mu_arr_pos, flux_up, u0, u_ToA_func, tau_grid = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0,
    tol=1e-3, BDRF_Fourier_modes=[0.05 / jnp.pi],
)
```

Because both `miejax_lite` and the solver are JAX-traceable, the **retrieval
Jacobian** of a ToA radiance with respect to a profile parameter is one autodiff call:

```python
import jax

def toa_radiance(theta):                              # theta parametrises r_e(tau)
    Leg = lambda tau: mie_avg_legendre(r_e_of_tau(tau, theta), wavelength, v_eff,
                                       precomp, m_real, m_imag)[1]
    om  = lambda tau: mie_avg_legendre(r_e_of_tau(tau, theta), wavelength, v_eff,
                                       precomp, m_real, m_imag)[0]
    _, _, _, u_func, _ = pydisort_riccati_jax(
        tau_bot, om, Leg, NQuad, mu0, I0, phi0, tol=1e-3)
    return interpolate(u_func, mu_arr_pos)(mu_obs, phi_obs)

dI_dtheta = jax.grad(toa_radiance)(theta0)
```

Parameters must enter through `omega_func` / `Leg_coeffs_func` (the autodiff path);
`tau_bot`, `mu0`, `NQuad` are validated in NumPy and are not differentiable. See the
worked examples in `docs/riccati_solver_intro.ipynb` and
`docs/riccati_solver_VOCALS_retrieval.ipynb`.

## Scope and deferred features

Not yet implemented (do not assume these are present):

- **Delta-M scaling** and **Nakajima–Tanaka corrections** — not applied.
- **Isotropic internal source** (thermal emission) — only the collimated beam source.
- **Non-ToA evaluation** — only $\tau{=}0$ is returned.
- **Retrieval loop** — the cost function, Gauss–Newton/LM iteration, and $r_e(\tau)$
  parameterisation are not yet implemented.

The **discrete adjoint is *not* a deferred feature**: differentiating through the solve
is free reverse-mode AD via diffrax's default `RecursiveCheckpointAdjoint`, and it is
verified to match finite differences. (`BacksolveAdjoint` etc. exist but the diffrax docs
prefer the checkpointed default; switching is a one-argument change, not new work.)

## Testing

```bash
cd tests && python -m pytest . -v          # full suite (57 tests; up to ~1 hour)
cd tests && python -m pytest 10_key_test.py 13_key_test.py 14_key_test.py -v   # ~10 min subset
```

Tests compare the full azimuthally-resolved `u_ToA_func(phi)` against `pydisort`:
directly for constant-property columns (tests 1–5, 8, 11), and by multi-layer
convergence (50/500 layers, $O(h^2)$) for $\tau$-varying $\omega$ and $g$, including
adiabatic cloud profiles (tests 6, 7, 9, 10). See the repository `CLAUDE.md` for the
full test map.

## References

- PythonicDISORT — https://pythonic-disort.readthedocs.io/en/latest/ (reference solver;
  see its comprehensive notebook section 3 for the DISORT derivation this code mirrors).
- `miejax_lite` — the differentiable Mie front-end providing $r_e(\tau)\to(\omega,g_l)$.
- `report_riccati_solver.pdf` (this repo) — derivation of the invariant-imbedding
  Riccati formulation, forward/backward sweeps, and the $N\times N$ boundary solve.
- Stamnes & Conklin (1984), *A new multi-layer discrete ordinate approach to radiative
  transfer in vertically inhomogeneous atmospheres.*
