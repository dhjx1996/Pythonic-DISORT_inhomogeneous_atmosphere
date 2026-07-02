# User guide — `pydisort_riccati_jax`

*The practical, code-first manual: install the package, run the forward solver, take
gradients, build Mie optics, run a retrieval, profile information content, and drive
the HPC production pipeline. Every code block below has been executed verbatim
(2026-07-02, CPU, float32 unless stated). For the math and design rationale, read the
companion [technical documentation](technical_documentation.md).*

---

## 1. What this is

A JAX, fully differentiable forward solver for the 1-D radiative transfer equation in
a plane-parallel atmosphere whose single-scattering albedo ω(τ) and phase function
p(τ; μ, φ) vary **continuously** with optical depth — plus the Gauss–Newton
optimal-estimation (OE) retrieval stack built on it, used to retrieve cloud
effective-radius profiles rₑ(τ) from multi-band, multi-angle top-of-atmosphere (ToA)
radiances:

```
rₑ(τ) ──optics_table──▶ (ω(τ), p(τ;μ,φ)) ──pydisort_riccati_jax──▶ u⁺(τ=0; μ, φ)
        (Mie table,          (the solver)        (the observable; differentiable
         differentiable)                          end-to-end w.r.t. rₑ, τ_bot, …)
```

Use plain [PythonicDISORT](https://pythonic-disort.readthedocs.io) instead when your
optical properties are piecewise-constant — it is exact and faster there. This solver
exists for the τ-varying retrieval forward-model case.

## 2. Install

Python ≥ 3.11. From the repo root:

```bash
pip install -e .                 # solver + retrieval stack
pip install -e ".[retrieval]"   # + netCDF4 (VOCALS I/O) and miepython (table builds)
pip install -e ".[test]"        # + pytest
```

Core dependencies: `numpy`, `scipy`, `jax`, `diffrax`, `PythonicDISORT` (the solver
imports its `subroutines`; it is not just a test reference). No install needed to
hack on the source — `tests/conftest.py` and the scripts add `src/` to `sys.path`;
in a notebook do `sys.path.insert(0, "<repo>/src")`.

**Precision switch (read this first):** the solver fixes its dtype at import.

```bash
export PYDISORT_RICCATI_JAX_X64=1   # float64 — BEFORE python/jupyter starts
```

Default is float32 with ODE tolerance ~1e-3 — fine for exploration and the 2-band
demos. **Production science (the 10-band, NQuad=48 observing system) requires
float64 + `SOLVER_TOL=1e-4`** — float32 destabilizes there (adaptive-step blowup).

## 3. Quickstart 1 — the forward solver (one call)

```python
import numpy as np
from pydisort_riccati_jax import pydisort_riccati_jax

tau_bot, NQuad, NLeg_all = 8.0, 16, 32
omega_func = lambda tau: 0.95 - 0.10 * tau / tau_bot        # single-scattering albedo
g_func = lambda tau: 0.85 - 0.05 * tau / tau_bot            # Henyey-Greenstein asymmetry
Leg_coeffs_func = lambda tau: g_func(tau) ** np.arange(NLeg_all)

mu_arr_pos, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0=0.9, I0=1.0, phi0=0.0,
    delta_M_scaling=True, NT_cor=True, NLeg_all=NLeg_all)

print(float(flux_up))                       # 0.14377 — ToA upward flux
print(np.asarray(u_func(np.pi)))            # ToA radiance at the N quadrature mu, phi=pi
```

- Inputs: `omega_func(τ)` → scalar, `Leg_coeffs_func(τ)` → `(NLeg_all,)` Legendre
  moments (χ₀=1), `NQuad ≥ 6` streams, solar geometry `(mu0, I0, phi0)`.
- Returns a 5-tuple, **upwelling-only** at ToA, size N = NQuad//2:
  `(mu_arr_pos, flux_up_ToA, u0_ToA, u_ToA_func, tau_grid)`. `u_ToA_func(φ)` is the
  azimuthally-resolved radiance at the quadrature μ's;
  `interpolate(u_ToA_func, mu_arr_pos)` gives barycentric μ-interpolation to
  arbitrary view angles.
- **Always turn on `delta_M_scaling=True, NT_cor=True` for peaked phase functions**
  and give `NLeg_all` enough moments (rule of thumb: ≳1.1× the largest Mie size
  parameter, including the droplet-distribution tail). Without them the finite-stream
  radiance rings negative — the same call above with the defaults produces a
  negative node. See guide §10 and the technical doc §4.
- Optional: `b_pos` (bottom diffuse source), `BDRF_Fourier_modes` (e.g.
  `[[0.06]]` = Lambertian albedo 0.06), `tol` (adaptive ODE tolerance).

## 4. Quickstart 2 — jit + gradients (the composable seam)

The one-shot call above is host-side. Inside optimization loops use the seam: build
the geometry once (`riccati_setup`, μ0 **static**), then solve/evaluate traceably.

```python
import jax, jax.numpy as jnp
from pydisort_riccati_jax import riccati_setup, riccati_solve, eval_radiance

setup = riccati_setup(NQuad, 1.0, 0.0, 0.9)     # (NQuad, I0, phi0, mu0); host-side, once

def toa_radiance(g, tau_bot):                   # g and tau_bot are TRACED
    omega = lambda tau: 0.95
    leg = lambda tau: g ** jnp.arange(NQuad)
    res = riccati_solve(setup, omega, leg, tau_bot)
    return eval_radiance(setup, res, 0.7, jnp.pi)     # radiance at mu=0.7, phi=pi

f = jax.jit(toa_radiance)                       # compiles once
g = jax.jit(jax.grad(toa_radiance, argnums=(0, 1)))   # reverse-mode discrete adjoint
f(0.85, 8.0)        # 0.06190
g(0.85, 8.0)        # (d/dg, d/dtau_bot) = (0.00952, 0.00156)
```

Rules of the seam:
- **Traced:** `tau_bot`, anything the optics closures capture. **Static (rebuild
  `setup` to change):** grid sizes, `I0/phi0/mu0`, BCs/BDRF, delta-M/TMS flags.
  Close `setup` over the jitted function; never pass it as a traced argument.
- Reverse mode (`jax.grad`, `jax.jacrev`) works out of the box. Forward mode
  (`jax.jacfwd` — cheaper when parameters ≪ observations) needs
  `riccati_setup(..., adjoint=diffrax.ForwardMode())`.
- Never wrap a traced output in `float()` inside traced code — it concretizes and
  kills the gradient.

## 5. Quickstart 3 — Mie optics + an OE retrieval

The production path: a miepython-grounded lookup table maps rₑ → (ω, Legendre)
differentiably; `RetrievalForward` wraps the multi-band multi-angle observation
operator; `gauss_newton_oe` runs the Levenberg–Marquardt-damped Gauss–Newton OE
retrieval. Small 2-band toy (float32, ~2 min CPU):

```python
import numpy as np
from pydisort_riccati_jax import noise_model as nm
from pydisort_riccati_jax import optics_table as ot
from pydisort_riccati_jax import retrieval_oe as roe

# r_e -> (omega, Q_ext, Legendre) table, built once (cache with build_or_load_table)
table = ot.build_re_table([1.64, 2.13], 4.0, 10.0, 16, v_eff=0.10,
                          n_radii=80, NLeg=160, n_gl=384)
opt_bands = [ot.select_channel(table, i) for i in range(2)]

fwd = roe.RetrievalForward(
    opt_bands, NQuad=8, mu0=0.9, I0=1.0, phi0=0.0,
    tau_bot=5.0, r_base=6.5,                    # first-guess anchors (never the truth)
    view_mu=np.linspace(0.9, 0.3, 4), view_phi=np.full(4, np.pi),
    BDRF_bands=[[0.06]] * 2, NLeg_all=160, NFourier=8, tol=1e-3,
    state_space="log", jac_mode="fwd",
    retrieve_tau_bot=True, retrieve_r_base=True, re_bounds=(4.0, 10.0))

# synthetic truth + noiseless observation y = F(truth); Se = the ASSUMED OCI noise
tau_truth = np.linspace(0.0, 5.0, 12)
re_truth = (6.5**5 + (9.0**5 - 6.5**5) * (1 - tau_truth / 5.0)) ** 0.2
y = roe.osse_observation(fwd, tau_truth, re_truth)
Se = roe.make_Se(fwd, y, nm.oci_swir())

s_nodes = np.array([0.0, 0.5])                  # free r_e nodes at normalized depth
x_a, Sa = roe.make_marine_sc_prior(s_nodes, r_top_prior=8.0, tau_bot_prior=4.0, log=True)
res = roe.gauss_newton_oe(fwd, y, s_nodes, x_a, Sa, Se,
                          n_iter=10, lm=1e-2, xtol=1e-4, max_n_outer=1)
post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)

r_nodes, r_base, tau_ret = fwd._split_state(res.x, s_nodes)
# tau_bot 4.78 (truth 5.0) | r_base 5.42 (prior-dominated: the base is radiatively
# shielded — expected physics, see the IC story) | DOFS 2.98 | converged True
re_curve = fwd.profile(res.x, s_nodes, np.linspace(0, float(tau_ret), 6))
```

The moving parts:
- **State** `x = [rₑ(s_nodes), r_base, τ_bot]` at **normalized depth** s = τ/τ_bot
  (nodes stretch with the retrieved τ_bot — the key to joint depth retrieval), in
  **log space** (`state_space="log"`, pair with a `log=True` prior).
- **Priors**: `make_marine_sc_prior` (generic, literature-grounded) or
  `make_climatology_prior` (leave-one-flight-out VOCALS ensemble). Principle: tight
  where the measurement is blind (cloud base), loose where it is strong (top, τ_bot).
- **Node selection** for real cases: `roe.select_num_modes(fwd, x_ref, s_ref, Se)`
  (noise-aware azimuthal-mode trim) and `roe.select_retrieval_grid(...)`
  (QRCP-ranked, noise/prior-whitened node placement + count). See
  `scripts/retrieval_worker.py` for the full production sequence.
- `posterior_diagnostics` → Ŝ, averaging kernel, DOFS, SIC, per-node data fraction.

## 6. Quickstart 4 — information-content profiling

"How much can this measurement resolve?" — answered on the **full ODE grid**
(independent of any retrieval-grid choice):

```python
from pydisort_riccati_jax import info_content as ic

prior_builder = lambda s: roe.make_marine_sc_prior(
    s, r_top_prior=8.0, tau_bot_prior=4.0)        # PHYSICAL-space prior here
post_ic, s_grid = ic.info_content_on_ode_grid(fwd, res.x, s_nodes, prior_builder, Se)
# DOFS 2.89, SIC 9.3 bits on 15 depth nodes

K, s_int = ic.jacobian_on_ode_grid(fwd, res.x, s_nodes)     # reusable Jacobian
_, Sa_pool = prior_builder(s_int)
spec = ic.info_spectrum(K, Sa_pool[:s_int.size, :s_int.size], Se)
spec.singular_values[:4]        # [20.4, 10.9, 2.7, 0.14] — whitened SNR spectrum
```

`jacobian_on_ode_grid` is the expensive step and depends only on state + geometry —
compute once, reuse across every prior and noise level. `flux_jacobian_on_ode_grid`
gives the angle-integrated (plane-albedo) spectral baseline. The campaign-scale
version of all this is `scripts/ic_worker_profile.py` / `ic_worker_mechanism.py`.

## 7. The production pipeline (HPC)

Three re-runnable batches over the 125 VOCALS profiles, each specified in
`hpc/AGENT_all125_{rad,ic,fr}.md` (read those before running; Slurm drivers in
`hpc/sbatch/`):

| Batch | Entry point | Product |
|---|---|---|
| `rad` | `scripts/generate_osse_radiances.py <idx> <parts>` → `consolidate` | `../data/osse_radiances.npz` (signature-gated truth radiances) |
| `ic`  | `scripts/ic_worker_profile.py` / `ic_worker_mechanism.py` | per-profile Jacobian/mechanism sidecars → aggregated by `scripts/ic_analysis_definitive.py` into the two notebook JSONs |
| `fr`  | `scripts/retrieval_worker.py <idx> runs/_fr_parts/<idx>` | per-profile A/B retrieval sidecars → metrics by `scripts/retrieval_analysis.py` |

Conventions:
- **`osse_config` is the single source of truth** for the observing system (bands,
  views, NQuad=48, NLEG_ALL=1536, NFOURIER, `signature()`), and for the default data
  locations. Change the observing system there and ONLY there — the signature hash
  gates every cache.
- Large data lives outside the repo at `../data/`; run outputs go to the untracked
  `runs/`.
- Checkpoint/resume: L1 (per-GN-iteration, always on via `checkpoint_path`) and L2
  (`FR_SETUP_CACHE=1`, caches the expensive per-profile setup). `FR_SETUP_ONLY=1`
  builds+caches the setup and exits (the CPU "setup farm" for later GPU runs).

Key environment variables (defaults in parentheses):

| Variable | Meaning |
|---|---|
| `PYDISORT_RICCATI_JAX_X64` | 1 = float64 (**required** for production) |
| `SOLVER_TOL` (1e-4) | adaptive ODE tolerance |
| `OPTICS_CACHE` (`../data/optics_table_10band_nleg1536_re20.npz`) | Mie table cache |
| `RADIANCE_CACHE` (`../data/osse_radiances.npz`) | truth radiances (signature-gated) |
| `VOCALS_DATA` (`../multispectral-retrieval-using-MODIS/VOCALS_REx_data`) | in-situ netCDFs |
| `MODE_MAP` (`scan`) | `vmap` = GPU bands×modes batching; `scan` on CPU |
| `ENSEMBLE_NQUAD` (48), `COST_RTOL` (0.01), `RADIANCE_TOL` | operating-point knobs |
| `FR_SETUP_CACHE`, `FR_SETUP_ONLY`, `FR_PIN_CORES`, `FR_SLOT_DIR` | resume/farm/affinity |
| `IC_MODE` (`priormean`\|`draw`), `IC_DEFINITIVE_OUT`, `IC_C_PARTS` | IC batch knobs |

On shared CPU nodes, `runtime_setup.setup()` must run **before JAX is imported**
(the workers do this) — it pins a disjoint per-node core slot; skipping it
oversubscribes XLA's thread pool. Importing the package itself never touches JAX.

## 8. Performance notes

- **CPU:** `mode_map="scan"`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` at
  1 CPU/task; more cores don't help a single column.
- **GPU:** `mode_map="vmap"` batches bands×modes in one SIMT launch (~2–5× per
  Jacobian at production shapes). Keep `NFOURIER` ceilings uniform per band —
  ragged ceilings disable the batch.
- μ0 is **static**: each distinct μ0 compiles once. Bin μ0 for per-scene work.
- `jac_mode="fwd"` when state dim < observation count (the retrieval regime).
- The compiled callables cache on the `RetrievalForward`; anything that changes
  `K_list` invalidates them automatically.

## 9. Testing

```bash
cd tests && python -m pytest . -v                       # default: float32 solver + retrieval (CI)
PYDISORT_RICCATI_JAX_X64=1 python -m pytest -m float64  # slow tight-tolerance partition
PYDISORT_HPC_GATES=1 python -m pytest hpc -m hpc -v     # production gates (cluster, hours)
```

The `tests/hpc/` gates are the standardized production checks: L1 resume
equivalence, L2 setup-cache equivalence, and the 3-profile golden cross-check that
must pass after any numerics-touching refactor.

## 10. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Negative ToA radiances | forward-peaked phase fn without delta-M/TMS, or `NLeg_all` too small (needs ≳1.1× the max size parameter incl. the size-distribution tail) → `delta_M_scaling=True, NT_cor=True`, raise `NLeg_all` |
| `max_steps` / CpuCallback crash | float32 at a heavy configuration, or `tol` too loose for a thick cloud → float64 + `SOLVER_TOL=1e-4` |
| `jax.jacfwd` errors about custom_vjp | rebuild with `riccati_setup(..., adjoint=diffrax.ForwardMode())` |
| Gradient is None/breaks | a `float()`/`np.asarray()` concretization inside traced code |
| "radiance-cache signature mismatch" | observing system changed vs the cache → regenerate with `scripts/generate_osse_radiances.py` (this is the gate working as designed) |
| 30+ threads on 1 CPU (cluster) | JAX imported before `runtime_setup.setup()` |
| GPU `ptxas` abort (transient) | resubmit — L1/L2 resume makes it ≤1 iteration lost |
