# Design Decisions

Settled design decisions and their rationale, recorded so they are not
re-litigated or misremembered (e.g. "is the discrete adjoint implemented?").
Open questions live in [`OUTSTANDING.md`](./OUTSTANDING.md).

Status tags: **[SETTLED]** decided and in effect · **[INVARIANT]** must never be violated.

---

## 1. Solver lineage: invariant-imbedding Riccati + implicit adaptive integration  [SETTLED]

The forward solver integrates the **matrix Riccati equation** (invariant imbedding) with
diffrax's **Kvaerno5** (L-stable ESDIRK, adaptive). **Minimising the number of integration
steps (τ-points) is a primary design motivation** — its payoff is **retrieval-grid viability**:
the implicit, L-stable solver produces a *minimal, artifact-free, profile-adaptive* set of
τ-points, and that set is the only a-priori-available, problem-derived candidate for the
retrieval grid (it is not pursued for runtime). The lineage that led here is recorded so the
dead ends are not retried:

- **Magnus + Redheffer star product** ("Report I"): unconditionally *stable* — works entirely
  with O(1)-bounded N×N reflection/transmission/source operators, avoiding the coexisting
  growing and decaying modes of the 2N×2N propagator. But its **step count is set by the
  ballistic eigenvalue** (`λ_max ∼ 1/μ_min ≈ 45` at NQuad=16, the single-scattering regime):
  `K ≳ λ_max·τ_bot`, even though the observable evolves on the much slower **diffusion
  eigenvalue** (`k_min ≈ √(3(1-ω)(1-ωg))`, the diffusion domain; ≈0.07 at ω=0.99, →0.02 at
  ω=0.999). τ=50, ω=0.99, g=0.85 → ≈2272 steps.
- **Three-domain decomposition** ("Report II"): split `[0,τ_bot]` into a non-diffusive top
  (beam active), a diffusion interior (beam negligible), and a non-diffusive near-surface
  layer; solve each independently and couple them with the (associative) star product — the
  aim being to step coarsely through the diffusion interior. **Discarded** because it produced
  essentially no step saving (462+914+914 = 2290 steps vs 2272 for a single Magnus solve, same
  wall time): the interior still required fine Magnus steps, so the split added bookkeeping for
  no benefit.
- **Resolution — one implicit stiff solver across the whole domain.** Integrating the Riccati
  equation with an implicit (Kvaerno5) method steps at the *diffusion* rate *everywhere*, so no
  domain split is needed. This is both **simpler** (a single solver, no domain boundaries to
  choose) and gives **far fewer τ-points**: ~35 adaptive steps for a τ=30 cloud (nearly
  NQuad-independent) vs thousands for Magnus. The τ-point reduction is the whole point.

**Why the minimal grid matters for retrieval (the candidate-pool argument).** An L-stable
solver places steps by *genuine* variation of the Riccati state, not by stiffness artifacts.
This makes the converged grid a **trustworthy *superset*** of the retrieval-informative
τ-points (it resolves all genuine variation, so it cannot miss an informative feature), while
keeping that superset *minimal* — which conditions the downstream sensitivity selection well
(a tight, well-placed pool vs the artifact-clustered thousands a Magnus grid would hand QRCP).
The retrieval grid is then a **sensitivity-selected *subset*** of this pool, not the whole
grid: the same contraction that guarantees the superset (state settles ⇒ all variation
resolved) also forbids using it wholesale (deep steps — the BoA imbedding boundary layer and
depth-attenuated deep features — carry little information). This superset/subset relationship,
and the alternative of decoupling via a smooth low-dim basis (not pursued here — a basis carries
its own a-priori commitment in its shape/training distribution and is static rather than
profile-adaptive; which route wins is left open), is the retrieval-grid material in §3 below.

**Retained from the rejected work:** the **Redheffer star product / interaction principle** (the
layer-composition rule underlying the Riccati derivation, and the up/down combination in the BC
solve) and the **N×N O(1)-bounded operator** formulation.

*(Distilled into report_riccati_solver.tex §"Solver Lineage" from the former
technical_reports/report_star_product_magnus.tex and report_diffusion_domain.tex — removed,
recoverable from git `99fb971`.)*

---

## 2. Numerical-stability invariant: NO POSITIVE EXPONENTS  [INVARIANT]

No intermediate quantity may contain `exp(+λ·τ)` with `λ > 0`, `τ > 0` — this overflows for
thick atmospheres. The Riccati formulation satisfies this by construction (the state stays
O(1); no positive exponents). Any future algorithm change (Magnus, doubling, SVD, …) must
preserve it.

---

## 3. Retrieval grid = sensitivity-selected subset of the ODE grid; information is ToA-weighted; retrievable DOF is small  [SETTLED — open sub-questions in OUTSTANDING G]

Three robust statements below. Two earlier claims — an exact **"rank-4 ceiling"** and full
**profile-independence** — are *deliberately not asserted* (no rigorous basis; measured under
biasing conditions). They are tracked in [OUTSTANDING G](./OUTSTANDING.md).

**(a) The ODE grid is a trustworthy candidate *pool*; the retrieval grid is a
sensitivity-selected *subset* of it — not the whole grid, and not a separate a-priori grid.**
The Kvaerno5 grid is placed for *solver accuracy*: being L-stable, it clusters by *genuine*
variation of the Riccati state, not by stiffness artifacts. That makes it a trustworthy
**superset** of the retrieval-informative τ-points — it resolves all genuine variation, so it
cannot *miss* an informative feature — while staying *minimal* (this is the retrieval-grid
payoff of §1's τ-point minimisation: a tight, well-placed pool conditions the selection far
better than the artifact-clustered thousands a Magnus grid would hand QRCP). The grid is a
*superset*, **not equal**, because **placement and selection use different criteria**:
- **Placement (ODE grid)** is by *state* variation. But genuine state variation ≠ optics
  variation: the state varies for two reasons — (i) real optics features, and (ii) the **BoA
  imbedding boundary layer**, where R relaxes from its IC toward R_∞ over the bottom stretch
  (contraction *forgetting the IC*; steepest stream μ_min≈0.02 settles ~50× faster). (ii) is
  *optics-independent* (occurs even for constant optics) and carries ≈zero information, so the
  ODE grid is **not an information map** — it is densest exactly where it is least informative
  (~14 of ~35 steps at BoA).
- **Selection (retrieval grid)** is by *sensitivity*, which decays monotonically with depth, so
  a deep point is pruned *regardless of why it was placed*.

This gives a strict nesting **{retrievable from ToA} ⊆ {information} ⊆ {variation}**: *variation*
= where the *state* changes (ODE grid); *information* = where the *optics* change (a real
feature exists); *retrievable* = the ToA-visible subset. The two gaps are the two contraction
faces — variation∖information = the IC boundary layer (**primal**), information∖retrievable =
depth-attenuated features (**adjoint**). Worked example: in
`adiabatic_cloud_with_drizzle.ipynb`, a near-ToA g-spike at τ=1 on a conservative τ=30 cloud
raised the step count 107→186, clustering on the spike *and* keeping the (uninformative) BoA
cluster. **So: select the retrieval grid as a sensitivity/QRCP-weighted subset of the ODE
pool.** *Which* ODE grid: the **m=0 grid alone** (the solver returns `tau_grid_m0` and discards
m≥1). Folding in the m≥1 grids — pool = the **union of the non-negligible modes' grids** — was
tested and **rejected** (OUTSTANDING §G, 2026-06-20): because optics ω(τ), gₗ(τ) are *shared*
across Fourier modes, the m≥1 grids only densify the *same* near-ToA region with near-collinear
duplicates, so the union adds no informative depth and merely lets QRCP over-concentrate near ToA
and abandon deep coverage (neutral-to-harmful — RF10 shielded RMSE 0.52→0.65 µm, drop-cap
58 %→187 %). The m=0 grid is thus a *complete* pool; the lone exception is a future v_e(τ)-resolving,
**polarized** forward model, where high-m modes would carry independent cloudbow vertical
information (OUTSTANDING §G flip condition). The alternative — *decoupling* the retrieval grid from the ODE grid via a smooth
low-dim basis (EOF/adiabatic-shape/NN) — is **not pursued here** (these are largely unexplored
beyond the adiabatic case): a basis carries its own a-priori commitment in its shape/training
distribution and is *static*, whereas the ODE grid is problem-derived, profile-adaptive, and
re-placeable. The ODE-pool route's complementary weakness — it inherits the *first-guess*
profile's bias — is correctable by lagged re-selection (see [OUTSTANDING B](./OUTSTANDING.md)).
Which route wins is left open.

**(b) Information is ToA-weighted with finite penetration — conditional on a thick cloud.**
The retrieved r_e is a vertical-weighting-function-weighted average concentrated toward cloud
top, with penetration depth set by band absorption (Platnick 2000 [verify specifics]).
Empirically, the column ℓ₂-norm of `∂u⁺/∂{ω,g}(τ)` (m=0) falls ~5 orders of magnitude from ToA
(~5×10⁻²) to BoA (~10⁻⁷); a δω=0.01 at BoA shifts u⁺(ToA) by ~2×10⁻⁸ relative — below any
noise. QRCP on the ToA Jacobian selects informative points in τ≲4, reaching depth only via a
forced BoA anchor. **Near-ToA features are detectable** — the τ=1 spike is captured (the trimmed
grid concentrates on it) — so this is *ToA-weighted*, not *near-ToA-blind*; detectability
degrades with depth (empirically a τ≈3–5 boundary, consistent with the literature cutoff). All
of this is conditional on optical thickness: for thin clouds (τ≲5–15) the whole column is in
view and the deep region is *not* prior-dominated.

**(c) The retrievable DOF is small — use a low-dimensional parameterisation.** The number of
independent directions in the ToA Jacobian is bounded by the stream count N (=8 at NQuad=16) and
is empirically a handful; multi-band data adds independent information only with diminishing
returns due to inter-band correlation (Coddington, Pilewskie & Vukicevic 2012 [verify] — for
thick clouds VIS/NIR become near-orthogonal, so a *few* complementary bands are non-redundant,
many are not). **Robust consequence, independent of the exact number:** parameterise r_e(τ) with
a *few shape parameters*, not a fine level-by-level grid; place what resolution there is near
cloud top by *sensitivity*, not by assumed profile curvature (adiabatic curvature is steepest at
*base* — where the observable is weakest); the deep r_e is prior-dominated, not measured.

**(d) The inter-node interpolation is part of the forward map, not a post-hoc step — and the
display must mirror it.** Once §3c fixes the nodes, the *function class between nodes* is the
remaining parameterisation choice. It lives **inside** F(x): the single lever
`retrieval_oe.RetrievalForward._re_of_tau` is what every solve, calibration, ODE-grid build,
Jacobian, and re-meshing re-map integrates, so it *defines what is retrieved* — **plot the result
with `RetrievalForward.profile()`** (which routes through the same lever) so the displayed curve
cannot drift from the forward model. *(Earlier framing of this as an independent post-hoc choice
was wrong — corrected here.)* **Current default: r_e⁵-linear (adiabatic)** — the adiabatic law *in
optical depth* is `r_e ∝ τ^(1/5)`: `r_e³ ∝ LWC ∝ geometric height z`, but `β ∝ r_e² ∝ z^(2/3)` makes
`τ ∝ z^(5/3)`, so `LWC ∝ τ^(3/5)` and `r_e ∝ τ^(1/5)` (≡ canonical `N_d ∝ τ^(1/2) r_e^(-5/2)`). So
`r_e⁵` is linear in τ; it represents the adiabatic prior *exactly*, gets per-segment curvature from two
endpoints (no grid-size coupling), and stays non-overshooting with finite base slope. It is C⁰ (kinked
at nodes). The function-class itself is **not settled** — it is an open inverse-problem bias–variance
lever (linear / adiabatic / PCHIP), tracked in
[OUTSTANDING §B′](./OUTSTANDING.md); bounded above by the integrator order (~C⁶, §1) and far more
tightly by the small DOF here.

*(Distilled into report_riccati_solver.tex §"The Retrieval Grid and Its Relation to the ODE
Grid" from the former technical_reports/boa_step_clustering_report.tex and the tests/supplementary
QRCP/Jacobian scripts — removed, recoverable from git `99fb971`; see also the retained
adiabatic_cloud_with_drizzle.ipynb. Detailed tables now in the report; multi-mode analysis
pointer in OUTSTANDING G.)*

---

## 4. Precision: float32 default, float64 opt-in  [SETTLED]

Production runs in **float32** with `tol ≈ 1e-3` (`atol = 1e-6`, safely above float32 eps
≈1.2e-7); retrieval accuracy is set by ~10–20% measurement noise, not float resolution, and
float32 keeps the adaptive step count low. Tighter `tol` (≤1e-4 on thick atmospheres) makes
the Kvaerno5 controller shrink `dt` forever (an Equinox max-steps error — a *controller*
failure, not instability). Set `PYDISORT_RICCATI_JAX_X64=1` for float64, where tight `tol`
(1e-8…1e-10) is reachable — required for finite-difference gradient checks. Tests are split
into a float32 default partition and a `@pytest.mark.float64` opt-in partition.

---

## 5. Discrete adjoint = free reverse-mode AD (DONE, verified) — NOT a separate feature  [SETTLED]

Differentiating through the solve is **free reverse-mode AD** via diffrax's default
`RecursiveCheckpointAdjoint` (a *discrete* adjoint, preferred over the hand-derived continuous
adjoint of LIDORT/SCIATRAN). **No separate adjoint code exists or is planned.** Verified:
`jax.grad` of `flux_up_ToA` w.r.t. an optical property matches finite differences to ~2e-10
(`tests/18_adjoint_test.py`, float64 partition). Prerequisite (already fixed): removing the
eager `float(flux_up_ToA)` that used to concretize and break grad-through-solve.

**Why not `BacksolveAdjoint` (diffrax's *continuous* adjoint).** `BacksolveAdjoint` is the
optimise-then-discretise / Neural-ODE adjoint: it stores only the final state and recovers the
gradient by integrating a separate adjoint ODE — and re-integrating the state — **backwards in
time**, for O(1) step-memory. We reject it for three reasons. (i) **Inconsistent gradient:** it
approximates the derivative of the *exact* ODE solution, not the exact derivative of the *discrete*
forward our loss actually evaluates, so ∇(numeric forward) ≠ numeric(∇ continuous) at finite
tolerance — a mismatch that degrades Gauss–Newton/line-search convergence; the discrete
`RecursiveCheckpointAdjoint` (and `ForwardMode`, §7) instead differentiate the scheme we run, so the
gradient is exact-by-construction (this is the §5 discrete-vs-continuous distinction). (ii)
**Unstable backward integration:** our Riccati system is stiff and strongly contractive (the reason
for the L-stable solver and the BoA boundary layer), and reversing a contractive flow is *expansive*
— it amplifies error and cannot recover the IC information the forward contraction erased, i.e. it
would reintroduce the growing-exponential problem the invariant-imbedding formulation exists to
remove (NO-POSITIVE-EXPONENTS, §2). diffrax's own guidance points the same way: `BacksolveAdjoint`
is intended for non-stiff problems and can be inaccurate, with checkpointing recommended for stiff
systems. (iii) **No memory pressure to justify the trade:** our solves are ~35 steps on tiny N×N
state (N ≤ 8/16), so the checkpointed adjoint's memory is already negligible — we would take on the
accuracy/stability cost to fix a problem we don't have. (`BacksolveAdjoint` is reverse-only too, so
it offers nothing for forward-mode.)

> This entry exists because we were previously unsure whether the adjoint had been
> implemented. It has — as autodiff, not as bespoke code.

---

## 6. Delta-M scaling (physical-τ form) + Nakajima–Tanaka TMS; IMS omitted  [SETTLED]

Opt-in (`delta_M_scaling=False`, `NT_cor=False` by default) fix for the negative ToA radiance of
forward-peaked phase functions ([OUTSTANDING A](./OUTSTANDING.md), now resolved *in the realistic
regime*). Delta-M removes the truncated forward peak (the m≥1 Fourier ringing source); TMS
restores the single scattering with the exact, untruncated phase function. Default-off keeps every
existing test and the un-scaled numerics bit-for-bit. **Correctness criterion is matching
PythonicDISORT's `NT_cor` (~1e-6), not strict non-negativity:** for peaks far under-resolved by the
stream count a residual negative lobe can remain — present in pydisort's `NT_cor` too, so intrinsic
to TMS, not this implementation (quantified in [OUTSTANDING A](./OUTSTANDING.md)).

**Activation = a single flag, with f derived internally (DISORT-style, *not* PythonicDISORT's
`f_arr`-as-data).** The truncation fraction is *always* `f(τ) = g_{NLeg}(τ)` — the first dropped
Legendre moment, i.e. the slice `Leg_coeffs_func(τ)[NLeg]` of the **same differentiable phase
function** the Mie front-end already produces. Passing `f` separately would be redundant and, in
the retrieval loop, would force `f` to track the current `rₑ(τ)` estimate *and* carry `∂f/∂rₑ`
consistently by hand; deriving it from the coefficients guarantees both and keeps the chain
`rₑ → Mie → (ω, gₗ, f) → solver → radiance` clean. PythonicDISORT's `f_arr` fits *its* static
forward model; the flag fits ours. No `f_func`/`f_arr` input exists.

**Physical-τ delta-M (the key simplification — no reparametrization).** Rather than integrate in
the scaled depth τ\*, the solver keeps integrating the Riccati ODE in **physical τ** (the existing
σ↔τ machinery and physical `tau_grid` are unchanged). Mapping the scaled-depth RTE back via the
Jacobian `dτ\*/dτ = scale_tau`, with `scale_tau(τ) = 1 − ω(τ)f(τ)`, collapses delta-M to three
pointwise substitutions using the **effective moments** `c_ℓ = g_ℓ − f` (= (1−f)·g\*_ℓ):

```
α_dM = M⁻¹(ω·D⁺[c]·W − scale_tau·I)      β_dM = M⁻¹(ω·D⁻[c]·W)
q±_dM = M⁻¹·scalar_fac·ω·exp(−τ\*(τ)/μ0)·X±[c]
```

i.e. the *only* changes to the existing einsum kernels are `c_ℓ` for `g_ℓ`, `−scale_tau·I` for
`−I`, and `τ\*` for `τ` in the beam exponent. `τ\*(τ) = ∫₀^τ (1−ωf)dτ′` is the lone cumulative
(non-pointwise) quantity; it is **azimuth-mode-independent**, so it is built once (fixed-grid
cumulative-trapezoid + `jnp.interp`, differentiable) before the Fourier loop. f=0 ⇒ c=g,
scale_tau=1, τ\*=τ ⇒ identical to the un-scaled solver. **NO-POSITIVE-EXPONENTS (§2) preserved:**
τ\*≥0 and only `exp(−τ\*/·)` ever appears.

**TMS (single-scattering), continuous physical-τ form.** Derived by mapping PythonicDISORT's
per-layer TMS coefficient `mathscr_B` to the continuum; the upwelling ToA correction is the single
1-D τ-quadrature

```
Δu⁺_i(φ) = (I0/(4π μ_i)) ∫₀^{τ_bot} ω(τ)·[p_full − (1−f)p_trunc](cosΘ_i(φ))
                                     · exp(−τ\*(τ)·(1/μ0 + 1/μ_i)) dτ,
```

whose angular bracket is the **missing forward-peak detail**
`Σ_{ℓ<NLeg}(2ℓ+1)f·P_ℓ + Σ_{ℓ≥NLeg}(2ℓ+1)g_ℓ·P_ℓ` (effective coefficients `b_ℓ = f` for `ℓ<NLeg`,
`g_ℓ` otherwise). Properties: the integrand is **smooth** (the angular sharpness lives in cosΘ,
evaluated *exactly* via `calculate_nu`), so a fixed Gauss–Legendre τ-quadrature (order ~128)
suffices; the exponent `1/μ0 + 1/μ_i` is a **sum** (no μ_i=μ0 singularity for upwelling) and
always negative (invariant preserved); it corrects the **intensity only** — `u0_ToA` and
`flux_up_ToA` stay the delta-M m=0 quantities (flux is delta-M-preserving). The correction is
analytic in μ, so `interpolate` adds it **directly at the requested μ** rather than interpolating
the sharp single-scattering peak (the smooth multiple-scattering field is what gets
μ-interpolated). It lives *outside* the Riccati sweeps and the Fourier loop, so it does not affect
the step count (delta-M, by reducing asymmetry, tends to *lower* it).

**IMS omitted by design.** Nakajima–Tanaka IMS corrects the **downward/transmitted** field **only —
by construction in DISORT itself** (STWLE 2000 DISORT report, App. A just after eq A.1: the −µ
argument indicates "we correct only downward not upward intensities"). This solver returns ToA
**upwelling** only, so there is no standard IMS to apply; the retrieval-grade codes LIDORT/VLIDORT
(Spurr) likewise omit IMS (exact single-scatter + streams, + a separate 2OS model). The residual
truncation error of sharp peaks is best handled by **more streams** if it is ever shown to matter;
δ-M+ (Lin & Stamnes 2018) was examined and deprioritised (ecosystem-confined, and its "same cost"
does not hold for our differentiable pipeline) — see [OUTSTANDING A](./OUTSTANDING.md).

*(Implemented in `src/_riccati_solver_jax.py` — `_compute_tau_star`,
`_legendre_weighted_sum_jax`, `_precompute_tms` / `_apply_tms` (split from the former
`_make_tms_func` so the TMS state is a traceable pytree carried by `SolveResult`; see §7), and the
delta-M branches of the α/β/q builders — and wired in `src/pydisort_riccati_jax.py`. Verified:
`tests/19_deltaM_test.py` (float32: regression, positivity, τ-varying Design-B match, flux
invariance, grad smoke) and `tests/20_deltaM_benchmark_test.py` (float64: Design-A convergence,
stream-convergence demonstration, exact single-layer match vs pydisort `NT_cor`, Mie-coupled, FD
gradients); `tests/supplementary/demo_deltaM_tms.py` is the radiance-vs-angle figure.)*

---

## 7. jit-able forward via a host-side setup / traceable solve split; DISORT azimuthal convergence  [SETTLED]

> **Update (OUTSTANDING §H, 2026-06).** Three things below were changed when the §H compile-memory
> OOM was fixed; the seam shape and bit-for-bit core are otherwise as described:
> 1. **μ0 is now static** (a `riccati_setup` parameter, not a `riccati_solve` arg). The in-trace
>    associated-Legendre recurrence `_assoc_legendre_neg_mu0_jax` is **removed** — `P_l^m(−μ0)` is
>    precomputed host-side with scipy. So wherever this section says "traced `mu0`" / "the recurrence",
>    read "static `mu0`, precomputed".
> 2. **The Fourier modes run under `lax.scan`** over padded `(NFourier, NLeg, N)` per-mode tensors
>    (not a Python-unrolled loop over ragged tensors). The mode body compiles **once** (O(1) compile
>    memory), which is what lifted the OOM ceiling.
> 3. **`calibrate_num_modes` (the relative Cauchy test) is removed.** Mode truncation is now the
>    noise-aware `retrieval_oe.select_num_modes` (S_ε); the solver default is all `NFourier` modes.

Resolves [OUTSTANDING C](./OUTSTANDING.md) (the retrieval-cost blocker). The forward model is now
**jit-able** through a thin, documented composable seam that separates the host-side SciPy setup
from a traceable solve — so the retrieval amortises one compile across hundreds of forward/gradient
evaluations instead of recompiling every call. The one-shot `pydisort_riccati_jax` still exists and
**delegates to the same core**, so its 5-tuple is unchanged (verified **bit-for-bit** — the seam
and the legacy entry produce identical `u_modes`, test 21b).

**The seam.** `riccati_setup(...) → SetupData` does everything `mu0`/`tau_bot`-independent once
(validation, double-Gauss quadrature, the **mu0-independent** per-mode Legendre tensors, per-mode
boundary conditions, source rescaling, barycentric weights). `riccati_solve(setup, omega_func,
Leg_coeffs_func, tau_bot, mu0, num_modes=K) → SolveResult` is then a **pure, jit/grad/jacfwd-able**
function of the **traced** inputs. `eval_radiance(setup, result, mu, phi)` is the traceable ToA
observable; `calibrate_num_modes(...)` is the host-side Cauchy control decision (below). Close
`setup` over the jitted function (it is a host-side object, **not** a traced argument).

**Traced vs static contract.** *Traced:* `tau_bot`, `mu0`, and the optics closures
`omega_func`/`Leg_coeffs_func`. *Static (baked into `setup`):* grid sizes, `I0`, `phi0`, the
boundary conditions, the BDRF, and the `delta_M`/`NT_cor` flags. Tracing `mu0` (a swath of solar
geometries reusing one compiled forward) costs a **custom associated-Legendre recurrence**: JAX has
no usable `P_l^m` (`lpmn` is deprecated, there is no `lpmv`), so `_assoc_legendre_neg_mu0_jax`
computes `P_l^m(−μ0)` via the fixed-`m` upward-in-`l` recurrence (Condon–Shortley phase, matching
`scipy.lpmv`). Gated and accepted (test 21a): float64 agreement ≈1e-15, float32 ≤1.5e-4 at the
high, Cauchy-truncated modes, gradient finite and FD-accurate, ~ms in-trace. The scipy `mu_arr_pos`
Legendre tensors are `mu0`-independent and stay in `setup`.

**Two host-side blockers, both fixed:** (i) the σ-grid build did `np.asarray(sol.ts)` + dynamic
slicing — fixed by a `save_grid` flag (`SaveAt(t1=True)` returns only the final state, no host sync,
`tau_bot` may be traced; the offline grid path keeps `SaveAt(steps=True)`); (ii) `_precompute_legendre`
called scipy on tracers — fixed by lifting it into `setup` (mu0-independent) and moving the
mu0-dependent term to the recurrence. The TMS state was likewise split into a traceable pytree
(`_precompute_tms`/`_apply_tms`) carried by `SolveResult`.

**DISORT azimuthal convergence (the "Cauchy criterion", STWLE2000 §3.7, p.89), implemented exactly.**
PythonicDISORT returned *functions* (so the criterion was inapplicable); to jit we must supply the
evaluation points, which puts us back in DISORT's situation. `calibrate_num_modes` forms the partial
sums `I_K(μ,φ)=Σ_{m=0}^{K} I^m(μ)·cos(m(φ0−φ))` over the **user** angles and stops at the first K
where `max_{μ,φ}|I^m·cos(m(φ0−φ))|/|I_K| ≤ ε_azim` holds **twice in succession** (counter resets on
a miss). `ε_azim` is user-set with a strong pseudo-hardcoded default `_DEFAULT_TOL_AZIM = 1e-3`
(DISORT requires ACCUR<0.01); `ε_azim = 0` ⇒ use all NFourier modes (ACCUR=0 semantics). The
one-shot entry defaults `tol_azim=0` (Cauchy off ⇒ bit-for-bit backward-compat). This is a
**concrete host-side** decision returning a Python `int K`; `riccati_solve(..., num_modes=K)` then
runs exactly K modes as a **static, Python-unrolled** loop (each mode keeps its natural ragged
`(NLeg−m,N)` tensors — **no pad/mask**), fully differentiable in **both** AD modes and jit-able
(compiles K mode-blocks once, caches). **No `vmap`, no `while_loop`** — the sequential Cauchy
early-stop and reverse-mode AD both rule them out; that is the whole reason K is a concrete int. The
TMS correction (single scatter, added outside the Fourier series) is **not** part of the test —
Cauchy governs only the diffuse multiple-scattering modes, matching DISORT. *(A single-trace
`scan`+pad/mask scheme that would make K a runtime mask — one compile for all K — is deferred; it
only pays if per-K cold compiles ever bind, see [OUTSTANDING C](./OUTSTANDING.md).)*

**AD mode requires choosing the adjoint.** Reverse-mode `jax.grad` (the verified discrete adjoint,
§5) is the **default** (diffrax `RecursiveCheckpointAdjoint`). Forward-mode `jax.jacfwd` (preferred
for small-DOF retrieval, [OUTSTANDING E](./OUTSTANDING.md)) **cannot** be applied to the default's
`custom_vjp`; build the setup with `adjoint=diffrax.ForwardMode()` for it. The solver exposes
`adjoint` on `riccati_setup` (threaded into `diffeqsolve`), defaulting to reverse so §5 and the FD
gradient tests are unchanged.

**NO-POSITIVE-EXPONENTS (§2) preserved** — the split changes only *where* quantities are computed,
not the Riccati state, which stays O(1).

*(Implemented in `src/pydisort_riccati_jax.py` — `SetupData`/`SolveResult`, `riccati_setup`
(static `mu0`, padded per-mode + scipy `P_l^m(−μ0)` tensors), `_fourier_solve` (the `lax.scan`
over modes), `riccati_solve`, `eval_radiance` — and `src/_riccati_solver_jax.py` — the
mode-index-free α/β/q builders, the `save_grid`/`adjoint` flags, `_precompute_tms`/`_apply_tms`.
Verified: `tests/21_jit_test.py` (seam↔jit↔legacy↔pydisort parity, with/without delta-M+TMS);
the FD adjoint tests `18`/`20e` were rerouted through the jitted seam.
`tests/supplementary/demo_jit_retrieval.py` is the recipe demo. The S_ε mode selector lives in
`src/retrieval_oe.py::select_num_modes`.)*

## 8. The r_e signal is phase-function-borne; Mie-Legendre table optics retained  [SETTLED]

`docs/jacobian_decomposition.ipynb` splits the retrieval Jacobian ∂R/∂r_e into its ω and gₗ
channels (freeze-one-channel closures; the split reproduces `RetrievalForward.jacobian` exactly
and is chain-rule additive) on three VOCALS truth profiles. Verdict — **the r_e retrieval is
not an ω retrieval**:

- The phase-function channel carries ~99% of the sensitivity for thin clouds at 1.24 µm,
  ~60–90% at 2.13 µm, and still 40% for the thick (τ=23) cloud at 2.13 µm. It also carries the
  *angular* information: the J_ω view-rows are near-parallel (~1 DOF however many views) while
  the J_g rows decorrelate across views — the multi-angle vertical DOF is g-borne.
- HG with the **exact** table g(r_e) is a 10–20σ forward error in the retrieval geometry
  (glory/backscatter, Θ ≈ 153–176°) and the HG retrieval is unfittable (DOFS 0.7 vs 2.6): the
  information rides on high-order Legendre moments (consistent with the §6 `NLeg_all` finding).
- The "exact AD Jacobian" motivation for tracing Mie fails on its own: `mie_avg`'s radius grid
  moves with r_e, so AD differentiates the oscillating quadrature artifact (differentiation is
  a high-pass; locally ~10×-wrong slopes at production `n_radii`), whereas the table's wide-
  baseline secant low-passes it (~5% median slope agreement at 2.13 µm). The table slope is the
  *better* ∂ω/∂r_e.

**Decision:** keep the (n_re, NLeg) Legendre table as the production optics path; the hybrid
traced-Mie-ω + HG pipeline is rejected. If a traced ω path is ever wanted (hyperspectral),
`mie_avg` first needs an r_e-independent radius grid. The GN state must be clamped to the table
support *inside* the forward map (bounded-state forward) — model error can otherwise drive
iterates to NaN optics. Details and figures: the notebook.

**Optics-backend swap → `optics_table` (miepython), JAX-Mie retired  [2026-06-24].** Since the
table is *built once outside the gradient* and consumed only through the differentiable
`table_lookup` interpolation, **autodiff never flows through Mie** — so the JAX-Mie front-end
(`miejax_lite.mie_avg_legendre`) buys nothing for the retrieval/IC Jacobian. The table is now
built with **miepython** (Bohren & Huffman; numba-accelerated) in `src/optics_table.py`
(gamma-averaged over v_eff=0.10; Segelstein water n,k vendored to `src/data/`, covering 0.55–3.7
µm), and `table_lookup` is ported there (the only piece still in JAX). `retrieval_oe` now imports
`table_lookup` from `optics_table`; **`miejax_lite` is retired from the production path** (repo
kept for legacy). Validated to round-off vs `miejax_lite` on the shared bands (|Δω|,|Δg|≈3–4e-6
across all 128 Legendre moments; `tests/supplementary/validate_optics_table.py`). The table is
**profile-independent**, so it is built once and disk-cached (`build_or_load_table`), shared by
every per-profile worker. Motivation: miepython reaches the strong-absorption bands (3.7 µm) the
band superset needs (§14), and is the field-reference Mie.

---

## 9. Lower boundary: Lambertian sea surface, albedo 0.06; the BDRF input *is* the albedo (NOT albedo/π)  [SETTLED]

**Surface.** The bottom boundary is a **Lambertian sea surface with albedo ρ_s = 0.06** — the
canonical broadband sea-surface value (Payne 1972, *J. Atmos. Sci.* 29, 959–970; reconfirmed by
Huang et al. 2019, *JGR Oceans* 124, 4856). Applied **identically** in the OSSE-truth and the
retrieval forward (a *known* boundary, not retrieved). One honest caveat: 0.06 is *broadband*
(visible-weighted), and our retrieval bands are NIR/SWIR where the ocean is darker (→ ~0 in the
SWIR, where liquid water absorbs), so a flat 0.06 is a mild **upper bound** on the surface
contribution — the safe direction for a known lower boundary, and not worth splitting per-band.

**Convention (verified).** `BDRF_Fourier_modes` are the surface-reflectance Fourier modes such
that, for a Lambertian surface, the m=0 mode **equals the albedo ρ_s directly** — i.e.
`BDRF_Fourier_modes=[ρ_s]`, **NOT** `[ρ_s/π]`. The BC solve builds `R_surf=(1+δ_{m0})·ρ·μ_j·W_j`,
whose hemispheric integral is exactly ρ. Confirmed two independent ways: (i) PythonicDISORT's own
docs (`omega_s = 0.1  # Ocean albedo is approximately 0.1`; `BDRF_Fourier_modes=[omega_s]`); (ii) a
single-bounce benchmark in this env (ω→0 absorbing slab, `I⁺(μ)=(A·μ₀I₀/π)·e^{−τ/μ₀}·e^{−τ/μ}`):
input `[0.1]` → effective albedo 0.0998; input `[0.1/π]` → 0.0318. The BC *formulation* (resolvent
`(I − R_surf·R_down)⁻¹` summing the cloud↔surface multiple reflections + an attenuated direct-beam
surface term) is correct and reproduces pydisort (`tests/5_test.py`, incl. albedo 0.1).

**Historical bug (flagged; sweep tracked in OUTSTANDING §J).** The repo's notebook/tests/demos
passed `[ρ/π]` throughout — i.e. surfaces **π× too dark** than labeled (notebook "dark ocean
0.05/π" = albedo 0.0159; test 5a "albedo 0.1" = 0.0318). This never broke a test (the same value is
fed to both the Riccati solver and pydisort ⇒ the solver-vs-reference comparison still holds) and
never biased a retrieval (surface applied consistently in truth + forward) — only the *nominal
physical albedo* was wrong. The retrieval/OSSE surface (the notebook) is now corrected to `[0.06]`.

---

## 10. Joint retrieval of r_e(τ) **and** the cloud base (τ_bot, r_base); normalized-depth grid  [SETTLED]

The VOCALS r_e(τ) retrieval previously fixed the cloud base `(τ_bot, r_base)` from the truth — a
threefold information leak (τ_bot, r_base, **and** the prior mean of the top radius r_top). It is
now a **joint** retrieval of the state `θ = [r_e(s-nodes…), r_base, τ_bot]` with all three made
leak-free. (`src/retrieval_oe.py`; `tests/supplementary/joint_dofs_experiment.py`,
`joint_osse_retrieval.py`, `smoke_joint_retrieval.py`.)

**(a) Normalized-depth parameterisation `s = τ/τ_bot ∈ [0,1]`  [SETTLED — INVARIANT for joint τ_bot].**
The r_e nodes live at fixed *normalized* depths `s`, base at `s=1`; absolute positions `s·τ_bot`
**stretch/compress with the retrieved τ_bot**. This is required for joint τ_bot retrieval:
absolute-τ nodes placed at a (thick) first-guess τ_bot strand below a thin cloud's base, and a
monotonicity guard then forbids τ_bot from dropping past them — the GN diverged (τ_bot→−5) or hit
the Kvaerno5 max-steps controller error on out-of-range optics. In `s` there is no crossing and no
guard. `re5-linear` in `s` ≡ in `τ` (linear rescale), so the adiabatic prior/lever is unchanged.
The single lever `RetrievalForward._re_of_tau` does the `τ→s` map; the grid (`OEResult.tau_nodes`,
`select_retrieval_grid`) is in `s`. **GN iterates are clamped** to the optics-table support and
physical τ_bot bounds (`_clamp_state`, projected GN; DESIGN §8 bounded-state forward). Verified
(smoke): the thin retrieval now converges from the climatology guess τ_bot 10.6→1.42 (truth 1.21),
the case that previously blew up.

**(b) Leak-free priors — broad (Option 2) headline, LOO climatology (Option 1) fallback  [SETTLED].**
The first guess and prior means come from a **leave-one-flight-out** VOCALS climatology
(`vocals_io.vocals_climatology`, robust median+MAD with a physical τ_bot filter — the CDP
integration yields artefact τ_bot up to ~1585 that wreck mean/σ) — never the target profile or its
flight. Headline prior = **broad weakly-informative** (`make_joint_prior` with generic marine-Sc
means r_top≈10, r_base≈12, τ_bot prior σ≈0.5·mean): the data sets τ_bot and the upper-cloud r_e,
the prior fills the shielded base. Fallback = the tighter LOO climatology
(`make_climatology_prior`) if the broad prior degrades the fit. Operationally: Option 2 needs no
scene-specific stats (generalises to any new scene); Option 1's analogue is a pre-built
regional/seasonal climatology that excludes the current observation. `r_base` joins the correlated
r_e block as the deepest node (it *is* r_e at the base); τ_bot is an independent broad scalar dim.

**(c) Information content — DOFS (linearized at truth; `joint_dofs_results.json`)  [SETTLED].**
Per-component `DOFS = tr(A) = Σ diag(A)` (`dofs_by_component`):

| cloud | bands | fixed-anchor | joint broad (prof/rbase/τbot) | joint clim |
|---|---|---|---|---|
| thin RF11 τ=1.2 | 1.24,2.13 | 1.49 | 2.59 (1.30/0.28/**1.00**) | 2.11 |
| thick RF03 τ=23 | 1.24,1.64,2.13 | 2.94 | 3.90 (2.85/0.09/**0.96**) | 2.96 |

- **τ_bot is essentially fully measured** (DOFS≈0.9–1.0; 1σ collapses prior~5 → post 0.06–0.07 thin,
  ~1–2 thick) and making it unknown costs the *profile* very little (1.49→1.30 thin, 2.94→2.85
  thick). **r_base is strongly depth-dependent** (NOT uniformly shielded): for the *thin* cloud the
  base is optically visible and the full-OSSE posterior removes ~90 % of its prior variance (DOFS
  ≈0.85, σ 8→2.6 µm, the retrieval pulls it well off the prior); for the *thick* cloud it is
  radiatively **shielded** (only ~30 % removed, DOFS ≈0.17, posterior ≈ prior). The at-truth DOFS
  experiment (0.28 thin / 0.09 thick) understates the thin case vs the at-retrieval posterior — the
  Jacobian ∂y/∂r_base is larger off the truth — but both agree thin ≫ thick. See the blind-spot
  bullet below.
- **DOFS is prior-dependent**: broad > climatology always (2.59>2.11, 3.90>2.96) — the tighter prior
  pre-supplies information, lowering the measurement's *marginal* DOFS. Report DOFS vs prior, not a
  single number.
- **The shielded base is a physical blind spot, not just a method limit.** A *sub-adiabatic* base
  (r_base **below** the upper-cloud r_e — the adiabatic prior assumes the opposite, r_base>r_top,
  since r_e∝τ^{1/5} grows downward) is the microphysical signature of sub-saturation /
  **re-evaporation** near cloud base (sub-cloud drizzle evaporation, entrainment mixing down). It is
  exactly the signal a reflected-sunlight retrieval cannot see for thick cloud: r_base DOFS collapses
  with optical thickness (0.28 thin → **0.09** thick here; 0.85→0.33 in the full OSSE), so for thick
  Sc the base radius — and any evaporation signature — is **unfalsifiable from reflectance** (you get
  the prior back). Marginally detectable only for optically thin cloud where photons reach the base.
  *Consequence:* base re-evaporation needs an active complement (cloud radar/lidar) or thin scenes;
  and the r_base *prior* should not be assumed adiabatic for scenes where evaporation is plausible —
  though for thick cloud that prior choice is, by construction, untestable by the measurement.
  **Empirical confirmation** (`subadiabatic_thin_retrieval.py` → notebook §13): starting from an
  adiabatic prior (r_base mean 12 > r_top), the joint retrieval **recovers a sub-adiabatic downturn
  only where the cloud is optically visible** — RF14 (τ≈2.5, decline through the upper cloud)
  captured 48 % of the true 3.4 µm drop (r_base 12→7.6, truth 6.0); RF05 (τ≈2.9, drop confined to the
  bottom ~40 %) **missed entirely** (r_base pushed to 9.6), despite near-equal τ. The discriminator
  is *where the structure sits relative to photon penetration*, not τ alone. The VOCALS cleaned-data
  floor is τ≈1.21 (RF11; the §11 headline, where r_base was pulled 12→7.7 toward truth 6.3, ~90 %
  measured), so τ≲1 — where a near-bottom drop might be fully recoverable — cannot be tested here.

**(d) Band reassessment — keep the weak-absorber, a conservative band is NOT worth it  [SETTLED].**
Swapping 1.24 µm (weak absorber) → 0.86 µm (conservative) *hurts*: thick τ_bot 1σ nearly doubles
(0.99→1.87) because a conservative band **saturates** for thick clouds (∂R/∂τ→0 as τ→∞) whereas a
weak absorber stays τ-sensitive, and 0.86 carries no r_e-gradient information that 1.24 has. The
bispectral "conservative band for τ" intuition does not pay because the joint inversion already
nails τ_bot. Keep `[1.24, (1.64), 2.13]`.

**(e) Joint > two-stage  [SETTLED — joint headline].** Because the joint inversion already
determines τ_bot tightly (c) **and** propagates its uncertainty into the r_e posterior, it gets the
two-stage robustness without the unquantified stage-1→stage-2 error leak. Two-stage is kept only as
a robustness fallback. A conservative band is *included in* the joint measurement rather than run as
a separate stage.

**(f) Adaptive node count — DOFS *is* a robust info measure here  [SETTLED].**
`auto_k_active` offers two estimators — `round(factor·DOFS)` and the noise-aware whitened-QRCP
filter factor `f_i=r_i²/(1+r_i²)` (not via DOFS), with `Σf_i ≈ DOFS` as a built-in cross-check and
DOFS-robustness probe. **OSSE verdict** (`joint_osse_results.json`, profile-only pool Jacobian at
the first guess): the independent estimators agree — `Σf_i` tracks DOFS to ≈5 % (thin 3.01 vs 2.88;
thick 2.53 vs 2.39), so **DOFS is an accurate, robust information measure for this problem**, not an
artefact of one estimator. The two node counts agree on the thin case (filter→4, dofs→4) and differ
by one on the thick (filter→3, dofs→4): the filter count is the more conservative — a thick cloud's
deeper nodes are redundant (saturation), and `f_i` collapses sharply (0.97, 0.87, 0.47, 0.21, …)
where DOFS's soft sum still counts the half-resolved 4th. Both auto values are **below** the
hardcoded production `k_active` (4 thin / 5 thick), i.e. the manual grids slightly over-resolve;
this is harmless because the extra nodes are prior-pulled (they land in the shielded base) but the
filter estimator is the principled choice.

**Implemented 2026-06-19 (filter wired in); threshold = Rodgers crossover 0.5 (updated 2026-06-20).**
`auto_k_active` is now **filter-only** (the `dofs` estimator and `factor` removed);
`select_retrieval_grid(k_active=None)` sets the count from the **full ODE grid** (the 20-node
resample — which *manufactured* collinear pool columns for thin clouds — is gone); the cut is exposed
as `filter_threshold` in data-fraction units. **Default `filter_threshold=0.5`** — Rodgers' **data/prior
crossover**: `f_i ≥ 0.5 ⇔ r_i ≥ 1 ⇔ SNR ≥ 1` on the noise-whitened Jacobian
`K̃ = Se^(−1/2)·K·diag(σ_prior)`, the boundary above which a direction is *measured* rather than
prior-dominated (Rodgers 2000 §2.4, `d_s = tr(A) = Σ λ_i²/(1+λ_i²)`; the filter factor is the
per-direction "fraction from data" — EODG retrieval-theory notes §8.5.2). Two reasons it is the right
default, *both* reinforcing the conservatism goal: **(i)** it keeps deviation-from-the-adiabatic-prior
to what the **data** license — a sub-0.5 node's "deviation" is prior/noise-driven (the §13
sub-adiabatic **overfit**), not a detection, so 0.5 protects the *credibility* of the method's
distinguishing feature rather than blunting it; **(ii)** being the SNR=1 crossover it is **invariant to
the noise level** (no re-tune when `Se` changes), unlike a tuned absolute cut.

**Supersedes the old `0.25`** (a *3 %*-noise tune). When the measurement noise was grounded to the
PACE-OCI **2 %** model (§12), 0.25 over-resolved → severe §13 overfit. A 0.25-vs-0.5 sweep at 2 %
(`tests/supplementary/sweep_threshold_2pct.py`) settled it: **thin unchanged** (k=5 both); **the
shielded RF10 §13 overfit fixed** (drop-cap 172 %→58 %, RMSE 0.69→0.52 — the culprit node `f=0.482`
is prior-dominated); **thick costs only +0.1 µm** (RMSE 0.81→0.91, *both* χ²≪1 ⇒ well-fit, no
structural under-resolution, and the dropped node sits *at* `f≈0.48`). That +0.1 µm is within the
retrieval's own uncertainty and comes from a coin-flip-at-the-crossover node — **not** "strong evidence
to deviate" from the principled default, so **no SIC escalation**. The old +51 %-RMSE RF08 concern that
motivated 0.25 was a 3 %-noise artefact; at 2 % the borderline node's `f` rises and the cost collapses.
(`tune_filter_threshold.py`/`thick_sweep2.py` were the superseded 3 %-noise sweeps.)

DOFS **left the selection path entirely** — it is now an information-content diagnostic only
(`src/info_content.py`, full ODE grid; `posterior_diagnostics` keeps per-retrieval DOFS+SIC). The
**SIC-peak selector** (SIC peaks at the RMSE-optimal k) is the documented **escalation** — used only if
a genuinely structured cloud is ever shown to be *structurally* under-resolved at 0.5 (χ² above the
noise floor), paid for then rather than by default.

**(g) Interpolation lever — second-order; re5-linear kept  [SETTLED].**
re5-linear vs plain-linear `_re_of_tau`, same data, same grid (model comparison):

| cloud | re5-linear (RMSE / ‖y−F‖ / DOFS) | linear (RMSE / ‖y−F‖ / DOFS) |
|---|---|---|
| thin RF11 | 0.53 µm / 9.9e-4 / 3.64 | 0.60 µm / 2.8e-3 / 3.11 |
| thick RF03 | 0.87 µm / 1.7e-2 / 3.43 | 0.96 µm / 1.1e-2 / 3.82 |

re5-linear wins the profile RMSE in both regimes, but the gap is **< 0.1 µm** — well inside the
retrieval's own uncertainty. **The user's intuition holds: for reasonable low-order monotonic
interpolation the choice is a minor, second-order lever, not a re-mesh-instability driver.**
re5-linear is kept as default — it is physically motivated (adiabatic `r_e ∝ τ^{1/5}`, so the
interpolant is linear in the natural variable) and marginally more accurate; the difference does not
warrant the extra machinery of a higher-order scheme.

**(h) Re-meshing — n_outer=1 is the right default; re-mesh did not help  [SETTLED].**
On the thin case, lagged re-meshing (`n_outer=2`) moved the deepest node (s 0.49→0.37) but the
profile RMSE got *worse* (0.53→0.70 µm) while the fit was unchanged (‖y−F‖≈1e-3) — re-pivoting a
well-fit but correlated node basis just churns placement (the "re-mesh instability" of OUTSTANDING
§G, now mild under normalized depth). The χ²-gate (`remesh_if_chi2_red_gt`) is therefore the correct
policy: **re-mesh only on a persistently high loss (structural misfit), freeze the grid otherwise.**

*Why "noise floor" ⇔ "well-fit" (the χ²-gate rationale, derived).* The gated quantity is the
**whitened** data misfit `χ²_red = ‖y−F(x̂)‖²_{Se⁻¹}/m = (1/m)·Σ_i (y_i−F_i)²/σ_{ε,i}²` — every
residual measured in units of *its own* measurement-noise σ. If the fit has captured all the
*systematic* signal, the residual is just noise, `r = y−F(x̂) ≈ ε ∼ N(0,Se)`; whitening gives
`Se^(−1/2)·r ∼ N(0,I_m)`, a sum of m squared unit-normals, so `E[‖r‖²_{Se⁻¹}] ≈ m` (≈ m−n for n
fitted parameters) and hence **`χ²_red ≈ 1`**. So `χ²_red ≲ 1` means the residuals are statistically
indistinguishable from noise — *nothing systematic is left for any node placement to explain*, which
is exactly what "well-fit" means (re-meshing cannot help, so freeze the grid); `χ²_red ≫ thr` (=2.0)
means the residual is many σ beyond what noise explains ⇒ a genuine **structural misfit** the current
parameterisation cannot reproduce ⇒ re-mesh. **Caveat for our OSSE:** `osse_observation` is
**noiseless** by default, so the residual floor is *not* a noise draw but **representation error**
(the few nodes + r_e⁵-interpolant vs the dense continuous truth), scored against the *assumed*
`Se=(0.03·R)²`. `χ²_red ≤ thr` then reads "the node model reproduces the synthetic data to within the
instrument's ~3 % precision" — the same gate, floored by parameterisation rather than a noise
realisation. *(Standard Rodgers OE goodness-of-fit; recorded here because the inline policy uses
"noise floor"/"structural misfit" as shorthand without the derivation.)*

For these well-fit VOCALS retrievals the gate keeps `n_outer` effectively 1 (re-mesh only on structural / high-loss misfit).

**Implemented 2026-06-19 (progressive escalation).** `n_outer` → **`max_n_outer`**, a capped
*escalation ladder*: **1** = off (select-once), **2** = re-mesh at *fixed* node count (placement
only), **3** = escalate to a *changed* count (the filter re-decides). It fires only on the **"both"
trigger** — reduced χ² > `remesh_if_chi2_red_gt` (default 2.0) **and** the re-selected grid would
actually move; the (recompiling) re-selection runs only at an *enabled* tier, while at a tier beyond
`max_n_outer` the code **warns on χ² alone** (a `RemeshWarning`, no wasted recompile) so a select-once
user still learns the fit wanted more. Default `max_n_outer=2`. Cost rationale (it's a *last resort*):
paying the 20-node-pool tax on every retrieval just to keep re-mesh recompile-free was the wrong
trade once the χ²-gate made re-mesh rare — so re-mesh now pays its own recompile when (rarely) invoked.

---

## 11. Prior design — grounded in VOCALS-REx data + literature + information content  [SETTLED — population-confirmed 2026-06-23]

The earlier broad prior was hand-picked and **inverted** (r_top=10 < r_base=12 — the
*converse* of an adiabatic profile, where r_e is largest at the top). Replaced by
`retrieval_oe.make_marine_sc_prior`, designed from a prior-sensitivity study
(`tests/supplementary/prior_investigation.py` → `docs/prior_investigation_results.json`)
plus the VOCALS-REx empirical distributions and the King/Vukićević retrieval literature
(AMT 18, 5299, 2025; the same 3-parameter r_top/r_base/τ adiabatic model we use).

**(a) Verified mechanism — how a shielded r_base still moves (resolves the "§12 is
suspiciously good" puzzle).** Linearized at truth, the retrieval is
`x̂ = x_a + A·(x_truth − x_a)` (A = averaging kernel). For thick RF03, A_base = **0.06**
(≈ no *direct* information), yet r_base moves off its prior. The driver is the **prior
off-diagonal correlation**, not measurement: with the same inverted prior (base=12), a
**diagonal** prior pins r_base at **11.44** (A_base→0.01), while the **correlated** prior
moves it to 9.72 — r_base is reconstructed by smooth extrapolation from the well-measured
upper-cloud nodes. So DOFS_base being ~0 yet r_base "retrieved well" is not measurement;
it is the prior's smoothness transferring the observable r_top downward. Its honest
posterior σ stays ≈ the prior σ. *Consequence:* §12's good r_base is luck (RF03's base
continues the visible trend); a true near-base anomaly would be missed (RF05, §13).

**(b) The three observability findings (averaging kernel at truth; thin RF11 / thick RF03).**

| param | A (thick) | prior-following | verdict |
|---|---|---|---|
| **r_top** | **0.98** | ~0 (x̂_top=11.4 for prior 8→16) | observable ⇒ prior barely matters |
| **r_base** | **0.06** | ~0.8 (x̂_base 3.6→8.4 for prior 4→10) | shielded ⇒ prior *is* the answer |
| **τ_bot** | **1.00** | 0 (x̂=23.2 for prior mean 5→25, σ 2→40) | fully measured ⇒ prior irrelevant |

(Thin RF11: r_top only ~0.25 vs thick 0.95 — the thin cloud top is genuinely harder to
resolve. The cause is **angular under-sampling, not stream count or single scattering**
(`thin_top_resolution.py`, `stream_view_thickthin.py`): A_top is *flat* in NQuad 16→32
(more streams don't help) but rises with **view-angle count/obliquity** at fixed NQuad —
thin 3-moderate 0.25 → 3-oblique 0.29 → 8-dense **0.39**; thick saturates at ~0.95 for any
choice. The radiance is dominated by the **node-bound multiple-scatter field** (the TMS
single-scatter *correction* is only ~3 % for both clouds), so each view angle is a
projection of the N node radiances: with n_view < N the node info is under-sampled, and
denser/oblique views recover more of ∂u_nodes/∂r_top up to the N-node ceiling. **Practical
upshot: thin retrievals want *more view angles* (dense angular sampling), not more streams;
thick is saturated on both.** τ_bot and r_base behave as for thick.)

**RULE (enforced as a warning in `RetrievalForward`): use ≥ NQuad//2 view angles.** Off-node
radiances are interpolations of the `N = NQuad//2` upwelling quadrature-node radiances (plus
the ~3 % per-angle TMS term), so fewer than N view angles under-samples that field and leaves
retrievable information unused. The notebook uses 8 views at NQuad=16. *Physical basis for the
thin/thick split:* multiple scattering drives a thick cloud's interior toward the **diffusion
limit**; ToA is never itself in the diffusion domain, but the upwelling ToA radiance is *fed by*
diffusion from below — hence smooth, moment-washed, node-bound, and stream-insensitive. A thin
cloud lacks that diffusive simplification, so its sparse top information is spread across the
angular field and must be sampled densely.

**(b′) Report DOFS *and* SIC (`posterior_diagnostics` now returns both).** The thin/thick result
exposes an inconsistency a single number hides: a *thin* cloud has little depth to vary (**few
DOF**) yet each feature is sharply measured and it *benefits from more streams/angles*; a *thick*
cloud varies more (**more DOF**) but diffusion caps the per-stream information so it *gains little
from more streams*. `DOFS = tr(A)` counts independent *features*; `SIC = ½ log₂|Sa Ŝ⁻¹|` (bits) is
the *magnitude* of information (how sharply each feature is pinned). Carry both — DOFS answers "how
many things", SIC answers "how well".

**(c) VOCALS-REx empirical distributions (n=125, τ_bot∈[0.3,60]).** r_top 9.5 ± 2.3 µm
(MAD; p95 14.0 — the ~14–15 µm **drizzle threshold** is the physical upper bound, big drops
fall out); r_base 5.7 ± **1.4** µm (MAD) — *narrower in absolute terms* than r_top, with a
tight robust core but a heavy tail (std 2.0 ≫ MAD 1.4 = the sub-saturation/drizzle
exceptions); τ_bot median 9.6, **MAD 9.5 ≈ median** (uninformative). 95 % of profiles have
r_top > r_base (adiabatic direction); corr(r_top, r_base) = 0.58; median ratio
r_base/r_top = 0.60 (literature 0.70).

**(d) The design — tight where the measurement is blind, loose where it is strong**
(`make_marine_sc_prior`): r_base **tight** (σ≈1.4, the microphysical core) and
**adiabatic-coupled** (mean = 0.65·r_top, clipped < r_top) — because it is prior-dominated,
a vague r_base prior would give a vague/wrong r_base; r_top **moderate** (σ≈2.3, effective
≲15 drizzle cap) — observable, so loose is fine; τ_bot **uninformative** (σ≈mean) — fully
data-determined. This is the optimal-estimation-principled prior and it is what the data,
literature, and sensitivity experiments jointly support. Sub-saturation profiles are the
rare tail we deliberately do not encode (capturing one is a bonus, §13).

**(e) The four prior builders — catalog** (`retrieval_oe.py`). A small hierarchy:
`make_adiabatic_prior` builds one correlated r_e block; `make_joint_prior` assembles the full
state from it; the two **grounded production priors** wrap `make_joint_prior` with data-set
numbers. All share the same Sₐ structure — an exponentially-correlated Gaussian
`Sₐ[i,j] = σ_i σ_j · exp(−|Δτ| / ℓ)` over the r_e + r_base block (correlation length `ℓ` default
`τ_bot/2`, i.e. 0.5 in normalized depth) with `τ_bot` appended **block-diagonal** (droplet size and
optical thickness are different physical quantities — no asserted cross-correlation). The mean is the
adiabatic r_e⁵-linear law `r_e ∝ τ^(1/5)`.

| builder | role | mean from | σ from (defaults) | leak-free? |
|---|---|---|---|---|
| `make_adiabatic_prior` | base block: r_e⁵-linear mean + correlated Sₐ (single r_e block) | adiabatic curve from `r_base` + `r_top_prior` | linear σ_top→σ_base (def **3.0 / 1.5**); exp-correlated, ℓ=τ_bot/2 | **caller-dependent** — leak-free iff `r_top_prior` is climatological; **§5 passes the truth → idealized, *not* leak-free** |
| `make_joint_prior` | assembles `[r_e nodes, r_base, τ_bot]`; r_base = deepest node (s=1); τ_bot block-diagonal | generic/clim `r_top_prior`, `r_base_prior`, `tau_bot_prior` | σ_top / σ_base / σ_τbot (def **5.0 / 2.0 / 0.5·τ_bot**) | yes when fed non-truth means |
| `make_marine_sc_prior` | **Option 2 — generic grounded marine-Sc** (production default) | `r_top_prior` (clim/MODIS); `r_base = 0.65·r_top` (adiabatic ratio, clipped < r_top); `tau_bot_prior` | σ_top 2.5, σ_base **1.5** (≈ VOCALS MAD 1.4), σ_τbot ~100 % (= τ_bot) | yes — fed climatological `r_top_prior`/`tau_bot_prior` |
| `make_climatology_prior` | **Option 1 — LOO VOCALS climatology** (strongest; the IC-profiling prior) | LOO ensemble **means** (r_top, r_base, τ_bot) | LOO ensemble **robust spreads** (1.4826·MAD ≈ 2.7 / 1.4 / 9.5) | yes — `vocals_climatology(exclude_flight=…)` never sees the truth's flight |

Routing: `make_marine_sc_prior` and `make_climatology_prior` both call `make_joint_prior` (which calls
`make_adiabatic_prior` for the r_e+r_base block), passing σ_base **explicitly** — so the base
builders' σ defaults apply only to *direct* callers.

**(f) Population confirmation (all-125 IC, NQuad=48; 2026-06-23).** The (a)/(b) mechanism — argued
from single profiles RF03/RF11 — now holds across **all 125 VOCALS profiles**
(`info_content_mechanism_all125.json`): under a *diagonal* Sₐ the full-view angular field directly
reaches only mid-cloud (deepest s with data-fraction≥0.5 ≈ **0.46 / 0.39 / 0.23** for thin/mid/thick),
while the *correlated* prior extends the reach to the base (≈ **1.00 / 0.99 / 0.89**) — the prior adds
~0.6 in normalized depth uniformly, and the flux-albedo's vertical localization is *entirely*
prior-mediated (direct reach ≈ 0). So the "tight where the measurement is blind" design (d) is
vindicated population-wide: the deep r_base is reconstructed by the prior's smoothness from the
observable upper cloud, not measured. (The prior-viability *ladder* of (e) — uninformative ≫ marine_sc
≈ climatology — was **not** re-run at NQuad=48; it remains the NQuad=32 Stage-1 pilot result.)

**(g) Hyperparameter scoping — prior correlation length ℓ (deferred, 2026-06-23).** The smoothness
term's correlation length is **ℓ = τ_bot/2 ≈ 0.5 in normalized depth** (`make_adiabatic_prior` default,
§(e)). It is a **Bayesian–Tikhonov smoothness hyperparameter, not an empirically-cited correlation
length** — the exp-correlated-smoothness *form* is standard OE practice (Rodgers 2000, *Inverse
Methods*), but the ℓ value is our design choice (no citation). It **is** an IC lever (longer ℓ →
smoother → fewer DOF; shorter → toward the measurement-rank ceiling). We **intentionally do not tune
it**: the IC profiling reports at the default ℓ, and a dedicated ℓ-sensitivity sweep is left to future
work (others may pick it up). The robustness runs partially bracket it — set i (the weak σ≈10-µm **diagonal**
prior, ℓ→0) vs the LOO prior (ℓ=0.5) — and an `IC_CORR_LENGTH` knob can isolate it cleanly if revisited.

**(h) Pending §11 rewrite (deferred, 2026-06-23).** (d)/(e) above still describe `make_marine_sc_prior`
(now **retired as the IC default**) and **mis-cite** the adiabatic r_base/r_top≈0.7 ratio as
"King/Vukićević AMT-2025" — that paper is **Buggee & Pilewskie (2025)**, doi:10.5194/amt-18-5299-2025
(the "King" was King & Vaughan = KV2012; "Vukićević" leaked in from CPV2012 — a garbled mash-up). The
IC profiling now uses the **LOO `make_climatology_prior`** as default; the weak-prior robustness rung
uses **σ≈10 µm on r_e (King & Vaughan 2012)** and a representative **r_top≈10 µm (Painemal & Zuidema
2011, doi:10.1029/2011JD016155)**. Full (d)/(e) rewrite to follow.

**(i) IC linearization point — recorded (2026-06-24).** The flexible-node retrieval state **is** the
5 `s_ref` nodes (+ r_base, τ_bot), so a linearization point must live in that basis. The "truth"
linearization is therefore the in-situ VOCALS profile **projected onto the 5-node basis** (`np.interp`
at s=[0,.2,.4,.6,.8]) then re5-interpolated to the ODE grid — **not** the literal (wiggly) aircraft
profile, which isn't in the state space. For near-uniform VOCALS Sc this 5-node sample is low-gradient
(e.g. RF11 idx89: raw 7.8–9.5 µm, 26 pts → sample 8.6–9.3 µm). **Headline = the LOO prior-MEAN
linearization (set iv):** the prior mean is the smooth 3-parameter r_e⁵-adiabatic curve
(`make_adiabatic_prior`: x_a = (r_base⁵ + (r_top⁵−r_base⁵)(1−s))^{1/5}, from the LOO r_top/r_base
means) — a clean, leak-free, population-representative, Rodgers-standard a-priori reference. The truth
(ii), realization (v), and ℓ=τ_bot (vi) linearizations are the **robustness ensemble** (IC invariant
to the lin point). **Realizations are demoted from the headline** (kept in the ensemble). **Construction
corrected 2026-06-24** (`roe.draw_climatology_realization`): a realization is a physical 3-param
r_e⁵-adiabatic draw — r_top, r_base from the LOO marginals (independent, rejection-sampled to
25≥r_top>r_base≥2), τ_bot = the truth (IC profiling) or LOO-drawn (full-retrieval synthetic truths).
This **replaces** the earlier per-node S_a cholesky draw, which gave *unphysically non-monotonic*
profiles: the prior *mean* is adiabatic, but its *covariance* is per-node correlated-Gaussian
(σ_iσ_j exp(−|Δτ|/ℓ), L684), so mean + per-node noise wanders off adiabatic — the cause of set v's
+22 % ODE steps, and why a realization is a less clean reference than the mean. Only set v (the draw
array) re-runs with the corrected construction; truth/priormean/mechanism unchanged.

---

## 12. Measurement-noise model — three-term σ(ρ), OCI-SWIR calibration-relative; default noiseless  [SETTLED — shot term + HARP2/polarized open in OUTSTANDING K]

**What "noise" means here (the conceptual fix).** The retrieval noise is **measurement noise
on the ToA radiances** — the instrument noise of the spaceborne radiometer — **not** uncertainty
in the VOCALS-REx in-situ truth. VOCALS is only the *ground-truth state* (it could equally be a
synthetic or GCM profile); what is noisy is the simulated reflectance the instrument would report.
So the noise model is grounded in the **PACE instrument specs**, not in the VOCALS distributions
(those ground the *prior*, §11 — a different object). The observable is the bidirectional
reflectance factor `ρ = π u / (μ0 I0)` (`RetrievalForward.forward`).

**The three-term σ(ρ) (general form).** Three independent error sources added in quadrature:

```
σ(ρ) = sqrt( (k_cal·ρ)²            # calibration / radiometric accuracy — flat-relative, does NOT average down
             + ρ·ρ_ref / SNR_ref²  # photon shot noise ∝ √ρ — relative size √(ρ_ref/ρ)/SNR_ref shrinks as ρ↑
             + floor² )            # read/dark/quantization — additive, signal-independent (dark pixels only)
```

`k_cal·ρ` is a *relative* gain uncertainty (a 2 % error is 2 % on any pixel); the shot term is
anchored at an instrument SNR_ref quoted at a reference reflectance ρ_ref (the L_typ analogue),
with `SNR(ρ) = SNR_ref·√(ρ/ρ_ref)`; the floor dominates only near zero signal.

**OCI-SWIR population = "Option B" (calibration-relative); shot term wired-but-off = "Option A"
deferred.** For our bands (1.24/1.64/2.13 µm = OCI SWIR) clouds are *bright*, so SNR is high and the
budget is **calibration-dominated** — the shot term is a small correction and A ≈ B in regime. We set
`k_cal` from the documented OCI/HARP2 **radiometric accuracy 1–3 %** (PACE MRD §3.7 absolute-gain
uncertainty; `oci_swir` default 0.02) and leave the shot term **off** (`snr_ref=∞`) because OCI's
SNR-at-L_typ table could not be cleanly sourced (the MRD tables are embedded *images*; the SNR spec is
in an external `.xlsx`; the conversion to reflectance further needs per-band F₀ + geometry). The
function keeps the shot coefficients exposed, so dropping in verified `snr_ref`/`ρ_ref` switches A on
with **no refactor** — tracked in [OUTSTANDING K](./OUTSTANDING.md).

**Honest caveat — calibration error is systematic, not random.** The `k_cal·ρ` term models absolute
gain uncertainty, which is *correlated across pixels in a scene* (a bias), whereas a diagonal `Se`
treats it as independent per-observation noise. For the **single-column** OSSE this is the standard
pragmatic choice (it sets the believable misfit scale and the χ²-gate floor, §10h); a multi-pixel /
scene retrieval should revisit it (off-diagonal `Se`) — see OUTSTANDING K.

**Default NOISELESS; the model is still used for `Se`.** `osse_observation(..., noise=None)` adds
nothing (the OSSE decision, §10b). A noise *realization* is opt-in: pass a `NoiseModel` (drawn via
`sample`, band-major over `fwd.n_bands`) or an explicit per-σ. Independently, the *assumed* covariance
the retrieval inverts is `Se = diag(σ²)` built by `make_Se(fwd, y, model)` — needed for weighting and
the χ²-gate **even with no perturbation**. The grounded `oci_swir()` model is the intended replacement
for the historical hand-picked `Se = diag((0.03·max(|y|,0.02))²)` (kept reproducible as
`generic_relative`).

**Scope (user-set 2026-06-19): OCI-SWIR intensity only.** HARP2 (VIS 0.44–0.87 µm multi-angle; its
headline **0.5 % DoLP** spec) cannot measure the SWIR retrieval bands; HARP2 and **polarized / DoLP
noise** belong to the polarized-cloudbow observable and are deferred (OUTSTANDING §I, §K).

**Posterior uncertainty budget — what `Ŝ` captures (transparency; deliberately *not* an OUTSTANDING
item).** The reported posterior 1σ is the Rodgers `Ŝ = (KᵀSe⁻¹K + Sa⁻¹)⁻¹` (`posterior_diagnostics`),
a function of only `Sa` (prior, §11) and `Se` (this section). It **captures** measurement noise, the
prior, and the finite-resolution *variance* — under-determined directions relax to prior σ, so a
shielded node honestly reports ≈prior σ. It does **not** capture two *bias* classes (sources of
uncertainty are endless; these are out of scope for a 1D OSSE and not viable with current
capabilities, so they are recorded here for honesty, not tracked as open work):
- **Imposed-shape bias (representation/state):** the adiabatic `r_e∝τ^{1/5}` inter-node interpolant
  (`_re_of_tau`), the adiabatic prior *mean* on unresolved directions, and the monotone basis's
  inability to express non-monotone structure (drizzle minimum, decoupled layers). A bias where
  reality is non-adiabatic, invisible to σ — exactly what §13 exercises. (The node *placement* is
  physics-driven via the ODE grid, so this is minimised by design; what remains is the fill-shape and
  the prior lean.)
- **Forward-model / physics error:** 1D-vs-3D (independent-pixel / plane-parallel bias), the Mie
  size-distribution idealization (single-mode gamma, fixed v_e), neglected gas/aerosol, the Lambertian
  ocean (§9), and numerical truncation (NQuad / NFourier / NLeg / float32). **All cancel in the OSSE by
  inverse crime** — the same forward generates and inverts the synthetic radiance — so they contribute
  **zero here** and switch on only against *real* OCI/HARP2 radiances. VOCALS-REx in-situ error is
  likewise out: the profile is the OSSE *truth* (exact by definition), and where it feeds the prior its
  instrument error is swamped by geophysical spread (§11c). The notebook mirrors this in §11b.

*(Implemented in `src/noise_model.py` — `NoiseModel` (three-term `sigma`/`Se`/`sample`), presets
`oci_swir` / `generic_relative` — and wired in `src/retrieval_oe.py` (`osse_observation` NoiseModel
dispatch, `make_Se`). Verified: `tests/supplementary/check_noise_model.py` (shot-off↔B, shot
calibration, per-band band-major coeffs, Se=diag(σ²), sample statistics, bright-cloud shot
subdominance, legacy match).)*

---

## 13. Performance — single-column latency-bound; batch columns on the GPU  [NOTE — empirical, 2026-06-08]

Cached *single-column* execution is dominated by **many sequential tiny matmuls** (NFourier modes × 2
sweeps × ~35 adaptive steps × 5 ESDIRK stages on N×N, N ≤ 8/16) — kernel-launch-latency-bound, so the
GPU is **not** faster than CPU per column (warm single-mode solve: CPU 2.1 ms vs GPU 28.9 ms — ~14×
*slower* on a T4). The analysis is **GPU-agnostic**: the binding costs (host-side XLA compile,
per-kernel launch latency) are set off-device.

**The retrieval is embarrassingly parallel across columns**, so `jax.vmap` over a batch turns the tiny
matmuls into device-filling batched matmuls. Warm per-column time vs batch size B (µs/column,
`tests/supplementary/batch_columns.py`):

| B | 1 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| **GPU** | 30592 | 2233 | **555** | 155 | 50 | **16** |
| **CPU** | 1908 | 1021 | **959** | 854 | 829 | 846 |

CPU per-column is ~flat (limited parallelism); GPU collapses ~1900× once a batch hides the latency.
**Crossover at B≈64; at B=4096 the GPU is ~53× faster per column.** So "GPU is not a lever" holds *per
single column*, but the right retrieval architecture is **jit (§7) + vmap a batch of columns onto the
GPU** — the batch, not the device, delivers the latency hiding. Real speed levers: fewer Fourier modes,
fewer adaptive steps (looser `tol`), and column batching — not the device itself.

*(Relocated from the former OUTSTANDING §D — empirical detail in `tests/supplementary/profile_solver.py`
and `batch_columns.py`, 2026-06-08, Tesla T4 vs CPU.)*

**CPU data-parallelism over columns — process-level, taskset-pinned (2026-06-21).** When no GPU is at
hand (the Stage-1 IC sweeps ran CPU-only), the moderate many-column workload is still well served — but
**not by giving one job more cores**. A single XLA process self-parallelizes one Jacobian to only
~10/16 cores (1028 % CPU observed), and the latency-bound tiny-matrix work shows **no positive
core-scaling**: a warm Jacobian eval took **287 s on 16 cores vs 215 s on 4 cores** (16 is *slower* —
XLA thread-pool over-subscription/spin on the N×N≤24 matrices). So the lever is **concurrency, not
cores-per-job**: run several columns as separate **taskset-pinned processes** confined to disjoint core
subsets (`tests/supplementary/_ic_parallel.py`: N workers × C cores, N·C = 16). 4×4 on the 16-core box
gives ≈4× throughput (each task ≈ as fast on 4 cores as on 16) at ~3 GB RAM total; the Stage-1 sweeps
ran this way (NQuad sweep 20 tasks / 84 min; population ensemble 10 / 93 min). For *all*-VOCALS a SLURM
**job array** (one profile/task, 4 cores each) is the same pattern at cluster scale. So: **GPU-vmap for
huge batches (above); taskset-pinned process-parallel on CPU for moderate many-profile sweeps** — both
exploit embarrassing parallelism *across* columns, never single-column core-scaling.

---

## 14. Information-content profiling (DEFINITIVE) — spectral verification + angular novelty  [SETTLED — 2026-06-24]

The definitive Stage-1 IC run (supersedes the pilot §15 / `info_content_stage1.py`). Reports Rodgers
**DOFS = tr(A) = Σ sᵢ²/(1+sᵢ²)** and Shannon **SIC = ½ Σ log₂(1+sᵢ²)** of the minimally-constrained
free-node **r_e(τ)** profile over all 125 physical VOCALS-REx profiles, NQuad=48, μ₀=0.9. Two legs:
the **spectral axis verifies the literature** (it *saturates*); the **angular axis is the novel
contribution** (it adds on top of the saturated spectral baseline and reaches the shielded base).

**Reframe / two legs.** The spectral leg reproduces the hyperspectral-albedo saturation of
**Coddington–Pilewskie–Vukićević 2012** (JGR, doi:10.1029/2011JD016771) and the bispectral / depth-
graded SWIR lineage (Nakajima–King 1990; Platnick 2000). SIC is the literature-comparable metric (the
absolute DOFS count is parameterization-specific — a *profile*, not a 2-parameter cloud-mean). The
angular leg — multi-angle radiance adding vertical DOFS at the saturated spectral baseline, and the
interchangeability test (spectral↔angular overlap only **pre**-saturation; Δ_ang(b)>0 at saturation) —
is the unique result.

**Band superset (10 bands; provenance in the notebook).** `[0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13,
2.26, 3.7, 4.05]` µm = HARP2 VIS/NIR (0.67 = 60 view angles) + OCI window/SWIR (1.038 is a clean
no-major-gas window band) + NK1990 **3.7** µm (strong absorption) + the **4.05** µm **VIIRS M13** band
(ω≈0.87 — slightly more absorbing than 3.7's window ω≈0.9; an *operational* MWIR channel) — testing
whether spectral IC has headroom beyond the 3.7 µm window band. (The 2.95 µm liquid-water absorption
*peak*, ω≈0.5, was rejected as too dark — no signal/sensitivity.) **Removed after review:** 2.00 µm (CO₂)
and 1.72 µm (CH₄-sensitive 1.6–1.7 region). No band order is baked into the workers; the **value-greedy**
order ({0.67,2.13} bispectral → 1.24,1.64 Platnick → fillers → 3.7 → 4.05 → redundant VIS) is applied
post-hoc, with a data-greedy cross-check.

**IC state = ALL r_e(τ) nodes INCLUDING the base (r_base = r_e at s=1); τ_bot held KNOWN.** r_base is
an r_e value so it stays *in* the profiled state (`info_content.jacobian_on_ode_grid(...,
include_base=True)`; the base joins the adiabatic prior block as the s=1 node, §11). τ_bot is *not* an
r_e quantity and is fully measured (A≈1), so holding it known isolates the r_e novelty and keeps the
s=τ/τ_bot depth grid well-posed. ⇒ the reported r_e IC is an **upper bound**. The τ_bot-unknown case is
a one-off sensitivity (`ic_tau_bot_check.py`, reported here, *not* in the notebook): promoting τ_bot to
an unknown is a clean ~1-DOF add (small r_e crosstalk) and the r_e IC is insensitive to linearizing
τ_bot at the prior mean vs the truth — confirming the τ_bot-known choice.

**Linearization = the LOO prior MEAN (set iv, HEADLINE); truth-linearization RETIRED.** The prior mean
= the ensemble **median**, sd = **1.4826·MAD** (robust-Gaussian; τ_bot is heavy-tailed — median 9.4 vs
mean 26, §11). Robustness levers retained: prior strength (loo / weak σ≈10 KV2012 / loo2x ℓ=1.0) and
linearization at a LOO **realization** (`draw`, set v). The pilot's truth-linearization (set ii) and
`info_content_linearity_probe.py` / `info_content_robust_truth.json` are removed.

**Noise = OCI 2 % calibration-relative** (`noise_model.oci_swir`), radiance *and* flux — matching the
pre-§15 retrievals (the pilot's 3 % was an inconsistency). Optics = the miepython table (§8).

**Raw-Jacobian caching (the architecture).** Each worker caches **K_full (all 10 bands × all views),
K_flux, s_int, the noise σ, the reflectance `y`, and the prior covariances** to a per-(mode,profile)
`.npz` sidecar (`ic_worker_profile.py`). Every figure quantity — the (n_bands × n_view) trade-off grid,
any band ordering/subset, Δ_ang/Δ_spec — is then a ~ms SVD of a *row subset* of one K_full, assembled
post-hoc (`ic_analysis_definitive.py` → `info_content_definitive.json`; the notebook only plots). No
forward solves in analysis. Mechanism (fig 5) from `ic_worker_mechanism.py` (same definitive config).

**Two post-hoc robustness tests (2026-06-25).** (1) **Noise-level sweep** — Sₑ is rebuilt from the cached
`y` at k_cal ∈ {1,2,3,5}% (the Jacobian is noise-independent), testing whether the angular/deep-base
gains — which ride near-0.5 filter factors, the most noise-sensitive modes — survive realistic radiometry.
(2) **Spectral-headroom test** — the 4.05 µm (VIIRS M13) band probes whether absorption *beyond* the
3.7 µm window opens a new vertical mode or confirms the saturation ceiling. Both are free SVD re-evaluations of
the cached K. The trade-off heatmaps also use a **uniform view axis** (1..N, not geometric).

**Per-band attribution = Shapley value [decided 2026-06-25].** Every single-band IC number in §15 (the
Test-2 bars, the substitution-vs-views curves, the free-node-vs-adiabatic comparison) is a **Shapley value** —
the band's average marginal contribution over *all* 2¹⁰ band subsets — not a single marginal. A single
marginal is superset-dependent and biased at either extreme: the **"9→10" leave-one-out** (`D(all)−D(all\b)`)
*under*-credits mutually-redundant bands (each VIS band ≈0 because another VIS band substitutes), while the
**"0→1" standalone** (`D({b})`) *over*-credits the τ/column anchor (every band ≈1 DOF alone, washing out all
contrast: free-node vs adiabatic ToA share becomes 19 % ≈ 20 %). Shapley is the unique *fair, superset-
independent* attribution and **sums to the total DOFS/SIC** (a genuine decomposition). It is tractable here
because `_fast_dofs_sic` exploits the **diagonal Sₑ**: `DOFS = tr((G+Sa⁻¹)⁻¹G)`, `SIC = ½log₂|I+Sa·G|` with
`G = (K/σ)ᵀ(K/σ)` an m×m (m = #nodes) matrix — O(rows·m²), no rows×rows inverse — verified bit-identical to
`posterior_diagnostics`. Conclusions are *sharper but identical in direction* under leave-one-out (e.g.
free-node ToA share Shapley 26 % vs adiabatic 16 %; LOO 39 % vs 7 %). `ic_analysis_definitive.py`:
`band_shapley`, `substitution_shapley`, `adiabatic_comparison`, `_shapley`, `_all_subset_metrics`.

**View-subset sampling — irregular golden-ratio (added 2026-06-26).** The view-sweep figures (Fig 2/3/3b)
pick V-view subsets of the 32 fixed principal-plane angles. A *regular* (uniform-µ) subset **aliases** with
the angular Jacobian of conservative-scattering bands (ω≈1, r_e signal purely angular) — the Fig 3b
0.55-µm V=16 notch — and, oppositely, *over-credits* the smooth magnitude signal of absorbing bands (a
regular grid optimally covers a smooth signal); **both are regular-grid sampling biases** (confirmed from
the raw K_full caches: 100 % of random 16-view subsets beat the regular one for 0.55; the absorbing 3.7 is
monotone, and where the regular grid *helps* it that is the opposite-sign bias). `spread_idx` now selects a
deterministic **golden-ratio (low-discrepancy) irregular** subset — both endpoints anchored (nadir +
most-oblique, so the full fan is spanned), only the interior irregular. The curves smooth and the artifacts
vanish; **conclusions are unchanged** (full-view DOFS 4.67 / SIC 19.4 / saturation n_sat=6 / band-greedy
order all identical — they live at the pinned nadir/full-view endpoints). Irregular is also the *realistic*
choice (POLDER/HARP2/3MI have fixed irregular angles). The forward K is at 32 fixed angles, so only the
diagnostic *subset selection* is irregularised; the production retrieval (full n_view=NQuad//2 fan) is
unaffected. `substitution_shapley` `band_idx` now includes band 0 (0.55) for the Fig 3b conservative-band
example.

*Files: `tests/supplementary/{ic_worker_profile,ic_worker_mechanism,ic_analysis_definitive,
ic_tau_bot_check}.py`, `src/{optics_table,info_content}.py`; handoff `AGENT_all125_ic.md`→`_fr.md`; figures
in notebook §15.*

## 15. Full r_e(τ) retrievals — log-space state, BP2026 convergence, oracle-adiabatic floor  [SETTLED — 2026-06-26]

The **capstone**: turn the §14 information-content claim into actual retrievals. Invert all 125 physical
VOCALS-REx profiles by joint Gauss–Newton OE (state `x = [r_e(s_nodes), r_base, τ_bot]`), NQuad=48,
μ₀=0.9, the §14 10-band × 24-view system, in **two prior configs** sharing one compiled forward — **A**
LOO-climatology prior mean (headline), **B** one LOO climatology *realization* (τ_bot sampled) as both
prior mean and first guess (robustness to where the regulariser is centred). **Headline = does the
free-node retrieval beat the best-possible adiabatic profile?** Three code upgrades (`src/retrieval_oe.py`):

**(a) Log-space state (`state_space='log'`).** Retrieve `x' = ln(x)` for the *whole* positive state
(r_e nodes, r_base **and** τ_bot) — BP2026 §2.4, who report it essential for GN convergence (Maahn et al.
86 %→100 %). Positivity is then free; the optics-table clamp only enforces support. The transform lives
entirely in `_split_state`/`_clamp_state`/`_encode_state` (a single `exp` at decode), so
forward/Jacobian/profile/grid-selection inherit it and autodiff returns the chain-ruled `K' = K·diag(x)`
**for free** — verified **exact at float64** (the production precision; forward parity 2e-12,
Jacobian-chain-rule 2e-11; a ~0.75 % log-vs-linear gap at float32 is just adaptive-step noise from
`exp(ln x)≠x` at 1e-7, and is in any case *not* a model error because the OSSE `y` and the GN forward run
the identical log path). The prior is transformed
by the **delta method** `to_log_prior(x_a,Sa)`: `x_a'=ln(x_a)`, `Sa'=D Sa Dᵀ`, `D=diag(1/x_a)` (validated
vs a Monte-Carlo log-normal) — exposed as `log=True` on `make_{joint,climatology,marine_sc}_prior`; the
Bayesian-Tikhonov correlation now lives on *fractional* r_e. The whole-state choice (vs radii-only) is the
natural reading of BP2026's "log-transform the state" and gives a clean lognormal τ_bot prior (τ_bot spans
1–50, so log is natural). **No §15-notebook IC re-run is needed**: posterior IC is reportable in linear µm
by un-chain-ruling `K_lin = K_log/r_e`, and DOFS/SIC are invariant under the reparam. *Grid selection
stays in physical space* (the noise-aware QRCP filter whitening is dimensionally correct there, and is
≈ invariant to the log reparam since `diag(r)·diag(σ_log) ≈ diag(σ_phys)`).

**(b) Robust Gauss–Newton — adaptive Levenberg–Marquardt + BP2026 convergence.** Examining the
per-iteration cost trajectory (rigour over results) exposed that the plain projected-GN `_gn_inner`
**oscillates**: a noiseless OSSE has a near-flat minimum (the misfit floors at the k-node *representation*
error, χ²≪1) and undamped GN **overshoots** it — the cost bounces and the last iterate is not the best.
This was **latent all along** — the old `joint_osse_results.json` runs show 2–5 cost up-steps and
`converged=False` in *both* linear and re5, thin and thick, even at 2–3 bands; it went unexamined because
the cost still *ended* low and the smoke test only checked `final ≤ initial` — **not** new to the
10-band/log capstone. `_gn_inner` is now an **adaptive LM** solve: a step is accepted only if it lowers the
full cost J; on a reject the damping λ is raised (×4) and retried, on an accept eased (×0.5) — guaranteeing
**monotonic descent**, so the returned iterate is the best found (verified: oscillation up-steps 2–5 → 0).
It improves *every* retrieval (the queued notebook re-run inherits it). On that monotone cost the BP2026
stops layer: **criterion 1 (primary): cost stagnation** — the data-misfit norm `φ = ‖y−F‖_{Sε⁻¹}` (= √χ²,
the OE form of BP's RSS) improves by a fraction **below `cost_rtol`** (`None` ⇒ no cost-stop; the LM
no-further-decrease / step-norm / `n_iter` stops only). `cost_rtol` **ships at 0.01**, erring tight (the unit
test reaches the same minimum as a far-tighter reference and the worker re-validation converges + beats
the floor) — NOT BP's 3 % on faith. A threshold-insensitivity sweep (`tune_cost_rtol.py`, COST_RTOL ∈
{5,3,1,0.3,0.1}% vs a tight reference) runs as **verification** and selects the **loosest value still on
the RMSE/DOFS/profile plateau** (it may relax 0.01 toward BP's 3 % if 0.03 proves on-plateau — a compute
saving, not a correctness change, since 0.01 errs tight). Erring tight matters because in a noiseless OSSE
an over-loose threshold under-converges and *inflates* our RMSE, which would unfairly
flatter the adiabatic floor — the wrong direction for the headline. **Verification** (idx 105, mid) found
`cost_rtol` **immaterial under the LM**: all of {5,3,1,0.3,0.1}% reach the *identical* converged result
(RMSE 0.2265, DOFS 4.67) in the *same* 7 iterations — the LM's no-further-decrease / step-norm criteria
bind before `cost_rtol` does, so 0.01 carries **no** convergence-vs-compute trade-off (thin/thick
verification is a cheap follow-up during the HPC run). **Criterion 2: noise floor** —
stop if reduced χ² `≤ chi2_floor` while non-increasing — is **implemented but default INACTIVE**
(`chi2_floor=None`): the Sε magnitude is not reliably profiled (the user's instruction), so the
noise-floor stop is coded and available but off, per §10h.

**(c) Oracle best-fit-adiabatic floor (`best_fit_adiabatic`).** The headline baseline (no adiabatic-OE or
bispectral re-run; BP2025/26 comparisons are literature-cited). Pure NumPy/SciPy (no RT), so every metric
is a post-hoc re-computation: fit `(r_top, r_base)` of the re5-linear curve to the **truth** at known
τ_bot by bounded least squares (`metric='rmse'` for ΔRMSE; `metric='maha'`, Cholesky-whitened by Ŝ⁻¹, for
the adiabatic lower bound `d²_adia,min` of the posterior Mahalanobis diagnostic). It is a *generous* floor:
`(r_top,r_base)` are fit to the truth (NOT pinned to its endpoints) and τ_bot is handed over as known
(which our retrieval must itself infer). The fit spans the full 2-parameter re5 family (**not** constrained
to `r_top≥r_base`) — i.e. the best fit within the retrieval's own re5-linear class, the like-for-like
"collapse our k-node state to 2 adiabatic DOF and fit perfectly" baseline; a monotone-constrained variant
is a one-line bounds change and, being post-hoc, is not baked into any run. ΔRMSE = RMSE_adia − RMSE_ours
(>0 ⇒ we beat the floor; ≈0 near-adiabatic).

**Observation = noiseless OSSE** (recap of §10b/§12): `y = F(x_truth)` with no noise realization added;
`Se` (oci_swir 2 % calibration-relative) enters only as the assumed weighting / posterior covariance, so
`x̂−x_truth` is pure regularisation/smoothing **bias** and `Ŝ` carries the analytic noise variance (no
Monte-Carlo). Retrievals run on the HPC (`AGENT_all125_fr.md`, repurposed from the IC handoff) in
**float64** (`PYDISORT_RICCATI_JAX_X64=1`), as the IC run did — **required, not optional** for the 10-band
forward. The discriminator is a 2×2 regime gap: the **notebook** retrievals ran *float32* but only on the
**2-band** bispectral pair `[1.24,2.13]` (even at NQuad=48/64, dense thin/mid/thick truths) — fine; the
**IC** run used the **10-band** superset at NQuad=48 on all 125 dense truths but at **float64** — fine;
so **10-band × float32 had never been run**. Two probes localised it (and refuted the obvious guesses —
DESIGN values rigour over a tidy story): `band_float32_probe.py` shows **all 10 band *forwards* pass alone
at float32** (so it is neither band *count* nor a single fragile band — bands are independent
`riccati_solve`s), and a forward-vs-Jacobian check shows the **smooth forward and the small
retrieval-grid Jacobian (6 columns) both pass** too. By elimination the trigger is the one remaining op:
the **grid-selection pool Jacobian** (`jacobian_on_grid` — a ~30-column **forward-mode AD** through the
adaptive integrator). Forward-mode AD augments each band's ODE with tangent states; that augmented system
is stiffer than the plain forward, and the float32 rtol-floor (§"Tolerance flooring") is calibrated for the
*forward*, not for forward-AD-through-the-solver. At the worst-case scale here (10 bands × ~30 tangent
columns) it exceeds the `max_steps` budget. Nothing prior hit it: the notebook's float32 Jacobians were
2-band / ~6-column (small enough); the IC run's big 10-band Jacobians were float64. The failure is thus in
**grid selection, before any GN step** — **not** a log-space artefact. (float64 has the roundoff headroom
and integrates the augmented system cleanly, as the IC run proved on all 125.) *Band-count is also a lever
if float32 were ever wanted:* each forward-mode tangent propagates through **all** band-ODEs at once, so
dropping bands shrinks the augmented solve, and the §14 IC shows the 10-band set is **redundant** for r_e
information — but float64 is the clean fix and all-10-at-float64 is harmless, so we keep them. float64 integrates cleanly (proven on all 125 by the IC run) and keeps
the posterior IC directly comparable to the float64 §14. Raw sidecars bundle back to jovyan where
`retrieval_analysis.py` computes every metric (RMSE/ΔRMSE, LWP bias, Mahalanobis, posterior IC). Grid
fixed per profile (`max_n_outer=1` — a clean A-vs-B comparison; a structural-misfit χ²>thr still warns,
flagged in the sidecar).

*Files: `src/retrieval_oe.py` (`state_space`/`to_log_prior`, `cost_rtol`/`chi2_floor`,
`best_fit_adiabatic`); `tests/supplementary/{retrieval_worker,tune_cost_rtol}.py`; handoff
`AGENT_all125_fr.md`; analysis `retrieval_analysis.py` + notebook §16 (post-results).*
