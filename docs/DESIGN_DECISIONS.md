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
pool.** The alternative — *decoupling* the retrieval grid from the ODE grid via a smooth
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

**Decision:** keep the (n_re, NLeg) Legendre table (`miejax_lite.table_lookup`) as the
production optics path; the hybrid traced-Mie-ω + HG pipeline is rejected. If a traced ω path
is ever wanted (hyperspectral), `mie_avg` first needs an r_e-independent radius grid. The GN
state must be clamped to the table support *inside* the forward map (bounded-state forward) —
model error can otherwise drive iterates to NaN optics. Details and figures: the notebook.

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

**(f) Adaptive node count (SO1) — DOFS *is* a robust info measure here  [SETTLED].**
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
filter estimator is the principled choice to switch to. **Recommendation: `method="filter"`** as the
default (no arbitrary `factor`, integer count, conservative under saturation); keep `dofs` as the
DOFS-proportional alternative.

**(g) Interpolation lever (SO2a) — second-order; re5-linear kept  [SETTLED].**
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

**(h) Re-meshing (SO2b) — n_outer=1 is the right default; re-mesh did not help  [SETTLED].**
On the thin case, lagged re-meshing (`n_outer=2`) moved the deepest node (s 0.49→0.37) but the
profile RMSE got *worse* (0.53→0.70 µm) while the fit was unchanged (‖y−F‖≈1e-3) — re-pivoting a
well-fit but correlated node basis just churns placement (the "re-mesh instability" of OUTSTANDING
§G, now mild under normalized depth). The χ²-gate (`remesh_if_chi2_red_gt`) is therefore the correct
policy: **re-mesh only on a persistently high loss (structural misfit), freeze the grid otherwise.**
For these well-fit VOCALS retrievals the gate keeps `n_outer` effectively 1, matching the SO2b
"disable if it destabilises / only re-mesh on high loss" rule.

---

## 11. Prior design — grounded in VOCALS-REx data + literature + information content  [SETTLED]

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

(Thin RF11: r_top only ~0.23 / ~30 %-prior-following — *information-starved thin cloud*,
**and possibly a discrete-ordinate resolution artefact**: near-top sensitivity is
single-scattering-dominated and oblique-angle / high-moment sensitive, so it needs more
streams; see `thin_top_resolution.py`. τ_bot and r_base behave as for thick.)

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
