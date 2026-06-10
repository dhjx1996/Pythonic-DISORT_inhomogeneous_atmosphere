# Outstanding Problems & Decisions

Open items, kept deliberately prominent. Settled rationale is in
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md).

Tags: **[BLOCKER]** must fix before retrieval works · **[DECISION]** a choice to make ·
**[BUG]** known-wrong behaviour · **[DEFERRED]** wanted, not yet started.

---

## A. Negative ToA radiances — forward model was physically wrong  [RESOLVED in the realistic regime]

**Resolved** by delta-M scaling + the Nakajima–Tanaka **TMS** correction (opt-in
`delta_M_scaling=True, NT_cor=True`; see `DESIGN_DECISIONS.md` §6). The symptom: for a realistic
forward-peaked cloud phase function (g₁≈0.85) the reconstructed ToA **radiance** went negative
(the m≥1 Fourier modes rang from the truncated forward peak) even though `flux_up_ToA` and the
m=0 mode `u0` stayed positive — the textbook finite-stream truncation artifact, which is why
flux-based tests passed and the DYAMOND flux lookup tables never exposed it.

- **Fix:** delta-M removes the forward peak from the truncated expansion (smooth, non-negative
  multiple-scattering field); TMS adds back the single scattering computed with the exact,
  untruncated phase function at the exact scattering angle. Correctness is verified by matching
  PythonicDISORT's own `NT_cor` solution to ~1e-6 (`tests/20_deltaM_benchmark_test.py`, exact
  single-layer + τ-varying), and the negative radiance is removed in the realistic regime
  (g≈0.85, NQuad=8: raw min −0.023 → corrected +0.043; `tests/19_deltaM_test.py`).
- **Residual at extreme peaks (known truncation limitation, not a bug).** Strict non-negativity is
  *not* guaranteed at finite streams: for very sharp peaks under-resolved by the stream count
  (measured g=0.9 at NQuad=16) delta-M+TMS *reduces* the negativity ~80 % (raw min −0.128 →
  corrected −0.025) but leaves a residual — and PythonicDISORT's `NT_cor` shows the *same* residual
  (we match it to ~1e-6), confirming it is intrinsic to the method class, not our implementation.
- **If it ever bites, the lever is more streams.** First *measure* whether the residual matters at
  the actual retrieval geometries/bands and an achievable NQuad (cloud retrievals observe
  back/side-scatter, while truncation error concentrates at ≲20° forward-aureole angles — likely
  below noise for us). If it does matter, raise **NQuad** — universal, monotone, no new failure
  modes, and cheap for us (jit+batch makes the bigger N×N matmuls GPU-friendly). **δ-M+**
  ([Lin & Stamnes 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC8051203/)) was examined and
  **deprioritised**: adoption is confined to the Stamnes ecosystem (DISORT/AccuRT; not LIDORT/
  VLIDORT/libRadtran/SHDOM), its wins are for extreme peaks (HG g=0.999, oceanic Fournier–Forand)
  outside our regime, and its "same cost" holds for the RT solve but **not** our differentiable
  pipeline — it replaces the trivial slice `f=g_{NLeg}` with a per-layer nonlinear (c,σ)
  moment-matching solve that would sit inside the autodiffed hot path every retrieval iteration.
- **IMS not implemented — and correctly so.** DISORT's IMS corrects **downward intensities only by
  construction** (STWLE 2000 = Stamnes, Tsay, Wiscombe, Laszlo & Evans, *DISORT report*, App. A,
  just after eq A.1: the −µ argument, "we correct only downward not upward intensities"). So there
  is **no standard upward IMS** to apply to our ToA-upwelling observable, and the retrieval-grade
  codes LIDORT/VLIDORT (Spurr) omit IMS too (exact single-scatter + streams, + a separate 2OS
  model). See `DESIGN_DECISIONS.md` §6.

### A′. Thin-cloud + Mie + off-nadir radiance needs many TMS moments  [RESOLVED — `NLeg_all` deficit]

**Root cause: too few Legendre moments in the TMS single-scatter, not streams or precision.**
Building the OSSE forward for a **thin** marine-Sc profile (RF11, τ_bot≈1.2) at the **2.13 µm Mie**
phase function with `NLeg_all=32`, the azimuthally-resolved ToA reflectance over a (μ,φ) grid was
**erratic, non-smooth in μ, sign-flipping** — R to +0.7…+0.9 (≈10× the physical ~0.05–0.2), and
near-nadir (μ=0.95) swung +0.06→−0.05 across azimuth where it should be ~flat; **1.24 µm was worse**
(|R|>1). It was **not** stream count (NQuad=16 ≈ 24, both erratic) and **not** float32 (a float64
run reproduced it bit-for-similar). **Raising `NLeg_all` 32→128 fixes it completely**: all R
positive (min +0.078), smooth in μ, near-nadir azimuth-flat, both bands physical, at NQuad=16.

**Why.** TMS adds the exact single scatter via `p_full(cosΘ)` reconstructed as a *truncated*
Legendre sum to `NLeg_all`. A Mie cloud phase function (size parameter x≈24 at 2.13 µm) has ≈2x+10
≈ 60 significant moments; truncating at 31 gives a **Gibbs-oscillating, sign-changing `p_full`** —
garbage that varies with cosΘ (hence μ,φ). **Thin clouds are single-scatter-dominated, so the bad
`p_full` *is* most of the signal** — which is exactly why the notebook's **thick τ=12** cloud
looked fine (+0.27: multiple-scatter dominates, TMS is a minor correction) while thin broke, and
why **flux / m=0** stayed physical (no sharp single-scatter angular structure). PythonicDISORT's
own NT validation (`pydisotest/5_test`) uses the **Cloud C.1 phase function with 300 moments** and
**NQuad=48** — i.e. ~300 moments is standard for an NT-corrected cloud; our 32 was ~10× too few.
**`NLeg_all` is cheap** — it feeds only the TMS τ-quadrature, *not* the unrolled per-mode diffuse
solve (item H) — so 128–300 costs negligible memory (unlike NQuad). **Barycentric Gauss–Legendre
μ-interpolation is *not* at fault** (well-conditioned, Lebesgue ~log n, unlike the cubic splines
Sta1982 critiques); with good moments the off-node field is smooth. *(Source-function integration
(Sta1982) — exact intensity at arbitrary μ — is **deprioritised**: unclear it beats barycentric,
and PythonicDISORT notes the interpolation route is easier and likely cheaper. Noted, not pursued.)*

**Irreducible limit — the ~10° forward (solar-aureole) exclusion.** Even at 300 moments / NQuad=48,
PythonicDISORT must **discard intensities within ~10° of the forward-scatter / solar-aureole
direction** (documented there and elsewhere); the truncated-`p_full` error concentrates at the
near-forward peak and is not removable by more moments/streams. **Not a concern for our observable:**
ToA *upwelling* with μ0=0.6 has minimum scattering angle Θ≈37° (grazing, φ=0), ≈74° at μ=μ0, →180°
at backscatter — so a back/side-scatter envelope (μ≥0.5, φ≈π/2…π) never samples the aureole. Keep
view geometry there; avoid near-forward (small-Θ, φ≈0, grazing-μ) directions.

**Accuracy budget.** PythonicDISORT's 1% test tolerance is far tighter than our ~10–20% measurement
noise — so `NLeg_all=128` (smooth, positive, physical) is **ample**; no need to chase 256/300 to 1%.

**Resolution / settings:** use **`NLeg_all ≥ 128`** for Mie clouds with NT_cor (`retrieval_oe`
default bumped 32→128). Keep views in the back/side-scatter envelope. Multi-angle multi-band is
**viable** at NQuad=16 — no flux-only pivot needed.

**Why the test suite missed it** (the user's question): tests `19`/`20` use **smooth analytic
Henyey–Greenstein** (`g**l`, monotone decay) — never a structured Mie phase function whose high
moments matter; **moderate/thick τ (5–10)** — never single-scatter-dominated thin τ≈1; assert
non-negativity **only at quadrature nodes**; and **explicitly restrict** strict-positivity to
g≤0.85. The pydisort comparisons assert *agreement* (`rel_tol=1e-2`), so a shared low-`NLeg_all`
TMS artifact would pass while both are wrong. **Action (open):** add a **thin-cloud, Mie,
off-nadir (μ,φ)-grid physical-plausibility test** (smoothness + magnitude + positivity, not just
pydisort agreement) and an **`NLeg_all` convergence check** for NT_cor. Interacts with item H
(NQuad/unroll ceiling) and item G (angular-DOF).

---

## B. Optics interpolation: τ-axis vs r_e-table  [DECISION]

**Underlying problem:** supplying the solver continuously-varying ω(τ), gₗ(τ) with **neither**
a τ-discretization **nor** an r_e-table is too expensive — full Mie inside the ODE vector field
never finishes compiling (see item C). So some interpolation is required. Two candidates:

1. **τ-axis** — interpolate τ → (ω, gₗ); node values = Mie(r_e(τ_node)), rebuilt each retrieval
   iteration (~9 Mie calls/iter). Gradient = **exact Mie at nodes** + linear interp between
   (cheapest true full-Mie backprop). Linear interp of moments is a convex combination ⇒
   preserves phase-function non-negativity and ω∈[0,1] (cubic/barycentric can overshoot).
   **Weakness:** node *placement* in τ depends on an a-priori profile guess; a wrong guess
   biases the retrieval (Rodgers representation error). Rescuable by lagged mesh re-selection
   (run X opt steps, re-mesh from the current iterate = inexact multilevel optimisation;
   r-refinement is recompile-free, h-refinement costs one reusable recompile per node-count).
   Distinguish *placement* error (re-meshing fixes) from *resolution* error (needs more nodes).

2. **r_e-axis table** (= paper Appendix B map B1: r_e → (ω, gₗ) on a log(r_e−3) grid) — built
   **once** over the physical size range (~4–21 µm), profile-independent; the solver evaluates
   r_e(τ) at its own adaptive points and looks up. **No τ-placement problem at all.** Gradient
   = table-slope (inexact-but-consistent Mie Jacobian); fine for Gauss–Newton, whose fixed
   point is set by forward accuracy, not the Jacobian.

**Trade:** exact per-iteration Mie gradient + placement fragility (τ-axis) vs robust/simple +
inexact-Jacobian (r_e-axis). Item 3 of `DESIGN_DECISIONS.md` (ToA-concentrated, ~rank-4,
profile-independent retrieval grid) argues *few, sensitivity-placed* nodes suffice and favours
robustness. **Current lean:** prototype the r_e-axis table; reserve the τ-axis + lagged
re-selection for if the lookup-slope Jacobian proves too inexact for the final fit.

### B′. The r_e(τ) profile parameterisation — the inter-node interpolation lever  [OPEN — explore]

*Distinct from the optics-table axis above:* this is how the retrieval state (r_e at the few grid
nodes) becomes the continuous **profile** r_e(τ) the solver integrates. It is **one localised
lever** — `retrieval_oe.RetrievalForward._re_of_tau` (a single `jnp.interp`) — through which the
forward, calibration, ODE-grid, Jacobian, and (via `RetrievalForward.profile`) the **display** all
route. **Key realisation:** it is **part of the forward map F(x), not a post-hoc/independent
choice** — it defines *what is retrieved*, so any change must (and now does) propagate to the
displayed curve too (`profile()` mirrors it). *(We — and the user — initially mis-filed this as an
independent downstream interpolation; corrected.)*

**Current default = r_e⁵-linear (adiabatic)** `jnp.interp(τ, knots, vals**5)**(1/5)`. The adiabatic
effective radius grows as `r_e ∝ τ^(1/5)` in optical depth: `r_e³ ∝ LWC ∝ geometric height z`, and the
extinction `β ∝ r_e² ∝ z^(2/3)` makes `τ = ∫β dz ∝ z^(5/3)`, so `LWC ∝ τ^(3/5)` and **`r_e ∝ τ^(1/5)`**
(≡ the canonical adiabatic `N_d ∝ τ^(1/2) r_e^(-5/2)`; verified numerically — `r_e⁵` linear in
`(1−τ/τ_bot)` to ~1e-12). So `r_e⁵` is what is linear in τ. It **represents the adiabatic prior
exactly**, gets per-segment curvature from the two endpoint values ⇒ **no grid-size coupling**, and
with finite `r_base` keeps `dr_e/dτ` finite at base (no cusp). It is still C⁰ (kinked at nodes).
Baselines saved for comparison: `docs/retrieval_baseline_linear_class.json` (thin, linear class) and
`docs/retrieval_thick_RF03_tau23.json` (thick, re5-linear).

**The candidates and the axis that separates them — node-based smoothness vs grid coupling:**

- **linear** (`jnp.interp(τ, knots, vals)`) — the "impute-nothing" baseline; C⁰, no coupling.
- **r_e⁵-linear** *(default, adiabatic)* — C⁰, no coupling, prior-coherent. Imputes the *adiabatic*
  shape per segment (r_e ∝ τ^(1/5); consistent with the adiabatic S_a prior mean). The §B′ model
  comparison probes it against **linear** — insensitivity ⇒ the data can't distinguish shapes (linear
  fine by Occam); a real difference ⇒ shape info, pick by evidence.
- **PCHIP (monotone-cubic, C¹)** — the only listed class that **de-kinks** the profile (so it would
  de-artifact the class↔ODE-grid↔re-meshing loop below). **But it couples to grid size:** a node-based
  C¹ class sets nodal slopes from *neighbours*, so it degrades to linear at 2 points and only gains
  real curvature at ≥3 — and that curvature is then a finite-difference artifact, not data, at low
  node count. So PCHIP is **deferred until the node-count/DOF supports it** (revisit candidate for the
  **thick-cloud** retrieval, where there are more nodes and higher DOF). *(Natural/global cubic is out
  — it overshoots, breaking positivity/monotonicity.)*
- **C¹-without-coupling exists only by leaving interpolation behind:** prescribe nodal slopes from a
  model (adiabatic-slope cubic Hermite) or fit a low-dim **shape/EOF basis** — but that is the
  *parametric-basis architecture* (DESIGN §3 alternative), not a drop-in `_re_of_tau`; for DOFS≈2 it
  may even be the more honest framing (retrieve ~2 adiabatic parameters, not 4 nodes + interpolant).

**Two recorded subtleties (do not lose):**
1. **class↔grid↔re-meshing kink-coupling.** A C⁰ profile's kinks sit at the nodes, so the ODE grid
   (placed by the ~C⁶ error estimator) *clusters at the nodes*, and QRCP/re-meshing then re-select
   near them — partly self-referential (also taints the *first* pool, built on the coarse first-guess
   grid). C⁰ classes (linear, r_e⁵-linear) keep this; only a C¹ class removes it. It is **second-order**
   (a mild step-count/selection effect), so not worth importing PCHIP's grid-coupling to fix *here*.
2. **probe the class by model comparison, not grid stability.** "Grid doesn't move ⇒ profile near
   prior ⇒ adiabatic class confirmed" is **confounded** — at DOFS≈2 the profile is prior-dominated
   regardless, and the kink artifact pins the grid. The clean test is to retrieve the *same data*
   under {linear, r_e⁵-linear, PCHIP} and compare fit χ² / posterior: insensitivity ⇒ data can't
   distinguish shapes (linear fine by Occam); a real difference ⇒ data carry shape info, pick by
   evidence. **One profile is not definitive** — needs a multi-profile study.

**How to change:** edit `_re_of_tau` once; forward, `calibrate`, `ode_grid`, Jacobian, the re-meshing
re-map in `gauss_newton_oe`, pool sampling in `select_retrieval_grid`, and `profile()` all follow.
The cleanest *promotion* makes the **state itself** the new class's coefficients (e.g. node values of
r_e⁵, or basis amplitudes) so parameterisation, prior mean, and display are coincident by construction.

---

## C. jit-ability of the solver — the retrieval-cost lever  [RESOLVED]

**Resolved** by a host-side **setup / traceable solve split** (the composable seam; see
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) §7). `riccati_setup(...)` runs all the SciPy,
`mu0`/`tau_bot`-independent work once; `riccati_solve(setup, omega_func, Leg_coeffs_func, tau_bot,
mu0, num_modes=K)` is then a pure, **jit / grad / jacfwd-able** function of the traced inputs, and
`eval_radiance(...)` is the traceable observable. The one-shot `pydisort_riccati_jax` delegates to
the same core (5-tuple unchanged, **bit-for-bit** — test 21b). Cold→warm caching confirmed (e.g.
NQuad=6 thin: jit forward ~42 s compile → ~0.3 s cached, no recompile across varying
`tau_bot`/`mu0`; reverse-`grad` ~3 min compile then cached); `tests/supplementary/demo_jit_retrieval.py`
runs the live recipe.

The two host-side blockers and their fixes:

1. `_kvaerno5_integrate` did `np.asarray(sol.ts)` + dynamic slicing to build `tau_grid`. Fix: a
   `save_grid` flag — `SaveAt(t1=True)` (only the final state is needed for ToA) on the jit path,
   no host sync, `tau_bot` may be traced; the offline grid path keeps `SaveAt(steps=True)`.
2. `_precompute_legendre` called `scipy.special` on what become tracers. Fix: it is
   `mu0`-independent, so it moves wholesale into `setup`. *(Originally the one `mu0`-dependent term
   `P_l^m(−μ0)` was computed in-trace by a custom associated-Legendre recurrence so `mu0` could be
   traced. **Superseded by §H:** `mu0` is now **static**, so that term is also precomputed host-side
   with scipy and the in-trace recurrence was removed.)*

**Mode count.** *(Originally the exact DISORT azimuthal-convergence (Cauchy) criterion, STWLE2000
§3.7 p.89, via `calibrate_num_modes`.* **Superseded by §H:** that relative partial-sum test saturated
at low signal and was removed; mode truncation is now the noise-aware `retrieval_oe.select_num_modes`.
The solver default is all `NFourier` modes; `riccati_solve(..., num_modes=K)` runs exactly K.) The AD
and static caveats still hold: reverse-`grad` (the discrete adjoint, §5) is the default; forward-`jacfwd`
needs `riccati_setup(..., adjoint=diffrax.ForwardMode())`; `phi0`, `I0`, `mu0`, and the boundary
conditions are static (baked into `setup`). The old callable-BDRF-at-traced-`mu0` jit hazard is gone
now that `mu0` is static — BDRFs are evaluated host-side at the static `mu0` in `riccati_setup`.

**Single-trace `scan`+pad/mask — IMPLEMENTED (§H).** The deferred optimisation landed: the unrolled
`for m in range(K)` is replaced by a single `lax.scan` over the **padded** `(NFourier, NLeg, N)`
per-mode tensors (`l<m` rows zeroed). One compile serves every K, and — the actual motivation — the
mode body compiles **once** (O(1) compile memory in mode count), which lifted the NQuad=24 / jacrev
OOM ceiling.

**Empirical history (the evidence that motivated the split; 2026-06-08, NQuad=8, T4 vs CPU; see
`tests/supplementary/profile_solver.py`).** *Recompile-every-call:* three identical unjitted calls
took 60.2/57.9/59.7 s on GPU and 50.9/49.4/50.3 s on CPU — **zero speedup on repeat**, confirming
every call recompiled. *Host-side compile:* a cProfile attributed ~54 s of the ~60 s to
`jax…trace_to_jaxpr` + `pjit._trace_for_jit` (16× = 8 modes × 2 sweeps via `diffrax.diffeqsolve`) —
Python tracing + XLA *lowering*, not device execution; the GPU sat idle (≈18 % slower than CPU).
*The fix prototype worked:* a jit-able single-mode forward R-solve (`SaveAt(t1=True)` + numpy GL
nodes) compiled in ~2 s then ran cached in 2–29 ms (≈100–1000× cold→warm) — now realised in full by
the seam above.

---

## D. GPU is latency-bound for this solver  [NOTE — empirically confirmed]

Cached execution is dominated by **many sequential tiny matmuls** (NFourier modes × 2 sweeps ×
~35 adaptive steps × 5 ESDIRK stages on N×N, N≤8/16). This is kernel-launch-latency-bound, so the
GPU is *not* faster than CPU per single column. Real speed levers: fewer Fourier modes, fewer
adaptive steps (looser `tol`), and batching across columns (below) — not the device itself.

**Measured (2026-06-08, `tests/supplementary/profile_solver.py`).** The results below were taken on
a Tesla T4, but the analysis is **GPU-agnostic**: the binding costs are host-side XLA compilation
and per-kernel launch latency, both set off-device, so the conclusions are about software structure
and batch regime, not the specific accelerator. Cached warm execution of the jit-able single-mode
solve: **CPU 2.1 ms vs GPU 28.9 ms — the GPU is 14× slower** (each tiny 4×4-matrix kernel launch is
latency-bound, with no parallelism to exploit). The levers behave as claimed: NFourier 8→2 cut the
unjitted call 62→14 s (GPU) / 51→14 s (CPU, ~linear); looser `tol` 1e-3→1e-2 cut the *cached* warm
time 28.9→22.2 ms (GPU) / 2.1→1.6 ms (CPU) — but barely moved the unjitted call (62→61 s), because
there compile, not step count, dominates.

**Important qualifier — this is the *single-column* regime; batching across columns flips it
(`tests/supplementary/batch_columns.py`).** Per-column launch/dispatch latency is set off-device,
so the single-column result is a property of the workload, not the GPU. But the retrieval is
embarrassingly parallel across columns: `jax.vmap` over a batch turns the tiny matmuls into batched
matmuls that fill the device. Measured warm per-column time vs batch size B (µs/column):

| B | 1 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| **GPU** | 30592 | 2233 | **555** | 155 | 50 | **16** |
| **CPU** | 1908 | 1021 | **959** | 854 | 829 | 846 |

CPU per-column is ~flat (limited parallelism); GPU per-column collapses ~1900× once there is a
batch to hide latency behind. **Crossover at B≈64; at B=4096 the GPU is ~53× faster than CPU per
column.** So "GPU not a lever" holds *per single column*, but the right retrieval architecture is
**jit (item C) + vmap a batch of columns onto the GPU** — it is the batch, not the device, that
delivers the latency hiding.

---

## E. Retrieval loop not yet implemented  [DEFERRED]

The cost function `J(θ)`, Gauss–Newton/LM iteration, and r_e(τ) profile parameterisation
(report §"Toward Retrieval") are not built. Forward-mode preferred for small p (≤~15 params),
reverse-mode for large p (crossover p≈15–20 at m=10 observations).

## F. Other deferred forward-model features  [DEFERRED]

Isotropic internal source (only the collimated beam is handled) and non-ToA depth evaluation
(only τ=0 is returned). *(Delta-M scaling and the Nakajima–Tanaka TMS correction are now
implemented — see item A and `DESIGN_DECISIONS.md` §6. IMS remains out of scope — it is
downward-only by construction in DISORT and is likewise omitted by LIDORT/VLIDORT; see item A.)*

**Adjoint robustness (minor).** A reverse-mode `grad` can NaN (singular lineax solve) on an
*aggressively steep* synthetic r_e profile — not real VOCALS-REx, which is finite-slope; quick fix
is a least-squares adjoint solver, `AutoLinearSolver(well_posed=False)`. Low priority.

---

## G. Retrieval information content — what is actually retrievable  [DECISION] [open]

The robust part is settled in [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) §3 (ToA-weighted,
small DOF, ODE grid ≠ retrieval grid). These sub-claims are **not** established and should not
be relied on until re-examined:

- **No rigorous "rank-4 ceiling."** Earlier SVD/QR analysis found ~4–6 dominant singular
  directions, but (i) the hard upper bound is the stream count N (=8 at NQuad=16), with no
  symmetry argument reducing 8→4; (ii) "4" is a soft, threshold-dependent count (the >0.1%
  threshold already gives 6); (iii) it was measured emphasising the m=0 mode, without delta-M,
  at N=8, and for a single geometry/thickness — all of which can suppress it. Claim only "small
  DOF," not "4."
- **Multi-mode / angular DOF was contaminated by the missing delta-M (item A — now fixed).** In
  `adiabatic_cloud_with_drizzle.ipynb` the per-Fourier-mode ‖∂u/∂g‖ is *larger* for several
  m≥1 modes (e.g. m=7 ≈ 0.5) than for m=0 (≈ 0.1) — but m≥1 is exactly where the radiance rang
  without delta-M. So whether higher azimuthal modes carry genuine extra information could not be
  judged until delta-M/TMS was fixed; the QRCP grids in that notebook sum all modes and inherited
  the contamination. **With item A now resolved, re-run the rank/Jacobian analysis with
  `delta_M_scaling=True, NT_cor=True` before drawing any angular-DOF conclusions.**
- **Profile-independence unproven.** Demonstrated only for a localised g-spike on one smooth
  adiabatic base; the angular-collapse depth depends on ω/band; globally different profiles
  (thin, multi-layer, inversion) untested.
- **Multi-band gains are real but saturating** (Coddington, Pilewskie & Vukicevic 2012):
  additional bands add information diminishingly due to inter-band correlation; the
  vertical-resolution gain rests on penetration-depth diversity (Platnick 2000), which is
  modest. Do not assume multi-band lifts the DOF far.

**To resolve:** re-run the Jacobian / effective-rank analysis *after* delta-M, across optical
thicknesses and solar/viewing geometries, using the combined r_e channel
(`J_ω·dω/dr_e + J_g·dg/dr_e`) and stacked bands. Also verify the Platnick (2000) and CPV (2012)
specifics before they are cited in the report.

*Starting point:* the prior multi-mode / full-radiance rank study (three tiers — baseline u₀,
full-u with all 16 Fourier modes = 128 rows, and NQuad=32; ToA rank stayed 4 in all, with
per-mode BoA decay ‖J^{m=1}‖≈9e-12, ‖J^{m=2}‖≈1e-16) lives in the removed
`technical_reports/boa_step_clustering_report.tex` — recoverable from git `99fb971`. **Treat its
conclusions as contaminated** (built on the un-delta-M'd m≥1 modes); re-derive, don't cite.

### Per-mode ODE grids and the retrieval grid  [DEFERRED — logged]

*(Significance is to retrieval-grid **quality**, not compute — cost is secondary here. Surfaced
while deciding not to `vmap` the Fourier modes, item C / §7; logged for the retrieval-grid work.)*

The ODE grid that §3a uses as the retrieval-grid candidate **pool** is, today, the **m=0 grid
alone**: the solver computes a grid per Fourier mode but returns only `tau_grid_m0` and **discards
the m≥1 grids**. m=0 is a defensible default — it carries the slowest (diffusion) eigenvalue and
the beam source, so it is typically the *densest* single grid and the largest single-channel
superset, and it holds the flux plus the bulk of the ToA-weighted, small-DOF information
(§3b/c). But each Fourier mode is an **independent angular information channel** with its **own**
ODE grid; discarding the m≥1 grids likely throws away retrieval-grid information. Significance:

- **Pool completeness (improves the §3a *superset*).** The observable is `Σ_m u_m·cos(m(φ0−φ))`;
  different modes can be sensitive to different τ-depths, so the m=0 grid can *miss* τ-features
  informative only for the angular (m≥1) channels. The principled pool is plausibly the **union of
  the non-negligible modes' grids** (`∪_m {variation_m} ⊇ ∪_m {retrievable_m}`) — a strict
  generalization of §3a's "subset of the m=0 grid."
- **Selection precision (improves the §3a *subset*).** QRCP/sensitivity selection currently runs on
  the **summed** Jacobian (it blends modes; this §G already flags it). A **per-mode sensitivity
  decomposition** prunes more precisely: keep a τ-point only if *some* non-negligible mode is
  sensitive to it; drop one whose only support is a Cauchy-negligible mode, even where the summed
  Jacobian gave it modest weight.
- **Decides the open angular-DOF question (above).** Re-running the (now delta-M-corrected) per-mode
  Jacobian/grid analysis answers *where in τ* each azimuthal mode places its sensitivity: if the
  m≥1 modes cluster at the **same** near-ToA depths as m=0, they add angular detail at **no new
  vertical resolution** (pool stays ≈ m=0); if they place steps **deeper/differently**, they carry
  **complementary vertical information** and the pool must be the union. This is the decisive test
  of whether higher azimuthal modes lift the retrievable *vertical* DOF.
- **Ties to the Cauchy stop (item C / §7).** The same K from the azimuthal-convergence criterion
  that truncates the *forward* also names the **non-negligible modes** whose grids should form the
  retrieval-grid pool — so the Cauchy machinery feeds the grid construction directly.
- **Caveat — variation ≠ information.** A per-mode grid is placed by that mode's *state* variation,
  which includes the optics-independent **BoA imbedding boundary layer** (≈zero information) present
  in *every* mode. So the per-mode *grids* enlarge the pool, but it is the per-mode *sensitivity*
  (the adjoint face, §3a) that does the informative pruning.
- **Step-count as a cheap amplitude predictor (hypothesis, secondary).** A mode whose Riccati state
  barely leaves its IC (few adaptive steps) plausibly yields a small ‖u_m‖; if so, step count
  a-priori predicts mode negligibility. The *true* Cauchy signal is the amplitude ‖u_m‖ (what
  DISORT tests); step count is only a correlate — to be checked, not assumed.

**Net:** the "best" retrieval grid is plausibly the **sensitivity-selected subset of the union of
the non-negligible (Cauchy-K) modes' ODE grids** — generalizing §3a from the m=0 grid to the full
angular-channel set. **Prerequisite:** retain the per-mode grids (and per-mode `u_m` /
sensitivities) in the offline `return_grid=True` path (currently discarded). Implementation
deferred; this records the design so the retrieval-grid work can pick it up.

### Re-mesh instability ⇒ the node basis is correlated  [OPEN — flagged 2026-06, revisit soon]

*Surfaced building the thick-cloud (RF03, τ≈23) retrieval. Recorded for the upcoming retrieval-grid
/ parameterisation work; the demo sidesteps it with `n_outer=1` (select-once).*

**Symptom.** With the lagged re-mesh (`gauss_newton_oe(n_outer=2)`), the first QRCP selection at the
adiabatic first guess gave a clean, well-spread grid `[0, 1.2, 3.5, 6.6, 13]`; the **re-selection at
the post-fit state** returned `[0, 0.65, 2.9, 12.6, 13.8]` — two nodes **clustered deep** (Δτ≈1.2,
both ±3 µm) and the mid-cloud (τ≈6) abandoned. The retrieved **profile barely changed** between the
two; only the **node placement** jumped.

**Diagnosis — correlated node-sensitivity columns, re-pivoted at a moved linearization point.**
Two compounding causes:
1. **The node basis is non-orthogonal.** Because `r_e(τ)` is an *interpolant*, moving any one node's
   value changes the radiance over the whole τ-span of its two adjacent segments, so neighbouring
   nodes' Jacobian columns `∂y/∂r_e(τ_j)` overlap heavily. Evidence: the averaging-kernel diagonal is
   **spread**, `A_ii ≈ [0.61, 0.39, 0.30, 0.40, 0.66]` — *no* node is independently resolved; the
   2.36 DOFS are shared. QRCP must pivot among **near-collinear** columns, which is ill-conditioned:
   a small change in the linearization point reshuffles the pivot order. The re-mesh supplies exactly
   that, so the set jumps.
2. **k_active > DOFS — by design, not a defect.** `k_active=5` with DOFS≈2.3 leaves ~2 nodes in
   the near-null space, *prior-regularized on purpose*: we **want** a few more nodes than DOFS,
   because (i) a node basis is never fully independent — `DOFS = tr(A) < n_nodes` is intrinsic, the
   nodes are not the independent quantities — and (ii) letting the prior fill the under-determined
   directions is the point of regularized OE, a feature. What is unstable is **only where QRCP
   *places* those surplus nodes**: with no data to pin them, pivoting among near-collinear columns is
   arbitrary. (Their ±3 µm bars ≈ prior σ confirm they are prior-supplied.) So the result is fine —
   the cure is to stabilise *placement*, not to cut `k_active` down to DOFS (which would discard the
   deliberately prior-filled nodes).
The **deep-node leverage** also misleads the placement: with the base anchor fixed, a deep node's
wide `r_e⁵` segment reaches up into the *visible* upper cloud, so it gets non-trivial sensitivity
(high `A_ii`) despite being imprecise — the §B′ class↔grid coupling.

**Not a retrieval failure** — fit `‖y−F‖≈1e-2 ≈` noise, profile stable; only the *placement* is
unstable. So `n_outer=1` (freeze the grid) is a legitimate fix for the figure, not just hiding.

**Candidate fixes (revisit) — stabilise *placement*, keep k slightly > DOFS:**
- **Freeze the grid (`n_outer=1`).** Select once, don't re-pivot at the moved estimate. What the demo
  does; simplest, keeps k>DOFS, no recompiles. The lagged re-mesh stays available for the thin case
  where the grid genuinely needs refining.
- **Scale the Jacobian by prior σ before QRCP.** `select_retrieval_grid` runs QRCP on the **raw**
  Jacobian (the "(scaled)" in its docstring is aspirational); ranking by *information*
  (sensitivity × prior σ) rather than raw sensitivity places the surplus nodes more sensibly.
- **Rank-revealing cardinality (partial — note the tension).** The QRCP **R-diagonal** pivot
  magnitudes a-priori show where the resolvable directions end; useful as a *diagnostic*. But cutting
  `k_active` to ≈DOFS is **not** the goal — we deliberately keep a few prior-filled nodes — and it
  also breaks the **fixed-cardinality, recompile-free** re-mesh design.
- **The real answer — leave the node basis.** Retrieve a few **orthogonal shape modes** (SVD/EOF of
  the sensitivity or prior) instead of N correlated node values: then k = DOFS by construction, there
  is nothing to re-pivot, and "where did the info come from" is a clean per-mode statement. This is
  exactly [§B′](#b-the-r_eτ-profile-parameterisation--the-inter-node-interpolation-lever--open--explore)'s
  "leave interpolation behind, fit a low-dim shape/EOF basis." **The flapping re-mesh is empirical
  motivation for that architecture**, not just a nuisance.

### Planned change — auto-select the node count `k_active` from a QRCP threshold  [TODO, record only]

Today `k_active` is a **hardcoded hyperparameter** (4 thin / 5 thick); `select_retrieval_grid`
QRCP-*ranks* the pool but the *count* is manual. Replace it with a **data-driven threshold**, computed
**once** at the first guess and **frozen** for the retrieval (so the forward/Jacobian still compile
once — mirrors how `select_num_modes` fixes the Fourier-mode count). Keep `k` **slightly above the
resolvable rank** on purpose, so the prior still fills a margin (per above: nodes are never independent,
and prior influence is a feature — so the target is *≈DOFS + small margin*, **not** exactly DOFS).

The QRCP already produces the signal: the **pivoted R-diagonal** `r_1 ≥ r_2 ≥ …` is each node's
*marginal* information (its residual sensitivity after orthogonalising against the already-chosen
nodes) — a surrogate singular-value spectrum of the node basis. Candidate thresholds:

- **Noise-aware (preferred — mirrors `select_num_modes`).** Whiten first:
  `K̃ = Se^(−1/2) · K · Sa^(1/2)` (rows by noise, columns by the prior square-root; `diag(σ_prior)` is
  the cheap approximation that ignores prior correlations). Then the R-diagonal is in **SNR units**, and
  the per-direction **filter factor** `f_i = r_i²/(1+r_i²)` is literally "fraction from data" (Rodgers
  — the *same* quantity as the UQ data-fraction bar), with `Σ f_i ≈ DOFS`. Keep all directions with
  `r_i ≳ 1` (data-dominated, `f_i ≳ ½`) **plus a small fixed margin** of prior-dominated ones ⇒ `k`
  self-consistently lands at ≈DOFS + margin with **no circular post-hoc DOFS**. *(This also finally
  implements the column-scaling `select_retrieval_grid`'s docstring already claims.)*
- **Relative cutoff (simpler, less principled).** Keep while `r_i/r_1 > ε` (ε≈0.05–0.1) — numerical-rank
  style, but raw units ⇒ `ε` is a feel parameter, not tied to the noise.
- **Cumulative information.** Keep until `Σ_{≤k} r_i² / Σ r_i² ≥ frac` (PCA-style) — robust-ish but
  `frac` is opaque.

Constraints: always include cloud-top (τ≈0); clamp `k ∈ [1, k_max]`; choose-once-and-freeze
(recompile-free); the **margin is the explicit "let the prior fill" knob**. Prereq: the
whitening/column-scaling (QRCP currently runs on the **raw** Jacobian).

---

## H. Fourier-mode unroll → compile-memory; revisit vmap/scan + static-mu0  [RESOLVED — 2026-06]

**Resolution (2026-06).** Implemented all three levers below as a unit:
1. **`lax.scan` over the Fourier modes** (chosen over `vmap` by a de-risk + profiling prototype that
   ran the identical padded-tensor + static-μ0 mode body under both, asserting forward + `jacrev` +
   `jacfwd` parity vs the old unrolled solve at ~1e-15). **Memory-first decision:** both compile the
   mode body **once** (O(1) graph), but `scan` is sequential (one mode's working set live at a time)
   and preserves each mode's *independent* adaptive stepping, whereas `vmap` materialises all
   modes×columns simultaneously and forces a uniform step count. Since not-OOMing outranks speed
   (and we batch *columns*, not modes, operationally), `scan` won. The ragged `(NLeg−m, N)` per-mode
   tensors are **padded** to `(NFourier, NLeg, N)` with the `l<m` rows zeroed, so the body is
   mode-index-free (the mode index is the scanned axis, not a Python int).
2. **Static μ0.** `P_l^m(−μ0)` is precomputed host-side with scipy (`_padded_legendre_modes`) into a
   `(NFourier, NLeg)` table; the in-trace `_assoc_legendre_neg_mu0_jax` recurrence is **removed**.
   `mu0` moved from a `riccati_solve` arg to a `riccati_setup` parameter.
3. **Cauchy → S_ε.** `calibrate_num_modes` (the relative azimuthal partial-sum test, which saturated
   at low signal) is **removed**; mode truncation is now the noise-aware `retrieval_oe.select_num_modes`
   — keep the smallest `K` whose dropped modes each contribute `< ⅓·min σ_ε` to the ToA reflectance.
   It is a *runtime* optimisation (the scan already removed the compile-memory necessity), so the
   solver default is "all `NFourier` modes."

Verified: the float32 suite is numerically unchanged (all `NFourier` is the default); the two OOM
cases now compile — `characterize_geom.py 24` (forward) and `check_jac.py 16` (NFourier=16 `jacrev`
at NQuad=16). The historical analysis that motivated this is kept below.

---

`riccati_solve` runs the K azimuthal modes as a **Python-unrolled loop** (DESIGN §7: "static,
Python-unrolled … each mode keeps its natural ragged `(NLeg−m, N)` tensors — no pad/mask"), so the
**compiled graph contains K independent copies of the whole Kvaerno5 Riccati solve**. Building the
VOCALS forward at **NQuad=24** (Cauchy K=19) **OOMs the XLA→LLVM compiler** ("Cannot allocate
memory" — a *compile-time* graph-size failure, not a runtime array allocation; reproduced
forward-only, no `jacrev`). Compile memory grows ~linearly in K and K grows with NQuad, so it is
~quadratic in NQuad overall. **NQuad=16 is the current CPU ceiling**; 24 streams is not a lot for
sharply-peaked cloud phase functions, so this binds. (GPU has more memory — item D — but is
latency-bound and does not target the root cause.)

**The root cause is graph size, so the fix is to stop unrolling / stop tracing what we don't vary:**

1. **`vmap`/`lax.scan` the Fourier modes (the big lever; DESIGN §7's deferred "scan+pad/mask").**
   Compile **one** mode-block and reuse it ⇒ compile memory **O(1) in K** instead of O(K). Cost:
   pad the ragged `(NLeg−m, N)` tensors to `(NLeg, N)` (modest runtime waste). **Resolves the
   earlier "Cauchy is incompatible with vmap" objection:** it isn't. Run the Cauchy
   azimuthal-convergence test **once** to fix the integer K (it can be a separate, *untraced* /
   low-memory pass), **then** `vmap`/`scan` the now-fixed K modes — the early-stop is no longer in
   the differentiated/compiled path. Alternatively, **if the Cauchy criterion is not buying us
   memory or runtime, drop it** and use a fixed mode count. Decide by measurement.
2. **Static `mu0` (don't trace what we don't vary).** We trace `mu0` "because we can," which forces
   the in-trace associated-Legendre recurrence `_assoc_legendre_neg_mu0_jax` (`P_l^m(−μ0)`) into
   *every* mode-block (DESIGN §7). For a **single-column** retrieval `mu0` is fixed and never
   differentiated, so tracing it only bloats the graph for a geometry-swath-reuse benefit we don't
   use. A static-`mu0` path would precompute `P_l^m(−μ0)` host-side with SciPy (as the `mu_arr_pos`
   tensors already are) and drop the recurrence from the graph. Secondary for memory (the N×N
   implicit solve dominates), but free single-column and composes with (1).

Interacts with item A′ (a robust multi-angle thin-cloud forward wants *more* modes → makes this
OOM worse, and float64 doubles it) and item G's per-mode-grid work (which also wants the per-mode
machinery).

**Confirmed this session (the OOM now bites the *Jacobian*):** at NQuad=16, NLeg_all=128 the
**forward compiles (181 s) but `jacrev` OOMs** when K saturates at 16 — the reverse tape over 16
unrolled mode-blocks is too big. **Workaround adopted for the demo — fix `NFourier=8`, `tol_azim=0`
(no in-loop Cauchy):** the *absolute* mode amplitudes decay fast after delta-M, so NFourier=8
reproduces NFourier=16 reflectance to <1 % (y≈[0.108,0.221,0.286] both), and at K=8 **both the GN
and pool `jacrev` compile cleanly (~460 s each, no OOM)**. This is "Q2" in action: the relative
Cauchy test was saturating over a tiny denominator and misleading us into 16 modes.

**Agreed next work package (user-approved, do as a unit, after the retrieval skeleton):**
- **Q2 — demote Cauchy from in-loop selector to offline diagnostic** (PythonicDISORT-style: return
  per-mode info, user fixes `NFourier`). The right signal is the **absolute** amplitude ‖u_m‖, not
  the relative ratio (which saturates at low signal). *A change of metric (absolute vs relative) is
  on the table — clear the specific diagnostic with the user before implementing.* Fixing `NFourier`
  offline also removes calibrate's wasted full-`NFourier` solve and is the prerequisite for ↓.
- **Q1 — `vmap`/`scan` the (now fixed-K) modes** + optional static-`mu0`. The ceiling-lifter:
  O(K)→O(1) compile memory, unblocks NQuad→48 and non-fragile `jacrev`.

*(Decision this session: demo proceeds at NQuad=16, NFourier=8, NLeg_all=128, autodiff Jacobian —
multi-angle multi-band, no flux-only pivot needed.)*
