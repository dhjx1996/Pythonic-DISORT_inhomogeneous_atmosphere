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

**Current default = r_e³-linear (adiabatic)** `jnp.interp(τ, knots, vals**3)**(1/3)`. Chosen because
`r_e³ ∝ LWC ∝ τ` so it is physically natural, **represents the adiabatic prior exactly** (verified
~7e-7), and gets per-segment curvature from the two endpoint values ⇒ **no grid-size coupling**.
It is still C⁰ (kinked at nodes). Baseline of the *previous* linear-class run saved for comparison:
`docs/retrieval_baseline_linear_class.json`.

**The candidates and the axis that separates them — node-based smoothness vs grid coupling:**

- **linear** (`jnp.interp(τ, knots, vals)`) — the "impute-nothing" baseline; C⁰, no coupling.
- **r_e³-linear** *(default)* — C⁰, no coupling, physical, prior-coherent. Imputes the *adiabatic*
  shape per segment (a defensible physical prior, consistent with the S_a we already use).
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
   grid). C⁰ classes (linear, r_e³-linear) keep this; only a C¹ class removes it. It is **second-order**
   (a mild step-count/selection effect), so not worth importing PCHIP's grid-coupling to fix *here*.
2. **probe the class by model comparison, not grid stability.** "Grid doesn't move ⇒ profile near
   prior ⇒ adiabatic class confirmed" is **confounded** — at DOFS≈2 the profile is prior-dominated
   regardless, and the kink artifact pins the grid. The clean test is to retrieve the *same data*
   under {linear, r_e³-linear, PCHIP} and compare fit χ² / posterior: insensitivity ⇒ data can't
   distinguish shapes (linear fine by Occam); a real difference ⇒ data carry shape info, pick by
   evidence. **One profile is not definitive** — needs a multi-profile study.

**How to change:** edit `_re_of_tau` once; forward, `calibrate`, `ode_grid`, Jacobian, the re-meshing
re-map in `gauss_newton_oe`, pool sampling in `select_retrieval_grid`, and `profile()` all follow.
The cleanest *promotion* makes the **state itself** the new class's coefficients (e.g. node values of
r_e³, or basis amplitudes) so parameterisation, prior mean, and display are coincident by construction.

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
   `mu0`-independent, so it moves wholesale into `setup`; the one `mu0`-dependent term
   `P_l^m(−μ0)` is computed in-trace by a **custom associated-Legendre recurrence**
   (`_assoc_legendre_neg_mu0_jax`) so `mu0` can be traced (DESIGN_DECISIONS §7; gate test 21a).

**Mode count via the exact DISORT azimuthal-convergence (Cauchy) criterion** (STWLE2000 §3.7 p.89):
`calibrate_num_modes` returns a concrete `int K ≤ NFourier` from a user-set `ε_azim` (strong
default 1e-3; `0` ⇒ all modes), and `riccati_solve(..., num_modes=K)` runs exactly K modes as a
static Python-unrolled loop — differentiable in both AD modes, jit-able, no `vmap`/`while_loop`
(DESIGN_DECISIONS §7). **AD-mode caveat:** reverse-`grad` (the discrete adjoint, §5) is the default;
forward-`jacfwd` needs `riccati_setup(..., adjoint=diffrax.ForwardMode())` (the reverse default is a
`custom_vjp` that cannot be forward-differentiated). **Residual caveats:** a *callable* BDRF
evaluated at a *traced* `mu0` is not jit-able (it calls NumPy/SciPy on the beam cosine — matrix
BDRFs and no-BDRF are fine); `phi0`, `I0`, and the boundary conditions remain static (baked into
`setup`).

**Deferred micro-optimisation — single-trace `scan`+pad/mask (only if cold-compile K-traces ever
bind).** K is a static int, so the unrolled `for m in range(K)` bakes K into the graph *structure*:
each distinct K pays one cold compile. The optimisation replaces the unroll with a single `lax.scan`
over a fixed `NFourier`, **pads** each mode's ragged `(NLeg−m,N)` tensors to a uniform `(NLeg,N)`
and **masks** the `l<m` rows, so K enters only as a runtime scan-length/mask — one compile serves
every K (re-calibrating K never recompiles). It trades the clean per-mode einsum kernels for
padded+masked ones; it pays off only if many frequent re-calibrations *and* a wide-ranging,
oscillating K *and* those recompiles dominating wall time all hold. In the recommended usage K is
calibrated once per geometry/regime and is stable (a tiny, cached set of K-traces), so this is
deferred until shown to bind.

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

---

## H. Fourier-mode unroll → compile-memory; revisit vmap/scan + static-mu0  [OPEN — found 2026-06]

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
