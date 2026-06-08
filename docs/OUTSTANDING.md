# Outstanding Problems & Decisions

Open items, kept deliberately prominent. Settled rationale is in
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md).

Tags: **[BLOCKER]** must fix before retrieval works · **[DECISION]** a choice to make ·
**[BUG]** known-wrong behaviour · **[DEFERRED]** wanted, not yet started.

---

## A. Negative ToA radiances — forward model is physically wrong  [BUG] [BLOCKER]

For a realistic forward-peaked cloud phase function (g₁≈0.85), the reconstructed ToA
**radiance** goes negative, even though `flux_up_ToA` and the m=0 mode `u0` stay positive.
The negatives appear in the **Fourier azimuth reconstruction** at the quadrature nodes
(m≥1 modes ring), *before* any μ-interpolation — so they are **not** an interpolation artifact
(linear-spline μ-interpolation still returns negatives), and **not** fixed by the optics
choice in item B (same optics → same solve → same ringing).

- **Likely cause:** no **delta-M scaling** and no **Nakajima–Tanaka (TMS) correction** (both
  deferred). This is the textbook cause of negative intensities for forward-peaked phase
  functions with finite streams; angle-integrated fluxes stay correct, which is why
  flux-based tests pass and the DYAMOND flux lookup tables never exposed it.
- **Next step:** confirm by comparing `u_func(φ)` at the nodes against a `pydisort` reference
  and checking the per-mode `u_m` decay; then implement delta-M (+ likely TMS).
- A radiance-observable retrieval cannot proceed until this is fixed.

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

---

## C. jit-ability of the solver — the retrieval-cost lever  [BLOCKER]

The retrieval needs `jax.jit` to amortise compilation across hundreds of forward/grad
evaluations, but `pydisort_riccati_jax` is currently **differentiable yet not jit-able as a
whole** (it interleaves SciPy-based setup with the traceable solve). Unjitted, every call
recompiles (~135 s, identical on CPU and GPU since compilation is host-side → the GPU sits
idle). Two host-side blockers, both with known fixes:

1. `_kvaerno5_integrate` does `np.asarray(sol.ts)` + dynamic slicing to build `tau_grid`.
   Fix: `SaveAt(t1=True)` (only the final state is needed for ToA) — no host sync, no dynamic
   shapes.
2. `_precompute_legendre` calls `scipy.special` on the quadrature nodes; under `jit` everything
   is a tracer. Fix: feed it the **numpy** GL nodes (deterministic from static N; numpy ops
   aren't traced) — keeps SciPy, just off the traced path.

**Action:** refactor into a non-jit `setup(...)` (quadrature + Legendre tables, run once) and a
jit-able `solve(params)`. With both fixes, a jitted forward measured ~117 s compile then ~5 s
cached; jitted grad ~442 s compile then ~19.5 s cached (≈27× per-eval speedup over recompiling).

---

## D. GPU is latency-bound for this solver  [NOTE]

Cached execution is dominated by **many sequential tiny matmuls** (NFourier modes × 2 sweeps ×
~35 adaptive steps × 5 ESDIRK stages on N×N, N≤8/16). This is kernel-launch-latency-bound, so
a T4 is *not* faster than CPU. Real speed levers: fewer Fourier modes, fewer adaptive steps
(looser `tol`), not the GPU.

---

## E. Retrieval loop not yet implemented  [DEFERRED]

The cost function `J(θ)`, Gauss–Newton/LM iteration, and r_e(τ) profile parameterisation
(report §"Toward Retrieval") are not built. Forward-mode preferred for small p (≤~15 params),
reverse-mode for large p (crossover p≈15–20 at m=10 observations).

## F. Other deferred forward-model features  [DEFERRED]

Delta-M scaling (see item A), Nakajima–Tanaka (TMS) corrections, isotropic internal source
(only the collimated beam is handled), and non-ToA depth evaluation (only τ=0 is returned).

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
- **Multi-mode / angular DOF is contaminated by the missing delta-M (item A).** In
  `adiabatic_cloud_with_drizzle.ipynb` the per-Fourier-mode ‖∂u/∂g‖ is *larger* for several
  m≥1 modes (e.g. m=7 ≈ 0.5) than for m=0 (≈ 0.1) — but m≥1 is exactly where the radiance rings
  without delta-M. So whether higher azimuthal modes carry genuine extra information cannot be
  judged until delta-M/TMS is fixed; the QRCP grids in that notebook sum all modes and inherit
  the contamination.
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
