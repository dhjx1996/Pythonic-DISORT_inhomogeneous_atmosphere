# Design Decisions

Settled design decisions and their rationale, recorded so they are not
re-litigated or misremembered (e.g. "is the discrete adjoint implemented?").
Open questions live in [`OUTSTANDING.md`](./OUTSTANDING.md).

Status tags: **[SETTLED]** decided and in effect · **[INVARIANT]** must never be violated.

---

## 1. Solver lineage: invariant-imbedding Riccati + implicit adaptive integration  [SETTLED]

The forward solver integrates the **matrix Riccati equation** (invariant imbedding) with
diffrax's **Kvaerno5** (L-stable ESDIRK, adaptive). **Minimising the number of integration
steps (τ-points) is the primary optimisation target** — the forward model runs inside the
retrieval loop, so step count, more than per-step cost, dominates total run time. This driver
recurs throughout the design; the lineage that led here is recorded so the dead ends are not
retried:

- **Magnus + Redheffer star product** ("Report I"): unconditionally *stable* — works entirely
  with O(1)-bounded N×N reflection/transmission/source operators, avoiding the coexisting
  growing and decaying modes of the 2N×2N propagator. But its **step count is set by the
  ballistic eigenvalue**: `K ≳ λ_max·τ_bot`, with `λ_max ∼ 1/μ_min ≈ 45` (NQuad=16), even
  though the observable evolves on the much slower diffusion scale `k_min ≈ 0.02`
  (τ=50, ω=0.99, g=0.85 → ≈2272 steps).
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

**Retained from the rejected work:** the **Redheffer star product** (now in the BC solver) and
the **N×N O(1)-bounded operator** formulation.

*(Distilled from technical_reports/report_star_product_magnus.tex and report_diffusion_domain.tex,
being removed.)*

---

## 2. Numerical-stability invariant: NO POSITIVE EXPONENTS  [INVARIANT]

No intermediate quantity may contain `exp(+λ·τ)` with `λ > 0`, `τ > 0` — this overflows for
thick atmospheres. The Riccati formulation satisfies this by construction (the state stays
O(1); no positive exponents). Any future algorithm change (Magnus, doubling, SVD, …) must
preserve it.

---

## 3. Retrieval grid ≠ solver grid; information is ToA-weighted; retrievable DOF is small  [SETTLED — open sub-questions in OUTSTANDING G]

Three robust statements below. Two earlier claims — an exact **"rank-4 ceiling"** and full
**profile-independence** — are *deliberately not asserted* (no rigorous basis; measured under
biasing conditions). They are tracked in [OUTSTANDING G](./OUTSTANDING.md).

**(a) The adaptive ODE grid is not the retrieval grid.** The Kvaerno5 grid is placed for
*solver accuracy* — it clusters wherever the optics or the Riccati state vary fast: near sharp
optics features *and* at BoA (a stiff R-transient from the steepest quadrature stream,
μ_min≈0.02 settling ~50× faster than the shallowest, then *forgotten* by contraction). In
`adiabatic_cloud_with_drizzle.ipynb`, adding a near-ToA g-spike at τ=1 to a conservative τ=30
cloud raised the adaptive step count 107→186, clustering steps on the spike *and* keeping the
BoA cluster — yet the BoA steps carry negligible retrieval information. **Step placement ≠
information placement.**

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

*(Distilled from technical_reports/boa_step_clustering_report.tex and the tests/supplementary
QRCP/Jacobian scripts — removed; see also the retained adiabatic_cloud_with_drizzle.ipynb.
Detailed tables move to report_riccati_solver.tex during integration.)*

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

> This entry exists because we were previously unsure whether the adjoint had been
> implemented. It has — as autodiff, not as bespoke code.
