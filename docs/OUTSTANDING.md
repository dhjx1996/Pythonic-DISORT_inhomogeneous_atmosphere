# Outstanding Problems & Decisions

Open items, kept deliberately prominent. Settled rationale is in
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md). **Resolved / out-of-scope items are collapsed to a
one-line pointer** — the `## letter` headers are retained because both docs cross-reference them by
letter; the full rationale lives in the linked DESIGN section. The genuinely-open items are **K**,
**L**, and **M**. *(Revised 2026-07-02 with the repository refactor — `CHANGELOG.md`; per-knob evidence
in [`hyperparameter_audit_2026-07.md`](./hyperparameter_audit_2026-07.md). Re-synced 2026-07-16
after the ve_rerun→main merge: ve046 canonicalization, retrieval-grid IC, results-manifest
retirement.)*

Tags: **[BLOCKER]** must fix before retrieval works · **[DECISION]** a choice to make ·
**[BUG]** known-wrong behaviour · **[DEFERRED]** wanted, not yet started ·
**[OUT OF SCOPE]** deliberately not pursued now · **[RESOLVED]** / **[NOTE]** → folded into DESIGN.

---

## A. Negative ToA radiances — delta-M + Nakajima–Tanaka TMS  [RESOLVED → DESIGN §6]

Forward-peaked-Mie ToA radiance went negative (truncated forward peak ⇒ m≥1 Fourier ringing); fixed
by opt-in delta-M scaling + the TMS single-scatter correction (`delta_M_scaling=True, NT_cor=True`).
The `f=g_{NLeg}` derivation, the residual-at-extreme-peaks limit, the "lever is more streams" call, the
δ-M+ deprioritisation, and IMS-omitted-by-design are all in DESIGN §6.

### A′. Thin-cloud Mie off-nadir TMS needs many moments  [RESOLVED → DESIGN §6]

Resolved by `NLeg_all ≥ 128` (too few moments ⇒ a Gibbs-oscillating `p_full` ⇒ erratic,
sign-flipping thin-cloud reflectance, since thin clouds are single-scatter-dominated). **Test
coverage added 2026-06-19** (`tests/22_thin_mie_test.py`): a structured gamma-averaged Mie phase
function (2.13 µm, ~70 significant moments) on a thin τ=1.2 cloud, off-nadir (μ,φ) back/side-scatter
envelope, asserting **NLeg_all convergence** (R(128)≈R(192) to 0.3 %, R(32) ~6× off) + **physical
plausibility** (positivity, magnitude, and smoother than the under-resolved field) — the failure mode
is reproduced at NLeg_all=32 (R∈[−1.3, 0.9]). The test asserts plausibility, not pydisort agreement
(a shared low-NLeg_all TMS artifact would pass an agreement check while both are wrong); it was
modelled in spirit on PythonicDISORT's many-moment `pydisotest` test 5.

---

## B. Optics interpolation: τ-axis vs r_e-table  [DECISION SETTLED 2026-06-12 → DESIGN §8]

Settled: keep the `(n_re, NLeg)` Mie–Legendre **r_e-table** (`optics_table.table_lookup`;
miepython-built since the miejax_lite retirement, DESIGN §8) as the production optics path (profile-independent, no τ-placement problem, consistent table-slope Jacobian);
the τ-axis + lagged re-selection is the documented fallback if the lookup-slope Jacobian ever proves
too inexact. The hybrid traced-Mie-ω + HG variant was rejected. Rationale: DESIGN §8.

### B′. The r_e(τ) profile parameterisation lever  [SETTLED second-order → DESIGN §10g; residual deferred]

How the node values become the continuous profile the solver integrates — the single localised lever
`RetrievalForward._re_of_tau` (part of F(x): it *defines what is retrieved*, and `profile()` mirrors it
for display). **Settled:** for reasonable low-order monotonic interpolants the choice is a **minor,
second-order lever** — re5-linear (adiabatic `r_e∝τ^{1/5}`) vs plain-linear differ by <0.1 µm RMSE,
inside the retrieval's own uncertainty (DESIGN §10g model comparison); **re5-linear is the default**
(physically motivated, marginally more accurate). **Residual (low-priority, deferred):** a multi-profile
confirmation (re5≈linear shown on 2 profiles only) and the C¹ **PCHIP** class (deferred until the
node-count/DOF supports it). The deeper alternative — **leave interpolation behind for a low-dim
shape/EOF basis** — is **CLOSED (user, 2026-06-21): rejected.** An EOF-from-ensemble basis over-constrains
every retrieved profile to the training-distribution shapes (`span{φ_i}`), compromising the
**minimally-constrained** feature that distinguishes this method from the adiabatic / 2–3-param literature.
The free sensitivity-placed-node parameterisation is **retained**. If node-difference wiggles ever warrant
suppression, the lever to *research* is a **2nd-difference (curvature) Tikhonov penalty** — not needed now:
the demo swings (e.g. idealized §5) were an artifact of a deliberately **loose σ_base=10**, and a
climatologically-tight base prior (DESIGN §11; VOCALS r_base MAD≈1.4 µm) already removes them. (This
supersedes DESIGN §3a's "smooth-basis route, left open.")

**Update 2026-07-11 — the "not needed now" premise is partially refuted (idx-110, ve046).** On a *real*
low-SIC shielded-base profile the tight, depth-decreasing σ_base does not suppress the swing — it *creates*
a dr_e/ds **kink** at the data/prior seam: the data-pinned top holds truth (~18 µm) while the base is yanked
to the climatology *mean* (~6 vs truth 10.2). Root cause: the prior kernel `exp(−|Δτ|/ℓ)` is Ornstein–Uhlenbeck
(first-difference smoothness only, C⁰/non-differentiable paths) → it never penalises **curvature**. Two candidate
fixes, both real, both risky: **(1) depth-increasing correlation** (tie the shielded base to the data-rich column
above, not the prior mean) — attacks base *accuracy*, but fights the existing depth-decreasing σ (needs σ_base
loosened too), imposes a vertical-coherence bias (biases genuinely-adiabatic bases high), and risks a near-singular
deep `Sa` block; **(2) 2nd-difference (curvature) Tikhonov** (the lever above) — one-line GN insertion
(`H_gn += λ·L₂ᵀL₂`, `rhs −= λ·L₂ᵀL₂·x`; no solver / positive-exp risk), removes the kink (C¹), but needs the
**non-uniform** 3-pt stencil (QRCP nodes are irregular), a *small* λ (else it washes out real drizzle/entrainment
structure — the minimally-constrained feature), and it smooths the *symptom* while base accuracy stays wrong.
**Hard ceiling:** neither adds information — the shielded base (SIC≈13) stays prior-dominated, so its posterior σ
stays wide/biased; this targets *plausibility*, not accuracy. Diagnostic before implementing: idx-110 deep-node
averaging kernels / `data_fraction`.

**Why the over-smoothing worry is smaller than it looks (user, 2026-07-11):** the true cloud is *not* smooth, but
the **retrieved** profile ought to be — **radiative smoothing** band-limits what the measurement can resolve, so
sub-resolution structure is unretrievable *by construction*. A curvature penalty therefore enforces the estimator's
**correct smoothness class**, it does not discard retrievable signal (there is none at that scale); the kink is a
genuine artifact of the C⁰ OU prior, not real structure the penalty would erase. This reframes fix (2) from "risky
cosmetic" to "well-posed regularisation," and motivates turning it **ON** (not merely opt-in-for-research) — with a
*calibrated* λ (small; the penalty self-localises to the data-poor deep nodes because it only bites where
`KᵀSε⁻¹K` is small). **Counter-caveat (user, same day): self-localisation only protects if λ stays small —
too strong a penalty smooths *across* the radiative-resolution scale and genuinely papers over *retrievable*
structure (e.g. the well-constrained top gradient). Calibrate λ to relax the data-poor deep kink while leaving
high-`KᵀSε⁻¹K` nodes ≈ fixed; do NOT choose λ for maximal smoothness.**

**Status: 2nd-difference Tikhonov IMPLEMENTED 2026-07-11** as `curvature_lambda` in `_gn_inner`/`gauss_newton_oe`
(+ `CURVATURE_LAMBDA` env, `retrieval_worker`). Non-uniform 3-pt stencil (`_second_difference_operator`), penalty
`P=λ·L₂ᵀWL₂` (W = interval quadrature weights → grid-independent curvature energy ∫(f″)²ds; a raw L₂ᵀL₂ scales
~1/h⁴ and makes λ non-transferable across profiles) over the log-r_e node+r_base block, τ_bot excluded;
**λ=0 ⇒ bit-identical** to the un-penalised solve. Linearised calibration (2026-07-11, from stored posteriors):
energy-weighted **λ≈0.3–1** relaxes the idx-110 kink 72–76 % (r_base +0.24→+0.73 µm toward truth) while moving
well-constrained thin-cloud nodes only ~0.15–0.38σ — the accuracy-vs-over-smoothing knob to set per campaign.
Depth-increasing correlation (fix 1) is now **implemented** (2026-07-16) as the `CORR_LENGTH` env knob
and **quantified** in §L "Prior top–base correlation": our prior imposes top↔base corr ≈ 0.135
(`corr_length=0.5`) where the VOCALS in-situ covariance (BP2026 recipe) is ≈ 0.84; 0.135 stays the
primary configuration, strong-corr (0.84) sensitivity campaigns queued (`hpc/AGENT_strongcorr_pipeline.md`);
flagged to tune for ops.
**Adopted in production (2026-07-15/16):
the canonical ve046 campaign (`runs/_ve046_tik_fr_parts`, summary
`docs/cached_results/retrieval_summary_ve046.json`) ran with λ=1.0; §15's retrieval-grid IC uses those
sidecars (with the curvature term excluded from the IC prior — DESIGN §17).**

**OPEN (2026-07-21): the penalty reference-curvature question.** The implemented (and published)
penalty is on the **absolute** profile, `½λ‖L₂ x‖²`, whose zero-penalty set is the constant+linear
(straight-in-log) profiles. The re5-linear adiabatic prior mean is *curved* in log-r_e, so this form
**penalises the adiabatic bend itself** — it biases the retrieval toward straight-in-log. Noticed while
writing the paper (the free-node method quietly carries a mild anti-curvature bias).
- The obvious "fix" — penalise the departure `½λ‖L₂(x−x̄)‖²` (adiabatic mean → null space, `S_a⁻¹→S_a⁻¹+P`
  same mean) — does **not** cleanly solve it: it only relocates the bias from "toward straight" to "toward
  the *climatological* adiabat's curvature," and that reference is a poor per-cloud proxy. In log-space the
  adiabatic curvature is dominated by the base and scales as `⅕[1−(r_top/r_base)⁵]²` — a **5th power of the
  endpoint ratio**, so it swings ~an order of magnitude across the plausible ratio range (ratio 1.4→2.0 gives
  base-curvature ∝ 19→960). Even the best-fit (oracle) adiabat departs substantially in curvature from the
  climatological prior mean.
- Why it is probably second-order either way: that curvature lives **at the base = the veiled region**,
  where the `S_a` prior already pulls x→x̄ and shape is conceded unretrievable (DESIGN §14/§15). Absolute
  and departure forms differ mainly *there*; the data-constrained upper cloud (mild curvature, data-dominated)
  is barely touched. Residual exposure = the transition node at the data/veil seam; the 0.15σ figure above is
  one thick profile (idx-110) and does not bound this across thin/mid/deep regimes.
- **Decision (2026-07-21): keep the published absolute form; do NOT switch.** Not because A is better —
  it isn't claimed to be — but because it is what the campaign ran, and neither reference is assumption-free
  (de-kinking vs impose-no-shape are in tension). The paper carries the adiabatic-shape penalisation as an
  explicit caveat (user-authored).
- **The way to actually settle it (deferred):** retrieve a regime-spanning ~6 profiles under (A) absolute,
  (B) departure, and (C) α→0 pure de-kink, overlay the profiles in the *data-constrained* region. Flat ⇒
  report as an implementation detail, results insensitive to the penalty reference (the stronger claim). Not
  flat ⇒ genuine finding; then consider the principled shape-neutral target: reference curvature to the
  **self-consistent adiabat through the retrieval's own endpoints** (penalise non-adiabaticity, not
  deviation-from-climatology; nonlinear, more machinery).

---

## C. jit-ability of the solver — the retrieval-cost lever  [RESOLVED → DESIGN §7]

Resolved by the host-side **setup / traceable-solve split** (the composable seam
`riccati_setup`/`riccati_solve`/`eval_radiance`; the one-shot entry delegates, 5-tuple bit-for-bit).
Cold→warm caching confirmed. The traced/static contract and the two host-side blockers (σ-grid build,
`_precompute_legendre`) are in DESIGN §7; the `lax.scan`-over-modes follow-up is §H below.

---

## D. GPU is latency-bound (single column) — batch across columns  [NOTE → DESIGN §13]

Cached single-column execution is kernel-launch-latency-bound (NFourier × 2 sweeps × ~35 steps × 5
stages of tiny N×N matmuls), so per column the GPU is *not* faster than CPU. The retrieval is
embarrassingly parallel across columns, and `vmap` over a batch flips it (crossover B≈64, ~53× at
B=4096). The measurement and the batch-crossover table are moved to **DESIGN §13**.

---

## E. Retrieval loop  [RESOLVED → DESIGN §10]

Implemented in `src/pydisort_riccati_jax/retrieval_oe.py`: cost `J(θ)`, Rodgers GN/LM (`gauss_newton_oe`), the
normalized-depth `_re_of_tau` parameterisation, the Tikhonov priors, QRCP grid selection, posterior
UQ/DOFS/SIC, and the OSSE harness. It is a **joint** retrieval of `[r_e(s-nodes), r_base, τ_bot]`
(DESIGN §10). Demonstrated on thin (RF11) and thick (RF03) in the VOCALS notebook.

---

## F. Other forward-model features — isotropic source, non-ToA depth, adjoint robustness  [OUT OF SCOPE]

Out of current scope (user, 2026-06-19); recorded so they are not mistaken for undiscovered gaps:
- **Isotropic internal source** and **non-ToA-depth evaluation** — only the collimated beam and the
  τ=0 ToA upwelling field are handled. Wanted eventually, not on the current path.
- **Adjoint robustness (minor):** reverse-mode `grad` can NaN (singular lineax solve) on an
  *aggressively steep* synthetic r_e profile — not real VOCALS-REx (finite-slope); quick fix if ever
  needed is `AutoLinearSolver(well_posed=False)`.

(Delta-M/TMS are implemented and IMS is omitted by design — DESIGN §6.)

---

## G. Retrieval information content — what is actually retrievable  [RESOLVED 2026-06/07 → DESIGN §14/§15]

**Closed by the definitive all-125 IC profiling (DESIGN §14, on the FIXED delta-M/TMS forward) and
the full-retrieval campaign (§15/§16):** the angular-DOF question (angular novelty quantified;
per-mode grids rejected — sub-item below), profile-(in)dependence (measured across the 125-profile
population, regime-vs-τ), and multi-band saturation (N_sat = 7 bands; data-greedy order from the
NK1990 bispectral pair) all have population-level answers. The remaining *conditionality* (single
solar geometry μ0=0.9) is tracked in §L. The historical caveats below are kept as the record of
what the pre-§14 analyses could and could not support:

- **No rigorous "rank-4 ceiling."** Earlier SVD/QR analysis found ~4–6 dominant singular
  directions, but (i) the hard upper bound is the stream count N (=8 at NQuad=16), with no
  symmetry argument reducing 8→4; (ii) "4" is a soft, threshold-dependent count (the >0.1%
  threshold already gives 6); (iii) it was measured emphasising the m=0 mode, without delta-M,
  at N=8, and for a single geometry/thickness — all of which can suppress it. Claim only "small
  DOF," not "4."
- **Multi-mode / angular DOF was contaminated by the missing delta-M (item A — since fixed; the
  §14 re-run superseded this).** In
  `adiabatic_cloud_with_drizzle.ipynb` the per-Fourier-mode ‖∂u/∂g‖ is *larger* for several
  m≥1 modes (e.g. m=7 ≈ 0.5) than for m=0 (≈ 0.1) — but m≥1 is exactly where the radiance rang
  without delta-M. So whether higher azimuthal modes carry genuine extra information could not be
  judged until delta-M/TMS was fixed; the QRCP grids in that notebook sum all modes and inherited
  the contamination. **Done: the §14 definitive profiling IS that re-run** (delta-M/TMS on, NLEG_ALL=1536,
  all 125 profiles).**
- **Profile-independence unproven.** Demonstrated only for a localised g-spike on one smooth
  adiabatic base; the angular-collapse depth depends on ω/band; globally different profiles
  (thin, multi-layer, inversion) untested.
- **Multi-band gains are real but saturating** (Coddington, Pilewskie & Vukicevic 2012):
  additional bands add information diminishingly due to inter-band correlation; the
  vertical-resolution gain rests on penetration-depth diversity (Platnick 2000), which is
  modest. Do not assume multi-band lifts the DOF far.

*(G-core was answered by DESIGN §14; the bullets above and the Starting point below are
historical background only.)*

*Starting point:* the prior multi-mode / full-radiance rank study (three tiers — baseline u₀,
full-u with all 16 Fourier modes = 128 rows, and NQuad=32; ToA rank stayed 4 in all, with
per-mode BoA decay ‖J^{m=1}‖≈9e-12, ‖J^{m=2}‖≈1e-16) lives in the removed
`technical_reports/boa_step_clustering_report.tex` — recoverable from git `99fb971`. **Treat its
conclusions as contaminated** (built on the un-delta-M'd m≥1 modes); re-derive, don't cite.

### Per-mode ODE grids and the retrieval grid  [INVESTIGATED 2026-06-20 → keep m=0; DESIGN §3a]

**Verdict: rejected. The m=0 ODE grid stays the sole retrieval-grid pool.** The hypothesis — that
the discarded m≥1 grids carry complementary vertical information, so the "best" pool is the **union
of the non-negligible (Cauchy-K) modes' grids** — was tested directly
(`per_mode_grid_investigation.py`, a faithful monkeypatch retaining every mode's forward ODE
grid; pruned 2026-07 — git history) and does **not** hold: the union is neutral-to-harmful on every VOCALS
case (OCI 2 % noise, `filter_threshold=0.5`).

- **Placement (TEST 1).** Every mode m=0…15 has a near-identical grid (~17–19 steps; same
  near-ToA/mid/deep split ≈2–3 / 5 / 10–11). The ~10–11 deep (s>0.85) steps in *every* mode are the
  universal BoA imbedding boundary layer (≈zero info, §3a). **The modes densify the same regions;
  they do not place steps at new informative depths** — because optics ω(τ), gₗ(τ) are *shared*
  across modes (every mode integrates the same Riccati structure), so a real optics feature already
  varies the m=0 state. Confirmed: the union-only nodes are overwhelmingly **mid/deep** re-samples
  (THIN 8 near-ToA / 27 mid / 6 deep) of already-covered, prior-dominated depths — not new near-ToA
  features.
- **Pool (TEST 2).** Union ≈ 3× m=0 (THIN 17→56, THICK 18→53), but the extra columns are
  near-collinear near-ToA duplicates.
- **Selection + recovery (TEST 3/3b) — decisive.** Offering QRCP the denser pool makes it
  **over-concentrate near ToA and abandon the deep coverage** the sparser m=0 grid was forced to
  provide, with equal-or-worse truth recovery:

  | case | m=0 grid (k: dense/near-base RMSE) | union grid (k: dense/near-base RMSE) |
  |---|---|---|
  | THIN  RF11 (τ1.2)    | k5: 0.388 / 0.391 µm        | k6: 0.384 / **0.427**          |
  | THICK RF03 (τ23)     | k3: **0.905** / 1.505       | k5: 0.915 / 1.531 (χ² 0.32→0.19, no recovery gain) |
  | RF10 shielded (τ4.9) | k5: **0.516** / **0.723**, drop-cap 58 % | k6: **0.646** / **0.977**, drop-cap **187 %** |

  RF10 is the clear harm: the union bunched 5/6 nodes into s≤0.21, under-sampled the weakly-
  informative base, and the over-concentrated near-ToA fit overshot the drop (cap 58 %→187 %, RMSE
  +25 %). The m=0 grid's relative near-ToA sparseness is a **feature** — it forces QRCP to spread
  nodes across the informative depth range.

This also disposes of the two sub-ideas. A **per-mode sensitivity decomposition** for selection is
moot — the summed-Jacobian QRCP already uses the *full observable* (the right measure), and the
modes carry no independent vertical sensitivity here. The **intersection** is strictly worse than
m=0 (a lossy subset: it breaks the superset guarantee, shrinks as modes are added, and collapses
onto the zero-information BoA layer).

**Flip condition — the one regime where this reverses.** The modes are redundant *only* because the
high-order Legendre moments are slaved to r_e(τ) through the fixed-`v_eff` Mie table. Extend the
forward to a **τ-varying effective variance v_e(τ)** (size-distribution width changing with depth —
entrainment, drizzle onset at base) and v_e modulates exactly the **high-order** moments
(cloudbow/rainbow sharpness) *decoupled* from the low-order asymmetry; since Fourier mode m couples
only to moments l≥m, high-m modes would then resolve a depth m=0 smooths over. That information
lives in the **polarized** cloudbow (scalar intensity plateaus at DOFS≈1), so the per-mode grids
become live **only** for a polarized, v_e(τ)-resolving forward model. Until then m=0 is a complete
pool. *(Recorded for the v_e side-project; revisit there.)*

### Re-mesh instability ⇒ correlated node basis  [RESOLVED → DESIGN §10h; EOF residual in §3a]

The lagged-re-mesh *placement* flapping (QRCP re-pivoting near-collinear node columns at a moved
linearization point) is **resolved in practice**: the default is `max_n_outer=2`, but the **χ²-gate**
(`remesh_if_chi2_red_gt` — re-mesh only on structural misfit) plus **normalized-depth** `s` (which
removed the deep-node wide-`r_e⁵` leverage that misled placement) make re-mesh fire **only very rarely**
— effectively select-once for well-fit VOCALS retrievals (DESIGN §10h; the `n_outer=2` experiment did
not help). `k > DOFS` is **kept on purpose** (the prior-filled margin is a feature, DESIGN §10f), *not*
cut to DOFS. The deeper fix it gestured at — leaving the correlated node basis for an **orthogonal
shape/EOF basis** — is an *architecture* question, not an instability: its real payoff is *placement
stability* (nothing to re-pivot), **not** "k = DOFS" (you still keep a prior-filled margin). That
alternative is logged in DESIGN §3a (the smooth-low-dim-basis route, "left open") and B′.

### Auto-select the node count `k_active`  [IMPLEMENTED → DESIGN §10f]

Implemented as `retrieval_oe.auto_k_active`: the noise-aware whitened-QRCP **filter**
`f_i = r_i²/(1+r_i²)` (with `Σf_i ≈ DOFS` as a built-in cross-check), wired into
`select_retrieval_grid(k_active=None)`; `filter_threshold=0.5` since the 2 %-noise re-sweep
(DESIGN §10f — the earlier 0.25 was tuned on the retired 3 % noise). DOFS left the *selection*
path (now an info-content diagnostic only).

---

## H. Fourier-mode unroll → compile-memory  [RESOLVED — 2026-06 → DESIGN §7]

The Python-unrolled K-mode loop put K copies of the Kvaerno5 solve in the graph and OOM'd the XLA
compiler (NQuad≥24 forward; NFourier=16 `jacrev` at NQuad=16). Resolved as a unit: **`lax.scan`** over
padded `(NFourier, NLeg, N)` per-mode tensors (mode body compiles once, O(1) in K), **static μ0**
(`P_l^m(−μ0)` precomputed host-side, the in-trace recurrence removed), and the **S_ε** mode selector
replacing the relative-Cauchy test. Details: DESIGN §7.

---

## I. Polarized single-scattering cloudbow forward — v_e / cloud-top r_e  [OUT OF SCOPE — deferred]

A second observable orthogonal to the scalar ToA radiance: the polarized cloudbow — the only accurate
lever for droplet effective *variance* v_e, and a sharpener for cloud-top r_e. Prototyped and validated
on the **`ve_retrieval`** branch (`src/polarized_mie.py`, `src/cloudbow_retrieval.py`). **Set aside
until further notice** (user, 2026-06-19); full assessment + merge plan:
`ve_retrieval:docs/ve_retrieval/ASSESSMENT.md`. **Stale pointer (2026-07-16 audit): the `ve_retrieval`
branch no longer exists locally or on origin — the prototype survives only in external clones/backups;
locate or re-derive before reviving this item.** Its instrument-noise counterpart (HARP2 / DoLP) is
parked in §K.

---

## J. BDRF specified as [ρ/π] — π-too-dark mislabeling  [RESOLVED 2026-06-17 → DESIGN §9]

Fixed: removed `/π` from every `BDRF_Fourier_modes` call site and regenerated the reference `.npz`;
all affected float32 tests pass at the physically-correct albedos. The convention (`[ρ_s]`, **not**
`[ρ_s/π]`) and the single-bounce verification are documented in DESIGN §9.

---

## K. Measurement-noise model — shot term (Option A) and HARP2/polarized noise deferred  [DECISION / DEFERRED]

The infrastructure is **built and settled** (`src/noise_model.py`; σ(ρ) = calibration-relative +
floor in quadrature; the OCI-SWIR default; `osse_observation(noise=)` + `NoiseModel.Se`; default
noiseless — see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) §12). These pieces are **open**:

- **Shot term (Option A) — REMOVED from the code (2026-07-10 ponytail audit), pending OCI
  SNR-at-L_typ.** The `snr_ref`/`rho_ref` coefficients sat wired-but-off for weeks because OCI's
  SWIR SNR-at-L_typ table could not be cleanly sourced: the PACE MRD (`PACE-SYS-REQ-0019L`) §3.7
  tables are embedded **images** (no extractable text), the SNR requirement lives in an external
  `.xlsx` (`oci_functional_requirements_table2`), and converting a radiance-domain SNR to our
  reflectance units further needs per-band solar irradiance F₀ + a reference geometry.
  **To resolve:** obtain the OCI SWIR SNR + L_typ per band, convert L_typ→ρ_ref, and re-add the
  quadrature term ``ρ·ρ_ref/SNR_ref²`` inside ``NoiseModel.sigma`` (a one-line formula; git
  commit cb8abea9933f074b2e8b0f4fd42fa9134ba94f8a and earlier has the old fields).
  Low urgency: clouds are bright ⇒ calibration-dominated ⇒ the shot
  term is a small correction in our regime.

- **Calibration error is systematic, not random (diagonal-Se caveat).** `k_cal·ρ` is an absolute-gain
  uncertainty, correlated across a scene's pixels; a diagonal `Se` treats it as independent. Fine for
  the **single-column** OSSE (sets the misfit scale + χ²-gate floor); a **multi-pixel / scene**
  retrieval should model the correlated part (off-diagonal `Se` or a separate bias term). Revisit when
  the retrieval goes beyond one column.

- **HARP2 / polarized (DoLP) noise — deferred with the cloudbow observable.** HARP2 (VIS 0.44–0.87 µm,
  10/10/60/10 view angles, **0.5 % DoLP**, 1–3 % radiometric) cannot measure the SWIR retrieval bands,
  so it attaches to the **polarized single-scattering cloudbow** observable (§I), not the current
  scalar SWIR retrieval. When that observable lands: add a polarization-aware path (per-observation I
  vs Q/U/DoLP type) and a `harp2`/`spexone` preset (SPEXone DoLP 0.3 %). Out of scope until then
  (user-set 2026-06-19: ignore polarization / v_e for now).

- **Notebook adoption — DONE (2026-06-19).** All four OSSE `Se` sites (§5 idealized / §8 thin joint /
  §12 thick / §13 sub-adiabatic) now build `Se = nm.oci_swir().Se(y)` — the PACE
  OCI-SWIR model (calibration-relative ~2 %) — replacing the hand-picked `0.03·max(|y|,0.02)` floor;
  `noise_model` is imported and the §8 markdown + §11b document the change. The OSSE stays **noiseless**
  (Se is the assumed weighting/UQ covariance only). `select_num_modes`'s mode-selection Se
  in the **IC workers** was fixed 2026-07-02 (flat `0.005²·I` → the measured-radiance OCI Se;
  audit §2.1 — the notebook's small 2-band demo keeps its own local floor.

---

## L. Post-refactor validation + audit flags  [updated 2026-07-16]

The 2026-07-02 refactor (`CHANGELOG.md` = the HPC validation brief; per-knob evidence in
[`hyperparameter_audit_2026-07.md`](./hyperparameter_audit_2026-07.md)):

- ~~**Golden-gate sign-off.**~~ **RESOLVED (2026-07-06):** L1/L2 equivalence gates PASSED on the
  cluster; float32 suite 68/68 (CPU) + float64 26/26 (GPU) vs PythonicDISORT. The 3-profile
  golden cross-check is **retired** (stale different-grid reference — the retrieval is
  grid-sensitive across code versions, so it was never a valid cross-version gate; DESIGN §16).
- ~~**IC re-run decision.**~~ **RESOLVED (2026-07-06), then SUPERSEDED (2026-07-14/15):** the
  refactor-re-run IC bundle was subsequently ruled **erroneous** — adaptive-integrator Jacobian
  texture inflates dense-grid DOFS/SIC ~2.7× at production tolerance. The canonical IC product is
  now the **retrieval-grid IC** (`docs/cached_results/ic_retrieval_grid.json`, computed on the
  ve046 canonical sidecars' QRCP grids), gate-verified by the forced-k / dual-linearization /
  dual-tolerance demo and the signed-kernel tol6-vs-tol7 gate — DESIGN §17; notebook §15 rebuilt
  on it (2026-07-15/16; overflow in `docs/IC_extra.ipynb`).
- ~~**FR bundle transfer + capstone finalization.**~~ **RESOLVED (2026-07-16):** the capstone
  runs on the canonical ve046 sidecars (`runs/_ve046_tik_fr_parts`); metrics analyzed and
  cached (`docs/cached_results/retrieval_summary_ve046.json`, notebook §16 + `docs/FR_extra.ipynb`).
  The original v_e=0.10 `_fr_parts` bundle is superseded (DESIGN §16 header).
- **Optimizer vNext [IMPLEMENTED 2026-07-10 in `_gn_inner`].** From
  `docs/optimizer_critique.txt` (git `191afed`) + the batch-3 backtrack observations, all four
  accepted improvements landed: (i) cost stagnation now tests the **monotone total cost J**
  (`rel = (J−J_new)/J`), not the data-only φ (kills the trade-φ-for-prior false-positive class the
  old `abs(rel)` sign fix only narrowed); (ii) `xtol` tests the **actual clamped step** `x_new − x`,
  not the proposed `dx`, so a boundary-pinned solve exits step-small instead of via backtrack
  exhaustion; (iii) **gain-ratio (Nielsen) λ adaptation** — reject grows λ geometrically (ν: ×2,×4,…),
  accept eases by `max(⅓, 1−(2ρ−1)³)` — to shorten the expensive reject/backtrack chains; (iv)
  SciPy Cholesky (`assume_a='pos'`) + `H_gn = lhs_base + Sa_inv` hoisted out of the backtrack loop.
  Behaviour verified by `tests/23_retrieval_test.py` (23g truth recovery, 23h L1-resume equivalence,
  23i τ_bot); population-scale validation delivered by the ve046 campaigns (125/125 canonical). On
  **trust-region-vs-clamping**: a formal box-constrained TR is *not* warranted here — the bounds
  bind rarely (essentially the `re_max`-edge class), projected-step LM is standard practice for
  that regime, and TR radius control is functionally equivalent to LM damping (it would not
  remove the reject evals); revisit only if boundary-active retrievals become common (then a
  projected/reflective TR à la TRF is the right form).
- ~~**v_e-corrected OSSE [branch `ve_rerun`].**~~ **RESOLVED (2026-07-15; branch merged to main
  and deleted 2026-07-16):** the config-A re-run at `OSSE_VEFF=0.046` ran to 125/125 canonical
  and the notebook IC section (§15) was rebuilt on it. The glory question
  is answered — more sharply than anticipated: the v_e=0.10 world's 1.038 µm exact-backscatter
  anchor (×2.6) does not exist at v_e=0.046; sharp VIS glory-*ring* spikes (0.55 µm ≤×3.1,
  0.67 µm ≤×2.6, ~2.5° off backscatter) replace it. Feature-anchored angular information is
  v_e-conditional; Finding B is re-established on the v_e-robust angular envelope (DESIGN §17).
- **μ0 = 0.9 conditionality [scope].** All published IC/FR numbers are single-geometry; quantify
  DOFS/band-ranking sensitivity to μ0 (cheap spot-check) and adopt μ0 binning for operational
  per-scene work (compile-per-bin; STRATEGY §4).
- **Un-swept second-order knobs [low].** `corr_length` (prior smoothness) and `margin` (+1 node)
  have no dedicated sweep at the 2 % noise model; cheap 2-case sweeps when the HPC is idle.
- **Prior top–base correlation under-specified vs in-situ [flag, deliberate — tune for ops].**
  `corr_length=0.5` fixes the *only* knob setting the prior correlation between cloud-top and
  cloud-base r_e: `corr = exp(−1/0.5) ≈ **0.135**` (top↔base). The VOCALS-REx in-situ covariance
  (BP2026 recipe: mean over top/bottom 25 % of each profile, log space, 125 profiles) gives
  `corr(ln r_top, ln r_base) = **0.84**` (bootstrap 95 % CI [0.79, 0.88]; Spearman 0.85). We
  under-couple by design. Literature (all in `cloud_profile_retrieval/`): **KV2012** and **KR2012**
  use a *diagonal* `S_a` (corr 0); **BP2025** couples via the prior *mean* (`r_base,a=0.70·r_top`);
  **BP2026** — the log-space source we follow — operationalizes it in the *covariance*
  (eq 3, `S'_a=Cov(ln[r_top,r_bot,τ,IWV])`, the empirical ~0.84). **Decision 2026-07-16: keep the
  untuned 0.135 for research** — a weak tie lets the genuine depth information-collapse (shielded
  base, low DOFS/SIC) show through in the posterior instead of being masked by a strong prior tie
  whose 0.84 is a single-campaign (SE-Pacific marine Sc) statistic of uncertain generality. This is
  the quantified form of §B′ "fix (1) depth-increasing correlation" — **now implemented as the
  `CORR_LENGTH` env knob** (`scripts/retrieval_worker.py`, 2026-07-16; unset → ℓ=0.5, corr 0.135;
  `5.7355` → corr 0.84; main-retrieval prior only, stamped into each sidecar). **The sensitivity
  runs are scheduled:** three strong-corr (0.84) campaigns — FR, adiabatic, v_e-mismatch — queued
  in strict order behind the weak-corr adiabatic ablation (`hpc/AGENT_strongcorr_pipeline.md`).
  0.135 remains the default/primary configuration. **For operations this must still be tuned**
  (sweep `corr_length`, or replace `make_adiabatic_prior`'s kernel with a BP2026-style empirical
  `S_a`). The `ve046_adia` ablation (2026-07-16, `runs/_ve046_tik_adia_parts`) is where this
  matters most: base r_e rides on ~1 shielded DOF, so the 0.135-vs-0.84 gap most directly shapes
  its base/LWP results.
- **Spectral surface albedo [low].** Constant Lambertian 0.06 across 0.55–4.05 µm is crude
  (SWIR sea albedo ≈ 0.02); secondary under bright cloud — revisit if dark-scene bands matter.
- **Shot-noise term** — removed from the code pending OCI SNR tables (§K; re-add is one line).

---

## M. Encode uncertainty about the imposed structural assumptions  [DEFERRED — research]

*(Brainstorm 2026-07-21; motivated by notebook §17 "structure spectrum" + Finding 5.)*

**The gap.** The structure knobs — model class `m` (k=1 vs free-node), correlation length `ℓ`,
curvature strength `α` — are **fixed conditioning variables** `θ=(ℓ,α,m)`. OE delivers `p(x|y,θ)`
at a point and Rodgers `Ŝ` is the covariance *conditional on that point*; there is no prior `p(θ)`
and no marginalization `p(x|y)=∫p(x|y,θ)p(θ|y)dθ`. This is precisely why structural error is absent
from `Ŝ` and why the posterior is under-dispersed (§17 Finding 4: PIT 0.81→1.00 along the spectrum,
42 % of truths outside the 95 % ellipsoid — the §13b representation-bias blind spot). Encoding this
uncertainty **is** the principled fix for that under-dispersion.

**The enabling insight (why it's more than error-inflation).** §17 Finding 5 — the operating point
lives in the **radiance null space**, so `p(θ|y)≈p(θ)`; the data cannot sharpen the structure
hyper-posterior, hence the extra posterior width is inherently **hyper-prior-sourced**, not a
likelihood effect. Collapse the three dials to one spectrum coordinate `s∈[0,1]`, put `p(s)`, weight
by the evidence `Z(y|s)` (closed-form `χ² + logdet(K S_a Kᵀ + S_ε)` in the linear-Gaussian limit).
Then evidence-weighting **auto-recovers Finding 2's regime-dependence**: where structure is
radiance-loud (thin/mid, Finding 4 χ²ᵣ strain) `Z` falls with `s` → downweight structure → leans
free-node; where radiance-silent (deep veiled core) `Z` is flat → weights ride `p(s)` → posterior
becomes a mixture spanning adiabat↔free-node, wide and honest at depth. Turns "radiances cannot
choose the operating point" into "they choose to the extent they can, and admit ignorance where they
cannot."

**Menu, ranked by value-for-effort:**
1. **Law-of-total-variance retrofit [cheapest; reuses the 4 §17 campaigns].**
   `Var(x|y)=E_θ[Var(x|y,θ)] + Var_θ[E(x|y,θ)]`; the second (between-structure) term is the missing
   dispersion, computable *today* from the retrieved profiles already in hand. Add to `Ŝ`, re-run
   PIT. **This is the recommended first experiment** — one afternoon, no new campaigns; validates or
   kills the premise before investing in hierarchical machinery.
2. **Evidence-weighted BMA over a 1-D `s`-grid.** Full version of (1): mixture mean/cov,
   `w_k∝Z(y|s_k)p(s_k)`; multimodal at depth (breaks the Gaussian-ellipsoid PIT — report the mixture).
   Covariance part is doable on the **§15 sidecar Jacobians** (linear-Gaussian, cheap) without
   re-running nonlinear GN.
3. **Empirical-Bayes hyper-prior from the 0.84 CI.** The bootstrap CI `corr∈[0.79,0.88]` maps through
   `ℓ=−1/ln(corr)` to a ready-made `ℓ∈[4.24,7.82]` (sampling uncertainty, narrow). Distinct from the
   *epistemic* "does coupling apply to this cloud" spread — the full spectrum `ℓ:0.5↔5.74`
   (`corr:0.135↔0.84`, §L). Carry both; they answer different questions.
4. **Joint retrieval of `ln ℓ, ln α` as state.** Propagates structure uncertainty via the joint
   covariance off-diagonals; DOFS≈0 for the new components (null space) — the honest outcome. Awkward:
   `ℓ,α` enter the *prior*, not the forward operator → profiling cleaner than differentiating.
5. **Robust/credal priors (Γ-minimax).** No `p(θ)`; report the envelope of any functional (LWP,
   `r_base`, deep-`r_e`) over `ℓ∈[0.5,5.74]`. Envelope width = structural uncertainty; pairs with the
   LWP-bias comparison vs BP2025/26.
6. **Decision-theoretic operating point.** Formalizes Finding 5's "priced bias–variance policy":
   Bayes-risk min under model uncertainty (loss Ŵ₁, truth-population dist, `p(s)`) — makes the
   "deep-Sc population prices it the other way" remark quantitative.

**Limits to state in any writeup:** (a) turtles-up — `p(s)` is itself asserted structure (a milder
assumption than a fixed point, not zero); (b) null-space ⇒ the added deep width is *whatever `p(s)`
puts in — report sensitivity to it; (c) multimodality vs Gaussian `Ŝ` — (2)/(6) force mixtures,
(1)/(3)/(5) keep a Gaussian/interval story; (d) `α`↔`ℓ` are partly redundant (both →∞ collapse to a
template, §17) so the 1-D `s` coordinate is the right object to put uncertainty on, not a 2-D
`(ℓ,α)` grid; (e) scope — §17's spectrum is `ℓ,α,m`, but the same logic extends to the other imposed
choices (`r_base` ratio 0.65, QRCP node count, `v_eff`, prior mean); decide the boundary explicitly.
Related: §B′ (curvature penalty = the `α` dial), §L (the 0.135-vs-0.84 corr flag = the `ℓ` dial).
