# Outstanding Problems & Decisions

Open items, kept deliberately prominent. Settled rationale is in
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md). **Resolved / out-of-scope items are collapsed to a
one-line pointer** — the `## letter` headers are retained because both docs cross-reference them by
letter; the full rationale lives in the linked DESIGN section. The genuinely-open items are **G** and **K**.

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

Settled: keep the `(n_re, NLeg)` Mie–Legendre **r_e-table** (`miejax_lite.table_lookup`) as the
production optics path (profile-independent, no τ-placement problem, consistent table-slope Jacobian);
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
shape/EOF basis** (the more honest framing at DOFS≈2, and the resolution of the re-mesh instability
above) — is logged in DESIGN §3a as the smooth-basis route, "left open."

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

Implemented in `src/retrieval_oe.py`: cost `J(θ)`, Rodgers GN/LM (`gauss_newton_oe`), the
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

*(G-core — the angular/vertical-DOF, profile-independence, and multi-band-saturation questions — is the focus of a future session. The bulleted claims above and the **Starting point** below are **background**, not current actions; in particular the rank analysis must be re-run post-delta-M before any of it is relied on.)*

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

### Auto-select the node count `k_active`  [IMPLEMENTED — SO1 → DESIGN §10f]

Implemented as `retrieval_oe.auto_k_active`: the noise-aware whitened-QRCP **filter**
`f_i = r_i²/(1+r_i²)` (with `Σf_i ≈ DOFS` as a built-in cross-check), wired into
`select_retrieval_grid(k_active=None)` at `filter_threshold=0.25`. DOFS left the *selection* path (now
an info-content diagnostic only). OSSE verdict and threshold tuning: DESIGN §10f.

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
`ve_retrieval:docs/ve_retrieval/ASSESSMENT.md`. Its instrument-noise counterpart (HARP2 / DoLP) is
parked in §K.

---

## J. BDRF specified as [ρ/π] — π-too-dark mislabeling  [RESOLVED 2026-06-17 → DESIGN §9]

Fixed: removed `/π` from every `BDRF_Fourier_modes` call site and regenerated the reference `.npz`;
all affected float32 tests pass at the physically-correct albedos. The convention (`[ρ_s]`, **not**
`[ρ_s/π]`) and the single-bounce verification are documented in DESIGN §9.

---

## K. Measurement-noise model — shot term (Option A) and HARP2/polarized noise deferred  [DECISION / DEFERRED]

The infrastructure is **built and settled** (`src/noise_model.py`; the three-term σ(ρ); the
OCI-SWIR calibration-relative default; `osse_observation(noise=)` + `make_Se`; default noiseless —
see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) §12). These pieces are **open**:

- **Shot term (Option A) — wired but OFF, pending OCI SNR-at-L_typ.** The shot coefficients
  (`snr_ref`, `rho_ref`) exist but default to "off" because OCI's SWIR SNR-at-L_typ table could not be
  cleanly sourced: the PACE MRD (`PACE-SYS-REQ-0019L`) §3.7 tables are embedded **images** (no
  extractable text), the SNR requirement lives in an external `.xlsx` (`oci_functional_requirements_table2`),
  and converting a radiance-domain SNR to our reflectance units further needs per-band solar
  irradiance F₀ + a reference geometry. **To resolve:** obtain the OCI SWIR SNR + L_typ per band,
  convert L_typ→ρ_ref, and pass `snr_ref`/`rho_ref` to `oci_swir` (no code change). Low urgency:
  clouds are bright ⇒ calibration-dominated ⇒ the shot term is a small correction in our regime
  (`check_noise_model.test_bright_cloud_shot_subdominant`).

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
  §12 thick / §13 sub-adiabatic) now build `Se = roe.make_Se(fwd, y, nm.oci_swir())` — the PACE
  OCI-SWIR model (calibration-relative ~2 %) — replacing the hand-picked `0.03·max(|y|,0.02)` floor;
  `noise_model` is imported and the §8 markdown + §11b document the change. The OSSE stays **noiseless**
  (Se is the assumed weighting/UQ covariance only). `select_num_modes`'s own fixed `0.005²·I`
  mode-selection floor is a *separate* quantity and intentionally unchanged. (User re-runs the notebook.)
