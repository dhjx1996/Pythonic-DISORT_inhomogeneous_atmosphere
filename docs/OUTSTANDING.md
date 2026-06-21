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
shape/EOF basis** — is **CLOSED (user, 2026-06-21): rejected.** An EOF-from-ensemble basis over-constrains
every retrieved profile to the training-distribution shapes (`span{φ_i}`), compromising the
**minimally-constrained** feature that distinguishes this method from the adiabatic / 2–3-param literature.
The free sensitivity-placed-node parameterisation is **retained**. If node-difference wiggles ever warrant
suppression, the lever to *research* is a **2nd-difference (curvature) Tikhonov penalty** — not needed now:
the demo swings (e.g. idealized §5) were an artifact of a deliberately **loose σ_base=10**, and a
climatologically-tight base prior (DESIGN §11; VOCALS r_base MAD≈1.4 µm) already removes them. (This
supersedes DESIGN §3a's "smooth-basis route, left open.")

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

### Per-mode ODE grids and the retrieval grid  [INVESTIGATED 2026-06-20 → keep m=0; DESIGN §3a]

**Verdict: rejected. The m=0 ODE grid stays the sole retrieval-grid pool.** The hypothesis — that
the discarded m≥1 grids carry complementary vertical information, so the "best" pool is the **union
of the non-negligible (Cauchy-K) modes' grids** — was tested directly
(`tests/supplementary/per_mode_grid_investigation.py`, a faithful monkeypatch that retains every
mode's forward ODE grid) and does **not** hold: the union is neutral-to-harmful on every VOCALS
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
