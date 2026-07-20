# Hyperparameter & logic audit — 2026-07-02 (Updated on 07-19)

Point-in-time audit of every tuning knob and the scientific claim-chain, done as part
of the 2026-07 refactor (commits `0a1b678`…). Sources: the code as of this commit,
`docs/DESIGN_DECISIONS.md` (DD), `docs/OUTSTANDING.md` (OD), the VOCALS results
notebook, and `../literature/` (CPV2012, BP2025/BP2026, NK1990, KV2012, KR2012,
VOCALS_REx, marine_sc/AMT-2025). Verdicts: **OK** = grounded (probe, study, or
literature, with the evidence named); **FLAG** = act or justify before relying on it
further; **INERT** = never binding in practice.

## 1. Verdict table

| Knob | Value | Grounding | Verdict |
|---|---|---|---|
| `NQUAD` | 48 | convergence probes; DOF ceiling = N=24 streams (DD §3/§15); E7 "keep" | OK |
| `NLEG_ALL` / `N_GL` | 1536 / 4096 | the TMS moment-count burn: 128 rang negative at x≈228; positive by ~768, converged ~1024, margin to r_e=25 (osse_config note); n_gl ≥ 2.35·NLeg | OK — the historically burnt knob, now over-margined and documented |
| `NFOURIER` | [24]×10 | re-tune on the fixed forward: rel<1 % at K=24, worst case 0.55 µm/r_e=20/τ=2 | OK; **keep uniform** (ragged breaks the E1 bands×modes batch — noted in osse_config) |
| truth/retrieval `tol` | 1e-4 | probe §A3 (closed 2026-06-29): same χ²ᵣ floor as 1e-5, only setting stable at both τ ends; E7 "keep" | OK; thick-tail (τ≳36) profile non-uniqueness caveat stands (DD note) |
| float32 default `tol` | 1e-3 (+`_RTOL_FLOOR_F32`) | DD §4; production explicitly float64 (DD §15) | OK (non-production) |
| `max_steps` | 4096 | headroom over ~35-step solves; the float32 10-band blowup is caught by it | OK |
| `MU0` | 0.9 fixed | OSSE choice | **FLAG (scope; §2.1)** — all §14/§15 results are conditional on one solar geometry; operational work needs μ0 binning (μ0 is a static/compile arg — STRATEGY §4) |
| views | 24 = NQuad//2, μ∈[0.25,0.95], φ=π, golden-ratio irregular | ≥NQuad//2 rule (DD §11b, verified A_top 0.25→0.39); irregularity kills the 0.55 µm aliasing notch (DD §14); μ=0.25 plane-parallel edge documented | OK |
| `ALBEDO` | 0.06 Lambertian, all 10 bands | DD §9 (BDRF convention fixed there) | **FLAG (low)** — sea albedo is spectrally varying (VIS≈0.02–0.06 → SWIR≈0.02); constant 0.06 is crude, though secondary under bright cloud. Revisit if dark-scene bands matter |
| `RE_BOUNDS` | (2, 22) µm | VOCALS truth max 18.1 + margin; 25 also safe at NLEG_ALL=1536 | OK (sufficiency choice, documented) |
| `RE_GRID_N`/`N_RADII` | 181 / 4096 | table-resolution choices; ripple-free per optics_table docstring | OK |
| `V_EFF` | 0.10 / 0.046 (canon) | 0.1 is the Hansen–Travis typical marine-Sc width and 0.046 was inferred from VOCALS-REx data; v_e is unretrievable from scalar intensity (DOFS~1 plateau — polarized cloudbow needed, shelved) | OK with caveat: sensitivity unquantified at 2 % noise |
| noise `k_cal` | 2 % + 1e-3 floor | PACE MRD §3.7 radiometric accuracy 1–3 %, calibration-dominated for bright clouds (DD §12) | OK |
| shot term | removed from code 2026-07-10 (was OFF/never populated) | OD-K open (needs OCI SNR tables; re-add is one quadrature line) | **FLAG (open)** — inherited open item; matters most for the dark 3.7/4.05 µm bands |
| mode-trim `frac` | 1/3 of min σ_ε | "≪ noise" rule; determines `NFourier` in `pydisort_riccati_jax` | OK |
| `filter_threshold` | 0.5 | Rodgers SNR=1 data/prior crossover; 0.25→0.5 re-sweep at 2 % noise (DD §10f) | OK; OD §G still says 0.25 — doc drift to fix in the doc revision |
| `margin` | +1 node | heuristic ("one prior-filled direction") | **FLAG (low)** — untested lever; harmless with the tight base prior, but no sweep exists at 2 % noise |
| `k_max` | 8 | cap on the filter count | INERT (observed k=4–6) |
| `S_REF_MODES`/`S_COARSE` | 4 / 5 uniform nodes | setup grids; S_COARSE is now also the τ_bot pre-retrieval grid (E3) | OK pending the HPC golden gate (§2.2) |
| GN: `n_iter`/`lm`/`xtol`/`cost_rtol` | 12 / 1e-2 / 2e-3 / 0.01 | observed 5–10 accepts; cost_rtol tuned (DD §10h); E7 "keep" | OK |
| `chi2_floor` | OFF | Sε magnitude not reliably profiled (DD §10h) | OK (deliberate) |
| `max_n_outer` | 2 (default) | frozen grid for clean A-vs-B; re-mesh escalation gated | OK |
| pre-retrieval: `re_sigma_tight`/`n_iter`/`xtol` | 0.1 / **4** / **2e-2** | 0.1 pins r_e (physical basis: ω=1 VIS rows carry τ_bot); 4/2e-2 are the **new E2a diet** (was 8/5e-3) | OK |
| priors: `r_base_ratio` | 0.65·r_top, clipped < r_top | VOCALS median 0.60, King/Vukićević AMT-2025 ≈0.70 | OK |
| priors: `sigma_top`/`sigma_base` | ≈2.3–2.5 / ≈1.4–1.5 µm | VOCALS MADs; "tight where blind, loose where strong" (DD §11, population-confirmed) | OK |
| priors: `sigma_tau_bot` | ~100 % relative | τ_bot fully data-determined (A≈1) | OK |
| priors: `corr_length` | 0.5 (normalized depth) → top–base prior **corr 0.135** (Update: 5.74, i.e. **corr 0.84** has been tried) | default τ_bot/2 heritage; **in-situ VOCALS corr(ln r_top, ln r_base)=0.84** (BP2026 recipe), so we under-couple by design | **FLAG (§2.2)** — deliberate: weak coupling exposes the depth information-collapse rather than hiding it; KV2012/KR2012 use 0 (diagonal), BP2026 uses the empirical ~0.84; tune for ops (OD §L) |
| B-draw RNG *(results retired)* | `default_rng(2000+index)` | reproducibility across resumes (audited clean) | OK |
| `TAU_BOT_OK` | (0.3, 100) | degenerate-profile guard (idx-0 τ≈1585) | OK |
| LWP | (2/3)·∫r_e dτ | assumes Q_ext≈2; z-integral cross-check carried in the sidecar | OK (documented approximation) |

## 2. Flags in detail

### 2.1 Single solar geometry (scope, not a bug)
All published IC and FR numbers are at μ0=0.9. The conclusions notebook presents are
geometry-conditional; nothing in the code prevents other μ0 (rebuild `setup`), but
no result yet quantifies how DOFS/band-value rankings move with μ0. Operational
per-scene work needs the μ0-binning strategy (STRATEGY §4) — deferred, documented.

### 2.2 The prior top–base correlation is under-specified vs the in-situ covariance (flag, deliberate for now)
`corr_length=0.5` sets the **only** knob controlling the prior correlation between
cloud-top r_e and the (shielded) cloud-base r_e. Through the exponential kernel
`S_a[i,j]=σ_iσ_j·exp(−|s_i−s_j|/ℓ)`, top (s=0) and base (s=1) get
`corr = exp(−1/0.5) = exp(−2) ≈ **0.135**` — nearly independent. **The VOCALS-REx
in-situ data says otherwise:** defining `r_top`/`r_base` as the mean over the top/bottom
25 % of each normalized-depth profile (the BP2026 recipe), the empirical
`corr(ln r_top, ln r_base)` over the 125 profiles is **+0.84** (Pearson-log; bootstrap
95 % CI [0.79, 0.88]; Spearman 0.85, so not a log/outlier artifact; monotone-stable
0.73→0.86 across a 10–30 % cutoff). We are imposing ~0.135 where the data supports ~0.84.

**Standard practice differs by school** (all four refs in `cloud_profile_retrieval/`):
- **KV2012** (King & Vaughan, eq 1 = our re⁵ law, state `[r_t,r_b,τ_c]`) and **KR2012**
  (Kokhanovsky & Rozanov, linear `[a_t,a_b]`+τ, `Q=S_a⁻¹`) both use a **diagonal `S_a`**
  — top and base *uncorrelated* (corr=0). Our 0.135 is already above their bar.
- **BP2025** (Buggee & Pilewskie, AMT 2025) couples top→base through the **prior mean**:
  `r_base,a = 0.70·r_top` (median VOCALS ratio), covariance not made off-diagonal.
- **BP2026** (the log-space source we already follow) operationalizes it in the
  **covariance**: eq (3) `S'_a = Cov(ln[r_top,r_bot,τ_c,IWV])` — the *full empirical*
  VOCALS covariance, off-diagonals included. This is the ~0.84 above.

## 3. Logic soundness — the claim chain (checked, sound)

1. **Forward validity.** Riccati solver ≡ pydisort references across 22 test files +
   the notebook's O(h²) multilayer convergence; delta-M f=g_NLeg convention matches
   pydisort exactly (apples-to-apples, `reference.py`); TMS corrects the upwelling
   field only, IMS omitted — consistent with LIDORT practice (DD §6).
2. **Leak-free OSSE.** Verified in code: LOO climatology excludes the truth's flight
   (`vocals_io.vocals_climatology(exclude_flight=...)`); priors/first guesses never
   see truth; `osse_observation` truth-encoding is world-definition, not leakage;
   radiance cache is signature-gated and τ_bot cross-checked against the in-situ truth.
3. **Noiseless observation + assumed Se.** DD §10b/§12 decision: y is noise-free,
   Se enters as weighting/posterior only. Standard OSSE practice; consequence
   (DOFS/SIC reflect *assumed* noise) is stated where it matters.
4. **Retrieval design.** Log-state + LM monotone descent + cost-stagnation stop =
   BP2026 §2.4/lines 205-213; normalized-depth nodes make joint τ_bot retrieval
   well-posed (no node-crossing); QRCP node selection whitened by noise and prior
   (Rodgers filter factors).
5. **Signature discipline.** `signature()` fingerprints what y *means*; tol is an
   accuracy tag asserted separately — verified both directions in workers. The one
   deliberate hole — K-trim and solver-precision are unsigned — is now documented
   (E1 note) and safe because the truth tier never mode-trims.
