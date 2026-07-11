# Hyperparameter & logic audit — 2026-07-02

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
| `MU0` | 0.9 fixed | OSSE choice | **FLAG (scope)** — all §14/§15 results are conditional on one solar geometry; operational work needs μ0 binning (μ0 is a static/compile arg — STRATEGY §4) |
| views | 24 = NQuad//2, μ∈[0.25,0.95], φ=π, golden-ratio irregular | ≥NQuad//2 rule (DD §11b, verified A_top 0.25→0.39); irregularity kills the 0.55 µm aliasing notch (DD §14); μ=0.25 plane-parallel edge documented | OK |
| `ALBEDO` | 0.06 Lambertian, all 10 bands | DD §9 (BDRF convention fixed there) | **FLAG (low)** — sea albedo is spectrally varying (VIS≈0.02–0.06 → SWIR≈0.02); constant 0.06 is crude, though secondary under bright cloud. Revisit if dark-scene bands matter |
| `RE_BOUNDS` | (2, 20) µm | VOCALS truth max 18.1 + margin; 25 also safe at NLEG_ALL=1536 | OK (sufficiency choice, documented) |
| `RE_GRID_N`/`N_RADII` | 32 / 600 | table-resolution choices; ripple-free per optics_table docstring | OK |
| `V_EFF` | 0.10 fixed | Hansen–Travis typical marine-Sc width; v_e is unretrievable from scalar intensity (DOFS~1 plateau — polarized cloudbow needed, shelved) | OK with caveat: results are conditional on v_e=0.10; sensitivity unquantified at 2 % noise |
| noise `k_cal` | 2 % + 1e-3 floor | PACE MRD §3.7 radiometric accuracy 1–3 %, calibration-dominated for bright clouds (DD §12) | OK |
| shot term | removed from code 2026-07-10 (was OFF/never populated) | OD-K open (needs OCI SNR tables; re-add is one quadrature line) | **FLAG (open)** — inherited open item; matters most for the dark 3.7/4.05 µm bands |
| mode-trim `frac` | 1/3 of min σ_ε | "≪ noise" rule | OK |
| IC workers' mode-trim `Se` | ~~flat (0.005)²·I~~ → measured-radiance OCI Se | was inconsistent with the oci_swir Se the diagnostics assume | **FIXED 2026-07-02** — see §2.1; re-run decision pending |
| `filter_threshold` | 0.5 | Rodgers SNR=1 data/prior crossover; 0.25→0.5 re-sweep at 2 % noise (DD §10f) | OK; OD §G still says 0.25 — doc drift to fix in the doc revision |
| `margin` | +1 node | heuristic ("one prior-filled direction") | **FLAG (low)** — untested lever; harmless with the tight base prior, but no sweep exists at 2 % noise |
| `k_max` | 8 | cap on the filter count | INERT (observed k=4–6) |
| `S_REF_MODES`/`S_COARSE` | 4 / 5 uniform nodes | setup grids; S_COARSE is now also the τ_bot pre-retrieval grid (E3) | OK pending the HPC golden gate (§2.2) |
| GN: `n_iter`/`lm`/`xtol`/`cost_rtol` | 12 / 1e-2 / 2e-3 / 0.01 | observed 5–10 accepts; cost_rtol tuned (DD §10h); E7 "keep" | OK |
| `chi2_floor` | OFF | Sε magnitude not reliably profiled (DD §10h) | OK (deliberate) |
| `max_n_outer` | 1 (FR) / 2 (default) | frozen grid for clean A-vs-B; re-mesh escalation gated | OK |
| pre-retrieval: `re_sigma_tight`/`n_iter`/`xtol` | 0.1 / **4** / **2e-2** | 0.1 pins r_e (physical basis: ω=1 VIS rows carry τ_bot); 4/2e-2 are the **new E2a diet** (was 8/5e-3) | **FLAG (gate)** — validated on the toy suite only; the production check is the 3-profile golden gate (§2.2) |
| priors: `r_base_ratio` | 0.65·r_top, clipped < r_top | VOCALS median 0.60, King/Vukićević AMT-2025 ≈0.70 | OK |
| priors: `sigma_top`/`sigma_base` | ≈2.3–2.5 / ≈1.4–1.5 µm | VOCALS MADs; "tight where blind, loose where strong" (DD §11, population-confirmed) | OK |
| priors: `sigma_tau_bot` | ~100 % relative | τ_bot fully data-determined (A≈1) | OK |
| priors: `corr_length` | 0.5 (normalized depth) | default τ_bot/2 heritage | **FLAG (low)** — the smoothness scale was never swept at 2 % noise; it shapes the Bayesian-Tikhonov coupling that moves the shielded base |
| B-draw RNG | `default_rng(2000+index)` | reproducibility across resumes (audited clean) | OK |
| `TAU_BOT_OK` | (0.3, 100) | degenerate-profile guard (idx-0 τ≈1585) | OK |
| LWP | (2/3)·∫r_e dτ | assumes Q_ext≈2; z-integral cross-check carried in the sidecar | OK (documented approximation) |

## 2. Flags in detail

### 2.1 IC mode-selection noise is not the retrieval noise (medium)
`scripts/ic_worker_profile.py:110` and `ic_worker_mechanism.py:71` select the mode
count with `Se = (0.005)²·I`, while everything downstream (Se weighting, DOFS/SIC)
uses `noise_model.oci_swir()` (2 %·ρ + 1e-3 floor). For dark SWIR scenes the oci σ
can sit near the 1e-3 floor — 5× *below* the flat 0.005 — so the trim threshold
(frac·min σ) is looser than the assumed measurement noise and can drop modes that are
not strictly sub-noise for the darkest observations. Production FR is consistent
(`retrieval_worker` passes the oci Se). **Status: FIXED in code (2026-07-02)** — both
IC workers now load the radiance record before mode selection and select against the
measured-radiance OCI Se (each worker's own view set). The published definitive
bundle still reflects the old flat value; a re-run therefore changes K selection
(same or MORE modes kept — the accuracy-safe direction) and shifts DOFS/SIC beyond
the E1 sub-noise note. The re-run decision sits with the user + HPC agent (see
CHANGELOG).

### 2.2 The E2a/E3 setup diet needs its production gate (gate-blocking)
`retrieve_tau_bot(n_iter=4, xtol=2e-2)` + pre-retrieval-on-`S_COARSE` change where
the per-profile grid anchor comes from. The toy suite (tests/23i) confirms anchor
recovery, but the anchor feeds grid selection, so the production check is grid
stability + retrieved-state agreement on real profiles across the τ range:
`tests/hpc/test_golden_profiles.py` (3-profile golden-diff) **must pass on the HPC
before the next production sweep** — as the fable assessment prescribed for E1–E6.

### 2.3 Single solar geometry (scope, not a bug)
All published IC and FR numbers are at μ0=0.9. The conclusions notebook presents are
geometry-conditional; nothing in the code prevents other μ0 (rebuild `setup`), but
no result yet quantifies how DOFS/band-value rankings move with μ0. Operational
per-scene work needs the μ0-binning strategy (STRATEGY §4) — deferred, documented.

### 2.4a Accepted latent gap: optics-table cache signature omits `n_gl` (low, by decision)
`optics_table._signature` keys the disk cache on (wavelengths, r_e bounds/grid, v_eff, NLeg)
but **not** `n_gl` — a table rebuilt at a different Gauss–Legendre projection order under the
same key would be wrongly reused. Judged consistent-enough on 2026-06-29 (`n_gl` is fixed at
4096 on the standard path); recorded here for awareness (ported from the retired STREAMLINE.md).

### 2.4 Un-swept second-order priors (low)
`corr_length=0.5` (normalized) and `margin=1` are the two remaining prior/selection
levers with no dedicated sweep at the 2 % noise model. Both are plausibly benign
(the base is prior-dominated by design; margin adds one prior-filled direction), but
neither has the evidence discipline the other knobs now have. Candidate for a cheap
2-case sweep when the HPC is next idle.

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
4. **IC methodology.** Spectral baseline = flux reflectance (plane albedo, m=0
   exact) — the CPV2012/King–Vaughan quantity; the angular-novelty analysis is the
   extension beyond CPV2012; DOFS/SIC per Rodgers (eq. 2.80 form verified in
   `posterior_diagnostics`); band-addition order data-greedy from the standard
   bispectral pair (NK1990) — reconstruction of the published figures verified
   bit-level (2026-07-02, `ic_analysis_definitive.py` reproduces both JSONs).
5. **Retrieval design.** Log-state + LM monotone descent + cost-stagnation stop =
   BP2026 §2.4/lines 205-213; normalized-depth nodes make joint τ_bot retrieval
   well-posed (no node-crossing); QRCP node selection whitened by noise and prior
   (Rodgers filter factors); oracle-adiabatic ΔRMSE is a like-for-like floor (same
   function class, generous τ_bot oracle).
6. **Signature discipline.** `signature()` fingerprints what y *means*; tol is an
   accuracy tag asserted separately — verified both directions in workers. The one
   deliberate hole — K-trim and solver-precision are unsigned — is now documented
   (E1 note) and safe because the truth tier never mode-trims.

## 4. Standing actions

1. HPC: run the golden gate (`tests/hpc/`, §2.2) before the next production sweep.
2. IC mode-trim Se: fixed in code (§2.1); decide the IC re-run (user + HPC agent).
3. Doc revision pass: fix OD §G filter_threshold drift; fold this audit's verdicts
   into DD where they harden decisions.
4. Idle-HPC candidates: corr_length/margin mini-sweeps (§2.4); μ0 sensitivity spot
   check (§2.3); OCI SNR tables → shot-noise term (OD-K).
