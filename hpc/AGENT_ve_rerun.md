# Delegated task — `ve_rerun`: the corrected-v_e OSSE campaign (config A only)

*(Hand-off spec on branch `ve_rerun`. **Merge/rebase onto latest `origin/main` before launch** —
main carries the batch-3 lesson banners in `AGENT_all125_{rad,fr}.md` and
`STRATEGY_hpc_retrieval_runs.md`; this branch adds only the two campaign enablers + this spec.
Results move by zip, never git. 2026-07-07.)*

## Why (notebook §5/§5c; OUTSTANDING §L)

The published campaign's optics assume a gamma DSD width **v_e = 0.10** — the standard
operational choice, but a poor one for VOCALS marine Sc: the per-profile zero-LWP-bias constant
is **0.037 (median) / 0.046 (min-RMS)**, and the measured column is *narrower at cloud top*
(median ~0.03) where the reflected radiance forms. Two consequences, handled two ways:

1. **LWP bookkeeping** (the big one): already handled **post-hoc** —
   `scripts/retrieval_analysis.py` reports LWP under the width corrections C(0.037), C(0.046)
   and the per-profile oracle C\*; to first order this *is* what a re-run would report, because
   the OSSE loop is v_e-self-consistent (truth radiances and retrieval forward share one table).
   **A full re-run is NOT needed for these numbers.**
2. **Optics realism** (what this campaign tests): the residual, second-order effect of the
   table itself — different ω/phase function ⇒ slightly different radiances, Jacobians and
   retrieval trajectories. The §15 penetration analysis (notebook Fig 0b) shows the
   near-conservative bands' *angular* information is anchored on v_e-sensitive single-scattering
   features (glory ×2.6 at 1.038 µm; cloudbow) — so running the OSSE at a defensible v_e both
   (a) validates the post-hoc first-order claim empirically and (b) hardens the multi-angle
   findings against the "your DSD width is wrong" review criticism.

## The two enablers (this branch; defaults preserve production exactly)

- **`OSSE_VEFF`** (env; `src/pydisort_riccati_jax/osse_config.py`): overrides `V_EFF`.
  `signature()` includes v_eff, so an override **re-keys the optics table, the radiance-cache
  gate and the L2 setup caches automatically** — a forgotten export is *refused*, not silently
  mixed. Verified: default ⇒ sig `d71a8559bbe457e8` (byte-identical to the published campaign);
  `OSSE_VEFF=0.046` ⇒ sig `25160fe82d1654f2`.
- **`FR_CONFIGS`** (env; `scripts/retrieval_worker.py`): subset of `"AB"` to run
  (default `"AB"` unchanged). This campaign runs **`FR_CONFIGS=A`** — config B (prior-draw
  robustness) was settled by batch-3 and is orthogonal to the v_e question.

**Everything else is FROZEN for comparability with the published campaign** — same code,
NQuad=48, bands/views, `COST_RTOL=0.01`, float64, `SOLVER_TOL=1e-4`, same optimizer (the
OUTSTANDING §L "optimizer vNext" items are deliberately NOT on this branch).

## v_e value — user's call at launch

`OSSE_VEFF=0.046` (min-RMS constant) is the recommended primary; 0.037 (zero-median) is the
alternative. **Per-profile v_e (option iii) is explicitly OUT of this campaign**: the pipeline
assumes one observing system per campaign (one signature), and the per-profile *oracle* question
is already answered post-hoc by the C\* column — no run needed.

## Procedure (order matters; the signature forces it anyway)

Set once in every job: `export OSSE_VEFF=0.046 FR_CONFIGS=A` (plus the usual env from the
templates).

1. **Optics table** (~4 min, once): `AGENT_all125_rad.md` Step 1 with
   `OPTICS_CACHE=$ROOT/../data/optics_table_10band_nleg1536_re20_ve046.npz` (a NEW path; the
   table's own signature embeds v_eff, so a wrong file is refused).
2. **`rad` batch** (the truth tier MUST be regenerated — new signature): per
   `AGENT_all125_rad.md`, writing `RADIANCE_CACHE=$ROOT/../data/osse_radiances_ve046.npz`.
   Expect sig `25160fe82d1654f2`, tol tag 1e-4, 125 valid + idx-0 skip.
3. **FR pilot (GO/NO-GO gate; ~12 profiles, A100/V100S):** run
   `FR_CONFIGS=A` on indices spanning the τ range + the known classes, e.g.
   `{20, 95 (thin) · 5, 47, 49, 55, 75 (mid) · 11, 28 (remesh-history) · 110, 119 (deep,
   low-confidence) · 13 (RF13 re_max-edge)}`, outputs to `runs/_fr_ve046_parts/`.
   Report to the primary: per-profile ΔRMSE and χ²_red vs the published config-A values, and
   the C(0.046)-corrected LWP vs the post-hoc-predicted column (they should agree to well
   within the profile scatter — that agreement IS the first-order-equivalence validation).
   **The primary decides** whether the remaining ~113 profiles run.
4. **Full array (on GO):** `AGENT_all125_fr.md` Step 2 verbatim, with the three env overrides
   and the new cache paths; L1+L2 on; 11:55 walls; unique log paths (`%A_%a`); hourly resubmit
   driver; post-run stopping-criterion audit (fr-spec banner item 4).
5. **Deliverable:** `fr_ve046_bundle.zip` = `_fr_ve046_parts/` (+ any supersession dirs +
   logs) **plus** the step-2 radiance bundle, to the workspace root for manual download.
   Manifest-style supersession records if any corrective re-runs were needed.

## Cost expectation (STRATEGY §1/§3)

Config-A-only roughly halves the GN phase; setup is unchanged (L2 makes it once-per-profile).
Budget ≈ 55–70 % of batch-3's per-profile wall on the same card class; the pilot measures it.

## Analysis (primary, jovyan — no cluster analysis)

`scripts/retrieval_analysis.py` on the new sidecars; notebook §16's loader accepts an alternate
parts dir. The headline comparisons: RMSE/ΔRMSE distribution shift vs the published A (expect
≈ nil), C-corrected LWP vs post-hoc prediction (expect agreement), and any change in the
per-band/angle structure of K on the pilot profiles (the §15 Fig 0b features are the place a
v_e change is *expected* to show).
