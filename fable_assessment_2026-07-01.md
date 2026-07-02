# Fable assessment — L1/L2/L3 caches + full FR code review (2026-07-01)

*(Scope: everything on the FR call path — `tests/supplementary/retrieval_worker.py` (incl. the
uncommitted L2 edits), `src/retrieval_oe.py` in full (forward class, mode/grid selectors, priors,
LM-GN inner loop + L1 checkpoint, τ_bot pre-retrieval, posterior), `tests/supplementary/osse_config.py`,
`tests/supplementary/runtime_setup.py`, `_fr_l2_test.py`, and the `_fr_*.sbatch` drivers. Written
mid-batch-3 with the L2 GPU gate in flight.)*

---

## 1. Cache verdicts

**L1 — GN-iteration checkpoint** (`retrieval_oe._save/_load_gn_checkpoint`, `_gn_inner`)
- *Correctness:* **Sound** — atomic tmp+`os.replace` writes, plain-numpy portability, exact resume
  semantics (post-ease `lm_cur` persisted, so the resumed LM trajectory reproduces the
  uninterrupted one); production-verified. One benign nuance: a kill in the window between the
  checkpoint write and the convergence tests (`cost_rtol`/`step_small` run *after* the save) can
  make a resumed run take *extra* accepted iterations — monotone descent means equal-or-better J,
  never a loss.
- *Importance/viability:* **HIGH, proven.** This is what makes 11:55-wall chunking and
  cheap preemption possible; keep permanently.

**L2 — setup cache** (`retrieval_worker.build_forward_and_obs`, `FR_SETUP_CACHE`)
- *Correctness:* **Audited clean, correctly gated** — atomic write, per-index file, config-gated
  load, jit invalidation mirrors `select_num_modes`, prior builders rebuilt exactly as the compute
  path; the same-platform bit-exact gate (`_fr_l2_test.py`) is the right test. Two hardenings
  before production wiring: **(i)** the `cfg` key (`prec|tol|NQ`) is too narrow — add
  `osse_config.signature()[1]` and the profile index to it (a future NFourier/band/view change
  would otherwise silently load a stale setup); **(ii)** note that cross-card reuse yields *a*
  valid deterministic setup, not *the* bitwise setup that card would have computed — the same
  class of fp-difference as today's cross-card L1 resume, and strictly better: it **freezes** the
  setup per profile, removing the current cross-wall setup-drift (a resumed config today recomputes
  its grid on a different card and continues a checkpoint against fp-shifted nodes).
- *Importance/viability:* **The highest-value gap in the system.** It eliminates the 1.9–11 h
  re-setup that dominates straggler walls (thin profiles on RTX8000 burn entire walls inside
  setup). Wire immediately on gate PASS; it is also a *rigor* improvement (frozen setup ⇒ cleaner
  A-vs-B and resume consistency). **OUTCOME (2026-07-01, post-write-up): gate 8683416 PASSED
  bit-exact (all five compared quantities dmax 0.0 on cache HIT) — wired into
  `_fr_gpu_realloc.sbatch` mid-run.**

**L3 — persistent JAX compile cache**
- *Correctness:* harmless as configured (default thresholds), but **measurably ineffective for
  FR** — the heavy forward/jacfwd executables never persist (only ~4 KB helper jits + the XLA
  autotune dir; 26,556 files / 107 MB of pure Lustre liability).
- *Importance/viability — the salvage question:* **Agree: delete.** (Data now; the feature code
  flagged for the main agent + user, per standing decision.) Salvaging would first require
  diagnosing *why* the executables don't land — most plausibly the per-band optics tables are
  closure-captured constants baked into the HLO (large, key-unstable) and/or the
  ForwardMode/custom-solve path — and even a perfect cache is bounded at ~10–15 % of a thick
  profile (one ~850 s compile vs ~18 × ~650 s executions). Not worth it for FR. Keep `_jax_cache`
  for IC (real `ptxas`-abort mitigation). The one future context where compile caching genuinely
  earns its keep is operational per-scene-μ₀ runs (μ₀ is a static arg ⇒ one compile per μ₀ bin,
  reused across many scenes — IC-like reuse); revisit only there. Note the refactor that would
  make FR executables cacheable (optics as traced args, §3-E5) is independently motivated — if it
  ever lands, re-measuring L3 costs nothing.

---

## 2. Correctness review — findings

**F1 — resume-skip `NameError` mislabels completed profiles as skipped. FOUND + FIXED mid-run.**
`retrieval_worker.main()`: when config A (or B) is resume-skipped (`_A.npz` exists), `mon_A`
was never bound, so the final `rec.update(..., A=mon_A, ...)` raised `NameError`, the broad
`except` caught it, and the combined `<i>.json` was written as `"skipped"` — a wrong monitoring
record, an incorrect coverage signal (resubmit loops re-paying full setup to crash identically),
and it had not yet fired only because no resume-skip had completed (19 indices were poised to hit
it, incl. the queued 2/3/5/6/7 and held 24). Fix (applied 2026-07-01, syntax- and
function-tested): **(a)** skip branches now require `.npz` *and* `.json` and reload the persisted
`mon` from the sidecar json; **(b)** `_persist` writes npz/json atomically (tmp + `os.replace`) —
they are resume sentinels and reload sources, so they may not be torn by a wall; **(c)** a
**finalize-only fast path**: if both configs are fully persisted, the combined `<i>.json` is
rebuilt from the sidecars in seconds (verified equal to a production record on grid/K_list/A/B;
adds `finalize_only=true`, omits the setup-only `tau_bot_pre`) — no setup re-pay, and
already-poisoned tasks self-heal on their next resubmit. Numerics untouched.

**F2 — L2 cache key too narrow** — see §1-L2(i); **applied** (signature + index now in `_cfg`).

**F7 — L2 cache write never landed (FOUND post-audit + FIXED; caught by the gate).**
`np.savez(<string path>)` appends `.npz` when the name lacks it — the atomic-write tmp name
(`…npz.tmp<pid>`) therefore never existed for `os.replace` → every L2 write failed with ENOENT,
and the first GPU gate run silently degraded to compute-vs-compute (a vacuous PASS). Fixed by
writing through an open file object (numpy does no extension munging on handles — same pattern as
`_save_gn_checkpoint` and the F1 `_persist` fix); the gate was cancelled and re-submitted on the
fixed code. Lesson recorded: an equivalence gate must FAIL LOUDLY when its premise (cache written,
then HIT) is not established — the re-armed watcher now also triggers on `cache write failed`, and
`_fr_l2_test.py` should assert the HIT actually occurred (check the cache file exists after
PASS 1) rather than rely on the log line.

**F3 — resume can overshoot the convergence boundary** — see §1-L1; benign (monotone), document.

**F4 — cross-wall setup recompute drift (pre-L2)** — a resumed task rebuilds its setup on whatever
card it lands on; ODE grids/QRCP are fp-card-dependent, so a config can resume its GN checkpoint
against minutely-shifted nodes (and A/B can end on minutely different grids). Self-consistent —
the checkpointed x is just a warm start against the (recorded-in-sidecar) grid actually used, and
a structural k-change would fail loudly on shape — so no rigor action needed; L2 removes it.

**F5 — gate scope** — `_fr_l2_test.py` proves same-platform bit-exactness only; cross-card L2
reuse is covered by the F4 argument, not by the gate. Fine for this run (state it in the docs).

**F6 — housekeeping** — `runtime_setup` slot-registry dirs (`.rad_core_slots/{job}_{node}`)
accumulate; purge at end-of-run (Lustre file-count hygiene).

*Audited clean (no action):* signature/tol gating of radiance + optics caches; truth-vs-cache
τ_bot cross-check; `_clamp_state` projection incl. log-space bounds; LM accept/reject monotonicity
+ backtrack cap; `cost_rtol`/`step_small`/no-descent stop logic; per-config checkpoint pathing and
ckpt deletion on persist; B-draw reproducibility across resumes (`default_rng(2000+index)`);
`max_n_outer=1` + `RemeshWarning` structural-misfit flagging; degenerate-profile guard;
`select_retrieval_views` layout; jacfwd-over-p choice (p≈7 ≪ m=240); affinity slot-claim
atomicity (O_EXCL + conservative reclaim).

---

## 3. Efficiency — full-refactor targets, ranked

**E1 — Restore the bands×modes GPU batch (the headline; ~2–5× on every Jacobian).**
`_forward_raw` only takes the 240-way `_forward_raw_vmap_bands` path when
`len(set(K_list)) == 1`. `select_num_modes` returns per-band K (e.g. `[23,24,…,19]`), so
**production FR silently degrades to the 10-band sequential Python loop** — each band a ~20-24-mode
vmap, i.e. exactly the latency-bound under-utilization the GPU measurements show. The trim itself
saves ~nothing on GPU (batch width is nearly free) — it was a CPU-scan optimization.
*Implementation:* on GPU (`mode_map=='vmap'`), set `fwd.K_list = [max(K_list)]*n_bands` after
`select_num_modes` (≈ +5–6 % modes vs the trimmed sum, for ~10× more SIMT width); extend the
batched path to `jacobian_on_grid` (the pool-Jacobian — it currently *always* uses the band loop,
so this also cuts the setup's grid-select phase) and to `mode_amplitudes` (10 sequential
full-NFourier solves today). *Validation:* uniform-K batched vs band-loop is already claimed
bit-identical (validated fwd+jacfwd per the docstring) — re-run that check; the K-pad adds only
modes each bounded < `(1/3)·min σ_ε` by construction (`select_num_modes` threshold), so golden-diff
2–3 profiles across the τ range + one A100/RTX8000 timing pair. **Interaction:** keep NFOURIER
ceilings *uniform* in future configs (per-band ceilings are a CPU-era saving that breaks
`_bands_share_setup` — venue-dependent trade, document in `osse_config`).

**E2 — τ_bot pre-retrieval diet (the user-flagged MAJOR INEFFICIENCY; ~30–50 % of setup).**
It is a full-width GN: up to 8 iterations × a k+2-tangent jacfwd on all 240 obs, for what is only
an informed prior anchor (τ_bot is free in the retrieval proper). In order of increasing
invasiveness: **(a)** loosen `n_iter` 8→3–4 and `xtol` 5e-3→2e-2 (halves the Jacobians; the
anchor's σ enters only the prior, tolerance for slack is high); **(b)** differentiate only the
free directions — jacfwd w.r.t. τ_bot (+r_base) = 1–2 tangents instead of k+2 (~4× per iteration;
one extra small compile, static argnums split); **(c)** band masking saves real compute only via a
3-band VIS forward (extra compile) — post-E1 the full-band eval is cheap enough that (c) is
probably moot. *Validation:* compare `tau_bot_pre`, the final grid, and the retrieved state on
~6 profiles spanning τ (the anchor feeds grid selection, so check grid stability, not just τ_pre).

**E3 — Drop the *initial* grid selection entirely (~800 s A100 / ~50 min RTX8000 per profile).**
The initial QRCP grid exists only to give the τ_bot pre-retrieval a grid — but the pre-retrieval
pins r_e tight, so grid quality is irrelevant to it: run it directly on the fixed `S_COARSE`
(5 nodes). Bonus: the docstring's compile-reuse rationale ("pre-retrieval at the initial grid
reuses the compiled Jacobian") is broken in practice — initial k=6 vs final k=5/4 recompiles
anyway (observed on idx-15…32, idx-95); `S_COARSE` gives p=7 = the *common* final shape, so reuse
*improves*. One `select_retrieval_grid` per profile instead of two. Validate as E2.

**E4 — L2 as a setup *farm* (future runs; composes with E2/E3).** Setups are deterministic and
platform-independent (the L2 design point): precompute all N setup caches on idle CPU `short`
slots (~278) before/alongside the GPU array — GPUs then run pure GN from the first minute.
Directly multiplies for noise-ensemble runs (M realizations per profile share one setup).

**E5 — Optics tables as traced args, not closure constants (medium effort).** The per-band
`omega/leg` arrays are captured in every jitted closure → baked into each HLO as constants:
bloats compiles, makes executables cache-unstable (the likely L3 failure), and re-traces on any
table change. Thread them as arguments (donated where possible). Payoff: faster/leaner compiles,
L3 re-measurable for free; prerequisite for any μ₀-binned operational cache.

**E6 — Fuse forward+Jacobian (`jax.linearize`).** After each accepted step the loop calls
`forward` (backtracks) then `jacobian` (which recomputes the primal internally) — one redundant
forward per iteration ≈ 5–10 % of the GN loop.

**E7 — Hyperparameters: keep.** f64 / tol=1e-4 / NQuad=48 / `cost_rtol=0.01` are probe-settled
(§A3, NFourier study, tune_cost_rtol) — no change proposed. `n_iter=12` cap is safe (observed 5–10
accepts). `lm` warm-start or A→B state reuse: rejected — B's whole purpose is the prior-centred
start, and lm adaptation converges in ~2 iterations anyway.

*Compound estimate:* E1 × (E2+E3) takes a thick profile from ~7 h to ~2–2.5 h on A100, and — the
operationally bigger effect — takes the thin-profile setup on slow cards from ≈ a full wall
(zero progress) to ~1.5–2 h, dissolving the straggler class; L2/E4 then makes every revisit ~free.

---

## 4. Mid-run vs post-run split

- **Done mid-run (numerics-neutral):** F1 fix (applied + tested); straggler→A100 routing;
  driver coverage semantics tightened (a `"skipped"` combined json for idx 1–125 counts as
  incomplete, and completeness = `_A.npz`+`_B.npz`+non-skipped `<i>.json`).
- **On L2 gate PASS (sanctioned):** wire `FR_SETUP_CACHE` into the resume sbatch with the F2 key
  hardening; resubmit the held checkpointed stragglers; commit L2.
- **Post-run only (numerics or comparability change — do NOT mix into this batch):** E1, E2, E3,
  E5, E6 — each with the validation recipe above, ideally landed together and re-gated on a
  3-profile golden-diff before the next production sweep. **ON HOLD: none of this starts without
  explicit user permission (user directive, 2026-07-01).**
