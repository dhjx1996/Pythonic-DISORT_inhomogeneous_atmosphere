# Delegated task — `ve046_adia`: the 2-point (adiabatic-constrained) ablation

*(Hand-off spec on branch `ve_rerun`. Results move by zip, never git. 2026-07-16.)*

## Why (user-requested ablation)

The canonical campaign retrieves the joint state `x = [r_e(s_nodes), r_base, tau_bot]` on a
QRCP-selected, noise-aware-filtered grid (`select_retrieval_grid` / `auto_k_active`), which picks
**k = 4-6** free interior r_e nodes per profile. Those interior nodes are the retrieval's only
means of representing a **non-adiabatic** profile shape — departure from the r_e^5-linear law
that is `RetrievalForward._re_of_tau`'s function class.

This campaign asks: **do those extra degrees of freedom actually contribute?** It runs the same
worker, priors, observing system, optimizer and truth against a retrieval whose r_e profile is
collapsed to its **2 adiabatic knots**, and compares against the canonical free-node result
profile-by-profile.

## The baseline this is ablating against — READ THIS FIRST

**The canonical run is `runs/_ve046_tik_fr_parts`** (config A, 125 profiles), driven by
`hpc/sbatch/_ve046_tik_gpu_wide_fq.sbatch` and its CPU/GPU siblings. It is **not** batch-3
(`runs/_fr_parts`, v_eff=0.10) and **not** the earlier ve046 pass
(`runs/_ve_rerun_fr_parts`) — both are superseded. Canonical config:

| knob | value |
|---|---|
| optics | `OSSE_VEFF=0.046`, `OSSE_RE_MAX=22`, table `optics_table_10band_nleg1536_re22_fq.npz` |
| quadrature | `OSSE_QUADRATURE=fixed OSSE_RE_GRID_N=181 OSSE_N_RADII=4096` (fq) — **but see the mix caveat** |
| truth | `RADIANCE_CACHE=osse_radiances_ve046_fq.npz`, sig `4069d4bba73f0bab`, tol 1e-4 |
| configs | `FR_CONFIGS=A` (config B settled by batch-3, orthogonal here) |
| solver | float64, `SOLVER_TOL=1e-4`, `COST_RTOL=0.01`, NQuad=48 |
| regularisation / mesh | `CURVATURE_LAMBDA=1.0`, `MAX_N_OUTER=2`, `REMESH_CHI2_THR=2` |
| code | `ve_rerun` HEAD — **includes `c2e7601` "Improved GN"**, merged 07-11 (`84114e7`) |

**Optimizer provenance (verified 2026-07-16, correcting an earlier misreading):** every one of
the 125 canonical `_A.npz` files has an mtime in 07-11 22:51 .. 07-14 18:46, i.e. **after** the
07-11 11:57 merge that brought Improved GN into `ve_rerun`. The canonical set therefore already
runs the current optimizer; a new run on `ve_rerun` HEAD is the **same optimizer generation**.
No freeze, no re-baseline, nothing untested on that axis.

**⚠️ The canonical set is MIXED-quadrature** (from the deliberate 07-13 mid-run fq flip, made
after the decisive gate showed 6/6 overlap `r >= 0.9955`):

- **idx 99-124 (24 profiles):** fq / `fixed`, osig `4069d4bba73f0bab` (sentinel radiances)
- **the other 101:** legacy `moving` / 32-pt, osig `33643d346cd40d3e`

This campaign runs **all-fq** — **user decision 2026-07-16, and it is settled, not a caveat to
re-litigate**: the optics-table change was already measured immaterial (the 07-12 decisive gate:
6/6 overlap `r >= 0.9955`, which is *why* the canonical was flipped mid-run), and all-fq is the
current operational paradigm. No matched legacy-quadrature control arm is warranted. The 24
fq-vs-fq canonical profiles (idx 99-124) remain available as a free internal cross-check if an
analysis ever wants one, but the population comparison against all 125 canonical profiles is the
intended one.

*(Cosmetic, do not be alarmed: 21 of the canonical combined `<i>.json` files say
`skipped: name 'mon_A' is not defined` — the known NameError fixed in `edfb399`. Their `_A.npz`
sidecars are valid and complete; only the combined json mislabels them.)*

## The one enabler (`scripts/retrieval_worker.py`)

**`FORCE_K_ACTIVE`** (env, int; unset = production default = unchanged behaviour): pins
`select_retrieval_grid`'s `k_active` instead of letting `auto_k_active`'s noise-aware filter
choose it. `select_retrieval_grid` **always** retains cloud-top (`s≈0`) first, then fills by QRCP
rank — so `FORCE_K_ACTIVE=1` yields **exactly one** free interior node at cloud-top. With the
`r_base` anchor always appended at `s=1` (`RetrievalForward._knots_vals`), the profile has
exactly 2 knots and interpolates re5-linear between them: **the adiabatic curve**, identical in
form to `make_adiabatic_prior`'s mean and to the 2-parameter family `best_fit_adiabatic` fits.
`tau_bot` is still retrieved — "2-point" describes the *profile shape* DOF.

This makes the retrieval itself adiabatic-constrained, which is the causal version of the
comparison `best_fit_adiabatic` already makes post-hoc (that one is an *oracle* fit to the truth
with tau_bot handed to it; this one must infer everything from radiances).

Also threaded into the L2 setup-cache key (`|k{FORCE_K_ACTIVE}`) so a forced-k setup can never be
confused with a canonical one, and stamped into `<i>.json` as `force_k_active` for provenance.
Verified by synthetic-Jacobian unit check: `k_active=1` → exactly `s_sel=[0.0]`; `k_active=None`
and `k_active=3` (the existing fixed-count re-mesh path) unaffected.

**Two canonical knobs go naturally inert at k=1 — expected, not a bug:**

- `CURVATURE_LAMBDA=1.0`: `_second_difference_operator([0.0, 1.0])` → shape `(0,2)` → `P = None`.
  Two knots carry no curvature. (Verified.)
- `MAX_N_OUTER=2`: the re-mesh re-selects at `fixed_k=1`, which always returns cloud-top →
  `grid_changed=False` → `break`.

Both are set anyway so the config matches canonical nominally. Neither confounds the comparison:
they regularise/refine degrees of freedom the ablation removes.

## Procedure

1. **No optics/rad rebuild.** Both caches exist and are signature-correct:
   `optics_table_10band_nleg1536_re22_fq.npz` (v_eff=0.046, re=[2,22]/181, quad=fixed/4096) and
   `osse_radiances_ve046_fq.npz` (sig `4069d4bba73f0bab`, tol 1e-4).
   *(Filename trap: `optics_table_10band_nleg1536_re20_fq.npz` currently holds v_eff=0.046/re22
   content despite its name — the 2026-07-12 mismatch-pilot overwrite. Never trust a fq table by
   filename; check its `signature` field. The re22 table above is correct.)*
2. **Gate** (`hpc/sbatch/_ve046_tik_adia_gate.sbatch`): idx-47, A100, writes
   `runs/_ve046_tik_adia_parts/47`. Retained as a template for future forced-k work; the
   2026-07-16 run **skipped it by user decision** (go straight to the wide array).
3. **Wide array** (`hpc/sbatch/_ve046_tik_adia_wide.sbatch`): `--array=1-125%60`. L1+L2 on,
   11:59 walls, `nice=2000`, unique `%A_%a` logs, idempotent resubmit, straggler routing per
   `STRATEGY §3`. **Precondition:** the fq truth cache must cover 1..125 (see below).
4. **Post-run audit** (`AGENT_all125_fr.md` banner item 4): population-wide final-iteration
   `rel`/`chi2_red` sweep; continuation-test suspicious stops.
5. **Deliverable:** `ve046_adia_bundle.zip` = `runs/_ve046_tik_adia_parts/` +
   `runs/_ve046_adia_logs/`, to the workspace root for manual download (never git).

## The idx-125 gap — diagnosed and CLOSED (2026-07-16)

`osse_radiances_ve046_fq.npz` originally covered **idx 1..124 only**. Diagnosis: the fq cache was
built by **four incremental gap-fill batches** — `fqR46` (8 idx), `fqR46ic` (85), `fqR46idx66` (1),
`fqRfull46` (91..124, 31) — whose **union is exactly 1..124**. idx-125 was **never submitted by
any of them**. Nothing special about the profile: RF14, τ_bot=2.462, 21 pts, passes the
`TAU_BOT_OK` guard (only idx-0, τ≈1585, is degenerate), present in the legacy
`osse_radiances_ve046.npz`, and the canonical FR retrieved it fine (against legacy-quad truth).
It is the same off-by-one class as the idx0/idx1 trap in `profile_index_convention` — the arrays
topped out at 124 instead of 125.

**Closed:** job `9058577` (`--array=125`, `--constraint=a100|v100s` per user directive, since this
index alone must compute its radiance from scratch) → `runs/_rad_fq_ve046_parts/osse_125.npz`,
then `generate_osse_radiances.py consolidate runs/_rad_fq_ve046_parts
../data/osse_radiances_ve046_fq.npz` → cache now 1..125, sig `4069d4bba73f0bab` unchanged
(consolidate is signature-gated and additive; the sidecars for 1..124 are untouched).

⚠️ `consolidate` uses a plain non-atomic `np.savez` on a **shared** cache. Do it only while no job
is reading that file (check `squeue`), and **before** launching the array — never mid-run.

## Cost expectation

Setup (mode select + τ_bot pre-retrieval) is unpriced by L2 across campaigns — the `_cfg` key
differs by `|k1`, so it is paid fresh, comparable to the canonical's setup cost. The GN phase
should be **cheaper** than canonical: 1 free r_e node instead of 4-6 → narrower Jacobian, smaller
per-iteration solve. Config A only. Net: at or below canonical per-profile wall on the same card
class; the gate measures it.

## Analysis (primary, jovyan — no cluster analysis)

`scripts/retrieval_analysis.py` on the new sidecars vs `runs/_ve046_tik_fr_parts`. Headline: per
profile `w1_ours` and `chi2_red` under the 2-point constraint vs the canonical free-node result.
The extra DOF earn their keep if the free-node retrieval is materially better on genuinely
non-adiabatic truths and roughly at parity on near-adiabatic ones; population-wide parity says
they do not. `d_w1 = w1_adia - w1_ours` is directly interpretable here: the ablation arm is
constrained to the same family the `w1_adia` oracle fits, so `w1_ours -> w1_adia` measures how
much of the oracle's advantage was oracle knowledge (true τ_bot, fit to truth) rather than DOF.
**Do the fq-vs-fq check on idx 99-124 first** (see the mix caveat above).
