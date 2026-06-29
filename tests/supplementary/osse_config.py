"""Canonical observing system for the VOCALS OSSE — the SINGLE SOURCE OF TRUTH
shared by the radiance generator, the IC worker, and the retrieval worker.

Why this module exists: we precompute the synthetic measurements ``y = F(truth)``
once (the "synthetic L1B", generate_osse_radiances.py) and have every downstream
worker LOAD them instead of regenerating. That is only safe if every worker assumes
the *identical* forward model. :func:`signature` captures the full observing-system
fingerprint; :func:`load_radiance` asserts the cache was generated with a matching
fingerprint before handing back ``y`` — the real-pipeline analog of "does this L1B
match my forward-model assumptions".

NFourier (azimuthal mode count) is **per band**, set by the convergence study
(``nfourier_study.py`` / DESIGN: NFourier study), NOT by the noise-aware per-profile
trim (``select_num_modes``, retained but deprecated for the OSSE — we do not trust the
noise estimate for truncation). The short-lambda / large-r_e bands need many modes
(principal-plane glory/rainbow); the absorbing long-lambda bands need few. Set << noise,
conservative.
"""
from math import pi
import hashlib
import json
import os
import numpy as np

# Adaptive Kvaerno5 ODE tolerance for the TRUTH tier (radiances + IC). STANDARD = 1e-4
# (default here; every sbatch also re-exports SOLVER_TOL=1e-4). Basis — §A3 probe: tol=1e-4
# is indistinguishable from tol=1e-5 at τ≲20 (0.2 % rmse); at the thickest profiles
# (τ≈36–51) the two converge to retrievals ~0.4 µm rmse apart (both converged — tol=1e-4
# was actually closer on τ_bot), within the practical-significance bar, so 1e-4 is adopted
# truth-tier-wide. (Earlier 3e-5 was a margin the probe made unnecessary; legacy 1e-3, the
# float32-era point, is retired. Caveat: §A3 measured the *retrieval*, not forward-radiance
# accuracy directly — 1e-4 is the user-standardized tier, 2026-06-29.)
#
# tol is DELIBERATELY NOT in signature(): the gate fingerprints what y *means* (the observing
# system, identical across accuracy tiers), whereas tol is *how accurately* y was computed and
# may legitimately differ between the truth cache and a looser operational retrieval forward.
# It is carried as an accuracy TAG on the cache (generate_osse_radiances writes it; load_radiance
# returns it; the IC + retrieval workers assert it against env RADIANCE_TOL) so a wrong-tol cache
# cannot be silently mixed in.
SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-4"))

# --- the fixed observing system -------------------------------------------------
BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]
NB = len(BANDS)
# Bands used for the τ_bot pre-retrieval (conservative scattering: ω=1.000 at
# 0.55/0.67/0.86 µm → reflectance dominated by total optical depth, r_e coupling
# only through g which is weak over the VOCALS r_e range).
VIS_BANDS = (0, 1, 2)
NQUAD = 48
N_PHYS = NQUAD // 2                                  # = 24 operational retrieval views (= NQuad//2)
N_VIEW_FULL = N_PHYS + 8                             # = 32 superset (8 extras = IC sanity check)
MU_HI, MU_LO = 0.95, 0.25                            # near-nadir / oblique envelope (θ≈18°..76°;
#                                                     plane-parallel + stream-interp validity, §μ0.25)
MU0 = 0.9
I0 = 1.0
PHI0 = 0.0
# NLEG_ALL = the FULL phase-function moment count, used ONLY for the Nakajima-Tanaka
# TMS single-scatter correction (NOT the discrete-ordinate solve, which truncates at
# NLeg=NQuad=48 per DISORT). 128 was catastrophically too few: the short-λ Mie forward
# peak (size parameter up to ~228 at 0.55 µm / r_e=20) Gibbs-rings the 128-moment
# reconstruction NEGATIVE, so TMS injected unphysical (min ≈ −2.5) radiances for EVERY
# cloud at the short bands — contaminating the original capstone AND the §15 IC. For the
# r_e≤20 clamp (truth max 18.1 µm + margin), the gamma-averaged 0.55 µm phase function
# stays positive by ~768 moments and is well-converged by ~1024. STANDARD = 1536, a
# deliberate conservative margin that ALSO keeps the sharper r_e=25 phase function
# (size parameter ~285 at 0.55 µm) positive and converged — so 1536 covers an r_e clamp
# raised to 25 with no negative-radiance issue (DESIGN: TMS NLeg_all study). n_gl must be
# ≥ ~2.35×NLeg_all to project accurately; 4096/1536≈2.67 satisfies this (3072 too coarse).
NLEG_ALL = 1536
N_GL = 4096
V_EFF = 0.10
ALBEDO = 0.06                                        # Lambertian sea-surface (BDRF)
RE_BOUNDS = (2.0, 20.0)                              # optics-table support / GN clamp. STANDARD = 20
#   (VOCALS truth max r_e = 18.1 µm + ~2 margin; r_e never exceeds 18 in VOCALS). 25 also works with
#   NO ISSUE — NLEG_ALL=1536 keeps the r_e=25 phase function positive & converged — so 20 is a
#   sufficiency choice (avoid over-extending the table), NOT a correctness limit. (If the clamp is
#   actually raised to 25, re-confirm NFOURIER: the mode study saw K≈27 at r_e=25 vs 24 tuned at 20.)
RE_GRID_N = 32                                       # optics-table r_e grid points
N_RADII = 600                                        # optics-table gamma-quadrature radii
RE_CLASS = "re5-linear"
# Single-sided principal-plane fan (φ=π), but with IRREGULAR view-zenith spacing — a
# golden-ratio low-discrepancy set over θ∈[arccos 0.95, arccos 0.25]≈[18°,76°], both
# endpoints anchored. Why irregular (DESIGN §14): real hyper-angular instruments (HARP2)
# sample irregularly, AND a *regular* μ-grid aliases with the conservative-band angular
# Jacobian (the spurious §15 0.55 µm DOFS notch). Distributing in θ (the instrument-natural
# along-track angle) is more realistic than uniform-in-μ. The set is built incrementally so
# it is NESTED: the first N_PHYS=24 (generation order) are the OPERATIONAL retrieval fan;
# the first N_VIEW_FULL=32 are the superset (the 8 extras fall in the 24's gaps — the IC
# angular-saturation sanity check). Capturing the *spirit* of HARP2 (many/wide/irregular),
# not its cloudbow-dense fore-aft geometry (irrelevant here). Both share ONE radiance cache.
def _irregular_views(n_total, n_retr, mu_hi, mu_lo):
    th_hi = np.degrees(np.arccos(mu_hi))             # near-nadir end (small θ)
    th_lo = np.degrees(np.arccos(mu_lo))             # oblique end (large θ)
    g = 0.6180339887498949                           # golden ratio (most-irrational → low-discrepancy)
    order, i = [0.0, 1.0], 1                          # endpoints anchored first
    while len(order) < n_total and i < 10000:
        t = (i * g) % 1.0
        if all(abs(t - o) > 0.012 for o in order):   # reject near-duplicates
            order.append(t)
        i += 1
    order = order[:n_total]
    th = lambda ts: th_hi + np.array(sorted(ts)) * (th_lo - th_hi)
    th_full = th(order)                              # sorted nadir→oblique (32)
    th_retr = th(order[:n_retr])                     # the operational 24 (first in gen order)
    mu_full = np.cos(np.radians(th_full))
    retr_idx = sorted(int(np.argmin(np.abs(th_full - t))) for t in th_retr)
    return mu_full, np.array(retr_idx)


VIEW_MU_FULL, RETRIEVAL_VIEW_IDX = _irregular_views(N_VIEW_FULL, N_PHYS, MU_HI, MU_LO)
VIEW_PHI_FULL = np.full(N_VIEW_FULL, pi)
# operational retrieval fan = the 24-subset of the 32 superset
VIEW_MU = VIEW_MU_FULL[RETRIEVAL_VIEW_IDX]
VIEW_PHI = np.full(N_PHYS, pi)

# --- per-band azimuthal mode ceiling (from nfourier_study.py, 2026-06-27) ----------
# Stress-state convergence study (r_e=25 µm clamp ceiling, τ∈{2,40}, μ=0.25 principal
# plane): truncation tail < 1e-4 absolute reflectance AND < 0.1σ at K=27 for every band
# (all bands remain strongly forward-peaked at r_e=25, size parameter 39..206); rounded
# up to a clean, conservative 30 (margin over K=27 → tail ≪ 1e-5). ≪ noise (the standing
# rule). REPLACES the old default 8, which left a ~10% truncation error in the short-λ
# bands at oblique principal-plane views (the water glory/rainbow is azimuthally sharp).
NFOURIER = [24] * NB                                 # re-tuned on the FIXED forward (DESIGN: NFourier
#   re-tune): 0.55 µm / r_e=20 / thin τ=2 worst case → rel<1% (PythonicDISORT tol) at K=24, abs<1e-3
#   (≈ dark-scene noise) at K=22. The old 30 carried an over-strict 1e-4 + the retired r_e=25 margin.


def signature():
    """Stable fingerprint of the observing system + forward-model settings. Any change
    that alters ``y = F(truth)`` changes this hash, invalidating a stale radiance cache."""
    payload = dict(
        bands=[float(b) for b in BANDS], nquad=NQUAD, mu0=MU0, i0=I0, phi0=PHI0,
        nleg_all=NLEG_ALL, n_gl=N_GL, v_eff=V_EFF, albedo=ALBEDO, re_bounds=list(RE_BOUNDS),
        re_grid_n=RE_GRID_N, n_radii=N_RADII, re_class=RE_CLASS,
        view_mu_full=[round(float(m), 6) for m in VIEW_MU_FULL],   # 32-view superset (generation)
        retrieval_view_idx=[int(i) for i in RETRIEVAL_VIEW_IDX],   # the 24 operational columns
        view_phi=round(float(VIEW_PHI_FULL[0]), 6),                # single-sided principal plane
        nfourier=[int(v) for v in NFOURIER],
        delta_M=True, NT_cor=True, x64=True,
    )
    # NB: tol / operational precision are deliberately NOT in this gate. The gate
    # fingerprints what y *means* (the observing system), which is identical across
    # accuracy tiers; tol/precision are how *accurately* y was computed and legitimately
    # differ between the truth cache (tol*) and the operational retrieval forward. tol is
    # carried as an accuracy TAG on the cache instead (load_radiance returns it; consolidate
    # asserts one tol per cache) — so it cannot be mixed, without breaking the tiered load.
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload, h


def build_forward(opt_bands, *, tau_bot, r_base, views="retrieval", state_space="log",
                  jac_mode="fwd", retrieve_tau_bot=True, retrieve_r_base=True,
                  tol=SOLVER_TOL, mode_map="scan"):
    """Construct the canonical ``RetrievalForward`` (per-band NFourier). ``views`` selects
    the angular fan: ``'full'`` = the 32-view superset (radiance GENERATION + IC sweep);
    ``'retrieval'`` = the 24 operational views (the retrieval). ``tau_bot`` / ``r_base``
    are the leak-free first-guess anchors (climatology), not the truth.

    ``tol`` = the adaptive Kvaerno5 ODE tolerance (default SOLVER_TOL = 1e-4, the standardized
    truth-tier point — see the SOLVER_TOL note). ``mode_map`` = 'scan' (CPU) |
    'vmap' (GPU bands×modes 240-way; ~17× per jacfwd). Both are threaded so the radiance
    re-gen and the GPU path can dial precision/speed without touching call sites."""
    import retrieval_oe as roe
    vm, vp = (VIEW_MU_FULL, VIEW_PHI_FULL) if views == "full" else (VIEW_MU, VIEW_PHI)
    return roe.RetrievalForward(
        opt_bands, NQuad=NQUAD, mu0=MU0, I0=I0, phi0=PHI0, tau_bot=tau_bot, r_base=r_base,
        view_mu=vm, view_phi=vp, BDRF_bands=[[ALBEDO]] * NB,
        NLeg_all=NLEG_ALL, NFourier=NFOURIER, re_class=RE_CLASS, state_space=state_space,
        jac_mode=jac_mode, retrieve_tau_bot=retrieve_tau_bot,
        retrieve_r_base=retrieve_r_base, re_bounds=RE_BOUNDS, tol=tol, mode_map=mode_map)


def select_retrieval_views(y_full):
    """From a 32-view-superset (band-major) measurement ``y_full`` (shape NB*32), select
    the 24 operational-fan columns → shape NB*24, the retrieval observation vector."""
    y = np.asarray(y_full, float).reshape(NB, N_VIEW_FULL)
    return y[:, RETRIEVAL_VIEW_IDX].reshape(-1)


def load_optics(cache_path):
    """Build/load the canonical optics table and per-band channel views."""
    import optics_table as ot
    re_table = ot.build_or_load_table(BANDS, RE_BOUNDS[0], RE_BOUNDS[1], RE_GRID_N,
                                      V_EFF, cache_path=cache_path, NLeg=NLEG_ALL,
                                      n_radii=N_RADII, n_gl=N_GL)
    return [ot.select_channel(re_table, i) for i in range(NB)]


def load_radiance(cache_path, index):
    """Load the precomputed ``y = F(truth)`` for ``index`` from the radiance cache,
    asserting the cache's observing-system signature matches this module. Returns the
    full record (y + truth metadata)."""
    import numpy as _np
    d = _np.load(cache_path, allow_pickle=True)
    _, want = signature()
    got = str(d["signature_hash"]) if "signature_hash" in d else "<none>"
    if got != want:
        raise ValueError(
            f"radiance-cache signature mismatch for index {index}: cache={got} "
            f"!= config={want}. The cache was generated with a different observing "
            f"system / NFourier; regenerate with generate_osse_radiances.py.")
    idx = int(index)
    rec = {k: d[f"{idx}_{k}"] for k in
           ("y", "tau", "re", "r_base", "tau_bot", "lwc", "altitude", "flight")}
    # accuracy tag: the tol this TRUTH cache was generated at (None for pre-tag caches,
    # e.g. the 543… tol=1e-3 batch). The caller verifies it is the expected truth tol;
    # the operational retrieval forward runs at its OWN (looser) tol independently.
    rec["tol"] = float(d["tol"]) if "tol" in d.files and d["tol"].item() is not None else None
    return rec
