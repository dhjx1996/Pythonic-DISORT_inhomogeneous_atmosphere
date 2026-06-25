"""DEFINITIVE IC analysis — builds §15 figs 1–4 data from the cached raw Jacobians.

The per-profile workers cache K_full (all 9 bands × all views), K_flux, s_int, the noise σ
and the prior covariances. Every figure quantity below is then a ~ms SVD of a *row subset* of
one K_full — so the spectral-saturation curve, the angular-on-top curve, the (n_bands × n_view)
trade-off grid and the regime-vs-τ scatter are all assembled here post-hoc, off the cache (no
forward solves). Writes docs/cached_results/info_content_definitive.json for the notebook.

Usage:
  ic_analysis_definitive.py <results_dir> [<results_dir> ...]   # dirs of *.npz sidecars
  (groups by the ic_mode stored in each npz; headline = priormean/loo)

Band ordering is **value-greedy** (literature-motivated; see DESIGN §13) with a **data-greedy**
cross-check (per-profile forward selection on marginal nadir DOFS, averaged). No ordering is baked
into the workers — it is applied here to the cached K.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
from retrieval_oe import posterior_diagnostics            # noqa: E402
from info_content import info_spectrum                    # noqa: E402

BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]
# NK1990-faithful literature order (indices into BANDS): {0.67,2.13} standard bispectral baseline,
# then the NK1990-EXPLORED higher-absorption bands 1.64 (NK's 1.65) -> 3.7 -> 4.05 (4.05 even more
# absorbing than 3.7 = the spectral-headroom test), then the modern OCI depth-graded fillers
# {1.24,1.038,2.26}, then redundant VIS {0.86,0.55}. This is the order the literature actually
# explored (NK1990 Fig 5 compares 2.16 vs 3.70; both the data-greedy ranking and NK1990 put 1.64/3.7
# among the top bands), so the saturation curve front-loads the informative bands. (Name "value-greedy"
# kept internally for back-compat; it is a fixed literature order, cross-checked by data_greedy.)
VALUE_ORDER = [1, 6, 5, 8, 9, 4, 3, 7, 2, 0]
VALUE_LABELS = [f"{BANDS[i]:g}" for i in VALUE_ORDER]
OUT = Path(os.environ.get("IC_DEFINITIVE_OUT",
                           _src.parents[0] / "docs" / "cached_results" / "info_content_definitive.json"))


def spread_idx(k, nv_max):
    return np.unique(np.linspace(0, nv_max - 1, k).round().astype(int))


def load_dirs(dirs):
    """Load every *.npz sidecar, grouped by (ic_mode). Skip degenerate/empty."""
    groups = defaultdict(list)
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            try:
                z = np.load(f, allow_pickle=True)
                if "K_full" not in z.files:
                    continue
                rec = {k: z[k] for k in z.files}
                groups[str(z["ic_mode"])].append(rec)
            except (OSError, ValueError, KeyError):
                continue
    return groups


def _metrics(K, Se, Sa, s):
    post = posterior_diagnostics(K, Sa, Se)
    spec = info_spectrum(K, Sa, Se)
    ix = np.where(post.data_fraction >= 0.5)[0]
    return float(post.dofs), float(spec.sic), (float(s[ix.max()]) if len(ix) else 0.0)


def _fast_dofs_sic(K, sig, Sa, Sa_inv):
    """DOFS = tr((G+Sa⁻¹)⁻¹G), SIC = ½log₂|I+Sa·G| with G = KᵀSₑ⁻¹K and **diagonal** Sₑ=diag(σ²) — so
    G = (K/σ)ᵀ(K/σ) is m×m (m = #state nodes), O(rows·m²) regardless of #rows. Avoids the rows×rows
    inverse in ``posterior_diagnostics`` → makes exact 2ⁿ-subset Shapley tractable. Verified == it."""
    m = Sa.shape[0]
    if K.shape[0] == 0:
        return 0.0, 0.0
    Kw = K / sig[:, None]
    G = Kw.T @ Kw
    S_hat = np.linalg.inv(G + Sa_inv)
    dofs = float(np.trace(S_hat @ G))
    _, ld = np.linalg.slogdet(np.eye(m) + Sa @ G)
    return dofs, float(0.5 * ld / np.log(2.0))


def _shapley(dvals, nb):
    """Per-band Shapley value from {frozenset(subset): scalar} over all 2ⁿ subsets (sums to D(full))."""
    import math
    from itertools import combinations
    w = [math.factorial(k) * math.factorial(nb - k - 1) / math.factorial(nb) for k in range(nb)]
    sh = np.zeros(nb)
    for b in range(nb):
        rest = [x for x in range(nb) if x != b]
        for k in range(nb):
            for S in combinations(rest, k):
                sh[b] += w[k] * (dvals[frozenset(S + (b,))] - dvals[frozenset(S)])
    return sh


def _all_subset_metrics(c, views, Sa, Sa_inv, J=None):
    """{frozenset(bands): (dofs, sic)} over all 2ⁿ band subsets at the given views (J = adiabatic tangent
    or None for the free-node state). Uses the fast Sₑ-diagonal evaluator."""
    from itertools import combinations
    nb = c.nb
    out = {}
    for k in range(nb + 1):
        for S in combinations(range(nb), k):
            rows = c.rows(list(S), views) if S else np.empty(0, int)
            K = c.K_full[rows]
            if J is not None:
                K = K @ J
            out[frozenset(S)] = _fast_dofs_sic(K, c.sig[rows], Sa, Sa_inv)
    return out


class Cache:
    """One profile's cached Jacobian + convenient row/metric helpers."""

    def __init__(self, rec):
        self.tau_bot = float(rec["tau_bot"])
        self.K_full = np.asarray(rec["K_full"], float)
        self.K_flux = np.asarray(rec["K_flux"], float)
        self.sig = np.asarray(rec["sigma_full"], float)
        self.sig_flux = np.asarray(rec["sigma_flux"], float)
        self.s = np.asarray(rec["s_int"], float)
        self.nb = int(rec["n_bands"])
        self.nvm = int(rec["nv_max"])
        self.nphys = int(rec["n_phys"])
        self.Sa = {k[3:]: np.asarray(rec[k], float) for k in rec if k.startswith("Sa_")}
        self.y = np.asarray(rec["y_full"], float) if "y_full" in rec else None      # reflectance ->
        self.y_flux = np.asarray(rec["y_flux"], float) if "y_flux" in rec else None  # rebuild Se @ any noise
        self.x_lin = np.asarray(rec["x_lin"], float) if "x_lin" in rec else None     # [r_e(s_ref), r_base, tau]

    def rows(self, band_idx, views):
        return np.array([b * self.nvm + v for b in band_idx for v in views])

    def m_radiance(self, band_idx, V, prior="loo"):
        v = spread_idx(V, self.nvm)
        r = self.rows(band_idx, v)
        return _metrics(self.K_full[r], np.diag(self.sig[r] ** 2), self.Sa[prior], self.s)

    def m_nadir(self, band_idx, prior="loo"):
        return self.m_radiance(band_idx, 1, prior)

    def m_albedo(self, band_idx, prior="loo"):
        r = np.array(band_idx)
        return _metrics(self.K_flux[r], np.diag(self.sig_flux[r] ** 2), self.Sa[prior], self.s)


def _mean_std(rows):
    a = np.asarray(rows, float)
    return a.mean(0).tolist(), a.std(0).tolist()


def spectral_saturation(caches, prior="loo"):
    """Fig 1: albedo & nadir DOFS/SIC vs n_bands (value order, 2→9)."""
    nb_axis = list(range(2, len(VALUE_ORDER) + 1))
    alb_d, alb_s, nad_d, nad_s = [], [], [], []
    for c in caches:
        order = VALUE_ORDER[:c.nb]
        ad, as_, nd, ns = [], [], [], []
        for b in nb_axis:
            bset = order[:b]
            da, sa, _ = c.m_albedo(bset, prior)
            dn, sn, _ = c.m_nadir(bset, prior)
            ad.append(da); as_.append(sa); nd.append(dn); ns.append(sn)
        alb_d.append(ad); alb_s.append(as_); nad_d.append(nd); nad_s.append(ns)
    out = {"n_bands": nb_axis, "labels_added": VALUE_LABELS[1:len(VALUE_ORDER)],
           "value_order_labels": VALUE_LABELS}
    for nm, arr in [("albedo_dofs", alb_d), ("albedo_sic", alb_s),
                    ("nadir_dofs", nad_d), ("nadir_sic", nad_s)]:
        out[nm + "_mean"], out[nm + "_std"] = _mean_std(arr)
    # n_95: ILLUSTRATIVE only (we do NOT pick a hard saturation point) — the n_bands reaching 95 % of the
    # 10-band mean nadir DOFS (=6). The data-greedy curve has a soft knee; reported in findings as a data
    # point, not a meaningful threshold. Figures draw NO N_sat line; the angular trio uses all 10 bands.
    nd = np.asarray(out["nadir_dofs_mean"])
    out["n_sat"] = int(nb_axis[int(np.argmax(nd >= 0.95 * nd[-1]))])
    return out


def angular_on_top(caches, n_sat, prior="loo"):
    """Fig 2: at N_sat bands, DOFS/SIC/depth vs n_view (1→N→beyond).

    ``depth`` is the deepest node (normalized s=τ/τ_bot) with data-fraction≥0.5; ``depth_tau`` is
    the same reach in *optical-depth* units (s·τ_bot) — the physically-meaningful penetration that
    distinguishes thin from thick clouds (a thick cloud reaches a small s but a large τ)."""
    nvm = min(caches[0].nvm, 28)   # cap at 28 views (29-32 are TMS-extrapolated past N=24)
    ladder = list(range(1, nvm + 1))   # UNIFORM view-count spacing (step 1), 1..28
    D, S, Dep, Dept = [], [], [], []
    for c in caches:
        bset = VALUE_ORDER[:n_sat]
        d, s_, dep, dept = [], [], [], []
        for V in ladder:
            dd, ss, pp = c.m_radiance(bset, V, prior)
            d.append(dd); s_.append(ss); dep.append(pp); dept.append(pp * c.tau_bot)
        D.append(d); S.append(s_); Dep.append(dep); Dept.append(dept)
    out = {"n_view": ladder, "n_sat": n_sat, "n_phys": caches[0].nphys}
    for nm, arr in [("dofs", D), ("sic", S), ("depth", Dep), ("depth_tau", Dept)]:
        out[nm + "_mean"], out[nm + "_std"] = _mean_std(arr)
    return out


def tradeoff_grid(caches, prior="loo"):
    """Fig 3: (n_bands × n_view) mean DOFS/SIC grid + Δ_ang(b), Δ_spec(v)."""
    nvm = caches[0].nvm
    nb_axis = list(range(2, len(VALUE_ORDER) + 1))
    # UNIFORM view axis (1..28, step 1) — was geometric [1,2,4,8,16,24,32], which plots
    # at uniform pixel spacing with non-uniform labels (hard to read). Capped at 28 views:
    # 29-32 are TMS-extrapolated past the N=24 stream ceiling, so 28 is the honest max.
    v_axis = list(range(1, min(nvm, 28) + 1))
    Dg = np.zeros((len(caches), len(nb_axis), len(v_axis)))
    Sg = np.zeros_like(Dg)
    for ci, c in enumerate(caches):
        for bi, b in enumerate(nb_axis):
            bset = VALUE_ORDER[:b]
            for vi, V in enumerate(v_axis):
                dd, ss, _ = c.m_radiance(bset, V, prior)
                Dg[ci, bi, vi] = dd
                Sg[ci, bi, vi] = ss
    Dm, Sm = Dg.mean(0), Sg.mean(0)
    # Δ_ang(b) = D(b, V_max) − D(b, 1 view);  Δ_spec(v) = D(9, v) − D(2, v)
    return {"n_bands": nb_axis, "n_view": v_axis, "band_labels": VALUE_LABELS,
            "dofs_mean": Dm.tolist(), "sic_mean": Sm.tolist(),
            "delta_ang_dofs": (Dm[:, -1] - Dm[:, 0]).tolist(),
            "delta_ang_sic": (Sm[:, -1] - Sm[:, 0]).tolist(),
            "delta_spec_dofs": (Dm[-1, :] - Dm[0, :]).tolist(),
            "delta_spec_sic": (Sm[-1, :] - Sm[0, :]).tolist()}


def regime_vs_tau(caches, prior="loo"):
    """Fig 4: per-profile DOFS/SIC (albedo/nadir/fullview) vs τ_bot (for best-fit lines)."""
    pts = []
    for c in caches:
        full = VALUE_ORDER[:c.nb]
        da, sa, _ = c.m_albedo(full, prior)
        dn, sn, _ = c.m_nadir(full, prior)
        dv, sv, _ = c.m_radiance(full, c.nphys, prior)
        pts.append(dict(tau_bot=c.tau_bot, dofs_albedo=da, dofs_nadir=dn, dofs_fullview=dv,
                        sic_albedo=sa, sic_nadir=sn, sic_fullview=sv))
    return sorted(pts, key=lambda p: p["tau_bot"])


def data_greedy(caches, prior="loo"):
    """Cross-check: per-profile forward selection maximizing marginal nadir DOFS; mean rank +
    the mean best-first DOFS trajectory (k=1..nb), for the Fig-1 overlay vs the literature order."""
    nb = caches[0].nb
    ranks = np.zeros(nb)
    trajs = []
    for c in caches:
        chosen, remaining, traj = [], list(range(nb)), []
        for step in range(nb):
            best, best_d = None, -1.0
            for b in remaining:
                d, _, _ = c.m_nadir(chosen + [b], prior)
                if d > best_d:
                    best_d, best = d, b
            ranks[best] += step
            chosen.append(best); remaining.remove(best); traj.append(best_d)
        trajs.append(traj)
    mean_rank = (ranks / len(caches))
    order = list(np.argsort(mean_rank))
    return {"mean_rank": mean_rank.tolist(), "data_greedy_order": order,
            "data_greedy_labels": [f"{BANDS[i]:g}" for i in order],
            "value_greedy_order": VALUE_ORDER,
            "n_bands_axis": list(range(1, nb + 1)),
            "curve_dofs_mean": np.asarray(trajs).mean(0).tolist()}


def greedy_order(caches, seed=(1, 6), nview=1, prior="loo"):
    """THE band-addition order, **contextualised to ``nview`` views**: a FIXED 2-band bispectral baseline
    {0.67, 2.13} = indices (1, 6) — so n_bands=2 is the standard bispectral method — then population
    forward-greedy (each step adds the band maximising the mean DOFS *at ``nview`` views*). ``nview=1`` is
    the nadir context (the saturation curves + grid); ``nview=n_phys`` is the full-view context (the
    Δ_ang / angular-context curves). The two orders differ because multi-angle viewing unlocks the
    penetrating bands, so the most-informative-next band is not the same at nadir and at full view."""
    nb = caches[0].nb
    chosen = list(seed)
    remaining = [b for b in range(nb) if b not in chosen]
    while remaining:
        best, best_d = None, -1.0
        for b in remaining:
            d = float(np.mean([c.m_radiance(chosen + [b], nview, prior)[0] for c in caches]))
            if d > best_d:
                best_d, best = d, b
        chosen.append(best); remaining.remove(best)
    return chosen


def angular_context_saturation(caches, prior="loo"):
    """Spectral saturation in the FULL-VIEW (angular) context: bands added **full-view-greedy** (an order
    that differs from the nadir-context Fig 1), giving the full-view DOFS/SIC vs n_bands *and* Δ_ang(b)
    along that same full-view-optimal path. Answers 'how many bands does the multi-angle retrieval need
    before spectral saturates', and makes the Δ_ang annotations consistent with the full-view ordering."""
    nph = caches[0].nphys
    order = greedy_order(caches, seed=(1, 6), nview=nph, prior=prior)
    nb_axis = list(range(2, len(order) + 1))
    Fd, Fs, Nd, Ns = [], [], [], []
    for c in caches:
        fd, fs, nd, ns = [], [], [], []
        for b in nb_axis:
            bset = order[:b]
            d_f, s_f, _ = c.m_radiance(bset, nph, prior)
            d_n, s_n, _ = c.m_nadir(bset, prior)
            fd.append(d_f); fs.append(s_f); nd.append(d_n); ns.append(s_n)
        Fd.append(fd); Fs.append(fs); Nd.append(nd); Ns.append(ns)
    Fd, Fs, Nd, Ns = (np.asarray(a) for a in (Fd, Fs, Nd, Ns))
    labels = [f"{BANDS[i]:g}" for i in order]
    out = {"n_bands": nb_axis, "order": order, "labels": labels, "labels_added": labels[1:], "n_view": nph,
           "full_dofs_mean": Fd.mean(0).tolist(), "full_dofs_std": Fd.std(0).tolist(),
           "full_sic_mean": Fs.mean(0).tolist(), "full_sic_std": Fs.std(0).tolist(),
           "nadir_dofs_mean": Nd.mean(0).tolist(), "nadir_sic_mean": Ns.mean(0).tolist(),
           "delta_ang_dofs": (Fd.mean(0) - Nd.mean(0)).tolist(),
           "delta_ang_sic": (Fs.mean(0) - Ns.mean(0)).tolist()}
    fd = np.asarray(out["full_dofs_mean"])
    out["n_sat"] = int(nb_axis[int(np.argmax(fd >= 0.97 * fd[-1]))])
    return out


def substitution_shapley(caches, band_idx=(8, 9, 4), prior="loo", n_views=10):
    """Fig 3b: the **Shapley SHARE** (fraction of the total DOFS / SIC) commanded by selected bands vs
    #views — superset-INDEPENDENT (each band's fair share of the information, averaged over all 2ⁿ subsets;
    the Shapley values sum to the full-set DOFS/SIC). The absorbing bands' share **falls** with views
    (angle substitutes for them); the penetrating band's **rises** (angle complements it). Exact Shapley at
    each view count via the fast Sₑ-diagonal evaluator; uniform view grid 1..28 (n_views points)."""
    nb = caches[0].nb
    nvm = min(caches[0].nvm, 28)
    v_axis = [int(v) for v in np.unique(np.linspace(1, nvm, n_views).round().astype(int))]
    accD = {b: np.zeros(len(v_axis)) for b in band_idx}
    accS = {b: np.zeros(len(v_axis)) for b in band_idx}
    for c in caches:
        Sa = c.Sa[prior]; Sai = np.linalg.inv(Sa)
        for vi, V in enumerate(v_axis):
            M = _all_subset_metrics(c, spread_idx(V, c.nvm), Sa, Sai)
            shD = _shapley({S: m[0] for S, m in M.items()}, nb); shD = shD / shD.sum()
            shS = _shapley({S: m[1] for S, m in M.items()}, nb); shS = shS / shS.sum()
            for b in band_idx:
                accD[b][vi] += shD[b]; accS[b][vi] += shS[b]
    n = len(caches)
    out = {"n_view": v_axis, "band_idx": list(band_idx),
           "band_labels": [f"{BANDS[b]:g}" for b in band_idx]}
    for b in band_idx:
        out[f"share_dofs_{BANDS[b]:g}"] = (accD[b] / n).tolist()
        out[f"share_sic_{BANDS[b]:g}"] = (accS[b] / n).tolist()
    return out


def _adia_tangent(s, r_top, r_base):
    """J (n_nodes x 2) = ∂r_e(s)/∂(r_top, r_base) for the adiabatic r_e(s)=(r_base^5+(r_top^5-r_base^5)(1-s))^(1/5)."""
    re = (r_base ** 5 + (r_top ** 5 - r_base ** 5) * (1.0 - s)) ** 0.2
    return np.stack([(r_top ** 4) * (1.0 - s) / re ** 4, (r_base ** 4) * s / re ** 4], axis=1)


def adiabatic_comparison(caches, prior="loo"):
    """Does the FREE-NODE (non-adiabatic) r_e(τ) state use the ToA / shallow-penetration bands better than a
    2-parameter ADIABATIC state? Project the cached free-node Jacobian onto the adiabatic (r_top, r_base)
    tangent — K_adia = K_full @ J — and compare per-band **Shapley DOFS** (fair, superset-independent; sums
    to the full-set DOFS) and the ToA-band Shapley share, for both nadir and full-view. (The linearisation
    point IS the adiabatic prior mean, so the projection is exact there.)"""
    nb = caches[0].nb
    nph = caches[0].nphys
    toa = [8, 9]                                                       # 3.7, 4.05
    out = {"bands": list(BANDS), "labels": [f"{x:g}" for x in BANDS], "toa_idx": toa}
    for ctx, V in [("nadir", 1), ("fullview", nph)]:
        shF = np.zeros(nb); shA = np.zeros(nb); nuse = 0
        for c in caches:
            if c.x_lin is None:
                continue
            nuse += 1
            r_top, r_base = float(c.x_lin[0]), float(c.x_lin[5])
            J = _adia_tangent(c.s, r_top, r_base)
            bi = int(np.where(c.s >= 1.0 - 1e-9)[0][-1])               # base (r_base) node
            Sa = c.Sa[prior]; Sai = np.linalg.inv(Sa)
            Sa2 = Sa[np.ix_([0, bi], [0, bi])]; Sa2i = np.linalg.inv(Sa2)
            views = spread_idx(V, c.nvm)
            MF = _all_subset_metrics(c, views, Sa, Sai)                # free-node
            MA = _all_subset_metrics(c, views, Sa2, Sa2i, J=J)         # adiabatic projection
            shF += _shapley({S: m[0] for S, m in MF.items()}, nb)
            shA += _shapley({S: m[0] for S, m in MA.items()}, nb)
        shF /= max(nuse, 1); shA /= max(nuse, 1)
        out[ctx] = {"free_shapley": shF.tolist(), "adia_shapley": shA.tolist(),
                    "free_total": float(shF.sum()), "adia_total": float(shA.sum()),
                    "free_toa_share": float(shF[toa].sum() / shF.sum()),
                    "adia_toa_share": float(shA[toa].sum() / shA.sum())}
    return out


def band_shapley(caches, prior="loo"):
    """Test 2: each band's **Shapley** nadir contribution (DOFS & SIC), by wavelength — the fair,
    superset-INDEPENDENT attribution: a band's average marginal over ALL 2ⁿ subsets, summing to the
    full-set DOFS/SIC. Supersedes the leave-one-out marginal, which under-credits mutually-redundant
    bands (each VIS band's LOO is ~0 because another VIS band substitutes; Shapley splits the credit)."""
    nb = caches[0].nb
    accD = np.zeros(nb); accS = np.zeros(nb)
    for c in caches:
        Sa = c.Sa[prior]; Sai = np.linalg.inv(Sa)
        M = _all_subset_metrics(c, spread_idx(1, c.nvm), Sa, Sai)
        accD += _shapley({S: m[0] for S, m in M.items()}, nb)
        accS += _shapley({S: m[1] for S, m in M.items()}, nb)
    n = len(caches)
    return {"bands": list(BANDS), "labels": [f"{x:g}" for x in BANDS],
            "dofs_mean": (accD / n).tolist(), "sic_mean": (accS / n).tolist()}


def noise_sweep(caches, levels=(0.01, 0.02, 0.03, 0.05), prior="loo"):
    """Test 1: headline DOFS/SIC (albedo/nadir/fullview, full band set) vs OCI calibration-noise level.

    Free post-hoc — Sₑ is rebuilt from the cached reflectance ``y`` at each ``k_cal`` (the Jacobian is
    noise-independent), so this is a row-subset re-evaluation, not a re-run. The deep-base / angular
    gains rest on near-0.5 filter factors — the most noise-sensitive modes — so this is what defends
    the headline against realistic radiometry. Needs ``y_full``/``y_flux`` in the cache (newer workers)."""
    import noise_model as nm
    have = [c for c in caches if c.y is not None]
    nph = caches[0].nphys
    keys = ["dofs_albedo", "dofs_nadir", "dofs_fullview", "sic_albedo", "sic_nadir", "sic_fullview"]
    out = {"k_cal_pct": [round(100 * k, 1) for k in levels], "n": len(have)}
    for k in keys:
        out[k + "_mean"] = []
    if not have:
        return out                                         # older cache without y -> skip gracefully
    for lev in levels:
        acc = {k: [] for k in keys}
        for c in have:
            full = VALUE_ORDER[:c.nb]
            sigF = nm.oci_swir(k_cal=lev).sigma(c.y, n_bands=c.nb)
            sigA = nm.oci_swir(k_cal=lev).sigma(c.y_flux, n_bands=c.nb)
            rA = np.array(full)
            a_d, a_s, _ = _metrics(c.K_flux[rA], np.diag(sigA[rA] ** 2), c.Sa[prior], c.s)
            rn = c.rows(full, spread_idx(1, c.nvm))
            n_d, n_s, _ = _metrics(c.K_full[rn], np.diag(sigF[rn] ** 2), c.Sa[prior], c.s)
            rv = c.rows(full, spread_idx(nph, c.nvm))
            v_d, v_s, _ = _metrics(c.K_full[rv], np.diag(sigF[rv] ** 2), c.Sa[prior], c.s)
            for key, val in zip(keys, [a_d, n_d, v_d, a_s, n_s, v_s]):
                acc[key].append(val)
        for key in keys:
            out[key + "_mean"].append(float(np.mean(acc[key])))
    return out


def robustness(groups):
    """Scalar fullview DOFS/SIC across priors (loo/weak/loo2x) and modes (priormean/draw)."""
    out = {}
    for mode, recs in groups.items():
        caches = [Cache(r) for r in recs]
        if not caches:
            continue
        priors = sorted(set().union(*[set(c.Sa) for c in caches]))
        out[mode] = {}
        for pr in priors:
            dv = [c.m_radiance(VALUE_ORDER[:c.nb], c.nphys, pr)[0]
                  for c in caches if pr in c.Sa]
            sv = [c.m_radiance(VALUE_ORDER[:c.nb], c.nphys, pr)[1]
                  for c in caches if pr in c.Sa]
            out[mode][pr] = dict(dofs_fullview_mean=float(np.mean(dv)),
                                 dofs_fullview_std=float(np.std(dv)),
                                 sic_fullview_mean=float(np.mean(sv)), n=len(dv))
    return out


def main(dirs):
    global VALUE_ORDER, VALUE_LABELS
    groups = load_dirs(dirs)
    if not groups:
        raise SystemExit(f"no *.npz Jacobian sidecars found under {dirs}")
    head = "priormean" if "priormean" in groups else next(iter(groups))
    caches = [Cache(r) for r in groups[head]]
    print(f"loaded {sum(len(v) for v in groups.values())} sidecars; "
          f"headline mode={head} ({len(caches)} profiles)")
    # RETIRE the hand-set literature order (ill-defined). THE band-addition order = the standard
    # bispectral pair {0.67, 2.13} as a FIXED 2-band baseline, then data-greedy (population forward-
    # greedy on marginal nadir DOFS) for every band past the pair. Used for the order-dependent figures
    # (spectral saturation, trade-off grid); order-independent panels (LOO, Δ_ang endpoints, noise,
    # robustness) are unaffected.
    dg = data_greedy(caches)
    VALUE_ORDER = greedy_order(caches, seed=(1, 6), nview=1)   # NADIR context: saturation curves + grid
    VALUE_LABELS = [f"{BANDS[i]:g}" for i in VALUE_ORDER]
    sat = spectral_saturation(caches)
    result = dict(
        bands=BANDS, value_order=VALUE_ORDER, value_labels=VALUE_LABELS,
        headline_mode=head, n_profiles=len(caches),
        NQuad=int(caches[0].nphys * 2), modes=sorted(groups),
        spectral=sat,
        angular=angular_on_top(caches, caches[0].nb),   # angular trio on ALL 10 bands (no saturation pick)
        angular_context=angular_context_saturation(caches),   # FULL-VIEW context: Δ_ang + full-view saturation
        grid=tradeoff_grid(caches),
        substitution=substitution_shapley(caches),
        band_shapley=band_shapley(caches),
        adiabatic=adiabatic_comparison(caches),
        regime=regime_vs_tau(caches),
        data_greedy=dg,
        noise=noise_sweep(caches),
        robustness=robustness(groups))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=lambda o: (
        int(o) if isinstance(o, np.integer) else
        float(o) if isinstance(o, np.floating) else o.tolist())))
    print(f"N_sat={sat['n_sat']} bands | wrote {OUT}")
    print(f"  nadir DOFS (2→9 bands): {[round(x,2) for x in sat['nadir_dofs_mean']]}")
    print(f"  data-greedy band order: {VALUE_LABELS}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(Path(__file__).resolve().parents[1] / "tests" / "supplementary" / "results")])
