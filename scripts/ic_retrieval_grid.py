"""Retrieval-grid information-content profiling — pure post-processing (2026-07-15).

The user-approved IC reframing: profile information content ON THE OPERATIONAL
(QRCP-selected) retrieval grids, from the canonical campaign sidecars — no new
solves. Rationale (2026-07-14/15 texture forensics): the dense/ODE-grid Jacobian
texture inflates quadratic IC functionals without bound (phantom rank scales with
row/column count); the retrieval grid is the identifiable subspace where the same
functionals are texture-converged at production tolerance. QRCP-grid IC =
ODE-grid IC projected onto its identifiable part (§16 post-hoc justification of
the retrieval-grid derivation — user-approved sentence).

Per profile (sidecar K_log/Sa_log/sigma; pure OE prior, no curvature):
  * DOFS/SIC recomputed (sanity vs stored values)
  * per-BAND exact Shapley share of DOFS and SIC (10 players, subset enumeration
    over precomputed per-band Gram blocks)
  * per-VIEW leave-one-out ΔDOFS + exact 6-group angular Shapley (views grouped
    by mu into 6 adjacent groups of 4)
  * CPV2012-style sequential introduction (greedy most-informative-first):
    cumulative DOFS/SIC vs #bands (full 24-view fan) and vs #views (full 10-band)
  * band-saturation curve (the "hundreds of bands" proxy: plateau before 10)
  * matched-row-budget band-vs-angle DOFS (10b x 1v vs 2b x 5v, etc.)
  * information-pump statistic (Fig-5 companion): deepest-node posterior variance
    reduction under the full OU prior vs its DIAGONAL (decorrelated) version —
    the correlated-prior share of the deep-node constraint, all 125 profiles.

Initial-vs-final grid (user nuance B): the campaign never re-meshed (thr=2.0,
population max chi2=0.18), so initial==final everywhere EXCEPT the three promoted
pathology3 records (idx49/54/57); their select-once IC comes from the
_idx{N}_pre_pathology3/ backups and is compared explicitly.

Mismatch campaign: EXCLUDED (user ruling 2026-07-15 — cautionary tale only).

Usage: ic_retrieval_grid.py [parts_dir] [out.json]
       defaults: runs/_ve046_tik_fr_parts  docs/cached_results/ic_retrieval_grid.json
"""
import sys
import json
import itertools
from pathlib import Path

import numpy as np

BISPECTRAL = (0.67, 2.13)
N_VGROUP = 6


def _dofs_sic(G, Sa):
    """DOFS = tr[(G+Sa^-1)^-1 G]; SIC = 1/2 log2 det(I + L^T G L), L=chol(Sa)."""
    p = G.shape[0]
    Sa_inv = np.linalg.inv(Sa)
    dofs = float(np.trace(np.linalg.solve(G + Sa_inv, G)))
    L = np.linalg.cholesky(Sa)
    M = np.eye(p) + L.T @ G @ L
    sign, logdet = np.linalg.slogdet(M)
    return dofs, float(0.5 * logdet / np.log(2.0))


def shapley_shares(vals, n):
    """Exact Shapley values from a complete subset→scalar map (bitmask keys over
    ``n`` players; sums to vals[full])."""
    from math import factorial
    w = [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
    sh = np.zeros(n)
    for mask in range(1 << n):
        s = bin(mask).count("1")
        for i in range(n):
            if not mask >> i & 1:
                sh[i] += w[s] * (vals[mask | (1 << i)] - vals[mask])
    return sh


def shapley(grams, Sa, n_players):
    """Exact Shapley shares of DOFS and SIC over Gram blocks (subset enumeration)."""
    n = n_players
    # value of every subset (bitmask -> (dofs, sic))
    vals = {0: (0.0, 0.0)}
    for mask in range(1, 1 << n):
        G = sum(grams[i] for i in range(n) if mask >> i & 1)
        vals[mask] = _dofs_sic(G, Sa)
    sh_d = shapley_shares({m: v[0] for m, v in vals.items()}, n)
    sh_s = shapley_shares({m: v[1] for m, v in vals.items()}, n)
    return sh_d, sh_s, vals


def greedy_curve(grams, Sa, n_players):
    """Most-informative-first sequential introduction; cumulative (dofs, sic) + order."""
    chosen, order, curve = 0, [], []
    G = None
    for _ in range(n_players):
        best, best_v = None, (-1.0, None)
        for i in range(n_players):
            if chosen >> i & 1:
                continue
            Gi = grams[i] if G is None else G + grams[i]
            v = _dofs_sic(Gi, Sa)
            if v[0] > best_v[0]:
                best, best_v, best_G = i, v, Gi
        chosen |= 1 << best
        order.append(best); curve.append(best_v); G = best_G
    return order, curve


def analyze(npz_path):
    z = dict(np.load(npz_path, allow_pickle=True))
    K = np.asarray(z["K_log"], float)                  # (m, p) log-space
    Sa = np.asarray(z["Sa_log"], float)
    sig = np.asarray(z["sigma"], float)
    bands = np.asarray(z["bands"], float)
    vmu = np.asarray(z["view_mu"], float)
    NB, NV = len(bands), len(vmu)
    p = K.shape[1]
    k = int(z["k"])
    Kw = K / sig[:, None]                              # noise-whitened rows
    Kr = Kw.reshape(NB, NV, p)

    # Gram blocks
    g_band = [np.einsum("vp,vq->pq", Kr[b], Kr[b]) for b in range(NB)]
    g_view = [np.einsum("bp,bq->pq", Kr[:, v], Kr[:, v]) for v in range(NV)]
    order_mu = np.argsort(-vmu)                        # nadir-most first
    vgroups = np.array_split(order_mu, N_VGROUP)
    g_vgroup = [sum(g_view[v] for v in grp) for grp in vgroups]

    G_full = sum(g_band)
    dofs_full, sic_full = _dofs_sic(G_full, Sa)

    # per-band exact Shapley; per-view LOO; angular-group Shapley
    sh_band_d, sh_band_s, _ = shapley(g_band, Sa, NB)
    sh_vg_d, sh_vg_s, _ = shapley(g_vgroup, Sa, N_VGROUP)
    loo_view = [dofs_full - _dofs_sic(G_full - g_view[v], Sa)[0] for v in range(NV)]

    # sequential curves + band saturation
    band_order, band_curve = greedy_curve(g_band, Sa, NB)
    view_order, view_curve = greedy_curve(g_view, Sa, NV)

    # matched row budgets
    bi = [int(np.argmin(np.abs(bands - b))) for b in BISPECTRAL]
    def budget(bsel, vsel):
        G = sum(np.einsum("p,q->pq", Kr[b, v], Kr[b, v]) for b in bsel for v in vsel)
        return _dofs_sic(G, Sa)[0]
    vpick = lambda n: [int(order_mu[i]) for i in np.linspace(0, NV - 1, n).astype(int)]
    budgets = {"10bx1v": budget(range(NB), vpick(1)), "2bx5v": budget(bi, vpick(5)),
               "10bx2v": budget(range(NB), vpick(2)), "2bx10v": budget(bi, vpick(10)),
               "10bx4v": budget(range(NB), vpick(4)), "2bx20v": budget(bi, vpick(20))}

    # information pump: deepest r_e node, posterior var reduction, corr vs diag prior
    Sa_diag = np.diag(np.diag(Sa))
    S_hat = np.linalg.inv(G_full + np.linalg.inv(Sa))
    S_hat_d = np.linalg.inv(G_full + np.linalg.inv(Sa_diag))
    j = k - 1                                          # deepest r_e node index
    red_corr = 1.0 - S_hat[j, j] / Sa[j, j]
    red_diag = 1.0 - S_hat_d[j, j] / Sa_diag[j, j]

    return dict(
        index=int(z["index"]), k=k, p=p,
        tau_bot_truth=float(z["truth_tau_bot"]),
        s_nodes=[round(float(s), 4) for s in np.asarray(z["s_grid"])],
        dofs=dofs_full, sic=sic_full,
        dofs_stored=float(z["dofs"]), sic_stored=float(z["sic"]),
        shapley_band_dofs=[round(float(x), 4) for x in sh_band_d],
        shapley_band_sic=[round(float(x), 4) for x in sh_band_s],
        shapley_vgroup_dofs=[round(float(x), 4) for x in sh_vg_d],
        vgroup_mu=[[round(float(vmu[v]), 3) for v in grp] for grp in vgroups],
        loo_view_dofs=[round(float(x), 5) for x in loo_view],
        band_order=[float(bands[i]) for i in band_order],
        band_curve_dofs=[round(d, 4) for d, _ in band_curve],
        band_curve_sic=[round(s, 4) for _, s in band_curve],
        view_curve_dofs=[round(d, 4) for d, _ in view_curve],
        budgets={kk: round(vv, 4) for kk, vv in budgets.items()},
        pump_deep_reduction_corr=round(float(red_corr), 4),
        pump_deep_reduction_diag=round(float(red_diag), 4),
    )


def main():
    PARTS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/_ve046_tik_fr_parts")
    OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/cached_results/ic_retrieval_grid.json")
    recs, errs = [], []
    for pth in sorted(PARTS.glob("[0-9]*_A.npz"), key=lambda q: int(q.name.split("_")[0])):
        try:
            recs.append(analyze(pth))
        except Exception as e:                          # noqa: BLE001
            errs.append((pth.name, str(e)[:120]))
    # initial-vs-final for the promoted pathology records
    initial_final = {}
    for n in (49, 54, 57):
        bak = PARTS / f"_idx{n}_pre_pathology3" / f"{n}_A.npz"
        if bak.exists():
            try:
                initial_final[str(n)] = dict(final=analyze(PARTS / f"{n}_A.npz"),
                                             initial=analyze(bak))
            except Exception as e:                      # noqa: BLE001
                errs.append((f"init-final idx{n}", str(e)[:120]))

    d = np.array([r["dofs"] for r in recs])
    sanity = np.max([abs(r["dofs"] - r["dofs_stored"]) for r in recs])
    print(f"n={len(recs)} errs={errs or 'none'}  DOFS med={np.median(d):.3f} "
          f"(stored-recompute max diff {sanity:.2e})")
    sh = np.array([r["shapley_band_dofs"] for r in recs])
    bands = [f"{b:g}" for b in np.load(next(PARTS.glob('[0-9]*_A.npz')), allow_pickle=True)["bands"]]
    print("median per-band Shapley DOFS share:")
    for b, v in zip(bands, np.median(sh, axis=0)):
        print(f"   {b:>6}um: {v:.3f}")
    bc = np.array([r["band_curve_dofs"] for r in recs])
    print("band-saturation (median cumulative DOFS, greedy):",
          [round(float(x), 2) for x in np.median(bc, axis=0)])
    pump_c = np.array([r["pump_deep_reduction_corr"] for r in recs])
    pump_d = np.array([r["pump_deep_reduction_diag"] for r in recs])
    print(f"pump: deep-node var reduction corr={np.median(pump_c):.3f} "
          f"diag={np.median(pump_d):.3f} (median over {len(recs)})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(records=recs, initial_vs_final=initial_final,
                                   errors=errs, parts_dir=str(PARTS))))
    print("saved", OUT)


if __name__ == "__main__":
    main()
