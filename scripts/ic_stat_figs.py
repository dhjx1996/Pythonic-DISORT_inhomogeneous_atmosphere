"""Finding A/B population statistics figures from ic_retrieval_grid.json (2026-07-15).

Pure post-processing of docs/cached_results/ic_retrieval_grid.json (125 ve046-canonical
profiles, 0 errors) -- no RT, no cluster dependency. Renders:

  Fig A (spectral, Finding A) -- left: per-band exact-Shapley DOFS share (population
         median + IQR, 10 bands, colored by wavelength); right: greedy sequential-
         introduction saturation curve (CPV2012-style, most-informative-band-first),
         normalized cumulative DOFS vs #bands, population median + IQR.
  Fig B (angular, Finding B) -- left: 6-group angular Shapley DOFS share vs view-mu
         group (population median + IQR); right: matched-row-budget bands-vs-angles
         comparison for the 3 row-budget pairs (10bx1v/2bx5v, 10bx2v/2bx10v,
         10bx4v/2bx20v), population median with IQR whiskers.

idx49/57 (k=2, p=4 -- the pathology3 give-up grids) are excluded from all medians per
the standing caveat; n_included printed on each figure.

Usage: ic_stat_figs.py [ic_retrieval_grid.json] [out_prefix]
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/cached_results/ic_retrieval_grid.json")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/ve046"
EXCLUDE = {49, 57}   # k=2 give-up grids (pathology3), flagged not represented by "typical" IC

BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]
BAND_C = {0.55: "#5778a4", 0.67: "#6a9f58", 0.86: "#85b6b2", 1.038: "#e7ca60",
          1.24: "#a87c9f", 1.64: "#f1a2a9", 2.13: "#967662", 2.26: "#b8b0ac",
          3.7: "#d1615d", 4.05: "#e49444"}

d = json.loads(IN.read_text())
recs = [r for r in d["records"] if r["index"] not in EXCLUDE]
n = len(recs)
print(f"loaded {len(d['records'])} records, using {n} (excluded {sorted(EXCLUDE)})")


def med_iqr(rows):
    a = np.asarray(rows, float)
    return np.median(a, axis=0), np.percentile(a, 25, axis=0), np.percentile(a, 75, axis=0)


def figA():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    shap = np.asarray([r["shapley_band_dofs"] for r in recs], float)   # (n, 10), BANDS order
    med, q1, q3 = np.median(shap, axis=0), np.percentile(shap, 25, axis=0), np.percentile(shap, 75, axis=0)
    x = np.arange(len(BANDS))
    colors = [BAND_C[b] for b in BANDS]
    axL.bar(x, med, yerr=[med - q1, q3 - med], color=colors, capsize=3, edgecolor="#333", lw=0.5)
    axL.set_xticks(x)
    axL.set_xticklabels([f"{b:g}" for b in BANDS], rotation=45)
    axL.set_xlabel("band (µm)")
    axL.set_ylabel("Shapley DOFS share")
    axL.set_title("per-band exact Shapley (median ± IQR)", fontsize=10)
    axL.grid(alpha=0.25, lw=0.5, axis="y")

    curve = np.asarray([r["band_curve_dofs"] for r in recs], float)    # (n, 10) cumulative, greedy rank order
    frac = curve / curve[:, -1:]
    fmed, fq1, fq3 = np.median(frac, axis=0), np.percentile(frac, 25, axis=0), np.percentile(frac, 75, axis=0)
    rank = np.arange(1, 11)
    axR.plot(rank, fmed, color="#5778a4", lw=1.8, marker="o", ms=4)
    axR.fill_between(rank, fq1, fq3, color="#5778a4", alpha=0.2, lw=0)
    axR.axhline(1.0, color="#999", lw=0.6, ls=":")
    axR.set_xlabel("# bands added (greedy, most-informative-first)")
    axR.set_ylabel("cumulative DOFS / final DOFS")
    axR.set_title("spectral saturation (median ± IQR)", fontsize=10)
    axR.set_ylim(0, 1.05)
    axR.grid(alpha=0.25, lw=0.5)
    axR.annotate(f"1 band = {fmed[0]*100:.0f}%", (1, fmed[0]), textcoords="offset points",
                 xytext=(8, -12), fontsize=8.5)
    axR.annotate(f"4 bands = {fmed[3]*100:.0f}%", (4, fmed[3]), textcoords="offset points",
                 xytext=(8, -14), fontsize=8.5)

    fig.suptitle(f"Fig A (Finding A) — spectral information saturation, "
                 f"ve046 canonical (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_figA_spectral_saturation.png", dpi=150)
    print("saved", f"{PREFIX}_figA_spectral_saturation.png")
    print(f"  band_curve frac: rank1={fmed[0]:.3f} rank4={fmed[3]:.3f} rank10={fmed[-1]:.3f}")
    print(f"  shapley_band_dofs median: {dict(zip(BANDS, np.round(med, 4)))}")


def figB():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    vshap = np.asarray([r["shapley_vgroup_dofs"] for r in recs], float)   # (n, 6)
    vmed, vq1, vq3 = np.median(vshap, axis=0), np.percentile(vshap, 25, axis=0), np.percentile(vshap, 75, axis=0)
    vmu = np.asarray(recs[0]["vgroup_mu"], float)   # (6, 4) groups of 4 mu each — same grouping every profile
    vmu_mid = vmu.mean(axis=1)
    g = np.arange(1, 7)
    axL.plot(g, vmed, color="#d1615d", lw=1.8, marker="o", ms=4)
    axL.fill_between(g, vq1, vq3, color="#d1615d", alpha=0.2, lw=0)
    axL.set_xticks(g)
    axL.set_xticklabels([f"µ≈{m:.2f}" for m in vmu_mid], rotation=30, fontsize=8)
    axL.set_xlabel("view-mu group (nadir → oblique)")
    axL.set_ylabel("Shapley DOFS share (6-group)")
    axL.set_title("angular Shapley (median ± IQR) — never →0", fontsize=10)
    axL.set_ylim(0, 1.0)
    axL.grid(alpha=0.25, lw=0.5)

    pairs = [("10bx1v", "2bx5v"), ("10bx2v", "2bx10v"), ("10bx4v", "2bx20v")]
    labels = ["10 rows\n(10b×1v vs 2b×5v)", "20 rows\n(10b×2v vs 2b×10v)", "40 rows\n(10b×4v vs 2b×20v)"]
    bmed = {k: np.median([r["budgets"][k] for r in recs]) for pair in pairs for k in pair}
    bq1 = {k: np.percentile([r["budgets"][k] for r in recs], 25) for pair in pairs for k in pair}
    bq3 = {k: np.percentile([r["budgets"][k] for r in recs], 75) for pair in pairs for k in pair}
    xg = np.arange(len(pairs))
    w = 0.32
    bandsK = [p[0] for p in pairs]
    anglesK = [p[1] for p in pairs]
    axR.bar(xg - w / 2, [bmed[k] for k in bandsK], width=w, color="#5778a4", label="bands (10b × Nv)",
            yerr=[[bmed[k] - bq1[k] for k in bandsK], [bq3[k] - bmed[k] for k in bandsK]], capsize=3)
    axR.bar(xg + w / 2, [bmed[k] for k in anglesK], width=w, color="#e49444", label="angles (2b × Nv)",
            yerr=[[bmed[k] - bq1[k] for k in anglesK], [bq3[k] - bmed[k] for k in anglesK]], capsize=3)
    axR.set_xticks(xg)
    axR.set_xticklabels(labels, fontsize=8.5)
    axR.set_ylabel("DOFS (matched row budget)")
    axR.set_title("matched-row-budget: bands > angles (median ± IQR)", fontsize=10)
    axR.legend(fontsize=8, frameon=False)
    axR.grid(alpha=0.25, lw=0.5, axis="y")

    fig.suptitle(f"Fig B (Finding B) — angular axis: informative but not "
                 f"interchangeable with bands, ve046 canonical (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_figB_angular_budget.png", dpi=150)
    print("saved", f"{PREFIX}_figB_angular_budget.png")
    print(f"  vgroup shapley median: {np.round(vmed, 4).tolist()}")
    for bk, ak in pairs:
        print(f"  {bk}={bmed[bk]:.3f} vs {ak}={bmed[ak]:.3f}  Δ={bmed[bk]-bmed[ak]:+.3f}")


if __name__ == "__main__":
    figA()
    figB()
