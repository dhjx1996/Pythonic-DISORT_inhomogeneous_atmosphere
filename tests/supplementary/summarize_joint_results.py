"""Synthesise the joint-retrieval experiments.

Reads docs/joint_dofs_results.json (information content, linearized at truth) and
docs/joint_osse_results.json (full leak-free GN retrievals) and prints a compact
summary. Pure host-side; run any time after the experiments have written (partial) results:

    /tmp/jaxve/bin/python tests/supplementary/summarize_joint_results.py
"""
import json
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
DOFS = _root / "docs" / "joint_dofs_results.json"
OSSE = _root / "docs" / "joint_osse_results.json"


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def joint_dofs(dofs):
    print("\n" + "=" * 78)
    print("Information content (DOFS), linearized at the truth scene")
    print("=" * 78)
    if not dofs:
        print("  (no DOFS results yet)")
        return
    print(f"{'config':<20}{'bands':<22}{'A fixed':>8}{'B broad':>9}"
          f"{'  (prof/rbase/taub)':>22}{'C clim':>8}")
    for label, r in dofs.items():
        A, B, C = r["A_fixed_anchor"], r["B_joint_broad"], r["C_joint_clim"]
        decomp = f"({B['profile_dofs']:.2f}/{B['r_base_dofs']:.2f}/{B['tau_bot_dofs']:.2f})"
        print(f"{label:<20}{str(r['bands']):<22}{A['dofs']:>8.2f}{B['dofs']:>9.2f}"
              f"{decomp:>22}{C['dofs']:>8.2f}")
    print("\n  Reading:")
    print("  - A vs B 'prof': cost of making (tau_bot,r_base) unknown = how much the")
    print("    profile DOFS drops when the anchors are no longer given.")
    print("  - B 'taub' vs 'rbase': the measurement constrains tau_bot but barely r_base")
    print("    (shielded base) -- the per-component DOFS split.")
    print("  - B vs C: prior-dependence of DOFS (broad supplies more data-DOF; the")
    print("    tighter climatology prior lowers DOFS by pre-constraining).")
    print("  - *_current vs *_conservative: band reassessment (conservative 0.86um for")
    print("    tau_bot vs the 1.24um weak absorber).")
    for label, r in dofs.items():
        B = r["B_joint_broad"]
        print(f"    {label}: tau_bot 1sig prior {B['tau_bot_prior_sigma']:.2f} -> "
              f"post {B['tau_bot_sigma']:.2f}; r_base prior "
              f"{B['r_base_prior_sigma']:.2f} -> post {B['r_base_sigma']:.2f}")


def joint_osse(osse):
    print("\n" + "=" * 78)
    print("Full joint retrievals (leak-free; first guess = climatology)")
    print("=" * 78)
    if not osse:
        print("  (no OSSE results yet)")
        return
    for label, r in osse.items():
        t = r["truth"]
        ret = r["retrieved"]
        print(f"\n  {label} [{r['flight']}, {r['re_class']}, n_outer={r['n_outer']}]:")
        print(f"    converged={r['converged']}  ||y-F||={r['resid_norm']:.2e}  "
              f"DOFS={r['dofs']:.2f} (prof {r['dofs_profile']:.2f}/rbase "
              f"{r['dofs_r_base']:.2f}/taub {r['dofs_tau_bot']:.2f})")
        print(f"    tau_bot: truth {t['tau_bot']:.2f} -> ret {ret['tau_bot']:.2f} "
              f"(prior {r['clim']['tau_bot_mean']:.2f}, post 1sig "
              f"{r['tau_bot_post_sigma']:.2f})")
        print(f"    r_base : truth {t['r_base']:.2f} -> ret {ret['r_base']:.2f} "
              f"(post 1sig {r['r_base_post_sigma']:.2f})")
        print(f"    profile RMSE {r['profile_rmse']:.2f} um  on grid "
              f"{[round(x,2) for x in r['tau_grid']]}")


def node_count(osse):
    print("\n" + "=" * 78)
    print("Adaptive node count: auto_k_active (filter vs dofs) + DOFS robustness")
    print("=" * 78)
    if not osse:
        print("  (no OSSE results yet)")
        return
    print(f"{'config':<18}{'used k':>7}{'filter k':>9}{'dofs k':>8}"
          f"{'Sum_f':>8}{'DOFS':>7}{'  (agree?)':>12}")
    for label, r in osse.items():
        f, d = r["so1"]["filter"], r["so1"]["dofs"]
        agree = "yes" if abs(f["sum_filter_factor"] - d["dofs"]) < 0.3 else "DIFFER"
        print(f"{label:<18}{r['k_active_used']:>7}{r['k_filter']:>9}{r['k_dofs']:>8}"
              f"{f['sum_filter_factor']:>8.2f}{d['dofs']:>7.2f}{agree:>12}")
    print("\n  Sum_f ~ DOFS is the built-in cross-check (whitened-QRCP filter factors vs")
    print("  tr(A)); agreement => DOFS is a consistent info measure for this basis.")


def interp_model(osse):
    print("\n" + "=" * 78)
    print("Interpolation model comparison: re5-linear vs linear (same data)")
    print("=" * 78)
    pairs = [("thin", "thin_re5_n1", "thin_linear_n1"),
             ("thick", "thick_re5_n1", "thick_linear_n1")]
    for tag, a, b in pairs:
        if a in osse and b in osse:
            ra, rb = osse[a], osse[b]
            print(f"  {tag}: re5  ||y-F||={ra['resid_norm']:.2e} RMSE={ra['profile_rmse']:.2f} "
                  f"DOFS={ra['dofs']:.2f}")
            print(f"  {tag}: lin  ||y-F||={rb['resid_norm']:.2e} RMSE={rb['profile_rmse']:.2f} "
                  f"DOFS={rb['dofs']:.2f}")
            dres = abs(ra["resid_norm"] - rb["resid_norm"])
            distinguishable = dres >= 0.2 * max(ra["resid_norm"], rb["resid_norm"])
            verdict = "CAN" if distinguishable else "CANNOT"
            print(f"        -> data {verdict} distinguish the shapes "
                  f"(|Δ||y-F|||={dres:.2e}; compare to the noise/fit level).")
        else:
            print(f"  {tag}: (need {a} and {b})")


def remesh(osse):
    print("\n" + "=" * 78)
    print("Re-meshing / n_outer: placement stability (thin re5, n_outer 1 vs 2)")
    print("=" * 78)
    if "thin_re5_n1" in osse and "thin_re5_n2" in osse:
        r1, r2 = osse["thin_re5_n1"], osse["thin_re5_n2"]
        print(f"  n_outer=1 grid: {[round(x,2) for x in r1['tau_grid']]}  "
              f"||y-F||={r1['resid_norm']:.2e} RMSE={r1['profile_rmse']:.2f}")
        print(f"  n_outer=2 grid: {[round(x,2) for x in r2['tau_grid']]}  "
              f"||y-F||={r2['resid_norm']:.2e} RMSE={r2['profile_rmse']:.2f}")
        print("  -> did re-meshing move the grid, and did it help the fit/RMSE or just")
        print("     churn placement (OUTSTANDING G 're-mesh instability')?")
    else:
        print("  (need thin_re5_n1 and thin_re5_n2)")


if __name__ == "__main__":
    dofs, osse = _load(DOFS), _load(OSSE)
    print(f"DOFS configs: {list(dofs)}\nOSSE variants: {list(osse)}")
    joint_dofs(dofs)
    joint_osse(osse)
    node_count(osse)
    interp_model(osse)
    remesh(osse)
