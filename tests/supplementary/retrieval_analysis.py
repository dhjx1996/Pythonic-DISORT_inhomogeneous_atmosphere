"""retrieval_analysis.py — post-hoc metric suite for the all-125 `fr` retrievals.

Consumes the per-(index, config) sidecar ``*_{A,B}.npz`` files written by
``retrieval_worker._persist`` (fields defined in ``retrieve_one``) and computes the full
suite DESIGN_DECISIONS §15 specifies: **RMSE / ΔRMSE, LWP-bias, Mahalanobis (+ adiabatic
floor), posterior IC (DOFS/SIC/data_fraction)**. Pure NumPy/SciPy — no RT, no re-run;
every number is a re-computation from the saved arrays. Nothing is committed: it writes a
JSON summary and prints tables (results are delivered by zip, per project rule).

LWP definition (pinned by the user, 2026-07-01): **piecewise-adiabatic**, consistent with
the retrieval's own re⁵-linear interpolation between nodes. From the extinction–LWC–r_e
relation β = 3·LWC/(2ρ_w r_e) with Q_ext=2, LWP = (2ρ_w/3)∫₀^{τ_bot} r_e(τ) dτ; with r_e in
µm, τ dimensionless and ρ_w=1000 kg/m³ the unit constant collapses to
**LWP[g/m²] = (2/3)∫ r_e[µm] dτ**. On an re⁵-linear segment (r_e⁵ affine in τ) this integrates
in closed form, giving the piecewise sum
    LWP = (5/9) · Σ_i Δτ_i · (r_{i+1}⁶ − r_i⁶)/(r_{i+1}⁵ − r_i⁵)      [g/m²]
which reduces to the textbook single-layer adiabatic LWP = (5/9)ρ_w τ_bot r_top (Wood/Bennartz).
This is the *retrieval-native* LWP; BP2025/26 report adiabatic LWP for their retrievals, so it
is the like-for-like comparison. Truth LWP is the in-situ vertical integral ∫LWC dz.

**Q_ext=2 (geometric optics) — settled 2026-07-01.** Our whole OSSE is geometric-optics:
``vocals_io`` builds truth τ with QEXT_GEOM=2.0 and r_e² (not the Mie ⟨r²⟩), and the forward
carries that band-independent τ. So (2/3)∫r_e dτ is *self-consistent* with our τ, but it
overestimates true in-situ ∫LWC dz by ~+8–21% (median ~16%) — a geometric-optics + fixed-v_eff
(0.1) artifact present even for a PERFECT retrieval, NOT retrieval error. BP2026's Eq (9) avoids
it (full Mie + measured v_eff≈0.07–0.08) but their whole forward is real-Mie; we can't match it
by swapping Q_ext without rebuilding the OSSE, so we keep Q_ext=2 and DECOMPOSE the bias so
nothing is hidden:
    lwp_bias_z (vs in-situ ∫LWC dz) = lwp_skill (vs (2/3)∫r_e_truth dτ) + geom-optics artifact
Report the *skill* term as the BP-spirit comparison; report the *absolute* term (magnitude-
comparable to BP2026's 17.7%) with the artifact broken out. CAVEATS to state with any BP
comparison: (i) this geom-optics/fixed-v_eff artifact; (ii) we do NOT retrieve above-cloud water
vapor, which BP found essential; (iii) noiseless OSSE. Our number is a best-case, WV-free,
Q_ext=2 bound — directionally comparable to BP's real-world 17.7%, not a strict "we beat them."

Usage:
    python retrieval_analysis.py <sidecar_dir> [--out summary.json]
    python retrieval_analysis.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import retrieval_oe as roe  # noqa: E402  (best_fit_adiabatic — the maha adiabatic floor)

RE_MAX_DEFAULT = 20.0        # production optics-table ceiling (RF13 edge check)
THICK_TAU = 36.0             # §A3 thick-tail regime boundary (conditioning, not mis-fit)


# ────────────────────────────── LWP (piecewise adiabatic) ──────────────────────────────
def lwp_piecewise_adiabatic(r_e, s, tau_bot):
    """Exact LWP [g/m²] of the re⁵-linear profile through breakpoints ``(s, r_e)``.

    ``s`` normalized depth τ/τ_bot∈[0,1], ``r_e`` in µm. Per segment (r_e⁵ affine in s):
    ∫ r_e ds = (5/6)·Δs·(r₁⁶−r₀⁶)/(r₁⁵−r₀⁵) → LWP = (2/3)·τ_bot·Σ(...). The r₁→r₀ (constant
    r_e) limit is r̄·Δs.
    """
    r_e = np.asarray(r_e, float)
    s = np.asarray(s, float)
    o = np.argsort(s)
    r_e, s = r_e[o], s[o]
    integ = 0.0
    for j in range(len(s) - 1):
        ds = s[j + 1] - s[j]
        if ds <= 0:
            continue
        r0, r1 = r_e[j], r_e[j + 1]
        d5 = r1 ** 5 - r0 ** 5
        if abs(d5) < 1e-9 * max(r0 ** 5, 1.0):          # r_e ≈ const → ∫ = r̄ Δs
            integ += 0.5 * (r0 + r1) * ds
        else:
            integ += (5.0 / 6.0) * ds * (r1 ** 6 - r0 ** 6) / d5
    return float((2.0 / 3.0) * tau_bot * integ)


def lwp_trapz_tau(r_e_dense, tau_dense):
    """LWP [g/m²] = (2/3)∫ r_e dτ by trapezoid (dense-grid cross-check / truth-τ variant)."""
    return float((2.0 / 3.0) * np.trapezoid(np.asarray(r_e_dense, float),
                                            np.asarray(tau_dense, float)))


# ────────────────────────────── Mahalanobis ──────────────────────────────
def _phys_reblock_cov(S_hat_log, r_e_nodes):
    """Delta-method log→physical covariance of the r_e block: Ŝ_phys = J Ŝ_log Jᵀ,
    J = diag(∂r_e/∂log r_e) = diag(r_e). ``S_hat_log`` sliced to the first k rows/cols."""
    k = len(r_e_nodes)
    J = np.diag(np.asarray(r_e_nodes, float))
    return J @ np.asarray(S_hat_log, float)[:k, :k] @ J.T


def mahalanobis(delta, cov):
    """dᴹ² = δᵀ Σ⁻¹ δ (solve, not explicit inverse)."""
    delta = np.asarray(delta, float)
    return float(delta @ np.linalg.solve(np.asarray(cov, float), delta))


# ────────────────────────────── per-sidecar analysis ──────────────────────────────
def _coerce_str(v):
    try:
        return str(np.asarray(v).item())
    except Exception:
        return str(v)


def analyze_sidecar(npz_path, *, re_max=RE_MAX_DEFAULT):
    """Compute the full metric row for one sidecar npz. Returns a flat dict of scalars."""
    d = dict(np.load(npz_path, allow_pickle=True))
    k = int(d["k"])
    s_grid = np.asarray(d["s_grid"], float)
    tau_bot_ret = float(d["tau_bot_ret"])
    re_nodes = np.asarray(d["re_nodes_ret"], float)
    r_base_ret = float(d["r_base_ret"])
    x_hat_log = np.asarray(d["x_hat_log"], float)
    S_hat_log = np.asarray(d["S_hat_log"], float)

    warns = []
    # sanity: log-state encoding is log(r_e) — validates x_truth reconstruction below
    if not np.allclose(np.exp(x_hat_log[:k]), re_nodes, rtol=1e-4, atol=1e-4):
        warns.append("log-state≠log(r_e): x_truth encoding assumption may be wrong")

    # ---- LWP (piecewise adiabatic) + biases ----
    re_bp = np.concatenate([re_nodes, [r_base_ret]])       # nodes + base breakpoint (s=1)
    s_bp = np.concatenate([s_grid, [1.0]])
    lwp_ours = lwp_piecewise_adiabatic(re_bp, s_bp, tau_bot_ret)
    lwp_ours_trapz = lwp_trapz_tau(d["re_ours_dense"], np.asarray(d["s_dense"], float) * tau_bot_ret)
    if abs(lwp_ours - lwp_ours_trapz) > 0.02 * max(abs(lwp_ours), 1.0):
        warns.append(f"LWP piecewise vs dense disagree ({lwp_ours:.2f} vs {lwp_ours_trapz:.2f})")
    lwp_truth_z = float(abs(np.trapezoid(np.asarray(d["truth_lwc"], float),
                                         np.asarray(d["truth_altitude"], float))))
    lwp_truth_tau = lwp_trapz_tau(d["truth_re"], d["truth_tau"])
    lwp_bias_z = lwp_ours - lwp_truth_z                    # absolute: vs in-situ ∫LWC dz
    lwp_bias_tau = lwp_ours - lwp_truth_tau                # SKILL: vs (2/3)∫r_e_truth dτ (artifact cancels)
    lwp_geomopt_artifact = lwp_truth_tau - lwp_truth_z     # Q_ext=2/v_eff floor (perfect-retrieval bias vs in-situ)

    # ---- RMSE / ΔRMSE (dense, thickness-neutral) ----
    re_ours_d = np.asarray(d["re_ours_dense"], float)
    re_truth_d = np.asarray(d["re_truth_dense"], float)
    re_adia_d = np.asarray(d["re_adia_dense"], float)
    rmse_ours = float(np.sqrt(np.mean((re_ours_d - re_truth_d) ** 2)))
    rmse_adia = float(np.sqrt(np.mean((re_adia_d - re_truth_d) ** 2)))
    d_rmse = rmse_adia - rmse_ours                          # >0 ⇒ we beat the oracle floor

    # ---- Mahalanobis (full-state log; r_e-block physical + adiabatic floor) ----
    s_tr = np.asarray(d["truth_tau"], float) / float(d["truth_tau_bot"])
    o = np.argsort(s_tr)
    re_truth_nodes = np.interp(s_grid, s_tr[o], np.asarray(d["truth_re"], float)[o])
    x_truth_log = np.log(np.concatenate(
        [re_truth_nodes, [float(d["truth_r_base"]), float(d["truth_tau_bot"])]]))
    d2_full = mahalanobis(x_hat_log - x_truth_log, S_hat_log)          # k+2 dof, log space
    S_phys = _phys_reblock_cov(S_hat_log, re_nodes)
    d2_re = mahalanobis(re_nodes - re_truth_nodes, S_phys)             # r_e block, physical
    try:
        adia_m = roe.best_fit_adiabatic(s_grid, re_truth_nodes, tau_bot_ret,
                                        metric="maha", Sinv=np.linalg.inv(S_phys))
        d2_adia_min = float(adia_m["d2"])
    except Exception as e:                                             # singular block, etc.
        d2_adia_min = float("nan")
        warns.append(f"maha floor failed: {type(e).__name__}")

    flight = _coerce_str(d["flight"])
    top_node = float(re_nodes[0])
    row = dict(
        index=int(d["index"]), flight=flight, config=_coerce_str(d["config"]), k=k,
        tau_bot_ret=tau_bot_ret, tau_bot_truth=float(d["truth_tau_bot"]),
        converged=bool(d["converged"]), n_gn=int(d["n_gn"]),
        chi2_red=float(d["chi2_red"]), structural_misfit=bool(d["structural_misfit"]),
        # fidelity
        rmse_ours=rmse_ours, rmse_adia=rmse_adia, d_rmse=d_rmse,
        # LWP
        lwp_ours=lwp_ours, lwp_truth_z=lwp_truth_z, lwp_truth_tau=lwp_truth_tau,
        lwp_bias_z=lwp_bias_z, lwp_bias_tau=lwp_bias_tau,
        lwp_relbias_z=(100.0 * lwp_bias_z / lwp_truth_z) if lwp_truth_z else float("nan"),
        # both ÷ in-situ (BP denominator) so relbias_z == lwp_skill_pct + lwp_geomopt_artifact_pct exactly
        lwp_skill_pct=(100.0 * lwp_bias_tau / lwp_truth_z) if lwp_truth_z else float("nan"),
        lwp_geomopt_artifact_pct=(100.0 * lwp_geomopt_artifact / lwp_truth_z) if lwp_truth_z else float("nan"),
        # Mahalanobis
        d2_full=d2_full, d2_re=d2_re, d2_adia_min=d2_adia_min,
        # posterior IC
        dofs=float(d["dofs"]), sic=float(d["sic"]),
        dofs_profile=float(d["dofs_profile"]), dofs_r_base=float(d["dofs_r_base"]),
        dofs_tau_bot=float(d["dofs_tau_bot"]),
        # regime / edge flags
        top_node_re=top_node,
        re_max_edge=bool(top_node >= re_max - 1.0),        # E: within 1 µm of the ceiling
        thick_tail=bool(float(d["truth_tau_bot"]) >= THICK_TAU),
        warnings=warns,
    )
    return row


# ────────────────────────────── aggregation / report ──────────────────────────────
def _stats(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), min=float(a.min()), max=float(a.max()))


def summarize(rows):
    """Aggregate per config + overall; surface the review flags (A–F checklist)."""
    out = {"n_sidecars": len(rows), "by_config": {}, "flags": {}}
    for cfg in sorted({r["config"] for r in rows}):
        R = [r for r in rows if r["config"] == cfg]
        out["by_config"][cfg] = {
            "n": len(R),
            "converged_frac": float(np.mean([r["converged"] for r in R])),
            "rmse_ours": _stats([r["rmse_ours"] for r in R]),
            "d_rmse": _stats([r["d_rmse"] for r in R]),
            "lwp_bias_z": _stats([r["lwp_bias_z"] for r in R]),
            # LWP-% vs in-situ, reported in BP2026's two forms: signed mean = Fig-5 "bias"
            # (hyperspectral +6.8%); mean|%| = Table-2 "avg % difference" (hyperspectral 17.7%,
            # bispectral 45.2%, W&H 21.1%). Our target-to-beat is those two BP numbers.
            "lwp_relbias_z_pct": _stats([r["lwp_relbias_z"] for r in R]),
            "lwp_signed_bias_pct": float(np.nanmean([r["lwp_relbias_z"] for r in R])),
            "lwp_avg_abs_pct_diff": float(np.nanmean(np.abs([r["lwp_relbias_z"] for r in R]))),
            # decomposition (absolute vs in-situ = retrieval SKILL + geom-optics/v_eff artifact):
            # skill = BP-spirit comparison (artifact-free); artifact = perfect-retrieval Q_ext=2 floor
            "lwp_skill_signed_pct": float(np.nanmean([r["lwp_skill_pct"] for r in R])),
            "lwp_skill_avg_abs_pct": float(np.nanmean(np.abs([r["lwp_skill_pct"] for r in R]))),
            "lwp_geomopt_artifact_pct_mean": float(np.nanmean([r["lwp_geomopt_artifact_pct"] for r in R])),
            "d2_re": _stats([r["d2_re"] for r in R]),
            "d2_adia_min": _stats([r["d2_adia_min"] for r in R]),
            "dofs": _stats([r["dofs"] for r in R]),
            "sic": _stats([r["sic"] for r in R]),
        }
    out["flags"] = {
        "non_converged": [(r["index"], r["config"]) for r in rows if not r["converged"]],
        "structural_misfit": [(r["index"], r["config"]) for r in rows if r["structural_misfit"]],
        "re_max_edge (E: RF13 etc.)": [(r["index"], r["flight"], round(r["top_node_re"], 1))
                                       for r in rows if r["re_max_edge"]],
        "thick_tail_tau>=%g" % THICK_TAU: sorted({r["index"] for r in rows if r["thick_tail"]}),
        "analysis_warnings": [(r["index"], r["config"], r["warnings"]) for r in rows if r["warnings"]],
    }
    return out


def _print_table(rows):
    hdr = (f"{'idx':>4} {'flt':>5} {'c':>1} {'τ_bot':>6} {'conv':>4} {'RMSE':>6} "
           f"{'ΔRMSE':>6} {'LWPbias':>8} {'rel%':>6} {'d²_re':>7} {'d²adia':>7} {'DOFS':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["index"], r["config"])):
        print(f"{r['index']:>4} {r['flight']:>5} {r['config']:>1} {r['tau_bot_truth']:>6.1f} "
              f"{str(r['converged']):>4} {r['rmse_ours']:>6.2f} {r['d_rmse']:>+6.2f} "
              f"{r['lwp_bias_z']:>+8.1f} {r['lwp_relbias_z']:>+6.1f} {r['d2_re']:>7.2f} "
              f"{r['d2_adia_min']:>7.2f} {r['dofs']:>5.2f}")


def run(sidecar_dir, out_json=None, re_max=RE_MAX_DEFAULT):
    paths = sorted(glob.glob(os.path.join(sidecar_dir, "*_[AB].npz")))
    if not paths:
        raise SystemExit(f"no *_A.npz / *_B.npz sidecars under {sidecar_dir}")
    rows, failed = [], []
    for p in paths:
        try:
            rows.append(analyze_sidecar(p, re_max=re_max))
        except Exception as e:                              # never let one bad file abort the pass
            failed.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
    _print_table(rows)
    summary = summarize(rows)
    summary["failed_sidecars"] = failed
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if out_json:
        Path(out_json).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {out_json}  ({len(rows)} rows, {len(failed)} failed)")
    return rows, summary


# ────────────────────────────── self-test (no sidecars needed) ──────────────────────────────
def _selftest():
    ok = True

    # 1. single-layer adiabatic LWP → textbook (5/9) τ_bot r_top  (r_base→0)
    r_top, tb = 10.0, 20.0
    lwp2 = lwp_piecewise_adiabatic([r_top, 1e-3], [0.0, 1.0], tb)          # 2 breakpoints = exact adiabat
    exp = (5.0 / 9.0) * tb * r_top
    print(f"[1] adiabatic LWP piecewise={lwp2:.3f}  expected (5/9)τ_bot r_top={exp:.3f}")
    ok &= abs(lwp2 - exp) < 1e-2

    # 2. piecewise == dense trapezoid for the SAME re5-linear curve
    s = np.linspace(0, 1, 400)
    re = (1e-3 ** 5 + (r_top ** 5 - 1e-3 ** 5) * (1 - s)) ** 0.2
    lwp_dense = lwp_trapz_tau(re, s * tb)
    print(f"[2] piecewise={lwp2:.3f}  dense-trapz={lwp_dense:.3f}  (agree ⇒ formula ✓)")
    ok &= abs(lwp2 - lwp_dense) < 5e-2

    # 3. constant-r_e limit: LWP = (2/3) r τ_bot
    lwpc = lwp_piecewise_adiabatic([8.0, 8.0], [0.0, 1.0], 15.0)
    print(f"[3] constant r_e=8 LWP={lwpc:.3f}  expected (2/3)*8*15={(2/3)*8*15:.3f}")
    ok &= abs(lwpc - (2 / 3) * 8 * 15) < 1e-6

    # 4. Mahalanobis with identity cov = squared Euclidean
    delta = np.array([1.0, -2.0, 0.5])
    print(f"[4] maha(I)={mahalanobis(delta, np.eye(3)):.3f}  expected ‖δ‖²={float(delta@delta):.3f}")
    ok &= abs(mahalanobis(delta, np.eye(3)) - float(delta @ delta)) < 1e-9

    # 5. end-to-end on a synthetic sidecar (exercises loader + every metric path)
    import tempfile
    k = 4
    s_grid = np.linspace(0.0, 0.75, k)
    re_nodes = np.array([11.0, 10.0, 8.5, 7.0])
    tb2, r_base = 12.0, 6.0
    s_dense = np.linspace(0, 1, 200)
    re_bp = np.concatenate([re_nodes, [r_base]]); s_bp = np.concatenate([s_grid, [1.0]])
    re_ours_dense = np.interp(s_dense, s_bp, re_bp ** 5) ** 0.2          # re5-linear dense
    truth_tau = np.linspace(0, tb2, 60)
    truth_re = np.interp(truth_tau / tb2, s_bp, re_bp) + 0.3             # slightly off truth
    S_hat_log = np.diag(np.r_[np.full(k, 0.01), 0.04, 1e-4])            # log-space cov
    sc = dict(index=13, flight="RF13", config="A", k=k, s_grid=s_grid,
              tau_bot_ret=tb2, truth_tau_bot=tb2, r_base_ret=r_base,
              re_nodes_ret=re_nodes, x_hat_log=np.log(np.r_[re_nodes, r_base, tb2]),
              S_hat_log=S_hat_log, s_dense=s_dense, re_ours_dense=re_ours_dense,
              re_truth_dense=np.interp(s_dense, truth_tau / tb2, truth_re),
              re_adia_dense=re_ours_dense.copy(), truth_tau=truth_tau, truth_re=truth_re,
              truth_lwc=np.linspace(0.35, 0.0, 60), truth_altitude=np.linspace(900, 1200, 60),
              truth_r_base=r_base, truth_r_top=11.0, converged=True, n_gn=6,
              chi2_red=0.9, structural_misfit=False, dofs=2.7, sic=9.0,
              dofs_profile=1.6, dofs_r_base=0.1, dofs_tau_bot=1.0)
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "prof13_A.npz"), **sc)
        row = analyze_sidecar(os.path.join(td, "prof13_A.npz"))
    print(f"[5] end-to-end row: LWP_ours={row['lwp_ours']:.1f} bias_z={row['lwp_bias_z']:+.1f} "
          f"d²_re={row['d2_re']:.2f} d²adia={row['d2_adia_min']:.2f} "
          f"RF13-edge={row['re_max_edge']} warn={row['warnings']}")
    ok &= np.isfinite(row["lwp_ours"]) and np.isfinite(row["d2_re"]) and not row["warnings"]
    ok &= row["re_max_edge"] is False        # top node 11 µm, ceiling 20 → not an edge case

    print("\nSELFTEST:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar_dir", nargs="?", help="directory of *_{A,B}.npz sidecars")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--re-max", type=float, default=RE_MAX_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.sidecar_dir:
        ap.error("sidecar_dir required (or --selftest)")
    run(a.sidecar_dir, out_json=a.out, re_max=a.re_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
