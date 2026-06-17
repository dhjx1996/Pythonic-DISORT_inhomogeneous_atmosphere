"""Can the joint retrieval CAPTURE a sub-adiabatic base on a THIN cloud?

A sub-adiabatic base (r_e drops toward cloud bottom — the re-evaporation
signature) is invisible to reflectance for THICK cloud (the base is shielded;
DESIGN §10c). For THIN cloud the base is partially visible, so this is the regime
where it might be recoverable. We test two real VOCALS thin profiles:

  * RF05 (τ≈2.9): the textbook NON-MONOTONIC shape — r_e rises adiabatically
    (5.6→8.9) then drops to 6.8 at base.
  * RF14 (τ≈2.5): a clean strong MONOTONIC sub-adiabatic decline (9.4→6.0).

The retrieval starts from the BROAD adiabatic prior (r_top=10 < r_base=12, i.e.
the *opposite* curvature) so any downturn in the result is data-driven. We report
whether the retrieved curve bends down near the base and how much of the true
downturn is recovered, and save a figure.

  /tmp/jaxve/bin/python -u tests/supplementary/subadiabatic_thin_retrieval.py
"""
import json
import sys
import time
from math import pi
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vocals_io as vio
import retrieval_oe as roe
from miejax_lite import build_re_table, mie_legendre_precompute, select_channel

DATA = ("/home/jovyan/cloud_profile_retrieval/"
        "multispectral-retrieval-using-MODIS/VOCALS_REx_data")
OUTPNG = _root / "docs" / "subadiabatic_thin_retrieval.png"
OUTJSON = _root / "docs" / "subadiabatic_thin_results.json"

NQuad, NLeg_all, NFourier, v_eff = 16, 128, 8, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50])
view_phi = np.array([pi, pi, pi])
ALBEDO = 0.06
BANDS = [1.24, 1.64, 2.13]                       # 3-band ladder for vertical leverage
BROAD = dict(r_top_prior=10.0, r_base_prior=12.0, sigma_top=5.0, sigma_base=8.0)
K_ACTIVE = 5
_precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
TARGETS = [("RF05", 2.88), ("RF14", 2.46)]


def run(flight, target_tau):
    t0 = time.perf_counter()
    profs = vio.load_all_profiles(DATA)
    truth = vio.pick_profile([p for p in profs if p.flight == flight], target_tau)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    print(f"\n=== {flight}: tau_bot={truth.tau_bot:.2f} r_top={truth.r_top:.2f} "
          f"peak={truth.r_e.max():.2f} r_base={truth.r_base:.2f} ===", flush=True)

    opt_bands = [select_channel(build_re_table(BANDS, 2.0, 25.0, 32, v_eff,
                                               _precomp, n_radii=600), i)
                 for i in range(len(BANDS))]
    sig_tau = 0.5 * clim["tau_bot_mean"]
    fwd = roe.RetrievalForward(
        opt_bands, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[ALBEDO]] * len(BANDS),
        NLeg_all=NLeg_all, NFourier=NFourier, re_class="re5-linear",
        retrieve_tau_bot=True, retrieve_r_base=True)

    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)

    s_ref = np.linspace(0.0, 1.0, 6)[:-1]
    x_ref, _ = roe.make_joint_prior(s_ref, tau_bot_prior=clim["tau_bot_mean"],
                                    sigma_tau_bot=sig_tau, **BROAD)
    roe.select_num_modes(fwd, x_ref, s_ref, Se)
    s_grid, _, _ = roe.select_retrieval_grid(fwd, x_ref, s_ref, K_ACTIVE)
    print(f"    QRCP grid in s: {np.round(s_grid, 3)}  modes K={fwd.K_list}", flush=True)

    # BROAD ADIABATIC prior (r_base 12 > r_top 10 -> wants the curve to RISE to base);
    # GN starts here (x0=None -> x_a), so a downturn in the result is data-driven.
    prior_builder = (lambda sn: roe.make_joint_prior(
        sn, tau_bot_prior=clim["tau_bot_mean"], sigma_tau_bot=sig_tau, **BROAD))
    x_a, Sa = prior_builder(s_grid)
    t1 = time.perf_counter()
    res = roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, n_iter=15, lm=1e-2,
                              xtol=2e-3, n_outer=1)
    post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
    dby = roe.dofs_by_component(post, len(res.tau_nodes), retrieve_r_base=True,
                                retrieve_tau_bot=True)
    kk = len(res.tau_nodes)
    s = np.asarray(res.tau_nodes)
    re_ret = np.asarray(res.x[:kk])
    r_base_ret, tau_bot_ret = float(res.x[kk]), float(res.x[kk + 1])
    truth_at = np.interp(s * tau_bot_ret, truth.tau, truth.r_e)
    prof_rmse = float(np.sqrt(np.mean((re_ret - truth_at) ** 2)))
    print(f"    GN [{time.perf_counter()-t1:.0f}s] cost {res.cost_history[0]:.2e}->"
          f"{res.cost_history[-1]:.2e} ||y-F||={np.linalg.norm(res.y-res.Fx):.2e}",
          flush=True)

    # --- "downturn captured?" diagnostics, on a dense profile -----------------
    # Compare the upper-cloud max of the RETRIEVED curve to the retrieved base:
    # a negative (base - peak) => the retrieved curve bends DOWN (sub-adiabatic).
    s_dense = np.linspace(0.0, 1.0, 200)
    re_dense_ret = fwd.profile(res.x, s, s_dense * tau_bot_ret)
    re_dense_pr = fwd.profile(res.x_a, s, s_dense * tau_bot_ret)
    re_dense_tru = np.interp(s_dense * tau_bot_ret, truth.tau, truth.r_e)
    drop_ret = float(re_dense_ret.max() - re_dense_ret[-1])    # retrieved downturn
    drop_pr = float(re_dense_pr.max() - re_dense_pr[-1])       # prior downturn (~ -ve, rises)
    drop_tru = float(re_dense_tru.max() - re_dense_tru[-1])    # true downturn
    captured = drop_ret / drop_tru if drop_tru > 1e-9 else float("nan")
    print(f"    tau_bot ret {tau_bot_ret:.2f} (truth {truth.tau_bot:.2f}, prior "
          f"{clim['tau_bot_mean']:.1f})", flush=True)
    print(f"    r_base  ret {r_base_ret:.2f} (truth {truth.r_base:.2f}, prior 12.0, "
          f"post 1σ {post.error[kk]:.2f}); DOFS r_base={dby['r_base']:.2f}", flush=True)
    print(f"    PROFILE ret={np.round(re_ret,2)} truth={np.round(truth_at,2)} "
          f"RMSE={prof_rmse:.2f}", flush=True)
    print(f"    DOWNTURN (peak-minus-base): truth {drop_tru:+.2f}  retrieved "
          f"{drop_ret:+.2f}  prior {drop_pr:+.2f}  ->  captured {captured*100:.0f}% "
          f"of the true drop", flush=True)
    verdict = ("CAPTURED" if drop_ret > 0.3 and captured > 0.3 else
               "partial" if drop_ret > 0.1 else "MISSED (prior won)")
    print(f"    VERDICT: {verdict}", flush=True)

    rec = dict(flight=flight, tau_bot_truth=truth.tau_bot, tau_bot_ret=tau_bot_ret,
               r_top_truth=truth.r_top, r_base_truth=truth.r_base,
               r_base_ret=r_base_ret, r_base_post_sigma=float(post.error[kk]),
               dofs=post.dofs, dofs_r_base=dby["r_base"], dofs_profile=dby["profile"],
               s_grid=s.tolist(), re_ret=re_ret.tolist(), truth_at=truth_at.tolist(),
               profile_rmse=prof_rmse, drop_truth=drop_tru, drop_ret=drop_ret,
               drop_prior=drop_pr, captured_frac=captured, verdict=verdict,
               runtime_s=time.perf_counter() - t0)
    plot = dict(truth_tau=np.asarray(truth.tau), truth_re=np.asarray(truth.r_e),
                s_dense=s_dense, re_dense_ret=np.asarray(re_dense_ret),
                re_dense_pr=np.asarray(re_dense_pr), tau_bot_ret=tau_bot_ret,
                tau_bot_pr=float(clim["tau_bot_mean"]), s=s, re_ret=re_ret,
                re_err=post.error[:kk], r_base_ret=r_base_ret,
                r_base_err=float(post.error[kk]), truth=truth, rec=rec)
    return plot


def main():
    plots = [run(*t) for t in TARGETS]
    fig, axes = plt.subplots(1, len(plots), figsize=(6.4 * len(plots), 4.6))
    if len(plots) == 1:
        axes = [axes]
    for ax, P in zip(axes, plots):
        rec = P["rec"]
        ax.plot(P["truth_re"], P["truth_tau"], "-", color="k", alpha=0.5, lw=1.6,
                label="truth (in situ)")
        ax.plot(P["re_dense_pr"], P["s_dense"] * P["tau_bot_pr"], "--", color="C1",
                alpha=0.7, label=fr"adiabatic prior ($\tau_b$={P['tau_bot_pr']:.1f})")
        ax.plot(P["re_dense_ret"], P["s_dense"] * P["tau_bot_ret"], "-", color="C0",
                lw=2.2, label=fr"retrieved ($\tau_b$={P['tau_bot_ret']:.2f})")
        ax.errorbar(P["re_ret"], P["s"] * P["tau_bot_ret"], xerr=P["re_err"],
                    fmt="o", color="C0", capsize=3, ms=5)
        ax.errorbar([P["r_base_ret"]], [P["tau_bot_ret"]], xerr=[P["r_base_err"]],
                    fmt="s", color="C0", capsize=4, label=r"retrieved base $\pm1\sigma$")
        ax.plot([P["truth"].r_base], [P["truth"].tau_bot], "X", color="red", ms=12,
                zorder=5, label="truth base")
        ax.set_ylim(1.15 * max(P["tau_bot_ret"], P["truth"].tau_bot), 0)
        ax.set_xlabel(r"$r_e\ [\mu m]$"); ax.set_ylabel(r"$\tau$")
        ax.set_title(f"{rec['flight']}: downturn captured {rec['captured_frac']*100:.0f}% "
                     f"({rec['verdict']})\nr_base {rec['r_base_truth']:.1f}(true) "
                     f"<- {rec['r_base_ret']:.1f}(ret) <- 12.0(prior)", fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle("Thin sub-adiabatic base: can reflectance recover the "
                 "re-evaporation downturn?", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPNG, dpi=110)
    print(f"\nsaved figure -> {OUTPNG}")
    OUTJSON.write_text(json.dumps([P["rec"] for P in plots], indent=2, default=float))
    print(f"saved results -> {OUTJSON}")


if __name__ == "__main__":
    main()
