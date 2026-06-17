"""Full joint OSSE retrievals: the headline result + the SO2a / SO2b / SO1 probes.

Runs end-to-end Gauss-Newton joint retrievals (state = [r_e nodes, r_base,
tau_bot]) on the thin (RF11) and thick (RF03) VOCALS truths, with leak-free
priors, and reports retrieved-vs-truth for the profile AND the two newly-unknown
anchors. One driver covers several objectives by varying a few knobs:

  * PO   — headline broad-prior joint retrieval (thin + thick); retrieved tau_bot
           / r_base vs truth; DOFS decomposition.
  * SO2a — re_class in {re5-linear, linear}: a MODEL COMPARISON on the same data
           (fit chi^2 / profile RMSE / DOFS), the clean test of the interpolation
           lever (OUTSTANDING B'). Also logs the QRCP node placement per class.
  * SO2b — n_outer in {1, 2}: does lagged re-meshing help or destabilise node
           placement (OUTSTANDING G "re-mesh instability")?
  * SO1  — auto_k_active (filter + dofs estimators) is reported from the pool
           Jacobian at the first guess (no extra compile), next to the fixed
           k_active actually used.

Incremental JSON -> docs/joint_osse_results.json. Select variants by name on the
CLI, else all run.  /tmp/jaxve/bin/python -u tests/supplementary/joint_osse_retrieval.py [name ...]
"""
import json
import sys
import time
from math import pi
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

import numpy as np

import vocals_io as vio
from miejax_lite import build_re_table, mie_legendre_precompute, select_channel
import retrieval_oe as roe

DD = ("/home/jovyan/cloud_profile_retrieval/"
      "multispectral-retrieval-using-MODIS/VOCALS_REx_data")
OUT = _root / "docs" / "joint_osse_results.json"

NQuad, NLeg_all, NFourier, v_eff = 16, 128, 8, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50])
view_phi = np.array([pi, pi, pi])
ALBEDO = 0.06
BROAD = dict(r_top_prior=10.0, r_base_prior=12.0, sigma_top=5.0, sigma_base=8.0)

# label, target_tau, restrict, bands, k_active, re_class, n_outer
VARIANTS = [
    ("thin_re5_n2",    1.0, None, [1.24, 2.13], 4, "re5-linear", 2),  # PO headline
    ("thin_re5_n1",    1.0, None, [1.24, 2.13], 4, "re5-linear", 1),  # SO2b
    ("thin_linear_n1", 1.0, None, [1.24, 2.13], 4, "linear",     1),  # SO2a
    ("thick_re5_n1",  23.3, "RF03", [1.24, 1.64, 2.13], 5, "re5-linear", 1),  # PO
    ("thick_linear_n1", 23.3, "RF03", [1.24, 1.64, 2.13], 5, "linear",  1),  # SO2a
]
_precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


def _save(rec):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[rec["label"]] = rec
    OUT.write_text(json.dumps(data, indent=2, default=float))
    print(f"  saved -> {OUT.name} [{rec['label']}]", flush=True)


def run(label, target_tau, restrict, bands, k_active, re_class, n_outer):
    t0 = time.perf_counter()
    profs = vio.load_all_profiles(DD)
    pool = [p for p in profs if p.flight == restrict] if restrict else profs
    truth = vio.pick_profile(pool, target_tau)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    print(f"\n=== {label}: {truth.flight} tau_bot={truth.tau_bot:.2f} "
          f"r_top={truth.r_top:.2f} r_base={truth.r_base:.2f}; {re_class} "
          f"n_outer={n_outer} ===", flush=True)

    opt_bands = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff,
                                               _precomp, n_radii=600), i)
                 for i in range(len(bands))]
    sig_tau = 0.5 * clim["tau_bot_mean"]
    fwd = roe.RetrievalForward(
        opt_bands, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[ALBEDO]] * len(bands),
        NLeg_all=NLeg_all, NFourier=NFourier, re_class=re_class,
        retrieve_tau_bot=True, retrieve_r_base=True)

    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)

    # mode count + QRCP grid at the broad first guess
    tau_ref = np.linspace(0.0, clim["tau_bot_mean"], 5)[:-1]
    x_ref, _ = roe.make_joint_prior(tau_ref, tau_bot_prior=clim["tau_bot_mean"],
                                    sigma_tau_bot=sig_tau, **BROAD)
    roe.select_num_modes(fwd, x_ref, tau_ref, Se)
    tau_coarse = np.linspace(0.0, clim["tau_bot_mean"], 6)[:-1]
    x_fg, _ = roe.make_joint_prior(tau_coarse, tau_bot_prior=clim["tau_bot_mean"],
                                   sigma_tau_bot=sig_tau, **BROAD)
    tau_grid, _, info = roe.select_retrieval_grid(fwd, x_fg, tau_coarse, k_active)
    k = len(tau_grid)
    print(f"    QRCP grid({k}): {np.round(tau_grid,2)}; modes K={fwd.K_list}",
          flush=True)

    # SO1: what would auto_k_active pick? (pool Jacobian already computed)
    Sa_pool = roe.make_adiabatic_prior(info["tau_pool"], clim["tau_bot_mean"],
                                       clim["r_base_mean"], BROAD["r_top_prior"],
                                       sigma_top=BROAD["sigma_top"],
                                       sigma_base=BROAD["sigma_base"])[1]
    k_filter, if_f = roe.auto_k_active(info["K_pool"], Se, Sa_pool, method="filter")
    k_dofs, if_d = roe.auto_k_active(info["K_pool"], Se, Sa_pool, method="dofs",
                                     factor=1.5)
    print(f"    SO1 auto_k: filter->{k_filter} (Sum_f={if_f['sum_filter_factor']:.2f}, "
          f"n_data={if_f['n_data_dominated']}), dofs->{k_dofs} "
          f"(DOFS={if_d['dofs']:.2f}); used k={k}", flush=True)

    # joint prior on the grid + GN
    prior_builder = (lambda tn: roe.make_joint_prior(
        tn, tau_bot_prior=clim["tau_bot_mean"], sigma_tau_bot=sig_tau, **BROAD))
    x_a, Sa = prior_builder(tau_grid)
    t1 = time.perf_counter()
    res = roe.gauss_newton_oe(fwd, y, tau_grid, x_a, Sa, Se, n_iter=12, lm=1e-2,
                              xtol=2e-3, n_outer=n_outer, k_active=k_active,
                              prior_builder=prior_builder)
    post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
    dby = roe.dofs_by_component(post, len(res.tau_nodes), retrieve_r_base=True,
                                retrieve_tau_bot=True)
    print(f"    GN [{time.perf_counter()-t1:.0f}s] converged={res.converged} "
          f"cost {res.cost_history[0]:.2e}->{res.cost_history[-1]:.2e} "
          f"||y-F||={np.linalg.norm(res.y-res.Fx):.2e}", flush=True)

    kk = len(res.tau_nodes)
    re_ret = res.x[:kk]
    r_base_ret, tau_bot_ret = float(res.x[kk]), float(res.x[kk + 1])
    truth_at = np.interp(res.tau_nodes, truth.tau, truth.r_e)
    prof_rmse = float(np.sqrt(np.mean((re_ret - truth_at) ** 2)))
    print(f"    PROFILE  ret={np.round(re_ret,2)} truth={np.round(truth_at,2)} "
          f"RMSE={prof_rmse:.2f} um", flush=True)
    print(f"    tau_bot  ret={tau_bot_ret:.2f} truth={truth.tau_bot:.2f} "
          f"(prior {clim['tau_bot_mean']:.2f}+-{sig_tau:.2f}, post 1σ "
          f"{post.error[kk+1]:.2f})", flush=True)
    print(f"    r_base   ret={r_base_ret:.2f} truth={truth.r_base:.2f} "
          f"(prior {BROAD['r_base_prior']:.1f}, post 1σ {post.error[kk]:.2f})",
          flush=True)
    print(f"    DOFS={post.dofs:.2f} (prof {dby['profile']:.2f}/rbase "
          f"{dby['r_base']:.2f}/taub {dby['tau_bot']:.2f})", flush=True)

    rec = dict(
        label=label, flight=truth.flight, re_class=re_class, n_outer=n_outer,
        bands=bands, k_active_used=kk, k_filter=k_filter, k_dofs=k_dofs,
        so1=dict(filter=if_f, dofs=if_d),
        tau_grid=res.tau_nodes.tolist(),
        truth=dict(tau_bot=truth.tau_bot, r_top=truth.r_top, r_base=truth.r_base),
        clim={k_: clim[k_] for k_ in ("r_top_mean", "r_base_mean", "tau_bot_mean")},
        retrieved=dict(profile=re_ret.tolist(), r_base=r_base_ret,
                       tau_bot=tau_bot_ret),
        truth_on_grid=truth_at.tolist(), profile_rmse=prof_rmse,
        converged=bool(res.converged), cost_history=list(map(float, res.cost_history)),
        resid_norm=float(np.linalg.norm(res.y - res.Fx)),
        dofs=post.dofs, dofs_profile=dby["profile"], dofs_r_base=dby["r_base"],
        dofs_tau_bot=dby["tau_bot"], error=post.error.tolist(),
        tau_bot_post_sigma=float(post.error[kk + 1]),
        r_base_post_sigma=float(post.error[kk]),
        tau_bot_prior_sigma=sig_tau, runtime_s=time.perf_counter() - t0)
    _save(rec)
    return rec


if __name__ == "__main__":
    which = sys.argv[1:] or [v[0] for v in VARIANTS]
    for v in VARIANTS:
        if v[0] in which:
            try:
                run(*v)
            except Exception as e:
                import traceback
                print(f"!! {v[0]} FAILED: {e}", flush=True)
                traceback.print_exc()
    print("\nDONE.", flush=True)
