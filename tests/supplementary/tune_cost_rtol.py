"""Tune the BP2026 cost-stagnation threshold COST_RTOL — threshold-insensitivity study.

BP2026 (lines 205-213) stop Gauss–Newton when the data-misfit cost changes by < 3 %
between iterations. We do NOT adopt 3 % on faith: this script sweeps
COST_RTOL ∈ {5,3,1,0.3,0.1} % on benchmark thin/mid/thick VOCALS profiles and
compares each stopped retrieval to a TIGHT reference (cost_rtol off, n_iter=25,
xtol=1e-5 — fully converged). A good threshold is the **loosest one still on the
plateau**: where RMSE(r_e), DOFS and the retrieved profile are indistinguishable
from the tight reference (Δrmse, Δdofs, max profile dev all negligible), erring
tight (an over-loose threshold under-converges and *inflates* our RMSE, which would
flatter the adiabatic floor — the wrong direction for the headline).

Reuses the production pipeline (retrieval_worker.build_forward_and_obs) so the
tuned threshold is measured on EXACTLY the retrieval the HPC array runs. Config A
(LOO prior mean) only — the threshold is a numerical-convergence property,
prior-config-independent. Writes docs/cached_results/cost_rtol_sweep.json.

Run:  /srv/conda/envs/notebook/bin/python tests/supplementary/tune_cost_rtol.py [idx ...]
"""
import sys
import json
import time
import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1] / "src"))
import retrieval_oe as roe                                          # noqa: E402
import vocals_io as vio                                            # noqa: E402

# import the worker module by path (it is a script, not a package)
_spec = importlib.util.spec_from_file_location("rw", _here / "retrieval_worker.py")
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)

OUT = _here.parents[1] / "docs" / "cached_results" / "cost_rtol_sweep.json"
BENCH = [95, 105, 42]                                              # thin (RF11) / mid (RF12) / thick (RF06)
SWEEP = [0.05, 0.03, 0.01, 0.003, 0.001]                          # 5 % → 0.1 %
# plateau tolerances (a threshold is "on the plateau" if all three hold vs the tight ref)
TOL_RMSE, TOL_DOFS, TOL_PROF = 0.01, 0.02, 0.05                    # µm, DOFS, µm


def _profile_and_metrics(fwd, res, s_grid, truth):
    """Dense retrieved profile + RMSE(vs truth) + DOFS for one GN result."""
    tau_bot_ret = float(np.asarray(fwd._split_state(res.x, s_grid)[2]))
    re_ours = np.asarray(fwd.profile(res.x, s_grid, rw.S_DENSE * tau_bot_ret))
    s_truth = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(s_truth)
    re_truth = np.interp(rw.S_DENSE, s_truth[o], np.asarray(truth.r_e, float)[o])
    rmse = float(np.sqrt(np.mean((re_ours - re_truth) ** 2)))
    post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
    return re_ours, rmse, float(post.dofs), tau_bot_ret, len(res.cost_history)


def run_one(idx, profiles):
    truth = profiles[idx]
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    t0 = time.time()
    fwd, y, Se, s_grid, _pb_phys, pb_log = rw.build_forward_and_obs(truth, clim)
    x_a_log, Sa_log = roe.make_climatology_prior(s_grid, clim, log=True)
    common = dict(x0=x_a_log, lm=1e-2, max_n_outer=1, prior_builder=pb_log, warn=False)

    # TIGHT reference (effectively fully converged): a cost_rtol far below the sweep
    # (5e-4 ≪ min sweep 1e-3) so it stops on stagnation rather than burning all n_iter.
    ref = roe.gauss_newton_oe(fwd, y, s_grid, x_a_log, Sa_log, Se,
                              n_iter=20, xtol=1e-5, cost_rtol=5e-4, **common)
    re_ref, rmse_ref, dofs_ref, tb_ref, n_ref = _profile_and_metrics(fwd, ref, s_grid, truth)

    rows = []
    for ct in SWEEP:
        res = roe.gauss_newton_oe(fwd, y, s_grid, x_a_log, Sa_log, Se,
                                  n_iter=12, xtol=2e-3, cost_rtol=ct, **common)
        re_o, rmse, dofs, tb, n_gn = _profile_and_metrics(fwd, res, s_grid, truth)
        d_rmse, d_dofs = rmse - rmse_ref, dofs - dofs_ref
        d_prof = float(np.max(np.abs(re_o - re_ref)))
        on_plateau = (abs(d_rmse) < TOL_RMSE and abs(d_dofs) < TOL_DOFS and d_prof < TOL_PROF)
        rows.append(dict(cost_rtol=ct, rmse=round(rmse, 4), d_rmse=round(d_rmse, 4),
                         dofs=round(dofs, 3), d_dofs=round(d_dofs, 3),
                         d_prof=round(d_prof, 4), n_gn=n_gn, tau_bot=round(tb, 3),
                         converged=bool(res.converged), on_plateau=on_plateau))
    rec = dict(index=idx, flight=truth.flight, tau_bot=round(float(truth.tau_bot), 3),
               k=len(s_grid), ref=dict(rmse=round(rmse_ref, 4), dofs=round(dofs_ref, 3),
                                       tau_bot=round(tb_ref, 3), n_gn=n_ref),
               sweep=rows, runtime_s=round(time.time() - t0, 1))
    print(f"\n=== idx {idx} {truth.flight} τ_bot={truth.tau_bot:.2f}  (ref: RMSE={rmse_ref:.3f} "
          f"DOFS={dofs_ref:.2f} n={n_ref}) [{rec['runtime_s']:.0f}s] ===", flush=True)
    print(f"  {'cost_rtol':>9} {'n_gn':>4} {'RMSE':>7} {'ΔRMSE':>7} {'DOFS':>6} {'ΔDOFS':>6} "
          f"{'Δprof':>6} {'plateau':>7}", flush=True)
    for r in rows:
        print(f"  {r['cost_rtol']*100:>7.1f}% {r['n_gn']:>4} {r['rmse']:>7.3f} {r['d_rmse']:>+7.3f} "
              f"{r['dofs']:>6.2f} {r['d_dofs']:>+6.2f} {r['d_prof']:>6.3f} "
              f"{'YES' if r['on_plateau'] else 'no':>7}", flush=True)
    return rec


def main():
    which = [int(a) for a in sys.argv[1:]] or BENCH
    profiles = vio.load_all_profiles(rw.DATA)
    recs = [run_one(i, profiles) for i in which]

    # recommendation: loosest cost_rtol on the plateau for EVERY benchmark profile
    on_all = [ct for ct in SWEEP
              if all(any(r['cost_rtol'] == ct and r['on_plateau'] for r in rec['sweep'])
                     for rec in recs)]
    rec_ct = max(on_all) if on_all else min(SWEEP)
    summary = dict(benchmarks=recs, sweep=SWEEP, tol=dict(rmse=TOL_RMSE, dofs=TOL_DOFS, prof=TOL_PROF),
                   plateau_all_profiles=on_all, recommended_cost_rtol=rec_ct)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nPLATEAU (all benchmark profiles): {on_all}")
    print(f"RECOMMENDED cost_rtol = {rec_ct} (loosest on the joint plateau, erring tight)")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
