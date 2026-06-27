"""Full r_e(τ) OE retrieval worker — one VOCALS profile BY INDEX, both prior configs.

Usage:  retrieval_worker.py <profile_index> <out_prefix>
        writes  <out_prefix>_A.npz / <out_prefix>_A.json   (config A: LOO prior mean)
                <out_prefix>_B.npz / <out_prefix>_B.json   (config B: LOO prior realization)
        and a combined  <out_prefix>.json  (slim monitoring record for both configs).

This is the CAPSTONE retrieval (DESIGN §16; the IC profiling is done). For one
in-situ truth profile it runs the joint Gauss–Newton optimal-estimation retrieval
of the full state ``x = [r_e(s_nodes), r_base, τ_bot]`` in **log space**, in two
prior configurations that share ONE compiled forward (only x_a/x0/Sa differ):

  A "LOO prior"  [HEADLINE]   — prior mean x_a = leave-one-flight-out climatology
                                median adiabatic anchor; first guess x0 = x_a.
  B "LOO prior realization"   — one draw_climatology_realization (τ_bot SAMPLED from
                                climatology) is BOTH x_a and x0; Sa = the same
                                climatology covariance. Tests where the regularizer
                                is centred.

Truth = the real VOCALS profile in both. Observation = NOISELESS OSSE
(y = F(x_truth), no noise realization added; DESIGN §10b/§12); Se (oci_swir 2 %
calibration-relative) enters only as the assumed weighting / posterior covariance.
Observing system = the §15 multi-angle × 10-band superset (principal-plane fan, 24
views = NQuad//2, μ0=0.9, NQuad=48).

Three §16 upgrades vs the pre-§15 retrievals (all in retrieval_oe):
  * state_space='log' (BP2026 §2.4) + a log-space climatology prior (to_log_prior);
  * BP2026 cost-stagnation convergence (cost_rtol; chi2_floor coded but INACTIVE);
  * the oracle best-fit-adiabatic floor (best_fit_adiabatic) — the RMSE lower bound
    under the adiabatic constraint, computed post-hoc here for monitoring.

RAW sidecar (the product; every metric is a post-hoc computation by
retrieval_analysis.py): the log-space Jacobian K, prior Sa, posterior S_hat, A,
DOFS/SIC, the retrieved state, the noise σ, y/Fx, the truth arrays (incl. lwc +
altitude for the z-resolved LWP_truth), and convenience dense profiles. Grid
selection is done ONCE at the climatology first guess (in PHYSICAL space, where the
noise-aware filter whitening is dimensionally correct and ≈ invariant to the log
reparam); both configs then retrieve on that fixed grid (max_n_outer=1, so the grid
never moves — a clean A-vs-B comparison — while a structural-misfit χ²>thr still
WARNS, flagged in the sidecar). Degenerate / failed profiles write {skipped:...}.
Index-addressed.

Env:  VOCALS_DATA   in-situ flight netCDFs (default = the jovyan path).
      ENSEMBLE_NQUAD (48 — the converged operating point; N = NQuad//2 streams).
      OPTICS_CACHE   cached miepython table (.npz; profile-INDEPENDENT, built once,
                     shared with the IC run when the signature matches).
      COST_RTOL      BP criterion-1 relative-misfit threshold (tuned; see
                     tune_cost_rtol.py / DESIGN §10h). Default 0.01.
"""
import os
import sys
import json
import time
import warnings
from math import pi
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_src = _here.parents[1] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import optics_table as ot                                          # noqa: E402
import jax.numpy as _jnp                                            # noqa: E402

# float64 is REQUIRED (set PYDISORT_RICCATI_JAX_X64=1): at float32 the dense in-situ
# truth forward and steep GN iterates hit the adaptive solver's max_steps on the
# NQuad=48 optics (DESIGN §15). Warn loudly rather than fail silently into a SKIP.
if _jnp.result_type(float) != _jnp.float64:
    print("WARNING: running in float32 — set PYDISORT_RICCATI_JAX_X64=1; the NQuad=48 "
          "retrieval is expected to hit max_steps at float32 (DESIGN §15).", flush=True)

# ---------------------------------------------------------------------------
# Fixed observing system / numerics (identical to the §15 IC "fullview" system)
# ---------------------------------------------------------------------------
DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQ = int(os.environ.get('ENSEMBLE_NQUAD', '48'))
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_10band.npz'))
COST_RTOL = float(os.environ.get('COST_RTOL', '0.01'))             # BP crit-1 (tuned); chi2_floor INACTIVE
VERBOSE = os.environ.get('FR_VERBOSE', '1') not in ('0', '', 'false')  # per-stage + per-GN-iter timing logs
N_PHYS = NQ // 2
mu0, NLeg_all, v_eff = 0.9, 128, 0.10
# 10-band instrument superset (ascending λ) — the §15 superset, verbatim.
BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7, 4.05]
NB = len(BANDS)
ALBEDO = 0.06                                                     # Lambertian sea-surface (BDRF)
NOISE = nm.oci_swir()                                             # OCI 2 % calibration-relative + 1e-3 floor
# Principal-plane fan: exactly NQuad//2 = 24 views, μ 0.95 → 0.25 (no under-sampling).
VIEW_MU, VIEW_PHI = np.linspace(0.95, 0.25, N_PHYS), np.full(N_PHYS, pi)
S_REF_MODES = np.linspace(0.0, 1.0, 5)[:-1]                       # coarse grid for mode selection
S_COARSE = np.linspace(0.0, 1.0, 6)[:-1]                          # first-guess pool grid for QRCP
S_DENSE = np.linspace(0.0, 1.0, 50)                              # thickness-neutral RMSE / LWP grid
TAU_BOT_OK = (0.3, 100.0)                                         # degenerate-profile guard


def build_forward_and_obs(truth, clim, *, optics_cache=OPTICS_CACHE):
    """Build the log-space forward, the noiseless observation + Se, select the
    azimuthal mode count and the QRCP retrieval grid (all at the climatology first
    guess). Returns ``(fwd, y, Se, s_grid, pb_phys, pb_log)`` — the pieces SHARED by
    both prior configs (one compiled forward)."""
    _t = time.time()
    re_table = ot.build_or_load_table(BANDS, 2.0, 25.0, 32, v_eff,
                                      cache_path=optics_cache, NLeg=NLeg_all, n_radii=600)
    opt = [ot.select_channel(re_table, i) for i in range(NB)]
    fwd = roe.RetrievalForward(
        opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
        tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],    # leak-free first-guess anchor
        view_mu=VIEW_MU, view_phi=VIEW_PHI, BDRF_bands=[[ALBEDO]] * NB,
        NLeg_all=NLeg_all, state_space='log', jac_mode='fwd',
        retrieve_tau_bot=True, retrieve_r_base=True)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] optics table + RetrievalForward ready", flush=True)

    # NOISELESS OSSE: y = F(x_truth); Se = assumed weighting only.
    _t = time.time()
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = roe.make_Se(fwd, y, NOISE)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] osse_observation (1st forward, compile-incl)", flush=True)

    # PHYSICAL prior builder drives the noise-aware grid filter (dimensionally correct,
    # and ≈ invariant to the log reparam since diag(r)·diag(σ_log) ≈ diag(σ_phys));
    # the LOG prior builder is the actual GN regulariser (and re-mesh rebuild).
    pb_phys = lambda sn: roe.make_climatology_prior(sn, clim)
    pb_log = lambda sn: roe.make_climatology_prior(sn, clim, log=True)

    # mode count + QRCP grid at the (encoded) climatology first guess
    _t = time.time()
    x_ref = fwd._encode_state(roe.make_climatology_prior(S_REF_MODES, clim)[0])
    roe.select_num_modes(fwd, x_ref, S_REF_MODES, Se)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] select_num_modes -> K={fwd.K_list}", flush=True)
    _t = time.time()
    x_fg = fwd._encode_state(roe.make_climatology_prior(S_COARSE, clim)[0])
    s_grid, _, _ = roe.select_retrieval_grid(
        fwd, x_fg, S_COARSE, k_active=None, Se=Se, prior_builder=pb_phys)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] select_retrieval_grid "
              f"(grid-selection Jacobian) -> k={len(s_grid)}", flush=True)
    return fwd, y, Se, s_grid, pb_phys, pb_log


def retrieve_one(fwd, y, Se, s_grid, x_a, x0, Sa, truth, pb_log, *, index,
                 cost_rtol=COST_RTOL, chi2_floor=None, config="A"):
    """Run ONE Gauss–Newton OE retrieval (a given prior/first-guess) on the fixed
    grid and assemble the raw sidecar dict + slim monitoring scalars. All inputs that
    differ between configs are x_a / x0 / Sa; everything else (fwd, y, Se, s_grid) is
    shared. No grid move (max_n_outer=1) — only a structural-misfit warning."""
    k = len(s_grid)
    if VERBOSE:
        print(f"  [{index}] config {config}: GN retrieve on k={k} grid ...", flush=True)
    _t = time.time()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        res = roe.gauss_newton_oe(
            fwd, y, s_grid, x_a, Sa, Se, x0=x0, n_iter=12, lm=1e-2, xtol=2e-3,
            cost_rtol=cost_rtol, chi2_floor=chi2_floor,
            max_n_outer=1, prior_builder=pb_log, remesh_if_chi2_red_gt=2.0, warn=True,
            verbose=VERBOSE)
    if VERBOSE:
        print(f"  [{index}] config {config}: GN done in {time.time()-_t:.0f}s "
              f"({len(res.cost_history)} accepted iters, converged={res.converged})", flush=True)
    structural_misfit = any(isinstance(w.message, roe.RemeshWarning) for w in wlog)

    post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)         # log-space Ŝ/A/DOFS/SIC
    dby = roe.dofs_by_component(post, k, retrieve_r_base=True, retrieve_tau_bot=True)

    # physical decode of the retrieved state
    r_nodes, r_base_ret, tau_bot_ret = (np.asarray(v, float) for v in
                                        fwd._split_state(res.x, s_grid))
    r_nodes = np.atleast_1d(r_nodes)
    r_base_ret, tau_bot_ret = float(r_base_ret), float(tau_bot_ret)

    # dense, thickness-neutral profiles (RT-free — pure interpolation)
    re_ours = fwd.profile(res.x, s_grid, S_DENSE * tau_bot_ret)     # retrieved r_e(s_dense)
    s_truth = np.asarray(truth.tau, float) / truth.tau_bot
    o = np.argsort(s_truth)
    re_truth_dense = np.interp(S_DENSE, s_truth[o], np.asarray(truth.r_e, float)[o])
    rmse_ours = float(np.sqrt(np.mean((re_ours - re_truth_dense) ** 2)))

    # oracle best-fit adiabatic floor (known τ_bot; r_top/r_base fit to truth)
    adia = roe.best_fit_adiabatic(S_DENSE, re_truth_dense, truth.tau_bot, metric='rmse')
    rmse_adia = adia['rmse']
    d_rmse = rmse_adia - rmse_ours                                  # >0 ⇒ we beat the floor

    # LWP (z-free optical-depth integral) + the Q_ext=2 consistency check
    tau_dense = S_DENSE * tau_bot_ret
    lwp_ours = float((2.0 / 3.0) * np.trapezoid(re_ours, tau_dense))
    lwp_adia = float((2.0 / 3.0) * np.trapezoid(np.asarray(adia['re_fit']), tau_dense))
    lwp_truth_z = float(abs(np.trapezoid(np.asarray(truth.lwc, float),
                                         np.asarray(truth.altitude, float))))
    lwp_truth_tau = float((2.0 / 3.0) * np.trapezoid(np.asarray(truth.r_e, float),
                                                     np.asarray(truth.tau, float)))

    chi2_red = float((res.y - res.Fx) @ np.linalg.inv(res.Se) @ (res.y - res.Fx)) / max(len(res.y), 1)

    sidecar = dict(
        index=int(index), flight=getattr(truth, 'flight', '?'), config=config,
        state_space='log', cost_rtol=float(cost_rtol),
        chi2_floor=(float(chi2_floor) if chi2_floor is not None else np.nan),
        bands=np.asarray(BANDS), view_mu=VIEW_MU, NQuad=NQ, n_view=N_PHYS,
        s_grid=np.asarray(s_grid), k=k,
        # --- raw OE outputs (LOG space; analysis un-chain-rules K_lin=K_log/r_e) ---
        x_hat_log=np.asarray(res.x), x_a_log=np.asarray(res.x_a),
        K_log=np.asarray(res.K), Sa_log=np.asarray(res.Sa),
        S_hat_log=np.asarray(post.S_hat), A_log=np.asarray(post.A),
        error_log=np.asarray(post.error), data_fraction=np.asarray(post.data_fraction),
        dofs=float(post.dofs), sic=float(post.sic),
        dofs_profile=float(dby['profile']), dofs_r_base=float(dby['r_base']),
        dofs_tau_bot=float(dby['tau_bot']), dofs_profile_nodes=np.asarray(dby['profile_nodes']),
        sigma=np.asarray(NOISE.sigma(y, n_bands=NB)), y=np.asarray(res.y), Fx=np.asarray(res.Fx),
        # --- physical retrieved + dense convenience profiles ---
        re_nodes_ret=r_nodes, r_base_ret=r_base_ret, tau_bot_ret=tau_bot_ret,
        s_dense=S_DENSE, re_ours_dense=np.asarray(re_ours), re_truth_dense=re_truth_dense,
        re_adia_dense=np.asarray(adia['re_fit']),
        adia_r_top=float(adia['r_top']), adia_r_base=float(adia['r_base']),
        # --- truth (incl. z-resolved fields for LWP_truth) ---
        truth_tau=np.asarray(truth.tau, float), truth_re=np.asarray(truth.r_e, float),
        truth_lwc=np.asarray(truth.lwc, float), truth_altitude=np.asarray(truth.altitude, float),
        truth_tau_bot=float(truth.tau_bot), truth_r_base=float(truth.r_base),
        truth_r_top=float(truth.r_top),
        # --- convergence / health ---
        converged=bool(res.converged), n_gn=len(res.cost_history),
        cost_history=np.asarray(res.cost_history, float), chi2_red=chi2_red,
        structural_misfit=bool(structural_misfit), K_list=np.asarray(fwd.K_list))

    mon = dict(config=config, converged=bool(res.converged), n_gn=len(res.cost_history),
               chi2_red=round(chi2_red, 4), structural_misfit=bool(structural_misfit),
               tau_bot_ret=round(tau_bot_ret, 3), tau_bot_truth=round(float(truth.tau_bot), 3),
               r_base_ret=round(r_base_ret, 3), r_top_ret=round(float(r_nodes[0]), 3),
               rmse_ours=round(rmse_ours, 4), rmse_adia=round(rmse_adia, 4),
               d_rmse=round(d_rmse, 4), dofs=round(float(post.dofs), 3),
               sic=round(float(post.sic), 3),
               dofs_split=dict(profile=round(float(dby['profile']), 3),
                               r_base=round(float(dby['r_base']), 3),
                               tau_bot=round(float(dby['tau_bot']), 3)),
               lwp_ours=round(lwp_ours, 2), lwp_adia=round(lwp_adia, 2),
               lwp_truth_z=round(lwp_truth_z, 2), lwp_truth_tau=round(lwp_truth_tau, 2))
    return sidecar, mon


def main():
    index = int(sys.argv[1])
    out_prefix = sys.argv[2]
    profiles = vio.load_all_profiles(DATA)
    truth = profiles[index]
    flight = getattr(truth, 'flight', '?')

    rec = dict(index=index, flight=flight, NQuad=NQ, cost_rtol=COST_RTOL,
               state_space='log')
    try:
        if not (TAU_BOT_OK[0] <= float(truth.tau_bot) <= TAU_BOT_OK[1]) \
                or len(np.asarray(truth.tau)) < 5:
            raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f}, npts={len(truth.tau)})")
        rec["tau_bot"] = float(truth.tau_bot)
        clim = vio.vocals_climatology(profiles, exclude_flight=flight)

        t0 = time.time()
        fwd, y, Se, s_grid, pb_phys, pb_log = build_forward_and_obs(truth, clim)
        print(f"[{index}] {flight} tau={truth.tau_bot:.1f}: built fwd + selected "
              f"grid({len(s_grid)}) in s={np.round(s_grid,3)}, K={fwd.K_list} "
              f"[{time.time()-t0:.0f}s]", flush=True)

        # shared LOG climatology prior on the selected grid (Sa_log shared by A & B)
        x_a_clim_log, Sa_log = roe.make_climatology_prior(s_grid, clim, log=True)

        # config A — LOO prior mean is x_a and x0
        sc_A, mon_A = retrieve_one(fwd, y, Se, s_grid, x_a_clim_log, x_a_clim_log,
                                   Sa_log, truth, pb_log, index=index, config="A")
        # config B — one climatology realization (τ_bot SAMPLED) is x_a and x0; Sa shared
        draw, info = roe.draw_climatology_realization(
            clim, s_grid, rng=np.random.default_rng(2000 + index), tau_bot=None)
        x_draw_log = fwd._encode_state(draw)
        sc_B, mon_B = retrieve_one(fwd, y, Se, s_grid, x_draw_log, x_draw_log,
                                   Sa_log, truth, pb_log, index=index, config="B")
        sc_B["draw_info"] = json.dumps({k_: (float(v) if not isinstance(v, str) else v)
                                        for k_, v in info.items()})

        for tag, sc in (("A", sc_A), ("B", sc_B)):
            np.savez(f"{out_prefix}_{tag}.npz", **sc)
            Path(f"{out_prefix}_{tag}.json").write_text(json.dumps(
                {kk: vv for kk, vv in (mon_A if tag == "A" else mon_B).items()}))

        rec.update(grid=np.round(s_grid, 4).tolist(), K_list=list(map(int, fwd.K_list)),
                   runtime_s=round(time.time() - t0, 1), A=mon_A, B=mon_B,
                   npz=[f"{Path(out_prefix).name}_A.npz", f"{Path(out_prefix).name}_B.npz"])
        print(f"[{index}] {flight} tau={truth.tau_bot:.1f} DONE [{time.time()-t0:.0f}s] | "
              f"A: dRMSE={mon_A['d_rmse']:+.3f} (ours {mon_A['rmse_ours']:.3f} vs adia "
              f"{mon_A['rmse_adia']:.3f}) DOFS={mon_A['dofs']:.2f} conv={mon_A['converged']} | "
              f"B: dRMSE={mon_B['d_rmse']:+.3f} DOFS={mon_B['dofs']:.2f} conv={mon_B['converged']}",
              flush=True)
    except Exception as e:                                          # noqa: BLE001
        rec["skipped"] = str(e)[:200]
        print(f"[{index}] {flight}: SKIPPED {rec['skipped']}", flush=True)

    Path(f"{out_prefix}.json").write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
