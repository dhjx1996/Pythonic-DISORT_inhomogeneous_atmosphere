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
sys.path.insert(0, str(_here))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import osse_config as oc                                            # noqa: E402
import jax.numpy as _jnp                                            # noqa: E402

# Accuracy tiers: the radiance CACHE (truth) is high-accuracy (float64, tol*); the
# RETRIEVAL here may run at OPERATIONAL precision/tol (PYDISORT_RICCATI_JAX_X64=0/1,
# SOLVER_TOL) — a cheaper, KNOWN, accepted bias (the operationally-realistic setup).
# float32 viability is being measured (probe #3): higher error is fine, de-stabilization
# (NaN / max_steps / non-convergence) is not. The truth tier always runs float64/tol*.
_PREC = "float64" if _jnp.result_type(float) == _jnp.float64 else "float32"

# ---------------------------------------------------------------------------
# Observing system / numerics — SINGLE SOURCE OF TRUTH = osse_config (the radiance
# cache, the IC run, and this worker MUST share it). This replaces the old standalone
# block whose NLeg_all=128 was the pre-TMS-fix value (garbage short bands).
# ---------------------------------------------------------------------------
DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_10band_nleg1536_re20.npz'))
RADIANCE_CACHE = Path(os.environ.get('RADIANCE_CACHE', _here.parents[2]
                                     / 'rad_bundle' / 'osse_radiances_125.npz'))
SOLVER_TOL = oc.SOLVER_TOL                                         # operational ODE tol (env SOLVER_TOL)
MODE_MAP = os.environ.get('MODE_MAP', 'scan')                      # 'vmap' = GPU bands×modes
COST_RTOL = float(os.environ.get('COST_RTOL', '0.01'))            # BP crit-1 (tuned); chi2_floor INACTIVE
VERBOSE = os.environ.get('FR_VERBOSE', '1') not in ('0', '', 'false')
NQ, N_PHYS, NB = oc.NQUAD, oc.N_PHYS, oc.NB                        # 48, 24, 10
BANDS, ALBEDO = oc.BANDS, oc.ALBEDO
NOISE = nm.oci_swir()                                             # OCI 2 % calibration-relative + 1e-3 floor
VIEW_MU, VIEW_PHI = oc.VIEW_MU, oc.VIEW_PHI                        # the irregular 24-view operational fan
S_REF_MODES = np.linspace(0.0, 1.0, 5)[:-1]                       # coarse grid for mode selection
S_COARSE = np.linspace(0.0, 1.0, 6)[:-1]                          # first-guess pool grid for QRCP
S_DENSE = np.linspace(0.0, 1.0, 50)                              # thickness-neutral RMSE / LWP grid
TAU_BOT_OK = (0.3, 100.0)                                         # degenerate-profile guard


def build_forward_and_obs(truth, clim, index, *, optics_cache=OPTICS_CACHE):
    """Build the log-space OPERATIONAL forward (precision via env, tol=SOLVER_TOL,
    mode_map=MODE_MAP), LOAD the high-accuracy radiance observation from the cache (the
    truth tier — NOT regenerated here), build Se, and select the azimuthal mode count +
    QRCP retrieval grid.

    Two-phase grid selection: first select at the LOO-prior ``tau_bot_mean``, run a cheap
    τ_bot pre-retrieval (:func:`retrieval_oe.retrieve_tau_bot`) on that grid to get a
    per-profile estimate, then re-select at the updated ``tau_bot``. The pre-retrieval
    pins r_e nodes tight and uses all bands — conservative-band (ω=1) rows dominate the
    τ_bot signal through the prior weighting, achieving the same physical effect as a
    VIS-only subset (``osse_config.VIS_BANDS`` is the documented physical motivation)
    while reusing the already-compiled full forward (zero extra JIT cost).

    Returns ``(fwd, y, Se, s_grid, pb_phys, pb_log, truth_tol, tau_bot_pre)``
    — shared by both prior configs (one compiled forward). ``tau_bot_pre`` is the
    pre-retrieved τ_bot (physical float) logged in the monitoring record."""
    _t = time.time()
    opt = oc.load_optics(optics_cache)                               # canonical NLeg_all=1536 table
    fwd = oc.build_forward(
        opt, tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
        views="retrieval", state_space="log", jac_mode="fwd",
        tol=SOLVER_TOL, mode_map=MODE_MAP)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] optics + RetrievalForward "
              f"({_PREC}, tol={SOLVER_TOL:.0e}, mode_map={MODE_MAP}) ready", flush=True)

    # OBSERVATION — signature-gated, cross-checked against truth (rigor over results).
    _t = time.time()
    rec = oc.load_radiance(RADIANCE_CACHE, index)
    if abs(float(rec["tau_bot"]) - float(truth.tau_bot)) > 1e-6:
        raise ValueError(f"cache/truth mismatch idx {index}: cache tau_bot="
                         f"{float(rec['tau_bot']):.4f} != VOCALS {float(truth.tau_bot):.4f}")
    truth_tol = rec.get("tol")                                       # the cache's accuracy tag
    _exp = os.environ.get("RADIANCE_TOL")                            # expected truth tol (optional gate)
    if _exp is not None and truth_tol is not None \
            and abs(truth_tol - float(_exp)) > 1e-12:
        raise ValueError(f"radiance cache tol {truth_tol} != expected RADIANCE_TOL {_exp} "
                         f"— wrong-accuracy cache; refusing (rigor over results).")
    y = oc.select_retrieval_views(rec["y"])
    Se = roe.make_Se(fwd, y, NOISE)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] loaded radiance cache "
              f"({RADIANCE_CACHE.name}, truth tol={truth_tol}) -> y[{y.size}]; "
              f"retrieval forward tol={SOLVER_TOL:.0e} ({_PREC})", flush=True)

    # MODE COUNT + INITIAL GRID at the climatology first guess.
    _t = time.time()
    x_ref = fwd._encode_state(roe.make_climatology_prior(S_REF_MODES, clim)[0])
    roe.select_num_modes(fwd, x_ref, S_REF_MODES, Se)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] select_num_modes -> K={fwd.K_list}", flush=True)
    _t = time.time()
    x_fg = fwd._encode_state(roe.make_climatology_prior(S_COARSE, clim)[0])
    pb_phys0 = lambda sn: roe.make_climatology_prior(sn, clim)      # clim-prior builder for initial grid
    s_grid_init, _, _ = roe.select_retrieval_grid(
        fwd, x_fg, S_COARSE, k_active=None, Se=Se, prior_builder=pb_phys0)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] initial select_retrieval_grid "
              f"-> k={len(s_grid_init)}", flush=True)

    # τ_BOT PRE-RETRIEVAL on the initial grid — reuses the compiled full forward (zero
    # extra JIT cost). r_e nodes pinned tight; conservative-band rows dominate τ_bot.
    _t = time.time()
    _clim_tau_prior = clim["tau_bot_mean"]
    tau_bot_pre, sigma_tau_pre = roe.retrieve_tau_bot(
        fwd, y, Se, clim, s_grid_init)
    clim = dict(clim, tau_bot_mean=tau_bot_pre, tau_bot_std=sigma_tau_pre)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] τ_bot pre-retrieval -> "
              f"{tau_bot_pre:.2f} ± {sigma_tau_pre:.2f} "
              f"(truth={truth.tau_bot:.2f}, clim_prior={_clim_tau_prior:.2f})", flush=True)

    # FINAL GRID at the per-profile τ_bot anchor. Prior builders close over updated clim.
    pb_phys = lambda sn: roe.make_climatology_prior(sn, clim)
    pb_log = lambda sn: roe.make_climatology_prior(sn, clim, log=True)
    _t = time.time()
    x_fg2 = fwd._encode_state(roe.make_climatology_prior(S_COARSE, clim)[0])
    s_grid, _, _ = roe.select_retrieval_grid(
        fwd, x_fg2, S_COARSE, k_active=None, Se=Se, prior_builder=pb_phys)
    if VERBOSE:
        print(f"    [build +{time.time()-_t:.0f}s] final select_retrieval_grid "
              f"(at tau_bot_pre={tau_bot_pre:.2f}) -> k={len(s_grid)}", flush=True)
    return fwd, y, Se, s_grid, pb_phys, pb_log, truth_tol, tau_bot_pre


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
               state_space='log', precision=_PREC, tol=SOLVER_TOL, mode_map=MODE_MAP,
               radiance_cache=RADIANCE_CACHE.name)
    try:
        if not (TAU_BOT_OK[0] <= float(truth.tau_bot) <= TAU_BOT_OK[1]) \
                or len(np.asarray(truth.tau)) < 5:
            raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f}, npts={len(truth.tau)})")
        rec["tau_bot"] = float(truth.tau_bot)
        clim = vio.vocals_climatology(profiles, exclude_flight=flight)

        t0 = time.time()
        fwd, y, Se, s_grid, pb_phys, pb_log, truth_tol, tau_bot_pre = \
            build_forward_and_obs(truth, clim, index)
        rec["radiance_tol"] = truth_tol
        rec["tau_bot_pre"] = round(tau_bot_pre, 3)
        print(f"[{index}] {flight} tau={truth.tau_bot:.1f}: built fwd + selected "
              f"grid({len(s_grid)}) in s={np.round(s_grid,3)}, K={fwd.K_list} "
              f"tau_bot_pre={tau_bot_pre:.2f} [{time.time()-t0:.0f}s]", flush=True)

        # shared LOG climatology prior on the selected grid (Sa_log shared by A & B)
        x_a_clim_log, Sa_log = roe.make_climatology_prior(s_grid, clim, log=True)

        # Persist each config's artifacts the moment that config finishes, so a
        # later config-B wall/crash cannot erase an already-converged config A.
        def _persist(tag, sc, mon):
            np.savez(f"{out_prefix}_{tag}.npz", **sc)
            Path(f"{out_prefix}_{tag}.json").write_text(
                json.dumps({kk: vv for kk, vv in mon.items()}))

        # config A — LOO prior mean is x_a and x0
        sc_A, mon_A = retrieve_one(fwd, y, Se, s_grid, x_a_clim_log, x_a_clim_log,
                                   Sa_log, truth, pb_log, index=index, config="A")
        _persist("A", sc_A, mon_A)
        # config B — one climatology realization (τ_bot SAMPLED) is x_a and x0; Sa shared
        draw, info = roe.draw_climatology_realization(
            clim, s_grid, rng=np.random.default_rng(2000 + index), tau_bot=None)
        x_draw_log = fwd._encode_state(draw)
        sc_B, mon_B = retrieve_one(fwd, y, Se, s_grid, x_draw_log, x_draw_log,
                                   Sa_log, truth, pb_log, index=index, config="B")
        sc_B["draw_info"] = json.dumps({k_: (float(v) if not isinstance(v, str) else v)
                                        for k_, v in info.items()})
        _persist("B", sc_B, mon_B)

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
