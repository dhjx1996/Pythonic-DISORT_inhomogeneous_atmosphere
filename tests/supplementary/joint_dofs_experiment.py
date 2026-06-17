"""PO information-content study: how does retrieving (τ_bot, r_base) jointly —
and the choice of prior — change the DOFS?

For each (cloud, band-set) config this builds ONE joint forward, selects a QRCP
retrieval grid AT THE TRUTH SCENE, evaluates the joint Jacobian
K = ∂y/∂[r_e nodes, r_base, τ_bot] at the truth state ONCE (the standard OSSE
information-content linearization — achievable info for this cloud; NOT a
retrieval leak, since only the linearization point uses truth while the priors
below stay leak-free), then reuses K for three prior configurations (the DOFS is
cheap host linear algebra once K exists):

  A. fixed-anchor baseline  — τ_bot, r_base KNOWN (the legacy retrieval): the
     r_e-node column block of K + the r_e prior block. DOFS_A.
  B. joint + broad prior     — weakly-informative (Option 2, the headline).
  C. joint + climatology     — leave-one-flight-out VOCALS prior (Option 1).

Reported per config: total DOFS, the per-component split (profile / r_base /
τ_bot via diag(A)), posterior 1σ on r_base and τ_bot, and the data-fraction of
each. This isolates (i) the cost of making the two anchors unknown (A vs B
profile-part) and (ii) the prior-dependence of DOFS (B vs C) — both PO
sub-questions — and, by varying ``bands``, re-assesses band selection.

Results are appended incrementally to docs/joint_dofs_results.json so partial
runs are not lost. Run:
    /tmp/jaxve/bin/python tests/supplementary/joint_dofs_experiment.py
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
OUT = _root / "docs" / "joint_dofs_results.json"

# shared geometry + numerics (notebook production settings)
NQuad, NLeg_all, NFourier, v_eff = 16, 128, 8, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50])      # ~25, 50, 60 deg view zenith
view_phi = np.array([pi, pi, pi])           # principal plane, backscatter
ALBEDO = 0.06

# broad (Option-2) prior hyper-parameters — generic marine-Sc, leak-free
BROAD = dict(r_top_prior=10.0, r_base_prior=12.0, sigma_top=5.0, sigma_base=8.0)

# configs: (label, target_tau, restrict_flight, bands, k_active)
CONFIGS = [
    ("thin_current", 1.0, None, [1.24, 2.13], 4),
    ("thin_conservative", 1.0, None, [0.86, 2.13], 4),
    ("thick_current", 23.3, "RF03", [1.24, 1.64, 2.13], 5),
    ("thick_conservative", 23.3, "RF03", [0.86, 1.64, 2.13], 5),
]

_precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


def truth_state(truth, s_nodes):
    """Joint state [r_e at normalized depths s_nodes, r_base, tau_bot] from truth.

    The grid is normalized depth s=τ/τ_bot, so r_e is sampled at absolute
    τ = s·τ_bot_truth.
    """
    re = np.interp(np.asarray(s_nodes) * truth.tau_bot, truth.tau, truth.r_e)
    return np.concatenate([re, [truth.r_base, truth.tau_bot]])


def _save(record):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[record["label"]] = record
    OUT.write_text(json.dumps(data, indent=2, default=float))
    print(f"  saved -> {OUT.name} [{record['label']}]")


def run_config(label, target_tau, restrict_flight, bands, k_active):
    t0 = time.perf_counter()
    profs = vio.load_all_profiles(DD)
    pool = [p for p in profs if p.flight == restrict_flight] if restrict_flight else profs
    truth = vio.pick_profile(pool, target_tau)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    print(f"\n=== {label}: {truth.flight} tau_bot={truth.tau_bot:.2f} "
          f"r_top={truth.r_top:.2f} r_base={truth.r_base:.2f}; bands={bands} ===")
    print(f"    LOO clim: r_top={clim['r_top_mean']:.1f}±{clim['r_top_std']:.1f} "
          f"r_base={clim['r_base_mean']:.1f}±{clim['r_base_std']:.1f} "
          f"tau_bot={clim['tau_bot_mean']:.1f}±{clim['tau_bot_std']:.1f}")

    opt_bands = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff,
                                               _precomp, n_radii=600), i)
                 for i in range(len(bands))]
    # The DOFS is the *information-content* diagnostic, so the Jacobian and the
    # QRCP grid are evaluated **at the truth scene** (standard OSSE practice: it
    # gives the achievable info content for this cloud, and the grid spans the true
    # [0, τ_bot]). This is NOT a retrieval leak — only the linearization point uses
    # truth; the PRIORS (Sa, below) stay leak-free (broad / LOO climatology). The
    # leak-free *retrieval* (first guess from climatology, GN finds τ_bot) is the
    # companion joint_osse_retrieval.py, which reports DOFS at the retrieved state.
    fwd = roe.RetrievalForward(
        opt_bands, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=truth.tau_bot, r_base=truth.r_base,
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[ALBEDO]] * len(bands),
        NLeg_all=NLeg_all, NFourier=NFourier,
        retrieve_tau_bot=True, retrieve_r_base=True)

    # synthetic measurement from truth (noiseless); 3% radiometric + floor
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    sigma_tau_broad = 0.5 * clim["tau_bot_mean"]

    # S_eps azimuthal-mode count at the truth scene (grids in normalized depth s)
    s_ref = np.linspace(0.0, 1.0, 5)[:-1]
    Kmodes = roe.select_num_modes(fwd, truth_state(truth, s_ref), s_ref, Se)
    print(f"    obs m={fwd.m}, S_eps modes K={Kmodes}, "
          f"setup {time.perf_counter()-t0:.0f}s")

    # QRCP retrieval grid at the truth scene (joint truth state), returned in s
    s_coarse = np.linspace(0.0, 1.0, 6)[:-1]
    t1 = time.perf_counter()
    s_grid, _, _ = roe.select_retrieval_grid(
        fwd, truth_state(truth, s_coarse), s_coarse, k_active)
    k = len(s_grid)
    print(f"    QRCP grid ({k}) in s: {np.round(s_grid,3)} "
          f"(τ≈{np.round(s_grid*truth.tau_bot,2)})  [{time.perf_counter()-t1:.0f}s]")

    # leak-free joint prior on the grid + the ONE Jacobian AT THE TRUTH STATE
    x_a_broad, Sa_broad = roe.make_joint_prior(
        s_grid, tau_bot_prior=clim["tau_bot_mean"],
        sigma_tau_bot=sigma_tau_broad, **BROAD)
    t2 = time.perf_counter()
    Kjac = np.asarray(fwd.jacobian(truth_state(truth, s_grid), s_grid))     # (m,k+2)
    print(f"    Jacobian {Kjac.shape} [{time.perf_counter()-t2:.0f}s]")

    # --- A: fixed-anchor baseline (drop r_base, τ_bot columns + r_e prior block)
    K_fix = Kjac[:, :k]
    Sa_fix = Sa_broad[:k, :k]
    post_fix = roe.posterior_diagnostics(K_fix, Sa_fix, Se)

    # --- B: joint + broad ; C: joint + climatology -------------------------
    post_broad = roe.posterior_diagnostics(Kjac, Sa_broad, Se)
    x_a_clim, Sa_clim = roe.make_climatology_prior(s_grid, clim)
    post_clim = roe.posterior_diagnostics(Kjac, Sa_clim, Se)

    def summary(post, Sa, joint):
        d = roe.dofs_by_component(post, k, retrieve_r_base=joint,
                                  retrieve_tau_bot=joint)
        out = {"dofs": post.dofs, "profile_dofs": d["profile"],
               "error": post.error.tolist(),
               "data_fraction": post.data_fraction.tolist()}
        if joint:
            out.update(r_base_dofs=d["r_base"], tau_bot_dofs=d["tau_bot"],
                       r_base_sigma=float(post.error[k]),
                       tau_bot_sigma=float(post.error[k + 1]),
                       r_base_prior_sigma=float(np.sqrt(Sa[k, k])),
                       tau_bot_prior_sigma=float(np.sqrt(Sa[k + 1, k + 1])))
        return out

    rec = dict(
        label=label, flight=truth.flight, bands=bands, k_active=k,
        s_grid=s_grid.tolist(), tau_grid=(s_grid * truth.tau_bot).tolist(),
        m=fwd.m, K_modes=list(map(int, Kmodes)),
        truth=dict(tau_bot=truth.tau_bot, r_top=truth.r_top, r_base=truth.r_base),
        clim={k_: clim[k_] for k_ in ("r_top_mean", "r_top_std", "r_base_mean",
                                      "r_base_std", "tau_bot_mean", "tau_bot_std", "n")},
        A_fixed_anchor=summary(post_fix, Sa_fix, False),
        B_joint_broad=summary(post_broad, Sa_broad, True),
        C_joint_clim=summary(post_clim, Sa_clim, True),
        runtime_s=time.perf_counter() - t0,
    )
    print(f"    DOFS:  A(fixed)={post_fix.dofs:.2f}  "
          f"B(broad)={post_broad.dofs:.2f} (prof {rec['B_joint_broad']['profile_dofs']:.2f}"
          f"/rbase {rec['B_joint_broad']['r_base_dofs']:.2f}"
          f"/taub {rec['B_joint_broad']['tau_bot_dofs']:.2f})  "
          f"C(clim)={post_clim.dofs:.2f}")
    print(f"    tau_bot 1σ: prior {rec['B_joint_broad']['tau_bot_prior_sigma']:.2f} "
          f"-> post {rec['B_joint_broad']['tau_bot_sigma']:.2f} (broad); "
          f"r_base 1σ: prior {rec['B_joint_broad']['r_base_prior_sigma']:.2f} "
          f"-> post {rec['B_joint_broad']['r_base_sigma']:.2f}")
    _save(rec)
    return rec


if __name__ == "__main__":
    which = sys.argv[1:] or [c[0] for c in CONFIGS]
    for cfg in CONFIGS:
        if cfg[0] in which:
            try:
                run_config(*cfg)
            except Exception as e:                       # keep going; log + continue
                import traceback
                print(f"!! {cfg[0]} FAILED: {e}")
                traceback.print_exc()
    print("\nDONE.")
