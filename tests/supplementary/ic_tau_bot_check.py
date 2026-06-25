"""One-off τ_bot-unknown sensitivity check (NOT a §15 headline — DESIGN §13 note).

The definitive IC profiling holds **τ_bot known** and reports the r_e(τ) profile IC as
an upper bound. This script quantifies the two worries that choice trades against, on a
few thin/mid/thick representative profiles (node basis: r_e at s_ref + r_base + τ_bot):

  (i)  CROSSTALK — does promoting τ_bot to an UNKNOWN steal/scramble r_e information, or
       is it a clean ~1-DOF add? Compare the r_e-block (profile + base) DOFS with τ_bot
       FIXED (drop its column/row) vs τ_bot RETRIEVED (full joint, LOO climatology prior
       gives τ_bot a mildly-informative median+MAD Gaussian), at τ_bot linearized = truth.
  (ii) LINEARIZATION — the far-from-truth worry: re-do the joint with τ_bot linearized at
       the PRIOR MEAN instead of the truth and see whether the r_e-block IC moves.

τ_bot self-DOFS ≈ 1 (and small r_e shift) ⇒ the τ_bot-known headline is a faithful upper
bound. Full-view radiance, OCI 2 % noise, 9-band superset, LOO prior, priormean r_e
linearization — matching ic_worker_profile.py. Reported in DESIGN_DECISIONS, not the notebook.

Run (notebook env):
  JAX_PLATFORMS=cpu /srv/conda/envs/notebook/bin/python tests/supplementary/ic_tau_bot_check.py
"""
import sys
import os
import json
import time
from pathlib import Path
from math import pi

import numpy as np

_here = Path(__file__).resolve().parent
_src = _here.parents[1] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
import noise_model as nm                                            # noqa: E402
import optics_table as ot                                          # noqa: E402

DATA = os.environ.get('VOCALS_DATA',
                      '/home/jovyan/cloud_profile_retrieval/'
                      'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQ = int(os.environ.get('ENSEMBLE_NQUAD', '48'))
OPTICS_CACHE = Path(os.environ.get('OPTICS_CACHE', _here / 'optics_table_9band.npz'))
OUT = os.environ.get('OUT', str(_here.parents[1] / 'docs' / 'cached_results'
                                / 'info_content_tau_bot_check.json'))
TARGETS = [3.0, 10.0, 24.0]                                         # thin / mid / thick τ_bot
mu0, NLeg_all, v_eff = 0.9, 128, 0.10
N_PHYS = NQ // 2
BANDS = [0.55, 0.67, 0.86, 1.038, 1.24, 1.64, 2.13, 2.26, 3.7]
NB = len(BANDS)
NOISE = nm.oci_swir()
VIEW_MU, VIEW_PHI = np.linspace(0.95, 0.25, N_PHYS), np.full(N_PHYS, pi)
s_ref = np.linspace(0.0, 1.0, 6)[:-1]
n_re_nodes = len(s_ref)                                             # interior r_e nodes (base is separate)

profiles = vio.load_all_profiles(DATA)
phys = [(i, p) for i, p in enumerate(profiles)
        if 0.3 <= float(p.tau_bot) <= 100.0 and len(np.asarray(p.tau)) >= 5]
reps = []
for tgt in TARGETS:
    i, p = min(phys, key=lambda ip: abs(ip[1].tau_bot - tgt))
    reps.append((i, p, tgt))

re_table = ot.build_or_load_table(BANDS, 2.0, 25.0, 32, v_eff,
                                  cache_path=OPTICS_CACHE, NLeg=NLeg_all, n_radii=600)
opt = [ot.select_channel(re_table, i) for i in range(NB)]


def rblock_dofs(post):
    """r_e-block DOFS = profile (interior) + base self-information (the base IS an r_e value)."""
    comp = roe.dofs_by_component(post, n_re_nodes, retrieve_r_base=True, retrieve_tau_bot=True)
    return comp["profile"] + comp["r_base"], comp.get("tau_bot", 0.0)


records = []
for idx, truth, tgt in reps:
    t0 = time.time()
    flight = getattr(truth, 'flight', '?')
    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    fwd = roe.RetrievalForward(opt, NQuad=NQ, mu0=mu0, I0=1.0, phi0=0.0,
                               tau_bot=truth.tau_bot, r_base=truth.r_base,
                               view_mu=VIEW_MU, view_phi=VIEW_PHI, BDRF_bands=[[0.06]] * NB,
                               NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True,
                               jac_mode='fwd')
    xa, Sa = roe.make_climatology_prior(s_ref, clim)               # 7×7 joint [re(5), r_base, τ_bot]
    xa = np.asarray(xa); Sa = np.asarray(Sa)
    # priormean r_e linearization (set iv); the two τ_bot linearizations:
    x_truthtb = np.concatenate([xa[:n_re_nodes + 1], [truth.tau_bot]])      # τ_bot = truth
    x_priortb = np.concatenate([xa[:n_re_nodes + 1], [clim["tau_bot_mean"]]])  # τ_bot = prior mean

    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = NOISE.Se(y, n_bands=NB)
    roe.select_num_modes(fwd, x_truthtb, s_ref, (0.005 ** 2) * np.eye(fwd.m))

    K_truth = np.asarray(fwd.jacobian(x_truthtb, s_ref))           # m × 7 (last col = ∂y/∂τ_bot)
    K_prior = np.asarray(fwd.jacobian(x_priortb, s_ref))

    # τ_bot KNOWN: drop the τ_bot column/row (6×6 r_e+base system), at the truth-τ_bot lin point.
    post_known = roe.posterior_diagnostics(K_truth[:, :-1], Sa[:-1, :-1], Se)
    re_known = float(post_known.dofs)                              # all 6 cols are r_e values
    # τ_bot UNKNOWN (full joint), two linearizations:
    post_u_truth = roe.posterior_diagnostics(K_truth, Sa, Se)
    post_u_prior = roe.posterior_diagnostics(K_prior, Sa, Se)
    re_u_truth, tb_u_truth = rblock_dofs(post_u_truth)
    re_u_prior, tb_u_prior = rblock_dofs(post_u_prior)

    rec = dict(index=idx, flight=flight, tau_bot=float(truth.tau_bot), target=tgt,
               tau_bot_prior_mean=float(clim["tau_bot_mean"]),
               re_block_dofs_tau_known=re_known,
               re_block_dofs_tau_unknown_truthlin=re_u_truth,
               re_block_dofs_tau_unknown_priorlin=re_u_prior,
               tau_bot_self_dofs_truthlin=tb_u_truth,
               tau_bot_self_dofs_priorlin=tb_u_prior,
               crosstalk_shift=re_u_truth - re_known,              # (i): r_e change when τ_bot unknown
               linearization_shift=re_u_prior - re_u_truth,        # (ii): far-from-truth τ_bot lin effect
               total_dofs_unknown=float(post_u_truth.dofs))
    records.append(rec)
    print(f"[{idx}] {flight} τ={truth.tau_bot:.1f} ({tgt:.0f}): r_e-block DOFS "
          f"known={re_known:.3f} unknown={re_u_truth:.3f} (Δ={rec['crosstalk_shift']:+.3f}); "
          f"τ_bot self={tb_u_truth:.3f}; lin(truth→prior) Δr_e={rec['linearization_shift']:+.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

summary = dict(
    bands=BANDS, NQuad=NQ, noise="oci_swir_2pct", n_re_nodes=n_re_nodes,
    note=("Node-basis sensitivity (r_e at s_ref + base + τ_bot). τ_bot self-DOFS≈1 and small "
          "crosstalk/linearization shift ⇒ τ_bot-known headline is a faithful r_e upper bound."),
    mean_crosstalk_shift=float(np.mean([r["crosstalk_shift"] for r in records])),
    mean_tau_bot_self_dofs=float(np.mean([r["tau_bot_self_dofs_truthlin"] for r in records])),
    mean_linearization_shift=float(np.mean([r["linearization_shift"] for r in records])),
    records=records)
Path(OUT).write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUT}\n  mean τ_bot self-DOFS {summary['mean_tau_bot_self_dofs']:.3f} "
      f"(≈1 = clean add); mean r_e crosstalk {summary['mean_crosstalk_shift']:+.3f} DOFS; "
      f"mean lin shift {summary['mean_linearization_shift']:+.3f} DOFS")
