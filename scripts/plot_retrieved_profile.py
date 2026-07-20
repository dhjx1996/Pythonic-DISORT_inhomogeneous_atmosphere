"""Standard single-profile retrieval diagnostic plot (2026-07-15).

Format fixed to docs/figures/bad_Adiabest-fit.png, the reference template: this IS
the default format for plotting retrieved profiles (see project memory) -- reuse
this script rather than one-off inline plotting code.

All curves in absolute optical depth tau (0 = cloud top, tau_bot = cloud base):
  * truth (in situ)       -- raw VOCALS-REx r_e(tau) samples, grey
  * prior                 -- prior mean shape on the PRE-RETRIEVED tau_bot support
                              (tau_bot_pre from the {idx}.json setup sidecar; user rule
                              2026-07-16), dense via fwd.profile, orange dashed. NOT the
                              climatological tau_bot in x_a_log: rendering on that support
                              kinks into a clamped vertical tail wherever the cloud is
                              thicker than the climatology (s>1 clamps to r_base).
  * best-fit adiabat (oracle, W1) -- re_adia_dense from the sidecar (fit against truth,
                              so on the TRUTH tau_bot axis), green dash-dot. NO top/base
                              markers (user rule 2026-07-18: the circles belong to the k=1
                              retrieval overlay, not the oracle)
  * retrieved             -- re_ours_dense (on the retrieved tau_bot axis), blue solid,
                              with node markers +-1 sigma (delta method from S_hat_log)
  * truth base            -- black star at (truth_r_base, truth_tau_bot)
  * retrieved base        -- red X +-1 sigma, distinguished from the interior nodes

Title reports chi2_red, tau_bot (retrieved vs truth), and DOFS (not SIC -- 2026-07-15).

Optional overlay (PLOT_K1_PARTS=<parts_dir>): the k=1 RETRIEVED adiabat from the matched
adiabatic-retrieval campaign (ve046_adia_bundle_1137436) -- the {idx}_{config}.npz sidecar's
re_ours_dense on ITS OWN retrieved tau_bot support, purple SOLID (user 2026-07-20: both
retrieved profiles get equal visual weight) with top/base circle markers, chi2_red in the label
(large chi2_red = the adiabat model class cannot fit this cloud's radiances). Off by default
to keep the standard format unchanged. (The retired PLOT_COMPROMISE overlay is gone -- the
truth-fed compromise rung was replaced by this real competitor, 2026-07-17.)

Usage: plot_retrieved_profile.py <idx> [config=A] [parts_dir=runs/_ve046_tik_fr_parts]
                                  [out=docs/figures/ve046_idx{idx}_profile.png]
Env: OSSE_VEFF/OSSE_RE_MAX/OPTICS_CACHE/RADIANCE_CACHE default to the ve046 fq
     campaign (override for a different campaign's parts_dir); CURVATURE_LAMBDA
     defaults to 1.0 (the ve046 campaign's value) for the title only.
     PLOT_CAMPAIGN_TAG (default "ve046") titles the campaign — set it when plotting a
     parts_dir whose retrieval was not the self-consistent ve046 one (e.g. the mismatched-v_e
     run), so the figure cannot be mistaken for a self-consistent result.

     NB the optics env must keep describing the TRUTH world even for a mismatch parts_dir:
     load_optics derives its expected signature from the env, so pointing OPTICS_CACHE at a
     table the env does not describe SILENTLY REBUILDS AND OVERWRITES that file. The table
     loaded here only feeds fwd.profile (prior-shape interpolation, optics-independent), so
     the truth-world default is both correct and the safe choice.
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("OSSE_VEFF", "0.046")
os.environ.setdefault("OSSE_RE_MAX", "22")
os.environ.setdefault("OSSE_QUADRATURE", "fixed")
os.environ.setdefault("OSSE_RE_GRID_N", "181")
os.environ.setdefault("OSSE_N_RADII", "4096")
os.environ.setdefault("OPTICS_CACHE", str(_WORKSPACE / "data" / "optics_table_10band_nleg1536_re22_fq.npz"))
os.environ.setdefault("RADIANCE_CACHE", str(_WORKSPACE / "data" / "osse_radiances_ve046_fq.npz"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYDISORT_RICCATI_JAX_X64", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pydisort_riccati_jax import osse_config as oc                   # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

S_DENSE = np.linspace(0.0, 1.0, 50)                                  # matches retrieval_worker.py


def main():
    idx = int(sys.argv[1])
    config = sys.argv[2] if len(sys.argv) > 2 else "A"
    parts_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "runs/_ve046_tik_fr_parts")
    out = sys.argv[4] if len(sys.argv) > 4 else f"docs/figures/ve046_idx{idx}_profile.png"
    curvature_lambda = float(os.environ.get("CURVATURE_LAMBDA", "1.0"))
    campaign_tag = os.environ.get("PLOT_CAMPAIGN_TAG", "ve046")

    z = dict(np.load(parts_dir / f"{idx}_{config}.npz", allow_pickle=True))
    s_grid = np.asarray(z["s_grid"], float)
    tau_bot_ret = float(z["tau_bot_ret"])
    truth_tau_bot = float(z["truth_tau_bot"])
    x_a_log = np.asarray(z["x_a_log"], float)
    S_hat_diag = np.diag(np.asarray(z["S_hat_log"], float))
    re_nodes_ret = np.asarray(z["re_nodes_ret"], float)
    r_base_ret = float(z["r_base_ret"])

    # the prior is ALWAYS rendered on the PRE-RETRIEVED tau_bot support (user, 2026-07-16):
    # the prior r_e(s) shape lives in normalized depth, and tau_bot_pre is the support the
    # retrieval operated on. The climatological tau_bot inside x_a_log is display-misleading
    # (clamps to r_base past its own base wherever the cloud is thicker than the climatology).
    tau_bot_pre = float(json.loads((parts_dir / f"{idx}.json").read_text())["tau_bot_pre"])

    opt = oc.load_optics(oc.OPTICS_CACHE)
    fwd = oc.build_forward(opt, tau_bot=tau_bot_ret, r_base=r_base_ret,
                           views="retrieval", jac_mode="fwd")
    tau_dense_ret = S_DENSE * tau_bot_ret
    tau_dense_adia = S_DENSE * truth_tau_bot
    tau_dense_pre = S_DENSE * tau_bot_pre
    x_a_plot = x_a_log.copy()
    x_a_plot[-1] = np.log(tau_bot_pre)                               # prior shape on the pre-retrieved support
    re_prior_dense = fwd.profile(x_a_plot, s_grid, tau_dense_pre)

    sigma_nodes = re_nodes_ret * np.sqrt(S_hat_diag[:len(s_grid)])
    sigma_rbase = r_base_ret * np.sqrt(S_hat_diag[len(s_grid)])
    tau_nodes = s_grid * tau_bot_ret

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(z["truth_re"], z["truth_tau"], color="grey", lw=1.3, label="truth (in situ)")
    ax.plot(re_prior_dense, tau_dense_pre, color="tab:orange", ls="--", lw=1.8, label="prior")
    ax.plot(z["re_adia_dense"], tau_dense_adia, color="tab:green", ls="-.", lw=1.6,
            label="best-fit adiabat (oracle, W1)")
    k1_parts = os.environ.get("PLOT_K1_PARTS", "")
    if k1_parts:
        # PLOT_OVERLAY_LABEL relabels the overlay when the parts dir is not the matched
        # k=1 campaign (e.g. overlaying the weak-l FR counterpart for vetting)
        olabel = os.environ.get("PLOT_OVERLAY_LABEL", "retrieved adiabat (k=1)")
        z1 = dict(np.load(Path(k1_parts) / f"{idx}_{config}.npz", allow_pickle=True))
        re1 = np.asarray(z1["re_ours_dense"], float)
        tau1 = S_DENSE * float(z1["tau_bot_ret"])
        ax.plot(re1, tau1, color="tab:purple", ls="-", lw=2.2,
                label=f"{olabel}, $\\chi^2_r$={float(z1['chi2_red']):.2g}")
        # top/base circles live on the k=1 RETRIEVAL (user rule 2026-07-18) — never on the
        # oracle; ±1σ error bars (delta method from the overlay's own S_hat_log, 2026-07-19)
        S1 = np.diag(np.asarray(z1["S_hat_log"], float))
        n1 = len(np.asarray(z1["s_grid"], float))
        re_nodes1 = np.asarray(z1["re_nodes_ret"], float)
        sig1 = [re_nodes1[0] * np.sqrt(S1[0]), float(z1["r_base_ret"]) * np.sqrt(S1[n1])]
        ax.errorbar([re1[0], re1[-1]], [tau1[0], tau1[-1]], xerr=sig1, fmt="o",
                    color="tab:purple", ms=9, capsize=3, zorder=5,
                    label=("overlay top/base $\\pm1\\sigma$" if "k=1" not in olabel
                           else "k=1 top/base $\\pm1\\sigma$"))
    ax.plot(z["re_ours_dense"], tau_dense_ret, color="tab:blue", lw=2.2, label="retrieved")
    ax.plot(float(z["truth_r_base"]), truth_tau_bot, "*", color="k", ms=15, zorder=7,
            label="truth base")
    ax.errorbar(re_nodes_ret, tau_nodes, xerr=sigma_nodes, fmt="o", color="tab:blue",
                ms=6, capsize=3, zorder=6, label=r"nodes $\pm1\sigma$")
    ax.errorbar([r_base_ret], [tau_bot_ret], xerr=[sigma_rbase], fmt="X", color="tab:red",
                ms=12, capsize=3, zorder=7, label=r"retrieved base $\pm1\sigma$")

    ax.invert_yaxis()
    ax.set_xlabel(r"$r_e$ [µm]")
    ax.set_ylabel(r"optical depth $\tau$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")   # dynamic: keep the legend off the retrievals (user rule 2026-07-18)
    ax.set_title(
        f"idx-{idx} ({str(z['flight'])})  {campaign_tag}, curvature Tikhonov lambda={curvature_lambda:g}\n"
        f"$\\chi^2_\\mathrm{{red}}$={float(z['chi2_red']):.4f},  "
        f"$\\tau_\\mathrm{{bot}}$={tau_bot_ret:.2f} (truth {truth_tau_bot:.2f}),  "
        f"DOFS={float(z['dofs']):.2f}", fontsize=11)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
