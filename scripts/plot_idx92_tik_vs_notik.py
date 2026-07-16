"""idx92 tik-vs-notik overlay (2026-07-16): bad_Adiabest-fit.png template, two retrieved
curves (canonical curvature_lambda=1 vs the no-Tikhonov re-run) on shared truth/prior/adiabat.

Usage: plot_idx92_tik_vs_notik.py
"""
import sys
import os
import json
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

S_DENSE = np.linspace(0.0, 1.0, 50)
IDX = 92
RUNS = [
    ("runs/_ve046_tik_fr_parts", "tik (canonical, lambda=1)", "tab:blue", "-"),
    ("runs/_ve046_notik_pathology_parts", "no-Tikhonov (lambda=0)", "tab:purple", "--"),
]


def main():
    out = "docs/figures/ve046_idx92_tik_vs_notik.png"
    opt = oc.load_optics(oc.OPTICS_CACHE)

    fig, ax = plt.subplots(figsize=(7, 8))
    z0 = None
    stats = {}
    for parts_dir, label, color, ls in RUNS:
        z = dict(np.load(Path(parts_dir) / f"{IDX}_A.npz", allow_pickle=True))
        if z0 is None:
            z0 = z
        stats[label] = (float(z["chi2_red"]), float(z["dofs"]))
        s_grid = np.asarray(z["s_grid"], float)
        tau_bot_ret = float(z["tau_bot_ret"])
        r_base_ret = float(z["r_base_ret"])
        re_nodes_ret = np.asarray(z["re_nodes_ret"], float)
        S_hat_diag = np.diag(np.asarray(z["S_hat_log"], float))
        sigma_nodes = re_nodes_ret * np.sqrt(S_hat_diag[:len(s_grid)])
        sigma_rbase = r_base_ret * np.sqrt(S_hat_diag[len(s_grid)])
        tau_nodes = s_grid * tau_bot_ret
        tau_dense_ret = S_DENSE * tau_bot_ret

        ax.plot(z["re_ours_dense"], tau_dense_ret, color=color, ls=ls, lw=2.2,
                label=f"retrieved: {label}")
        ax.errorbar(re_nodes_ret, tau_nodes, xerr=sigma_nodes, fmt="o", color=color,
                    ms=6, capsize=3, zorder=6)
        ax.errorbar([r_base_ret], [tau_bot_ret], xerr=[sigma_rbase], fmt="X", color=color,
                    ms=12, capsize=3, zorder=7)

        if parts_dir == "runs/_ve046_tik_fr_parts":
            tau_bot_pre = json.loads((Path(parts_dir) / f"{IDX}.json").read_text())["tau_bot_pre"]
            fwd = oc.build_forward(opt, tau_bot=tau_bot_ret, r_base=r_base_ret,
                                   views="retrieval", jac_mode="fwd")
            # profile() reads tau_bot from x itself -- swap x_a_log's trailing tau_bot
            # entry (unconditioned climatological mean) to the pre-retrieved tau_bot_pre
            # so the prior's [0,1] shape displays at the right scale (memory fix 2026-07-16).
            x_a_prior_log = np.asarray(z["x_a_log"], float).copy()
            x_a_prior_log[-1] = np.log(tau_bot_pre)
            tau_dense_prior = S_DENSE * tau_bot_pre
            re_prior_dense = fwd.profile(x_a_prior_log, s_grid, tau_dense_prior)
            ax.plot(re_prior_dense, tau_dense_prior, color="tab:orange", ls="--", lw=1.8,
                    label="prior")

    truth_tau_bot = float(z0["truth_tau_bot"])
    tau_dense_adia = S_DENSE * truth_tau_bot
    ax.plot(z0["truth_re"], z0["truth_tau"], color="grey", lw=1.3, label="truth (in situ)")
    ax.plot(z0["re_adia_dense"], tau_dense_adia, color="tab:green", ls="-.", lw=1.6,
            label="best-fit adiabat (oracle, W1)")
    ax.plot([z0["re_adia_dense"][0], z0["re_adia_dense"][-1]],
            [tau_dense_adia[0], tau_dense_adia[-1]], "o", color="tab:green", ms=10,
            zorder=5, label="adiabat top/base")
    ax.plot(float(z0["truth_r_base"]), truth_tau_bot, "*", color="k", ms=15, zorder=8,
            label="truth base")

    ax.invert_yaxis()
    ax.set_xlabel(r"$r_e$ [µm]")
    ax.set_ylabel(r"optical depth $\tau$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    chi2_t, dofs_t = stats["tik (canonical, lambda=1)"]
    chi2_n, dofs_n = stats["no-Tikhonov (lambda=0)"]
    ax.set_title(
        f"idx-{IDX} ({str(z0['flight'])})  ve046, curvature Tikhonov lambda=1 vs 0\n"
        f"tik: $\\chi^2_\\mathrm{{red}}$={chi2_t:.4f} DOFS={dofs_t:.2f}  |  "
        f"notik: $\\chi^2_\\mathrm{{red}}$={chi2_n:.4f} DOFS={dofs_n:.2f}",
        fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
