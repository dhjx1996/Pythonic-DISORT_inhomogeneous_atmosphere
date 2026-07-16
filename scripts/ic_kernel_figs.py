"""Fig 0a/0b + Fig 5 mechanism tier from dense kernel-probe npz (2026-07-15).

Consumes the ic_worker_wiggle sidecars (K_full (n_bands*n_views, n_s), s_grid,
sigma_full, re_grid, bands, view_mu) for the three representative profiles
(98 thin / 55 medium / 39 thick) and renders:

  Fig 0a — SIGNED Eq-3 penetration kernels w(s) = normalized d(ln I)/d(ln r_e(s))
           at near-nadir, spectral ladder per profile. The signed kernel is the
           validated primary object (Pla2000 Eq-3 criterion, 2026-07-14 tiers);
           |K| rectifies sign-flip texture into fake tail mass in the
           thin-cloud absorbing regime (tol5/6/7 ladder, 2026-07-15).
  Fig 0b — the angular axis: signed-kernel depth centroid vs view mu (left) and
           noise-relative signal amplitude vs mu (right), per band; glory at
           mu_v = mu_0 = 0.9 marked.
  Fig 5 mechanism — per-node fractional variance reduction (data_fraction) vs
           depth on the DENSE grid under the full correlated OU prior vs its
           diagonal (decorrelated) version: where the correlation pumps
           measurement information below the kernels' reach.

Pure post-processing (NumPy + local prior build; no RT). Usage:
    ic_kernel_figs.py <parts_tol7_dir> <out_prefix>
e.g. ic_kernel_figs.py runs/_wiggle_mie_ve046_tol7_parts docs/figures/ve046
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pydisort_riccati_jax import vocals_io as vio                    # noqa: E402
from pydisort_riccati_jax import retrieval_oe as roe                 # noqa: E402

VOCALS = "/burg-archive/apam/projects/multispectral-retrieval-using-MODIS/VOCALS_REx_data"
PARTS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/_wiggle_mie_ve046_tol7_parts")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/ve046"
IDXS = (98, 55, 39)          # thin / medium / thick
MU0 = 0.9

BAND_C = {0.55: "#5778a4", 0.67: "#6a9f58", 0.86: "#85b6b2", 1.038: "#e7ca60",
          1.24: "#a87c9f", 1.64: "#f1a2a9", 2.13: "#967662", 2.26: "#b8b0ac",
          3.7: "#d1615d", 4.05: "#e49444"}


def load(idx):
    """Raw physical kernel K = ∂I/∂r_e(s) — the validated Eq-3 object
    (platnick_eq3_validation: r*_signed = Σ K_j r_e(s_j)/Σ K_j on these rows)."""
    z = dict(np.load(PARTS / f"{idx}.npz", allow_pickle=True))
    s = np.asarray(z["s_grid"], float)
    K = np.asarray(z["K_full"], float).reshape(len(z["bands"]), len(z["view_mu"]), -1)
    return z, s, K


def signed_w(K_row, s):
    """Signed Eq-3 kernel, unit signed area."""
    a = np.trapezoid(K_row, s)
    return K_row / a


def fig0a():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    for ax, idx in zip(axes, IDXS):
        z, s, K = load(idx)
        bands = np.asarray(z["bands"], float)
        inad = int(np.argmax(np.asarray(z["view_mu"], float)))
        for b, lam in enumerate(bands):
            w = signed_w(K[b, inad], s)
            ax.plot(w, s, color=BAND_C[round(float(lam), 3)], lw=1.6,
                    label=f"{lam:g} µm")
        ax.axvline(0, color="#999", lw=0.7)
        ax.set_title(f"idx{idx} ({str(z['flight'])}, τ_bot={float(z['tau_bot']):.1f})",
                     fontsize=10)
        ax.set_xlabel("w(s) — signed Eq-3 kernel (unit area)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("normalized depth  s = τ/τ_bot")
    axes[0].invert_yaxis()
    axes[2].legend(fontsize=7.5, loc="lower right", frameon=False, ncol=2)
    fig.suptitle("Fig 0a — signed penetration kernels w(s) ∝ ∂I/∂r_e(s), "
                 "near-nadir view", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_fig0a_signed_kernels.png", dpi=150)
    print("saved", f"{PREFIX}_fig0a_signed_kernels.png")


def _smooth_nan(y, k=5):
    """NaN-aware running mean for DISPLAY (per-view signed centroids jitter — real,
    tol6/tol7-converged texture, per the 2026-07-15 full-fan gate check — not noise;
    smoothing is cosmetic, matching the original §16 Fig 0b convention)."""
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    yz = np.where(valid, y, 0.0)
    pad = k // 2
    yz_p = np.r_[np.repeat(yz[0], pad), yz, np.repeat(yz[-1], pad)]
    v_p = np.r_[np.repeat(float(valid[0]), pad), valid.astype(float), np.repeat(float(valid[-1]), pad)]
    num = np.convolve(yz_p, np.ones(k), mode="valid")
    den = np.convolve(v_p, np.ones(k), mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def fig0b():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), sharex=True)
    for col, idx in enumerate(IDXS):
        z, s, K = load(idx)
        bands = np.asarray(z["bands"], float)
        vmu = np.asarray(z["view_mu"], float)
        sig = np.asarray(z["sigma_full"], float).reshape(len(bands), len(vmu))
        order = np.argsort(vmu)
        for b, lam in enumerate(bands):
            cen = np.array([np.trapezoid(signed_w(K[b, v], s) * s, s) for v in order])
            # signed centroid is undefined where the row's lobes cancel (|∫w| ≪ ∫|w|,
            # e.g. crossing the cloudbow) — mask those views instead of plotting 1/0 spikes
            purity = np.array([abs(np.trapezoid(K[b, v], s))
                               / np.trapezoid(np.abs(K[b, v]), s) for v in order])
            cen = np.where(purity > 0.5, cen, np.nan)
            cen = _smooth_nan(cen, k=5)
            # noise-relative signal: ∫|K/σ| ds (ic_analysis_definitive amp_rel convention)
            amp = np.trapezoid(np.abs(K[b, order]), s, axis=1) / sig[b, order]
            c = BAND_C[round(float(lam), 3)]
            axes[0, col].plot(vmu[order], cen, color=c, lw=1.5, label=f"{lam:g}")
            axes[1, col].plot(vmu[order], amp / amp[np.argmax(vmu[order])],
                              color=c, lw=1.5)
        axes[0, col].set_ylim(0, 1.05)
        for r in (0, 1):
            axes[r, col].axvline(MU0, color="#999", lw=0.8, ls=":")
            axes[r, col].grid(alpha=0.25, lw=0.5)
        axes[0, col].set_title(f"idx{idx} (τ_bot={float(z['tau_bot']):.1f})", fontsize=10)
        axes[1, col].set_xlabel("view µ  (dotted: glory µ=µ0)")
    axes[0, 0].set_ylabel("kernel depth centroid ⟨s⟩ (signed)")
    axes[1, 0].set_ylabel("noise-relative signal (× near-nadir)")
    axes[0, 2].legend(fontsize=7, ncol=2, frameon=False, title="µm", title_fontsize=7)
    fig.suptitle("Fig 0b — the angular axis: depth coverage (top, 5-view running mean)\n"
                 "and usable signal (bottom) across the view fan", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{PREFIX}_fig0b_angular.png", dpi=150)
    print("saved", f"{PREFIX}_fig0b_angular.png")


def fig5_mechanism():
    profiles = vio.load_all_profiles(VOCALS)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    for ax, idx in zip(axes, IDXS):
        z, s, K = load(idx)
        sig = np.asarray(z["sigma_full"], float)
        Kw = (np.asarray(z["K_full"], float)
              * np.asarray(z["re_grid"], float)[None, :]) / sig[:, None]
        clim = vio.vocals_climatology(profiles, exclude_flight=str(z["flight"]))
        _, Sa = roe.make_climatology_prior(s, clim, log=True,
                                           retrieve_r_base=False,
                                           retrieve_tau_bot=False)
        Sa = np.asarray(Sa, float)
        G = Kw.T @ Kw
        for Sa_i, lab, c in ((Sa, "correlated (OU) prior", "#5778a4"),
                             (np.diag(np.diag(Sa)), "diagonal prior", "#d1615d")):
            S_hat = np.linalg.inv(G + np.linalg.inv(Sa_i))
            df = 1.0 - np.diag(S_hat) / np.diag(Sa_i)
            ax.plot(df, s, color=c, lw=1.8, label=lab)
        ax.set_title(f"idx{idx} (τ_bot={float(z['tau_bot']):.1f})", fontsize=10)
        ax.set_xlabel("data fraction  1 − Ŝ_jj/S_a,jj")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1.02)
    axes[0].set_ylabel("normalized depth  s = τ/τ_bot")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.suptitle("Fig 5 (mechanism) — the correlation pump on the dense grid: "
                 "measurement constraint vs depth, correlated vs decorrelated prior",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_fig5_mechanism.png", dpi=150)
    print("saved", f"{PREFIX}_fig5_mechanism.png")


if __name__ == "__main__":
    fig0a()
    fig0b()
    fig5_mechanism()
