"""Is the thin-cloud A_top deficit (~0.23 at NQuad=16) physical or a discrete-
ordinates resolution artifact? Near-top sensitivity in a thin cloud is single-
scattering-dominated and oblique-angle / high-moment sensitive; NQuad sets both
the angular grid and the diffuse Legendre truncation. We recompute the averaging
kernel at the truth for the THIN profile (RF11) over NQuad and view-angle
obliquity. If A_top climbs with streams/obliquity, the thin H1 deficit is
numerical (-> r_top is observable, H1 holds), not fundamental.

  /tmp/jaxve/bin/python -u tests/supplementary/thin_top_resolution.py
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
import retrieval_oe as roe
from miejax_lite import build_re_table, mie_legendre_precompute, select_channel

DATA = ("/home/jovyan/cloud_profile_retrieval/"
        "multispectral-retrieval-using-MODIS/VOCALS_REx_data")
OUT = _root / "docs" / "thin_top_resolution_results.json"
NLeg_all, v_eff, mu0, I0, phi0 = 128, 0.10, 0.6, 1.0, 0.0
BANDS = [1.24, 2.13]
_precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)

MODERATE = (np.array([0.90, 0.65, 0.50]), np.array([pi, pi, pi]))          # ~26-60 deg
OBLIQUE = (np.array([0.90, 0.65, 0.50, 0.35, 0.25]),                        # add 70-76 deg
           np.array([pi, pi, pi, pi, pi]))
# label, NQuad, view_set, NFourier
CONFIGS = [
    ("NQ16_moderate", 16, MODERATE, 8),
    ("NQ24_moderate", 24, MODERATE, 8),
    ("NQ32_moderate", 32, MODERATE, 8),
    ("NQ24_oblique",  24, OBLIQUE, 12),
    ("NQ32_oblique",  32, OBLIQUE, 12),
]


def run(label, NQuad, view, NFourier):
    t0 = time.perf_counter()
    profs = vio.load_all_profiles(DATA)
    truth = vio.pick_profile([p for p in profs if p.flight == "RF11"], 1.0)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    vm, vp = view
    ob = [select_channel(build_re_table(BANDS, 2.0, 25.0, 32, v_eff, _precomp,
                                        n_radii=600), i) for i in range(len(BANDS))]
    fwd = roe.RetrievalForward(ob, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
                               tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
                               view_mu=vm, view_phi=vp, BDRF_bands=[[0.06]] * len(BANDS),
                               NLeg_all=NLeg_all, NFourier=NFourier,
                               retrieve_tau_bot=True, retrieve_r_base=True)
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    k = 4
    s_grid = np.linspace(0.0, 1.0, k + 1)[:-1]
    x_truth = np.concatenate([np.interp(s_grid * truth.tau_bot, truth.tau, truth.r_e),
                              [truth.r_base, truth.tau_bot]])
    K = np.asarray(fwd.jacobian(x_truth, s_grid), float)
    # adiabatic prior; measure A_top and how much the top follows its prior
    def A_and_follow(r_top_prior):
        x_a, Sa = roe.make_joint_prior(s_grid, tau_bot_prior=clim["tau_bot_mean"],
                                       r_top_prior=r_top_prior, r_base_prior=6.0,
                                       sigma_top=5.0, sigma_base=8.0,
                                       sigma_tau_bot=0.5 * clim["tau_bot_mean"])
        post = roe.posterior_diagnostics(K, Sa, Se)
        x_hat = x_a + post.A @ (x_truth - x_a)
        return np.diag(post.A), x_hat[0], post.dofs
    Ad, xt8, dofs = A_and_follow(8.0)
    _, xt16, _ = A_and_follow(16.0)
    follow = (xt16 - xt8) / 8.0          # d x_hat_top / d prior  (0=data wins, 1=prior wins)
    rec = dict(label=label, NQuad=NQuad, n_view=len(vm), NFourier=NFourier,
               A_top=float(Ad[0]), A_profile=float(Ad[:k].sum()), dofs=float(dofs),
               xtop_prior8=float(xt8), xtop_prior16=float(xt16),
               top_prior_following=float(follow), m=int(fwd.m),
               runtime_s=time.perf_counter() - t0)
    print(f"  {label:15s} m={fwd.m:2d} -> A_top={rec['A_top']:.2f} "
          f"A_profile={rec['A_profile']:.2f} DOFS={dofs:.2f} | top-prior-following "
          f"{follow:.2f} (x_top 8->{xt8:.1f}, 16->{xt16:.1f}; truth {truth.r_top:.1f}) "
          f"[{rec['runtime_s']:.0f}s]", flush=True)
    return rec


if __name__ == "__main__":
    print("THIN RF11 top-resolution scan (A_top vs NQuad & view obliquity):", flush=True)
    out = []
    for c in CONFIGS:
        try:
            out.append(run(*c))
        except Exception as e:
            import traceback; print(f"!! {c[0]} FAILED: {e}"); traceback.print_exc()
        OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {OUT.name}")
