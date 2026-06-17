"""Put the joint-retrieval prior on solid footing: VOCALS-REx empirical
distributions of (r_top, r_base, tau_bot) + a linearized prior-sensitivity study
that tests three hypotheses and verifies the r_base correlation mechanism.

Method (cheap, rigorous): linearize the forward at the TRUTH joint state -> one
Jacobian K per scene. The linearized OE retrieval is x_hat = x_a + A (x_truth -
x_a) with averaging kernel A = S_hat K^T Se^-1 K. So:
  * a parameter with A_ii ~ 1 IGNORES its prior (data wins);
  * a parameter with A_ii ~ 0 just RETURNS its prior mean.
Varying the prior (means/sigmas/correlation) and reading x_hat + posterior sigma
answers "does this prior matter?" without many full GN runs.

  H1 r_top : observable -> prior should not matter (A_top ~ 1).
  H2 r_base: shielded for thick -> prior dominates (A_base ~ 0); needs a
             well-chosen, narrow prior. Visible for thin.
  H3 tau_bot: should be uninformative; tau_bot still retrievable (A ~ 1) from
             conservative + absorbing bands.
  Mechanism 3: r_base moves off an inverted prior mean ONLY via the correlated
             prior (off-diagonal A); a DIAGONAL prior pins r_base at its mean.

  /tmp/jaxve/bin/python -u tests/supplementary/prior_investigation.py
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
OUT = _root / "docs" / "prior_investigation_results.json"
NQuad, NLeg_all, NFourier, v_eff = 16, 128, 8, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50]); view_phi = np.array([pi, pi, pi])
_precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
results = {}


# ---------------------------------------------------------------------------
# 1. VOCALS-REx empirical distributions of r_top, r_base, tau_bot
# ---------------------------------------------------------------------------
def empirical_distributions():
    profs = vio.load_all_profiles(DATA)
    rt = np.array([p.r_top for p in profs])
    rb = np.array([p.r_base for p in profs])
    tb = np.array([p.tau_bot for p in profs])
    keep = (tb >= 0.3) & (tb <= 60)          # physical filter (artefact tau_bot)
    rt, rb, tb = rt[keep], rb[keep], tb[keep]
    thick = tb >= 8.0                         # base shielded subset

    def stats(x):
        return dict(n=int(x.size), mean=float(np.mean(x)), median=float(np.median(x)),
                    std=float(np.std(x)), mad=float(1.4826 * np.median(np.abs(x - np.median(x)))),
                    p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                    min=float(x.min()), max=float(x.max()))
    out = dict(
        n=int(rt.size),
        r_top=stats(rt), r_base=stats(rb), tau_bot=stats(tb),
        r_top_thick=stats(rt[thick]), r_base_thick=stats(rb[thick]),
        r_base_minus_top=stats(rb - rt),               # <0 => adiabatic (top bigger)
        corr_rtop_rbase=float(np.corrcoef(rt, rb)[0, 1]),
        corr_rtop_taub=float(np.corrcoef(rt, tb)[0, 1]),
        corr_rbase_taub=float(np.corrcoef(rb, tb)[0, 1]),
        frac_rtop_gt_rbase=float(np.mean(rt > rb)),      # adiabatic direction fraction
    )
    print("=== VOCALS-REx empirical distributions (n=%d, tau_bot in [0.3,60]) ===" % rt.size)
    for k in ("r_top", "r_base", "tau_bot"):
        s = out[k]
        print(f"  {k:8s} mean {s['mean']:5.2f} median {s['median']:5.2f} "
              f"std {s['std']:5.2f} MAD {s['mad']:5.2f}  [p5 {s['p5']:.1f}, p95 {s['p95']:.1f}] "
              f"range [{s['min']:.1f},{s['max']:.1f}]")
    print(f"  r_base relative spread (MAD/median) {out['r_base']['mad']/out['r_base']['median']:.2f} "
          f"vs r_top {out['r_top']['mad']/out['r_top']['median']:.2f}  "
          f"(H2: is r_base narrower?)")
    print(f"  thick-only (tau>=8): r_top {out['r_top_thick']['median']:.1f}+-{out['r_top_thick']['mad']:.1f}, "
          f"r_base {out['r_base_thick']['median']:.1f}+-{out['r_base_thick']['mad']:.1f}")
    print(f"  frac(r_top>r_base) = {out['frac_rtop_gt_rbase']:.2f}  (adiabatic direction)")
    print(f"  corr(r_top,r_base)={out['corr_rtop_rbase']:.2f}  corr(r_base,tau)={out['corr_rbase_taub']:.2f}  "
          f"corr(r_top,tau)={out['corr_rtop_taub']:.2f}")
    return out


# ---------------------------------------------------------------------------
# 2. Linearized prior-sensitivity via the averaging kernel at truth
# ---------------------------------------------------------------------------
def build_scene(flight, target_tau, bands, k_active):
    profs = vio.load_all_profiles(DATA)
    pool = [p for p in profs if p.flight == flight]
    truth = vio.pick_profile(pool, target_tau)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    ob = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, _precomp,
                                        n_radii=600), i) for i in range(len(bands))]
    fwd = roe.RetrievalForward(ob, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
                               tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
                               view_mu=view_mu, view_phi=view_phi,
                               BDRF_bands=[[0.06]] * len(bands), NLeg_all=NLeg_all,
                               NFourier=NFourier, retrieve_tau_bot=True,
                               retrieve_r_base=True)
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    # fixed grid in s (uniform, k_active interior nodes incl top)
    s_grid = np.linspace(0.0, 1.0, k_active + 1)[:-1]
    # TRUTH joint state on the grid (linearization point)
    x_truth = np.concatenate([np.interp(s_grid * truth.tau_bot, truth.tau, truth.r_e),
                              [truth.r_base, truth.tau_bot]])
    K = np.asarray(fwd.jacobian(x_truth, s_grid), float)
    return dict(truth=truth, clim=clim, s_grid=s_grid, x_truth=x_truth, K=K, Se=Se,
                k=len(s_grid))


def linpost(scene, x_a, Sa):
    """Linearized posterior at truth: x_hat = x_a + A (x_truth - x_a); returns
    x_hat, posterior sigma, and per-component DOFS (diag A)."""
    post = roe.posterior_diagnostics(scene["K"], Sa, scene["Se"])
    A = post.A
    x_hat = np.asarray(x_a) + A @ (scene["x_truth"] - np.asarray(x_a))
    k = scene["k"]
    return dict(x_hat=x_hat, sigma=post.error, dofs=post.dofs,
                A_diag=np.diag(A), x_top=float(x_hat[0]), x_base=float(x_hat[k]),
                x_taub=float(x_hat[k + 1]), sig_base=float(post.error[k]),
                sig_taub=float(post.error[k + 1]),
                A_top=float(np.diag(A)[0]), A_base=float(np.diag(A)[k]),
                A_taub=float(np.diag(A)[k + 1]))


def joint_prior(s_grid, *, r_top, r_base, tau_bot, sig_top, sig_base, sig_taub, corr=0.5):
    return roe.make_joint_prior(s_grid, tau_bot_prior=tau_bot, r_top_prior=r_top,
                                r_base_prior=r_base, sigma_top=sig_top,
                                sigma_base=sig_base, sigma_tau_bot=sig_taub,
                                corr_length=corr)


def sensitivity(scene, tag):
    tr = scene["truth"]; sg = scene["s_grid"]
    print(f"\n=== {tag}: {tr.flight} tau_bot={tr.tau_bot:.2f} r_top={tr.r_top:.2f} "
          f"r_base={tr.r_base:.2f} (k={scene['k']}) ===")
    rec = dict(flight=tr.flight, tau_bot=tr.tau_bot, r_top=tr.r_top, r_base=tr.r_base,
               cases={})
    # baseline adiabatic prior (Option 1 numbers)
    base = dict(r_top=12.0, r_base=6.0, tau_bot=float(scene["clim"]["tau_bot_mean"]),
                sig_top=5.0, sig_base=8.0, sig_taub=0.5 * scene["clim"]["tau_bot_mean"])
    cases = {
        "adiabatic_base": dict(base),
        "inverted(bug)": {**base, "r_top": 10.0, "r_base": 12.0},
        "inverted_DIAGONAL": {**base, "r_top": 10.0, "r_base": 12.0, "corr": 1e-6},
        # H1 r_top: vary mean far from truth
        "rtop_lo(8)": {**base, "r_top": 8.0}, "rtop_hi(16)": {**base, "r_top": 16.0},
        # H2 r_base: vary mean + width
        "rbase_lo(4)": {**base, "r_base": 4.0}, "rbase_hi(10)": {**base, "r_base": 10.0},
        "rbase_narrow(2)": {**base, "sig_base": 2.0}, "rbase_wide(12)": {**base, "sig_base": 12.0},
        # H3 tau_bot: vary mean + width
        "taub_lo(5)": {**base, "tau_bot": 5.0}, "taub_hi(25)": {**base, "tau_bot": 25.0},
        "taub_tight(2)": {**base, "sig_taub": 2.0}, "taub_vague(40)": {**base, "sig_taub": 40.0},
    }
    for name, cfg in cases.items():
        x_a, Sa = joint_prior(sg, **{k: cfg[k] for k in
                              ("r_top", "r_base", "tau_bot", "sig_top", "sig_base", "sig_taub")},
                              corr=cfg.get("corr", 0.5))
        lp = linpost(scene, x_a, Sa)
        rec["cases"][name] = dict(prior_top=cfg["r_top"], prior_base=cfg["r_base"],
                                  prior_taub=cfg["tau_bot"], x_top=lp["x_top"],
                                  x_base=lp["x_base"], x_taub=lp["x_taub"],
                                  A_top=lp["A_top"], A_base=lp["A_base"], A_taub=lp["A_taub"],
                                  sig_base=lp["sig_base"], sig_taub=lp["sig_taub"])
        print(f"  {name:18s} prior(top {cfg['r_top']:.0f} base {cfg['r_base']:.0f} "
              f"taub {cfg['tau_bot']:.0f}) -> x_hat(top {lp['x_top']:5.2f} base {lp['x_base']:5.2f} "
              f"taub {lp['x_taub']:5.2f})  A(top {lp['A_top']:.2f} base {lp['A_base']:.2f} "
              f"taub {lp['A_taub']:.2f})")
    return rec


if __name__ == "__main__":
    t0 = time.perf_counter()
    results["empirical"] = empirical_distributions()
    thin = build_scene("RF11", 1.0, [1.24, 2.13], 4)
    results["thin"] = sensitivity(thin, "THIN")
    thick = build_scene("RF03", 23.3, [1.24, 1.64, 2.13], 5)
    results["thick"] = sensitivity(thick, "THICK")
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved -> {OUT.name}  [{time.perf_counter()-t0:.0f}s]")
