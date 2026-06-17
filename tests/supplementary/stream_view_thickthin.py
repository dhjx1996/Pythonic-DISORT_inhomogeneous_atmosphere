"""Thin vs thick: is ToA-radiance information node-bound (multiple-scatter,
interpolated) or per-view-angle (single-scatter, TMS)? And does thick need fewer
streams than thin (multiple scattering washes out high moments)?

Mechanism (eval_radiance): out = barycentric_interp(mu_nodes) [diffuse, node-bound]
                                 + TMS(mu_obs) [single-scatter, exact angle, full moments].

Tests, at the TRUTH averaging kernel A (x̂ = x_a + A(x_truth−x_a)):
  1. single-scatter FRACTION |TMS|/|full| at the view angles (thin >> thick?).
  2. view-CHOICE at FIXED NQuad=16: 3-moderate vs 3-oblique vs 8-dense view angles
     -> does A_top change with choice/count (thin) or saturate/node-bound (thick)?
  3. thick NQuad scan {16,24,32} (moderate) vs the thin scan (flat) — does thick
     info saturate with streams even faster?

  /tmp/jaxve/bin/python -u tests/supplementary/stream_view_thickthin.py
"""
import json, sys, time
from math import pi
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
import numpy as np
import jax.numpy as jnp
import vocals_io as vio
import retrieval_oe as roe
from miejax_lite import build_re_table, mie_legendre_precompute, select_channel, table_lookup
from pydisort_riccati_jax import riccati_setup, riccati_solve, eval_radiance

DATA = ("/home/jovyan/cloud_profile_retrieval/"
        "multispectral-retrieval-using-MODIS/VOCALS_REx_data")
OUT = _root / "docs" / "stream_view_thickthin_results.json"
NLeg_all, v_eff, mu0, I0, phi0 = 128, 0.10, 0.6, 1.0, 0.0
_pre = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
MOD3 = np.array([0.90, 0.65, 0.50])
OBL3 = np.array([0.55, 0.40, 0.28])
DENSE8 = np.linspace(0.95, 0.20, 8)


def scene(flight, target_tau, bands, NQuad, vm, NFourier=8):
    profs = vio.load_all_profiles(DATA)
    truth = vio.pick_profile([p for p in profs if p.flight == flight], target_tau)
    clim = vio.vocals_climatology(profs, exclude_flight=truth.flight)
    ob = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, _pre, n_radii=600), i)
          for i in range(len(bands))]
    vp = np.full(len(vm), pi)
    fwd = roe.RetrievalForward(ob, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
                               tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
                               view_mu=np.asarray(vm), view_phi=vp,
                               BDRF_bands=[[0.06]] * len(bands), NLeg_all=NLeg_all,
                               NFourier=NFourier, retrieve_tau_bot=True, retrieve_r_base=True)
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    k = 4 if target_tau < 5 else 5
    sg = np.linspace(0.0, 1.0, k + 1)[:-1]
    xt = np.concatenate([np.interp(sg * truth.tau_bot, truth.tau, truth.r_e),
                         [truth.r_base, truth.tau_bot]])
    K = np.asarray(fwd.jacobian(xt, sg), float)
    x_a, Sa = roe.make_marine_sc_prior(sg, r_top_prior=11.0, tau_bot_prior=clim["tau_bot_mean"])
    post = roe.posterior_diagnostics(K, Sa, Se)
    A = np.diag(post.A)
    return dict(truth=truth, clim=clim, fwd=fwd, bands=bands, NQuad=NQuad, n_view=len(vm),
                A_top=float(A[0]), A_profile=float(A[:k].sum()), dofs=float(post.dofs), k=k)


def ss_fraction(flight, target_tau, band, NQuad):
    """|TMS single-scatter| / |full radiance| at the moderate view angles, one band."""
    profs = vio.load_all_profiles(DATA)
    truth = vio.pick_profile([p for p in profs if p.flight == flight], target_tau)
    opt = select_channel(build_re_table([band], 2.0, 25.0, 32, v_eff, _pre, n_radii=600), 0)
    tb = float(truth.tau_bot)
    s_knots = jnp.asarray(np.append(truth.tau / tb, 1.0))
    vals = jnp.asarray(np.append(truth.r_e, truth.r_e[-1]))
    def om(t): return table_lookup(opt, jnp.interp(t / tb, s_knots, vals))[0]
    def leg(t): return table_lookup(opt, jnp.interp(t / tb, s_knots, vals))[1]
    frac = []
    for nt in (True, False):
        setup = riccati_setup(NQuad, I0, phi0, mu0, NFourier=8, NLeg_all=NLeg_all,
                              BDRF_Fourier_modes=[0.06], delta_M_scaling=True,
                              NT_cor=nt, tol=1e-3)
        res = riccati_solve(setup, om, leg, tb)
        r = np.array([float(eval_radiance(setup, res, m, pi)) for m in MOD3])
        frac.append(r)
    full, no_tms = frac
    return float(np.mean(np.abs(full - no_tms) / np.abs(full)))


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = {"thin": {}, "thick": {}}
    TH = ("thin", "RF11", 1.0, [1.24, 2.13])
    TK = ("thick", "RF03", 23.3, [1.24, 1.64, 2.13])

    print("=== single-scatter fraction |TMS|/|full| (NQuad=16, 2.13um) ===", flush=True)
    for tag, fl, tt, bands in (TH, TK):
        f = ss_fraction(fl, tt, 2.13, 16)
        res[tag]["ss_fraction_213"] = f
        print(f"  {tag:5s} {fl}: SS fraction = {f*100:.1f}%", flush=True)

    print("\n=== view-CHOICE / count at FIXED NQuad=16 (A_top, DOFS) ===", flush=True)
    for tag, fl, tt, bands in (TH, TK):
        res[tag]["view"] = {}
        for vlabel, vm in (("3-moderate", MOD3), ("3-oblique", OBL3), ("8-dense", DENSE8)):
            s = scene(fl, tt, bands, 16, vm)
            res[tag]["view"][vlabel] = dict(A_top=s["A_top"], A_profile=s["A_profile"],
                                            dofs=s["dofs"], n_view=s["n_view"])
            print(f"  {tag:5s} {vlabel:11s} (m={len(vm)*len(bands)}) A_top={s['A_top']:.2f} "
                  f"A_profile={s['A_profile']:.2f} DOFS={s['dofs']:.2f}", flush=True)

    print("\n=== thick NQuad scan (3-moderate) — does it saturate faster than thin? ===", flush=True)
    res["thick"]["nquad"] = {}
    for NQ in (16, 24, 32):
        s = scene("RF03", 23.3, [1.24, 1.64, 2.13], NQ, MOD3)
        res["thick"]["nquad"][NQ] = dict(A_top=s["A_top"], A_profile=s["A_profile"], dofs=s["dofs"])
        print(f"  thick NQ{NQ} A_top={s['A_top']:.2f} A_profile={s['A_profile']:.2f} "
              f"DOFS={s['dofs']:.2f}", flush=True)

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved -> {OUT.name}  [{time.perf_counter()-t0:.0f}s]", flush=True)
