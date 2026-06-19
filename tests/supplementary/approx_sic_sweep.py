"""APPROXIMATE-variant SIC(k)+DOFS(k) sweep — the cheap operational selector.

Unlike fg_sic_dofs_sweep.py (which evaluated a *fresh* joint Jacobian on each candidate
k-grid → one XLA compile per k), this builds the JOINT pool Jacobian ONCE (one compile, on the
full ODE-grid interior + the r_base/tau_bot columns) and gets SIC(k)/DOFS(k) for every k by
SUB-SELECTING columns + a determinant ratio — no per-k Jacobian eval or recompile. It is therefore
APPROXIMATE (sub-selecting the full-pool Jacobian ignores the interpolant change with node set), and
the point is to show (a) the selection cost collapses and (b) whether the approximate SIC peak still
lands at the RMSE-optimal k (thin~4, RF03=3, RF08=4). Demonstration only — production uses the fixed
filter_threshold=0.25.

    /tmp/jaxve/bin/python tests/supplementary/approx_sic_sweep.py
"""
import sys
import json
import time
from pathlib import Path
from math import pi

import numpy as np
from scipy.linalg import qr

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import vocals_io as vio                                              # noqa: E402
import retrieval_oe as roe                                          # noqa: E402
from miejax_lite import (mie_legendre_precompute, build_re_table,   # noqa: E402
                         select_channel)

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NQuad, NLeg_all, v_eff = 16, 128, 0.10
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.linspace(0.95, 0.25, 8)
view_phi = np.full(view_mu.size, pi)

profiles = vio.load_all_profiles(DATA)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)


def build_case(truth, bands):
    clim = vio.vocals_climatology(profiles, exclude_flight=truth.flight)
    opt = [select_channel(build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600), i)
           for i in range(len(bands))]
    prior_builder = (lambda sn: roe.make_marine_sc_prior(
        sn, r_top_prior=clim['r_top_mean'], tau_bot_prior=clim['tau_bot_mean']))
    fwd = roe.RetrievalForward(
        opt, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
        tau_bot=clim['tau_bot_mean'], r_base=clim['r_base_mean'],
        view_mu=view_mu, view_phi=view_phi, BDRF_bands=[[0.06]] * len(bands),
        NLeg_all=NLeg_all, retrieve_tau_bot=True, retrieve_r_base=True, jac_mode='fwd')
    s_ref = np.linspace(0.0, 1.0, 6)[:-1]
    x_ref, _ = prior_builder(s_ref)
    roe.select_num_modes(fwd, x_ref, s_ref, (0.005 ** 2) * np.eye(fwd.m))
    y = roe.osse_observation(fwd, truth.tau, truth.r_e)
    Se = np.diag((0.03 * np.maximum(np.abs(y), 0.02)) ** 2)
    return prior_builder, fwd, s_ref, Se


CASES = [
    ("THIN ", vio.pick_profile(profiles, target_tau=1.0), [1.24, 2.13], [2, 3, 4, 5], 4),
    ("RF03 ", vio.pick_profile([p for p in profiles if p.flight == 'RF03'], 23.3),
     [1.24, 1.64, 2.13], [3, 4, 5, 6], 3),
    ("RF08 ", vio.pick_profile([p for p in profiles if p.flight == 'RF08'], 26.3),
     [1.24, 1.64, 2.13], [3, 4, 5, 6], 4),
]

out = []
t_all0 = time.time()
for label, truth, bands, ks, rmse_opt_k in CASES:
    prior_builder, fwd, s_ref, Se = build_case(truth, bands)

    # --- ONE-TIME: joint pool Jacobian on the full ODE-grid interior (+ r_base/tau_bot) ---
    t0 = time.time()
    cur_tau_bot = float(fwd.tau_bot)
    tau_pool = fwd.ode_grid(prior_builder(s_ref)[0], s_ref)
    s_pool = np.unique(np.clip(tau_pool / cur_tau_bot, 0.0, 1.0))
    s_int = s_pool[s_pool < 1.0 - 1e-6]                       # interior r_e nodes
    x_pool, Sa_pool = prior_builder(s_int)                    # joint first guess on the pool
    K_full = np.asarray(fwd.jacobian(x_pool, s_int))          # (m, n_int + 2)  ONE compile
    t_pooljac = time.time() - t0
    n_int = s_int.size

    # QRCP rank on the whitened r_e block (filter ranking), cloud-top always first
    w, V = np.linalg.eigh(Se)
    Se_half_inv = (V / np.sqrt(w)) @ V.T
    sig = np.sqrt(np.clip(np.diag(Sa_pool)[:n_int], 0.0, None))
    _, _, piv = qr(Se_half_inv @ K_full[:, :n_int] @ np.diag(sig), mode="economic", pivoting=True)
    top = int(np.argmin(s_int))
    order = [top] + [int(p) for p in piv if int(p) != top]   # cloud-top + QRCP order

    # --- PER-k: sub-select columns + determinant ratio (no Jacobian eval) ---
    t1 = time.time()
    rows = []
    for k in ks:
        idx = sorted(order[:k], key=lambda j: s_int[j])      # k r_e nodes, sorted by depth
        s_sel = s_int[idx]
        cols = list(idx) + [n_int, n_int + 1]                # r_e cols + r_base + tau_bot
        K_k = K_full[:, cols]
        _, Sa_k = prior_builder(s_sel)
        post = roe.posterior_diagnostics(K_k, Sa_k, Se)
        rows.append(dict(k=int(k), dofs=float(post.dofs), sic=float(post.sic)))
    t_loop = time.time() - t1

    sic_peak_k = max(rows, key=lambda r: r['sic'])['k']
    ok = "MATCH" if sic_peak_k == rmse_opt_k else "MISMATCH"
    print(f"\n{label} {truth.flight} tau_bot={truth.tau_bot:.1f}  (RMSE-optimal k={rmse_opt_k})",
          flush=True)
    for r in rows:
        print(f"  k={r['k']}: DOFS {r['dofs']:.2f}   SIC {r['sic']:.2f}", flush=True)
    print(f"  -> approx SIC peak k={sic_peak_k} [{ok}];  "
          f"pool-Jacobian (1 compile+eval) {t_pooljac:.1f}s, per-k loop {1e3 * t_loop:.1f} ms "
          f"for {len(ks)} k's", flush=True)
    out.append(dict(label=label.strip(), flight=truth.flight, rmse_opt_k=rmse_opt_k,
                    sic_peak_k=int(sic_peak_k), match=(sic_peak_k == rmse_opt_k),
                    t_pooljac_s=t_pooljac, t_loop_ms=1e3 * t_loop, rows=rows))

print(f"\nTOTAL wall-time (incl. Mie builds + forwards): {time.time() - t_all0:.0f}s", flush=True)
p = Path(__file__).resolve().parents[2] / "docs" / "approx_sic_sweep.json"
p.write_text(json.dumps(out, indent=2))
print(f"wrote {p}", flush=True)
