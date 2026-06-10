"""Thick-cloud r_e(τ) OE retrieval on the RF05 VOCALS-REx profile (τ_bot≈12).

Companion to smoke_retrieval.py (thin τ≈1.2). Exercises the ceiling-lifted solver
(scan-over-modes + S_ε selector) at depth: 3-band absorption ladder × multi-angle,
adiabatic prior, lagged re-meshing. Prints the deep-node prior-dominance diagnostic
(per-node data-vs-prior variance split) the notebook section will present.

Usage:  JAX_PLATFORMS=cpu python supplementary/thick_retrieval.py \
            [k_active=5] [target_tau=12] [flight=any]
"""
import sys, time
from pathlib import Path
from math import pi

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import vocals_io as vio
from miejax_lite import mie_legendre_precompute, build_re_table, select_channel
import retrieval_oe as roe

DD = "/home/jovyan/cloud_profile_retrieval/multispectral-retrieval-using-MODIS/VOCALS_REx_data"
K_ACTIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TARGET_TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
FLIGHT = sys.argv[3] if len(sys.argv) > 3 else None
N_OUTER = int(sys.argv[4]) if len(sys.argv) > 4 else 2

t0 = time.perf_counter()
profs = vio.load_all_profiles(DD)
if FLIGHT is not None:
    profs = [p for p in profs if p.flight == FLIGHT]
thick = vio.pick_profile(profs, TARGET_TAU)
print(f"profile {thick.flight} tau_bot={thick.tau_bot:.2f} n={thick.tau.size} "
      f"r_top={thick.r_top:.2f} r_base={thick.r_base:.2f} "
      f"re[{thick.r_e.min():.1f},{thick.r_e.max():.1f}]")

# --- optics: 3-band absorption ladder ---------------------------------------
bands = [1.24, 1.64, 2.13]                 # weak -> strong water absorption
NQuad, NFourier, NLeg_all, v_eff = 16, 8, 128, 0.10
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
table = build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600)
opt_bands = [select_channel(table, i) for i in range(len(bands))]
print(f"setup+table {time.perf_counter()-t0:.1f}s  (bands={bands})")

# --- geometry: nadir-ish + oblique back-scatter views -----------------------
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50])
view_phi = np.array([pi, pi, pi])              # principal plane, back-scatter
BDRF = [[0.05 / pi]] * len(bands)              # dark ocean, per band

fwd = roe.RetrievalForward(
    opt_bands, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=thick.tau_bot, r_base=thick.r_base,
    view_mu=view_mu, view_phi=view_phi, BDRF_bands=BDRF, NLeg_all=NLeg_all,
    NFourier=NFourier)
print(f"{len(bands)} bands x {view_mu.size} angles = {fwd.m} obs")

# --- S_ε mode selection (thick cloud -> few azimuthal modes) -----------------
tau_ref = np.linspace(0.0, thick.tau_bot, 6)[:-1]
x_ref, _ = roe.make_adiabatic_prior(tau_ref, thick.tau_bot, thick.r_base,
                                    r_top_prior=thick.r_top)
t1 = time.perf_counter()
Se_sel = (0.005 ** 2) * np.eye(fwd.m)
K = roe.select_num_modes(fwd, x_ref, tau_ref, Se_sel)
print(f"select_num_modes K={K}  ({time.perf_counter()-t1:.1f}s)")

# --- OSSE observation -------------------------------------------------------
t1 = time.perf_counter()
y = roe.osse_observation(fwd, thick.tau, thick.r_e)
print(f"OSSE y ({fwd.n_bands}x{fwd.n_view}):\n{np.round(y.reshape(fwd.n_bands, fwd.n_view),4)}"
      f"  (forward compile {time.perf_counter()-t1:.1f}s)")
assert np.all(y > 0), "unphysical negative reflectance"

# --- retrieval grid ---------------------------------------------------------
t1 = time.perf_counter()
tau_sel, re_sel, info = roe.select_retrieval_grid(fwd, x_ref, tau_ref,
                                                  k_active=K_ACTIVE)
print(f"grid: pool={info['tau_pool'].size} -> {K_ACTIVE} selected "
      f"tau={np.round(tau_sel,3)} ({time.perf_counter()-t1:.1f}s)")

# --- prior + observation error ----------------------------------------------
prior_builder = lambda tn: roe.make_adiabatic_prior(
    tn, thick.tau_bot, thick.r_base, r_top_prior=thick.r_top,
    sigma_top=3.0, sigma_base=10.0)
x_a, Sa = prior_builder(tau_sel)
sigma_obs = 0.03 * np.maximum(np.abs(y), 0.02)     # ~3% radiometric + floor
Se = np.diag(sigma_obs ** 2)

# --- Gauss-Newton OE with lagged re-meshing ---------------------------------
t1 = time.perf_counter()
# xtol=5e-3: the near-adiabatic prior starts GN already near the optimum, so the
# meaningful convergence scale is ~0.05 µm (errors are 0.5–3 µm), not the thin
# section's tighter 2e-3 (which here would just chase negligible <0.05 µm steps).
res = roe.gauss_newton_oe(fwd, y, tau_sel, x_a, Sa, Se, n_iter=15, lm=1e-2,
                          xtol=5e-3, n_outer=N_OUTER, k_active=K_ACTIVE,
                          prior_builder=prior_builder)
print(f"GN+remesh: converged={res.converged} iters={len(res.cost_history)} "
      f"cost {res.cost_history[0]:.3e} -> {res.cost_history[-1]:.3e} "
      f"({time.perf_counter()-t1:.1f}s)")
print(f"  final grid tau = {np.round(res.tau_nodes,3)}")
truth_at = np.interp(res.tau_nodes, thick.tau, thick.r_e)
print(f"  x_a       = {np.round(res.x_a,3)}")
print(f"  retrieved = {np.round(res.x,3)}")
print(f"  truth     = {np.round(truth_at,3)}")

# --- UQ + three candidate per-node "measurement vs prior" metrics -----------
post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
A_diag = np.diag(post.A)                              # sums to DOFS exactly
A_rowsum = post.A.sum(axis=1)                         # weight on truth in x̂=Ax+(I-A)xa
var_red = post.data_fraction                          # 1 - Ŝ_ii/Sa_ii
print(f"UQ: DOFS={post.dofs:.2f}  error(1σ)={np.round(post.error,3)} µm")
print(f"  A_ii   (diag, ΣA_ii=DOFS) = {np.round(A_diag,2)}  (sum {A_diag.sum():.2f})")
print(f"  A rowsum (value weight)   = {np.round(A_rowsum,2)}")
print(f"  var.red. 1-Ŝ_ii/Sa_ii     = {np.round(var_red,2)}")
print(f"  fit residual ‖y-F‖={np.linalg.norm(res.y-res.Fx):.2e}")
print(f"TOTAL {time.perf_counter()-t0:.1f}s")

# --- save artifact (companion to docs/retrieval_baseline_linear_class.json) --
import json
out = dict(
    profile=f"{thick.flight} thick marine Sc, tau_bot={thick.tau_bot:.2f}",
    function_class="re5-linear (adiabatic)", date="2026-06-10",
    config=dict(NQuad=NQuad, NFourier=NFourier, NLeg_all=NLeg_all,
                bands_um=bands, view_mu=view_mu.tolist(),
                view_phi_pi=[1, 1, 1], mu0=mu0, k_active=K_ACTIVE, n_outer=N_OUTER,
                select_num_modes_K=K, prior="adiabatic + Bayesian-Tikhonov"),
    results=dict(
        y=np.round(y, 4).tolist(),
        final_grid_tau=np.round(res.tau_nodes, 3).tolist(),
        retrieved=np.round(res.x, 3).tolist(),
        truth_at_nodes=np.round(truth_at, 3).tolist(),
        prior_xa=np.round(res.x_a, 3).tolist(),
        DOFS=round(float(post.dofs), 3),
        error_1sigma_um=np.round(post.error, 3).tolist(),
        A_diag=np.round(A_diag, 3).tolist(),
        A_rowsum=np.round(A_rowsum, 3).tolist(),
        var_reduction=np.round(var_red, 3).tolist(),
        converged=bool(res.converged),
        cost0=float(res.cost_history[0]), cost1=float(res.cost_history[-1]),
        resid_norm=float(np.linalg.norm(res.y - res.Fx))))
jpath = (Path(__file__).resolve().parents[2] / "docs"
         / f"retrieval_thick_{thick.flight}_tau{int(round(thick.tau_bot))}.json")
jpath.write_text(json.dumps(out, indent=2))
print(f"saved {jpath}")
