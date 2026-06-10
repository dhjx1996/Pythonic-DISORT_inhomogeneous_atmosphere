"""Smoke test: end-to-end r_e(τ) OE retrieval on a thin VOCALS-REx profile."""
import sys, time
from pathlib import Path
from math import pi

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import numpy as np
import jax.numpy as jnp

import vocals_io as vio
from miejax_lite import mie_legendre_precompute, build_re_table, select_channel
import retrieval_oe as roe

DD = "/home/jovyan/cloud_profile_retrieval/multispectral-retrieval-using-MODIS/VOCALS_REx_data"

t0 = time.perf_counter()
profs = vio.load_all_profiles(DD)
thin = vio.pick_profile(profs, 1.0)
print(f"profile {thin.flight} tau_bot={thin.tau_bot:.2f} n={thin.tau.size} "
      f"r_top={thin.r_top:.2f} r_base={thin.r_base:.2f}")

# --- optics table: single band for the smoke test ---------------------------
NQuad, NFourier, NLeg_all, v_eff = 16, 8, 128, 0.10
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
band = select_channel(build_re_table([2.13], 2.0, 25.0, 32, v_eff, precomp,
                                     n_radii=600), 0)
print(f"setup+table {time.perf_counter()-t0:.1f}s")

# --- geometry: nadir + two oblique views ------------------------------------
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.90, 0.65, 0.50])         # nadir-ish .. ~60deg (PACE/SPEXone envelope)
view_phi = np.array([pi, pi, pi])              # principal plane (backscatter)
BDRF = [[0.05 / pi]]                            # one band

fwd = roe.RetrievalForward(
    [band], NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=thin.tau_bot, r_base=thin.r_base,
    view_mu=view_mu, view_phi=view_phi, BDRF_bands=BDRF, NLeg_all=NLeg_all,
    NFourier=NFourier)

# reference state for mode selection: adiabatic first guess on a coarse grid
tau_ref = np.linspace(0.0, thin.tau_bot, 5)[:-1]               # 4 interior nodes
x_ref, _ = roe.make_adiabatic_prior(tau_ref, thin.tau_bot, thin.r_base,
                                    r_top_prior=thin.r_top)
t1 = time.perf_counter()
# Pick the azimuthal mode count from the noise floor (S_ε selector) up front, so
# the forward compiles once at the chosen K. Use a representative noise for
# selection (the retrieval's own Se is built from the OSSE y below).
Se_sel = (0.005 ** 2) * np.eye(fwd.m)
K = roe.select_num_modes(fwd, x_ref, tau_ref, Se_sel)
print(f"select_num_modes K={K}  ({time.perf_counter()-t1:.1f}s)")

# --- OSSE observation from the dense truth ----------------------------------
t1 = time.perf_counter()
y = roe.osse_observation(fwd, thin.tau, thin.r_e)
print(f"OSSE y (n={y.size}): {np.round(y,4)}  (forward compile {time.perf_counter()-t1:.1f}s)")

# --- retrieval grid by QRCP on the ODE pool ---------------------------------
t1 = time.perf_counter()
tau_sel, re_sel, info = roe.select_retrieval_grid(fwd, x_ref, tau_ref, k_active=4)
print(f"grid: pool={info['tau_pool'].size} nodes -> selected tau={np.round(tau_sel,3)} "
      f"({time.perf_counter()-t1:.1f}s)")

# --- prior on the selected grid ---------------------------------------------
x_a, Sa = roe.make_adiabatic_prior(tau_sel, thin.tau_bot, thin.r_base,
                                   r_top_prior=thin.r_top)
Se = np.diag(np.full(y.size, (0.01 * np.abs(y).max() + 1e-4) ** 2))   # ~1% radiometric

# --- Gauss-Newton OE with lagged re-meshing ---------------------------------
prior_builder = lambda tn: roe.make_adiabatic_prior(
    tn, thin.tau_bot, thin.r_base, r_top_prior=thin.r_top)
t1 = time.perf_counter()
res = roe.gauss_newton_oe(fwd, y, tau_sel, x_a, Sa, Se, n_iter=15, lm=1e-3,
                          n_outer=3, k_active=4, prior_builder=prior_builder)
print(f"GN+remesh: converged={res.converged} iters={len(res.cost_history)} "
      f"cost {res.cost_history[0]:.3e} -> {res.cost_history[-1]:.3e} "
      f"({time.perf_counter()-t1:.1f}s)")
print(f"  final grid tau = {np.round(res.tau_nodes,3)}  (was {np.round(tau_sel,3)})")
# rebuild prior on final grid for UQ (gauss_newton_oe already returns it)
x_a, Sa = res.x_a, res.Sa
print(f"  x_a       = {np.round(res.x_a,3)}")
print(f"  retrieved = {np.round(res.x,3)}")
truth_at_nodes = np.interp(res.tau_nodes, thin.tau, thin.r_e)
print(f"  truth     = {np.round(truth_at_nodes,3)}")

# --- UQ ---------------------------------------------------------------------
post = roe.posterior_diagnostics(res.K, res.Sa, res.Se)
print(f"UQ: DOFS={post.dofs:.2f}  error(1σ)={np.round(post.error,3)} µm")
print(f"  fit residual ‖y-F‖={np.linalg.norm(res.y-res.Fx):.2e}")
print(f"TOTAL {time.perf_counter()-t0:.1f}s")
