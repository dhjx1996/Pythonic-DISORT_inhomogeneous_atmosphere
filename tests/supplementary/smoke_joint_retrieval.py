"""Smoke test for the JOINT r_e(τ) + r_base + τ_bot retrieval refactor.

Cheap (NQuad=6, 1 band, 1 view) plumbing check for the PO refactor of
``retrieval_oe`` — it does NOT run a full GN retrieval (that is the expensive
background job). It asserts:

  1. leave-one-flight-out climatology is computable and leak-free;
  2. ``make_joint_prior`` / ``make_climatology_prior`` return the (k + n_extra)
     joint state + SPD covariance;
  3. ``_split_state`` decodes the joint vector correctly;
  4. the joint forward at the TRUTH state == the legacy fixed-anchor forward at
     the truth profile (bit-for-bit-ish — validates the decode is transparent);
  5. ``∂y/∂r_base`` and ``∂y/∂τ_bot`` are finite and non-zero (the new unknowns
     actually flow through autodiff).

Run:  /tmp/jaxve/bin/python tests/supplementary/smoke_joint_retrieval.py
"""
import sys
import time
from math import pi
from pathlib import Path

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))

import numpy as np

import vocals_io as vio
from miejax_lite import build_re_table, mie_legendre_precompute, select_channel
import retrieval_oe as roe

DD = ("/home/jovyan/cloud_profile_retrieval/"
      "multispectral-retrieval-using-MODIS/VOCALS_REx_data")

t0 = time.perf_counter()
profs = vio.load_all_profiles(DD)
thin = vio.pick_profile(profs, 1.0)
print(f"profile {thin.flight} tau_bot={thin.tau_bot:.3f} n={thin.tau.size} "
      f"r_top={thin.r_top:.2f} r_base={thin.r_base:.2f}")

# --- (1) leave-one-flight-out climatology -----------------------------------
clim = vio.vocals_climatology(profs, exclude_flight=thin.flight)
assert thin.flight not in clim["flights"], "LEAK: target flight in climatology"
print(f"[1] LOO climatology (n={clim['n']}, excl {thin.flight}): "
      f"r_top={clim['r_top_mean']:.2f}±{clim['r_top_std']:.2f}, "
      f"r_base={clim['r_base_mean']:.2f}±{clim['r_base_std']:.2f}, "
      f"tau_bot={clim['tau_bot_mean']:.2f}±{clim['tau_bot_std']:.2f}")

# --- optics + cheap geometry -------------------------------------------------
NQuad, NFourier, NLeg_all, v_eff = 6, 4, 128, 0.10
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
band = select_channel(build_re_table([2.13], 2.0, 25.0, 32, v_eff, precomp,
                                     n_radii=600), 0)
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.65])
view_phi = np.array([pi])
BDRF = [[0.06]]
print(f"setup+table {time.perf_counter()-t0:.1f}s")

# coarse retrieval grid in NORMALIZED depth s=τ/τ_bot (3 free nodes incl. top s=0)
s_nodes = np.linspace(0.0, 1.0, 4)[:-1]
k = s_nodes.size

# --- (2) joint priors --------------------------------------------------------
x_broad, Sa_broad = roe.make_joint_prior(
    s_nodes, tau_bot_prior=clim["tau_bot_mean"], r_top_prior=10.0,
    r_base_prior=12.0, sigma_tau_bot=0.5 * clim["tau_bot_mean"])
x_clim, Sa_clim = roe.make_climatology_prior(s_nodes, clim)
assert x_broad.shape == (k + 2,) and Sa_broad.shape == (k + 2, k + 2)
assert x_clim.shape == (k + 2,) and Sa_clim.shape == (k + 2, k + 2)
assert np.all(np.linalg.eigvalsh(Sa_broad) > 0), "broad Sa not SPD"
assert np.all(np.linalg.eigvalsh(Sa_clim) > 0), "clim Sa not SPD"
print(f"[2] joint prior shapes OK: state dim {k}+2={k+2}; "
      f"x_broad={np.round(x_broad,2)}")

# --- joint forward: prior τ_bot / r_base (leak-free first guess) -------------
fwd = roe.RetrievalForward(
    [band], NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=clim["tau_bot_mean"], r_base=clim["r_base_mean"],
    view_mu=view_mu, view_phi=view_phi, BDRF_bands=BDRF, NLeg_all=NLeg_all,
    NFourier=NFourier, retrieve_tau_bot=True, retrieve_r_base=True)

# --- (3) _split_state decode -------------------------------------------------
x_test = np.array([9.0, 10.0, 11.0, 12.5, 7.3])     # [3 nodes, r_base, tau_bot]
rn, rb, tb = fwd._split_state(x_test, s_nodes)       # float32 in the venv
assert np.allclose(np.asarray(rn), [9.0, 10.0, 11.0], atol=1e-4)
assert np.isclose(float(rb), 12.5, atol=1e-4) and np.isclose(float(tb), 7.3, atol=1e-4)
print(f"[3] _split_state OK: r_nodes={np.asarray(rn)}, r_base={float(rb)}, "
      f"tau_bot={float(tb)}")

# --- (4) joint-at-truth == legacy fixed-anchor-at-truth ----------------------
t1 = time.perf_counter()
y_joint = roe.osse_observation(fwd, thin.tau, thin.r_e)     # joint truth state
print(f"    joint forward compile+run {time.perf_counter()-t1:.1f}s")
fwd_fix = roe.RetrievalForward(
    [band], NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=thin.tau_bot, r_base=thin.r_base,
    view_mu=view_mu, view_phi=view_phi, BDRF_bands=BDRF, NLeg_all=NLeg_all,
    NFourier=NFourier)                                       # legacy fixed-anchor
y_fix = roe.osse_observation(fwd_fix, thin.tau, thin.r_e)
rel = np.abs(y_joint - y_fix) / np.abs(y_fix)
# float32 solver-parity tolerance: the joint (traced τ_bot) and fixed (constant
# τ_bot) paths compile the τ/τ_bot division differently ⇒ adaptive-step noise at
# the solver's float32 tol≈1e-3 level (DESIGN §4); not bit-identical like the old
# absolute-τ forward, but equal to within solver tolerance.
assert np.all(rel < 2e-3), f"joint != fixed at truth beyond solver tol: rel={rel}"
print(f"[4] joint-at-truth == fixed-anchor (float32 solver tol): "
      f"y={np.round(y_joint,5)}, max rel diff {rel.max():.2e}")

# --- (5) gradient flows to r_base and τ_bot ----------------------------------
t2 = time.perf_counter()
K = np.asarray(fwd.jacobian(x_broad, s_nodes))               # (m, k+2)
print(f"    jacobian compile+run {time.perf_counter()-t2:.1f}s")
assert K.shape == (fwd.m, k + 2)
col_rbase, col_taubot = K[:, k], K[:, k + 1]
assert np.all(np.isfinite(K)), "non-finite Jacobian"
assert np.linalg.norm(col_rbase) > 0, "∂y/∂r_base is zero"
assert np.linalg.norm(col_taubot) > 0, "∂y/∂τ_bot is zero"
print(f"[5] gradient flow OK: ||∂y/∂r_base||={np.linalg.norm(col_rbase):.3e}, "
      f"||∂y/∂τ_bot||={np.linalg.norm(col_taubot):.3e}")

# --- DOFS decomposition smoke (uses the broad-prior Jacobian) ----------------
y_broad = fwd.forward(x_broad, s_nodes)
Se_d = np.diag((0.03 * np.maximum(np.abs(y_broad), 0.02)) ** 2)
post = roe.posterior_diagnostics(K, Sa_broad, Se_d)
dby = roe.dofs_by_component(post, k, retrieve_r_base=True, retrieve_tau_bot=True)
print(f"[+] DOFS={post.dofs:.2f}  profile={dby['profile']:.2f} "
      f"r_base={dby['r_base']:.2f} tau_bot={dby['tau_bot']:.2f}")

# --- (6) GN CONVERGENCE from a far first guess (the normalized-depth + clamp fix)
# Thin truth τ_bot=1.21 but the climatology first guess is τ_bot≈10.6 — exactly the
# case the absolute-τ parameterisation could not retrieve (nodes stranded below the
# thin base ⇒ crossing / max-solver-steps). In normalized s the nodes track τ_bot.
t3 = time.perf_counter()
Se_obs = np.diag((0.03 * np.maximum(np.abs(y_joint), 0.02)) ** 2)
res = roe.gauss_newton_oe(fwd, y_joint, s_nodes, x_broad, Sa_broad, Se_obs,
                          n_iter=8, lm=1e-2, xtol=2e-3, n_outer=1)
tb_ret = float(res.x[k + 1])
assert np.all(np.isfinite(res.x)), "GN produced non-finite state"
assert res.cost_history[-1] <= res.cost_history[0], "GN did not reduce cost"
assert tb_ret > 0, f"τ_bot went unphysical: {tb_ret}"
# τ_bot should head DOWN from the thick first guess (10.6) toward the thin truth (1.21)
assert tb_ret < clim["tau_bot_mean"], (
    f"τ_bot did not move toward thin truth: {tb_ret} vs prior {clim['tau_bot_mean']:.1f}")
print(f"[6] GN converges from far guess [{time.perf_counter()-t3:.1f}s]: "
      f"τ_bot {clim['tau_bot_mean']:.1f}(prior)->{tb_ret:.2f}(ret), truth {thin.tau_bot:.2f}; "
      f"cost {res.cost_history[0]:.2e}->{res.cost_history[-1]:.2e}, finite & physical")
print(f"\nALL JOINT-REFACTOR SMOKE CHECKS PASSED ({time.perf_counter()-t0:.0f}s)")
