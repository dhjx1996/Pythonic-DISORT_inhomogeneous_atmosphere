"""Isolate barycentric-interpolation error: evaluate radiance AT the quadrature
nodes (no interpolation) vs OFF-node, for the thin Mie cloud.

Sta1982: discrete-ordinate intensity is accurate only at the quadrature angles;
standard interpolation to other angles oscillates. If at-node R is smooth/physical
and off-node is erratic, μ-interpolation (not TMS/diffuse) is the culprit.
Usage: check_atnode.py [NQuad] [NLeg_all]
"""
import sys, time
from pathlib import Path
from math import pi
_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
import numpy as np
import jax.numpy as jnp
import vocals_io as vio
from miejax_lite import mie_legendre_precompute, build_re_table, select_channel, table_lookup
from pydisort_riccati_jax import riccati_setup, riccati_solve, eval_radiance

NQuad = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NLeg_all = int(sys.argv[2]) if len(sys.argv) > 2 else 32
v_eff = 0.10
DD = "/home/jovyan/cloud_profile_retrieval/multispectral-retrieval-using-MODIS/VOCALS_REx_data"
thin = vio.pick_profile(vio.load_all_profiles(DD), 1.0)
print(f"profile {thin.flight} tau_bot={thin.tau_bot:.2f}  NQuad={NQuad} NLeg_all={NLeg_all}")

precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
table = build_re_table([1.24, 2.13], 2.0, 25.0, 32, v_eff, precomp, n_radii=600)
mu0, I0, phi0 = 0.6, 1.0, 0.0
BDRF = [0.05 / pi]
interior = thin.tau < thin.tau_bot - 1e-9
knots = jnp.asarray(np.r_[thin.tau[interior], thin.tau_bot])
vals = jnp.asarray(np.r_[thin.r_e[interior], thin.r_base])
phi_grid = np.array([0.0, pi / 2, pi])

for bi, name in [(1, "2.13µm"), (0, "1.24µm")]:
    opt = select_channel(table, bi)
    om = lambda tau: table_lookup(opt, jnp.interp(tau, knots, vals))[0]
    leg = lambda tau: table_lookup(opt, jnp.interp(tau, knots, vals))[1]
    setup = riccati_setup(NQuad, I0, phi0, mu0, NLeg_all=NLeg_all,
                          BDRF_Fourier_modes=BDRF,
                          delta_M_scaling=True, NT_cor=True, tol=1e-3)
    mu_nodes = np.asarray(setup.mu_nodes)            # the exact quadrature angles
    K = setup.NFourier
    t = time.perf_counter()
    res = riccati_solve(setup, om, leg, thin.tau_bot, num_modes=K)
    R_node = np.asarray(eval_radiance(setup, res, jnp.asarray(mu_nodes), jnp.asarray(phi_grid))) * pi / (mu0 * I0)
    # off-node: midpoints between adjacent nodes (worst case for interpolation)
    mu_mid = 0.5 * (mu_nodes[:-1] + mu_nodes[1:])
    R_mid = np.asarray(eval_radiance(setup, res, jnp.asarray(mu_mid), jnp.asarray(phi_grid))) * pi / (mu0 * I0)
    print(f"\n=== {name}  K={K}  ({time.perf_counter()-t:.0f}s) ===")
    print("  AT NODES (no interp):   mu        fwd      cross     back")
    for i, mu in enumerate(mu_nodes):
        print(f"    {mu:.3f}   | " + "  ".join(f"{R_node[i,j]:+.4f}" for j in range(3)))
    print(f"  at-node min R = {R_node.min():+.4f}   smooth_in_mu(nadir col back)? span={R_node[:,2].min():+.3f}..{R_node[:,2].max():+.3f}")
    print("  OFF-NODE (midpoints, interpolated):")
    for i, mu in enumerate(mu_mid):
        print(f"    {mu:.3f}   | " + "  ".join(f"{R_mid[i,j]:+.4f}" for j in range(3)))
    print(f"  off-node min R = {R_mid.min():+.4f}")
