"""Characterize forward-reflectance sign over a (mu,phi) grid vs NQuad and band.

For the thin VOCALS profile, build the seam forward at a given NQuad for two
bands (2.13 = least forward-peaked, 1.24 = most), solve once per band, and print
the bidirectional reflectance R = pi u/(mu0 I0) over a grid of view zenith
(mu) x relative azimuth (phi). Tells us the usable multi-angle envelope.
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

NQuad = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NLeg_all = int(sys.argv[2]) if len(sys.argv) > 2 else 32
v_eff = 0.10
DD = "/home/jovyan/cloud_profile_retrieval/multispectral-retrieval-using-MODIS/VOCALS_REx_data"
thin = vio.pick_profile(vio.load_all_profiles(DD), 1.0)
print(f"profile {thin.flight} tau_bot={thin.tau_bot:.2f}  NQuad={NQuad}")

precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
table = build_re_table([1.24, 2.13], 2.0, 25.0, 32, v_eff, precomp, n_radii=600)

mu0, I0, phi0 = 0.6, 1.0, 0.0
BDRF = [[0.05 / pi]]
# truth profile nodes (interior; base anchored)
interior = thin.tau < thin.tau_bot - 1e-9
knots = jnp.asarray(np.r_[thin.tau[interior], thin.tau_bot])
vals = jnp.asarray(np.r_[thin.r_e[interior], thin.r_base])

mu_grid = np.array([0.95, 0.8, 0.65, 0.5, 0.4, 0.3])
phi_grid = np.array([0.0, pi / 2, pi])          # forward / cross / back
for bi, name in [(1, "2.13µm"), (0, "1.24µm")]:
    opt = select_channel(table, bi)
    om = lambda tau: table_lookup(opt, jnp.interp(tau, knots, vals))[0]
    leg = lambda tau: table_lookup(opt, jnp.interp(tau, knots, vals))[1]
    setup = riccati_setup(NQuad, I0, phi0, mu0, NLeg_all=NLeg_all,
                          BDRF_Fourier_modes=BDRF[0],
                          delta_M_scaling=True, NT_cor=True, tol=1e-3)
    K = setup.NFourier      # all modes — the actual compile-memory stressor
    t = time.perf_counter()
    res = riccati_solve(setup, om, leg, thin.tau_bot, num_modes=K)
    R = np.asarray(eval_radiance(setup, res, jnp.asarray(mu_grid),
                                 jnp.asarray(phi_grid))) * pi / (mu0 * I0)
    print(f"\n=== {name}  K={K}  ({time.perf_counter()-t:.0f}s) ===")
    print("  mu \\ phi |   0(fwd)   90(cross)  180(back)")
    for i, mu in enumerate(mu_grid):
        print(f"   {mu:.2f}    | " + "  ".join(f"{R[i,j]:+.4f}" for j in range(len(phi_grid))))
    print(f"  min R = {R.min():+.4f}  (negative ⇒ unphysical)")
