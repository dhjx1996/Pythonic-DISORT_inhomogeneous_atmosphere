"""Does capping NFourier keep the forward physical AND let jacrev compile (no OOM)?
Usage: check_jac.py [NFourier]"""
import sys, time
from pathlib import Path
from math import pi
_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
import numpy as np
import vocals_io as vio
from miejax_lite import mie_legendre_precompute, build_re_table, select_channel
import retrieval_oe as roe

NFourier = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NQuad, NLeg_all, v_eff = 16, 128, 0.10
DD = "/home/jovyan/cloud_profile_retrieval/multispectral-retrieval-using-MODIS/VOCALS_REx_data"
thin = vio.pick_profile(vio.load_all_profiles(DD), 1.0)
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
bands = [1.24, 2.13]
table = build_re_table(bands, 2.0, 25.0, 32, v_eff, precomp, n_radii=600)
opt_bands = [select_channel(table, i) for i in range(len(bands))]
mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu, view_phi = np.array([0.90, 0.65, 0.50]), np.array([pi, pi, pi])
fwd = roe.RetrievalForward(opt_bands, NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=thin.tau_bot, r_base=thin.r_base, view_mu=view_mu, view_phi=view_phi,
    BDRF_bands=[[0.05]]*len(bands), NLeg_all=NLeg_all, NFourier=NFourier)
print(f"{len(bands)} bands x {view_mu.size} angles = {fwd.m} obs")
tau_ref = np.linspace(0.0, thin.tau_bot, 5)[:-1]
x_ref, _ = roe.make_adiabatic_prior(tau_ref, thin.tau_bot, thin.r_base, r_top_prior=thin.r_top)
# Keep the FULL NFourier (no select_num_modes truncation): this script's job is
# to prove the jacrev compiles at the full mode count — the OOM stressor (§H).
print(f"NFourier={NFourier} -> K_list={fwd.K_list} (all modes; OOM stress test)")
t = time.perf_counter(); y = np.asarray(fwd.forward(x_ref, tau_ref))
print(f"forward y={np.round(y,4)} positive={np.all(y>0)} ({time.perf_counter()-t:.0f}s)")
t = time.perf_counter()
try:
    J = np.asarray(fwd.jacobian(x_ref, tau_ref))   # GN Jacobian (small p), jacrev
    print(f"GN jacrev OK shape={J.shape} finite={np.all(np.isfinite(J))} ({time.perf_counter()-t:.0f}s)")
except Exception as e:
    print(f"GN jacrev FAILED: {type(e).__name__}: {str(e)[:120]}")
t = time.perf_counter()
try:
    Kp = fwd.jacobian_on_grid(x_ref, tau_ref)      # pool Jacobian, jacrev
    print(f"pool jacrev OK shape={Kp.shape} ({time.perf_counter()-t:.0f}s)")
except Exception as e:
    print(f"pool jacrev FAILED: {type(e).__name__}: {str(e)[:120]}")
