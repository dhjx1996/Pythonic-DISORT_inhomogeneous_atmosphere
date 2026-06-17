"""Forward-only positivity + S_ε mode-selection check at production stream count."""
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
thin = vio.pick_profile(vio.load_all_profiles(DD), 1.0)

NQuad, NFourier, NLeg_all, v_eff = 16, 16, 32, 0.10
precomp = mie_legendre_precompute(max_nstop=512, NLeg=NLeg_all)
band = select_channel(build_re_table([2.13], 2.0, 25.0, 32, v_eff, precomp, n_radii=600), 0)

mu0, I0, phi0 = 0.6, 1.0, 0.0
view_mu = np.array([0.9, 0.65, 0.4])
view_phi = np.array([pi, pi, pi])
fwd = roe.RetrievalForward([band], NQuad=NQuad, mu0=mu0, I0=I0, phi0=phi0,
    tau_bot=thin.tau_bot, r_base=thin.r_base, view_mu=view_mu, view_phi=view_phi,
    BDRF_bands=[[0.05]], NLeg_all=NLeg_all, NFourier=NFourier)

tau_ref = np.linspace(0.0, thin.tau_bot, 5)[:-1]
x_ref, _ = roe.make_adiabatic_prior(tau_ref, thin.tau_bot, thin.r_base, r_top_prior=thin.r_top)
t = time.perf_counter()
# Noise-aware azimuthal mode selection (replaces the old relative Cauchy test):
# a representative PACE-like 0.5%-reflectance 1σ noise floor.
Se = (0.005 ** 2) * np.eye(fwd.m)
K = roe.select_num_modes(fwd, x_ref, tau_ref, Se)
print(f"NQuad={NQuad} NFourier={NFourier} -> S_eps K={K}  saturated={K[0]>=NFourier}  ({time.perf_counter()-t:.0f}s)")
t = time.perf_counter()
y = np.asarray(fwd.forward(x_ref, tau_ref))
print(f"forward y (R) at view_mu={view_mu}: {np.round(y,4)}  all_positive={np.all(y>0)}  ({time.perf_counter()-t:.0f}s)")
# OSSE truth forward too
yt = roe.osse_observation(fwd, thin.tau, thin.r_e)
print(f"OSSE truth y: {np.round(yt,4)}  all_positive={np.all(yt>0)}")
