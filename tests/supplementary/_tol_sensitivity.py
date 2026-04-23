"""Run pydisort_riccati_jax at a single tolerance, save results to .npz.
Usage: python _tol_sensitivity.py <tol> <outfile>
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from math import pi
from pydisort_riccati_jax import pydisort_riccati_jax

tol = float(sys.argv[1])
outfile = sys.argv[2]

tau_bot = 30
omega_top, omega_bot = 0.85, 0.96
g_top, g_bot = 0.865, 0.820
NQuad = 16; NLeg = NQuad
mu0, I0, phi0 = 0.5, 1.0, 0.0

tau_spike = 15.0; sigma_w = 0.5
delta_omega = -0.15; delta_g = 0.04

omega_func = lambda tau: (omega_top + (omega_bot - omega_top) * tau / tau_bot
    + delta_omega * jnp.exp(-0.5 * ((tau - tau_spike) / sigma_w) ** 2))
g_func = lambda tau: (g_top + (g_bot - g_top) * tau / tau_bot
    + delta_g * jnp.exp(-0.5 * ((tau - tau_spike) / sigma_w) ** 2))
Leg_coeffs_func = lambda tau: g_func(tau) ** jnp.arange(NLeg)

phi_test = np.array([0.0, pi/4, pi/2, pi])

mu_arr, flux_up, u0, u_func, tau_grid = pydisort_riccati_jax(
    tau_bot, omega_func, Leg_coeffs_func, NQuad, mu0, I0, phi0, tol=tol)

u_phi = u_func(phi_test)

np.savez(outfile, tol=tol, tau_grid=tau_grid, u0=u0, u_phi=u_phi,
         mu_arr=mu_arr, flux_up=flux_up)
