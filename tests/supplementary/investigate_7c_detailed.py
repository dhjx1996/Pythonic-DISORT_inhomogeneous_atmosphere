"""
Detailed investigation of test_7c BDRF discrepancy.

Extracts Riccati intermediate operators (R_up, T_up, s_up, R_down, T_down, s_down)
for m=0 and manually reconstructs the BC solve to pinpoint where the offset enters.
"""
import sys
from pathlib import Path
from math import pi
import numpy as np

_tests_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_tests_dir.parent / "src"))
sys.path.insert(0, str(_tests_dir))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PythonicDISORT import subroutines
from PythonicDISORT.pydisort import pydisort
from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
    _riccati_forward_jax,
    _riccati_backward_jax,
)

# ---- Problem setup (test_7c) ----
NQuad = 8
NLeg = NQuad
NFourier = NQuad
N = NQuad // 2
tau_bot = 1.0
mu0, I0, phi0 = 0.5, 1.0, 0.0
rho = 0.3
BDRF = [rho / pi]
omega_func = lambda tau: 0.90 + (0.60 - 0.90) * tau / tau_bot
g_func = lambda tau: 0.70 + (0.30 - 0.70) * tau / tau_bot

def Leg_coeffs_func(tau):
    g = g_func(tau)
    return g ** np.arange(NLeg)

# ---- Quadrature ----
mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos_jax = jnp.array(mu_arr_pos)
W_jax = jnp.array(W)
M_inv = 1.0 / mu_arr_pos_jax

# ---- Rescaling ----
rescale_factor = np.max((I0, 0.0, 0.0))  # b_pos=0, b_neg=0
I0_scaled = I0 / rescale_factor  # = 1.0
I0_div_4pi = I0_scaled / (4 * pi)

print("=" * 70)
print("SETUP")
print("=" * 70)
print(f"mu_arr_pos = {mu_arr_pos}")
print(f"W          = {W}")
print(f"I0_div_4pi = {I0_div_4pi:.10e}")
print(f"BDRF[0]    = {BDRF[0]:.10e}")

# ====================================================================
# PART 1: Riccati m=0 operators (at very tight tolerance)
# ====================================================================
print("\n" + "=" * 70)
print("PART 1: Riccati m=0 operators (tol=1e-10)")
print("=" * 70)

m = 0
tol = 1e-10
m_equals_0 = True

leg_data = _precompute_legendre(m, NLeg, mu_arr_pos_jax, mu0)
alpha_func, beta_func = _make_alpha_beta_funcs_jax(
    omega_func, Leg_coeffs_func, m, leg_data,
    mu_arr_pos_jax, W_jax, M_inv, N,
)
q_up, q_down = _make_q_funcs_jax(
    omega_func, Leg_coeffs_func, m, leg_data,
    mu_arr_pos_jax, M_inv, mu0, I0_div_4pi, m_equals_0, N,
)

# Forward sweep: R_up, T_up, s_up
R_up, T_up, s_up, tau_grid = _riccati_forward_jax(
    alpha_func, beta_func, tau_bot, N, tol,
    q_up_func=q_up, q_down_func=q_down,
)

# Backward sweep: R_down, T_down, s_down
R_down, T_down, s_down, _ = _riccati_backward_jax(
    alpha_func, beta_func, tau_bot, N, tol,
    q_up_func=q_up, q_down_func=q_down,
)

print(f"Forward steps:  {len(tau_grid)-1}")
print(f"||R_up||  = {float(jnp.linalg.norm(R_up)):.6e}")
print(f"||T_up||  = {float(jnp.linalg.norm(T_up)):.6e}")
print(f"||s_up||  = {float(jnp.linalg.norm(s_up)):.6e}")
print(f"s_up      = {np.array(s_up)}")
print(f"||R_down||= {float(jnp.linalg.norm(R_down)):.6e}")
print(f"||T_down||= {float(jnp.linalg.norm(T_down)):.6e}")
print(f"||s_down||= {float(jnp.linalg.norm(s_down)):.6e}")
print(f"s_down    = {np.array(s_down)}")

# ====================================================================
# PART 2: Manual BC solve WITH BDRF
# ====================================================================
print("\n" + "=" * 70)
print("PART 2: Manual BC solve WITH BDRF (m=0)")
print("=" * 70)

BDRF_mode = BDRF[0]  # rho/pi (scalar)
mu_arr_pos_times_W = mu_arr_pos_jax * W_jax

R_surf = (1 + int(m_equals_0)) * BDRF_mode * mu_arr_pos_times_W[None, :]
print(f"R_surf (m=0):\n{np.array(R_surf)}")

# Beam surface term
mathscr_X_pos = (mu0 * I0_div_4pi * 4) * BDRF_mode * jnp.ones(N)
beam_surface_term = mathscr_X_pos * jnp.exp(-tau_bot / mu0)
print(f"mathscr_X_pos = {np.array(mathscr_X_pos)}")
print(f"beam_surface_term = {np.array(beam_surface_term)}")

# b_pos=0, b_neg=0
b_pos_m = jnp.zeros(N)
b_neg_m = jnp.zeros(N)
b_pos_eff = b_pos_m + beam_surface_term

# BC system
LHS = jnp.eye(N) - R_surf @ R_down
RHS_val = R_surf @ (T_down @ b_neg_m + s_down) + b_pos_eff
print(f"\nR_surf @ s_down     = {np.array(R_surf @ s_down)}")
print(f"b_pos_eff           = {np.array(b_pos_eff)}")
print(f"Total RHS           = {np.array(RHS_val)}")

I_plus_bot = jnp.linalg.solve(LHS, RHS_val)
print(f"I_plus_bot          = {np.array(I_plus_bot)}")

I_plus_top_bdrf = (R_up @ b_neg_m + T_up @ I_plus_bot + s_up).real
print(f"I_plus_top (m=0, BDRF) = {np.array(I_plus_top_bdrf)}")

# ====================================================================
# PART 3: Manual BC solve WITHOUT BDRF
# ====================================================================
print("\n" + "=" * 70)
print("PART 3: Manual BC solve WITHOUT BDRF (m=0)")
print("=" * 70)

LHS_nb = jnp.eye(N)
RHS_nb = b_pos_m  # = 0
I_plus_bot_nb = RHS_nb  # = 0
I_plus_top_no_bdrf = (T_up @ I_plus_bot_nb + s_up).real
print(f"I_plus_top (m=0, no BDRF) = {np.array(I_plus_top_no_bdrf)}")
print(f"BDRF contribution (m=0)   = {np.array(I_plus_top_bdrf - I_plus_top_no_bdrf)}")

# ====================================================================
# PART 4: Multilayer pydisort reference (5000 layers)
# ====================================================================
print("\n" + "=" * 70)
print("PART 4: Multilayer pydisort reference (NL=5000)")
print("=" * 70)

NLayers = 5000
edges = np.linspace(0, tau_bot, NLayers + 1)
mids = 0.5 * (edges[:-1] + edges[1:])
tau_arr = edges[1:]
omega_arr = np.array([omega_func(t) for t in mids])
Leg_arr = np.array([Leg_coeffs_func(t) for t in mids])

# WITH BDRF
mu_arr, Fp, Fm, u0f, uf = pydisort(
    tau_arr, omega_arr, NQuad,
    Leg_arr, float(mu0), float(I0), float(phi0),
    NLeg=NLeg, NFourier=NQuad,
    only_flux=False,
    b_pos=0, b_neg=0,
    BDRF_Fourier_modes=list(BDRF),
)
# Evaluate at tau=0 for each phi, extract upwelling m=0 mode
# The u0f function returns the zeroth Fourier mode
u0_pydisort_bdrf = u0f(0)[:N]
print(f"u0_pydisort (m=0, BDRF) = {u0_pydisort_bdrf}")

# WITHOUT BDRF
mu_arr_nb, Fp_nb, Fm_nb, u0f_nb, uf_nb = pydisort(
    tau_arr, omega_arr, NQuad,
    Leg_arr, float(mu0), float(I0), float(phi0),
    NLeg=NLeg, NFourier=NQuad,
    only_flux=False,
    b_pos=0, b_neg=0,
    BDRF_Fourier_modes=[],
)
u0_pydisort_no_bdrf = u0f_nb(0)[:N]
print(f"u0_pydisort (m=0, no BDRF) = {u0_pydisort_no_bdrf}")
print(f"BDRF contribution (m=0) = {u0_pydisort_bdrf - u0_pydisort_no_bdrf}")

# ====================================================================
# PART 5: Comparison
# ====================================================================
print("\n" + "=" * 70)
print("PART 5: Comparison (m=0 Fourier mode)")
print("=" * 70)

ric_m0 = np.array(I_plus_top_bdrf) * rescale_factor  # undo rescaling
pyd_m0 = u0_pydisort_bdrf

diff = ric_m0 - pyd_m0
scale = max(float(np.max(np.abs(pyd_m0))), 1e-8)
print(f"Riccati m=0  = {ric_m0}")
print(f"pydisort m=0 = {pyd_m0}")
print(f"Diff (abs)   = {diff}")
print(f"Diff (rel)   = {diff / scale}")
print(f"Max |diff|   = {float(np.max(np.abs(diff))):.6e}")
print(f"Max rel diff = {float(np.max(np.abs(diff))) / scale:.6e}")

# Compare no-BDRF m=0 modes
ric_m0_nb = np.array(I_plus_top_no_bdrf) * rescale_factor
pyd_m0_nb = u0_pydisort_no_bdrf
diff_nb = ric_m0_nb - pyd_m0_nb
scale_nb = max(float(np.max(np.abs(pyd_m0_nb))), 1e-8)
print(f"\n--- Without BDRF ---")
print(f"Riccati m=0  = {ric_m0_nb}")
print(f"pydisort m=0 = {pyd_m0_nb}")
print(f"Max |diff|   = {float(np.max(np.abs(diff_nb))):.6e}")
print(f"Max rel diff = {float(np.max(np.abs(diff_nb))) / scale_nb:.6e}")

# Compare BDRF-only contributions
ric_bdrf_contrib = ric_m0 - ric_m0_nb
pyd_bdrf_contrib = pyd_m0 - pyd_m0_nb
bdrf_diff = ric_bdrf_contrib - pyd_bdrf_contrib
scale_bdrf = max(float(np.max(np.abs(pyd_bdrf_contrib))), 1e-8)
print(f"\n--- BDRF contribution only ---")
print(f"Riccati BDRF contrib  = {ric_bdrf_contrib}")
print(f"pydisort BDRF contrib = {pyd_bdrf_contrib}")
print(f"Diff                  = {bdrf_diff}")
print(f"Max |diff|            = {float(np.max(np.abs(bdrf_diff))):.6e}")
print(f"Max rel diff          = {float(np.max(np.abs(bdrf_diff))) / scale_bdrf:.6e}")

# ====================================================================
# PART 6: Verify star-product identity R_up = R_down^T (for symmetric problem)
# ====================================================================
print("\n" + "=" * 70)
print("PART 6: Operator diagnostics")
print("=" * 70)
print(f"R_up symmetric? max|R_up - R_up.T| = {float(jnp.max(jnp.abs(R_up - R_up.T))):.4e}")
print(f"R_down symmetric? max|R_down - R_down.T| = {float(jnp.max(jnp.abs(R_down - R_down.T))):.4e}")
print(f"T_up symmetric? max|T_up - T_up.T| = {float(jnp.max(jnp.abs(T_up - T_up.T))):.4e}")
print(f"T_down symmetric? max|T_down - T_down.T| = {float(jnp.max(jnp.abs(T_down - T_down.T))):.4e}")

# For the m=0 mode of an azimuthally-symmetric problem, all matrices should be symmetric
# Check reciprocity: T_up = T_down^T for m=0
print(f"T_up = T_down^T? max|T_up - T_down.T| = {float(jnp.max(jnp.abs(T_up - T_down.T))):.4e}")
print(f"R_up = R_down? (NOT expected for tau-varying)  max|R_up - R_down| = {float(jnp.max(jnp.abs(R_up - R_down))):.4e}")

# ====================================================================
# PART 7: s_down decomposition -- what is the surface seeing?
# ====================================================================
print("\n" + "=" * 70)
print("PART 7: What does the surface see?")
print("=" * 70)
# Total downwelling at surface from all sources:
# I-(tau_bot) = T_down @ b_neg + R_down @ I+(tau_bot) + s_down
# With b_neg=0:
# I-(tau_bot) = R_down @ I_plus_bot + s_down
I_minus_bot = R_down @ I_plus_bot + s_down
print(f"s_down (beam-generated downwelling)  = {np.array(s_down)}")
print(f"R_down @ I+_bot (reflected upwelling)= {np.array(R_down @ I_plus_bot)}")
print(f"Total I-(tau_bot)                    = {np.array(I_minus_bot)}")

# Verify surface BC: I+(tau_bot) = R_surf @ I-(tau_bot) + beam_surface_term
check_surface = R_surf @ I_minus_bot + beam_surface_term
residual = I_plus_bot - check_surface
print(f"\nSurface BC check: I+_bot - (R_surf @ I-_bot + beam_sfc) = {np.array(residual)}")
print(f"  max |residual| = {float(jnp.max(jnp.abs(residual))):.4e}")

# ====================================================================
# PART 8: Compare I+(tau_bot) and I-(tau_bot) from pydisort
# ====================================================================
print("\n" + "=" * 70)
print("PART 8: pydisort I+(tau_bot) and I-(tau_bot) vs Riccati")
print("=" * 70)

u0_at_bot_bdrf = u0f(tau_arr[-1])
pyd_Iplus_bot = u0_at_bot_bdrf[:N]
pyd_Iminus_bot = u0_at_bot_bdrf[N:]
print(f"pydisort I+(tau_bot, m=0) = {pyd_Iplus_bot}")
print(f"Riccati  I+(tau_bot, m=0) = {np.array(I_plus_bot) * rescale_factor}")
print(f"Diff I+(tau_bot)          = {pyd_Iplus_bot - np.array(I_plus_bot) * rescale_factor}")
print()
print(f"pydisort I-(tau_bot, m=0) = {pyd_Iminus_bot}")
print(f"Riccati  I-(tau_bot, m=0) = {np.array(I_minus_bot) * rescale_factor}")
print(f"Diff I-(tau_bot)          = {pyd_Iminus_bot - np.array(I_minus_bot) * rescale_factor}")

# ====================================================================
# PART 9: Verify s_down independently via no-BDRF downwelling at tau_bot
# ====================================================================
print("\n" + "=" * 70)
print("PART 9: Verify backward sweep (s_down) via no-BDRF case")
print("=" * 70)
# Without BDRF: I+(tau_bot) = 0, b_neg = 0
# I-(tau_bot) = T_down @ 0 + R_down @ 0 + s_down = s_down
# So s_down should equal the beam-generated downwelling at tau_bot
u0_at_bot_nb = u0f_nb(tau_arr[-1])
pyd_Iminus_bot_nb = u0_at_bot_nb[N:]
print(f"pydisort I-(tau_bot, m=0, no BDRF)  = {pyd_Iminus_bot_nb}")
print(f"Riccati s_down (m=0)                = {np.array(s_down) * rescale_factor}")
diff_sdown = pyd_Iminus_bot_nb - np.array(s_down) * rescale_factor
print(f"Diff                                = {diff_sdown}")
print(f"Max |diff|                          = {float(np.max(np.abs(diff_sdown))):.6e}")

# s_up verification (already confirmed at ToA, but be explicit)
print(f"\npydisort I+(0, m=0, no BDRF) = {u0_pydisort_no_bdrf}")
print(f"Riccati s_up (m=0)           = {np.array(s_up) * rescale_factor}")
diff_sup = u0_pydisort_no_bdrf - np.array(s_up) * rescale_factor
print(f"Diff                         = {diff_sup}")
print(f"Max |diff|                   = {float(np.max(np.abs(diff_sup))):.6e}")

# ====================================================================
# PART 10: T_up verification via known input
# ====================================================================
print("\n" + "=" * 70)
print("PART 10: Verify T_up — decompose BDRF contribution")
print("=" * 70)
ric_T_up_ones = np.array(T_up @ jnp.ones(N))
ric_I_plus_bot_val = float(I_plus_bot[0])  # uniform for Lambertian
print(f"I_plus_bot (uniform) = {ric_I_plus_bot_val:.10e}")
print(f"T_up @ ones(N)       = {ric_T_up_ones}")
ric_bdrf_only = ric_I_plus_bot_val * ric_T_up_ones
pyd_bdrf_only = pyd_m0 - u0_pydisort_no_bdrf
print(f"Riccati BDRF contrib = {ric_bdrf_only}")
print(f"pydisort BDRF contrib= {pyd_bdrf_only}")

# If T_up is correct, what I_plus_bot would pydisort need?
pyd_implied_Iplus_bot = pyd_bdrf_only / ric_T_up_ones
print(f"\nImplied pydisort I+_bot per mu (should be uniform if T_up matches):")
print(f"  = {pyd_implied_Iplus_bot}")
print(f"  std / mean = {float(np.std(pyd_implied_Iplus_bot) / np.mean(pyd_implied_Iplus_bot)):.6e}")
print(f"  Compare to Riccati I+_bot = {ric_I_plus_bot_val:.10e}")
