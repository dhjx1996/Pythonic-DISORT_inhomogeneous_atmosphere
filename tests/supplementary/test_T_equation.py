"""
Test whether the T equation dT/dσ = T(α+βR) or dT/dσ = (α+βR)T is correct.

We compare both formulations against pydisort's BDRF contribution for the
test_7c problem, where T_up is the only untested operator.
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
import diffrax

from PythonicDISORT import subroutines
from PythonicDISORT.pydisort import pydisort
from _riccati_solver_jax import (
    _precompute_legendre,
    _make_alpha_beta_funcs_jax,
    _make_q_funcs_jax,
)

# ---- Problem setup (test_7c) ----
NQuad = 8
NLeg = NQuad
N = NQuad // 2
tau_bot = 1.0
mu0, I0, phi0 = 0.5, 1.0, 0.0
rho = 0.3
BDRF = [rho / pi]
omega_func = lambda tau: 0.90 + (0.60 - 0.90) * tau / tau_bot
g_func = lambda tau: 0.70 + (0.30 - 0.70) * tau / tau_bot

def Leg_coeffs_func(tau):
    return g_func(tau) ** np.arange(NLeg)

mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)
mu_arr_pos_jax = jnp.array(mu_arr_pos)
W_jax = jnp.array(W)
M_inv = 1.0 / mu_arr_pos_jax

rescale_factor = 1.0
I0_div_4pi = I0 / (4 * pi)

m = 0
m_equals_0 = True
tol = 1e-10

leg_data = _precompute_legendre(m, NLeg, mu_arr_pos_jax, mu0)
alpha_func, beta_func = _make_alpha_beta_funcs_jax(
    omega_func, Leg_coeffs_func, m, leg_data,
    mu_arr_pos_jax, W_jax, M_inv, N,
)
q_up_m, q_down_m = _make_q_funcs_jax(
    omega_func, Leg_coeffs_func, m, leg_data,
    mu_arr_pos_jax, M_inv, mu0, I0_div_4pi, m_equals_0, N,
)


def integrate_forward(T_variant="T_aR_bR"):
    """
    Forward sweep (bottom to top) with configurable T equation.
    T_variant="T_aR_bR": dT = T @ (α + β@R)  [current code]
    T_variant="aR_bR_T": dT = (α + β@R) @ T
    T_variant="aR_Rb_T": dT = (α + R@β) @ T  [Redheffer star product derivation]
    T_variant="T_aR_Rb": dT = T @ (α + R@β)
    """
    tb = float(tau_bot)

    def vector_field(sigma, state, args):
        R = state['R']
        T = state['T']
        s = state['s']

        alpha = alpha_func(tb - sigma)
        beta = beta_func(tb - sigma)

        dR = alpha @ R + R @ alpha + R @ beta @ R + beta

        if T_variant == "T_aR_bR":
            dT = T @ (alpha + beta @ R)
        elif T_variant == "aR_bR_T":
            dT = (alpha + beta @ R) @ T
        elif T_variant == "aR_Rb_T":
            dT = (alpha + R @ beta) @ T
        elif T_variant == "T_aR_Rb":
            dT = T @ (alpha + R @ beta)

        q1 = q_down_m(tb - sigma)
        q2 = q_up_m(tb - sigma)
        ds = (alpha + R @ beta) @ s + R @ q1 + q2

        return {'R': dR, 'T': dT, 's': ds}

    y0 = {
        'R': jnp.zeros((N, N)),
        'T': jnp.eye(N),
        's': jnp.zeros(N),
    }

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=tol, atol=tol * 1e-3)

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=tb,
        dt0=None, y0=y0,
        stepsize_controller=controller,
        max_steps=4096,
    )

    R = sol.ys['R'][-1]
    T = sol.ys['T'][-1]
    s = sol.ys['s'][-1]
    return R, T, s


def integrate_backward(T_variant="T_aR_bR"):
    """Backward sweep (top to bottom)."""
    tb = float(tau_bot)

    def vector_field(sigma, state, args):
        R = state['R']
        T = state['T']
        s = state['s']

        alpha = alpha_func(sigma)
        beta = beta_func(sigma)

        dR = alpha @ R + R @ alpha + R @ beta @ R + beta

        if T_variant == "T_aR_bR":
            dT = T @ (alpha + beta @ R)
        elif T_variant == "aR_bR_T":
            dT = (alpha + beta @ R) @ T
        elif T_variant == "aR_Rb_T":
            dT = (alpha + R @ beta) @ T
        elif T_variant == "T_aR_Rb":
            dT = T @ (alpha + R @ beta)

        q1 = q_up_m(sigma)
        q2 = q_down_m(sigma)
        ds = (alpha + R @ beta) @ s + R @ q1 + q2

        return {'R': dR, 'T': dT, 's': ds}

    y0 = {
        'R': jnp.zeros((N, N)),
        'T': jnp.eye(N),
        's': jnp.zeros(N),
    }

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=tol, atol=tol * 1e-3)

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=tb,
        dt0=None, y0=y0,
        stepsize_controller=controller,
        max_steps=4096,
    )

    R = sol.ys['R'][-1]
    T = sol.ys['T'][-1]
    s = sol.ys['s'][-1]
    return R, T, s


def bc_solve(R_up, T_up, s_up, R_down, T_down, s_down):
    """BC solve with BDRF, b_pos=0, b_neg=0."""
    BDRF_mode = BDRF[0]
    mu_W = mu_arr_pos_jax * W_jax
    R_surf = 2 * BDRF_mode * mu_W[None, :]
    mathscr_X = (mu0 * I0_div_4pi * 4) * BDRF_mode * jnp.ones(N)
    beam_sfc = mathscr_X * jnp.exp(-tau_bot / mu0)

    b_neg = jnp.zeros(N)
    LHS = jnp.eye(N) - R_surf @ R_down
    RHS = R_surf @ (T_down @ b_neg + s_down) + beam_sfc
    I_plus_bot = jnp.linalg.solve(LHS, RHS)
    I_plus_top = (R_up @ b_neg + T_up @ I_plus_bot + s_up).real
    return I_plus_top


# ---- pydisort reference ----
NLayers = 5000
edges = np.linspace(0, tau_bot, NLayers + 1)
mids = 0.5 * (edges[:-1] + edges[1:])
tau_arr = edges[1:]
omega_arr = np.array([omega_func(t) for t in mids])
Leg_arr = np.array([Leg_coeffs_func(t) for t in mids])

_, _, _, u0f, _ = pydisort(
    tau_arr, omega_arr, NQuad,
    Leg_arr, float(mu0), float(I0), float(phi0),
    NLeg=NLeg, NFourier=NQuad, only_flux=False,
    BDRF_Fourier_modes=list(BDRF),
)
pyd_m0 = u0f(0)[:N]

# ---- Test all four T formulations ----
variants = ["T_aR_bR", "aR_bR_T", "aR_Rb_T", "T_aR_Rb"]
labels = {
    "T_aR_bR": "T(α+βR) [current]",
    "aR_bR_T": "(α+βR)T",
    "aR_Rb_T": "(α+Rβ)T [Redheffer]",
    "T_aR_Rb": "T(α+Rβ)",
}

results = {}
for v in variants:
    print(f"Computing {labels[v]}...")
    R_up, T_up, s_up = integrate_forward(v)
    R_dn, T_dn, s_dn = integrate_backward(v)
    u = np.array(bc_solve(R_up, T_up, s_up, R_dn, T_dn, s_dn))
    results[v] = u

scale = max(float(np.max(np.abs(pyd_m0))), 1e-8)

print("\n" + "=" * 70)
print("COMPARISON vs pydisort m=0 (5000 layers)")
print("=" * 70)
print(f"pydisort           = {pyd_m0}")
print()

for v in variants:
    u = results[v]
    diff = u - pyd_m0
    maxdiff = float(np.max(np.abs(diff)))
    reldiff = maxdiff / scale
    print(f"{labels[v]:24s}: u = {u}  max|diff| = {maxdiff:.4e}  rel = {reldiff:.4e}")
