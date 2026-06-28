"""Benchmark: vmap-over-Fourier-modes vs the production lax.scan, any backend.

  vmap_probe.py scan    -> eval-only forward+jacfwd time + peak mem, production scan-over-modes
  vmap_probe.py vmap    -> same, with the modes batched via jax.vmap

Config = osse_config (NQuad=48, NFourier=24, NLeg_all=1024), float64, jac_mode='fwd', 1 band.
Backend is whatever JAX_PLATFORMS selects (cpu | cuda). On CPU, vmap was measured ~3x SLOWER
(XLA-CPU doesn't parallelize a batched implicit solve); the open question is whether a GPU
(SIMT over the batch) flips that. Each run prints y[:3]+sum so vmap-vs-scan / GPU-vs-CPU
agreement (bit-identity modulo cross-backend rounding) is verifiable from the logs.

Reference CPU numbers (jovyan, 8 threads, warm): scan fwd 67s / jac 206s; vmap fwd 181s / jac 639s.
"""
import os
import sys
import time
import resource

os.environ.setdefault("JAX_PLATFORMS", "cpu")                     # agent overrides to 'cuda'
os.environ.setdefault("PYDISORT_RICCATI_JAX_X64", "1")           # solver requires float64
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, _here)
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import pydisort_riccati_jax as P
import retrieval_oe as roe
import osse_config as oc

MODE = sys.argv[1] if len(sys.argv) > 1 else "scan"
TABLE = os.path.join(_here, "optics_table_10band_nleg1024_re20.npz")
KMODES = 24
opt_all = oc.load_optics(TABLE)

s6 = np.linspace(0.0, 0.999, 6)
r_top, r_base, tb = 12.0, 8.0, 12.0
re6 = (r_base ** 5 + (r_top ** 5 - r_base ** 5) * (1 - s6)) ** 0.2


def build_1band(bi):
    fwd = roe.RetrievalForward(
        [opt_all[bi]], NQuad=48, mu0=0.9, I0=1.0, phi0=0.0, tau_bot=tb, r_base=r_base,
        view_mu=oc.VIEW_MU, view_phi=oc.VIEW_PHI, BDRF_bands=[[0.06]], NLeg_all=oc.NLEG_ALL,
        NFourier=KMODES, state_space="log", jac_mode="fwd",
        retrieve_tau_bot=True, retrieve_r_base=True, re_bounds=oc.RE_BOUNDS)
    fwd.K_list = [KMODES]
    return fwd


def mem_report():
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)   # GB
    dev = ""
    try:
        st = jax.local_devices()[0].memory_stats()
        peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
        if peak:
            dev = f"  device peak {peak / 1024**3:.2f} GB"
    except Exception:
        pass
    return f"host RSS {host:.2f} GB{dev}"


# vmap variant of _fourier_solve: faithful copy, lax.scan -> jax.vmap over modes
def _fourier_solve_vmap(setup, omega_func, Leg_coeffs_func, tau_bot, *,
                        num_modes, return_grid, Leg_coeffs_tms_func=None):
    N, NLeg, K = setup.N, setup.NLeg, num_modes
    if Leg_coeffs_tms_func is None:
        Leg_coeffs_tms_func = Leg_coeffs_func
    if setup.delta_M_scaling:
        f_of_tau = lambda tau: Leg_coeffs_func(tau)[NLeg]
        tau_star_eval, tau_star_bot = P._compute_tau_star(omega_func, f_of_tau, tau_bot)
    else:
        tau_star_eval, tau_star_bot = None, tau_bot

    def one_mode(wp_m, ap_pos_m, ap_neg_m, amu0_m, bpos_m, bneg_m, R_raw_m, beam_m, mz):
        alpha_func, beta_func = P._make_alpha_beta_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m,
            setup.W_jax, setup.M_inv, N, NLeg, setup.delta_M_scaling)
        q_up, q_down = P._make_q_funcs_jax(
            omega_func, Leg_coeffs_func, wp_m, ap_pos_m, ap_neg_m, amu0_m,
            setup.M_inv, setup.mu0, setup.I0_div_4pi, mz, N, NLeg,
            setup.delta_M_scaling, tau_star_eval)
        R_up, T_up, s_up, _ = P._riccati_forward_jax(
            alpha_func, beta_func, tau_bot, N, setup.tol,
            q_up_func=q_up, q_down_func=q_down, save_grid=False, adjoint=setup.adjoint)
        R_down, T_down, s_down, _ = P._riccati_backward_jax(
            alpha_func, beta_func, tau_bot, N, setup.tol,
            q_up_func=q_up, q_down_func=q_down, save_grid=False, adjoint=setup.adjoint)
        return P._solve_bc_riccati_jax(
            R_up, T_up, T_down, R_down, s_up, s_down, N, bpos_m, bneg_m, R_raw_m, beam_m, mz,
            setup.mu_arr_pos_jax, setup.W_jax, setup.I0_div_4pi, setup.mu0, tau_star_bot)

    stacks = (setup.weighted_poch_modes[:K], setup.asso_leg_pos_modes[:K],
              setup.asso_leg_neg_modes[:K], setup.asso_leg_mu0_modes[:K],
              setup.b_pos_modes[:K], setup.b_neg_modes[:K],
              setup.bdrf_R_modes[:K], setup.bdrf_beam_modes[:K], setup.m_is_zero[:K])
    u_modes_arr = jax.vmap(one_mode)(*stacks)
    if setup.rescale_factor > 0:
        u_modes_arr = u_modes_arr * setup.rescale_factor
    tms_data = None
    if setup.NT_cor:
        tms_data = P._precompute_tms(
            omega_func, Leg_coeffs_tms_func, tau_star_eval, tau_bot, setup.mu0, setup.phi0,
            setup.I0_orig_div_4pi, NLeg, setup.NLeg_all, setup.NT_quad_order)
    return P.SolveResult(u_modes=u_modes_arr, tms_data=tms_data, tau_grid=None)


print(f"[{MODE}] JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
if MODE == "vmap":
    P._fourier_solve = _fourier_solve_vmap
fwd = build_1band(0)
x = fwd._encode_state(np.append(np.append(re6, r_base), tb))

t = time.time(); y = np.asarray(fwd.forward(x, s6)); print(f"[{MODE}] forward compile+eval {time.time()-t:.0f}s  y[:3]={np.round(y[:3],6)} sum={y.sum():.6f} n_neg={(y<0).sum()}", flush=True)
t = time.time(); J = np.asarray(fwd.jacobian(x, s6)); print(f"[{MODE}] jacfwd compile+eval {time.time()-t:.0f}s  J.shape={J.shape}", flush=True)
for _ in range(2):
    _ = np.asarray(fwd.forward(x, s6))                            # warm
t = time.time()
for _ in range(3):
    _ = np.asarray(fwd.forward(x, s6))
tfe = (time.time() - t) / 3.0
_ = np.asarray(fwd.jacobian(x, s6))
t = time.time()
for _ in range(2):
    _ = np.asarray(fwd.jacobian(x, s6))
tje = (time.time() - t) / 2.0
print(f"[{MODE}] EVAL-ONLY: forward {tfe:.2f}s  jacfwd {tje:.2f}s   (1 band, K=24, float64)", flush=True)
print(f"[{MODE}] {mem_report()}", flush=True)
