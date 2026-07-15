"""Wiggle-provenance probe — ToA radiance Jacobian on a DENSE uniform depth grid,
with the phase function switchable between the full Mie table and a Henyey-Greenstein
surrogate with MATCHED (omega, g).

Purpose (2026-07 investigation): the Fig-0a weighting functions w(s) ~ |dI/dr_e(s)|
show non-monotone "wiggles" at depth for the short-wavelength (near-conservative)
bands. Hypothesis: they are the r_e-interference structure of the Mie phase function
(glory / cloudbow / supernumerary bows) swept by the smooth r_e(s) linearization
profile. Discriminating experiment: HG has NO secondary phase-function features
(smooth in both angle and r_e via g(r_e) alone), so
    * wiggles persist under HG  -> spurious (numerics / grid),
    * wiggles vanish under HG   -> Mie fine-structure is the driver.
The 'mie' phase re-runs the SAME code path as the definitive IC set A (priormean
linearization) but on the dense grid, so the HG/Mie pair differs ONLY in the phase
function (same grid, same code, same Se construction).

Usage:  ic_worker_wiggle.py <profile_index> <out.json>     (sidecar <out.npz>)
Env:    IC_PHASE       mie | hg                (default mie)
        N_SGRID        dense-grid points        (default 51, s in [0,1] incl. base)
        N_RE_TABLE     if set, build/load a FINER r_e-grid optics table with this many
                       r_e points at OPTICS_CACHE (default: production 32-point table via
                       oc.load_optics). The 2x2 {mie,hg}x{32,fine} factorial separates
                       angular Mie features / physical g-ripples / table-grid artifacts
                       (table_lookup is piecewise-linear -> its r_e-derivative is a
                       STAIRCASE with n_re-1 steps; a coarse grid imprints on K).
        ENSEMBLE_NQUAD (default 48), SOLVER_TOL, OPTICS_CACHE, VOCALS_DATA as usual.

HG construction: the optics table stores chi_l with chi_0=1, chi_1=g; HG is exactly
chi_l = g^l, with g = the Mie table's own chi_1(r_e) per band -> omega(r_e) and
g(r_e) (and their r_e-derivatives, up to table resolution) match Mie; every moment
l>=2 collapses to the g^l geometric decay. The observation y is generated
self-consistently in the same phase-function world (osse_observation, noiseless).
"""
import sys
import os
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pydisort_riccati_jax import runtime_setup                       # noqa: E402
runtime_setup.setup()                                              # affinity pin BEFORE JAX
from pydisort_riccati_jax import vocals_io as vio                    # noqa: E402
from pydisort_riccati_jax import retrieval_oe as roe                 # noqa: E402
from pydisort_riccati_jax import noise_model as nm                   # noqa: E402
from pydisort_riccati_jax import osse_config as oc                   # noqa: E402

PHASE = os.environ.get("IC_PHASE", "mie")
N_SGRID = int(os.environ.get("N_SGRID", "51"))
NQ = int(os.environ.get("ENSEMBLE_NQUAD", "48"))
NOISE = nm.oci_swir()
s_ref = np.linspace(0.0, 1.0, 6)[:-1]                # retrieval grid (linearization nodes)

idx, out = int(sys.argv[1]), sys.argv[2]
profiles = vio.load_all_profiles(oc.VOCALS_DATA)
truth = profiles[idx]
flight = getattr(truth, "flight", "?")

try:
    if PHASE not in ("mie", "hg"):
        raise ValueError(f"bad IC_PHASE={PHASE!r}")
    if not (0.3 <= float(truth.tau_bot) <= 100.0) or len(np.asarray(truth.tau)) < 5:
        raise ValueError(f"degenerate (tau_bot={truth.tau_bot:.2f})")

    n_re_env = os.environ.get("N_RE_TABLE")
    if n_re_env:
        from pydisort_riccati_jax import optics_table as ot
        re_table = ot.build_or_load_table(
            oc.BANDS, oc.RE_BOUNDS[0], oc.RE_BOUNDS[1], int(n_re_env), oc.V_EFF,
            cache_path=oc.OPTICS_CACHE, NLeg=oc.NLEG_ALL, n_radii=oc.N_RADII,
            n_gl=oc.N_GL)
        opt = [ot.select_channel(re_table, i) for i in range(oc.NB)]
    else:
        opt = oc.load_optics(oc.OPTICS_CACHE)
    if PHASE == "hg":
        import jax.numpy as jnp
        hg_opt = []
        for band in opt:
            leg = np.asarray(band["leg"])            # (n_re, NLeg); chi_0=1, chi_1=g
            g = leg[:, 1:2]
            ells = np.arange(leg.shape[1], dtype=float)[None, :]
            band = dict(band)
            band["leg"] = jnp.asarray(g ** ells)     # HG: chi_l = g^l, g=g_Mie(r_e)
            hg_opt.append(band)
        opt = hg_opt

    clim = vio.vocals_climatology(profiles, exclude_flight=flight)
    # Texture-free Jacobian modes (2026-07-14): FIXED_N_STEPS = frozen uniform
    # StepTo grid (full-Newton — measured ~10-20x adaptive, K-only use at most);
    # FREEZE_STEP_GRADS=1 = adaptive controller with stop-gradient'ed step
    # decisions (native cost). The y generation + mode selection ALWAYS run on
    # the plain adaptive forward (their accuracy is tolerance-grade already and
    # the frozen forward is needlessly expensive there); only the dense Jacobian
    # uses the frozen variant, inheriting the adaptive run's mode selection.
    _fns = os.environ.get("FIXED_N_STEPS")
    _fsg = os.environ.get("FREEZE_STEP_GRADS") in ("1", "true", "True")
    fwd = oc.build_forward(opt, tau_bot=float(truth.tau_bot), r_base=float(truth.r_base),
                           views="full", jac_mode="fwd",
                           mode_map=os.environ.get("MODE_MAP", "scan"))
    fwd_K = fwd
    if _fns or _fsg:
        fwd_K = oc.build_forward(opt, tau_bot=float(truth.tau_bot),
                                 r_base=float(truth.r_base),
                                 views="full", jac_mode="fwd",
                                 mode_map=os.environ.get("MODE_MAP", "scan"),
                                 fixed_n_steps=int(_fns) if _fns else None,
                                 freeze_step_grads=_fsg)

    # priormean linearization — identical to the definitive IC set A (smooth adiabatic
    # LOO prior mean; NOT the wiggly in-situ truth), tau_bot at the truth.
    xa_sref = np.asarray(roe.make_climatology_prior(s_ref, clim)[0])
    nb6 = len(s_ref) + 1
    x_lin_phys = np.concatenate([xa_sref[:nb6], [truth.tau_bot]])
    x_lin = fwd._encode_state(x_lin_phys)

    t0 = time.time()
    # self-consistent observation (same phase-function world), noiseless -> OCI Se
    y = np.asarray(roe.osse_observation(fwd, truth.tau, truth.r_e))
    sig = NOISE.sigma(y, n_bands=oc.NB)
    print(f"[{idx}] {flight} tau={truth.tau_bot:.1f} phase={PHASE}: "
          f"y done in {time.time()-t0:.0f}s; mode selection + dense Jacobian...", flush=True)
    roe.select_num_modes(fwd, x_lin, s_ref, np.diag(sig ** 2))
    if fwd_K is not fwd:
        fwd_K.K_list = list(fwd.K_list)              # inherit the noise-aware mode trim

    s_grid = np.linspace(0.0, 1.0, N_SGRID)          # DENSE uniform grid incl. base s=1
    tau_bot = float(truth.tau_bot)
    re_grid = fwd_K.profile(x_lin, s_ref, s_grid * tau_bot)
    K = np.asarray(fwd_K.jacobian_on_grid(re_grid, s_grid, tau_bot))
    if not np.abs(K).max() > 0:
        raise ValueError(f"NULL Jacobian (maxabs={np.abs(K).max():.2e})")
    if not np.all(np.isfinite(K)):
        raise ValueError("non-finite Jacobian")

    npz_path = Path(out).with_suffix(".npz")
    np.savez_compressed(
        npz_path, index=idx, flight=flight, tau_bot=tau_bot, NQuad=NQ, phase=PHASE,
        n_bands=oc.NB, nv_max=oc.N_VIEW_FULL, bands=np.asarray(oc.BANDS),
        view_mu=oc.VIEW_MU_FULL, s_grid=s_grid, re_grid=np.asarray(re_grid),
        K_full=K, sigma_full=sig, y_full=y, x_lin=np.asarray(x_lin_phys),
        v_eff=oc.V_EFF, K_list=np.asarray(fwd.K_list), n_re_table=int(opt[0]["n_re"]),
        fixed_n_steps=int(_fns) if _fns else 0, freeze_step_grads=int(_fsg))
    rec = dict(index=idx, flight=flight, tau_bot=tau_bot, phase=PHASE, NQuad=NQ,
               n_sgrid=N_SGRID, npz=npz_path.name, K_maxabs=float(np.abs(K).max()),
               runtime_s=round(time.time() - t0, 1))
    print(f"[{idx}] {flight} phase={PHASE} DONE [{time.time()-t0:.0f}s] "
          f"K {K.shape} maxabs={np.abs(K).max():.3e} -> {npz_path.name}", flush=True)
except Exception as e:                                              # noqa: BLE001
    rec = dict(index=idx, flight=flight, tau_bot=float(getattr(truth, "tau_bot", 0.0)),
               phase=PHASE, skipped=str(e)[:200])
    print(f"[{idx}] {flight}: SKIPPED {rec['skipped']}", flush=True)

Path(out).write_text(json.dumps(rec))
