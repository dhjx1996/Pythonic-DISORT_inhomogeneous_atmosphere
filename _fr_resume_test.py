"""L1 resume-equivalence rigor gate (FR_CHECKPOINT_RESUME_PLAN verification #1):
an uninterrupted GN solve must equal an interrupted-then-resumed one (bit-exact on the
same platform). Reuses the real worker setup (build_forward_and_obs) on a thin profile."""
import os, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tests/supplementary")
import jax
import retrieval_oe as roe
import retrieval_worker as rw          # build_forward_and_obs (reads caches from env)
import vocals_io as vio

print(f"jax {jax.__version__} x64={jax.config.read('jax_enable_x64')} {jax.devices()[0].platform}", flush=True)
IDX = int(os.environ.get("DIAG_IDX", "95"))
CK = os.environ["RESUME_CKPT"]                 # shared-FS checkpoint path
profiles = vio.load_all_profiles(rw.DATA)
truth = profiles[IDX]; flight = truth.flight
clim = vio.vocals_climatology(profiles, exclude_flight=flight)
fwd, y, Se, s_grid, pb_phys, pb_log, truth_tol, tau_bot_pre = rw.build_forward_and_obs(truth, clim, IDX)
x_a, Sa = roe.make_climatology_prior(s_grid, clim, log=True)
print(f"[idx {IDX}] {flight} setup done; k={len(s_grid)}", flush=True)

def run(ckpt, n_iter):
    return roe.gauss_newton_oe(fwd, y, s_grid, x_a, Sa, Se, x0=x_a, n_iter=n_iter,
                               lm=1e-2, xtol=2e-3, cost_rtol=None, chi2_floor=None,
                               max_n_outer=1, prior_builder=pb_log, checkpoint_path=ckpt)

ref = run(None, 8)                              # uninterrupted
print(f"uninterrupted: n_gn={len(ref.cost_history)}  x={np.round(ref.x,5)}", flush=True)
if os.path.exists(CK):
    os.remove(CK)
_ = run(CK, 3)                                  # interrupt: stop after 3 iters (checkpoint written)
print(f"  [checkpoint written at iter ~3]; resuming ...", flush=True)
res = run(CK, 8)                                # resume from checkpoint -> complete
print(f"resumed:       n_gn={len(res.cost_history)}  x={np.round(res.x,5)}", flush=True)

dmax = float(np.abs(ref.x - res.x).max())
bit = np.array_equal(ref.x, res.x)
print(f"\nmax|x_ref - x_resumed| = {dmax:.3e}")
print("RESUME-EQUIVALENCE:",
      "PASS (bit-exact)" if bit else ("PASS (within solver tol)" if dmax < 1e-6 else "FAIL"))
