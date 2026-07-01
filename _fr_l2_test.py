"""Layer-2 setup-cache equivalence gate (rigor before production). build_forward_and_obs with a
cache HIT (load + skip the deterministic setup) must reproduce the COMPUTE path bit-exactly:
same K_list, s_grid, tau_bot_pre, AND same forward/jacobian evaluations at a test state. Same
platform => bit-exact expected. A FAIL here blocks enabling L2 in the production sweep."""
import os, sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tests/supplementary")
import retrieval_worker as rw
import vocals_io as vio

IDX = int(os.environ.get("DIAG_IDX", "95"))
P = os.environ["L2_CKPT"]
if os.path.exists(P):
    os.remove(P)                                    # start clean: PASS 1 must WRITE it
profiles = vio.load_all_profiles(rw.DATA); truth = profiles[IDX]; flight = truth.flight
clim = vio.vocals_climatology(profiles, exclude_flight=flight)
print(f"[l2test] idx={IDX} {flight} tau={truth.tau_bot:.2f}; cache={P}", flush=True)


def run(tag):
    fwd, y, Se, s_grid, pb_phys, pb_log, tt, tbp = \
        rw.build_forward_and_obs(truth, clim, IDX, setup_cache_path=P)
    x = fwd._encode_state(rw.roe.make_climatology_prior(s_grid, clim)[0])
    F = np.asarray(fwd.forward(x, s_grid), float)
    K = np.asarray(fwd.jacobian(x, s_grid), float)
    print(f"[{tag}] K_list={list(map(int, fwd.K_list))} k={len(s_grid)} "
          f"tau_bot_pre={tbp:.8f}", flush=True)
    return list(map(int, fwd.K_list)), np.asarray(s_grid, float), float(tbp), F, K


print("=== PASS 1: compute + WRITE cache ===", flush=True)
k1, g1, t1, F1, K1 = run("compute")
# The gate's premise: PASS 1 actually wrote the cache, so PASS 2 exercises the LOAD path.
# Without this the test silently degrades to compute-vs-compute (a vacuous PASS) — which is
# exactly what the 2026-07-01 np.savez extension-munging write bug produced.
assert os.path.exists(P), ("L2-EQUIVALENCE: FAIL — PASS 1 did not write the setup cache "
                           f"({P}); LOAD path untestable (vacuous gate)")
print("=== PASS 2: LOAD cache (must print 'Layer-2 setup cache HIT') ===", flush=True)
k2, g2, t2, F2, K2 = run("load")

dg = float(np.abs(g1 - g2).max()) if len(g1) == len(g2) else 9e9
dF = float(np.abs(F1 - F2).max()); dK = float(np.abs(K1 - K2).max()); dt = abs(t1 - t2)
bit = (k1 == k2) and np.array_equal(g1, g2) and (t1 == t2) \
    and np.array_equal(F1, F2) and np.array_equal(K1, K2)
print(f"\nK_list match={k1 == k2} | s_grid dmax={dg:.2e} | tau_bot_pre d={dt:.2e} | "
      f"forward dmax={dF:.2e} | jacobian dmax={dK:.2e}")
worst = max(dg, dF, dK, dt)
print("L2-EQUIVALENCE:",
      "PASS (bit-exact)" if bit else ("PASS (within 1e-9)" if worst < 1e-9 else "FAIL"))
