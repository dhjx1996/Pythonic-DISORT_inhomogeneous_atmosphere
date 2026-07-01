"""Cross-check a retrieved FR result against the golden GPU-probe bundle for the same index.
Usage: _fr_golden_compare.py <idx> [ours_A_npz]   (default ours = _fr_parts/<idx>_A.npz)

Primary comparison is on the platform-INVARIANT dense retrieved profile re_ours_dense(s_dense)
and scalars (tau_bot_ret, dofs, sic, rmse) -- these are grid-independent, so a CPU result is
comparable to the GPU-generated golden even if the QRCP node grid differs slightly. x_hat_log is
compared too but only meaningfully when the node grids match (k + s_grid)."""
import numpy as np, sys

idx = int(sys.argv[1])
ours_p = sys.argv[2] if len(sys.argv) > 2 else f"docs/cached_results/_fr_parts/{idx}_A.npz"
gold_p = f"tests/supplementary/precision_probe_out/probe_{idx}_tol1e-4_A.npz"
o = np.load(ours_p, allow_pickle=True)
g = np.load(gold_p, allow_pickle=True)

print(f"=== GOLDEN CROSS-CHECK idx {idx}  (config A headline) ===")
print(f"  ours = {ours_p}")
ko, kg = int(o["k"]), int(g["k"])
print(f"  NQuad ours={int(o['NQuad'])} golden={int(g['NQuad'])} | k ours={ko} golden={kg}")
print(f"  s_grid ours  ={np.round(np.asarray(o['s_grid'],float),4)}")
print(f"  s_grid golden={np.round(np.asarray(g['s_grid'],float),4)}")

# platform-invariant dense profile (um) -- the robust comparison
ro = np.asarray(o["re_ours_dense"], float); rg = np.asarray(g["re_ours_dense"], float)
dprof = np.abs(ro - rg)
print(f"  re_ours_dense (um): max|d|={dprof.max():.3e}  rms|d|={np.sqrt(np.mean(dprof**2)):.3e}")

# truth-dense should be identical (same profile) -- sanity
rt_o = np.asarray(o["re_truth_dense"], float); rt_g = np.asarray(g["re_truth_dense"], float)
print(f"  re_truth_dense agree? max|d|={np.abs(rt_o-rt_g).max():.2e} (should be ~0)")

for kf in ("tau_bot_ret", "r_base_ret", "dofs", "sic"):
    vo, vg = float(o[kf]), float(g[kf])
    print(f"  {kf:12s}: ours={vo:.5f} golden={vg:.5f}  d={vo-vg:+.2e}")
print(f"  n_gn ours={int(o['n_gn'])} golden={int(g['n_gn'])} | converged ours={bool(o['converged'])} golden={bool(g['converged'])}")
print(f"  cost_history ours  ={np.round(np.asarray(o['cost_history'],float),4)}")
print(f"  cost_history golden={np.round(np.asarray(g['cost_history'],float),4)}")
if ko == kg and np.allclose(np.asarray(o['s_grid'],float), np.asarray(g['s_grid'],float), atol=1e-6):
    dx = np.abs(np.asarray(o['x_hat_log'],float) - np.asarray(g['x_hat_log'],float))
    print(f"  x_hat_log (grids match): max|d|={dx.max():.3e}")

m = dprof.max()
verdict = ("TIGHT MATCH (<1e-4 um -> resume+correctness confirmed)" if m < 1e-4 else
           "CLOSE (<5e-2 um -> correct; expected CPU/GPU or version drift)" if m < 5e-2 else
           "MISMATCH (>5e-2 um -> INVESTIGATE)")
print(f"VERDICT idx {idx}: {verdict}  [max|re_dense diff|={m:.3e} um]")
