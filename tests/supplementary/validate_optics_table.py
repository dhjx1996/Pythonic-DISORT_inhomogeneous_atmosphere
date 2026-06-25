"""Validation gate: miepython optics table (src/optics_table.py) vs miejax_lite.

Compares ω, g (=Leg[1]), Q_ext, and the full Legendre vector on the bands shared
with the legacy JAX-Mie build, at matched build settings (max_nstop=512, n_gl=1024,
NLeg=128) so any difference is the reimplementation, not the config. Then sanity-
checks the new strong-absorption band 3.7 µm (outside CPV2012's range), which has
no miejax reference here but must be physically well-behaved for the solver.

Run (notebook env):  /srv/conda/envs/notebook/bin/python tests/supplementary/validate_optics_table.py
"""
import sys
from pathlib import Path

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
sys.path.insert(0, "/home/jovyan/cloud_profile_retrieval/miejax_lite")  # miejax_lite (legacy ref;
#   point at the package PARENT, not the workspace root, to avoid namespace shadowing)

import optics_table as ot                                            # noqa: E402

SHARED = [0.66, 0.86, 1.24, 1.64, 2.13]
RE_MIN, RE_MAX, N_RE, V_EFF, NLEG = 2.0, 25.0, 32, 0.10, 128
MAXN, NGL, NRADII = 512, 1024, 600                                   # match miejax build

print("building miepython table (matched config) ...", flush=True)
mp_tab = ot.build_re_table(SHARED, RE_MIN, RE_MAX, N_RE, V_EFF,
                           n_radii=NRADII, NLeg=NLEG, n_gl=NGL, max_nstop=MAXN)

print("building miejax_lite reference ...", flush=True)
from miejax_lite import mie_legendre_precompute, build_re_table as jax_build  # noqa: E402
precomp = mie_legendre_precompute(max_nstop=MAXN, NLeg=NLEG)
jx = jax_build(SHARED, RE_MIN, RE_MAX, N_RE, V_EFF, precomp, n_radii=NRADII)
jx_omega = np.asarray(jx["omega"]); jx_leg = np.asarray(jx["leg"]); jx_qext = np.asarray(jx["qext"])

dw = np.abs(mp_tab["omega"] - jx_omega)
dg = np.abs(mp_tab["leg"][:, :, 1] - jx_leg[:, :, 1])
dq = np.abs(mp_tab["qext"] - jx_qext)
dl = np.abs(mp_tab["leg"] - jx_leg)
print("\n=== miepython vs miejax_lite (shared bands, matched config) ===")
print(f"  |Δω|     max {dw.max():.2e}  mean {dw.mean():.2e}")
print(f"  |Δg|     max {dg.max():.2e}  mean {dg.mean():.2e}")
print(f"  |ΔQ_ext| max {dq.max():.2e}  mean {dq.mean():.2e}")
print(f"  |ΔLeg|   max {dl.max():.2e}  mean {dl.mean():.2e}  (all {NLEG} moments)")
for bi, lam in enumerate(SHARED):
    print(f"   band {lam:>5} µm: |Δω|max {dw[bi].max():.1e}  |Δg|max {dg[bi].max():.1e}")
gate = (dw.max() < 2e-3) and (dg.max() < 2e-3) and (dq.max() < 5e-3)
print(f"\nGATE (ω,g<2e-3, Q_ext<5e-3): {'PASS' if gate else 'FAIL'}")

print("\n=== 3.7 µm sanity (strong absorption; no miejax ref) ===")
hot = ot.build_re_table([3.7], RE_MIN, RE_MAX, N_RE, V_EFF, n_radii=NRADII, NLeg=NLEG)
re = np.linspace(RE_MIN, RE_MAX, N_RE)
om, g = hot["omega"][0], hot["leg"][0, :, 1]
print(f"  ω(3.7) range [{om.min():.4f}, {om.max():.4f}]  (absorbing → <1, expect ~0.5–0.8)")
print(f"  g(3.7) range [{g.min():.4f}, {g.max():.4f}]   χ_0 range "
      f"[{hot['leg'][0,:,0].min():.5f}, {hot['leg'][0,:,0].max():.5f}] (≡1)")
ok37 = (0.2 < om.min()) and (om.max() < 1.0) and np.all(np.diff(om) < 0) and (g.min() > 0.5)
print(f"  ω monotone↓ in r_e: {np.all(np.diff(om) < 0)} ; 3.7 µm usable: {ok37}")
print(f"\nOVERALL: {'PASS' if (gate and ok37) else 'REVIEW'}")
