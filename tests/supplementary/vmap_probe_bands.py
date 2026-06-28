"""Benchmark the GPU SECOND axis: bands x modes vmap vs band-looped modes-vmap vs scan.

The A100 modes-only probe (docs/cached_results/results.md) already showed vmap-over-modes
jacfwd ~7x faster than CPU-scan on ONE band. This probe asks the follow-up: does ALSO
batching the 10 bands into the same vmap (240-way: 10 bands x 24 modes) buy MORE on the
A100 (it was only 1.26/40 GB full → very under-used), or does the adaptive-solver lock-step
eat it? Batching bands forces the vmapped while_loop to run to the MAX adaptive step count
across bands — the absorbing bands take a few more (per-band m=0: 21/21/22/22/23 over
0.55→3.7 µm), so the cheap bands idle ~10%. The net of {SIMT fill gain} vs {lock-step cost}
is exactly what these three runs measure.

  vmap_probe_bands.py scan        CPU/GPU band-loop, modes-SCAN          (production CPU path)
  vmap_probe_bands.py vmap_loop   GPU band-loop, modes-VMAP per band     (axis 1 only; bands serial)
  vmap_probe_bands.py vmap_both   GPU one vmap over bands x modes (240)  (axis 2; the GPU target)

Full operational osse_config: 10 bands, NQuad=48, NFourier=24, NLeg_all=1024, float64,
jac_mode='fwd', 24 principal-plane views. EVAL-ONLY = warm (compile excluded). Prints
y sum / n_neg so all three are checkable for bit-identity (modulo cross-backend rounding),
and device peak mem so the 240-way A100 footprint is on record.

Backend = JAX_PLATFORMS (cpu | cuda). Run vmap_loop/vmap_both with the GPU python; scan with
either (CPU for the reference). The decisive numbers:
  (i)  vmap_both jacfwd  vs  vmap_loop jacfwd   → does batching bands add value on GPU?
  (ii) vmap_both jacfwd  vs  CPU-scan jacfwd    → end-to-end second-axis win.
"""
import os
import sys
import time
import resource

os.environ.setdefault("JAX_PLATFORMS", "cpu")                    # agent overrides to 'cuda'
os.environ.setdefault("PYDISORT_RICCATI_JAX_X64", "1")          # solver requires float64
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))
sys.path.insert(0, _here)
import numpy as np
import jax
import retrieval_oe as roe
import osse_config as oc

MODE = sys.argv[1] if len(sys.argv) > 1 else "scan"
assert MODE in ("scan", "vmap_loop", "vmap_both"), f"bad mode {MODE!r}"
TABLE = os.path.join(_here, "optics_table_10band_nleg1024_re20.npz")
opt_all = oc.load_optics(TABLE)                                  # the 10 operational bands

# representative thick exemplar (re5-linear, 6 interior nodes + base + tau_bot) — exercises
# the absorbing-band extra-step lock-step the probe is here to measure.
s6 = np.linspace(0.0, 0.999, 6)
r_top, r_base, tb = 12.0, 8.0, 12.0
re6 = (r_base ** 5 + (r_top ** 5 - r_base ** 5) * (1 - s6)) ** 0.2


def build(mode_map):
    fwd = roe.RetrievalForward(
        opt_all, NQuad=oc.NQUAD, mu0=oc.MU0, I0=oc.I0, phi0=oc.PHI0, tau_bot=tb,
        r_base=r_base, view_mu=oc.VIEW_MU, view_phi=oc.VIEW_PHI,
        BDRF_bands=[[oc.ALBEDO]] * oc.NB, NLeg_all=oc.NLEG_ALL, NFourier=oc.NFOURIER,
        re_class=oc.RE_CLASS, state_space="log", jac_mode="fwd",
        retrieve_tau_bot=True, retrieve_r_base=True, re_bounds=oc.RE_BOUNDS,
        mode_map=mode_map)
    return fwd


def mem_report():
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)   # GB
    dev = ""
    try:
        st = jax.local_devices()[0].memory_stats()
        peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
        if peak:
            dev = f"  device peak {peak / 1024 ** 3:.2f} GB"
    except Exception:
        pass
    return f"host RSS {host:.2f} GB{dev}"


print(f"[{MODE}] JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
fwd = build("scan" if MODE == "scan" else "vmap")
if MODE == "vmap_loop":
    fwd._bands_share_setup = False          # force band-LOOP; modes still vmapped per band
elif MODE == "vmap_both":
    assert fwd._bands_share_setup and len(set(fwd.K_list)) == 1, "bands not shareable!"
path = ("band-loop, modes-scan" if MODE == "scan" else
        "band-loop, modes-vmap" if MODE == "vmap_loop" else "ONE vmap: bands x modes")
print(f"[{MODE}] path = {path}   ({oc.NB} bands, K={fwd.K_list[0]}, NQuad={oc.NQUAD}, float64)",
      flush=True)

x = fwd._encode_state(np.append(np.append(re6, r_base), tb))
t = time.time(); y = np.asarray(fwd.forward(x, s6))
print(f"[{MODE}] forward compile+eval {time.time() - t:.0f}s  y.shape={y.shape} "
      f"sum={y.sum():.6f} n_neg={(y < 0).sum()}", flush=True)
t = time.time(); J = np.asarray(fwd.jacobian(x, s6))
print(f"[{MODE}] jacfwd compile+eval {time.time() - t:.0f}s  J.shape={J.shape}", flush=True)

for _ in range(2):
    _ = np.asarray(fwd.forward(x, s6))                          # warm
t = time.time()
for _ in range(3):
    _ = np.asarray(fwd.forward(x, s6))
tfe = (time.time() - t) / 3.0
_ = np.asarray(fwd.jacobian(x, s6))                            # warm
t = time.time()
for _ in range(2):
    _ = np.asarray(fwd.jacobian(x, s6))
tje = (time.time() - t) / 2.0
print(f"[{MODE}] EVAL-ONLY: forward {tfe:.2f}s  jacfwd {tje:.2f}s", flush=True)
print(f"[{MODE}] {mem_report()}", flush=True)
