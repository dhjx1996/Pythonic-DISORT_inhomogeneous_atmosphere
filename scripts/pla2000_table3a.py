"""Platnick (2000) Table-3a validation — BOTH stages in one script (merged 2026-07-14).

Stage "retrieval" — recreate column 1 ("Retrieval") of Pla2000 Table 3a with OUR solver +
homogeneous-equivalent matching: the verification that our 'retrieval algorithm' (in the
Eq-3-validation sense: match a homogeneous cloud's reflectance to the structured cloud's)
reproduces the published numbers.

Platnick's setup (his §3.1, Tables 1-3a; §5 text): profiles B (adiabatic,
r_e(τ) = (r_top^5 − (r_top^5 − r_base^5)·τ/τ_c)^{1/5}, τ from cloud TOP) and D (linear,
r_e = r_top − (r_top − r_base)·τ/τ_c); clouds (τ_c; r_base→r_top): (15; 4→10),
(10; 6→15), (8; 5→12), (5; 8→12); built from Δτ=0.25 homogeneous layers with INTEGER
r_e nearest the analytic value at the layer midpoint; bands 1.6/2.2/3.7 µm; μ0=0.65,
μ=0.85, AZIMUTHALLY-AVERAGED bidirectional reflectance (→ our u0, m=0 only,
NFourier=1); black surface; τ quoted at wavelength-independent Q_e=2 (→ band-shared τ);
retrieval = closest homogeneous-library reflectance at KNOWN τ_c.

Published column-1 targets hardcoded below. Spec items his paper leaves open (noted,
loosen the agreement bar to ~0.3-0.5 µm): size-distribution width (we use gamma
v_eff=0.1), refractive index source (we use Segelstein), his adding/doubling angular
resolution. Solar-only at 3.7 µm matches his "cloud emission removed without error".

Stage "kernel" — the BULLETPROOF comparison: score OUR derivative kernel on HIS OWN eight
clouds. For each cloud compute the layer Jacobian K_j = dR/dr_e(layer j) by ONE
reverse-mode AD pass through the solver, form the Eq-(3) estimates

    r*_signed = sum(K_j r_j)/sum(K_j),      r*_abs = sum(|K_j| r_j)/sum(|K_j|),

and score |r* - retrieval| against Platnick's PUBLISHED w_m and w_N estimate errors on
the SAME clouds (his Table 3a columns). Identical clouds, identical criterion, fair ratio.
The kernel tolerance is env-overridable (PLA_TOL, default 1e-6 = Jacobian-grade per the
2026-07-14 tolerance finding); a non-default PLA_TOL writes kernel_result_tol{TOL}.json.

Runs standalone (no osse_config observing system): builds a throwaway 3-band Mie table
(fixed quadrature) and drives pydisort_riccati_jax directly.

Usage:  pla2000_table3a.py [retrieval|kernel|all]      (default all)
        The kernel stage auto-runs the retrieval stage first if its outputs
        (runs/_pla2000_table3a/{mie3band.npz,result.json}) are missing.
"""
import sys
import os
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pydisort_riccati_jax import runtime_setup                       # noqa: E402
runtime_setup.setup()
import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
from pydisort_riccati_jax import optics_table as ot                  # noqa: E402
from pydisort_riccati_jax.solver import pydisort_riccati_jax         # noqa: E402
from scipy.interpolate import PchipInterpolator                      # noqa: E402

OUT = Path("runs/_pla2000_table3a"); OUT.mkdir(parents=True, exist_ok=True)
TBL_PATH = OUT / "mie3band.npz"
BANDS = np.array([1.6, 2.2, 3.7])
MU0, MU_V = 0.65, 0.85
NQUAD, NLEG = 32, 256
RETRIEVAL_TOL = 1e-5                                  # stage "retrieval" (radiance-grade)
KERNEL_TOL = float(os.environ.get("PLA_TOL", "1e-6"))  # stage "kernel" (Jacobian-grade)
CLOUDS = [(15.0, 4.0, 10.0), (10.0, 6.0, 15.0), (8.0, 5.0, 12.0), (5.0, 8.0, 12.0)]
LIB_RE = np.arange(3.0, 17.0 + 1e-9, 1.0)
# Pla2000 Table 3a, order = bands (1.6, 2.2, 3.7):
# column 1 ("Retrieval, µm") alone:
PUBLISHED = {("D", 15.0): (7.3, 8.1, 9.4), ("D", 10.0): (10.8, 11.8, 13.4),
             ("D", 8.0): (8.0, 9.1, 10.5), ("D", 5.0): (10.0, 10.3, 10.9),
             ("B", 15.0): (8.8, 9.3, 9.9), ("B", 10.0): (13.0, 13.6, 14.5),
             ("B", 8.0): (10.2, 10.6, 11.4), ("B", 5.0): (10.4, 10.7, 11.2)}
# (retrieval, w_m estimate, w_N estimate) per (profile, tc, band):
PUB_FULL = {("D", 15.0): [(7.3, 7.9, 8.1), (8.1, 8.3, 8.4), (9.4, 9.3, 9.3)],
            ("D", 10.0): [(10.8, 11.3, 12.0), (11.8, 11.7, 12.2), (13.4, 13.3, 13.3)],
            ("D", 8.0): [(8.0, 8.9, 9.7), (9.1, 9.1, 9.7), (10.5, 10.3, 10.4)],
            ("D", 5.0): [(10.0, 10.2, 10.3), (10.3, 10.3, 10.7), (10.9, 10.8, 11.0)],
            ("B", 15.0): [(8.8, 9.1, 9.3), (9.3, 9.3, 9.4), (9.9, 9.9, 9.9)],
            ("B", 10.0): [(13.0, 13.2, 13.7), (13.6, 13.5, 13.9), (14.5, 14.4, 14.5)],
            ("B", 8.0): [(10.2, 10.5, 10.9), (10.6, 10.6, 11.0), (11.4, 11.4, 11.4)],
            ("B", 5.0): [(10.4, 10.6, 11.0), (10.7, 10.7, 11.0), (11.2, 11.1, 11.3)]}


def load_or_build_table():
    """Throwaway 3-band Mie table (monochromatic band centers, per DESIGN §8b)."""
    if TBL_PATH.exists():
        return ot.load_table(TBL_PATH)
    t0 = time.time()
    tbl = ot.build_re_table(BANDS, 2.0, 20.0, 73, 0.1, n_radii=2048, NLeg=NLEG,
                            n_gl=1024, quadrature="fixed")
    ot.save_table(tbl, TBL_PATH)
    print(f"3-band Mie table built [{time.time()-t0:.0f}s]", flush=True)
    return tbl


def profile_re(kind, tau, tc, rb, rt):
    if kind == "B":                                   # adiabatic (x=1): exponent 1/5
        return (rt**5 - (rt**5 - rb**5) * tau / tc) ** 0.2
    return rt - (rt - rb) * tau / tc                  # D: linear in tau


# ---------------------------------------------------------------------------
# Stage 1 — recreate the published "Retrieval" column
# ---------------------------------------------------------------------------

def stage_retrieval(tbl):
    re_axis = np.linspace(tbl["re_min"], tbl["re_max"], int(tbl["n_re"]))
    OMEGA = np.asarray(tbl["omega"])                  # (3, n_re)
    LEG = np.asarray(tbl["leg"])                      # (3, n_re, NLEG)

    def optics_of_re(b, re_vals):
        """(omega, leg) linearly interpolated in r_e for band b (numpy, host-side)."""
        om = np.interp(re_vals, re_axis, OMEGA[b])
        lg = np.empty((len(np.atleast_1d(re_vals)), NLEG))
        for l in range(NLEG):
            lg[:, l] = np.interp(re_vals, re_axis, LEG[b, :, l])
        return om, lg

    def solve_layers(b, edges, re_layers):
        """Azimuthally-averaged u0 at MU_V for a layered cloud (piecewise-constant optics)."""
        om, lg = optics_of_re(b, re_layers)
        om_j, lg_j, ed_j = jnp.asarray(om), jnp.asarray(lg), jnp.asarray(edges)
        def omega_func(tau):
            i = jnp.clip(jnp.searchsorted(ed_j, tau, side="right") - 1, 0, len(re_layers) - 1)
            return om_j[i]
        def leg_func(tau):
            i = jnp.clip(jnp.searchsorted(ed_j, tau, side="right") - 1, 0, len(re_layers) - 1)
            return lg_j[i]
        mu_pos, _, u0, _, _ = pydisort_riccati_jax(
            float(edges[-1]), omega_func, leg_func, NQUAD, MU0, 1.0, 0.0,
            NLeg=NQUAD, NLeg_all=NLEG, NFourier=1, tol=RETRIEVAL_TOL, delta_M_scaling=True)
        return float(np.interp(MU_V, np.asarray(mu_pos), np.asarray(u0)))

    t0 = time.time()
    libraries = {}                                    # (tc, band) -> PCHIP of R(re)
    results, rows = {}, []
    for tc, rb, rt in CLOUDS:
        edges = np.arange(0.0, tc + 1e-9, 0.25)
        mids = edges[:-1] + 0.125
        for b in range(3):
            if (tc, b) not in libraries:
                R = [solve_layers(b, np.array([0.0, tc]), np.array([c])) for c in LIB_RE]
                libraries[(tc, b)] = PchipInterpolator(LIB_RE, R)
                print(f"library tc={tc} band={BANDS[b]} done [{time.time()-t0:.0f}s]", flush=True)
        for kind in ("D", "B"):
            re_layers = np.round(profile_re(kind, mids, tc, rb, rt))  # INTEGER r_e per layer
            got = []
            for b in range(3):
                Rc = solve_layers(b, edges, re_layers)
                lib = libraries[(tc, b)]
                grid = np.linspace(LIB_RE[0], LIB_RE[-1], 2000)
                got.append(float(grid[np.argmin(np.abs(lib(grid) - Rc))]))
            pub = PUBLISHED[(kind, tc)]
            results[f"{kind}_{tc:g}"] = dict(ours=[round(g, 2) for g in got], published=list(pub))
            for b in range(3):
                rows.append((kind, tc, BANDS[b], got[b], pub[b], got[b] - pub[b]))
            print(f"profile {kind} tc={tc}: ours={[f'{g:.1f}' for g in got]} "
                  f"published={pub} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'prof':>4} {'tc':>4} {'band':>5} {'ours':>6} {'Pla2000':>8} {'delta':>6}")
    for kind, tc, band, g, p, d in rows:
        print(f"{kind:>4} {tc:>4.0f} {band:>5.1f} {g:>6.2f} {p:>8.1f} {d:>+6.2f}")
    d = np.array([r[5] for r in rows])
    print(f"\n|delta|: median={np.median(np.abs(d)):.2f} max={np.abs(d).max():.2f} µm "
          f"(Pla2000 internal w_m-vs-retrieval agreement was 0.1-0.9 µm)")
    (OUT / "result.json").write_text(json.dumps(dict(
        rows=[dict(profile=r[0], tc=r[1], band=r[2], ours=round(r[3], 2),
                   published=r[4], delta=round(r[5], 2)) for r in rows],
        spec_notes="v_eff=0.1 gamma assumed; Segelstein index; NFourier=1 (azimuthal avg); "
                   "band-shared tau (Qe=2 scaling); solar-only", runtime_s=round(time.time()-t0, 1))))
    print("saved", OUT / "result.json")


# ---------------------------------------------------------------------------
# Stage 2 — score OUR Eq-(3) kernel on Platnick's own clouds
# ---------------------------------------------------------------------------

def stage_kernel(tbl):
    re_axis = np.linspace(tbl["re_min"], tbl["re_max"], int(tbl["n_re"]))
    OMEGA = np.asarray(tbl["omega"]); LEG = np.asarray(tbl["leg"])
    ours_ret = json.loads((OUT / "result.json").read_text())["rows"]  # our recreated retrievals

    def make_u0_fn(b, edges):
        re_ax = jnp.asarray(re_axis); om_ax = jnp.asarray(OMEGA[b]); LG = jnp.asarray(LEG[b])
        ed = jnp.asarray(edges); nlay = len(edges) - 1
        def u0_of_re(re_layers):
            i = jnp.clip(jnp.searchsorted(re_ax, re_layers) - 1, 0, len(re_axis) - 2)
            t = (re_layers - re_ax[i]) / (re_ax[i + 1] - re_ax[i])
            om_l = om_ax[i] * (1 - t) + om_ax[i + 1] * t
            lg_l = LG[i] * (1 - t)[:, None] + LG[i + 1] * t[:, None]
            def omega_func(tau):
                j = jnp.clip(jnp.searchsorted(ed, tau, side="right") - 1, 0, nlay - 1)
                return om_l[j]
            def leg_func(tau):
                j = jnp.clip(jnp.searchsorted(ed, tau, side="right") - 1, 0, nlay - 1)
                return lg_l[j]
            mu_pos, _, u0, _, _ = pydisort_riccati_jax(
                float(edges[-1]), omega_func, leg_func, NQUAD, MU0, 1.0, 0.0,
                NLeg=NQUAD, NLeg_all=NLEG, NFourier=1, tol=KERNEL_TOL, delta_M_scaling=True)
            return jnp.interp(MU_V, mu_pos, u0)
        return u0_of_re

    t0 = time.time()
    rows = []
    for tc, rb, rt in CLOUDS:
        edges = np.arange(0.0, tc + 1e-9, 0.25)
        mids = edges[:-1] + 0.125
        for kind in ("D", "B"):
            re_layers = np.round(profile_re(kind, mids, tc, rb, rt)).astype(float)
            for b in range(3):
                K = np.asarray(jax.grad(make_u0_fn(b, edges))(jnp.asarray(re_layers)))
                r_sgn = float(np.sum(K * re_layers) / np.sum(K))
                r_abs = float(np.sum(np.abs(K) * re_layers) / np.sum(np.abs(K)))
                ret_pub, wm_pub, wn_pub = PUB_FULL[(kind, tc)][b]
                ret_ours = next(r["ours"] for r in ours_ret
                                if r["profile"] == kind and r["tc"] == tc
                                and abs(r["band"] - BANDS[b]) < 0.01)
                rows.append(dict(profile=kind, tc=tc, band=float(BANDS[b]),
                                 ret_pub=ret_pub, ret_ours=ret_ours, wm_pub=wm_pub, wn_pub=wn_pub,
                                 r_signed=round(r_sgn, 2), r_abs=round(r_abs, 2)))
                print(f"{kind} tc={tc:g} {BANDS[b]}um: K done, r*_signed={r_sgn:.2f} "
                      f"(pub ret {ret_pub}, pub w_m {wm_pub}) [{time.time()-t0:.0f}s]", flush=True)
                # Every (cloud, band) closure compiles fresh and is used ONCE; keeping the
                # cached executables ratchets memory until LLVM codegen mmap fails
                # ("Failed to materialize symbols" / "LLVM compilation error: Cannot
                # allocate memory" — 3 job casualties, 2026-07-14).
                jax.clear_caches()

    print(f"\n{'prof':>4} {'tc':>4} {'band':>5} {'ret_pub':>8} {'ret_ours':>8} {'wm_pub':>7} "
          f"{'signed':>7} {'|e_wm|':>6} {'|e_sgn|':>7} {'|e_abs|':>7}")
    e_wm, e_sg, e_ab, e_wn = [], [], [], []
    for r in rows:
        # score every estimate against the SAME target: the published retrieval column
        ewm = abs(r["wm_pub"] - r["ret_pub"]); esg = abs(r["r_signed"] - r["ret_pub"])
        eab = abs(r["r_abs"] - r["ret_pub"]); ewn = abs(r["wn_pub"] - r["ret_pub"])
        e_wm.append(ewm); e_sg.append(esg); e_ab.append(eab); e_wn.append(ewn)
        print(f"{r['profile']:>4} {r['tc']:>4.0f} {r['band']:>5.1f} {r['ret_pub']:>8.1f} "
              f"{r['ret_ours']:>8.2f} {r['wm_pub']:>7.1f} {r['r_signed']:>7.2f} "
              f"{ewm:>6.2f} {esg:>7.2f} {eab:>7.2f}")
    for name, e in (("Pla2000 w_m", e_wm), ("Pla2000 w_N", e_wn),
                    ("OURS signed", e_sg), ("OURS |K|", e_ab)):
        e = np.array(e)
        print(f"{name:>12}: |err| median={np.median(e):.3f} mean={e.mean():.3f} max={e.max():.2f} um")
    out_json = OUT / ("kernel_result.json" if "PLA_TOL" not in os.environ
                      else f"kernel_result_tol{KERNEL_TOL:g}.json")
    out_json.write_text(json.dumps(dict(
        rows=rows, tol=KERNEL_TOL, note="errors scored vs the PUBLISHED retrieval column; "
        "our solver recreates that column (result.json)",
        runtime_s=round(time.time() - t0, 1))))
    print("saved", out_json)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("retrieval", "kernel", "all"):
        raise SystemExit(f"usage: {sys.argv[0]} [retrieval|kernel|all]")
    tbl = load_or_build_table()
    if stage in ("retrieval", "all") or not (OUT / "result.json").exists():
        stage_retrieval(tbl)
    if stage in ("kernel", "all"):
        stage_kernel(tbl)
