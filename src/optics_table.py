"""optics_table.py — miepython-grounded r_e → (ω, Q_ext, Legendre) lookup table.

The production replacement for the JAX-Mie build path
(``miejax_lite.mie_avg_legendre``). **The retrieval / information-content
Jacobian does not differentiate through Mie**: the differentiable forward gets
optics via ``table_lookup(opt, r_e(τ))`` — a *differentiable linear interpolation*
of this precomputed table — and the table itself (``opt``) is built once, outside
the gradient. So the only thing that must stay in JAX is the tiny ``table_lookup``
hot path; the *table build* is offline and is done here with **miepython** (Bohren
& Huffman; numba-accelerated via miepython's own JIT) over the gamma size
distribution. ``miejax_lite`` is retained for legacy / validation only.

Why swap (DESIGN_DECISIONS §13): autodiff independence from Mie (above) means the
JAX-Mie front-end buys nothing for the retrieval, while miepython is the field
reference implementation and trivially reaches the strong-absorption bands (e.g.
3.7 µm) that motivated the band superset.

API mirrors ``miejax_lite._table`` so it is a drop-in for the workers:

    build_re_table(wavelengths, re_min, re_max, n_re, v_eff, ...) -> table dict
    select_channel(table, channel)   -> opt  (single-λ r_e-table)
    table_lookup(opt, r_e)           -> (omega, leg)   [JAX, differentiable in r_e]

plus disk caching (the table is **profile-independent** — built once, loaded by
every per-profile worker / HPC array task):

    build_or_load_table(...)  -> table dict   (load if a matching cache exists)
    save_table / load_table

The Legendre extraction is bit-for-bit the same definition miejax_lite uses
(P(μ)=2(|S1|²+|S2|²)/(x²·Q_sca), χ_ℓ = ½∫P P_ℓ dμ by Gauss–Legendre, χ_0≡1,
g=χ_1), but the Mie coefficients a_n, b_n come from ``miepython.an_bn`` — validated
to match ``miepython.efficiencies`` (Q_ext/Q_sca/g) to round-off and miejax_lite on
the shared bands (see ``tests/supplementary/validate_optics_table.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.special

import jax.numpy as jnp  # only for the differentiable table_lookup hot path

__all__ = [
    "water_refractive_index",
    "build_re_table",
    "build_or_load_table",
    "save_table",
    "load_table",
    "select_channel",
    "table_lookup",
]

_DATA = Path(__file__).resolve().parent / "data" / "segelstein81_index.csv"
_TABLE_RI = np.genfromtxt(_DATA, delimiter=",", skip_header=1)
_WL, _N, _K = _TABLE_RI[:, 0], _TABLE_RI[:, 1], _TABLE_RI[:, 2]  # μm, n, k(>0)


def water_refractive_index(wavelength):
    """Liquid-water ``(m_real, m_imag)`` at ``wavelength`` (μm); ``m = n − k·i``.

    Segelstein (1981) table (0.2–5.0 µm, covers the 0.55–3.7 µm band set), linear
    interpolation. ``m_imag`` is returned **negative** (the miepython / DISORT
    sign convention). Vendored from miejax_lite so the production path carries no
    miejax dependency.
    """
    wl = float(wavelength)
    if wl < _WL[0] or wl > _WL[-1]:
        raise ValueError(f"wavelength {wl} µm outside [{_WL[0]}, {_WL[-1]}] µm")
    return float(np.interp(wl, _WL, _N)), -float(np.interp(wl, _WL, _K))


# ---------------------------------------------------------------------------
# Angular precompute (μ-only) for the exact Legendre projection
# ---------------------------------------------------------------------------
def _legendre_precompute(max_nstop, NLeg, n_gl):
    """π_n/τ_n angular functions + Legendre basis on ``n_gl`` Gauss–Legendre nodes.

    Numpy port of ``miejax_lite.mie_legendre_precompute`` (same recurrences), used
    to project the Mie phase function onto Legendre moments χ_ℓ, ℓ < ``NLeg``.
    """
    mu, w = np.polynomial.legendre.leggauss(n_gl)            # nodes in [-1, 1]
    pi_mat = np.zeros((n_gl, max_nstop))
    tau_mat = np.zeros((n_gl, max_nstop))
    pi_mat[:, 0] = 1.0                                       # π_1
    pi_nm2 = np.zeros(n_gl)                                  # π_0
    for n in range(1, max_nstop + 1):
        k = n - 1
        tau_mat[:, k] = n * mu * pi_mat[:, k] - (n + 1) * pi_nm2
        if n < max_nstop:
            pi_mat[:, n] = ((2 * n + 1) * mu * pi_mat[:, k] - (n + 1) * pi_nm2) / n
        pi_nm2 = pi_mat[:, k].copy()
    Pl_mat = np.empty((NLeg, n_gl))
    for ell in range(NLeg):
        Pl_mat[ell, :] = scipy.special.eval_legendre(ell, mu)
    n = np.arange(1, max_nstop + 1, dtype=float)
    scale = (2.0 * n + 1.0) / ((n + 1.0) * n)
    return dict(pi_mat=pi_mat, tau_mat=tau_mat, Pl_mat=Pl_mat, w_gl=w,
                scale=scale, cn=(2.0 * n + 1.0), max_nstop=max_nstop)


def _wiscombe_nstop(x):
    return int(np.floor(x + 4.05 * x ** (1.0 / 3.0) + 2.0)) + 1


def _gamma_weights(r_eff, v_eff, r):
    """Hansen & Travis (1974) modified-gamma n(r); n(r) ∝ r^α exp(−b r) with
    α = 1/v_eff − 3, b = 1/(v_eff r_eff). Normalized to unit area (the norm cancels
    in the ratios below; kept for clarity). Matches miejax ``_gamma_dist_weights``."""
    alpha = 1.0 / v_eff - 3.0
    b = (alpha + 3.0) / r_eff
    log_N = (alpha + 1.0) * np.log(b) - scipy.special.gammaln(alpha + 1.0)
    return np.exp(log_N + alpha * np.log(r) - b * r)


def _mie_radius_block(m, x, pc):
    """Per-radius (Q_ext, Q_sca, χ_ℓ) for a vector of size parameters ``x``.

    a_n, b_n from ``miepython.an_bn`` (numba-accelerated), padded/truncated to
    ``pc['max_nstop']``; bulk efficiencies and the Legendre projection use the same
    Bohren–Huffman reductions miejax_lite uses (validated to round-off)."""
    import miepython as mp

    M = pc["max_nstop"]
    nr = x.shape[0]
    A = np.zeros((nr, M), complex)
    B = np.zeros((nr, M), complex)
    for i, xi in enumerate(x):
        a, b = mp.an_bn(m, float(xi), 0)                    # length nstop(xi); n_pole=0 (miepython 3.x)
        ns = min(a.shape[0], M)
        A[i, :ns] = a[:ns]
        B[i, :ns] = b[:ns]
    cn = pc["cn"]                                           # 2n+1
    x2 = x ** 2
    qext = 2.0 / x2 * ((A.real + B.real) @ cn)
    qsca = 2.0 / x2 * ((np.abs(A) ** 2 + np.abs(B) ** 2) @ cn)
    # Amplitude functions S1, S2 at the GL nodes; phase function P(μ) → χ_ℓ.
    sa = A * pc["scale"]
    sb = B * pc["scale"]
    S1 = sa @ pc["pi_mat"].T + sb @ pc["tau_mat"].T        # (nr, n_gl)
    S2 = sa @ pc["tau_mat"].T + sb @ pc["pi_mat"].T
    P = 2.0 * (np.abs(S1) ** 2 + np.abs(S2) ** 2) / (x2[:, None] * qsca[:, None])
    chi = 0.5 * (pc["w_gl"] * P) @ pc["Pl_mat"].T          # (nr, NLeg); χ_0=1, χ_1=g
    return qext, qsca, chi


def build_re_table(wavelengths, re_min, re_max, n_re, v_eff, *,
                   n_radii=600, NLeg=128, n_gl=1024, max_nstop=None, m=None):
    """Build the (r_e, λ) → (ω, Leg_coeffs, Q_ext) table with miepython.

    Drop-in for ``miejax_lite.build_re_table`` (same output dict keys), but the
    Mie is miepython (offline, numba). One r_e-table per entry of ``wavelengths``
    on a **uniform** r_e grid (``n_re`` points over ``[re_min, re_max]`` µm), each
    gamma-averaged over the size distribution (``v_eff``) on a ``3·r_e`` radius grid
    (``n_radii`` points) — identical construction to the JAX-Mie path.

    ``max_nstop`` (Mie series length) defaults to the global Wiscombe order for the
    largest size parameter (2π·3·re_max/min λ), so the largest droplets at the
    shortest band are **not** truncated. ``m`` optionally overrides the per-band
    water refractive index (sequence of ``(n, k)`` with ``k>0``)."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, float))
    re_grid = np.linspace(re_min, re_max, n_re)
    if max_nstop is None:
        x_max = 2.0 * np.pi * (3.0 * re_max) / float(wavelengths.min())
        max_nstop = _wiscombe_nstop(x_max) + 4
    pc = _legendre_precompute(int(max_nstop), int(NLeg), int(n_gl))

    omega = np.empty((wavelengths.size, n_re))
    leg = np.empty((wavelengths.size, n_re, NLeg))
    qext = np.empty((wavelengths.size, n_re))
    for wi, lam in enumerate(wavelengths):
        if m is None:
            m_real, m_imag = water_refractive_index(float(lam))
        else:
            m_real, m_imag = m[wi]
        mc = complex(m_real, m_imag if m_imag <= 0 else -m_imag)
        for ri, re in enumerate(re_grid):
            r = np.linspace(1e-3, 3.0 * float(re), n_radii)
            x = 2.0 * np.pi * r / float(lam)
            qe, qs, chi = _mie_radius_block(mc, x, pc)
            w = _gamma_weights(float(re), v_eff, r) * r ** 2  # Q_sca·r²·n(r) weight base
            tw = np.trapezoid(w, r)
            sca_int = np.trapezoid(qs * w, r)
            omega[wi, ri] = sca_int / np.trapezoid(qe * w, r)
            qext[wi, ri] = np.trapezoid(qe * w, r) / tw
            wq = qs * w
            leg[wi, ri] = np.trapezoid(wq[:, None] * chi, r, axis=0) / np.trapezoid(wq, r)

    return {
        "wavelengths": np.asarray(wavelengths),
        "re_min": float(re_min), "re_max": float(re_max), "n_re": int(n_re),
        "dr": float((re_max - re_min) / (n_re - 1)), "v_eff": float(v_eff),
        "NLeg": int(NLeg), "max_nstop": int(max_nstop),
        "omega": omega, "leg": leg, "qext": qext,
    }


# ---------------------------------------------------------------------------
# Channel resolution + differentiable lookup (ported from miejax_lite._table)
# ---------------------------------------------------------------------------
def _opt1d(table, omega, leg, qext, wavelength):
    return {"re_min": table["re_min"], "re_max": table["re_max"],
            "n_re": table["n_re"], "dr": table["dr"],
            "omega": jnp.asarray(omega), "leg": jnp.asarray(leg),
            "qext": jnp.asarray(qext), "wavelength": float(wavelength)}


def select_channel(table, channel):
    """Resolve an exact built wavelength by index → single-λ r_e-table (``opt``)."""
    return _opt1d(table, table["omega"][channel], table["leg"][channel],
                  table["qext"][channel], np.asarray(table["wavelengths"])[channel])


def table_lookup(opt, r_e):
    """O(1) differentiable linear lookup of ``(omega, Leg_coeffs)`` at ``r_e``.

    Ported verbatim from ``miejax_lite.table_lookup`` — the one piece that stays in
    JAX (the per-τ hot path inside the solver). Out-of-range ``r_e`` clamps to the
    grid ends (gradient → 0); differentiable in ``r_e`` (gradient = table slope)."""
    re_min, dr, n_re = opt["re_min"], opt["dr"], opt["n_re"]
    omega_grid, leg_grid = opt["omega"], opt["leg"]
    idx = jnp.clip((r_e - re_min) / dr, 0.0, n_re - 1.0)
    i0 = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n_re - 2)
    frac = idx - i0
    omega = jnp.take(omega_grid, i0, axis=0) * (1.0 - frac) \
        + jnp.take(omega_grid, i0 + 1, axis=0) * frac
    fl = jnp.expand_dims(frac, -1)
    leg = jnp.take(leg_grid, i0, axis=0) * (1.0 - fl) \
        + jnp.take(leg_grid, i0 + 1, axis=0) * fl
    return omega, leg


# ---------------------------------------------------------------------------
# Disk cache (table is profile-independent: build once, load everywhere)
# ---------------------------------------------------------------------------
def _signature(wavelengths, re_min, re_max, n_re, v_eff, NLeg):
    wl = ",".join(f"{w:.4f}" for w in np.atleast_1d(wavelengths))
    return f"wl=[{wl}] re=[{re_min},{re_max}]/{n_re} veff={v_eff} NLeg={NLeg}"


def save_table(table, path):
    """Persist a built table to ``.npz`` (arrays + a provenance signature)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signature=_signature(table["wavelengths"], table["re_min"],
                                        table["re_max"], table["n_re"],
                                        table["v_eff"], table["NLeg"]),
             wavelengths=np.asarray(table["wavelengths"]),
             re_min=table["re_min"], re_max=table["re_max"], n_re=table["n_re"],
             dr=table["dr"], v_eff=table["v_eff"], NLeg=table["NLeg"],
             max_nstop=table["max_nstop"], omega=table["omega"],
             leg=table["leg"], qext=table["qext"])


def load_table(path):
    """Load a table saved by :func:`save_table`."""
    z = np.load(Path(path), allow_pickle=True)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k])
            for k in ("re_min", "re_max", "n_re", "dr", "v_eff", "NLeg",
                      "max_nstop", "wavelengths", "omega", "leg", "qext")}


def build_or_load_table(wavelengths, re_min, re_max, n_re, v_eff, *,
                        cache_path, NLeg=128, **kw):
    """Load the cached table if its signature matches, else build and cache it.

    The optics table depends only on (bands, r_e grid, v_eff, NLeg) — **not** on the
    cloud profile — so per-profile workers / HPC array tasks share one cache instead
    of each rebuilding it (the build is the expensive Mie pass; the table is reused)."""
    cache_path = Path(cache_path)
    want = _signature(wavelengths, re_min, re_max, n_re, v_eff, NLeg)
    if cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=True)
            if str(z["signature"]) == want:
                return load_table(cache_path)
        except (OSError, KeyError, ValueError):
            pass
    table = build_re_table(wavelengths, re_min, re_max, n_re, v_eff, NLeg=NLeg, **kw)
    save_table(table, cache_path)
    return table
