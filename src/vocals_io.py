"""vocals_io.py — VOCALS-REx in-situ cloud-profile loader for r_e(τ) retrieval.

Reads a VOCALS-REx C-130 flight netCDF (EOL dataset 89.002), extracts clean
vertical penetrations of a marine-stratocumulus layer, and returns each as a
truth profile ``(τ_grid, r_e(τ))`` (with cloud-base optical depth ``τ_bot`` and
base radius ``r_base``) for the OSSE retrieval demonstration.

This is a deliberately *simplified* Python port of Andrew Buggee's MATLAB
``readVocalsRex`` / ``find_verticalProfiles_VOCALS_REx_ver2``:

- **CDP-only optics.** We use only the Cloud Droplet Probe (``CCDP_RWO``, radii
  ≈1–25 µm); the large-drizzle 2DC probe is dropped. The in-cloud r_e range that
  sized the optics table ([2,25] µm) was itself CDP ``M₃/M₂``, and marine Sc
  droplet optics at the retrieval bands are CDP-dominated. (Noted in the docs.)
- **1 Hz.** The 10 Hz CDP bins (``sps10``) and 25 Hz altitude (``sps25``) are
  averaged to the 1 Hz ``Time`` grid.
- **Single-layer profile finder.** Contiguous in-cloud, vertically-traversed
  segments — without the MATLAB multilayer-reconciliation branches (not needed
  to pick one or two demonstration profiles).

The physics (r_e = ratio of 3rd to 2nd moment, τ = π ∫ Q_ext r_e² N_c dz from
cloud top down) follows the MATLAB exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- thresholds (Painemal & Zuidema 2011, as used in the MATLAB) -------------
LWC_THRESHOLD = 0.03    # g/m^3  — cloud boundary (P&Z 2011)
NC_THRESHOLD = 25.0     # #/cm^3 — in-cloud number-concentration floor
RHO_LW = 1.0e6          # g/m^3  — density of liquid water
QEXT_GEOM = 2.0         # geometric-optics extinction efficiency (λ ≪ r)


@dataclass
class CloudProfile:
    """One vertical penetration, ordered cloud-top (τ=0) → cloud-base (τ_bot)."""
    tau: np.ndarray        # optical depth, increasing from 0 at cloud top
    r_e: np.ndarray        # effective radius [µm] at each τ node
    v_eff: np.ndarray      # effective variance (Hansen & Travis 1974) at each node
    altitude: np.ndarray   # altitude [m] (decreasing: top → base)
    total_Nc: np.ndarray   # number concentration [cm^-3]
    lwc: np.ndarray        # liquid water content [g/m^3]
    ascending: bool        # was the aircraft ascending through the cloud?
    flight: str            # source flight id (e.g. 'RF01')

    @property
    def tau_bot(self) -> float:
        return float(self.tau[-1])

    @property
    def r_base(self) -> float:
        return float(self.r_e[-1])

    @property
    def r_top(self) -> float:
        return float(self.r_e[0])


# ----------------------------------------------------------------------------
# 1. Read + reduce one flight to 1 Hz CDP moments
# ----------------------------------------------------------------------------
def read_flight(path: str | Path) -> dict:
    """Load a VOCALS-REx flight netCDF; return 1 Hz CDP-derived fields.

    Returns a dict with ``time`` (s), ``altitude`` (m), ``total_Nc`` (cm^-3),
    ``r_e`` (µm, M₃/M₂), ``lwc`` (g/m^3), and ``bin_radii`` (µm, CDP centres).
    """
    import netCDF4 as nc

    path = Path(path)
    ds = nc.Dataset(path)
    try:
        ccdp = ds.variables["CCDP_RWO"]
        if ccdp.getncattr("DataQuality") != "Good":
            raise ValueError(f"{path.name}: CDP DataQuality not 'Good'")

        # CellSizes are *diameter* bin edges (µm); radii = /2. FirstBin/LastBin
        # (1-based, inclusive) select the valid bins.
        edges_d = np.asarray(ccdp.getncattr("CellSizes"), dtype=float)
        first = int(ccdp.getncattr("FirstBin"))
        last = int(ccdp.getncattr("LastBin"))
        r_edges = edges_d[first - 1:last + 1] / 2.0            # µm, (nbin+1,)
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])            # µm, (nbin,)

        # CCDP_RWO: (Time, sps10, Vector31) #/cm^3 per bin. Average sps10 → 1 Hz,
        # mask fill values, slice to the valid bins.
        Nc_bins = np.asarray(ccdp[:], dtype=float)             # (T, 10, 31)
        Nc_bins = np.where(Nc_bins <= ccdp.getncattr("_FillValue") + 1, 0.0, Nc_bins)
        Nc_bins = np.nan_to_num(Nc_bins, nan=0.0)
        Nc_bins = Nc_bins[:, :, first - 1:last].mean(axis=1)   # (T, nbin) #/cm^3

        time = np.asarray(ds.variables["Time"][:], dtype=float)
        altitude = np.asarray(ds.variables["ALTX"][:], dtype=float)
        if altitude.ndim > 1:                                  # (Time, sps25)
            altitude = altitude.mean(axis=1)
    finally:
        ds.close()

    # CDP moments (Hansen–Travis 1974): total N_c, r_e = M3/M2, v_eff, LWC.
    total_Nc = Nc_bins.sum(axis=1)                             # cm^-3
    m2 = (Nc_bins * r_cent ** 2).sum(axis=1)
    m3 = (Nc_bins * r_cent ** 3).sum(axis=1)
    m4 = (Nc_bins * r_cent ** 4).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_e = np.where(m2 > 0, m3 / m2, 0.0)                   # µm
        v_eff = np.where((m2 > 0) & (r_e > 0), m4 / (r_e ** 2 * m2) - 1.0, 0.0)
    # LWC = 4/3 π ρ Σ N r³, with N in cm^-3 and r in cm → g/m^3.
    r_cent_cm = r_cent * 1e-4
    lwc = 4.0 / 3.0 * np.pi * RHO_LW * (Nc_bins * r_cent_cm ** 3).sum(axis=1)

    return dict(flight=path.name.split(".")[0], time=time, altitude=altitude,
                total_Nc=total_Nc, r_e=r_e, v_eff=v_eff, lwc=lwc,
                bin_radii=r_cent)


# ----------------------------------------------------------------------------
# 2. Find clean vertical in-cloud profiles
# ----------------------------------------------------------------------------
def find_profiles(flight: dict, *, lwc_threshold: float = LWC_THRESHOLD,
                  nc_threshold: float = NC_THRESHOLD, min_len: int = 15,
                  min_depth: float = 30.0, monotone_frac: float = 0.85
                  ) -> list[CloudProfile]:
    """Extract contiguous in-cloud, vertically-traversed segments.

    A profile is a maximal run where ``total_Nc > nc_threshold`` and
    ``lwc > lwc_threshold``, spanning ≥ ``min_depth`` m of altitude with the
    aircraft moving monotonically (≥ ``monotone_frac`` of steps one sign).
    """
    alt, Nc, lwc, re = (flight["altitude"], flight["total_Nc"],
                        flight["lwc"], flight["r_e"])
    in_cloud = (Nc > nc_threshold) & (lwc > lwc_threshold) & (re > 0)

    profiles: list[CloudProfile] = []
    edges = np.diff(in_cloud.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if in_cloud[0]:
        starts = np.r_[0, starts]
    if in_cloud[-1]:
        ends = np.r_[ends, in_cloud.size]

    for s, e in zip(starts, ends):
        sl = slice(s, e)
        a = alt[sl]
        if a.size < min_len or (a.max() - a.min()) < min_depth:
            continue
        dz = np.diff(a)
        if dz.size == 0:
            continue
        frac_up = np.mean(dz > 0)
        ascending = frac_up >= 0.5
        if max(frac_up, 1 - frac_up) < monotone_frac:   # not a clean traverse
            continue
        prof = _build_profile(a, re[sl], flight["v_eff"][sl], Nc[sl],
                              lwc[sl], ascending, flight["flight"])
        if prof is not None:
            profiles.append(prof)
    return profiles


def _build_profile(altitude, re, ve, Nc, lwc, ascending, flight) -> CloudProfile | None:
    """Order top→base and integrate τ(z) = π ∫ Q_ext r_e² N_c dz from the top."""
    # Order so index 0 is cloud TOP (max altitude), last is cloud BASE.
    order = np.argsort(-altitude)
    altitude, re, ve, Nc, lwc = (altitude[order], re[order], ve[order],
                                 Nc[order], lwc[order])

    # τ from cloud top down: integrand [1/m] = π Q_ext r_e² N_c with r_e in m,
    # N_c in m^-3.  π Q (re·1e-6)² (Nc·1e6) = π Q · re²[µm] · Nc[cm^-3] · 1e-6.
    integrand = np.pi * QEXT_GEOM * re ** 2 * Nc * 1e-6        # 1/m
    depth = altitude[0] - altitude                            # 0 at top, grows
    tau = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(depth))])
    if not np.all(np.isfinite(tau)) or tau[-1] <= 0:
        return None
    return CloudProfile(tau=tau, r_e=re, v_eff=ve, altitude=altitude,
                        total_Nc=Nc, lwc=lwc, ascending=bool(ascending),
                        flight=flight)


# ----------------------------------------------------------------------------
# 3. Convenience: scan flights and pick by target thickness
# ----------------------------------------------------------------------------
def load_all_profiles(data_dir: str | Path, *, flights: list[str] | None = None,
                      **kw) -> list[CloudProfile]:
    """Read every (or selected) flight in ``data_dir`` and return all profiles."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.nc"))
    if flights is not None:
        paths = [p for p in paths if any(p.name.startswith(f) for f in flights)]
    out: list[CloudProfile] = []
    for p in paths:
        try:
            out.extend(find_profiles(read_flight(p), **kw))
        except (OSError, ValueError):
            continue
    return out


def pick_profile(profiles: list[CloudProfile], target_tau: float
                 ) -> CloudProfile:
    """Return the profile whose ``τ_bot`` is closest to ``target_tau``."""
    return min(profiles, key=lambda p: abs(p.tau_bot - target_tau))
