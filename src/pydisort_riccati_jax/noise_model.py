"""noise_model.py — per-observation measurement-noise models for the ToA OSSE.

The retrieval observable is the bidirectional reflectance factor
``ρ = π u / (μ0 I0)`` (``RetrievalForward.forward`` / ``osse_observation``). A
:class:`NoiseModel` maps an observation vector ``ρ`` → per-element 1σ noise
``σ(ρ)``, which is used two ways:

1. **Account for noise** — build the *assumed* error covariance
   ``Se = diag(σ²)`` that the retrieval inverts for weighting (always needed,
   even with no perturbation).
2. **Add noise** — draw a random realization ``y = ρ + N(0, σ)`` for a noisy
   synthetic measurement (:meth:`NoiseModel.sample`).

**Default is NOISELESS** (the OSSE decision, DESIGN §10b): ``osse_observation``
adds nothing unless a model is supplied. A model is still used to *build* ``Se``.

The conceptual point (recorded in DESIGN §12): "noise" here is **measurement
noise on the ToA radiances** — instrument noise of the spaceborne radiometer —
*not* uncertainty in the VOCALS-REx in-situ truth profiles (those are the ground
truth and could equally be synthetic/GCM). So the model is grounded in the
PACE instrument specs, not in VOCALS.

The three-term σ(ρ) (the general form; see DESIGN §12 for the physics)::

    σ(ρ) = sqrt( (k_cal·ρ)²            # calibration / radiometric accuracy (flat-relative)
                 + ρ·ρ_ref / SNR_ref²  # photon shot noise (∝ √ρ); OFF when SNR_ref = inf
                 + floor² )            # read/dark/quantization (additive, signal-independent)

The three sources are independent ⇒ added in quadrature. ``k_cal·ρ`` does *not*
average down with brightness (a 2 % gain error is 2 % on any pixel); the shot
term's *relative* size ``√(ρ_ref/ρ)/SNR_ref`` shrinks as the scene brightens, so
it is subdominant for bright clouds; the floor matters only for dark pixels.

**OCI-SWIR population (the production default, :func:`oci_swir`).** Calibration-
relative ("Option B"): ``k_cal`` from the documented OCI/HARP2 radiometric
accuracy (1–3 %; PACE MRD §3.7 absolute-gain uncertainty), shot term OFF, small
floor. The shot term ("Option A") is left wired-but-off because the OCI
SNR-at-L_typ table could not be cleanly sourced (the PACE MRD tables are embedded
images; the SNR spec lives in an external ``.xlsx``) — populate ``snr_ref`` /
``rho_ref`` later to switch it on with no refactor. See OUTSTANDING §K.

Scope (per the user's firm choices, 2026-06-19): **OCI-SWIR intensity only.**
HARP2 (VIS multi-angle) and polarized / DoLP noise are deferred with the
polarized-cloudbow observable (OUTSTANDING §I, §K).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Three-term per-observation reflectance-noise model (see module docstring).

    Each coefficient is either a **scalar** (applied to every observation) or a
    **per-band array** of length ``n_bands`` (broadcast over that band's view
    angles). The observation layout is **band-major** —
    ``y = [band0×views, band1×views, …]`` — matching ``RetrievalForward.forward``,
    so observation ``i`` belongs to band ``i // (m // n_bands)``.

    Parameters
    ----------
    k_cal : relative radiometric-accuracy coefficient (e.g. 0.02 = 2 %).
    snr_ref : SNR at the reference reflectance ``rho_ref`` (shot term). ``inf``
        (default) turns the shot term off → "Option B".
    rho_ref : reference reflectance at which ``snr_ref`` is quoted (the L_typ
        analogue, expressed in reflectance units).
    floor : additive reflectance floor (read/dark/quantization).
    name : label for provenance / plotting.
    """

    k_cal: float | np.ndarray
    snr_ref: float | np.ndarray = np.inf
    rho_ref: float | np.ndarray = 1.0
    floor: float | np.ndarray = 0.0
    name: str = "custom"

    @staticmethod
    def _per_obs(coeff, band, n_bands):
        arr = np.atleast_1d(np.asarray(coeff, float))
        if arr.size == 1:
            return np.full(band.shape, arr.item())
        if arr.size == n_bands:
            return arr[band]
        raise ValueError(
            f"coefficient must be scalar or length n_bands={n_bands}, got {arr.size}")

    def sigma(self, rho, n_bands=1):
        """Per-observation 1σ noise on the reflectance vector ``rho``.

        ``rho`` is the (band-major) observation vector of length ``m``; ``n_bands``
        is how many equal-size band blocks it splits into (``m % n_bands == 0``).
        """
        rho = np.abs(np.asarray(rho, float)).ravel()
        m = rho.size
        if n_bands < 1 or m % n_bands != 0:
            raise ValueError(f"len(rho)={m} not divisible by n_bands={n_bands}")
        band = np.repeat(np.arange(n_bands), m // n_bands)        # band-major
        kc = self._per_obs(self.k_cal, band, n_bands)
        snr = self._per_obs(self.snr_ref, band, n_bands)
        rr = self._per_obs(self.rho_ref, band, n_bands)
        fl = self._per_obs(self.floor, band, n_bands)
        with np.errstate(divide="ignore", invalid="ignore"):
            shot2 = np.where(np.isfinite(snr) & (snr > 0), rho * rr / snr ** 2, 0.0)
        return np.sqrt((kc * rho) ** 2 + shot2 + fl ** 2)

    def Se(self, rho, n_bands=1):
        """Assumed measurement-error covariance ``diag(σ²)`` (what the retrieval inverts)."""
        s = self.sigma(rho, n_bands)
        return np.diag(s ** 2)

    def sample(self, rho, n_bands=1, seed=0):
        """A Gaussian noise *realization* ``ρ + N(0, σ)`` (for a noisy synthetic obs)."""
        rho = np.asarray(rho, float)
        s = self.sigma(rho, n_bands).reshape(rho.shape)
        rng = np.random.default_rng(seed)
        return rho + rng.standard_normal(rho.shape) * s


# ---------------------------------------------------------------------------
# Instrument presets
# ---------------------------------------------------------------------------

def oci_swir(k_cal=0.02, floor=1e-3, snr_ref=np.inf, rho_ref=1.0):
    """PACE **OCI SWIR** intensity noise — the production default (Option B).

    Calibration-relative: ``σ ≈ k_cal·ρ`` (plus a small floor), with ``k_cal``
    from the documented OCI/HARP2 radiometric accuracy of **1–3 %** (PACE MRD
    §3.7 absolute-gain uncertainty; the SWIR cloud-reflectance regime is
    calibration-dominated because clouds are bright, so the shot term is
    subdominant). Default ``k_cal=0.02`` (2 %) sits in that band.

    The shot term is **off by default** (``snr_ref=inf``) because OCI's
    SNR-at-L_typ table could not be cleanly sourced (MRD tables are images, SNR
    is in an external ``.xlsx``) — pass ``snr_ref`` / ``rho_ref`` (in reflectance
    units) to switch it on. ``k_cal`` may be a per-band array (length n_bands) to
    differentiate the SWIR channels. See OUTSTANDING §K.
    """
    return NoiseModel(k_cal=k_cal, snr_ref=snr_ref, rho_ref=rho_ref,
                      floor=floor, name="OCI-SWIR")


def generic_relative(rel=0.03, floor=6e-4):
    """The repo's historical hand-picked model, as a 3-term instance.

    Approximates the legacy inline ``Se = diag((0.03·max(|y|, 0.02))²)`` (a
    *relative* 3 % with a hard absolute floor at ``0.03·0.02 = 6e-4``) by the
    smooth quadrature form ``σ = sqrt((rel·ρ)² + floor²)``. Kept so pre-PACE
    OSSE results remain reproducible; ``oci_swir`` is the grounded replacement.
    """
    return NoiseModel(k_cal=rel, snr_ref=np.inf, rho_ref=1.0,
                      floor=floor, name="generic-relative")
