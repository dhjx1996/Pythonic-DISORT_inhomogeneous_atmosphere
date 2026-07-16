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

The σ(ρ) (see DESIGN §12 for the physics)::

    σ(ρ) = sqrt( (k_cal·ρ)²   # calibration / radiometric accuracy (flat-relative)
                 + floor² )   # read/dark/quantization (additive, signal-independent)

The two sources are independent ⇒ added in quadrature. ``k_cal·ρ`` does *not*
average down with brightness (a 2 % gain error is 2 % on any pixel); the floor
matters only for dark pixels. A photon shot-noise term (∝ √ρ) was removed
(2026-07-10, ponytail audit): the OCI SNR-at-L_typ table could not be cleanly
sourced, so it was never populated — re-add it in quadrature
(``ρ·ρ_ref / SNR_ref²``) when the spec is sourced. See OUTSTANDING §K.

**OCI-SWIR population (the production default, :func:`oci_swir`).** Calibration-
relative: ``k_cal`` from the documented OCI/HARP2 radiometric accuracy (1–3 %;
PACE MRD §3.7 absolute-gain uncertainty), small floor. The SWIR cloud-reflectance
regime is calibration-dominated because clouds are bright.

Scope (per the user's firm choices, 2026-06-19): **OCI-SWIR intensity only.**
HARP2 (VIS multi-angle) and polarized / DoLP noise are deferred with the
polarized-cloudbow observable (OUTSTANDING §I, §K).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Per-observation reflectance-noise model (see module docstring).

    Both coefficients are scalars, applied elementwise to every observation
    (a per-band array variant was removed 2026-07-16, ponytail audit: no caller
    ever differentiated the bands — re-add band broadcasting when one does).

    Parameters
    ----------
    k_cal : relative radiometric-accuracy coefficient (e.g. 0.02 = 2 %).
    floor : additive reflectance floor (read/dark/quantization).
    name : label for provenance / plotting.
    """

    k_cal: float
    floor: float = 0.0
    name: str = "custom"

    def sigma(self, rho):
        """Per-observation 1σ noise on the reflectance vector ``rho``."""
        rho = np.abs(np.asarray(rho, float)).ravel()
        return np.sqrt((self.k_cal * rho) ** 2 + self.floor ** 2)

    def Se(self, rho):
        """Assumed measurement-error covariance ``diag(σ²)`` (what the retrieval inverts)."""
        return np.diag(self.sigma(rho) ** 2)

    def sample(self, rho, seed=0):
        """A Gaussian noise *realization* ``ρ + N(0, σ)`` (for a noisy synthetic obs)."""
        rho = np.asarray(rho, float)
        rng = np.random.default_rng(seed)
        return rho + rng.standard_normal(rho.shape) * self.sigma(rho).reshape(rho.shape)


# ---------------------------------------------------------------------------
# Instrument presets
# ---------------------------------------------------------------------------

def oci_swir(k_cal=0.02, floor=1e-3):
    """PACE **OCI SWIR** intensity noise — the production default.

    Calibration-relative: ``σ ≈ k_cal·ρ`` (plus a small floor), with ``k_cal``
    from the documented OCI/HARP2 radiometric accuracy of **1–3 %** (PACE MRD
    §3.7 absolute-gain uncertainty). Default ``k_cal=0.02`` (2 %) sits in that
    band.
    """
    return NoiseModel(k_cal=k_cal, floor=floor, name="OCI-SWIR")
