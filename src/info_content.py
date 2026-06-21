"""info_content.py — measurement information-content profiling on the FULL ODE grid.

Deliberately separate from the retrieval solver. The retrieval picks a handful of
nodes (the noise-aware filter; ``retrieval_oe.select_retrieval_grid``), but the
*measurement's* intrinsic information content is a property of the full,
continuously-resolved column — so we characterise it on the adaptive ODE grid
itself (a trustworthy superset of the informative points, DESIGN §3a). The Rodgers
algebra is reused from ``retrieval_oe.posterior_diagnostics`` (DOFS = tr(A), SIC,
averaging kernel) — there is exactly one implementation of it.

The DOFS here is a *different number* from the per-retrieval ``posterior_diagnostics``
DOFS on the selected ``k_active`` nodes: this answers "how much can the measurement
resolve" (full grid), that answers "how much did this retrieval resolve" (its grid).
Label them distinctly when reporting (notebook §11/§12).

Note the information scale is set entirely by ``Se`` (the measurement-noise model):
a "noiseless" OSSE still assumes an ``Se``; see DESIGN. DOFS/SIC depend only on
``(K, Se, Sa)`` — never on a noise *realization*.

**Cost model (Stage-1 sweeps).** The expensive step is the autodiff Jacobian ``K``;
the Rodgers diagnostics (``posterior_diagnostics``) and the whitened singular
spectrum (``info_spectrum``) are O(ms) linear algebra. ``K`` depends only on the
state + observing geometry (bands, views, NQuad), **not** on the prior or ``Se`` —
so one ``K`` serves the entire prior ladder and every noise level. Compute ``K``
once with :func:`jacobian_on_ode_grid`, then derive all metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retrieval_oe import posterior_diagnostics  # the single DOFS/SIC/AK implementation


def jacobian_on_ode_grid(fwd, x, s_nodes):
    """ToA Jacobian ``K = ∂y/∂r_e`` and grid on the FULL interior ODE grid.

    Runs the adaptive solve at state ``x`` to obtain the ODE τ-grid, evaluates the
    r_e profile on it, and forms the ToA Jacobian on its **interior** normalized-depth
    nodes (the base node s=1 is dropped — r_base is a separate retrieved scalar).
    Prior-independent and ``Se``-independent: reuse the returned ``K`` across the
    whole prior ladder / noise levels.

    Returns ``(K, s_int)`` — ``K`` of shape ``(n_obs, n_int)`` and the interior
    normalized-depth grid ``s_int`` the columns live on.
    """
    x = np.asarray(x, float)
    cur_tau_bot = float(fwd._split_state(x, s_nodes)[2])
    tau_grid = fwd.ode_grid(x, s_nodes)                            # absolute τ (full grid)
    s_grid = np.unique(np.clip(tau_grid / cur_tau_bot, 0.0, 1.0))  # normalized
    interior = s_grid < 1.0 - 1e-6
    s_int = s_grid[interior]
    re_grid = fwd.profile(x, s_nodes, s_grid * cur_tau_bot)
    K = np.asarray(fwd.jacobian_on_grid(re_grid, s_grid, cur_tau_bot))[:, interior]
    return K, s_int


def info_content_on_ode_grid(fwd, x, s_nodes, prior_builder, Se):
    """DOFS / SIC / averaging kernel of the r_e(τ) profile on the FULL ODE grid.

    Thin wrapper: :func:`jacobian_on_ode_grid` for ``K`` + the r_e-block prior from
    ``prior_builder`` + :func:`retrieval_oe.posterior_diagnostics`.

    Parameters
    ----------
    fwd : RetrievalForward
        The differentiable forward (its ``jac_mode`` sets fwd/rev autodiff).
    x, s_nodes : the current joint state and its normalized-depth nodes.
    prior_builder : callable ``s -> (x_a, Sa)`` (e.g. a ``make_marine_sc_prior``
        closure); only the r_e-block of ``Sa`` is used.
    Se : observation error covariance (the noise model — sets the info scale).

    Returns
    -------
    (post, s_grid) : ``retrieval_oe.Posterior`` on the full ODE grid (``post.dofs``
        = tr(A), ``post.sic``, ``post.A``, ``post.error``) and the interior
        normalized-depth grid the diagnostics live on.
    """
    K, s_int = jacobian_on_ode_grid(fwd, x, s_nodes)
    _, Sa = prior_builder(s_int)
    n = s_int.size
    Sa_re = np.asarray(Sa, float)[:n, :n]                          # r_e block only
    return posterior_diagnostics(K, Sa_re, Se), s_int


@dataclass
class Spectrum:
    """Whitened information spectrum from one SVD of ``K̃ = Se^(-1/2) K Sa^(1/2)``."""
    singular_values: np.ndarray   # s_i — whitened singular values (SNR units), descending
    filter_factors: np.ndarray    # f_i = s_i²/(1+s_i²) — per-direction data fraction
    dofs: float                   # Σ f_i  (= tr(A), basis-free)
    sic: float                    # ½ Σ log₂(1+s_i²) [bits]


def info_spectrum(K, Sa, Se) -> Spectrum:
    """Basis-free whitened singular spectrum of the measurement — one SVD.

    Whiten the Jacobian by the noise (rows) and the full *correlated* prior
    (columns): ``K̃ = Se^(-1/2) · K · Sa^(1/2)`` (symmetric SPD square roots). The
    singular values ``{s_i}`` of ``K̃`` are the information content's eigenstructure
    in signal-to-noise units — how the resolving power is distributed across
    independent directions, not merely its total. From them:

    * ``f_i = s_i²/(1+s_i²)`` — Rodgers filter factor (the fraction of direction *i*
      supplied by the data; ``s_i = 1`` ⇔ data ties prior ⇔ ``f_i = 0.5``),
    * ``DOFS = Σ f_i`` and ``SIC = ½ Σ log₂(1+s_i²)`` [bits].

    These match :func:`retrieval_oe.posterior_diagnostics` to round-off — ``Σ f_i =
    tr(A)`` and the SIC is identical — by the similarity ``A ∼ K̃ᵀK̃ (I+K̃ᵀK̃)⁻¹``;
    this function surfaces the spectrum that underlies those scalars. Distinct from
    :func:`retrieval_oe.auto_k_active`, which whitens with ``diag(σ)`` + QRCP to get
    per-*node* (not per-direction) marginal information for node *selection*.

    Parameters
    ----------
    K : (n_obs, n) Jacobian ∂y/∂x.
    Sa : (n, n) prior covariance for the same state block (the full correlated
        matrix, not just its diagonal).
    Se : (n_obs, n_obs) observation error covariance.
    """
    K = np.asarray(K, float)
    Sa = np.asarray(Sa, float)
    Se = np.asarray(Se, float)
    # Se^{-1/2} (general SPD; Se is diagonal in practice).
    we, Ve = np.linalg.eigh(Se)
    Se_half_inv = (Ve / np.sqrt(we)) @ Ve.T
    # Sa^{1/2} (SPD correlated prior; clip tiny negative eigenvalues from round-off).
    wa, Wa = np.linalg.eigh(Sa)
    Sa_half = (Wa * np.sqrt(np.clip(wa, 0.0, None))) @ Wa.T
    K_tilde = Se_half_inv @ K @ Sa_half
    s = np.linalg.svd(K_tilde, compute_uv=False)        # whitened singular values
    s2 = s ** 2
    f = s2 / (1.0 + s2)
    return Spectrum(singular_values=s, filter_factors=f,
                    dofs=float(f.sum()), sic=float(0.5 * np.sum(np.log2(1.0 + s2))))
