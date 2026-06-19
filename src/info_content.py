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
"""
from __future__ import annotations

import numpy as np

from retrieval_oe import posterior_diagnostics  # the single DOFS/SIC/AK implementation


def info_content_on_ode_grid(fwd, x, s_nodes, prior_builder, Se):
    """DOFS / SIC / averaging kernel of the r_e(τ) profile on the FULL ODE grid.

    Runs the adaptive solve at state ``x`` to obtain the ODE τ-grid, forms the ToA
    Jacobian ∂y/∂r_e on its **interior** normalized-depth nodes, builds the r_e-block
    prior there, and returns the Rodgers diagnostics on that full grid.

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
    x = np.asarray(x, float)
    cur_tau_bot = float(fwd._split_state(x, s_nodes)[2])
    tau_grid = fwd.ode_grid(x, s_nodes)                            # absolute τ (full grid)
    s_grid = np.unique(np.clip(tau_grid / cur_tau_bot, 0.0, 1.0))  # normalized
    interior = s_grid < 1.0 - 1e-6
    s_int = s_grid[interior]
    re_grid = fwd.profile(x, s_nodes, s_grid * cur_tau_bot)
    K = np.asarray(fwd.jacobian_on_grid(re_grid, s_grid, cur_tau_bot))[:, interior]
    _, Sa = prior_builder(s_int)
    n = s_int.size
    Sa_re = np.asarray(Sa, float)[:n, :n]                          # r_e block only
    return posterior_diagnostics(K, Sa_re, Se), s_int
