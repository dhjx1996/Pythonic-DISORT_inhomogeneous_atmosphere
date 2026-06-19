"""retrieval_oe.py — optimal-estimation r_e(τ) retrieval over the Riccati seam.

Thin glue around the *existing* differentiable forward model
(``pydisort_riccati_jax`` seam + ``miejax_lite`` optics table) that turns a
multi-band, multi-angle ToA reflectance measurement into an effective-radius
profile ``r_e(τ)`` by Gauss–Newton optimal estimation with a Bayesian-Tikhonov
(correlated-Gaussian) prior, plus the posterior uncertainty quantification.

**Nothing here is a new forward model.** ``build_forward`` composes
``riccati_setup`` / ``riccati_solve`` / ``eval_radiance`` (the jit-able seam,
DESIGN_DECISIONS §7) with the precomputed ``table_lookup`` optics; the only new
code is the state→observation mapping and the OE/UQ linear algebra.

Design choices (see the plan + DESIGN_DECISIONS):

- **State** ``x`` = ``r_e`` at a handful of free τ-nodes (cloud top τ=0 always a
  node); cloud base ``(τ_bot, r_base)`` is a *fixed, known* anchor (simplification
  — the two hardest quantities to retrieve in thick cloud are deferred).
  ``r_e(τ)`` is the interpolant through the nodes + base anchor — **r_e⁵-linear
  (adiabatic) by default** (the adiabatic law in optical depth is r_e ∝ τ^(1/5) —
  see :meth:`RetrievalForward._re_of_tau`), set by that single lever.
  That interpolation is **part of the forward map** (it defines what is retrieved),
  *not* a post-hoc display choice; plot the result with :meth:`RetrievalForward.profile`
  so the curve mirrors ``F(x)``. The function-class is an open lever (linear /
  monotone-cubic PCHIP) — see OUTSTANDING §B′.
- **Observation** ``y`` = bidirectional reflectance ``R = π u / (μ0 I0)`` stacked
  over {bands} × {view angles incl. oblique}. Extra view angles are nearly free
  (one solve per band, evaluated at many (μ,φ)); oblique views are the lever for
  thin clouds (longer slant path, minimal penetration depth) and add vertical DOF.
- **Jacobian** ``K = ∂y/∂x`` by reverse-mode autodiff (``jax.jacrev``) through the
  jitted seam — the verified discrete adjoint (§5). (``jacfwd`` via a
  ``ForwardMode`` setup is the documented small-p optimisation; jacrev is used
  here for robustness and a single adjoint path.)
- **Retrieval grid** = sensitivity-selected subset of the adaptive ODE grid by
  QRCP (DESIGN §3); ``select_retrieval_grid``.
- **GN/UQ** done host-side in NumPy/SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from pydisort_riccati_jax import (
    riccati_setup, riccati_solve, eval_radiance,
    pydisort_riccati_jax, _barycentric_interpolate,
)
from miejax_lite import table_lookup


# ============================================================================
# 1. Forward / observation operator (thin glue over the seam)
# ============================================================================
class RetrievalForward:
    """Multi-band, multi-angle ToA-reflectance forward model for OE.

    Builds one host-side ``setup`` per band (geometry — incl. the **static**
    ``mu0`` — shared; the per-band BDRF and optics table differ), and exposes
    jitted ``forward`` / ``jacobian`` callables plus the ODE-grid and
    pool-sensitivity utilities used by the retrieval-grid selector. The per-band
    Fourier mode count ``K`` (``K_list``) defaults to the full ``NFourier`` and
    can be trimmed offline by the S_ε selector :func:`select_num_modes`.
    """

    def __init__(self, opt_bands, *, NQuad, mu0, I0, phi0, tau_bot, r_base,
                 view_mu, view_phi, BDRF_bands=None, NLeg_all=128, NFourier=8,
                 tol=1e-3, re_class="re5-linear", jac_mode="rev",
                 retrieve_tau_bot=False, retrieve_r_base=False,
                 re_bounds=(2.0, 25.0), tau_bounds=(0.1, 60.0)):
        # ``retrieve_tau_bot`` / ``retrieve_r_base`` promote the cloud-base anchor
        # ``(τ_bot, r_base)`` from *fixed known* values to **retrieved unknowns**
        # (the joint retrieval, PO). When True the corresponding quantity is read
        # from the state vector instead of the constructor; ``tau_bot`` / ``r_base``
        # then supply only the *first-guess / fallback* (a leak-free climatological
        # value, NOT the truth). State layout is always
        #     x = [r_e(τ_node_0..k-1),  (r_base if retrieve_r_base),
        #                               (τ_bot if retrieve_tau_bot)]
        # decoded by :meth:`_split_state`. Default False reproduces the legacy
        # fixed-anchor forward bit-for-bit (kept for the baseline comparison and so
        # the supplementary scripts are unchanged).
        # NLeg_all>=128: a Mie cloud phase function needs ~60+ moments for the
        # NT/TMS single-scatter; 32 gives a Gibbs-oscillating p_full that wrecks
        # thin-cloud (single-scatter-dominated) off-nadir radiance. See
        # docs/OUTSTANDING.md §A′. Cheap: NLeg_all feeds only the TMS quadrature.
        #
        # NFourier is now just the **static ceiling** on the azimuthal mode count.
        # Post the scan-the-modes refactor (OUTSTANDING §H) the mode body compiles
        # once via lax.scan, so running all NFourier modes no longer OOMs the
        # forward/jacrev — NFourier need not be held artificially small for memory.
        # Mode truncation is a *runtime* saving: pick num_modes <= NFourier offline
        # with the S_ε selector (:func:`select_num_modes`) from the per-mode
        # reflectance amplitudes vs the measurement noise, then bake it into
        # ``K_list``. (Default 8 suffices for the thin VOCALS case; raise it for
        # thick cloud, where the selector then trims it back down.)
        self.opt_bands = list(opt_bands)
        self.n_bands = len(self.opt_bands)
        self.mu0 = float(mu0)
        self.I0 = float(I0)
        self.tau_bot = float(tau_bot)
        self.r_base = float(r_base)
        if re_class not in ("re5-linear", "linear"):
            raise ValueError(f"unknown re_class {re_class!r}; "
                             "expected 're5-linear' or 'linear'")
        self.re_class = re_class             # profile parameterisation lever (§B′)
        self.retrieve_tau_bot = bool(retrieve_tau_bot)
        self.retrieve_r_base = bool(retrieve_r_base)
        # physical/table bounds for clamping GN iterates (DESIGN §8 bounded-state
        # forward): r_e to the optics-table support, τ_bot to a marine-Sc range.
        self.re_min, self.re_max = float(re_bounds[0]), float(re_bounds[1])
        self.tau_min, self.tau_max = float(tau_bounds[0]), float(tau_bounds[1])
        self.view_mu = jnp.asarray(view_mu, dtype=float)
        self.view_phi = jnp.asarray(view_phi, dtype=float)
        self.n_view = int(self.view_mu.shape[0])
        self.m = self.n_bands * self.n_view          # observation dimension
        # Off-node radiances are interpolated from the NQuad//2 upwelling quadrature
        # nodes (plus the small per-angle TMS term). Using fewer view angles than
        # NQuad//2 UNDER-SAMPLES that node radiance field — it leaves retrievable
        # information on the table (verified: thin-cloud A_top 0.25→0.39 going 3→8
        # views at NQuad=16; DESIGN_DECISIONS.md §11b). Pick the angles freely, but
        # use at least NQuad//2 of them.
        if self.n_view < NQuad // 2:
            import warnings
            warnings.warn(
                f"n_view={self.n_view} < NQuad//2={NQuad // 2}: the {NQuad // 2} "
                f"quadrature-node radiances are under-sampled, leaving retrievable "
                f"information unused. Use >= NQuad//2 view angles (DESIGN §11b).",
                stacklevel=2)
        if BDRF_bands is None:
            BDRF_bands = [()] * self.n_bands
        # Jacobian AD mode. The retrieval state p (~6-7) is smaller than the
        # observation count m = n_bands*n_view (~16-24), so forward-mode (one tangent
        # solve per input, INDEPENDENT of n_view — the sensitivity funnels through the
        # NQuad//2 quadrature nodes) is cheaper than reverse (one adjoint solve per
        # output) AND makes dense view angles ~free. Forward-mode needs ForwardMode()
        # setups (diffrax's reverse default is a custom_vjp that cannot be
        # forward-differentiated). Default 'rev' keeps the legacy adjoint path.
        if jac_mode not in ("rev", "fwd"):
            raise ValueError(f"jac_mode must be 'rev' or 'fwd', got {jac_mode!r}")
        self.jac_mode = jac_mode
        _adjoint = None
        if jac_mode == "fwd":
            import diffrax
            _adjoint = diffrax.ForwardMode()
        self.setups = [
            riccati_setup(NQuad, I0, phi0, mu0, NFourier=NFourier,
                          NLeg_all=NLeg_all, BDRF_Fourier_modes=bdrf,
                          delta_M_scaling=True, NT_cor=True, tol=tol,
                          adjoint=_adjoint)
            for bdrf in BDRF_bands
        ]
        self.K_list = [s.NFourier for s in self.setups]
        self._fwd_jit = None
        self._jac_jit = None
        self._jac_grid_jit = None

    # -- joint-state decode: free nodes (+ optional retrieved base / τ_bot) ---
    @property
    def n_extra(self):
        """Number of trailing state entries beyond the r_e nodes (0, 1, or 2)."""
        return int(self.retrieve_r_base) + int(self.retrieve_tau_bot)

    def _split_state(self, x, s_nodes):
        """Decode the joint state ``x`` → ``(r_nodes, r_base, τ_bot)``.

        ``r_nodes`` are the first ``len(s_nodes)`` entries (r_e at the free nodes at
        normalized depth ``s∈[0,1)``, incl. cloud top s=0). ``r_base`` / ``τ_bot``
        are read from the trailing entries when retrieved (the joint retrieval, PO),
        else fall back to the fixed constructor values ``self.r_base`` /
        ``self.tau_bot``. The returned ``r_base`` / ``τ_bot`` are **traced scalars**
        in joint mode, so ``∂y/∂r_base`` and ``∂y/∂τ_bot`` flow through autodiff
        (τ_bot is a traced ``riccati_solve`` arg by construction, DESIGN §7).
        """
        x = jnp.asarray(x, float)
        k = int(jnp.asarray(s_nodes).shape[0])       # static (shape known at trace)
        r_nodes = x[:k]
        idx = k
        if self.retrieve_r_base:
            r_base = x[idx]
            idx += 1
        else:
            r_base = self.r_base
        if self.retrieve_tau_bot:
            tau_bot = x[idx]
        else:
            tau_bot = self.tau_bot
        return r_nodes, r_base, tau_bot

    def _clamp_state(self, x, s_nodes):
        """Project a (host-side) state onto the physical/table bounds.

        Clamps the r_e nodes (and retrieved ``r_base``) to the optics-table support
        ``[re_min, re_max]`` and retrieved ``τ_bot`` to ``[tau_min, tau_max]``. Used
        between Gauss-Newton steps (projected GN) so an overshoot cannot drive the
        optics out of the table or ``τ_bot`` negative — the failure that hit the
        absolute-τ / unclamped retrieval (out-of-table r_e, τ_bot < 0, and the
        Kvaerno5 "max steps" controller error from the resulting stiff optics).
        """
        x = np.asarray(x, float).copy()
        k = int(np.asarray(s_nodes).shape[0])
        x[:k] = np.clip(x[:k], self.re_min, self.re_max)         # r_e nodes
        idx = k
        if self.retrieve_r_base:
            x[idx] = np.clip(x[idx], self.re_min, self.re_max)
            idx += 1
        if self.retrieve_tau_bot:
            x[idx] = np.clip(x[idx], self.tau_min, self.tau_max)
        return x

    # -- r_e(s) interpolant knots in normalized depth s=τ/τ_bot ---------------
    def _knots_vals(self, x, s_nodes):
        """Interpolation knots/values (in normalized depth) + the base optical depth.

        Returns ``(s_knots, vals, τ_bot)``: r_e node values at normalized depths
        ``s_knots = [s_nodes, 1.0]`` (free nodes in ``s∈[0,1)`` plus the base anchor
        at ``s=1``), and the (traced in joint mode) ``τ_bot``. No monotonicity guard
        is needed — ``s_knots`` are fixed in ``[0,1]`` regardless of the retrieved
        ``τ_bot`` (that is the whole point of the normalized-depth parameterisation;
        see :meth:`_re_of_tau`).
        """
        r_nodes, r_base, tau_bot = self._split_state(x, s_nodes)
        s_nodes = jnp.asarray(s_nodes, float)
        tau_bot = jnp.asarray(tau_bot, float)
        s_knots = jnp.concatenate([s_nodes, jnp.ones((1,))])
        vals = jnp.concatenate([jnp.asarray(r_nodes, float),
                                jnp.reshape(jnp.asarray(r_base, float), (1,))])
        return s_knots, vals, tau_bot

    def _re_of_tau(self, tau, s_knots, vals, tau_bot):
        """r_e at optical depth ``tau`` from node values at **normalized depth**
        ``s = τ/τ_bot ∈ [0,1]`` — the function-class lever (OUTSTANDING §B′; DESIGN §3d).

        **Normalized depth (not absolute τ) is the key to joint τ_bot retrieval.** The
        nodes live at fixed ``s∈[0,1]`` and the base at ``s=1``; multiplying by the
        retrieved ``τ_bot`` gives the absolute positions, so the nodes **stretch /
        compress with τ_bot** and never fall past the cloud base. The absolute-τ
        parameterisation breaks here: a grid placed at a thick first-guess τ_bot puts
        nodes below a thin cloud's base, and retrieving τ_bot downward then drives a
        node past τ_bot (the crossing that made the absolute-τ GN diverge / hit
        max-solver-steps). In ``s`` there is no crossing and no monotonicity guard.

        This is NOT a post-hoc interpolation — it is *inside* F(x); :meth:`profile`
        routes through here so the display mirrors the forward.

        **Default: r_e⁵-linear (adiabatic).** r_e ∝ τ^(1/5) (r_e³∝LWC∝z, β∝r_e²∝z^(2/3)
        ⇒ τ∝z^(5/3) ⇒ r_e∝τ^(1/5)). Since ``s`` is just a linear rescale of τ,
        **r_e⁵-linear in s ≡ r_e⁵-linear in τ** — the adiabatic law is unchanged by the
        normalization. C⁰; finite base slope. ``re_class`` switches the class here and
        only here so it propagates to forward / modes / ODE-grid / Jacobian / display.
        """
        s = tau / tau_bot                                  # absolute τ → normalized depth
        if self.re_class == "linear":
            return jnp.interp(s, s_knots, vals)
        return jnp.interp(s, s_knots, vals ** 5) ** (1.0 / 5.0)   # re5-linear (adiabatic)

    def profile(self, x, s_nodes, tau):
        """Evaluate the retrieved r_e at **absolute** optical depths ``tau`` exactly
        as the forward integrates it (the node grid ``s_nodes`` is normalized depth).

        Free node values ``x`` at normalized depths ``s_nodes`` plus the base anchor
        at ``s=1``, through :meth:`_re_of_tau` (which maps the queried absolute ``tau``
        to ``s=τ/τ_bot``). Use this for plotting / downstream so the displayed curve
        mirrors F(x) by construction. ``tau`` are absolute optical depths (e.g.
        ``linspace(0, retrieved_τ_bot, …)``).
        """
        s_knots, vals, tau_bot = self._knots_vals(x, s_nodes)
        return np.asarray(self._re_of_tau(jnp.asarray(tau, float), s_knots, vals,
                                          tau_bot))

    def _band_reflectance(self, opt, setup, K, s_knots, vals, tau_bot):
        def om(tau):
            return table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[0]

        def leg(tau):
            return table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[1]

        res = riccati_solve(setup, om, leg, tau_bot, num_modes=K)
        u = jnp.stack([eval_radiance(setup, res, self.view_mu[i], self.view_phi[i])
                       for i in range(self.n_view)])           # (n_view,)
        return jnp.pi * u / (self.mu0 * self.I0)

    def _forward_raw(self, x, s_nodes):
        s_knots, vals, tau_bot = self._knots_vals(x, s_nodes)
        return jnp.concatenate([
            self._band_reflectance(opt, setup, K, s_knots, vals, tau_bot)
            for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
        ])                                                     # (n_bands*n_view,)

    # -- per-mode reflectance amplitudes (drives the S_ε mode selector) -------
    def mode_amplitudes(self, x_ref, s_nodes):
        """Per-band, per-mode ToA-reflectance amplitude at a reference state.

        For each band runs ONE full-``NFourier`` solve and decomposes the ToA
        bidirectional reflectance into its azimuthal Fourier contributions at the
        view angles::

            contrib_m(μ_i, φ_i) = π · u_m(μ_i) · cos(m (φ0 − φ_i)) / (μ0 I0)

        (``u_m`` the m-th Fourier mode of the ToA upwelling field, barycentrically
        interpolated to the view μ). Returns ``amp`` of shape ``(n_bands,
        NFourier)`` with ``amp[b, m] = max_i |contrib_m|`` — the worst-case
        reflectance any single mode adds across the views. Consumed by
        :func:`select_num_modes`; the *absolute* per-mode amplitude (not a relative
        partial sum) is the meaningful quantity to compare against the noise floor.
        """
        s_knots, vals, tau_bot = self._knots_vals(x_ref, s_nodes)
        amps = []
        for opt, setup in zip(self.opt_bands, self.setups):
            def om(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[0]

            def leg(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[1]

            res = riccati_solve(setup, om, leg, tau_bot)        # all NFourier
            u_modes = res.u_modes                               # (NFourier, N)
            # u_m at each view μ: barycentric interp of each mode's (N,) vector.
            u_view = _barycentric_interpolate(
                self.view_mu, setup.mu_nodes, u_modes.T, setup.bary_weights
            )                                                   # (n_view, NFourier)
            m_arr = jnp.arange(u_modes.shape[0])
            cosm = jnp.cos(m_arr[None, :]
                           * (setup.phi0 - self.view_phi)[:, None])  # (n_view, NF)
            contrib = jnp.pi * u_view * cosm / (self.mu0 * self.I0)  # (n_view, NF)
            amps.append(np.asarray(jnp.max(jnp.abs(contrib), axis=0)))  # (NF,)
        return np.stack(amps)                                   # (n_bands, NF)

    # -- jitted forward + Jacobian (compiled once, cached) -------------------
    def forward(self, x, s_nodes):
        if self._fwd_jit is None:
            self._fwd_jit = jax.jit(self._forward_raw)
        return self._fwd_jit(jnp.asarray(x, float), jnp.asarray(s_nodes, float))

    def jacobian(self, x, s_nodes):
        """K = ∂y/∂x  (m × p) through the jitted seam. ``jac_mode='fwd'`` uses
        forward-mode (``jax.jacfwd``, p tangent solves — cheaper and n_view-
        independent when p < m); ``'rev'`` uses reverse-mode (``jax.jacrev``)."""
        if self._jac_jit is None:
            _jac = jax.jacfwd if self.jac_mode == "fwd" else jax.jacrev
            self._jac_jit = jax.jit(_jac(self._forward_raw, argnums=0))
        return self._jac_jit(jnp.asarray(x, float), jnp.asarray(s_nodes, float))

    # -- ODE grid (adaptive candidate pool, DESIGN §3) -----------------------
    def ode_grid(self, x, s_nodes):
        """Adaptive Kvaerno5 **absolute-τ** grid at the given state (first band).

        Integrates to the **current** ``τ_bot`` (the retrieved value in joint mode,
        decoded from ``x``), so the candidate pool tracks the estimated cloud depth.
        Returns absolute optical depths (the caller normalizes by ``τ_bot``).
        """
        s_knots, vals, tau_bot = self._knots_vals(x, s_nodes)
        opt = self.opt_bands[0]
        om = lambda tau: table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[0]
        leg = lambda tau: table_lookup(opt, self._re_of_tau(tau, s_knots, vals, tau_bot))[1]
        *_, tau_grid = pydisort_riccati_jax(
            float(tau_bot), om, leg, self.setups[0].NQuad, self.mu0, self.I0,
            self.setups[0].phi0, delta_M_scaling=True, NT_cor=True,
            NLeg_all=self.opt_bands[0]["leg"].shape[-1])
        return np.asarray(tau_grid, float)

    # -- pool sensitivity (parameterise r_e on a normalized-depth pool grid) --
    def jacobian_on_grid(self, re_vals, s_grid, tau_bot=None):
        """K_pool = ∂y/∂r_e(s_j) at the pool nodes ``s_grid`` (normalized depth; m×len).

        ``s_grid`` and ``tau_bot`` are **traced** arguments, so re-selections at a
        stable ODE-grid size reuse the compiled Jacobian (recompile-free lagged
        re-meshing); a new pool size compiles once and caches. ``s_grid`` are the
        pool nodes in normalized depth ``s=τ/τ_bot``; ``tau_bot`` is the integration
        limit (the current/retrieved cloud base; defaults to ``self.tau_bot``).
        """
        if tau_bot is None:
            tau_bot = self.tau_bot
        if self._jac_grid_jit is None:
            def fwd(rv, sg, tb):
                return jnp.concatenate([
                    self._band_reflectance(opt, setup, K, sg, rv, tb)
                    for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
                ])
            _jac = jax.jacfwd if self.jac_mode == "fwd" else jax.jacrev
            self._jac_grid_jit = jax.jit(_jac(fwd, argnums=0))
        return np.asarray(self._jac_grid_jit(jnp.asarray(re_vals, float),
                                             jnp.asarray(s_grid, float),
                                             jnp.asarray(tau_bot, float)))


def build_forward(*args, **kw):
    """Functional alias for :class:`RetrievalForward`."""
    return RetrievalForward(*args, **kw)


# ============================================================================
# 1b. Azimuthal mode-count selector (S_ε, replaces the relative Cauchy test)
# ============================================================================
def select_num_modes(fwd: RetrievalForward, x_ref, s_nodes, Se, *, frac=1/3.0):
    """Pick the per-band Fourier mode count ``K`` from the **measurement noise**.

    The old in-solver relative Cauchy test (STWLE2000 p.89) was removed with the
    scan-the-modes refactor (OUTSTANDING §H): it saturated (``K=NFourier``) for
    thin low-signal clouds and, more fundamentally, judged convergence against the
    *signal* rather than the *noise*. Here truncation is a noise-aware **runtime**
    optimisation — there is no point computing a mode whose ToA-reflectance
    contribution is small compared with what the instrument can measure.

    For each band, takes the per-mode reflectance amplitudes
    (:meth:`RetrievalForward.mode_amplitudes` at the reference state ``x_ref``) and
    keeps the smallest ``K`` such that **every** higher mode ``m >= K`` contributes
    less than ``frac · min σ_ε`` at every view — where ``σ_ε = √diag(Se)`` is the
    observation 1σ. ``frac = 1/3`` keeps the dropped-mode error well inside the
    noise. Sets and returns ``fwd.K_list``; invalidates the compiled callables (K is
    static / baked into the jitted forward).

    Parameters
    ----------
    fwd : RetrievalForward
    x_ref, s_nodes : reference state (joint, on a normalized-depth grid) at which to
        measure the mode amplitudes (the mode spectrum is weakly state-dependent —
        pick a representative first-guess profile, e.g. the prior mean).
    Se : (m, m) array — observation error covariance (reflectance²).
    frac : float — keep-threshold as a fraction of the minimum observation σ_ε.
    """
    sigma_eps = np.sqrt(np.diag(np.asarray(Se, float)))
    thresh = float(frac) * float(np.min(sigma_eps))
    amps = fwd.mode_amplitudes(x_ref, s_nodes)                 # (n_bands, NFourier)
    K_list = []
    for amp in amps:
        sig = np.where(amp >= thresh)[0]            # modes above the noise floor
        K = int(sig.max()) + 1 if sig.size else 1   # smallest K dropping only sub-noise modes
        K_list.append(max(1, min(K, amp.shape[0])))
    fwd.K_list = K_list
    fwd._fwd_jit = fwd._jac_jit = fwd._jac_grid_jit = None
    return fwd.K_list


# ============================================================================
# 2. Retrieval-grid selection: QRCP-trimmed subset of the adaptive ODE grid
# ============================================================================
def select_retrieval_grid(fwd: RetrievalForward, x, s_nodes, k_active=None, *,
                          Se=None, prior_builder=None, filter_threshold=0.5,
                          margin=1, re_of_tau=None, k_max=8):
    """Sensitivity-select **normalized-depth** nodes from the FULL ODE grid.

    1. Run the adaptive solve at the current state → ODE absolute-τ grid (the
       candidate pool — a trustworthy *superset* of the informative points,
       DESIGN §3a), normalize to ``s = τ/τ_bot ∈ [0,1]``. **No resampling**: the
       pool *is* the ODE grid (the old fixed-cardinality resample manufactured
       collinear pool columns for thin clouds — removed).
    2. Form the ToA Jacobian ``∂y/∂r_e(s_j)`` on the interior pool nodes (autodiff).
    3. **Node count** — if ``k_active is None`` the noise-aware filter
       (:func:`auto_k_active`, driven by ``filter_threshold``) sets it from the pool;
       otherwise the given count is used (fixed-count re-mesh). QR-with-column-
       pivoting ranks the interior nodes by independent information; keep the top
       ``k_active`` — always including cloud-top ``s≈0`` (most informative).

    Parameters
    ----------
    k_active : int or None
        None ⇒ the filter decides the count (needs ``Se`` and ``prior_builder``).
        An int ⇒ select exactly that many (fixed-count re-mesh; ``Se``/``prior_builder``
        not required).
    Se, prior_builder : observation covariance and ``prior_builder(s)->(x_a, Sa)``
        Required only when ``k_active is None`` (for the filter whitening / σ_prior).
    re_of_tau : callable or None
        Maps an **absolute** τ to current r_e (to evaluate the pool's r_e values).
        If None, uses the forward's own interpolant of ``(s_nodes, x)``.

    Returns ``(s_sel, re_sel, info)`` with ``info['k_active']`` the count used.
    """
    if k_active is None and (Se is None or prior_builder is None):
        raise ValueError("k_active=None (filter selection) requires Se and "
                         "prior_builder")
    # current cloud base (retrieved value in joint mode, else the fixed anchor)
    cur_tau_bot = float(fwd._split_state(x, s_nodes)[2])
    tau_pool = fwd.ode_grid(x, s_nodes)                        # absolute τ (full grid)
    s_pool = np.unique(np.clip(tau_pool / cur_tau_bot, 0.0, 1.0))   # normalized
    tau_pool_abs = s_pool * cur_tau_bot
    if re_of_tau is None:
        re_pool = fwd.profile(x, s_nodes, tau_pool_abs)        # same class as forward
    else:
        re_pool = np.asarray([float(re_of_tau(t)) for t in tau_pool_abs])

    K_pool = fwd.jacobian_on_grid(re_pool, s_pool, cur_tau_bot)    # (m, P)

    # Candidate set = interior nodes only (the base anchor s=1 is appended by the
    # forward / handled by r_base, not retrieved as a free node).
    interior = s_pool < 1.0 - 1e-6
    s_int = s_pool[interior]
    re_int = re_pool[interior]
    K_int = K_pool[:, interior]

    # node count: filter (k_active None) or fixed (re-mesh fixed-count path)
    if k_active is None:
        _, Sa_pool = prior_builder(s_int)                     # r_e block first
        sig = np.sqrt(np.clip(np.diag(Sa_pool)[:s_int.size], 0.0, None))
        k_active, kinfo = auto_k_active(K_int, Se, sig,
                                        filter_threshold=filter_threshold,
                                        margin=margin, k_max=k_max)
    else:
        kinfo = dict(filter_threshold=float(filter_threshold))
    k_active = int(min(k_active, s_int.size))                  # can't exceed interior pool

    # Column pivoting on the sensitivity matrix: rank interior nodes by indep. info.
    from scipy.linalg import qr
    _, _, piv = qr(K_int, mode="economic", pivoting=True)      # indexes into s_int

    # Always include cloud-top (s≈0, most informative); fill the rest by QRCP rank.
    top_idx = int(np.argmin(s_int))
    chosen = [top_idx]
    for p in piv:
        if len(chosen) >= k_active:
            break
        if int(p) != top_idx:
            chosen.append(int(p))
    if len(chosen) < k_active:                                 # degenerate pool: pad by depth
        for j in np.argsort(s_int):
            if len(chosen) >= k_active:
                break
            if j not in chosen:
                chosen.append(int(j))
    chosen = sorted(set(chosen), key=lambda j: s_int[j])
    s_sel = s_int[chosen]
    re_sel = re_int[chosen]
    info = dict(s_pool=s_pool, K_pool=K_pool, piv=piv, k_active=int(k_active))
    info.update({k: v for k, v in kinfo.items() if k != "k_active"})
    return s_sel, re_sel, info


# ----------------------------------------------------------------------------
# 2b. Data-driven retrieval-node count k_active  (SO1)
# ----------------------------------------------------------------------------
def auto_k_active(K_pool, Se, sigma_prior, *, filter_threshold=0.5, margin=1,
                  k_min=1, k_max=8):
    """How many r_e(τ) nodes the measurement can independently support (SO1).

    Noise-aware **filter** count from the pool Jacobian (DOFS is no longer used for
    selection — it is an information-content diagnostic only; see
    :mod:`info_content`). Whiten ``K̃ = Se^(-1/2) · K_pool · diag(σ_prior)`` (rows by
    noise, columns by the prior √variance), QRCP → pivoted R-diagonal
    ``r_1 ≥ r_2 ≥ …`` (each node's **marginal** information in SNR units). The
    Rodgers filter factor ``f_i = r_i²/(1+r_i²)`` is the *fraction of that
    direction's information that comes from the data*; keep the directions with
    ``f_i ≥ filter_threshold`` plus a fixed ``margin`` of prior-filled ones.

    ``filter_threshold`` is in **data-fraction units** (``f``): **0.5 ⇔ data ties the
    prior** (⇔ ``r_i ≥ 1`` ⇔ SNR ≥ 1) — Rodgers' data/prior crossover, the boundary above
    which a direction is *measured* rather than prior-dominated. **Default 0.5**: it keeps
    exactly the data-dominated directions (plus the fixed ``margin``), and — being the
    SNR=1 crossover on the *noise-whitened* Jacobian — it is **invariant to the noise
    level** (no re-tuning when ``Se`` changes), unlike a tuned absolute cut. A *lower*
    threshold admits prior-dominated (``f<0.5``) directions into the fit; with a loosened
    prior those fit noise (the §13 sub-adiabatic overfit). The earlier ``0.25`` was tuned
    on a **3 %**-noise sweep and was superseded when the measurement noise was grounded to
    the PACE-OCI **2 %** model: a 0.25-vs-0.5 sweep at 2 %
    (``tests/supplementary/sweep_threshold_2pct.py``) showed 0.5 leaves the thin case
    unchanged, fixes the shielded RF10 overfit (drop-cap 172 %→58 %, RMSE 0.69→0.52), and
    costs the thick case only +0.1 µm (within uncertainty, both χ²≪1, from a node sitting
    *at* f≈0.5). Computed **once** at the first guess so the count is frozen for the
    retrieval (the forward/Jacobian compile once); a node-count change is only revisited
    under the ``max_n_outer=3`` re-mesh escalation.

    ``sigma_prior`` is the r_e-block prior σ vector (length = number of pool
    columns). ``k`` is clamped to ``[k_min, min(k_max, n_obs, pool_size)]``;
    cloud-top is always retained by :func:`select_retrieval_grid`, not counted here.

    Returns ``(k, info)``.
    """
    K = np.asarray(K_pool, float)
    Se = np.asarray(Se, float)
    sig = np.asarray(sigma_prior, float)
    # Se^{-1/2} for general SPD observation covariance (here Se is diagonal).
    w, V = np.linalg.eigh(Se)
    Se_half_inv = (V / np.sqrt(w)) @ V.T

    K_tilde = Se_half_inv @ K @ np.diag(sig)       # whitened (SNR units)
    from scipy.linalg import qr
    _, R, _ = qr(K_tilde, mode="economic", pivoting=True)
    rdiag = np.abs(np.diag(R))
    f = rdiag ** 2 / (1.0 + rdiag ** 2)            # per-direction data fraction
    n_data = int(np.count_nonzero(f >= float(filter_threshold)))
    k = n_data + int(margin)

    k_hi = int(min(k_max, K.shape[0], K.shape[1]))
    k = int(np.clip(k, k_min, max(k_min, k_hi)))
    info = dict(k_active=k, filter_threshold=float(filter_threshold),
                n_data=n_data, margin=int(margin), sum_filter_factor=float(f.sum()),
                rdiag=rdiag.tolist(), filter_factors=f.tolist())
    return k, info


# ============================================================================
# 3. Prior (pluggable): adiabatic mean + Bayesian-Tikhonov covariance
# ============================================================================
def make_adiabatic_prior(tau_nodes, tau_bot, r_base, r_top_prior, *,
                         sigma_top=3.0, sigma_base=10.0, corr_length=None,
                         strength=1.0):
    """Adiabatic first guess ``x_a`` and correlated prior covariance ``S_a``.

    ``x_a`` is the r_e⁵-linear adiabatic profile (known base → prior top radius)
    sampled at ``tau_nodes`` — the adiabatic law *in optical depth* is r_e ∝ τ^(1/5)
    (r_e³ ∝ LWC ∝ height z, and β ∝ r_e² makes τ ∝ z^(5/3), so r_e ∝ τ^(1/5)).
    Coherent with the ``re5-linear`` forward class. ``S_a`` is a depth-increasing,
    exponentially correlated Gaussian::

        S_a[i,j] = σ_i σ_j exp(-|τ_i-τ_j| / ℓ),   σ(τ) tight at top, loose at base

    The off-diagonal correlation is the Bayesian-Tikhonov smoothness term; ``ℓ``
    (default τ_bot/2) and ``strength`` (a single σ-scale knob) are the only free
    regularisation levers. Same signature as a future *learned* prior
    (ensemble mean + sample/EOF covariance would slot in here).

    ``r_top_prior`` is the prior *belief* about the cloud-top radius (the top node is
    retrieved; this is only its prior mean). When the truth top is not known, the
    data-grounded climatological value is the **VOCALS-REx ensemble's own** cloud-top
    r_e distribution (125 profiles): mean ≈ **9.7 µm** (median 9.5, σ 2.3; range
    4.9–18.0), or ≈ **10.3 ± 2.2 µm** for the thick (τ_bot>8) subset — and that
    empirical σ≈2.2–2.3 µm is consistent with the ``sigma_top=3`` used here. (The OSSE
    demo instead passes the true top, deliberately: see notebook §12 — a perfectly
    anchored adiabatic prior makes the measurement's *departure* from adiabatic
    unambiguous.) This empirical mean+spread is the first rung of a fully learned prior.
    """
    tau_nodes = np.asarray(tau_nodes, float)
    # adiabatic r_e^5-linear: r_e^5 linear in optical depth (r_e ∝ τ^(1/5); see
    # docstring for the height-vs-optical-depth derivation).
    frac = 1.0 - tau_nodes / tau_bot                           # 1 at top, 0 base
    x_a = (r_base ** 5 + (r_top_prior ** 5 - r_base ** 5) * frac) ** (1.0 / 5.0)

    if corr_length is None:
        corr_length = max(tau_bot / 2.0, 1e-3)
    # depth-increasing σ: linearly from σ_top (τ=0) to σ_base (τ=τ_bot).
    sigma = strength * (sigma_top + (sigma_base - sigma_top) * tau_nodes / tau_bot)
    dt = np.abs(tau_nodes[:, None] - tau_nodes[None, :])
    Sa = (sigma[:, None] * sigma[None, :]) * np.exp(-dt / corr_length)
    Sa += 1e-9 * np.eye(len(tau_nodes))                        # SPD jitter
    return x_a, Sa


def make_joint_prior(s_nodes, *, tau_bot_prior, r_top_prior, r_base_prior,
                     retrieve_r_base=True, retrieve_tau_bot=True,
                     sigma_top=5.0, sigma_base=8.0, sigma_tau_bot=None,
                     corr_length=None, strength=1.0):
    """Joint prior over the retrieved state ``x = [r_e nodes, (r_base), (τ_bot)]``.

    The r_e nodes live at **normalized depth** ``s∈[0,1)`` — *and* ``r_base``, which
    **is** r_e at the base, so it joins the block as the deepest node at ``s=1`` —
    forming one correlated, depth-increasing-σ adiabatic block
    (:func:`make_adiabatic_prior` evaluated in normalized depth, i.e. with unit
    ``τ_bot`` so the adiabatic mean ``r_e∝s^(1/5)`` is τ_bot-independent). ``τ_bot``
    is appended as an **independent broad scalar** dimension (block-diagonal: cloud
    optical depth is a different physical quantity from droplet size, coupled only
    weakly, so we do not assert a cross-correlation in the prior).

    All means are **leak-free** — generic/climatological ``r_top_prior``,
    ``r_base_prior``, ``tau_bot_prior``, *never* the truth. Broad σ's make it the
    **weakly-informative (Option 2)** headline prior: the data sets ``τ_bot`` (the
    measurement constrains optical thickness directly) and the upper-cloud r_e,
    while the prior fills the radiatively shielded base. The ``retrieve_*`` flags
    must match the :class:`RetrievalForward` they will be used with.

    ``sigma_tau_bot`` defaults to ``0.5·τ_bot_prior`` (~50 % relative — broad);
    ``corr_length`` is in normalized-depth units (default 0.5).
    """
    s_nodes = np.asarray(s_nodes, float)
    # r_e block in NORMALIZED depth: base node at s=1; unit τ_bot ⇒ frac=1−s.
    nodes_aug = np.append(s_nodes, 1.0) if retrieve_r_base else s_nodes
    x_a, Sa = make_adiabatic_prior(
        nodes_aug, 1.0, r_base_prior, r_top_prior,
        sigma_top=sigma_top, sigma_base=sigma_base,
        corr_length=corr_length, strength=strength)
    if retrieve_tau_bot:
        if sigma_tau_bot is None:
            sigma_tau_bot = 0.5 * float(tau_bot_prior)
        x_a = np.append(x_a, float(tau_bot_prior))
        n = Sa.shape[0]
        Sa_aug = np.zeros((n + 1, n + 1))
        Sa_aug[:n, :n] = Sa
        Sa_aug[n, n] = float(sigma_tau_bot) ** 2
        Sa = Sa_aug
    return x_a, Sa


def make_climatology_prior(s_nodes, clim, *, retrieve_r_base=True,
                           retrieve_tau_bot=True, corr_length=None, strength=1.0):
    """Leave-one-flight-out VOCALS-REx climatological prior (Option 1, *fallback*).

    ``clim`` is the held-out ensemble summary from
    :func:`vocals_io.vocals_climatology` (computed EXCLUDING the target's own
    flight — leave-one-flight-out, so the prior never sees a statistic derived from
    the truth profile or any profile sharing its flight; the leak-free OSSE
    discipline). Means = ensemble means; σ's = ensemble spreads. This is the
    fallback if the broad Option-2 prior (:func:`make_joint_prior` with generic
    numbers) degrades the retrieval — and is then also the prior the
    information-content profiling would use.
    """
    return make_joint_prior(
        s_nodes, tau_bot_prior=clim["tau_bot_mean"],
        r_top_prior=clim["r_top_mean"], r_base_prior=clim["r_base_mean"],
        retrieve_r_base=retrieve_r_base, retrieve_tau_bot=retrieve_tau_bot,
        sigma_top=clim["r_top_std"], sigma_base=clim["r_base_std"],
        sigma_tau_bot=clim["tau_bot_std"], corr_length=corr_length,
        strength=strength)


def make_marine_sc_prior(s_nodes, *, r_top_prior, tau_bot_prior, r_base_ratio=0.65,
                         sigma_top=2.5, sigma_base=1.5, sigma_tau_bot=None,
                         retrieve_r_base=True, retrieve_tau_bot=True,
                         corr_length=None, strength=1.0):
    """Generic marine-Sc joint prior, grounded in VOCALS-REx data + literature.

    Replaces the earlier hand-picked (and inadvertently *inverted*, r_base>r_top)
    broad prior. The structure follows the optimal-estimation principle revealed by
    the prior-sensitivity study (``tests/supplementary/prior_investigation.py``;
    DESIGN_DECISIONS.md §11): **make the prior tight exactly where the measurement
    is blind, loose where it is strong.**

    * **r_top** — observable (averaging-kernel A_top≈1 for thick cloud), so its prior
      barely matters: a *moderate* ``sigma_top`` (≈ the VOCALS MAD 2.3 µm). Effective
      range ~6–15 µm (data p5–p95 6.7–14; the ~14–15 µm drizzle threshold is the
      physical upper bound — bigger droplets precipitate out). Pass a climatological
      or MODIS-retrieved ``r_top_prior``.
    * **r_base** — radiatively **shielded** for thick cloud (A_base≈0.06, ~80 %
      prior-dominated), so the prior *is* the answer there. We therefore make it
      **tight and adiabatic-coupled**, not vague: mean ``r_base = r_base_ratio·r_top``
      (the adiabatic ratio; VOCALS median 0.60, King/Vukićević AMT-2025 ≈0.70 — default
      0.65), **clipped < r_top** (the adiabatic constraint, satisfied by 95 % of VOCALS
      profiles and enforced as a hard bound in that literature), with a tight
      ``sigma_base`` ≈ the VOCALS robust core (MAD 1.4 µm). Sub-saturation /
      re-evaporation profiles are the rare heavy tail (std 2.0 ≫ MAD 1.4); we do not
      try to capture them in the prior — recovering one is a bonus (notebook §13).
    * **tau_bot** — fully data-determined from the bands (A≈1.00, prior irrelevant), and
      "average cloud thickness" is not a meaningful quantity (VOCALS MAD≈median), so the
      prior is deliberately **uninformative**: ``sigma_tau_bot`` defaults to
      ``tau_bot_prior`` (~100 % relative).

    All means are leak-free (climatological/generic, never the truth).
    """
    r_base_prior = min(float(r_base_ratio) * float(r_top_prior),
                       float(r_top_prior) - 0.5)      # adiabatic bound r_base < r_top
    if sigma_tau_bot is None:
        sigma_tau_bot = float(tau_bot_prior)          # uninformative (~100 % relative)
    return make_joint_prior(
        s_nodes, tau_bot_prior=tau_bot_prior, r_top_prior=r_top_prior,
        r_base_prior=r_base_prior, sigma_top=sigma_top, sigma_base=sigma_base,
        sigma_tau_bot=sigma_tau_bot, retrieve_r_base=retrieve_r_base,
        retrieve_tau_bot=retrieve_tau_bot, corr_length=corr_length, strength=strength)


# ============================================================================
# 4. Gauss–Newton optimal estimation (Rodgers n-form) + lagged re-meshing
# ============================================================================
@dataclass
class OEResult:
    x: np.ndarray              # retrieved joint state [r_e nodes, (r_base), (τ_bot)]
    tau_nodes: np.ndarray      # final retrieval grid in NORMALIZED depth s=τ/τ_bot ∈[0,1)
    x_a: np.ndarray            # prior mean on the final grid
    Sa: np.ndarray             # prior covariance on the final grid
    Se: np.ndarray             # observation error covariance
    K: np.ndarray              # final Jacobian
    y: np.ndarray              # observation
    Fx: np.ndarray             # final forward
    cost_history: list = field(default_factory=list)
    converged: bool = False


def _gn_inner(fwd, s_nodes, y, x0, x_a, Sa, Se, *, n_iter, lm, xtol):
    """Inner Gauss–Newton on a fixed (normalized-depth) retrieval grid (Rodgers n-form).

    ``s_nodes`` is the retrieval grid in normalized depth s=τ/τ_bot. Each iterate is
    projected onto the physical/table bounds (:meth:`RetrievalForward._clamp_state`)
    so an overshoot cannot drive the optics out of the table or τ_bot ≤ 0.
    """
    Se_inv = np.linalg.inv(Se)
    Sa_inv = np.linalg.inv(Sa)
    y = np.asarray(y, float)
    x = fwd._clamp_state(np.asarray(x0, float), s_nodes)
    history = []
    converged = False
    for _ in range(n_iter):
        Fx = np.asarray(fwd.forward(x, s_nodes), float)
        K = np.asarray(fwd.jacobian(x, s_nodes), float)       # (m, p)
        # cost J = ½‖y-F‖²_{Se⁻¹} + ½‖x-x_a‖²_{Sa⁻¹}
        r = y - Fx
        J = 0.5 * r @ Se_inv @ r + 0.5 * (x - x_a) @ Sa_inv @ (x - x_a)
        history.append(float(J))
        lhs = K.T @ Se_inv @ K + (1.0 + lm) * Sa_inv
        rhs = K.T @ Se_inv @ r - Sa_inv @ (x - x_a)
        dx = np.linalg.solve(lhs, rhs)
        x = fwd._clamp_state(x + dx, s_nodes)                 # projected GN step
        if np.linalg.norm(dx) < xtol * (np.linalg.norm(x) + xtol):
            converged = True
            break
    Fx = np.asarray(fwd.forward(x, s_nodes), float)
    K = np.asarray(fwd.jacobian(x, s_nodes), float)
    return x, K, Fx, history, converged


class RemeshWarning(UserWarning):
    """Emitted when re-meshing is triggered — or is warranted but disabled/capped."""


def gauss_newton_oe(fwd: RetrievalForward, y, s_nodes, x_a, Sa, Se, *,
                    x0=None, n_iter=12, lm=0.0, xtol=1e-4,
                    max_n_outer=2, prior_builder=None, filter_threshold=0.5,
                    margin=1, remesh_if_chi2_red_gt=2.0, warn=True):
    """Optimal estimation with **progressive lagged re-meshing** (Rodgers n-form).

    The retrieval grid ``s_nodes`` is in **normalized depth** s=τ/τ_bot∈[0,1) (so
    retrieving τ_bot never moves a node past the base). Inner loop: Rodgers
    Gauss–Newton on a fixed grid, projected onto physical bounds each step. The
    outer loop is a **gated last resort** that escalates progressively, capped by
    ``max_n_outer``:

    - ``max_n_outer=1`` → no re-meshing (select-once).
    - ``max_n_outer=2`` → may re-mesh with a **fixed node count** (placement only). [default]
    - ``max_n_outer=3`` → may further escalate to a **changed node count** (filter
      re-decides). Hard ceiling.

    Re-mesh fires only on the **"both"** trigger: reduced χ²
    ``= ‖y−F‖²_{Se⁻¹}/m > remesh_if_chi2_red_gt`` **and** the re-selected grid would
    actually move. The (recompiling) re-selection runs only at an *enabled* tier — at
    a tier *beyond* ``max_n_outer`` the warning fires on χ² **alone** (so a select-once
    user still learns the fit wanted more, without paying a recompile to find out).

    ``prior_builder(s_nodes) -> (x_a, Sa)`` is required for any re-meshing (rebuilds
    the prior on each re-selected grid and supplies σ_prior for the filter); a
    fixed-count re-selection (tier 2) keeps ``len(s_nodes)`` nodes, tier 3 lets the
    filter (``filter_threshold``) re-decide the count. ``warn`` toggles the
    :class:`RemeshWarning` messages. The initial grid is the ``s_nodes`` passed in
    (select it once with :func:`select_retrieval_grid` before calling).
    """
    import warnings
    s_nodes = np.asarray(s_nodes, float)
    x = np.asarray(x_a, float) if x0 is None else np.asarray(x0, float)
    Se_inv = np.linalg.inv(np.asarray(Se, float))
    m = len(np.asarray(y, float))

    x, K, Fx, hist, conv = _gn_inner(fwd, s_nodes, y, x, x_a, Sa, Se,
                                     n_iter=n_iter, lm=lm, xtol=xtol)
    full_hist = list(hist)

    def _chi2_red():
        # Whitened data misfit / m. χ²_red≈1 ⇔ residuals are noise-sized ⇔ "well-fit"
        # (no systematic signal left for re-meshing to capture); >thr ⇔ structural
        # misfit. Derivation + the noiseless-OSSE caveat: DESIGN_DECISIONS.md §10h.
        r0 = np.asarray(y, float) - np.asarray(Fx, float)
        return float(r0 @ Se_inv @ r0) / max(m, 1)

    thr = remesh_if_chi2_red_gt
    # Progressive escalation: n_outer=2 → fixed-count re-mesh, n_outer=3 → changed-count.
    n_outer = 2
    while prior_builder is not None and thr is not None and n_outer <= 3:
        chi2 = _chi2_red()
        if chi2 <= thr:
            break                                              # fit adequate → stop
        if n_outer > max_n_outer:                              # warranted but disabled/capped
            if warn and n_outer == 2:                          # max_n_outer=1: re-mesh off entirely
                warnings.warn(
                    f"Structural misfit (reduced χ²={chi2:.1f} > {thr}); re-meshing "
                    f"appears warranted — but it is disabled (max_n_outer={max_n_outer}). "
                    f"Returning the select-once result; consider raising max_n_outer "
                    f"or revising the prior/parameterization.", RemeshWarning, stacklevel=2)
            elif warn:                                         # max_n_outer=2: count change capped
                warnings.warn(
                    f"Misfit persists after fixed-count re-meshing (reduced "
                    f"χ²={chi2:.1f} > {thr}); a node-count change appears warranted but "
                    f"is capped (max_n_outer={max_n_outer}). Consider max_n_outer=3 or "
                    f"revising the prior.", RemeshWarning, stacklevel=2)
            break
        # enabled tier → "both": run the re-selection we will actually use.
        cur_x, cur_nodes = np.asarray(x, float), s_nodes
        _, cur_rbase, cur_taubot = fwd._split_state(cur_x, cur_nodes)
        cur_taubot = float(cur_taubot)
        re_of_tau = lambda t: fwd.profile(cur_x, cur_nodes, t)
        fixed_k = len(s_nodes) if n_outer == 2 else None       # tier 3 lets the count change
        new_s, _, _ = select_retrieval_grid(
            fwd, x, s_nodes, fixed_k, Se=Se, prior_builder=prior_builder,
            filter_threshold=filter_threshold, margin=margin, re_of_tau=re_of_tau)
        grid_changed = not (len(new_s) == len(s_nodes)
                            and np.allclose(new_s, s_nodes, atol=1e-3))
        if not grid_changed:
            break                                              # re-mesh would not help
        if warn and n_outer == 2:
            warnings.warn(
                f"Re-meshing (n_outer={n_outer}): persistent structural misfit "
                f"(reduced χ²={chi2:.1f} > {thr}) and node placement would shift; "
                f"re-selecting placement at fixed node count (k={len(s_nodes)}) — "
                f"this triggers a recompile.", RemeshWarning, stacklevel=2)
        elif warn:
            warnings.warn(
                f"Re-meshing (n_outer={n_outer}): misfit persists after placement "
                f"re-selection (reduced χ²={chi2:.1f} > {thr}); re-selecting both node "
                f"count and placement (k: {len(s_nodes)}→{len(new_s)}) — this triggers "
                f"a larger recompile.", RemeshWarning, stacklevel=2)
        # apply the re-mesh: re-map the r_e-node block onto the new normalized grid,
        # carrying the current base/τ_bot estimates into the trailing joint entries.
        x = np.asarray(fwd.profile(cur_x, cur_nodes, new_s * cur_taubot), float)
        if fwd.retrieve_r_base:
            x = np.append(x, float(cur_rbase))
        if fwd.retrieve_tau_bot:
            x = np.append(x, cur_taubot)
        x_a, Sa = prior_builder(new_s)
        s_nodes = new_s
        x, K, Fx, hist, conv = _gn_inner(fwd, s_nodes, y, x, x_a, Sa, Se,
                                         n_iter=n_iter, lm=lm, xtol=xtol)
        full_hist += hist
        n_outer += 1

    return OEResult(x=x, tau_nodes=s_nodes, x_a=x_a, Sa=Sa, Se=Se, K=K,
                    y=np.asarray(y), Fx=Fx, cost_history=full_hist, converged=conv)


# ============================================================================
# 5. Posterior uncertainty quantification (the deliverable-1 UQ)
# ============================================================================
@dataclass
class Posterior:
    S_hat: np.ndarray          # posterior covariance
    error: np.ndarray          # √diag(S_hat) — retrieval 1σ error per node
    A: np.ndarray              # averaging kernel matrix
    dofs: float                # degrees of freedom for signal = tr(A)
    sic: float                 # Shannon information content [bits] = ½ log₂|Sa Ŝ⁻¹|
    data_fraction: np.ndarray  # per-node 1 − Ŝ_ii/Sa_ii — measurement vs prior


def posterior_diagnostics(K, Sa, Se) -> Posterior:
    """Rodgers posterior covariance, averaging kernels, DOFS, SIC, and the per-node
    measurement-vs-prior split.

    ``Ŝ = (Kᵀ Sε⁻¹ K + Sa⁻¹)⁻¹``;  ``A = Ŝ Kᵀ Sε⁻¹ K``;  ``DOFS = tr(A)``.
    The averaging-kernel rows show *where in τ* each retrieved level draws its
    information (peaks spread through the column ⇔ vertical resolving power).

    **DOFS vs SIC** — two *complementary* information measures (report both):
    ``DOFS = tr(A)`` counts the number of *independent dimensions* the measurement
    constrains (how many features), thresholding each direction near 0/1. ``SIC =
    ½ log₂|Sa Ŝ⁻¹|`` (bits) is the total Shannon information — the *magnitude* of
    the variance reduction across all directions (a direction reduced 100× adds
    more bits than one reduced 2×). They can diverge: a thin cloud has *few* DOF
    (little depth to vary) yet each is sharply measured (high SIC per DOF), while a
    thick cloud has *more* DOF but diffusion caps the per-stream information (lower
    SIC per DOF). DOFS alone hides this; SIC exposes it.

    ``data_fraction[i] = 1 − Ŝ_ii / Sa_ii`` is the **fractional variance
    reduction** at node i — how much of that node's value the *measurement*
    pinned down versus what it inherited from the prior (0 = pure prior, 1 = fully
    measured); the plain-language per-node companion to the (scalar) DOFS/SIC.
    """
    K = np.asarray(K, float)
    Sa = np.asarray(Sa, float)
    Se_inv = np.linalg.inv(np.asarray(Se, float))
    Sa_inv = np.linalg.inv(Sa)
    KtSeK = K.T @ Se_inv @ K
    S_hat = np.linalg.inv(KtSeK + Sa_inv)
    A = S_hat @ KtSeK
    data_fraction = 1.0 - np.diag(S_hat) / np.diag(Sa)
    # Shannon information content (Rodgers eq. 2.80): H = ½ log₂(|Sa|/|Ŝ|) [bits].
    _, logdet_Sa = np.linalg.slogdet(Sa)
    _, logdet_Shat = np.linalg.slogdet(S_hat)
    sic = float(0.5 * (logdet_Sa - logdet_Shat) / np.log(2.0))
    return Posterior(S_hat=S_hat, error=np.sqrt(np.diag(S_hat)), A=A,
                     dofs=float(np.trace(A)), sic=sic, data_fraction=data_fraction)


def dofs_by_component(post, n_nodes, *, retrieve_r_base=False,
                      retrieve_tau_bot=False):
    """Split ``DOFS = tr(A) = Σ diag(A)`` into per-component contributions.

    ``diag(A)[i]`` is the information (in DOF units) the measurement supplies to
    state element ``i``, and they sum to the scalar DOFS. Groups them into the r_e
    **profile** (the first ``n_nodes`` nodes) and the retrieved scalars
    ``r_base`` / ``τ_bot`` when present — directly answering "of the DOFS gained by
    making the base/depth unknown, how much does the measurement actually supply to
    them vs the profile" (PO). Note these are *self*-information diagonals; the
    off-diagonal of A shows the (generally non-trivial) cross-talk between them.
    """
    d = np.diag(np.asarray(post.A, float))
    out = {"profile": float(d[:n_nodes].sum()),
           "profile_nodes": np.asarray(d[:n_nodes]).copy()}
    idx = n_nodes
    if retrieve_r_base:
        out["r_base"] = float(d[idx]); idx += 1
    if retrieve_tau_bot:
        out["tau_bot"] = float(d[idx]); idx += 1
    out["total"] = float(d.sum())
    return out


# ============================================================================
# 6. OSSE helper
# ============================================================================
def osse_observation(fwd: RetrievalForward, tau_truth, re_truth, *, noise=None,
                     seed=0):
    """Synthetic observation ``y = F(x_true)`` from an in-situ truth profile.

    The truth ``r_e(τ)`` is the dense in-situ profile; it is fed to the forward
    model directly, as node values at the truth's **normalized depths**
    ``s = τ/τ_bot``. In **joint** mode the truth ``τ_bot`` / ``r_base`` (= the last
    in-situ point) are appended to the state so the synthetic measurement is
    generated at the true cloud depth/base — this is *defining the synthetic world*,
    not a leak (the leak would be letting the *retrieval* know them, which the
    prior/first-guess no longer does).

    **Noiseless by default** (the OSSE decision, DESIGN §10b): the noise here is
    *measurement* noise on the ToA radiances (instrument noise), not VOCALS truth
    uncertainty (DESIGN §12). ``noise`` may be **(a)** a :class:`noise_model.NoiseModel`
    (a PACE/OCI instrument model — a Gaussian realization is drawn via its
    ``sample`` using ``fwd.n_bands`` for the band-major layout), or **(b)** an
    explicit per-observation σ (scalar or array). Build the matching assumed ``Se``
    with :func:`make_Se`.
    """
    tau_truth = np.asarray(tau_truth, float)
    re_truth = np.asarray(re_truth, float)
    tau_bot_truth = float(tau_truth[-1])
    r_base_truth = float(re_truth[-1])
    # interior truth nodes (exclude the base point — appended via the state/anchor),
    # expressed in normalized depth s=τ/τ_bot to match the forward parameterisation.
    interior = tau_truth < tau_bot_truth - 1e-9
    s_truth = tau_truth[interior] / tau_bot_truth
    x = re_truth[interior]
    if fwd.retrieve_r_base:
        x = np.append(x, r_base_truth)
    if fwd.retrieve_tau_bot:
        x = np.append(x, tau_bot_truth)
    y = np.asarray(fwd.forward(x, s_truth), float)
    if noise is not None:
        if hasattr(noise, "sample"):                 # a noise_model.NoiseModel
            y = np.asarray(noise.sample(y, n_bands=fwd.n_bands, seed=seed), float)
        else:                                        # explicit per-obs σ (scalar/array)
            rng = np.random.default_rng(seed)
            y = y + rng.normal(0.0, 1.0, size=y.shape) * np.asarray(noise)
    return y


def make_Se(fwd: RetrievalForward, y, noise_model):
    """Assumed measurement-error covariance ``Se = diag(σ²)`` from a NoiseModel.

    The OE counterpart to :func:`osse_observation`'s perturbation: it supplies the
    band-major ``fwd.n_bands`` so a :class:`noise_model.NoiseModel` can apply
    per-band coefficients. Use this in place of the old hand-picked
    ``Se = diag((0.03·max(|y|, 0.02))²)`` to ground the weighting in the PACE/OCI
    instrument model (DESIGN §12). Noiseless OSSE still needs ``Se`` for weighting.
    """
    return noise_model.Se(np.asarray(y, float), n_bands=fwd.n_bands)
