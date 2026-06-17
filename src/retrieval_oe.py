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
                 tol=1e-3, re_class="re5-linear",
                 retrieve_tau_bot=False, retrieve_r_base=False):
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
        self.view_mu = jnp.asarray(view_mu, dtype=float)
        self.view_phi = jnp.asarray(view_phi, dtype=float)
        self.n_view = int(self.view_mu.shape[0])
        self.m = self.n_bands * self.n_view          # observation dimension
        if BDRF_bands is None:
            BDRF_bands = [()] * self.n_bands
        self.setups = [
            riccati_setup(NQuad, I0, phi0, mu0, NFourier=NFourier,
                          NLeg_all=NLeg_all, BDRF_Fourier_modes=bdrf,
                          delta_M_scaling=True, NT_cor=True, tol=tol)
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

    def _split_state(self, x, tau_nodes):
        """Decode the joint state ``x`` → ``(r_nodes, r_base, τ_bot)``.

        ``r_nodes`` are the first ``len(tau_nodes)`` entries (r_e at the free
        nodes, incl. cloud top τ=0). ``r_base`` / ``τ_bot`` are read from the
        trailing entries when retrieved (the joint retrieval, PO), else fall back
        to the fixed constructor values ``self.r_base`` / ``self.tau_bot``. The
        returned ``r_base`` / ``τ_bot`` are **traced scalars** in joint mode, so
        ``∂y/∂r_base`` and ``∂y/∂τ_bot`` flow through autodiff (τ_bot is a traced
        ``riccati_solve`` arg by construction, DESIGN §7).
        """
        x = jnp.asarray(x, float)
        k = int(jnp.asarray(tau_nodes).shape[0])     # static (shape known at trace)
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

    # -- r_e(τ) interpolant knots: free nodes + base anchor at τ_bot ----------
    def _knots_vals(self, x, tau_nodes):
        """Interpolation knots/values + the (safeguarded) base optical depth.

        Returns ``(knots, vals, τ_bot)``: r_e at ``[tau_nodes, τ_bot]``, with
        ``τ_bot`` floored to just below... above the deepest free node so the
        ``jnp.interp`` abscissa stays strictly sorted even when the retrieved
        ``τ_bot`` is traced. The floor is a numerical guard only — a sensible
        ``τ_bot`` prior keeps it inactive (it would activate only if the retrieval
        drove τ_bot beneath an interior node, the pathological crossing case).
        """
        r_nodes, r_base, tau_bot = self._split_state(x, tau_nodes)
        tau_nodes = jnp.asarray(tau_nodes, float)
        tau_bot = jnp.asarray(tau_bot, float)
        if tau_nodes.shape[0] > 0:
            tau_bot = jnp.maximum(tau_bot, tau_nodes[-1] + 1e-6)
        knots = jnp.concatenate([tau_nodes, jnp.reshape(tau_bot, (1,))])
        vals = jnp.concatenate([jnp.asarray(r_nodes, float),
                                jnp.reshape(jnp.asarray(r_base, float), (1,))])
        return knots, vals, tau_bot

    def _re_of_tau(self, tau, knots, vals):
        """The profile parameterisation r_e(τ) from the node values — **the
        function-class lever** (OUTSTANDING §B′; DESIGN §3d).

        This is NOT a post-hoc / cosmetic interpolation: it is *inside* the forward
        map F(x), so it defines what is retrieved and what every solve integrates.
        Plot the result with :meth:`profile`, which routes through here so the
        display mirrors F(x).

        **Default: r_e⁵-linear (adiabatic).** The adiabatic effective radius grows as
        r_e ∝ τ^(1/5) in optical depth: r_e³ ∝ LWC ∝ geometric height z, and the
        extinction β ∝ r_e² ∝ z^(2/3) makes τ = ∫β dz ∝ z^(5/3), so LWC ∝ τ^(3/5) and
        r_e ∝ τ^(1/5) (equivalently the canonical adiabatic N_d ∝ τ^(1/2) r_e^(-5/2)).
        So r_e⁵ is what is linear in τ: it is interpolated linearly and 5th-rooted.
        This (i) is the adiabatic class, (ii) is coherent with the adiabatic prior
        (same 1/5 law ⇒ represents ``x_a`` exactly), and (iii) gets per-segment
        curvature from the two endpoint values, so it has **no grid-size coupling**.
        It is C⁰ (kinked at nodes); with finite ``r_base`` the slope at base stays
        finite (no root cusp).

        The class is the ``re_class`` constructor lever, switched **here and only
        here** so it propagates through forward / mode-amplitudes / ODE-grid /
        Jacobian / re-meshing / display by construction::

            "re5-linear" (default, adiabatic): jnp.interp(tau,knots,vals**5)**(1/5)
            "linear" (impute-nothing baseline):  jnp.interp(tau, knots, vals)
            PCHIP (C¹):  not wired in — needs ≥3 nodes, couples to node count, so its
                         curvature is an FD artifact at low DOF; the clean class test
                         is model comparison (linear vs adiabatic fit χ²), §B′.

        "Which class" is an inverse-problem bias–variance decision (OUTSTANDING §B′),
        bounded above by the integrator order (~C⁶, §1) and far more tightly by DOFS.
        """
        if self.re_class == "linear":
            return jnp.interp(tau, knots, vals)
        return jnp.interp(tau, knots, vals ** 5) ** (1.0 / 5.0)   # re5-linear (adiabatic)

    def profile(self, x, tau_nodes, tau):
        """Evaluate the retrieved r_e(τ) **exactly as the forward integrates it**.

        Free nodes ``x`` at ``tau_nodes`` plus the fixed base anchor ``(τ_bot,
        r_base)``, through :meth:`_re_of_tau`. Use this for plotting / downstream so
        the displayed curve mirrors F(x) by construction — the interpolation is part
        of the retrieval, not an independent post-hoc choice.
        """
        knots, vals, _ = self._knots_vals(x, tau_nodes)
        return np.asarray(self._re_of_tau(jnp.asarray(tau, float), knots, vals))

    def _band_reflectance(self, opt, setup, K, knots, vals, tau_bot):
        def om(tau):
            return table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]

        def leg(tau):
            return table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]

        res = riccati_solve(setup, om, leg, tau_bot, num_modes=K)
        u = jnp.stack([eval_radiance(setup, res, self.view_mu[i], self.view_phi[i])
                       for i in range(self.n_view)])           # (n_view,)
        return jnp.pi * u / (self.mu0 * self.I0)

    def _forward_raw(self, x, tau_nodes):
        knots, vals, tau_bot = self._knots_vals(x, tau_nodes)
        return jnp.concatenate([
            self._band_reflectance(opt, setup, K, knots, vals, tau_bot)
            for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
        ])                                                     # (n_bands*n_view,)

    # -- per-mode reflectance amplitudes (drives the S_ε mode selector) -------
    def mode_amplitudes(self, x_ref, tau_nodes):
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
        knots, vals, tau_bot = self._knots_vals(x_ref, tau_nodes)
        amps = []
        for opt, setup in zip(self.opt_bands, self.setups):
            def om(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]

            def leg(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]

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
    def forward(self, x, tau_nodes):
        if self._fwd_jit is None:
            self._fwd_jit = jax.jit(self._forward_raw)
        return self._fwd_jit(jnp.asarray(x, float), jnp.asarray(tau_nodes, float))

    def jacobian(self, x, tau_nodes):
        """K = ∂y/∂x  (m × p), reverse-mode through the jitted seam."""
        if self._jac_jit is None:
            self._jac_jit = jax.jit(jax.jacrev(self._forward_raw, argnums=0))
        return self._jac_jit(jnp.asarray(x, float), jnp.asarray(tau_nodes, float))

    # -- ODE grid (adaptive candidate pool, DESIGN §3) -----------------------
    def ode_grid(self, x, tau_nodes):
        """Adaptive Kvaerno5 τ-grid at the given state (first band, offline).

        Integrates to the **current** ``τ_bot`` (the retrieved value in joint mode,
        decoded from ``x``), so the candidate pool tracks the estimated cloud depth.
        """
        knots, vals, tau_bot = self._knots_vals(x, tau_nodes)
        opt = self.opt_bands[0]
        om = lambda tau: table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]
        leg = lambda tau: table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]
        *_, tau_grid = pydisort_riccati_jax(
            float(tau_bot), om, leg, self.setups[0].NQuad, self.mu0, self.I0,
            self.setups[0].phi0, delta_M_scaling=True, NT_cor=True,
            NLeg_all=self.opt_bands[0]["leg"].shape[-1])
        return np.asarray(tau_grid, float)

    # -- pool sensitivity (parameterise r_e on an arbitrary τ-grid) ----------
    def jacobian_on_grid(self, re_vals, tau_grid, tau_bot=None):
        """K_pool = ∂y/∂r_e(τ_j) at the pool nodes ``tau_grid`` (m × len).

        ``tau_grid`` and ``tau_bot`` are **traced** arguments, so re-selections at a
        stable ODE-grid size reuse the compiled Jacobian (recompile-free lagged
        re-meshing); a new pool size compiles once and caches. ``tau_bot`` is the
        integration limit (the current/retrieved cloud base); defaults to the fixed
        ``self.tau_bot`` for the non-joint path.
        """
        if tau_bot is None:
            tau_bot = self.tau_bot
        if self._jac_grid_jit is None:
            def fwd(rv, tg, tb):
                return jnp.concatenate([
                    self._band_reflectance(opt, setup, K, tg, rv, tb)
                    for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
                ])
            self._jac_grid_jit = jax.jit(jax.jacrev(fwd, argnums=0))
        return np.asarray(self._jac_grid_jit(jnp.asarray(re_vals, float),
                                             jnp.asarray(tau_grid, float),
                                             jnp.asarray(tau_bot, float)))


def build_forward(*args, **kw):
    """Functional alias for :class:`RetrievalForward`."""
    return RetrievalForward(*args, **kw)


# ============================================================================
# 1b. Azimuthal mode-count selector (S_ε, replaces the relative Cauchy test)
# ============================================================================
def select_num_modes(fwd: RetrievalForward, x_ref, tau_nodes, Se, *, frac=1/3.0):
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
    x_ref, tau_nodes : reference state at which to measure the mode amplitudes
        (the mode spectrum is weakly state-dependent — pick a representative
        first-guess profile, e.g. the prior mean).
    Se : (m, m) array — observation error covariance (reflectance²).
    frac : float — keep-threshold as a fraction of the minimum observation σ_ε.
    """
    sigma_eps = np.sqrt(np.diag(np.asarray(Se, float)))
    thresh = float(frac) * float(np.min(sigma_eps))
    amps = fwd.mode_amplitudes(x_ref, tau_nodes)               # (n_bands, NFourier)
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
def select_retrieval_grid(fwd: RetrievalForward, x, tau_nodes, k_active, *,
                          re_of_tau=None, k_pool=20):
    """Sensitivity-select ``k_active`` τ-nodes from the adaptive ODE pool.

    1. Run the adaptive solve at the current state → ODE τ-grid (candidate pool,
       a trustworthy *superset* of the informative points, DESIGN §3a).
    2. Form the ToA Jacobian ``∂y/∂r_e(τ_j)`` on that pool (autodiff).
    3. QR-with-column-pivoting (QRCP) on the (scaled) Jacobian ranks the nodes by
       independent information; keep the top ``k_active`` — always including the
       cloud-top node τ=0 (most informative) — and return them sorted.

    Parameters
    ----------
    re_of_tau : callable or None
        Maps the pool τ to current r_e (to evaluate the pool's r_e values). If
        None, uses the interpolant of ``(tau_nodes, x)`` + base anchor.
    """
    # current cloud base (retrieved value in joint mode, else the fixed anchor)
    cur_tau_bot = float(fwd._split_state(x, tau_nodes)[2])
    tau_pool = fwd.ode_grid(x, tau_nodes)
    tau_pool = np.unique(np.clip(tau_pool, 0.0, cur_tau_bot))
    # Resample the adaptive ODE grid to a FIXED cardinality (preserving its
    # density — uniform in node-index) so the pool Jacobian compiles once and
    # every lagged re-selection is recompile-free (tau_pool is a traced arg).
    if k_pool is not None and tau_pool.size != k_pool and tau_pool.size > 1:
        tau_pool = np.interp(np.linspace(0.0, tau_pool.size - 1, k_pool),
                             np.arange(tau_pool.size), tau_pool)
    if re_of_tau is None:
        re_pool = fwd.profile(x, tau_nodes, tau_pool)   # same class as the forward
    else:
        re_pool = np.asarray([float(re_of_tau(t)) for t in tau_pool])

    K_pool = fwd.jacobian_on_grid(re_pool, tau_pool, cur_tau_bot)   # (m, P)

    # Column pivoting on the sensitivity matrix: rank τ-nodes by independent info.
    from scipy.linalg import qr
    _, _, piv = qr(K_pool, mode="economic", pivoting=True)

    # Candidate set = interior nodes only (the fixed base anchor τ_bot is appended
    # by the forward, not retrieved). Always include the cloud-top node (τ≈0, most
    # informative). Return EXACTLY k_active nodes so the retrieval-grid size is
    # constant across lagged re-selections ⇒ r-refinement is recompile-free.
    interior = tau_pool < cur_tau_bot - 1e-6
    top_idx = int(np.argmin(tau_pool))
    chosen = [top_idx]
    for p in piv:                                              # QRCP order
        if len(chosen) >= k_active:
            break
        if int(p) != top_idx and interior[p]:
            chosen.append(int(p))
    if len(chosen) < k_active:                                 # degenerate pool: pad by depth
        for j in np.argsort(tau_pool):
            if len(chosen) >= k_active:
                break
            if j not in chosen and interior[j]:
                chosen.append(int(j))
    chosen = sorted(set(chosen), key=lambda j: tau_pool[j])
    tau_sel = tau_pool[chosen]
    re_sel = re_pool[chosen]
    return tau_sel, re_sel, dict(tau_pool=tau_pool, K_pool=K_pool, piv=piv)


# ----------------------------------------------------------------------------
# 2b. Data-driven retrieval-node count k_active  (SO1)
# ----------------------------------------------------------------------------
def auto_k_active(K_pool, Se, Sa_pool, *, method="filter", factor=1.5, margin=1,
                  k_min=1, k_max=8):
    """How many r_e(τ) nodes the measurement can independently support (SO1).

    Computed **once** at the first guess from the POOL Jacobian (so the chosen
    count is then frozen for the retrieval — the forward/Jacobian still compile
    once, mirroring :func:`select_num_modes` for the Fourier modes). Two
    estimators, returned with diagnostics so they can be cross-checked:

    - ``method="dofs"`` — ``k = round(factor · DOFS)`` (the user's SO1 proposal),
      ``factor ≥ 1`` so a few prior-filled nodes are kept beyond the resolvable
      rank (a node basis is never fully independent: ``DOFS = tr(A) < n_nodes``
      intrinsically, and letting the prior fill the surplus is a *feature* of
      regularised OE — OUTSTANDING §G). DOFS is the Rodgers ``tr(A)`` on the pool.

    - ``method="filter"`` *(default — NOT routed through DOFS)* — whiten
      ``K̃ = Se^(-1/2) · K_pool · diag(σ_prior)`` (rows by noise, columns by the
      prior √variance), QRCP → pivoted R-diagonal ``r_1 ≥ r_2 ≥ …`` (each node's
      **marginal** information in SNR units). Keep the **data-dominated**
      directions — filter factor ``f_i = r_i²/(1+r_i²) ≥ ½ ⇔ r_i ≥ 1`` (Rodgers;
      the literal "fraction from data") — plus a fixed ``margin`` of prior-filled
      ones. ``Σ f_i ≈ DOFS`` gives a built-in cross-check that the two estimators
      agree (and a robustness probe on the DOFS itself).

    Both are clamped to ``[k_min, min(k_max, n_obs, pool_size)]``; cloud-top is
    always retained by :func:`select_retrieval_grid`, not counted here.

    Returns ``(k, info)``.
    """
    K = np.asarray(K_pool, float)
    Se = np.asarray(Se, float)
    Sa = np.asarray(Sa_pool, float)
    # Se^{-1/2} for general SPD observation covariance (here Se is diagonal).
    w, V = np.linalg.eigh(Se)
    Se_half_inv = (V / np.sqrt(w)) @ V.T

    # filter-factor spectrum (always computed — it is the DOFS cross-check)
    sig = np.sqrt(np.clip(np.diag(Sa), 0.0, None))
    K_tilde = Se_half_inv @ K @ np.diag(sig)
    from scipy.linalg import qr
    _, R, _ = qr(K_tilde, mode="economic", pivoting=True)
    rdiag = np.abs(np.diag(R))
    f = rdiag ** 2 / (1.0 + rdiag ** 2)
    n_data = int(np.count_nonzero(rdiag >= 1.0))
    sum_f = float(f.sum())

    # DOFS on the pool (correlated Sa)
    Se_inv = np.linalg.inv(Se)
    Sa_inv = np.linalg.inv(Sa)
    KtSeK = K.T @ Se_inv @ K
    A = np.linalg.solve(KtSeK + Sa_inv, KtSeK)
    dofs = float(np.trace(A))

    if method == "dofs":
        k = int(round(float(factor) * dofs))
    elif method == "filter":
        k = n_data + int(margin)
    else:
        raise ValueError(f"unknown method {method!r}; 'filter' or 'dofs'")

    k_hi = int(min(k_max, K.shape[0], K.shape[1]))
    k = int(np.clip(k, k_min, max(k_min, k_hi)))
    info = dict(method=method, k_active=k, dofs=dofs, sum_filter_factor=sum_f,
                n_data_dominated=n_data, margin=int(margin), factor=float(factor),
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


def make_joint_prior(tau_nodes, *, tau_bot_prior, r_top_prior, r_base_prior,
                     retrieve_r_base=True, retrieve_tau_bot=True,
                     sigma_top=5.0, sigma_base=8.0, sigma_tau_bot=None,
                     corr_length=None, strength=1.0):
    """Joint prior over the retrieved state ``x = [r_e nodes, (r_base), (τ_bot)]``.

    The r_e nodes — *and* ``r_base``, which **is** r_e at the base, so it joins the
    block as the deepest r_e node at ``τ_bot_prior`` — form one correlated,
    depth-increasing-σ adiabatic block (:func:`make_adiabatic_prior`). ``τ_bot`` is
    appended as an **independent broad scalar** dimension (block-diagonal: cloud
    geometric/optical depth is a different physical quantity from droplet size,
    coupled only weakly, so we do not assert a cross-correlation in the prior).

    All means are **leak-free** — generic/climatological ``r_top_prior``,
    ``r_base_prior``, ``tau_bot_prior``, *never* the truth. Broad σ's make it the
    **weakly-informative (Option 2)** headline prior: the data sets ``τ_bot`` (a
    conservative band measures optical thickness directly) and the upper-cloud
    r_e, while the prior fills the radiatively shielded base. The ``retrieve_*``
    flags must match the :class:`RetrievalForward` they will be used with, so the
    state layout (and Sa block structure) line up.

    ``sigma_tau_bot`` defaults to ``0.5·τ_bot_prior`` (~50 % relative — broad).
    """
    tau_nodes = np.asarray(tau_nodes, float)
    # r_e block: extend the free nodes with the base node at τ_bot when r_base is
    # retrieved (make_adiabatic_prior's frac=0 there ⇒ its mean = r_base_prior).
    nodes_aug = np.append(tau_nodes, tau_bot_prior) if retrieve_r_base else tau_nodes
    x_a, Sa = make_adiabatic_prior(
        nodes_aug, tau_bot_prior, r_base_prior, r_top_prior,
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


def make_climatology_prior(tau_nodes, clim, *, retrieve_r_base=True,
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
        tau_nodes, tau_bot_prior=clim["tau_bot_mean"],
        r_top_prior=clim["r_top_mean"], r_base_prior=clim["r_base_mean"],
        retrieve_r_base=retrieve_r_base, retrieve_tau_bot=retrieve_tau_bot,
        sigma_top=clim["r_top_std"], sigma_base=clim["r_base_std"],
        sigma_tau_bot=clim["tau_bot_std"], corr_length=corr_length,
        strength=strength)


# ============================================================================
# 4. Gauss–Newton optimal estimation (Rodgers n-form) + lagged re-meshing
# ============================================================================
@dataclass
class OEResult:
    x: np.ndarray              # retrieved r_e at the (final) grid nodes
    tau_nodes: np.ndarray      # final retrieval-grid τ
    x_a: np.ndarray            # prior mean on the final grid
    Sa: np.ndarray             # prior covariance on the final grid
    Se: np.ndarray             # observation error covariance
    K: np.ndarray              # final Jacobian
    y: np.ndarray              # observation
    Fx: np.ndarray             # final forward
    cost_history: list = field(default_factory=list)
    converged: bool = False


def _gn_inner(fwd, y, tau_nodes, x0, x_a, Sa, Se, *, n_iter, lm, xtol):
    """Inner Gauss–Newton on a fixed retrieval grid (Rodgers n-form)."""
    Se_inv = np.linalg.inv(Se)
    Sa_inv = np.linalg.inv(Sa)
    y = np.asarray(y, float)
    x = np.asarray(x0, float).copy()
    history = []
    converged = False
    for _ in range(n_iter):
        Fx = np.asarray(fwd.forward(x, tau_nodes), float)
        K = np.asarray(fwd.jacobian(x, tau_nodes), float)     # (m, p)
        # cost J = ½‖y-F‖²_{Se⁻¹} + ½‖x-x_a‖²_{Sa⁻¹}
        r = y - Fx
        J = 0.5 * r @ Se_inv @ r + 0.5 * (x - x_a) @ Sa_inv @ (x - x_a)
        history.append(float(J))
        lhs = K.T @ Se_inv @ K + (1.0 + lm) * Sa_inv
        rhs = K.T @ Se_inv @ r - Sa_inv @ (x - x_a)
        dx = np.linalg.solve(lhs, rhs)
        x = x + dx
        if np.linalg.norm(dx) < xtol * (np.linalg.norm(x) + xtol):
            converged = True
            break
    Fx = np.asarray(fwd.forward(x, tau_nodes), float)
    K = np.asarray(fwd.jacobian(x, tau_nodes), float)
    return x, K, Fx, history, converged


def gauss_newton_oe(fwd: RetrievalForward, y, tau_nodes, x_a, Sa, Se, *,
                    x0=None, n_iter=12, lm=0.0, xtol=1e-4,
                    n_outer=1, k_active=None, prior_builder=None,
                    r_top_prior=None):
    """Optimal estimation with optional **lagged re-meshing** (outer loop).

    Inner loop: Rodgers Gauss–Newton on a fixed grid. Outer loop (``n_outer>1``):
    after each inner solve, re-select the retrieval grid by QRCP at the *current*
    estimate (``select_retrieval_grid``), re-map the state (current ``r_e(τ)``
    sampled at the new nodes) and rebuild the prior on the new nodes
    (``prior_builder``), then re-run the inner GN. This corrects the first-guess
    node-placement bias (OUTSTANDING B). ``n_outer=1`` ⇒ select-once.

    ``prior_builder(tau_nodes) -> (x_a, Sa)`` is required when ``n_outer>1`` to
    rebuild the prior on each new grid.
    """
    tau_nodes = np.asarray(tau_nodes, float)
    x = np.asarray(x_a, float) if x0 is None else np.asarray(x0, float)

    x, K, Fx, hist, conv = _gn_inner(fwd, y, tau_nodes, x, x_a, Sa, Se,
                                     n_iter=n_iter, lm=lm, xtol=xtol)
    full_hist = list(hist)

    for _ in range(max(0, n_outer - 1)):
        if k_active is None or prior_builder is None:
            break
        # current r_e(τ) via the forward's own parameterisation (the lever) so the
        # re-mesh re-mapping mirrors F(x) exactly — not an independent interpolation.
        cur_x, cur_nodes = np.asarray(x, float), tau_nodes
        re_of_tau = lambda t: fwd.profile(cur_x, cur_nodes, t)
        new_tau, new_re, _ = select_retrieval_grid(
            fwd, x, tau_nodes, k_active, re_of_tau=re_of_tau)
        if (len(new_tau) == len(tau_nodes)
                and np.allclose(new_tau, tau_nodes, atol=1e-3)):
            break                                              # grid stabilised
        # re-map the r_e-node block onto the new grid (same parameterisation),
        # carrying the current retrieved base/τ_bot estimates into the trailing
        # joint-state entries (the grid re-selection touches only the r_e nodes).
        _, cur_rbase, cur_taubot = fwd._split_state(cur_x, cur_nodes)
        x = np.asarray(fwd.profile(cur_x, cur_nodes, new_tau), float)
        if fwd.retrieve_r_base:
            x = np.append(x, float(cur_rbase))
        if fwd.retrieve_tau_bot:
            x = np.append(x, float(cur_taubot))
        x_a, Sa = prior_builder(new_tau)
        tau_nodes = new_tau
        # K may change size ⇒ recompiles once for this new k (rare).
        x, K, Fx, hist, conv = _gn_inner(fwd, y, tau_nodes, x, x_a, Sa, Se,
                                         n_iter=n_iter, lm=lm, xtol=xtol)
        full_hist += hist

    return OEResult(x=x, tau_nodes=tau_nodes, x_a=x_a, Sa=Sa, Se=Se, K=K, y=np.asarray(y),
                    Fx=Fx, cost_history=full_hist, converged=conv)


# ============================================================================
# 5. Posterior uncertainty quantification (the deliverable-1 UQ)
# ============================================================================
@dataclass
class Posterior:
    S_hat: np.ndarray          # posterior covariance
    error: np.ndarray          # √diag(S_hat) — retrieval 1σ error per node
    A: np.ndarray              # averaging kernel matrix
    dofs: float                # degrees of freedom for signal = tr(A)
    data_fraction: np.ndarray  # per-node 1 − Ŝ_ii/Sa_ii — measurement vs prior


def posterior_diagnostics(K, Sa, Se) -> Posterior:
    """Rodgers posterior covariance, averaging kernels, DOFS, and the per-node
    measurement-vs-prior split.

    ``Ŝ = (Kᵀ Sε⁻¹ K + Sa⁻¹)⁻¹``;  ``A = Ŝ Kᵀ Sε⁻¹ K``;  ``DOFS = tr(A)``.
    The averaging-kernel rows show *where in τ* each retrieved level draws its
    information (peaks spread through the column ⇔ vertical resolving power);
    DOFS is the number of independent pieces of profile information.

    ``data_fraction[i] = 1 − Ŝ_ii / Sa_ii`` is the **fractional variance
    reduction** at node i — how much of that node's value the *measurement*
    pinned down versus what it inherited from the prior (0 = pure prior, 1 = fully
    measured). It is the plain-language, per-node companion to the (scalar) DOFS
    and the (matrix) averaging kernel: a labelled "x% measured / (1−x)% prior" bar.
    """
    K = np.asarray(K, float)
    Sa = np.asarray(Sa, float)
    Se_inv = np.linalg.inv(np.asarray(Se, float))
    Sa_inv = np.linalg.inv(Sa)
    KtSeK = K.T @ Se_inv @ K
    S_hat = np.linalg.inv(KtSeK + Sa_inv)
    A = S_hat @ KtSeK
    data_fraction = 1.0 - np.diag(S_hat) / np.diag(Sa)
    return Posterior(S_hat=S_hat, error=np.sqrt(np.diag(S_hat)), A=A,
                     dofs=float(np.trace(A)), data_fraction=data_fraction)


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
    model directly (as the node values on its own τ-grid). In **joint** mode the
    truth ``τ_bot`` / ``r_base`` (= the last in-situ point) are appended to the
    state so the synthetic measurement is generated at the true cloud depth/base —
    this is *defining the synthetic world*, not a leak (the leak would be letting
    the *retrieval* know them, which the prior/first-guess no longer does).
    Noiseless by default (the OSSE decision); pass ``noise`` (a per-observation σ
    vector or scalar) to add a Gaussian realization.
    """
    tau_truth = np.asarray(tau_truth, float)
    re_truth = np.asarray(re_truth, float)
    tau_bot_truth = float(tau_truth[-1])
    r_base_truth = float(re_truth[-1])
    # interior truth nodes (exclude the base point — appended via the state/anchor)
    interior = tau_truth < tau_bot_truth - 1e-9
    x = re_truth[interior]
    if fwd.retrieve_r_base:
        x = np.append(x, r_base_truth)
    if fwd.retrieve_tau_bot:
        x = np.append(x, tau_bot_truth)
    y = np.asarray(fwd.forward(x, tau_truth[interior]), float)
    if noise is not None:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, 1.0, size=y.shape) * np.asarray(noise)
    return y
