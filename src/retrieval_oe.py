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
  ``r_e(τ)`` is the interpolant through the nodes + base anchor — **r_e³-linear
  (adiabatic) by default**, set by the single lever :meth:`RetrievalForward._re_of_tau`.
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
    riccati_setup, riccati_solve, calibrate_num_modes, eval_radiance,
    pydisort_riccati_jax,
)
from miejax_lite import table_lookup


# ============================================================================
# 1. Forward / observation operator (thin glue over the seam)
# ============================================================================
class RetrievalForward:
    """Multi-band, multi-angle ToA-reflectance forward model for OE.

    Builds one host-side ``setup`` per band (geometry shared; the per-band BDRF
    and optics table differ), calibrates the DISORT Cauchy mode count ``K`` per
    band at a reference state, and exposes jitted ``forward`` / ``jacobian``
    callables plus the ODE-grid and pool-sensitivity utilities used by the
    retrieval-grid selector.
    """

    def __init__(self, opt_bands, *, NQuad, mu0, I0, phi0, tau_bot, r_base,
                 view_mu, view_phi, BDRF_bands=None, NLeg_all=128, NFourier=8,
                 tol=1e-3, tol_azim=0.0):
        # NLeg_all>=128: a Mie cloud phase function needs ~60+ moments for the
        # NT/TMS single-scatter; 32 gives a Gibbs-oscillating p_full that wrecks
        # thin-cloud (single-scatter-dominated) off-nadir radiance. See
        # docs/OUTSTANDING.md §A′. Cheap: NLeg_all feeds only the TMS quadrature.
        #
        # NFourier fixed small (8), tol_azim=0 ⇒ NO in-loop Cauchy selector
        # (OUTSTANDING §H "Q2"): the relative azimuthal test saturates (K=NFourier)
        # for thin low-signal clouds anyway, while the *absolute* mode amplitudes
        # decay fast after delta-M (NFourier=8 reproduces NFourier=16 to <1%). A
        # fixed small NFourier both bounds the Fourier *unroll* (so jacrev fits in
        # memory — §H) and skips calibrate's wasted full-NFourier solve. Pick
        # NFourier offline from the per-mode amplitudes; do not re-select per solve.
        self.opt_bands = list(opt_bands)
        self.n_bands = len(self.opt_bands)
        self.mu0 = float(mu0)
        self.I0 = float(I0)
        self.tau_bot = float(tau_bot)
        self.r_base = float(r_base)
        self.view_mu = jnp.asarray(view_mu, dtype=float)
        self.view_phi = jnp.asarray(view_phi, dtype=float)
        self.n_view = int(self.view_mu.shape[0])
        self.m = self.n_bands * self.n_view          # observation dimension
        if BDRF_bands is None:
            BDRF_bands = [()] * self.n_bands
        self.setups = [
            riccati_setup(NQuad, I0, phi0, NFourier=NFourier, NLeg_all=NLeg_all,
                          BDRF_Fourier_modes=bdrf, delta_M_scaling=True,
                          NT_cor=True, tol=tol, tol_azim=tol_azim)
            for bdrf in BDRF_bands
        ]
        self.K_list = [s.NFourier for s in self.setups]
        self._fwd_jit = None
        self._jac_jit = None
        self._jac_grid_jit = None

    # -- r_e(τ) interpolant: free nodes + fixed base anchor -------------------
    def _knots_vals(self, x, tau_nodes):
        knots = jnp.concatenate([jnp.asarray(tau_nodes, float),
                                 jnp.asarray([self.tau_bot])])
        vals = jnp.concatenate([jnp.asarray(x, float),
                                jnp.asarray([self.r_base])])
        return knots, vals

    def _re_of_tau(self, tau, knots, vals):
        """The profile parameterisation r_e(τ) from the node values — **the
        function-class lever** (OUTSTANDING §B′; DESIGN §3d).

        This is NOT a post-hoc / cosmetic interpolation: it is *inside* the forward
        map F(x), so it defines what is retrieved and what every solve integrates.
        Plot the result with :meth:`profile`, which routes through here so the
        display mirrors F(x).

        **Default: r_e³-linear (adiabatic).** Since r_e³ ∝ LWC ∝ τ, r_e³ is
        interpolated linearly in τ and cube-rooted. This (i) is the physically
        natural class, (ii) is *coherent with the adiabatic prior* — it represents
        ``x_a`` exactly — and (iii) gets per-segment curvature from just the two
        endpoint values, so it has **no grid-size coupling** (well-defined down to
        one free node + base). It is still C⁰ (kinked at the nodes).

        Swap the class **here and only here** to propagate it through forward /
        calibrate / ODE-grid / Jacobian / re-meshing / display::

            linear (impute-nothing baseline):  jnp.interp(tau, knots, vals)
            PCHIP (C¹, overshoot-free):         needs ≥3 nodes for curvature and
                                                couples to node count — revisit at
                                                higher DOF (e.g. thick cloud); a
                                                node-based C¹ class cannot avoid that
                                                coupling (OUTSTANDING §B′).

        "Which class" is an inverse-problem bias–variance decision (OUTSTANDING §B′),
        bounded above by the integrator order (~C⁶, §1) and far more tightly by DOFS.
        """
        return jnp.interp(tau, knots, vals ** 3) ** (1.0 / 3.0)

    def profile(self, x, tau_nodes, tau):
        """Evaluate the retrieved r_e(τ) **exactly as the forward integrates it**.

        Free nodes ``x`` at ``tau_nodes`` plus the fixed base anchor ``(τ_bot,
        r_base)``, through :meth:`_re_of_tau`. Use this for plotting / downstream so
        the displayed curve mirrors F(x) by construction — the interpolation is part
        of the retrieval, not an independent post-hoc choice.
        """
        knots, vals = self._knots_vals(x, tau_nodes)
        return np.asarray(self._re_of_tau(jnp.asarray(tau, float), knots, vals))

    def _band_reflectance(self, opt, setup, K, knots, vals):
        def om(tau):
            return table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]

        def leg(tau):
            return table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]

        res = riccati_solve(setup, om, leg, self.tau_bot, self.mu0, num_modes=K)
        u = jnp.stack([eval_radiance(setup, res, self.view_mu[i], self.view_phi[i])
                       for i in range(self.n_view)])           # (n_view,)
        return jnp.pi * u / (self.mu0 * self.I0)

    def _forward_raw(self, x, tau_nodes):
        knots, vals = self._knots_vals(x, tau_nodes)
        return jnp.concatenate([
            self._band_reflectance(opt, setup, K, knots, vals)
            for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
        ])                                                     # (n_bands*n_view,)

    # -- calibration: set the per-band Cauchy K at a reference state ----------
    def calibrate(self, x_ref, tau_nodes):
        knots, vals = self._knots_vals(x_ref, tau_nodes)
        Ks = []
        for opt, setup in zip(self.opt_bands, self.setups):
            def om(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]

            def leg(tau, opt=opt):
                return table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]

            Ks.append(calibrate_num_modes(setup, om, leg, self.tau_bot,
                                          self.mu0, self.view_mu, self.view_phi))
        self.K_list = [int(k) for k in Ks]
        # K changed ⇒ invalidate compiled callables (K is static/baked).
        self._fwd_jit = self._jac_jit = self._jac_grid_jit = None
        return self.K_list

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
        """Adaptive Kvaerno5 τ-grid at the given state (first band, offline)."""
        knots, vals = self._knots_vals(x, tau_nodes)
        opt = self.opt_bands[0]
        om = lambda tau: table_lookup(opt, self._re_of_tau(tau, knots, vals))[0]
        leg = lambda tau: table_lookup(opt, self._re_of_tau(tau, knots, vals))[1]
        *_, tau_grid = pydisort_riccati_jax(
            self.tau_bot, om, leg, self.setups[0].NQuad, self.mu0, self.I0,
            self.setups[0].phi0, delta_M_scaling=True, NT_cor=True,
            NLeg_all=self.opt_bands[0]["leg"].shape[-1])
        return np.asarray(tau_grid, float)

    # -- pool sensitivity (parameterise r_e on an arbitrary τ-grid) ----------
    def jacobian_on_grid(self, re_vals, tau_grid):
        """K_pool = ∂y/∂r_e(τ_j) at the pool nodes ``tau_grid`` (m × len).

        ``tau_grid`` is a **traced** argument, so re-selections at a stable ODE-grid
        size reuse the compiled Jacobian (recompile-free lagged re-meshing); a new
        pool size compiles once and caches.
        """
        if self._jac_grid_jit is None:
            def fwd(rv, tg):
                return jnp.concatenate([
                    self._band_reflectance(opt, setup, K, tg, rv)
                    for opt, setup, K in zip(self.opt_bands, self.setups, self.K_list)
                ])
            self._jac_grid_jit = jax.jit(jax.jacrev(fwd, argnums=0))
        return np.asarray(self._jac_grid_jit(jnp.asarray(re_vals, float),
                                             jnp.asarray(tau_grid, float)))


def build_forward(*args, **kw):
    """Functional alias for :class:`RetrievalForward`."""
    return RetrievalForward(*args, **kw)


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
    tau_pool = fwd.ode_grid(x, tau_nodes)
    tau_pool = np.unique(np.clip(tau_pool, 0.0, fwd.tau_bot))
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

    K_pool = fwd.jacobian_on_grid(re_pool, tau_pool)           # (m, P)

    # Column pivoting on the sensitivity matrix: rank τ-nodes by independent info.
    from scipy.linalg import qr
    _, _, piv = qr(K_pool, mode="economic", pivoting=True)

    # Candidate set = interior nodes only (the fixed base anchor τ_bot is appended
    # by the forward, not retrieved). Always include the cloud-top node (τ≈0, most
    # informative). Return EXACTLY k_active nodes so the retrieval-grid size is
    # constant across lagged re-selections ⇒ r-refinement is recompile-free.
    interior = tau_pool < fwd.tau_bot - 1e-6
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


# ============================================================================
# 3. Prior (pluggable): adiabatic mean + Bayesian-Tikhonov covariance
# ============================================================================
def make_adiabatic_prior(tau_nodes, tau_bot, r_base, r_top_prior, *,
                         sigma_top=3.0, sigma_base=10.0, corr_length=None,
                         strength=1.0):
    """Adiabatic first guess ``x_a`` and correlated prior covariance ``S_a``.

    ``x_a`` is the r_e³-linear adiabatic profile (known base → prior top radius)
    sampled at ``tau_nodes``. ``S_a`` is a depth-increasing, exponentially
    correlated Gaussian::

        S_a[i,j] = σ_i σ_j exp(-|τ_i-τ_j| / ℓ),   σ(τ) tight at top, loose at base

    The off-diagonal correlation is the Bayesian-Tikhonov smoothness term; ``ℓ``
    (default τ_bot/2) and ``strength`` (a single σ-scale knob) are the only free
    regularisation levers. Same signature as a future *learned* prior
    (ensemble mean + sample/EOF covariance would slot in here).
    """
    tau_nodes = np.asarray(tau_nodes, float)
    # adiabatic r_e^3-linear: r_e^3 linear in optical depth (LWC ∝ τ).
    frac = 1.0 - tau_nodes / tau_bot                           # 1 at top, 0 base
    x_a = (r_base ** 3 + (r_top_prior ** 3 - r_base ** 3) * frac) ** (1.0 / 3.0)

    if corr_length is None:
        corr_length = max(tau_bot / 2.0, 1e-3)
    # depth-increasing σ: linearly from σ_top (τ=0) to σ_base (τ=τ_bot).
    sigma = strength * (sigma_top + (sigma_base - sigma_top) * tau_nodes / tau_bot)
    dt = np.abs(tau_nodes[:, None] - tau_nodes[None, :])
    Sa = (sigma[:, None] * sigma[None, :]) * np.exp(-dt / corr_length)
    Sa += 1e-9 * np.eye(len(tau_nodes))                        # SPD jitter
    return x_a, Sa


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
        cur_x, cur_nodes = x, tau_nodes
        re_of_tau = lambda t: fwd.profile(cur_x, cur_nodes, t)
        new_tau, new_re, _ = select_retrieval_grid(
            fwd, x, tau_nodes, k_active, re_of_tau=re_of_tau)
        if (len(new_tau) == len(tau_nodes)
                and np.allclose(new_tau, tau_nodes, atol=1e-3)):
            break                                              # grid stabilised
        # re-map state + prior onto the new grid (same parameterisation)
        x = fwd.profile(cur_x, cur_nodes, new_tau)
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


def posterior_diagnostics(K, Sa, Se) -> Posterior:
    """Rodgers posterior covariance, averaging kernels, and DOFS.

    ``Ŝ = (Kᵀ Sε⁻¹ K + Sa⁻¹)⁻¹``;  ``A = Ŝ Kᵀ Sε⁻¹ K``;  ``DOFS = tr(A)``.
    The averaging-kernel rows show *where in τ* each retrieved level draws its
    information (peaks spread through the column ⇔ vertical resolving power);
    DOFS is the number of independent pieces of profile information.
    """
    K = np.asarray(K, float)
    Se_inv = np.linalg.inv(np.asarray(Se, float))
    Sa_inv = np.linalg.inv(np.asarray(Sa, float))
    KtSeK = K.T @ Se_inv @ K
    S_hat = np.linalg.inv(KtSeK + Sa_inv)
    A = S_hat @ KtSeK
    return Posterior(S_hat=S_hat, error=np.sqrt(np.diag(S_hat)), A=A,
                     dofs=float(np.trace(A)))


# ============================================================================
# 6. OSSE helper
# ============================================================================
def osse_observation(fwd: RetrievalForward, tau_truth, re_truth, *, noise=None,
                     seed=0):
    """Synthetic observation ``y = F(x_true)`` from an in-situ truth profile.

    The truth ``r_e(τ)`` is the dense in-situ profile; it is fed to the forward
    model directly (as the node values on its own τ-grid, base anchor fixed at
    the truth base). Noiseless by default (the OSSE decision); pass ``noise`` (a
    per-observation σ vector or scalar) to add a Gaussian realization.
    """
    tau_truth = np.asarray(tau_truth, float)
    re_truth = np.asarray(re_truth, float)
    # interior truth nodes (exclude the base anchor, which fwd appends)
    interior = tau_truth < fwd.tau_bot - 1e-9
    y = np.asarray(fwd.forward(re_truth[interior], tau_truth[interior]), float)
    if noise is not None:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, 1.0, size=y.shape) * np.asarray(noise)
    return y
