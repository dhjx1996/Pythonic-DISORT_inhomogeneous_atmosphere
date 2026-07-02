# Comprehensive technical documentation — `pydisort_riccati_jax`

*The scientist's and mathematician's companion: the governing equations, the
invariant-imbedding Riccati formulation and its stability guarantee, the numerical
integration and differentiation machinery, the Mie optics, and the full
optimal-estimation retrieval + information-content methodology built on top. It
replaces the LaTeX report `report_riccati_solver.tex` (retired 2026-07-02; in git
history) — condensing its solver derivations and superseding its pre-retrieval-era
state — in the spirit of PythonicDISORT's "Comprehensive Documentation" of
DISORT. Markdown was chosen over LaTeX deliberately: it renders where the code
lives, diffs with it, and keeps equations one edit away from the implementation
they describe. For runnable examples see the [user guide](user_guide.md); for
per-knob evidence see the [hyperparameter audit](hyperparameter_audit_2026-07.md);
for the decision log see `DESIGN_DECISIONS.md`.*

Notation: $N_q$ = `NQuad` streams, $N = N_q/2$ per hemisphere; $\mu_i, w_i$ =
double-Gauss quadrature nodes/weights on $(0,1]$; $\tau \in [0, \tau_b]$ optical
depth (0 = ToA); $\omega(\tau)$ single-scattering albedo; $\chi_\ell(\tau)$
Legendre phase-function moments ($\chi_0 = 1$, $\chi_1 = g$); $\mu_0, I_0, \phi_0$
the collimated beam.

---

## 1. Problem statement

### 1.1 The radiative transfer equation

Monochromatic, unpolarized, plane-parallel scattering with a collimated beam:

$$\mu \frac{\partial u(\tau,\mu,\phi)}{\partial \tau} = u(\tau,\mu,\phi)
 - \frac{\omega(\tau)}{4\pi}\!\int_0^{2\pi}\!\!\int_{-1}^{1} p(\tau;\mu,\phi;\mu',\phi')\,
   u(\tau,\mu',\phi')\,d\mu' d\phi'
 - \frac{\omega(\tau)\,I_0}{4\pi}\, p(\tau;\mu,\phi;-\mu_0,\phi_0)\, e^{-\tau/\mu_0}.$$

What distinguishes this solver from DISORT/PythonicDISORT is the **continuously
τ-varying** $\omega(\tau)$ and $p(\tau;\cdot)$ — supplied as *functions*, not
per-layer constants — and that only the **upwelling field at ToA** is produced (the
retrieval observable). For piecewise-constant optics, eigendecomposition solvers
are exact and faster; use them there.

### 1.2 Fourier decomposition and discrete ordinates

Expanding $u = \sum_{m=0}^{M-1} u^m(\tau,\mu)\cos m(\phi_0-\phi)$ with the addition
theorem decouples the azimuth into `NFourier` independent modes. Discretizing each
mode on the $2N$ quadrature streams gives, per mode, a linear $2N$-system for
$\mathbf u = (\mathbf u^+, \mathbf u^-)$ (up-/downwelling at the positive nodes):

$$\frac{d}{d\tau}\begin{pmatrix}\mathbf u^+\\ \mathbf u^-\end{pmatrix}
 = \underbrace{\begin{pmatrix} -\boldsymbol\alpha(\tau) & -\boldsymbol\beta(\tau)\\
 \boldsymbol\beta(\tau) & \boldsymbol\alpha(\tau)\end{pmatrix}}_{\mathbf A(\tau)}
 \begin{pmatrix}\mathbf u^+\\ \mathbf u^-\end{pmatrix}
 + \begin{pmatrix} -\tilde{\mathbf Q}^+(\tau)\\ \tilde{\mathbf Q}^-(\tau)\end{pmatrix},
 \qquad
 \begin{aligned}
 \boldsymbol\alpha &= \mathbf M^{-1}\!\left(\omega\,\mathbf D^{+}\mathbf W - \mathbf I\right),\\
 \boldsymbol\beta  &= \mathbf M^{-1}\,\omega\,\mathbf D^{-}\mathbf W,
 \end{aligned}$$

with $\mathbf M = \mathrm{diag}(\mu_i)$, $\mathbf W = \mathrm{diag}(w_i)$, and
$\mathbf D^{\pm}$ the same-/opposite-hemisphere phase matrices assembled from the
$\chi_\ell$ via associated Legendre products (precomputed once with
`scipy.special`, contracted by `einsum` in the traced vector field — the τ-dependence
enters only through $\omega(\tau)$ and $\chi_\ell(\tau)$). $\tilde{\mathbf Q}^\pm
\propto \omega\,e^{-\tau/\mu_0}$ is the beam source. All modes share this structure
(mode-index-free builders on padded tensors), so the per-mode solve compiles **once**
and runs under `lax.scan` (CPU) or `vmap` (GPU) over modes.

## 2. The invariant-imbedding Riccati formulation

Solving the two-point boundary-value problem for $\mathbf A(\tau)$ by
eigendecomposition is unavailable (no layers) and by shooting is unstable: $\mathbf
A$ has growing modes $e^{+\lambda\tau}$ that overflow for thick clouds. Invariant
imbedding removes them **by construction**.

### 2.1 Scattering operators

For a slab $[\tau_a,\tau_b]$ define the four $N\times N$ operators and two source
vectors of its diffuse input–output map:

$$\mathbf I^+(\tau_a) = \mathbf R_{\rm up}\,\mathbf I^-(\tau_a) + \mathbf T_{\rm up}\,\mathbf I^+(\tau_b) + \mathbf s_{\rm up},
\qquad
\mathbf I^-(\tau_b) = \mathbf T_{\rm down}\,\mathbf I^-(\tau_a) + \mathbf R_{\rm down}\,\mathbf I^+(\tau_b) + \mathbf s_{\rm down}.$$

These are physical (bounded, non-negative) quantities regardless of thickness.

### 2.2 The Riccati system

Grow the slab from the bottom with thickness variable $\sigma$ (top of the current
slab at $\tau = \tau_b - \sigma$). Adding an infinitesimal layer — whose own
operators are $r = \boldsymbol\beta\,d\sigma$, $t = \mathbf I +
\boldsymbol\alpha\,d\sigma$ — via the Redheffer star product and keeping the
first-order multiple-reflection term $\mathbf R\boldsymbol\beta\mathbf R$ yields

$$\boxed{\;\frac{d\mathbf R}{d\sigma} = \boldsymbol\alpha\mathbf R + \mathbf R\boldsymbol\alpha
 + \mathbf R\boldsymbol\beta\mathbf R + \boldsymbol\beta,\qquad \mathbf R(0)=\mathbf 0\;}$$

with companion linear ODEs sharing the same integration:

$$\frac{d\mathbf T}{d\sigma} = (\boldsymbol\alpha + \mathbf R\boldsymbol\beta)\,\mathbf T,\quad \mathbf T(0)=\mathbf I;
\qquad
\frac{d\mathbf s}{d\sigma} = (\boldsymbol\alpha + \mathbf R\boldsymbol\beta)\,\mathbf s
 + \mathbf R\,\mathbf q_1 + \mathbf q_2,\quad \mathbf s(0)=\mathbf 0,$$

where $\mathbf q_{1,2}$ are the beam-source injections at the current top. A
**forward sweep** ($\sigma: 0\to\tau_b$, coefficients evaluated at
$\tau=\tau_b-\sigma$) builds $(\mathbf R_{\rm up},\mathbf T_{\rm up},\mathbf s_{\rm
up})$; a mirrored **backward sweep** builds $(\mathbf R_{\rm down},\mathbf T_{\rm
down},\mathbf s_{\rm down})$. The integration state is the PyTree
$\{\mathbf R\,(N{\times}N), \mathbf T\,(N{\times}N), \mathbf s\,(N)\}$ per mode.

### 2.3 The stability invariant — NO POSITIVE EXPONENTS

Every term on the right of the Riccati equation is non-negative ($\boldsymbol\beta
\propto \omega\mathbf M^{-1}\mathbf D^-\mathbf W \ge 0$; $\mathbf R(0)=0$), and
$\mathbf R$ is a physical reflectance: it grows monotonically toward its
semi-infinite-cloud limit and **stays O(1) for any thickness**. No intermediate
quantity of the algorithm contains $e^{+\lambda\tau}$, $\lambda>0$ — the failure
mode of shooting/transfer-matrix methods on thick clouds. *This is the repository's
hard invariant: any algorithmic change must preserve it* (`DESIGN_DECISIONS.md` §2).
The price is nonlinear stiffness near the diffusion regime, addressed by the
integrator choice (§5).

## 3. Boundary conditions and the observable

The surface (Lambertian albedo or general BDRF Fourier modes, plus optional bottom
diffuse source $\mathbf b^+$) supplies a reflection relation at $\tau_b$; combined
with the two sweeps' operators, a single $N\times N$ linear solve per mode yields
the upwelling field at ToA:

$$\mathbf u^{+,m}(0) \;=\; \big[\text{star-product of } (\mathbf R,\mathbf T,\mathbf s)_{\rm up/down} \text{ with the surface}\big]$$

(implementation: `_solve_bc_riccati_jax`, one `jnp.linalg.solve`). The azimuthal
series is then resummed, $u(0,\mu_i,\phi) = \sum_m u^{+,m}(\mu_i)\cos m(\phi_0-\phi)$,
and evaluated at arbitrary view zeniths by **barycentric interpolation** across the
quadrature nodes ($O(N)$, differentiable). The retrieval observable is this full
azimuthally-resolved $u(0,\mu,\phi)$ — not just flux — and the upward flux
$F^\uparrow(0) = 2\pi\sum_i \mu_i w_i u_0^{+}(\mu_i)$ uses the $m{=}0$ mode only
(exactly: higher modes integrate to zero over azimuth).

## 4. Delta-M scaling and the Nakajima–Tanaka (TMS) correction

Mie phase functions of cloud droplets are extremely forward-peaked (size parameter
$x = 2\pi r/\lambda$ up to several hundred). Truncating at $N_q$ streams/moments
makes the reconstructed phase function ring (Gibbs) and the computed radiance can go
**negative**. Two standard remedies are implemented (opt-in flags, ON in
production):

- **Delta-M** (Wiscombe 1977): split off a forward Dirac fraction
  $f(\tau) = \chi_{N_{\rm Leg}}(\tau)$ (the first dropped moment — "f-as-data") and
  rescale in **physical τ form**: the solve uses
  $d\tilde\tau = (1 - \omega f)\,d\tau$, $\tilde\omega = \omega(1-f)/(1-\omega f)$,
  $\tilde\chi_\ell = (\chi_\ell - f)/(1 - f)$. Fluxes are then accurate; radiances
  still carry truncation error.
- **TMS** (Nakajima & Tanaka 1988): replace the *singly-scattered* part of the
  delta-M radiance with the exact single scatter computed from the **full**
  $N_{\rm Leg,all}$-moment phase function at each view angle. Implemented for the
  upwelling field; the companion IMS correction is omitted **by design** — it
  corrects the downward field near the aureole, which this solver never outputs
  (LIDORT/VLIDORT make the same choice).

**Moment-count requirement.** TMS is only as good as the reconstructed full phase
function: it must be positive and converged at the largest size parameter present —
including the *size-distribution tail* (the gamma average integrates radii to
$3r_e$). Empirically the reconstruction turns positive at $\approx 1.1\times$ the
max size parameter. The production point ($r_e \le 20\,\mu$m, $\lambda \ge
0.55\,\mu$m, $x_{\max}\approx 685$) uses $N_{\rm Leg,all} = 1536$ with Gauss–Legendre
projection order $n_{\rm GL} = 4096 \ (\ge 2.35\,N_{\rm Leg,all}$). The historical
failure — 128 moments ringing the 0.55 µm bands negative and contaminating a whole
analysis — is the motivating burn; treat $N_{\rm Leg,all}$ as a *correctness*
parameter, not a tuning knob.

## 5. Numerical integration

The Riccati flow is integrated with **diffrax Kvaerno5/4** — an L-stable, adaptive
embedded ESDIRK pair. Why this integrator:

- **L-stability**: the flow is stiff where the diffusion limit is approached
  (thick, conservative); A-stability alone rings, explicit methods need
  $O(\tau_b/\epsilon)$ steps.
- **Adaptivity as information**: the accepted-step grid concentrates where the
  optics change — it is reused as the *candidate pool* for retrieval-node selection
  (§8.4), an unusual dual use.
- **Step-count economy**: the forward runs inside a Jacobian inside a retrieval
  loop, so accepted-step count dominates cost. It is nearly $N_q$-independent
  (~35 steps for a τ=30 cloud) — stream count buys angular resolution nearly free
  of ODE cost.

Tolerances: `tol` sets `rtol` with `atol = tol·1e-3`; in float32 `rtol` is floored
at $10^{-3}$ (asking a step controller to resolve below unit roundoff makes it
shrink $dt$ forever into the `max_steps=4096` guard). **Precision policy**: float32
(`tol≈1e-3`) is the exploratory default; the production observing system (10 bands,
$N_q=48$) is **float64 + `tol=1e-4`**, the probe-settled point where thin and thick
profiles both converge and a tighter decade changes fit quality not at all (only
cost). At $\tau \gtrsim 36$ the retrieved *profile* (not the fit) becomes
tol-sensitive — a conditioning signal at extreme depth, documented, not "fixed" by
tightening globally.

Two independent adaptive solves of the same problem (e.g. the sequential band loop
vs the GPU-batched vmap, or a plain forward vs the augmented Jacobian pass) agree to
$O(\text{tol})$, **not bitwise** — each is an equally valid tol-level solution.
Equivalence gates are therefore written against tol-scaled bounds (float32) or run
at float64 where the bound tightens by orders of magnitude.

## 6. Differentiability

The solver is differentiable end-to-end in JAX:

- **Reverse mode is the discrete adjoint.** `jax.grad` through the solve
  differentiates the actually-executed discrete algorithm (diffrax
  `RecursiveCheckpointAdjoint` by default) — there is no hand-derived adjoint ODE,
  and none is wanted: a continuous adjoint re-integrates *backward* through the
  same stiffness and re-introduces exactly the instability the imbedding removed.
  The discrete adjoint inherits the no-positive-exponents property.
- **Forward mode for retrievals.** With state dimension $p\sim 7 \ll m$
  observations, `jax.jacfwd` (p tangent solves, view-count-independent) beats
  reverse; it requires `riccati_setup(..., adjoint=diffrax.ForwardMode())`, which
  propagates tangents jointly with the primal in one augmented adaptive solve.
- **The seam.** `riccati_setup` (host-side; static geometry, quadrature, mode
  tensors, μ0) / `riccati_solve` (traced: $\tau_b$ and the optics closures) /
  `eval_radiance` (traced observable). Fourier modes run under `lax.scan`, so the
  mode body compiles once — O(1) compile memory in mode count. μ0 static means one
  (cheap) compile per solar geometry; operational per-scene work should bin μ0.
- Optics tables enter the jitted callables as **traced arguments** (stacked per
  band), never closure constants — constants bake multi-MB tables into every HLO,
  bloat compiles, and defeat compile caching.

## 7. Mie optics (`optics_table`)

The retrieval needs $r_e \mapsto (\omega, \chi_\ell, Q_{\rm ext})$ differentiable
and cheap inside the ODE hot loop. Exact Mie inside the trace is hopeless
(compile-time explosion); the production representation is a **precomputed table
with differentiable linear interpolation**:

1. Per band and per $r_e$ grid point (uniform, `n_re` points): Mie coefficients
   $a_n, b_n$ (miepython, Wiscombe order for the largest tail size parameter),
   amplitude functions on a GL angular grid, phase function
   $P(\mu) = 2(|S_1|^2{+}|S_2|^2)/(x^2 Q_{\rm sca})$, moments
   $\chi_\ell = \tfrac12\int P\,P_\ell\,d\mu$ by GL quadrature.
2. **Gamma averaging** over the droplet size distribution (Hansen–Travis modified
   gamma, $n(r)\propto r^{1/v_e-3}e^{-(1/v_e-3+3)r/r_e}$, effective variance
   $v_e = 0.10$ fixed): $Q_{\rm sca} r^2 n(r)$-weighted moments on a $3r_e$ radius
   grid. Averaging also smooths the Mie ripple that makes point-wise Mie Jacobians
   noisy.
3. `table_lookup(opt, r_e)`: linear interpolation in $r_e$; gradient = table slope;
   out-of-range clamps (zero gradient — pair with state bounds, §8.2). The solve
   interpolates only the $N_{\rm Leg}{+}1$ moments it needs; the TMS path pulls all
   $N_{\rm Leg,all}$.

Tables are profile-independent, built once, disk-cached with a provenance
signature, and shared by every worker.

## 8. The optimal-estimation retrieval (`retrieval_oe`)

### 8.1 Estimator

Rodgers MAP/OE: minimize

$$J(x) = \tfrac12\,\big(y - F(x)\big)^{\!\top} S_\epsilon^{-1} \big(y - F(x)\big)
 + \tfrac12\,(x - x_a)^{\!\top} S_a^{-1} (x - x_a)$$

by damped Gauss–Newton (Levenberg–Marquardt):

$$\delta x = \big(K^\top S_\epsilon^{-1} K + (1+\lambda)\,S_a^{-1}\big)^{-1}
 \big(K^\top S_\epsilon^{-1}(y - F(x)) - S_a^{-1}(x - x_a)\big),\qquad K = \partial F/\partial x,$$

with **accept-only-if-J-decreases**: a rejected step raises λ (×4) and retries, an
accepted one eases it (×0.5). Monotone descent makes the returned iterate the best
found and makes the stopping tests meaningful (a plain-GN oscillation around the
noiseless-OSSE's flat minimum was the observed failure that motivated LM).
Stopping (BP2026): no-further-decrease; data-misfit stagnation
$(\varphi_{k}-\varphi_{k+1})/\varphi_k <$ `cost_rtol`; step-norm; iteration cap.
Iterates are projected onto physical/table bounds between steps.

### 8.2 State parameterization

$x = [\,r_e(s_1..s_k),\ r_{\rm base},\ \tau_b\,]$ in **natural log** (positivity
automatic; BP2026 report log-state as decisive for GN convergence; the prior maps by
the delta method: $\ln x_a,\ D S_a D^\top$, $D = \mathrm{diag}(1/x_a)$).
The $r_e$ nodes live at **normalized depth** $s = \tau/\tau_b \in [0,1)$ with the
base anchored at $s=1$: retrieving $\tau_b$ stretches the node positions with it,
so nodes can never cross the cloud base — the failure that divergence-prone
absolute-τ grids hit. Between nodes the profile follows the **adiabatic class**,
$r_e^5$ linear in $s$: from $r_e^3 \propto {\rm LWC} \propto z$ and $d\tau \propto
r_e^2\,dz$ follows $\tau \propto z^{5/3}$, i.e. $r_e \propto \tau^{1/5}$ — and a
linear rescale $s = \tau/\tau_b$ leaves the class invariant. The same interpolant
is *inside* $F(x)$ and in every display (`fwd.profile`), so plots show exactly what
the forward integrated.

### 8.3 Priors — tight where blind, loose where strong

The prior sensitivity study inverted the naive instinct: the measurement pins the
cloud **top** ($A_{\rm top}\approx 1$) and **τ_b** (conservative VIS bands), while
the **base** is radiatively shielded (~80 % prior-dominated) — so the prior must be
*loose* on top/τ_b and *tight and adiabatically coupled* on the base
($r_{\rm base} = 0.65\,r_{\rm top}$, clipped $< r_{\rm top}$; VOCALS median 0.60,
literature ≈0.70). The $r_e$ block (base included, at $s{=}1$) is one correlated
adiabatic-mean Gaussian with depth-increasing σ and exponential correlation;
$\tau_b$ is an independent broad scalar (~100 % relative). Campaign work uses the
**leave-one-flight-out** VOCALS climatology (means/σ from the ensemble *excluding
the target's flight* — the OSSE leak-free discipline).

### 8.4 Retrieval-grid and mode selection

- **Node pool = the ODE grid.** The adaptive accepted steps of a solve at the
  current state (normalized by $\tau_b$) form a trustworthy superset of the
  informative depths. The pool Jacobian $K_{\rm pool} = \partial y/\partial
  r_e(s_j)$ is one `jacfwd`.
- **How many nodes:** whiten $\tilde K = S_\epsilon^{-1/2} K_{\rm pool}\,
  \mathrm{diag}(\sigma_{\rm prior})$; QRCP gives ordered marginal informations
  $r_1 \ge r_2 \ge\dots$; the Rodgers filter factor $f_i = r_i^2/(1+r_i^2)$ is the
  data fraction of direction $i$; keep the $f_i \ge 0.5$ (data ties prior — the
  SNR=1 crossover, noise-level-invariant) plus a margin of one. — **Where:** QRCP
  column pivoting on $K_{\rm pool}$, always retaining cloud top.
- **Azimuthal mode count:** per band, the smallest $K$ such that every mode $m \ge
  K$ contributes less than $\tfrac13\min\sigma_\epsilon$ reflectance at every view
  (truncation is a *runtime* optimization judged against the noise, not a
  convergence test against the signal). On the GPU path $K$ is padded uniform
  across bands — ragged $K$ would break the bands×modes SIMT batch for a sub-noise
  saving that only sequential (CPU) execution realizes.

### 8.5 Posterior diagnostics

$$\hat S = (K^\top S_\epsilon^{-1}K + S_a^{-1})^{-1},\qquad
 A = \hat S\,K^\top S_\epsilon^{-1}K,\qquad
 \mathrm{DOFS} = \mathrm{tr}\,A,\qquad
 \mathrm{SIC} = \tfrac12\log_2 \frac{|S_a|}{|\hat S|}\ \text{[bits]},$$

plus the per-node data fraction $1 - \hat S_{ii}/S_{a,ii}$. DOFS counts the
independent directions the measurement constrains; SIC weighs how *sharply* (a
direction reduced 100× adds more bits than one reduced 2×) — thin clouds show few
DOFS but high SIC per DOF, thick the reverse; report both.

## 9. Information-content methodology

The IC question ("what can the measurement resolve?") is answered on the **full ODE
grid**, independent of any retrieval-grid choice, with the same Rodgers algebra
(single implementation). The basis-free view is the whitened spectrum: one SVD of
$\tilde K = S_\epsilon^{-1/2} K\, S_a^{1/2}$ gives singular values $s_i$ (SNR
units), $f_i = s_i^2/(1+s_i^2)$, $\mathrm{DOFS} = \sum f_i$, $\mathrm{SIC} =
\tfrac12\sum\log_2(1+s_i^2)$. Two observables are profiled: the multi-angle
radiance Jacobian, and the angle-integrated **plane-albedo (flux-reflectance)
Jacobian** — the $m{=}0$-exact spectral quantity of the King–Vaughan/CPV2012
literature, which serves as the spectral-IC baseline against which the *angular*
information novelty is measured. Design choices that matter to the numbers: the
irregular (golden-ratio) view fan (a regular μ-grid aliases against the
conservative-band angular Jacobian), the 2 % calibration-relative OCI noise model
(the information scale is set entirely by the assumed $S_\epsilon$; a "noiseless"
OSSE still assumes one), and μ0 = 0.9 fixed (all published numbers are conditional
on it).

## 10. The OSSE pipeline and its rigor gates

Three batches over the 125 VOCALS-REx in-situ profiles:
**rad** (truth radiances $y = F(x_{\rm truth})$, float64/`tol=1e-4`, cached once)
→ **ic** (Jacobians at prior mean / climatology draws; mechanism records)
→ **fr** (the joint retrieval, two prior configs A/B sharing one compiled forward).
Discipline that makes the results defensible:

- **Signature gating.** `osse_config.signature()` hashes what $y$ *means* (bands,
  views, $N_q$, moments, delta-M/TMS flags…); every cache asserts it. `tol` is
  deliberately *outside* the signature (it is how accurately $y$ was computed, and
  legitimately differs between the truth tier and an operational forward) and is
  carried as an asserted accuracy tag instead.
- **Leak-freedom.** Priors/first guesses only ever see leave-one-flight-out
  statistics; the truth enters solely as the synthetic world's definition.
- **Resumability.** L1 = per-GN-iteration checkpoint (atomic, plain numpy,
  CPU↔GPU portable); L2 = per-profile setup cache (mode counts, grid, τ_b
  pre-retrieval — deterministic and platform-portable by design; key includes the
  observing-system signature, the profile index, and a code-semantics version).
  A persistent XLA compile cache (the would-be L3) was measured a no-op for the
  execution-bound retrieval and removed.
- **Gates** (`tests/hpc/`, opt-in): L1 resume-equivalence, L2 hit-vs-compute
  equivalence (including the assert that the cache was actually *written* — a
  write-path bug once degraded this gate to a vacuous pass), and the **3-profile
  golden cross-check** on platform-invariant retrieved quantities — the mandatory
  sign-off after any numerics-adjacent refactor.

## 11. Verification ladder

| Tier | What | Where |
|---|---|---|
| CI, every push | float32: solver vs pydisort references (Stamnes cases, BDRF, thick, τ-varying convergence O(h²), adaptive grid, μ-interp, adjoint smoke, delta-M/TMS positivity+regression, jit seam parity) + the retrieval stack (path equivalences at tol bounds, jacfwd↔jacrev, priors, posterior algebra, noiseless truth recovery, L1 resume, τ_b pre-retrieval, oracle floor) | `tests/*_test.py` |
| Weekly / on demand | float64 partition: tight-tolerance convergence, FD-vs-AD gradients, stringent delta-M benchmarks, the float64+`tol=1e-4` production-precision retrieval | `-m float64` |
| Per HPC campaign | production-scale gates + golden cross-check | `tests/hpc/`, `-m hpc` |

The reference implementation throughout is PythonicDISORT's `pydisort` (exact
eigendecomposition), wrapped apples-to-apples (same delta-M $f = \chi_{N_{\rm
Leg}}$ convention) in `pydisort_riccati_jax.reference`; τ-varying cases are checked
by $O(h^2)$ Richardson-style convergence of piecewise-constant approximations
toward the Riccati solution.

## 12. References

- Stamnes, K., S.-C. Tsay, W. Wiscombe, K. Jayaweera (1988): *Numerically stable
  algorithm for discrete-ordinate-method radiative transfer…* (DISORT). Appl. Opt. 27.
- Ho, D. J. X.: *PythonicDISORT* — https://pythonic-disort.readthedocs.io (the
  reference solver and the model for this documentation's two levels).
- Wiscombe, W. (1977): *The delta-M method…* J. Atmos. Sci. 34.
- Nakajima, T., M. Tanaka (1988): *Algorithms for radiative intensity calculations
  in moderately thick atmospheres…* (TMS/IMS). JQSRT 40.
- Nakajima, T., M. D. King (1990): *Determination of the optical thickness and
  effective particle radius…* (the bispectral method). J. Atmos. Sci. 47.
- Rodgers, C. D. (2000): *Inverse Methods for Atmospheric Sounding.* World Scientific.
- Coddington, O., P. Pilewskie, T. Vukicevic (2012): spectral cloud information
  content (CPV2012; the spectral-IC baseline this work's angular analysis extends).
- King, N., M. Vaughan (2012) (KV2012); Kokhanovsky & Rozanov (KR2012); Platnick
  (2000, Pla2000): vertical-profile information in cloud remote sensing.
- BP2025 / BP2026 (preprint): profile retrievals with log-state Gauss–Newton and
  cost-stagnation convergence — the retrieval-design source.
- Hansen, J. E., L. D. Travis (1974): *Light scattering in planetary atmospheres*
  (gamma size distribution).
- Wood, R., et al. (2011): VOCALS-REx overview. ACP 11.
- Kværnø, A. (2004): *Singly diagonally implicit Runge–Kutta methods with an
  explicit first stage* (the Kvaerno5/4 pair). BIT 44.
- Segelstein, D. (1981): the complex refractive index of water (M.S. thesis; the
  `src/pydisort_riccati_jax/data` table).

*(Local copies of the campaign-relevant papers are in `../literature/`.)*
