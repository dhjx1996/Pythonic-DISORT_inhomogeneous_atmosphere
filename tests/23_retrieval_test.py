"""Retrieval-stack coverage: RetrievalForward, mode/grid selection, priors,
posterior diagnostics, the Gauss-Newton OE loop, the L1 checkpoint, and the
tau_bot pre-retrieval — on a SMALL synthetic configuration (2 bands, NQuad=8,
tiny miepython optics table) so the whole file runs on CPU in the default
float32 partition.

This is the first collected coverage of the OE layer (previously exercised only
by the non-collected HPC workers). The synthetic truth lives exactly in the
forward's re5-linear class, so the noiseless GN retrieval must recover it to
within float32 forward noise -- a functional end-to-end gate, not a golden file.
"""
import os

import numpy as np
import pytest

from pydisort_riccati_jax import retrieval_oe as roe
from pydisort_riccati_jax import optics_table as ot

# --- small, deterministic configuration -------------------------------------
# Size-parameter budget: the gamma tail integrates radii to 3*re_max = 30 um, so
# the largest size parameter is x = 2*pi*30/1.64 ~ 115. A positive TMS phase-
# function reconstruction needs ~1.1x moments (the production NLEG_ALL=1536
# lesson at x~685) -> NLEG_ALL=160 keeps this toy PHYSICAL (positive radiances),
# unlike a too-small moment count which silently poisons the whole file.
BANDS = [1.64, 2.13]          # one weakly + one strongly absorbing SWIR band
NQUAD = 8
NLEG_ALL = 160
V_EFF = 0.10
RE_BOUNDS = (4.0, 10.0)
N_RE = 16
TAU_BOT = 5.0
R_TOP, R_BASE = 9.0, 6.5
S_NODES = np.array([0.0, 0.5])          # free r_e nodes (cloud top + mid)
VIEW_MU = np.linspace(0.9, 0.3, 4)      # 4 views >= NQuad//2
VIEW_PHI = np.full(4, np.pi)
SIGMA_EPS = 0.005                        # flat observation sigma (reflectance)


@pytest.fixture(scope="module")
def opt_bands():
    table = ot.build_re_table(BANDS, RE_BOUNDS[0], RE_BOUNDS[1], N_RE, V_EFF,
                              n_radii=80, NLeg=NLEG_ALL, n_gl=384)
    return [ot.select_channel(table, i) for i in range(len(BANDS))]


def make_fwd(opt_bands, mode_map="scan", **kw):
    kwargs = dict(NQuad=NQUAD, mu0=0.9, I0=1.0, phi0=0.0,
                  tau_bot=TAU_BOT, r_base=R_BASE,
                  view_mu=VIEW_MU, view_phi=VIEW_PHI,
                  BDRF_bands=[[0.06]] * len(BANDS),
                  NLeg_all=NLEG_ALL, NFourier=NQUAD, tol=1e-3,
                  re_class="re5-linear", state_space="log", jac_mode="fwd",
                  retrieve_tau_bot=True, retrieve_r_base=True,
                  re_bounds=RE_BOUNDS, mode_map=mode_map)
    kwargs.update(kw)
    return roe.RetrievalForward(opt_bands, **kwargs)


def truth_state():
    """Physical truth exactly in the re5-linear class."""
    re_nodes = (R_BASE ** 5 + (R_TOP ** 5 - R_BASE ** 5) * (1.0 - S_NODES)) ** 0.2
    return np.concatenate([re_nodes, [R_BASE], [TAU_BOT]])


@pytest.fixture(scope="module")
def fwd(opt_bands):
    return make_fwd(opt_bands)


@pytest.fixture(scope="module")
def y_truth(fwd):
    x_true = fwd._encode_state(truth_state())
    return np.asarray(fwd.forward(x_true, S_NODES), float)


def _Se(m):
    return (SIGMA_EPS ** 2) * np.eye(m)


# --- 23a: forward basics + scan-vs-vmap path equality ------------------------
def test_23a_forward_paths_agree(opt_bands, fwd, y_truth):
    assert y_truth.shape == (len(BANDS) * VIEW_MU.size,)
    assert np.all(np.isfinite(y_truth)) and np.all(y_truth > 0.0)

    # vmap maps the same modes; with uniform K the bands are batched in ONE vmap.
    # The two paths are equally valid ADAPTIVE solves at tol=1e-3 (floored in
    # float32): their step sequences may differ, so each is within ~tol of truth
    # and their difference is bounded by ~2*tol — NOT bit-identical on CPU/float32.
    # (The HPC golden gate re-checks this at float64/tol=1e-4 where it tightens.)
    fwd_v = make_fwd(opt_bands, mode_map="vmap")
    x_true = fwd._encode_state(truth_state())
    y_v = np.asarray(fwd_v.forward(x_true, S_NODES), float)
    np.testing.assert_allclose(y_v, y_truth, rtol=3e-3,
                               atol=3e-3 * float(np.max(np.abs(y_truth))))


# --- 23b: jacobian consistency (forward vs reverse AD) -----------------------
def test_23b_jacfwd_vs_jacrev(opt_bands, fwd):
    x_true = fwd._encode_state(truth_state())
    K_fwd = np.asarray(fwd.jacobian(x_true, S_NODES), float)
    p = len(S_NODES) + 2
    assert K_fwd.shape == (fwd.m, p)
    assert np.all(np.isfinite(K_fwd))

    fwd_rev = make_fwd(opt_bands, jac_mode="rev")
    K_rev = np.asarray(fwd_rev.jacobian(x_true, S_NODES), float)
    np.testing.assert_allclose(K_fwd, K_rev, rtol=2e-3, atol=2e-6)


# --- 23c: pool Jacobian path equality (scan vs vmap forwards) -----------------
def test_23c_jacobian_on_grid_paths_agree(opt_bands, fwd):
    s_pool = np.array([0.0, 0.25, 0.5, 0.75])
    re_pool = truth_state()[:2].mean() * np.ones(4)
    K_scan = fwd.jacobian_on_grid(re_pool, s_pool, TAU_BOT)
    assert K_scan.shape == (fwd.m, 4)

    fwd_v = make_fwd(opt_bands, mode_map="vmap")
    K_vmap = fwd_v.jacobian_on_grid(re_pool, s_pool, TAU_BOT)
    # float32 path-noise on the Jacobian is absolute (differencing amplifies the
    # ~1e-4-relative radiance noise on small elements) — scale atol to max|K|.
    np.testing.assert_allclose(K_vmap, K_scan, rtol=5e-3,
                               atol=1e-3 * float(np.max(np.abs(K_scan))))


# --- 23d: noise-aware mode selection ------------------------------------------
def test_23d_select_num_modes(opt_bands, y_truth):
    fwd2 = make_fwd(opt_bands)
    x_ref = fwd2._encode_state(truth_state())
    Se = _Se(fwd2.m)
    K_list = roe.select_num_modes(fwd2, x_ref, S_NODES, Se)
    assert all(1 <= K <= NQUAD for K in K_list)
    # dropped modes are each < (1/3)*min(sigma); total truncation stays well under noise
    y_trim = np.asarray(fwd2.forward(x_ref, S_NODES), float)
    assert np.max(np.abs(y_trim - y_truth)) < (NQUAD * SIGMA_EPS / 3.0)

    # (E1) on the vmap path the K_list is padded UNIFORM so the bands×modes batch
    # stays alive (a ragged K silently degraded production FR to a band loop)
    fwd_v = make_fwd(opt_bands, mode_map="vmap")
    K_v = roe.select_num_modes(fwd_v, x_ref, S_NODES, Se)
    assert len(set(K_v)) == 1 and K_v[0] == max(K_list)
    assert fwd_v._can_batch_bands


# --- 23e: priors ---------------------------------------------------------------
def test_23e_priors():
    x_a, Sa = roe.make_marine_sc_prior(S_NODES, r_top_prior=9.7, tau_bot_prior=6.0)
    p = len(S_NODES) + 2
    assert x_a.shape == (p,) and Sa.shape == (p, p)
    r_base_prior = x_a[len(S_NODES)]
    assert r_base_prior < 9.7                          # adiabatic bound r_base < r_top
    np.testing.assert_allclose(Sa, Sa.T)
    assert np.all(np.linalg.eigvalsh(Sa) > 0)          # SPD

    # log transform: exact mean map, delta-method covariance
    x_log, Sa_log = roe.to_log_prior(x_a, Sa)
    np.testing.assert_allclose(x_log, np.log(x_a))
    D = np.diag(1.0 / x_a)
    np.testing.assert_allclose(Sa_log, D @ Sa @ D.T)

    clim = dict(r_top_mean=9.7, r_top_std=2.3, r_base_mean=6.5, r_base_std=1.4,
                tau_bot_mean=6.0, tau_bot_std=5.0)
    x_c, Sa_c = roe.make_climatology_prior(S_NODES, clim, log=True)
    assert x_c.shape == (p,) and np.all(np.isfinite(Sa_c))

    # a climatology realization is a genuine adiabatic profile within bounds
    rng = np.random.default_rng(0)
    draw, info = roe.draw_climatology_realization(clim, S_NODES, rng=rng,
                                                  tau_bot=None, bounds=(2.0, 25.0))
    assert info["r_top"] > info["r_base"] and info["tau_bot"] > 0
    assert draw.shape == (p,)


# --- 23f: posterior diagnostics -------------------------------------------------
def test_23f_posterior_diagnostics():
    rng = np.random.default_rng(1)
    m_obs, p = 8, 4
    K = rng.standard_normal((m_obs, p))
    Sa = np.diag([4.0, 4.0, 1.0, 25.0])
    Se = 0.01 * np.eye(m_obs)
    post = roe.posterior_diagnostics(K, Sa, Se)
    assert 0.0 < post.dofs <= p
    assert post.sic > 0.0
    assert np.all(np.linalg.eigvalsh(post.S_hat) > 0)
    assert np.all(post.data_fraction >= 0.0) and np.all(post.data_fraction <= 1.0)
    d = roe.dofs_by_component(post, 2, retrieve_r_base=True, retrieve_tau_bot=True)
    np.testing.assert_allclose(d["profile"] + d["r_base"] + d["tau_bot"],
                               post.dofs, rtol=1e-12)


# --- 23g: end-to-end noiseless GN retrieval recovers the truth ------------------
@pytest.fixture(scope="module")
def gn_setup(fwd, y_truth):
    Se = _Se(fwd.m)
    clim = dict(r_top_mean=9.0, r_top_std=2.0, r_base_mean=7.5, r_base_std=1.0,
                tau_bot_mean=6.5, tau_bot_std=3.0)     # off-truth, leak-free-style
    x_a, Sa = roe.make_climatology_prior(S_NODES, clim, log=True)
    return Se, clim, x_a, Sa


def test_23g_gn_recovers_truth(fwd, y_truth, gn_setup):
    Se, clim, x_a, Sa = gn_setup
    res = roe.gauss_newton_oe(fwd, y_truth, S_NODES, x_a, Sa, Se,
                              n_iter=10, lm=1e-2, xtol=1e-4, max_n_outer=1)
    assert res.converged
    r_nodes, r_base, tau_bot = (np.asarray(v, float)
                                for v in fwd._split_state(res.x, S_NODES))
    truth = truth_state()
    # noiseless + truth in-class => sub-noise recovery of the well-observed state
    np.testing.assert_allclose(np.atleast_1d(r_nodes)[0], truth[0], rtol=0.02)
    assert abs(float(tau_bot) - TAU_BOT) < 0.3
    # final residual is far below the assumed noise (chi2_red << 1)
    r = res.y - res.Fx
    chi2_red = float(r @ np.linalg.inv(res.Se) @ r) / len(r)
    assert chi2_red < 0.1


# --- 23h: L1 checkpoint resume-equivalence (the in-suite gate) -------------------
def test_23h_l1_resume_equivalence(fwd, y_truth, gn_setup, tmp_path):
    Se, clim, x_a, Sa = gn_setup
    kw = dict(n_iter=6, lm=1e-2, xtol=1e-12, cost_rtol=None, max_n_outer=1)
    ref = roe.gauss_newton_oe(fwd, y_truth, S_NODES, x_a, Sa, Se, **kw)

    ck = str(tmp_path / "gn.ckpt.npz")
    kw_short = dict(kw, n_iter=3)
    roe.gauss_newton_oe(fwd, y_truth, S_NODES, x_a, Sa, Se,
                        checkpoint_path=ck, **kw_short)      # interrupt after 3
    assert os.path.exists(ck)
    res = roe.gauss_newton_oe(fwd, y_truth, S_NODES, x_a, Sa, Se,
                              checkpoint_path=ck, **kw)      # resume -> complete
    np.testing.assert_array_equal(ref.x, res.x)              # bit-exact on one platform


# --- 23i: tau_bot pre-retrieval ---------------------------------------------------
def test_23i_retrieve_tau_bot(fwd, y_truth, gn_setup):
    Se, clim, x_a, Sa = gn_setup
    tau_est, sigma_tau = roe.retrieve_tau_bot(fwd, y_truth, Se, clim, S_NODES)
    assert sigma_tau > 0.0
    assert abs(tau_est - TAU_BOT) < 1.0    # informed anchor, not the final answer


# --- 23k: fused forward+jacobian consistency (E6) -----------------------------------
def test_23k_forward_and_jacobian(fwd):
    x_true = fwd._encode_state(truth_state())
    y1 = np.asarray(fwd.forward(x_true, S_NODES), float)
    K1 = np.asarray(fwd.jacobian(x_true, S_NODES), float)
    y2, K2 = fwd.forward_and_jacobian(x_true, S_NODES)
    # K comes from the identical augmented computation -> tight agreement; the fused
    # primal rides the AUGMENTED adaptive solve, whose step sequence may differ from
    # the plain forward's by ~tol (equally valid solves), hence the looser y bound.
    np.testing.assert_allclose(np.asarray(K2, float), K1, rtol=1e-5,
                               atol=1e-5 * float(np.max(np.abs(K1))))
    np.testing.assert_allclose(np.asarray(y2, float), y1, rtol=3e-3,
                               atol=3e-3 * float(np.max(np.abs(y1))))


# --- 23j: oracle adiabatic floor ---------------------------------------------------
def test_23j_best_fit_adiabatic():
    s = np.linspace(0.0, 1.0, 40)
    re_adia = (R_BASE ** 5 + (R_TOP ** 5 - R_BASE ** 5) * (1.0 - s)) ** 0.2
    fit = roe.best_fit_adiabatic(s, re_adia, TAU_BOT)
    assert fit["success"] and fit["rmse"] < 1e-6       # exact class member -> 0
    np.testing.assert_allclose([fit["r_top"], fit["r_base"]], [R_TOP, R_BASE],
                               rtol=1e-4)


# --- 23l: production numerics tier (float64, tol=1e-4) end-to-end -------------------
@pytest.mark.float64
def test_23l_gn_float64_production_tol(opt_bands):
    """The production precision point (float64 + tol=1e-4, DESIGN §15) on the toy —
    the retrieval stack's only sub-HPC float64 coverage. The noiseless in-class truth
    must be recovered to a much deeper floor than the float32 case."""
    fwd64 = make_fwd(opt_bands, tol=1e-4)
    x_true = fwd64._encode_state(truth_state())
    y64 = np.asarray(fwd64.forward(x_true, S_NODES), float)
    Se = _Se(fwd64.m)
    clim = dict(r_top_mean=9.0, r_top_std=2.0, r_base_mean=7.5, r_base_std=1.0,
                tau_bot_mean=6.5, tau_bot_std=3.0)
    x_a, Sa = roe.make_climatology_prior(S_NODES, clim, log=True)
    res = roe.gauss_newton_oe(fwd64, y64, S_NODES, x_a, Sa, Se,
                              n_iter=10, lm=1e-2, xtol=1e-5, max_n_outer=1)
    assert res.converged
    r = res.y - res.Fx
    chi2_red = float(r @ np.linalg.inv(res.Se) @ r) / len(r)
    assert chi2_red < 1e-3
    _, _, tau_bot = fwd64._split_state(res.x, S_NODES)
    assert abs(float(tau_bot) - TAU_BOT) < 0.1
