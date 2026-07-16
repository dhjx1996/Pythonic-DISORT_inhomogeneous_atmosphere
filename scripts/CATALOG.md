# CATALOG — scripts/

## Production pipeline (HPC; specs in `hpc/AGENT_all125_{rad,fr}.md`)

| File | What it is |
|---|---|
| `generate_osse_radiances.py` | Precomputes the "synthetic L1B" — exact truth radiances `y = F(truth)` per profile, consolidated into the signature-gated `../data/osse_radiances*.npz` cache every downstream worker loads. |
| `retrieval_worker.py` | The production entry point: one profile's full Gauss–Newton r_e(τ) retrieval (A/B configs, τ_bot pre-retrieval, QRCP grid, L1/L2 checkpoint-resume) writing the per-index sidecars. |
| `retrieval_analysis.py` | Post-hoc metric suite over the `fr` sidecars — W1(r_e(τ)) + LWP-bias primary, Mahalanobis, DOFS/SIC — pure NumPy re-computation into a JSON summary and printed tables. |

## Information content (§15; all post-processing of existing sidecars except the wiggle probe)

| File | What it is |
|---|---|
| `ic_retrieval_grid.py` | The canonical IC product: per-profile DOFS/SIC, exact band/view-group Shapley, greedy saturation, matched-row budgets and the correlation-pump statistic on the operational QRCP grids → `docs/cached_results/ic_retrieval_grid.json`; also home of the one `shapley_shares` implementation. |
| `ic_kforce_demo.py` | Rigor gate (i) for the retrieval-grid reframing: forces k = 4..16 past QRCP's choice at two linearization points and two solver tolerances and shows DOFS plateaus → `ic_kforce_demo.json`. |
| `ic_worker_wiggle.py` | The one IC script that still solves RT: dense-depth-grid ToA Jacobian probe with the phase function switchable Mie↔HG and coarse↔fine optics tables (wiggle provenance; produced the cached `kernel_probe_ve046_*` sidecars = rigor gate (ii)). |
| `ic_figs.py` | Renders every §15 figure — signed Eq-3 kernels (0a), angular axis (0b), correlation pump (5) from the kernel probes, plus population Shapley/saturation/budget stats (A, B) from `ic_retrieval_grid.json`. |

## Literature validation + plotting

| File | What it is |
|---|---|
| `pla2000_table3a.py` | Reproduces Platnick (2000) Table 3a with our solver — both the homogeneous-equivalent "retrieval" column and the kernel stage — as an external-literature validation. |
| `platnick_eq3_validation.py` | Pla2000 Eq-3 check on the wiggle sidecars: the signed-kernel weighted r* against the homogeneous-equivalent retrieval (GPU compute stage → `eq3_<idx>.npz`). |
| `platnick_eq4_wm.py` | Pla2000 Eq-4 companion: maximum-penetration weighting w_m = (1/R)·dR/dτ from the production solver (→ `eq4_<idx>.npz`). |
| `plot_retrieved_profile.py` | THE standard single-profile retrieval diagnostic plot (truth / prior / retrieved ±σ in absolute τ; format fixed to the `bad_Adiabest-fit.png` template) — reuse this, don't write one-off plotting code. |
