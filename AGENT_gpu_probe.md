# Delegated task — A100 GPU probe #2: does batching the BANDS (on top of modes) add more?

*(Short hand-off for a Sonnet agent. Read-only experiment — **no commits, no changes to the production JAX
env**. Report numbers back; the primary decides. 2026-06-28.)*

> **Update (monitoring agent, 2026-06-28):** Step 3 was revised — **split** into a GPU-only job
> (`vmap_loop` + `vmap_both`) and a **separate CPU-partition** job (`scan` reference). The original
> single ~1 h A100 job ran the CPU-scan leg first, but the 10-band CPU-scan `jacfwd` *compile alone*
> is >40 min, so it consumed the walltime and the GPU legs never ran (the first attempt, job 8614739,
> TIMED OUT with no GPU numbers). The CPU reference needs no A100, so running it concurrently on a CPU
> node both fixes the timeout and stops burning GPU time on the slow reference.

## Why

Your probe #1 settled the first axis: vmap-over-**modes** jacfwd **28.3 s vs CPU-scan 197.3 s (~7×)** on one
band, beating GPU-scan ~2× — batching the azimuthal modes is a GPU (SIMT-over-batch) win. The full A100 peak
was only **1.26 / 40 GB** → the GPU was *badly* under-used at 24-way. So the follow-up: does ALSO batching the
**10 bands** into the same vmap (→ **240-way: 10 bands × 24 modes**) buy *more*, or does the adaptive solver's
lock-step eat it?

The tension this probe resolves: batching bands forces the vmapped `while_loop` to run to the **max** adaptive
step count across bands, masking the bands that converged early. The absorbing bands take a few more steps
(per-band m=0: **21/21/22/22/23** over 0.55→3.7 µm → ~10 % lock-step idle). So it's {SIMT-fill gain} vs
{~10 % lock-step cost}. The modes-vmap 7× *already pays* the same kind of penalty (per-mode steps 17–19) and
still won, so the prior is "bands help too" — but it's unmeasured, hence this probe. The primary has built and
**CPU-validated** the bands×modes path bit-identical to scan (forward rel 1.6e-13, jacfwd rel 3.1e-12,
n_neg=0); only the GPU *speed* is open.

## Paths

- Repo: `/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere`.
  **Sync first:** `git fetch origin && git reset --hard origin/main` (the bands×modes code + this probe are on
  `main`).
- `PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python` · `ROOT=<repo above>`.
- Benchmark (in the repo after sync): `tests/supplementary/vmap_probe_bands.py`.

## Step 1 — A100 + GPU JAX (you solved this in probe #1; reuse it)

```bash
salloc -C a100 --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=01:00:00
cd $ROOT && git fetch origin && git reset --hard origin/main
$PY -c "import jax; print(jax.devices())"          # must show a CudaDevice
```
Per your results.md §A, the conda `JAX` env has all the `nvidia-*-cu12` libs + jaxlib **0.10.2** but the
`jax-cuda12-plugin`/`-pjrt` are **0.9.2** (so JAX disables the plugin). Your fix — a `--system-site-packages`
**overlay venv** reusing the conda libs + jaxlib and installing only the matching **0.10.2** plugin/pjrt —
worked; rebuild it the same way and use `PYG=<overlay>/bin/python` for the cuda runs (the production env stays
untouched). Confirm `… -c "import jax; print(jax.devices())"` shows the A100 before proceeding.

## Step 2 — optics table (built once in probe #1; the probe self-loads it)

The probe loads `tests/supplementary/optics_table_10band_nleg1024_re20.npz` (already present from the rad
batch). No action unless it's missing (then rebuild as in `AGENT_all125_rad.md`).

## Step 3 — run the probes: **GPU-only job + a parallel CPU reference** (split; see Update note)

The full forward is 10 bands × NQuad=48 × NLeg_all=1024, so the `jacfwd` compile is much heavier than probe
#1 — on **CPU** the 10-band `scan` `jacfwd` *compile alone* is >40 min. Do **not** put the CPU leg in the GPU
job: run the two GPU legs on the A100 and the CPU reference on an ordinary CPU node, **concurrently**.

**(a) GPU job — the two vmap legs (the decisive numbers).** With `$PYG` from Step 1 (fits well under 1 h):
```bash
cd $ROOT
echo "=== GPU vmap_loop (modes only) ==="; JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 $PYG tests/supplementary/vmap_probe_bands.py vmap_loop
echo "=== GPU vmap_both (bands x modes) ==="; JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 $PYG tests/supplementary/vmap_probe_bands.py vmap_both
```

**(b) CPU reference — a SEPARATE non-GPU job** (no A100; ~2–3 h for the heavy CPU `jacfwd` compile):
```bash
# e.g. sbatch --partition=short --cpus-per-task=8 --mem=32G --time=03:00:00  (no --gres/-C a100)
JAX_PLATFORMS=cpu PYDISORT_RICCATI_JAX_X64=1 OMP_NUM_THREADS=8 $PY tests/supplementary/vmap_probe_bands.py scan
```

- `scan` = production CPU path (band-loop, modes-scan) — the number to beat.
- `vmap_loop` = GPU, axis 1 only (band-loop, modes-vmap; 10 sequential 24-way solves).
- `vmap_both` = GPU, axis 2 (one vmap over bands × modes; 240-way) — the target.

Each prints a `path =` line, a `forward …` line (`sum`/`n_neg`), the key `EVAL-ONLY: forward Xs jacfwd Ys`,
and a `device peak … GB` line.

## Report back

Paste the three `EVAL-ONLY` lines, the backend/device lines, and the mem lines. The decisive comparisons:

- **(i) Does batching bands help?** `vmap_both` jacfwd **vs** `vmap_loop` jacfwd. If `vmap_both` is clearly
  faster, the second axis (filling the under-used A100) beats the lock-step cost → build the full bands×modes
  GPU retrieval. If they're ~equal or `vmap_both` is slower, the lock-step ate the fill → ship axis-1 only
  (modes-vmap, bands looped), which already gives the 7×.
- **(ii) End-to-end:** `vmap_both` (or the winner) jacfwd **vs** CPU-scan jacfwd — the real per-solve speedup
  the retrieval will see.
- **Memory:** the `vmap_both` `device peak` — confirm the 240-way fits the A100 with headroom (expected ≈
  10–13 GB; flag if it's near 40).
- **Correctness:** confirm `forward sum` agrees across all three (modulo cross-backend rounding) and `n_neg=0`
  everywhere.

Then **stop** — no further action; the primary designs the GPU retrieval worker from these numbers.
