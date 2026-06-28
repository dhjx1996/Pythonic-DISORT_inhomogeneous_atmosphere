# Delegated task — A100 GPU probe: does batching the Fourier modes beat CPU?

*(Short hand-off for a Sonnet agent. Read-only experiment — **no commits, no changes to the production JAX
env**. Report numbers back; the primary decides. 2026-06-28.)*

## Why

We have a `jax.vmap`-over-Fourier-modes variant of the solver that is **bit-correct and composes with the
forward-mode adjoint**, but on **CPU it runs ~3× slower** than the production `lax.scan` (XLA-CPU does not
parallelize a batched implicit ODE solve; jovyan, 8 threads, warm: scan fwd 67 s / jac 206 s vs vmap fwd
181 s / jac 639 s). Batching independent small solves is a **GPU** pattern (SIMT over the batch), and the
A100 has real float64 (we are locked to float64). **Question: does GPU-vmap beat CPU-scan?** If yes, a GPU
retrieval path is worth building; if not, we ship the existing 2-day CPU batch and move on.

## Paths

- Repo: `/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere`.
  **Sync first:** `git fetch origin && git reset --hard origin/main`.
- `PY=/burg-archive/home/dh3065/miniconda3/envs/JAX/bin/python` · `ROOT=<repo above>`.
- Benchmark (already in the repo after sync): `tests/supplementary/vmap_probe.py`.

## Step 1 — get an A100 + a CUDA-enabled JAX

```bash
salloc -C a100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00
cd $ROOT && git fetch origin && git reset --hard origin/main
$PY -c "import jax; print(jax.devices())"          # must show a CudaDevice
```
If that prints only CPU, the env has no CUDA JAX. **Do not `pip install` into the production `JAX` env** (it
runs the CPU batches). Instead get a GPU JAX in *isolation*: prefer a cluster module (`module avail jax`/
`module load cuda`), else make a throwaway venv —
`python -m venv /tmp/jaxgpu && /tmp/jaxgpu/bin/pip install -U "jax[cuda12]" numpy scipy diffrax miepython numba`
— and use `PYG=/tmp/jaxgpu/bin/python` for the GPU runs only. Match the cluster's CUDA major version
(cuda12 vs cuda11). Confirm `…  -c "import jax; print(jax.devices())"` shows the A100 before proceeding.

## Step 2 — build the optics table once (~3–4 min, CPU; the probe also self-builds if absent)

```bash
$PY - <<EOF
import sys; sys.path.insert(0,'$ROOT/src'); sys.path.insert(0,'$ROOT/tests/supplementary')
import osse_config as oc; oc.load_optics('$ROOT/tests/supplementary/optics_table_10band_nleg1024_re20.npz')
print("optics OK")
EOF
```

## Step 3 — run the three probes (each ~5–12 min incl. compile)

Use the **GPU** python (`PYG`, or `$PY` if it already saw the A100) for the cuda runs, `$PY` for the CPU run:

```bash
cd $ROOT
echo "=== GPU vmap ==="; JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 $PYG tests/supplementary/vmap_probe.py vmap
echo "=== GPU scan ==="; JAX_PLATFORMS=cuda PYDISORT_RICCATI_JAX_X64=1 $PYG tests/supplementary/vmap_probe.py scan
echo "=== CPU scan ==="; JAX_PLATFORMS=cpu  PYDISORT_RICCATI_JAX_X64=1 OMP_NUM_THREADS=8 $PY tests/supplementary/vmap_probe.py scan
```

Each prints a `[mode] JAX backend:` line, a `forward …` line (with `y[:3]`/`sum`/`n_neg`), and the key
`[mode] EVAL-ONLY: forward Xs  jacfwd Ys` + a mem line.

## Report back

Paste the three `EVAL-ONLY` lines, the backend/device lines, and the mem lines. The decisive comparison is
**GPU-vmap `jacfwd` vs CPU-scan `jacfwd`** (jovyan CPU-scan was 206 s): a large GPU win (e.g. ≲ 50 s) means a
GPU retrieval path is worth designing; roughly-equal-or-worse means it isn't on this hardware. Also confirm
the `forward` `sum=` agrees across all three (bit-identity of vmap vs scan, modulo cross-backend rounding) and
that `n_neg=0` everywhere. Then **stop** — no further action; the primary decides next steps.
