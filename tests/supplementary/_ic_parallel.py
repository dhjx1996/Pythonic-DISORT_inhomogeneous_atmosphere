"""Generic data-parallel launcher for the (latency-bound, embarrassingly-parallel-over-
profiles) information-content workers.

One JAX/XLA process self-parallelizes a single Jacobian to only ~10/16 cores (the ODE
solve is partly sequential + small matrices), so the box is better used by running several
profiles CONCURRENTLY, each pinned (taskset) to a disjoint core subset. Memory is a
non-issue (~3 GB/process of 128). This is the pattern for Stage-2's all-126-column run too.

    # tasks on stdin, one worker-arg-line each:
    printf '1.0\\n2.0\\n...\\n' | python _ic_parallel.py <worker.py> <out.json> <n_conc> <cores_per>

Each task -> `taskset -c <slot-cores> python <worker> <line-args> <part.json>`; the worker
writes ONE json dict; the launcher throttles to n_conc and merges parts (flat list) -> out.json.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    worker, out_json, n_conc, cores_per = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    tasks = [ln.split() for ln in sys.stdin.read().splitlines() if ln.strip()]
    part_dir = Path("/tmp") / ("_ic_parts_" + Path(out_json).stem)   # intermediates: /tmp, not the results dir
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    core_sets = [f"{i*cores_per}-{i*cores_per+cores_per-1}" for i in range(n_conc)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("PYDISORT_RICCATI_JAX_X64", "1")

    todo = list(enumerate(tasks))
    free = list(range(n_conc))
    running = {}                                  # slot -> (proc, idx, task)
    t0 = time.time()
    print(f"{len(tasks)} tasks, {n_conc} concurrent x {cores_per} cores", flush=True)
    while todo or running:
        while free and todo:
            slot = free.pop(0)
            idx, task = todo.pop(0)
            part = part_dir / f"{idx:03d}.json"
            cmd = ["taskset", "-c", core_sets[slot], sys.executable, worker, *task, str(part)]
            running[slot] = (subprocess.Popen(cmd, env=env), idx, task)
            print(f"  [t+{time.time()-t0:4.0f}s] slot{slot}(cores {core_sets[slot]}) <- task{idx} {task}", flush=True)
        time.sleep(2)
        for slot, (p, idx, task) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                ok = (rc == 0 and (part_dir / f"{idx:03d}.json").exists())
                print(f"  [t+{time.time()-t0:4.0f}s] task{idx} {task} {'done' if ok else f'FAILED rc={rc}'}", flush=True)
                del running[slot]
                free.append(slot)

    parts = sorted(part_dir.glob("*.json"), key=lambda f: int(f.stem))
    merged = [json.loads(f.read_text()) for f in parts]
    Path(out_json).write_text(json.dumps(merged, indent=2))
    if len(merged) == len(tasks):
        shutil.rmtree(part_dir)                          # clean intermediates on full success
    print(f"\nmerged {len(merged)}/{len(tasks)} -> {out_json}  ({time.time()-t0:.0f}s wall)", flush=True)


if __name__ == "__main__":
    main()
