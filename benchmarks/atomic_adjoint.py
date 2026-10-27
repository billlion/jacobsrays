"""Compare serial/atomic adjoints in one OpenMP build, including contention.

Each requested thread count runs in a fresh process so OpenMP reads its
environment before initialization. Timings include allocation and validation;
no sparse matrix or per-thread volume is stored. Speed is never a test gate.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from jacobsrays import Grid, RayTransform, _native


def worker(args):
    if not _native.openmp_enabled:
        raise RuntimeError("rebuild with JACOBSRAYS_OPENMP=1 before benchmarking")
    rng = np.random.default_rng(2026)
    n, count = args.size, args.rays
    grid = Grid((n,)*3)
    results = []
    for case in ("distributed", "central", "identical"):
        if case == "distributed":
            starts = rng.uniform(0, n, (count, 3)); starts[:, 0] = -1
            ends = rng.uniform(0, n, (count, 3)); ends[:, 0] = n+1
        elif case == "central":
            directions = rng.normal(size=(count, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            starts, ends = n/2 - n*directions, n/2 + n*directions
        else:
            starts = np.tile([-1., n/2+.25, n/2+.25], (count, 1))
            ends = np.tile([n+1., n/2+.25, n/2+.25], (count, 1))
        serial = RayTransform(grid, starts, ends)
        atomic = RayTransform(grid, starts, ends, adjoint_mode="atomic")
        image = rng.normal(size=grid.image_shape)
        data = rng.normal(size=count)
        reference = serial.adjoint(data)
        ax = serial.direct(image)
        atomic.adjoint(data)
        times = {"serial": [], "atomic": [], "forward": []}
        max_error, max_dot_error = 0., 0.
        for repeat in range(args.repeat):
            # Alternate timing order to reduce systematic warm-up/order bias.
            order = ("serial", "atomic") if repeat % 2 == 0 else ("atomic", "serial")
            for mode in order:
                op = serial if mode == "serial" else atomic
                start = time.perf_counter()
                result = op.adjoint(data)
                times[mode].append(time.perf_counter()-start)
                if mode == "atomic":
                    error = np.linalg.norm(result-reference)/np.linalg.norm(reference)
                    scale = np.linalg.norm(ax)*np.linalg.norm(data) + np.linalg.norm(image)*np.linalg.norm(result)
                    dot_error = abs(np.vdot(ax, data)-np.vdot(image, result))/scale
                    if not (error < 5e-12 and dot_error < 5e-14):
                        raise AssertionError(f"{case}: transpose check failed: {error}, {dot_error}")
                    max_error = max(max_error, float(error))
                    max_dot_error = max(max_dot_error, float(dot_error))
            start = time.perf_counter()
            serial.direct(image)
            times["forward"].append(time.perf_counter()-start)
        stats = {mode: {"minimum_seconds": min(values), "median_seconds": float(np.median(values))}
                 for mode, values in times.items()}
        results.append(dict(case=case, requested_threads=args.worker_threads,
                            rays=count, grid=grid.shape, repeat=args.repeat, timings=stats,
                            median_speedup=stats["serial"]["median_seconds"]/stats["atomic"]["median_seconds"],
                            max_relative_adjoint_difference=max_error,
                            max_norm_scaled_dot_error=max_dot_error))
    return results


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=positive, default=[1, 2, 4, 8])
    parser.add_argument("--rays", type=positive, default=20000)
    parser.add_argument("--size", type=positive, default=64)
    parser.add_argument("--repeat", type=positive, default=5)
    parser.add_argument("--worker-threads", type=positive, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_threads:
        print(json.dumps(worker(args)))
        return
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    result = dict(date=datetime.date.today().isoformat(), platform=platform.platform(),
                  python=platform.python_version(), numpy=np.__version__, cpu=cpu,
                  cpu_count=os.cpu_count(),
                  affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  results=[])
    for threads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-threads", str(threads),
                   "--rays", str(args.rays), "--size", str(args.size), "--repeat", str(args.repeat)]
        run = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        result["results"].extend(json.loads(run.stdout))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
