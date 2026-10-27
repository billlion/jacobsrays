# SPDX-License-Identifier: Apache-2.0
"""Experimental Linux CPU oneMKL Sparse BLAS comparison; no package API changes.

The ctypes adapter is deliberately confined to this benchmark. It supports
real double CSR with LP64 indices, runs in a fresh process, and owns the
arrays referenced by its MKL handle. MKL is never imported by jacobsrays.
"""
import argparse
import ctypes as ct
import datetime
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy
from jacobsrays import Grid, RayTransform, TensorRayTransform, TransverseRayTransform
from jacobsrays import _native


class Descriptor(ct.Structure):
    _fields_ = [("type", ct.c_int), ("mode", ct.c_int), ("diag", ct.c_int)]


def checked(status):
    if status:
        raise RuntimeError(f"MKL Sparse BLAS status {status}")


def resident_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])*1024


def cpu_quota():
    path = Path("/sys/fs/cgroup/cpu.max")
    return path.read_text().strip() if path.exists() else "unavailable"


class MKL:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        self.descr = Descriptor(20, 42, 50)  # general, full, non-unit
        p, i, d = ct.c_void_p, ct.c_int, ct.c_double
        signatures = {
            "mkl_sparse_d_create_csr": [ct.POINTER(p), i, i, i, p, p, p, p],
            "mkl_sparse_destroy": [p],
            "mkl_sparse_set_mv_hint": [p, i, Descriptor, i],
            "mkl_sparse_set_mm_hint": [p, i, Descriptor, i, i, i],
            "mkl_sparse_optimize": [p],
            "mkl_sparse_d_mv": [i, d, p, Descriptor, p, d, p],
            "mkl_sparse_d_mm": [i, d, p, Descriptor, i, p, i, i, d, p, i],
        }
        for name, args in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes = args
            func.restype = i
        self.lib.MKL_Get_Max_Threads.argtypes = []
        self.lib.MKL_Get_Max_Threads.restype = i
        self.lib.MKL_Get_Version_String.argtypes = [p, i]
        self.lib.MKL_Get_Version_String.restype = None
        buf = ct.create_string_buffer(512)
        self.lib.MKL_Get_Version_String(buf, len(buf))
        self.version = buf.value.decode()


class MKLMatrix:
    def __init__(self, matrix, runtime):
        self.runtime = runtime
        self.handle = ct.c_void_p()
        self.closed = False
        if np.iscomplexobj(matrix.data):
            raise ValueError("real matrices only")
        self.matrix = matrix.tocsr(copy=True)
        self.matrix.sum_duplicates()
        self.matrix.sort_indices()
        self.matrix.check_format(full_check=True)
        if max(*self.matrix.shape, self.matrix.nnz) > np.iinfo(np.int32).max:
            raise ValueError("this benchmark adapter is limited to LP64-sized matrices")
        self.matrix.data = np.ascontiguousarray(self.matrix.data, dtype=np.float64)
        self.matrix.indices = np.ascontiguousarray(self.matrix.indices, dtype=np.int32)
        self.matrix.indptr = np.ascontiguousarray(self.matrix.indptr, dtype=np.int32)
        a = self.matrix
        if a.nnz == 0:
            return
        checked(runtime.lib.mkl_sparse_d_create_csr(ct.byref(self.handle), 0,
                *a.shape, a.indptr.ctypes.data, a.indptr[1:].ctypes.data,
                a.indices.ctypes.data, a.data.ctypes.data))
        try:
            for op in [10, 11]:
                checked(runtime.lib.mkl_sparse_set_mv_hint(self.handle, op, runtime.descr, 100))
                checked(runtime.lib.mkl_sparse_set_mm_hint(self.handle, op, runtime.descr, 101, 6, 100))
            checked(runtime.lib.mkl_sparse_optimize(self.handle))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle.value:
            checked(self.runtime.lib.mkl_sparse_destroy(self.handle))
            self.handle = ct.c_void_p()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def dot(self, values, transpose=False):
        if self.closed:
            raise RuntimeError("closed MKL handle")
        if np.iscomplexobj(values):
            raise ValueError("real arrays only")
        x = np.ascontiguousarray(values, dtype=np.float64)
        rows, cols = self.matrix.shape
        if transpose:
            rows, cols = cols, rows
        if x.ndim not in (1, 2) or x.shape[0] != cols:
            raise ValueError("incompatible input shape")
        result = np.empty((rows, *x.shape[1:]), dtype=np.float64)
        if result.size == 0 or cols == 0 or self.matrix.nnz == 0:
            result.fill(0)
            return result
        op = 11 if transpose else 10
        if x.ndim == 1:
            checked(self.runtime.lib.mkl_sparse_d_mv(op, 1., self.handle,
                    self.runtime.descr, x.ctypes.data, 0., result.ctypes.data))
        else:
            k = x.shape[1]
            checked(self.runtime.lib.mkl_sparse_d_mm(op, 1., self.handle,
                    self.runtime.descr, 101, x.ctypes.data, k, k, 0., result.ctypes.data, k))
        return result

    def __matmul__(self, x):
        return self.dot(x)

    @property
    def T(self):
        return Transpose(self)


class Transpose:
    def __init__(self, matrix):
        self.matrix = matrix

    def __matmul__(self, x):
        return self.matrix.dot(x, transpose=True)


class CachedTranspose:
    """Control: spend one extra CSR copy to make SciPy's adjoint gather rows."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.T = matrix.T.tocsr()

    def __matmul__(self, values):
        return self.matrix @ values


def compare(functions, repeat):
    reference = functions["scipy"]()
    errors = {}
    for name, function in functions.items():
        value = function()
        scale = max(np.linalg.norm(reference), np.finfo(float).tiny)
        error = float(np.linalg.norm(value-reference)/scale)
        if not np.isfinite(error) or error > 5e-12:
            raise AssertionError((name, error))
        errors[name] = error
    times = {name: [] for name in functions}
    names = list(functions)
    for rep in range(repeat):
        # Rotate and reverse order; no accuracy checks inside timed calls.
        order = names[rep % len(names):] + names[:rep % len(names)]
        if rep % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            value = functions[name]()
            times[name].append(time.perf_counter()-start)
            del value
    return {name: {"samples_seconds": samples, "median_seconds": statistics.median(samples),
                   "minimum_seconds": min(samples), "relative_error": errors[name]}
            for name, samples in times.items()}


def worker(args):
    runtime = MKL(args.library)
    rng = np.random.default_rng(20261027)
    n, count = args.size, args.rays
    grid = Grid((n,)*3)
    starts = rng.uniform(0, n, (count, 3)); starts[:, 0] = -1
    ends = rng.uniform(0, n, (count, 3)); ends[:, 0] = n+1
    start = time.perf_counter()
    scalar = RayTransform(grid, starts, ends)
    a = scalar.paths().to_csr()
    geometry_seconds = time.perf_counter()-start
    rss_before_setup = resident_bytes()
    start = time.perf_counter()
    mkl = MKLMatrix(a, runtime)
    setup_seconds = time.perf_counter()-start
    rss_after_setup = resident_bytes()
    sorted_a = mkl.matrix
    start = time.perf_counter()
    cached = CachedTranspose(sorted_a)
    cached_setup_seconds = time.perf_counter()-start
    x = rng.normal(size=grid.image_shape)
    y = rng.normal(size=count)
    # Component-major input is the layout used by the current tensor API.
    f = rng.normal(size=(6, *grid.image_shape))
    v = rng.normal(size=(count, 6))
    operations = {}
    operations["scalar_forward"] = compare({"scipy": lambda: a@x.ravel(),
            "scipy_sorted": lambda: sorted_a@x.ravel(), "mkl": lambda: mkl@x.ravel()}, args.repeat)
    operations["scalar_adjoint"] = compare({"scipy": lambda: a.T@y,
            "scipy_sorted": lambda: sorted_a.T@y,
            "scipy_cached_transpose": lambda: cached.T@y, "mkl": lambda: mkl.T@y}, args.repeat)
    operations["six_forward"] = compare({"scipy": lambda: a@f.reshape(6, -1).T,
            "scipy_sorted": lambda: sorted_a@f.reshape(6, -1).T,
            "mkl": lambda: mkl@f.reshape(6, -1).T}, args.repeat)
    operations["six_adjoint"] = compare({"scipy": lambda: a.T@v,
            "scipy_sorted": lambda: sorted_a.T@v,
            "scipy_cached_transpose": lambda: cached.T@v, "mkl": lambda: mkl.T@v}, args.repeat)
    for name, cls in [("lrt", TensorRayTransform), ("trt", TransverseRayTransform)]:
        op = cls(grid, starts, ends)
        data = y if name == "lrt" else v
        functions_direct, functions_adjoint = {}, {}
        def direct(matrix):
            op.matrix = matrix
            return op.direct(f)
        def adjoint(matrix):
            op.matrix = matrix
            return op.adjoint(data)
        for label, matrix in [("scipy", a), ("scipy_sorted", sorted_a),
                              ("scipy_cached_transpose", cached), ("mkl", mkl)]:
            functions_direct[label] = lambda matrix=matrix: direct(matrix)
            functions_adjoint[label] = lambda matrix=matrix: adjoint(matrix)
        operations[name+"_forward"] = compare(functions_direct, args.repeat)
        operations[name+"_adjoint"] = compare(functions_adjoint, args.repeat)
        ax, aty = direct(mkl), adjoint(mkl)
        denom = np.linalg.norm(ax)*np.linalg.norm(data)+np.linalg.norm(f)*np.linalg.norm(aty)
        dot_error = float(abs(np.vdot(ax, data)-np.vdot(f, aty))/denom)
        if not np.isfinite(dot_error) or dot_error > 5e-14:
            raise AssertionError((name, dot_error))
        operations[name+"_dot_error"] = dot_error
        del op
        gc.collect()
    mkl.close()
    return dict(threads=args.worker_threads, mkl_max_threads=runtime.lib.MKL_Get_Max_Threads(),
                mkl_version=runtime.version, grid=list(grid.shape), rays=count, nnz=a.nnz,
                csr_bytes=sum(z.nbytes for z in [a.data,a.indices,a.indptr]),
                geometry_seconds=geometry_seconds, mkl_setup_seconds=setup_seconds,
                scipy_transpose_setup_seconds=cached_setup_seconds,
                scipy_transpose_bytes=sum(z.nbytes for z in [cached.T.data,cached.T.indices,cached.T.indptr]),
                rss_before_mkl_setup=rss_before_setup, rss_after_mkl_setup=rss_after_setup,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                operations=operations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(sys.prefix)/"lib/libmkl_rt.so.3")
    parser.add_argument("--threads", type=int, nargs="+", default=[1,2,4,8])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--rays", type=int, default=20000)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--worker-threads", type=int)
    args = parser.parse_args()
    if min(args.size, args.rays, args.repeat, *args.threads) < 1:
        parser.error("positive dimensions, counts and thread numbers required")
    if args.worker_threads:
        print(json.dumps(worker(args)))
        return
    results = []
    for threads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                   OMP_WAIT_POLICY="PASSIVE",
                   MKL_NUM_THREADS=str(threads), MKL_DYNAMIC="FALSE", MKL_THREADING_LAYER="GNU",
                   MKL_INTERFACE_LAYER="LP64", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1")
        command = [sys.executable, __file__, "--worker-threads", str(threads),
                   "--size",str(args.size),"--rays",str(args.rays),"--repeat",str(args.repeat),
                   "--library",str(args.library)]
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        results.append(json.loads(result.stdout))
        print(f"Finished {threads} threads", file=sys.stderr, flush=True)
    cpuinfo = Path("/proc/cpuinfo").read_text()
    cpu = next(line.split(":",1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name"))
    print(json.dumps(dict(date=datetime.date.today().isoformat(), cpu=cpu,
        affinity=sorted(os.sched_getaffinity(0)), cpu_max=cpu_quota(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), repeat=args.repeat, threading_layer="GNU", interface="LP64",
        omp_wait_policy="PASSIVE", jacobsrays_openmp=bool(_native.openmp_enabled),
        results=results), indent=2))


if __name__ == "__main__":
    main()
