# SPDX-License-Identifier: Apache-2.0
"""Experimental rank-two tensor wrapper around TIGRE's scalar GPU projectors.

Only centred, untilted 3D parallel scans around z are supported. This is a
host-array prototype: six Ax/Atb calls, NumPy contractions, transfers included.
TIGRE's backprojector is NOT advertised as the exact discrete adjoint.
Nothing here is imported by the installed jacobsrays package.
"""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy
from jacobsrays import Grid, trace


TIGRE_COMMIT = "9195fd57faa1eba47dedfd45d95a9957d3910dbd"
COMPONENTS = ("xx", "sqrt(2)xy", "sqrt(2)xz", "yy", "sqrt(2)yz", "zz")


def checked_array(value, shape, dtype):
    if np.iscomplexobj(value):
        raise ValueError("real arrays required")
    value = np.asarray(value, dtype=dtype)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"expected finite array of shape {shape}")
    return np.ascontiguousarray(value)


@dataclass
class ParallelScan:
    """Explicit geometry shared by the CPU reference and TIGRE adapter.

    Physical coordinates are xyz; images are zyx; sinograms are (view, v, u).
    At angle zero rays run from +x to -x, u increases along +y, v along +z.
    This agrees with TIGRE's Python Siddon parallel kernel (including its
    internal v reversal), not the MATLAB memory layout. No offsets or tilts.
    """
    grid: Grid
    angles: np.ndarray
    detector_shape: tuple
    detector_spacing: tuple
    distance: float

    @classmethod
    def create(cls, shape=(64, 64, 64), spacing=(1., 1., 1.), views=32):
        if len(shape) != 3 or views < 1 or int(views) != views:
            raise ValueError("three grid dimensions and a positive integer view count required")
        extent = np.asarray(shape)*np.asarray(spacing)
        grid = Grid(tuple(shape), tuple(spacing), tuple(-extent/2))
        # Quantise acquisition parameters to TIGRE's precision before making
        # CPU rays; both implementations see the same specified angles.
        angles = np.asarray(.137 + np.arange(views)*np.pi/views, dtype=np.float32)
        du = float(np.float32(.85*min(spacing[:2])))
        dv = float(np.float32(spacing[2]))
        nu = int(math.ceil(np.hypot(*extent[:2])/du)) + 2
        return cls(grid, angles, (shape[2], nu), (dv, du),
                   float(np.float32(2*np.linalg.norm(extent))))

    @property
    def range_shape(self):
        return (len(self.angles), *self.detector_shape)

    @property
    def directions(self):
        a = self.angles.astype(np.float64)
        directions = np.column_stack((-np.cos(a), -np.sin(a), np.zeros_like(a)))
        return np.broadcast_to(directions[:, None, None, :], (*self.range_shape, 3)).reshape(-1, 3)

    def rays(self):
        nv, nu = self.detector_shape
        dv, du = self.detector_spacing
        v, u = np.meshgrid((np.arange(nv)-(nv-1)/2)*dv,
                           (np.arange(nu)-(nu-1)/2)*du, indexing="ij")
        a = self.angles.astype(np.float64)
        c, s = np.cos(a)[:, None, None], np.sin(a)[:, None, None]
        points = np.stack((-s*u, c*u, np.broadcast_to(v, self.range_shape)), axis=-1)
        axis = -self.directions.reshape(*self.range_shape, 3)
        return ((points+self.distance*axis).reshape(-1, 3),
                (points-self.distance*axis).reshape(-1, 3))

    def tigre_geometry(self, tigre):
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(self.grid.image_shape))
        geo.dVoxel = np.asarray(self.grid.spacing[::-1], dtype=np.float32)
        geo.sVoxel = geo.nVoxel*geo.dVoxel
        geo.nDetector = np.asarray(self.detector_shape, dtype=np.int32)
        geo.dDetector = np.asarray(self.detector_spacing, dtype=np.float32)
        geo.sDetector = geo.nDetector*geo.dDetector
        geo.DSO, geo.DSD = self.distance, 2*self.distance
        geo.offOrigin = np.zeros(3, dtype=np.float32)
        geo.offDetector = np.zeros(2, dtype=np.float32)
        geo.rotDetector = np.zeros(3, dtype=np.float32)
        geo.COR = 0.
        return geo


class MatrixScalar:
    """CPU reference, with an explicitly cached transpose when given a CSR."""
    def __init__(self, scan, matrix, dtype=np.float64):
        self.scan, self.dtype = scan, np.dtype(dtype)
        self.matrix = matrix
        self.transpose = matrix.T.tocsr() if hasattr(matrix.T, "tocsr") else matrix.T

    def direct(self, image):
        x = checked_array(image, self.scan.grid.image_shape, self.dtype)
        return np.asarray(self.matrix @ x.ravel()).reshape(self.scan.range_shape)

    def backproject(self, data):
        y = checked_array(data, self.scan.range_shape, self.dtype)
        return np.asarray(self.transpose @ y.ravel()).reshape(self.scan.grid.image_shape)

    def direct_components(self, image):
        # Match the existing tensor CPU implementation: one SpMM for all
        # components, not six slower SpMV calls as a straw-man baseline.
        return np.asarray(self.matrix @ image.reshape(6, -1).T)

    def backproject_components(self, values):
        return np.asarray(self.transpose @ values).T.reshape(6, *self.scan.grid.image_shape)


class TigreScalar:
    """Siddon forward and TIGRE 'matched' backprojection on one visible GPU.

    Atb is a backprojection candidate, not a guaranteed transpose of Ax.
    CUDA_VISIBLE_DEVICES is respected; no CPU fallback is permitted.
    """
    def __init__(self, scan):
        import tigre
        from tigre.utilities import gpu
        ids = gpu.GpuIds()
        if len(ids) == 0:
            raise RuntimeError("TIGRE found no visible CUDA GPU; refusing to skip GPU validation")
        self.gpuids = ids[0]
        self.gpu_names = gpu.getGpuNames()
        self.scan, self.dtype = scan, np.dtype(np.float32)
        self.tigre, self.geometry = tigre, scan.tigre_geometry(tigre)

    def direct(self, image):
        x = checked_array(image, self.scan.grid.image_shape, self.dtype)
        result = self.tigre.Ax(x, self.geometry, self.scan.angles,
                               projection_type="Siddon", gpuids=self.gpuids)
        return checked_array(result, self.scan.range_shape, self.dtype)

    def backproject(self, data):
        y = checked_array(data, self.scan.range_shape, self.dtype)
        result = self.tigre.Atb(y, self.geometry, self.scan.angles,
                                backprojection_type="matched", gpuids=self.gpuids)
        return checked_array(result, self.scan.grid.image_shape, self.dtype)


class TensorWrapper:
    """Rank-two LRT/TRT in jacobsrays' Frobenius-orthonormal component basis."""
    def __init__(self, scalar, kind):
        if kind not in ("lrt", "trt"):
            raise ValueError("kind must be lrt or trt")
        self.scalar, self.kind = scalar, kind
        self.dtype = scalar.dtype
        self.domain_shape = (6, *scalar.scan.grid.image_shape)
        self.range_shape = (math.prod(scalar.scan.range_shape),) + (() if kind == "lrt" else (6,))
        self.basis = np.zeros((6, 3, 3), dtype=self.dtype)
        for k, (i, j) in enumerate((i, j) for i in range(3) for j in range(i, 3)):
            self.basis[k, i, j] = self.basis[k, j, i] = 1 if i == j else 1/np.sqrt(2)
        u = scalar.scan.directions.astype(self.dtype)
        self.weights = np.einsum("ri,cij,rj->rc", u, self.basis, u)
        self.projections = np.eye(3, dtype=self.dtype)[None] - u[:, :, None]*u[:, None, :]

    def transverse(self, values):
        matrices = np.einsum("rc,cij->rij", values, self.basis)
        matrices = self.projections @ matrices @ self.projections
        return np.einsum("rij,cij->rc", matrices, self.basis)

    def direct(self, image):
        x = checked_array(image, self.domain_shape, self.dtype)
        if hasattr(self.scalar, "direct_components"):
            values = self.scalar.direct_components(x)
        else:
            values = np.column_stack([self.scalar.direct(component).ravel() for component in x])
        return np.sum(values*self.weights, axis=1) if self.kind == "lrt" else self.transverse(values)

    def backproject(self, data):
        y = checked_array(data, self.range_shape, self.dtype)
        values = y[:, None]*self.weights if self.kind == "lrt" else self.transverse(y)
        if hasattr(self.scalar, "backproject_components"):
            return self.scalar.backproject_components(values)
        return np.stack([self.scalar.backproject(values[:, k].reshape(self.scalar.scan.range_shape))
                         for k in range(6)])


def relative_error(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite result")
    difference, denominator = np.linalg.norm(actual-reference), np.linalg.norm(reference)
    # A zero reference with a nonzero result has no finite relative error.
    # JSON null preserves this fact without producing invalid Infinity JSON.
    if denominator == 0:
        return 0. if difference == 0 else None
    return float(difference/denominator)


def dot_check(x, y, ax, by):
    x, y, ax, by = (np.asarray(a, dtype=np.float64) for a in (x, y, ax, by))
    left, right = float(np.vdot(ax, y)), float(np.vdot(x, by))
    denominator = np.linalg.norm(ax)*np.linalg.norm(y)+np.linalg.norm(x)*np.linalg.norm(by)
    return dict(forward_dot=left, backprojection_dot=right,
                scaled_defect=float(abs(left-right)/max(denominator, np.finfo(float).tiny)))


def validate_gpu():
    """Extract tiny complete GPU forward matrices; compare with CPU weights.

    Backprojection is also compared with the transpose of TIGRE's OWN
    measured forward matrix, isolating its mismatch from geometric differences.
    No rescaling or fitted normalization is applied to either operator.
    """
    rng = np.random.default_rng(20261028)
    results = []
    for spacing in ((1., 1., 1.), (.75, 1.25, 1.5)):
        scan = ParallelScan.create((4, 5, 3), spacing, views=7)
        starts, ends = scan.rays()
        reference = trace(scan.grid, starts, ends).to_csr()
        gpu = TigreScalar(scan)
        matrix = np.empty(reference.shape, dtype=np.float32)
        unit = np.zeros(scan.grid.image_shape, dtype=np.float32)
        for j in range(reference.shape[1]):
            unit.fill(0)
            unit.flat[j] = 1
            matrix[:, j] = gpu.direct(unit).ravel()
        cpu = MatrixScalar(scan, reference)
        dense = reference.toarray()
        forward_error = relative_error(matrix, dense)
        max_coefficient_error = float(np.max(np.abs(matrix-dense)))
        item = dict(grid=list(scan.grid.shape), spacing=list(spacing),
                    rays=reference.shape[0], gpu_names=gpu.gpu_names,
                    forward_matrix_relative_error=forward_error,
                    max_coefficient_absolute_error=max_coefficient_error,
                    forward_passed=bool(forward_error <= 5e-4 and
                                        max_coefficient_error <= 5e-3*min(spacing)),
                    operators={})
        for kind in ("scalar", "lrt", "trt"):
            candidate = gpu if kind == "scalar" else TensorWrapper(gpu, kind)
            expected = cpu if kind == "scalar" else TensorWrapper(cpu, kind)
            domain = scan.grid.image_shape if kind == "scalar" else candidate.domain_shape
            shape = scan.range_shape if kind == "scalar" else candidate.range_shape
            trials = []
            for trial in range(3):
                # Include a positive trial so cancellation in random dot
                # products cannot conceal a normalization discrepancy.
                x = np.asarray(rng.normal(size=domain) if trial else np.ones(domain), dtype=np.float32)
                y = np.asarray(rng.normal(size=shape) if trial else np.ones(shape), dtype=np.float32)
                ax, by = candidate.direct(x), candidate.backproject(y)
                entry = dict(forward_relative_error=relative_error(ax, expected.direct(x)),
                             backprojection_vs_cpu_adjoint=relative_error(by, expected.backproject(y)),
                             dot=dot_check(x, y, ax, by))
                # Dense CPU products with the extracted GPU matrix are an
                # independent exact-transpose reference for this small case.
                own = MatrixScalar(scan, matrix.astype(np.float64))
                own = own if kind == "scalar" else TensorWrapper(own, kind)
                entry["backprojection_vs_own_forward_transpose"] = relative_error(by, own.backproject(y))
                trials.append(entry)
            item["operators"][kind] = trials
        results.append(item)
    return results


def time_operations(functions, repeat):
    # TIGRE's public API returns host arrays after device-to-host copies.
    # Consequently wall time includes completed GPU work and both transfers.
    for fn in functions.values():
        fn()
    samples = {name: [] for name in functions}
    names = list(functions)
    for rep in range(repeat):
        order = names[rep % len(names):] + names[:rep % len(names)]
        if rep % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            value = functions[name]()
            samples[name].append(time.perf_counter()-start)
            del value
    return {name: dict(samples_seconds=values, median_seconds=statistics.median(values))
            for name, values in samples.items()}


def benchmark(size, views, repeat, mkl_library=None):
    scan = ParallelScan.create((size,)*3, views=views)
    start = time.perf_counter()
    starts, ends = scan.rays()
    matrix = trace(scan.grid, starts, ends).to_csr()
    matrix.sort_indices()
    geometry_seconds = time.perf_counter()-start
    setup = {}
    with ExitStack() as stack:
        scalars = {}
        for dtype in (np.float64, np.float32):
            start = time.perf_counter()
            scalars[f"scipy_{np.dtype(dtype).name}"] = MatrixScalar(scan, matrix.astype(dtype), dtype)
            setup[f"scipy_{np.dtype(dtype).name}"] = time.perf_counter()-start
        if mkl_library:
            from mkl_probe import MKL, MKLMatrix
            runtime = MKL(Path(mkl_library))
            start = time.perf_counter()
            mkl = stack.enter_context(MKLMatrix(matrix, runtime))
            scalars["mkl_float64"] = MatrixScalar(scan, mkl)
            setup["mkl_float64"] = time.perf_counter()-start
        start = time.perf_counter()
        scalars["tigre_float32"] = TigreScalar(scan)
        setup["tigre_float32"] = time.perf_counter()-start
        rng = np.random.default_rng(20261028)
        result = dict(grid=list(scan.grid.shape), rays=matrix.shape[0], views=views,
                      detector_shape=list(scan.detector_shape), nnz=matrix.nnz,
                      scalar_csr_bytes=sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr)),
                      geometry_seconds=geometry_seconds, scalar_setup_seconds=setup,
                      operations={}, gpu_names=scalars["tigre_float32"].gpu_names)
        if mkl_library:
            result["mkl"] = dict(version=runtime.version, threads=runtime.lib.MKL_Get_Max_Threads())
        for kind in ("scalar", "lrt", "trt"):
            print(f"Timing {kind}", file=sys.stderr, flush=True)
            ops = {name: scalar if kind == "scalar" else TensorWrapper(scalar, kind)
                   for name, scalar in scalars.items()}
            domain = scan.grid.image_shape if kind == "scalar" else (6, *scan.grid.image_shape)
            shape = scan.range_shape if kind == "scalar" else ops["tigre_float32"].range_shape
            # Exactly the same float32-representable data on every backend;
            # the CPU double path promotes it, it does not get a different input.
            x = rng.normal(size=domain).astype(np.float32)
            y = rng.normal(size=shape).astype(np.float32)
            inputs = {name: (x.astype(op.dtype), y.astype(op.dtype)) for name, op in ops.items()}
            cpu_ax, cpu_by = ops["scipy_float64"].direct(x), ops["scipy_float64"].backproject(y)
            diagnostics = {}
            functions = {}
            for name, op in ops.items():
                bx, sy = inputs[name]
                ax, by = op.direct(bx), op.backproject(sy)
                diagnostics[name] = dict(forward_relative_error=relative_error(ax, cpu_ax),
                                         backprojection_vs_cpu_adjoint=relative_error(by, cpu_by),
                                         dot=dot_check(bx, sy, ax, by))
                functions[f"{name}_forward"] = lambda op=op, bx=bx: op.direct(bx)
                functions[f"{name}_backproject"] = lambda op=op, sy=sy: op.backproject(sy)
            result["operations"][kind] = dict(diagnostics=diagnostics,
                                               timings=time_operations(functions, repeat))
        return result


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--mkl-library", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.size, args.views, args.repeat) < 1:
        parser.error("size, views and repeat must be positive")
    report = dict(schema_version=1, status="started", date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  git_commit=command_output(["git", "rev-parse", "HEAD"]),
                  git_dirty=command_output(["git", "status", "--porcelain"]),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  platform=platform.platform(), expected_tigre_source_commit=TIGRE_COMMIT,
                  nvidia_smi=command_output(["nvidia-smi"]),
                  components=list(COMPONENTS), parameters=vars(args).copy(),
                  thread_environment={k: os.environ.get(k) for k in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  notes=["Experimental host-array wrapper; six scalar calls per tensor application.",
                         "TIGRE uses float32; CPU float32 and float64 controls are labelled separately.",
                         "Backprojection mismatch is reported, never fitted away or assumed zero.",
                         "Parallel views around z do not provide complete tensor inversion data.",
                         "Timings include transfers, NumPy contractions, checks and output allocation.",
                         "Geometry and one-off setup are recorded separately; this is not a solver benchmark."])
    report["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["parameters"].items()}
    exit_code = 1
    try:
        dist = importlib.metadata.distribution("tigre")
        report["tigre_version"] = dist.version
        report["tigre_install_source"] = json.loads(dist.read_text("direct_url.json") or "null")
        print("Checking complete small forward matrices and backprojection defects", file=sys.stderr, flush=True)
        report["validation"] = validate_gpu()
        if not all(case["forward_passed"] for case in report["validation"]):
            report["status"] = "forward_validation_failed"
            exit_code = 2
        else:
            if not args.validation_only:
                report["benchmark"] = benchmark(args.size, args.views, args.repeat, args.mkl_library)
            report["status"] = "completed_with_backprojection_diagnostics"
            exit_code = 0
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"{report['status']}: {args.output}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
