# SPDX-License-Identifier: Apache-2.0
"""CPU algebra/geometry tests only; real CUDA validation is an explicit job."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from jacobsrays import trace, TensorRayTransform, TransverseRayTransform


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "tigre_tensor.py"
spec = importlib.util.spec_from_file_location("tigre_tensor", BENCHMARK)
prototype = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prototype
spec.loader.exec_module(prototype)
ParallelScan, MatrixScalar, TensorWrapper = (prototype.ParallelScan,
                                           prototype.MatrixScalar, prototype.TensorWrapper)


class SingleComponentScalar:
    """Exercise the six-call path without claiming to emulate TIGRE/CUDA."""
    def __init__(self, scalar):
        self.scan, self.dtype = scalar.scan, scalar.dtype
        self.direct, self.backproject = scalar.direct, scalar.backproject


class TigrePrototypeTests(unittest.TestCase):
    def test_report_roundtrip_with_cpu_stand_in(self):
        # Check the complete reporting path using a CPU stand-in. This does
        # not validate any TIGRE kernel and cannot supply GPU timings.
        class CPUStandIn(MatrixScalar):
            def __init__(self, scan):
                super().__init__(scan, trace(scan.grid, *scan.rays()).to_csr().astype(np.float32), np.float32)
                self.gpu_names = ["CPU test stand-in; no CUDA execution"]
        from types import SimpleNamespace
        dist = SimpleNamespace(version="CPU-test-only", read_text=lambda name: None)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(prototype, "TigreScalar", CPUStandIn), \
                patch.object(prototype.importlib.metadata, "distribution", return_value=dist):
            output = Path(temp)/"result.json"
            self.assertEqual(prototype.main(["--size", "3", "--views", "3", "--repeat", "1",
                                             "--output", str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertTrue(all(v["forward_passed"] for v in result["validation"]))
            self.assertIn("trt", result["benchmark"]["operations"])
            # A real geometry failure must save the diagnostics and avoid timings.
            bad = [{"forward_passed": False, "forward_matrix_relative_error": 1.}]
            with patch.object(prototype, "validate_gpu", return_value=bad):
                self.assertEqual(prototype.main(["--output", str(output)]), 2)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "forward_validation_failed")
            self.assertNotIn("benchmark", result)
        self.assertIsNone(prototype.relative_error([1.], [0.]))

    def test_geometry_axis_directions_and_pixel_order(self):
        scan = ParallelScan.create((4, 6, 3), spacing=(.75, 1.25, 1.5), views=2)
        scan.angles = np.array([0., np.pi/2])
        starts, ends = scan.rays()
        shape = (*scan.range_shape, 3)
        starts, ends = starts.reshape(shape), ends.reshape(shape)
        mid = (starts+ends)/2
        nv, nu = scan.detector_shape
        v = (np.arange(nv)-(nv-1)/2)*scan.detector_spacing[0]
        u = (np.arange(nu)-(nu-1)/2)*scan.detector_spacing[1]
        assert_allclose(mid[0, :, :, 0], 0)
        assert_allclose(mid[0, :, :, 1], np.broadcast_to(u, (nv, nu)))
        assert_allclose(mid[0, :, :, 2], np.broadcast_to(v[:, None], (nv, nu)))
        assert_allclose(mid[1, :, :, 0], -np.broadcast_to(u, (nv, nu)), atol=1e-14)
        assert_allclose(mid[1, :, :, 1], 0, atol=1e-14)
        directions = (ends-starts).reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        assert_allclose(directions, scan.directions, atol=1e-15)
        # Independent analytic lengths for a rectangular box at angle zero.
        a = trace(scan.grid, starts.reshape(-1, 3), ends.reshape(-1, 3)).to_csr()
        lengths = (a @ np.ones(np.prod(scan.grid.shape))).reshape(scan.range_shape)
        expected = np.where(np.abs(u) < 6*1.25/2, 4*.75, 0.)
        assert_allclose(lengths[0], np.broadcast_to(expected, (nv, nu)), atol=1e-14)

    def test_wrappers_against_existing_operators_and_adjoint(self):
        rng = np.random.default_rng(42)
        scan = ParallelScan.create((5, 4, 3), spacing=(.75, 1.25, 1.5), views=7)
        starts, ends = scan.rays()
        a = trace(scan.grid, starts, ends).to_csr()
        for dtype, tolerance in ((np.float64, 2e-12), (np.float32, 3e-6)):
            scalar = MatrixScalar(scan, a.astype(dtype), dtype)
            for kind, reference_class in (("lrt", TensorRayTransform), ("trt", TransverseRayTransform)):
                reference = reference_class(scan.grid, starts, ends)
                for backend in (scalar, SingleComponentScalar(scalar)):
                    op = TensorWrapper(backend, kind)
                    x = rng.normal(size=op.domain_shape).astype(dtype)
                    y = rng.normal(size=op.range_shape).astype(dtype)  # arbitrary TRT data, not pre-projected
                    ax, by = op.direct(x), op.backproject(y)
                    self.assertEqual(ax.dtype, np.dtype(dtype))
                    self.assertEqual(by.dtype, np.dtype(dtype))
                    assert_allclose(ax, reference.direct(x), atol=tolerance, rtol=tolerance)
                    assert_allclose(by, reference.adjoint(y), atol=tolerance, rtol=tolerance)
                    defect = prototype.dot_check(x, y, ax, by)["scaled_defect"]
                    self.assertLess(defect, tolerance)

    def test_physical_constant_tensor_and_trace(self):
        scan = ParallelScan.create((4, 5, 3), views=5)
        a = trace(scan.grid, *scan.rays()).to_csr()
        scalar = MatrixScalar(scan, a)
        lrt, trt = TensorWrapper(scalar, "lrt"), TensorWrapper(scalar, "trt")
        f = np.array([[2., .3, -.7], [.3, -1., .4], [-.7, .4, 3.]])
        packed = np.array([2, np.sqrt(2)*.3, -np.sqrt(2)*.7, -1, np.sqrt(2)*.4, 3])
        x = np.broadcast_to(packed[:, None, None, None], trt.domain_shape)
        lengths = scalar.direct(np.ones(scan.grid.image_shape)).ravel()
        u = scan.directions
        p = np.eye(3)[None] - u[:, :, None]*u[:, None, :]
        expected = lengths[:, None, None]*(p @ f @ p)
        values = trt.direct(x)
        # Unpack independently, including the off-diagonal sqrt(2) factors.
        actual = np.zeros_like(expected)
        for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            actual[:, i, j] = actual[:, j, i] = values[:, k]/(1 if i == j else np.sqrt(2))
        assert_allclose(actual, expected, atol=2e-14)
        assert_allclose(np.einsum("rij,rj->ri", actual, u), 0, atol=2e-14)
        assert_allclose(np.trace(actual, axis1=1, axis2=2), np.trace(f)*lengths-lrt.direct(x), atol=2e-14)

    def test_invalid_input_and_diagnostic_mismatch(self):
        scan = ParallelScan.create((2, 3, 4), views=3)
        scalar = MatrixScalar(scan, trace(scan.grid, *scan.rays()).to_csr())
        op = TensorWrapper(scalar, "trt")
        for bad in (np.ones((6, 2, 3, 4)), np.ones(op.domain_shape, complex),
                    np.full(op.domain_shape, np.nan)):
            with self.assertRaises(ValueError):
                op.direct(bad)
        with self.assertRaises(ValueError):
            op.backproject(np.ones(6))
        # An approximate backprojector must remain visible in the diagnostics.
        x, y = np.ones(op.domain_shape), np.ones(op.range_shape)
        ax, by = op.direct(x), op.backproject(y)
        self.assertGreater(prototype.dot_check(x, y, ax, 2*by)["scaled_defect"], .01)

    def test_import_does_not_load_tigre_and_missing_gpu_fails(self):
        code = ("import runpy, sys; runpy.run_path(sys.argv[1]); "
                "assert 'tigre' not in sys.modules; "
                "import jacobsrays; assert 'tigre' not in sys.modules")
        subprocess.run([sys.executable, "-c", code, str(BENCHMARK)], check=True)
        # Exercise fail-closed behaviour without pretending this is a GPU test.
        class NoDevices:
            def __len__(self):
                return 0
        from types import SimpleNamespace
        fake_gpu = SimpleNamespace(GpuIds=NoDevices)
        with patch.dict(sys.modules, {"tigre": SimpleNamespace(),
                                      "tigre.utilities": SimpleNamespace(gpu=fake_gpu)}):
            with self.assertRaisesRegex(RuntimeError, "no visible CUDA GPU"):
                prototype.TigreScalar(ParallelScan.create((2, 2, 2), views=2))


if __name__ == "__main__":
    unittest.main()
