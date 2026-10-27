"""Matched transpose and conflicting-write regressions for the atomic adjoint."""
import importlib.util
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from jacobsrays import Grid, RayTransform, _native


class AtomicAdjointTests(unittest.TestCase):
    def test_mode_validation(self):
        grid = Grid((4, 3))
        starts, ends = [[-1, .5]], [[5, .5]]
        with self.assertRaises(ValueError):
            RayTransform(grid, starts, ends, adjoint_mode="unknown")
        if not _native.openmp_enabled:
            with self.assertRaisesRegex(RuntimeError, "OpenMP build"):
                RayTransform(grid, starts, ends, adjoint_mode="atomic")
            op = RayTransform(grid, starts, ends)
            with self.assertRaisesRegex(RuntimeError, "OpenMP build"):
                _native.backproject_atomic(op._starts, op._ends, *op._geometry, np.ones(1))

    @unittest.skipUnless(_native.openmp_enabled, "requires an OpenMP build")
    def test_colliding_rays(self):
        # All threads update the same four voxels. Integer contributions make
        # any lost update detectable without a floating-point tolerance.
        # Segment length eight makes every plane parameter exactly binary.
        grid = Grid((4, 3, 2))
        count = 16384
        starts = np.tile([-2., 1.5, .5], (count, 1))
        ends = np.tile([6., 1.5, .5], (count, 1))
        op = RayTransform(grid, starts, ends, adjoint_mode="atomic")
        rng = np.random.default_rng(61)
        for data in (np.ones(count), rng.integers(-5, 6, count).astype(float)):
            expected = np.zeros(grid.image_shape)
            expected[0, 1, :] = data.sum()
            for _ in range(3):
                assert_array_equal(op.adjoint(data), expected)

    @unittest.skipUnless(_native.openmp_enabled, "requires an OpenMP build")
    def test_random_geometry_and_transpose(self):
        rng = np.random.default_rng(62)
        for dim in (2, 3):
            grid = Grid((7, 5, 3)[:dim], (.3, .7, 1.1)[:dim], (-1, -2, -.5)[:dim])
            starts, ends = rng.uniform(-3, 4, (2, 2048, dim))
            serial = RayTransform(grid, starts, ends)
            atomic = RayTransform(grid, starts, ends, adjoint_mode="atomic")
            matrix = serial.paths().to_csr()
            x = rng.normal(size=grid.image_shape)
            y = rng.normal(size=len(starts))
            expected = (matrix.T @ y).reshape(grid.image_shape)
            assert_allclose(serial.adjoint(y), expected, rtol=2e-13, atol=2e-13)
            assert_array_equal(atomic.direct(x), serial.direct(x))
            ax = atomic.direct(x)
            for _ in range(3):
                aty = atomic.adjoint(y)
                assert_allclose(aty, expected, rtol=2e-13, atol=2e-13)
                scale = np.linalg.norm(ax)*np.linalg.norm(y) + np.linalg.norm(x)*np.linalg.norm(aty)
                self.assertLess(abs(np.vdot(ax, y)-np.vdot(x, aty)), 2e-14*scale)
            assert_allclose(atomic.as_scipy().rmatvec(y), expected.ravel(), rtol=2e-13, atol=2e-13)

    @unittest.skipUnless(_native.openmp_enabled, "requires an OpenMP build")
    def test_empty_miss_and_degenerate(self):
        grid = Grid((4, 3))
        for starts, ends in ((np.empty((0, 2)), np.empty((0, 2))),
                             ([[-1, 4], [.5, .5]], [[5, 4], [.5, .5]])):
            op = RayTransform(grid, starts, ends, adjoint_mode="atomic")
            assert_array_equal(op.adjoint(np.ones(len(starts))), np.zeros(grid.image_shape))

    @unittest.skipUnless(_native.openmp_enabled and importlib.util.find_spec("cil"),
                         "requires OpenMP and CIL")
    def test_cil_adapter_and_cgls(self):
        from cil.framework import ImageGeometry
        from cil.optimisation.algorithms import CGLS
        from jacobsrays.cil import JacobProjector
        rng = np.random.default_rng(63)
        ig = ImageGeometry(voxel_num_x=4, voxel_num_y=3, dtype=np.float64)
        starts, ends = rng.uniform(-4, 4, (2, 256, 2))
        op = JacobProjector(ig, starts, ends, adjoint_mode="atomic")
        serial = JacobProjector(ig, starts, ends)
        x, y = op.domain_geometry().allocate(), op.range_geometry().allocate()
        x.fill(rng.normal(size=ig.shape))
        y.fill(rng.normal(size=len(starts)))
        out = op.domain_geometry().allocate()
        self.assertIs(op.adjoint(y, out=out), out)
        assert_allclose(out.as_array(), serial.adjoint(y).as_array(), rtol=2e-13, atol=2e-13)
        ax = op.direct(x)
        scale = ax.norm()*y.norm() + x.norm()*out.norm()
        self.assertLess(abs(ax.dot(y)-x.dot(out)), 2e-14*scale)
        solver = CGLS(initial=op.domain_geometry().allocate(0), operator=op, data=ax)
        solver.run(10, verbose=0)
        self.assertLess((op.direct(solver.solution)-ax).norm(), .01*ax.norm())
