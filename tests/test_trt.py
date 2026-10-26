# SPDX-License-Identifier: Apache-2.0
"""Rank-two TRT against explicit physical matrices, not packed weight formulas."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from jacobsrays import Grid, TransverseRayTransform, TensorRayTransform


def pack(matrices):
    dim = matrices.shape[-1]
    return np.stack([matrices[..., i, j]*(1 if i == j else np.sqrt(2))
                     for i in range(dim) for j in range(i, dim)], axis=-1)


def unpack(values, dim):
    result = np.zeros((*values.shape[:-1], dim, dim))
    for k, (i, j) in enumerate((i, j) for i in range(dim) for j in range(i, dim)):
        result[..., i, j] = result[..., j, i] = values[..., k]/(1 if i == j else np.sqrt(2))
    return result


class TRTTests(unittest.TestCase):
    def test_analytic_constant_and_polarizations(self):
        # A central diagonal chord of a unit cube, with all six tensor entries.
        grid = Grid((2, 3, 4), (.5, 1/3, .25))
        op = TransverseRayTransform(grid, [-1, -1, -1], [2, 2, 2])
        F = np.array([[2., .3, -.7], [.3, -1., .4], [-.7, .4, 3.]])
        x = np.broadcast_to(pack(F)[:, None, None, None], op.domain_shape)
        u = np.ones(3)/np.sqrt(3)
        P = np.eye(3)-np.outer(u, u)
        result = unpack(op.direct(x), 3)[0]
        assert_allclose(result, np.sqrt(3)*P@F@P, atol=2e-14)
        assert_allclose(result@u, 0, atol=2e-14)
        # An orthonormal detector frame: three independent entries retain
        # both transverse diagonals AND their off-diagonal coupling.
        a = np.array([1., -1., 0.])/np.sqrt(2)
        b = np.cross(u, a)
        Q = np.column_stack((a, b))
        H = Q.T@result@Q
        assert_allclose(H, np.sqrt(3)*Q.T@F@Q, atol=2e-14)
        assert_allclose(np.linalg.norm(pack(H)), np.linalg.norm(pack(result)), atol=2e-14)
        for angle in (.2, .8, 1.4):
            q = a*np.cos(angle)+b*np.sin(angle)
            assert_allclose(q@result@q, np.sqrt(3)*(q@F@q), atol=2e-14)
        # The full TRT retains the transverse trace (unlike trace-free TRT).
        identity = np.broadcast_to(pack(np.eye(3))[:, None, None, None], op.domain_shape)
        assert_allclose(unpack(op.direct(identity), 3)[0], np.sqrt(3)*P, atol=2e-14)
        # Longitudinal and mixed longitudinal/transverse tensors vanish for this ray.
        for F in (np.outer(u, u), np.outer(u, a)+np.outer(a, u)):
            x = np.broadcast_to(pack(F)[:, None, None, None], op.domain_shape)
            assert_allclose(op.direct(x), 0, atol=2e-14)

    def test_voxel_oracle_adjoint_and_scipy(self):
        rng = np.random.default_rng(213)
        for dim in (2, 3):
            grid = Grid((3, 4, 2)[:dim], (.3, .7, 1.1)[:dim], (-.2, -1.3, .5)[:dim])
            lo = np.array(grid.origin)-1
            hi = np.array(grid.origin)+np.array(grid.shape)*grid.spacing+1
            starts, ends = rng.uniform(lo, hi, (2, 37, dim))
            op = TransverseRayTransform(grid, starts, ends)
            x = rng.normal(size=op.domain_shape)
            y = rng.normal(size=op.range_shape)  # Includes non-transverse data.
            tensors = unpack(x.reshape(op.ncomponents, -1).T, dim)
            # Independent geometry: slab-clip every ray against every voxel.
            expected = []
            for s, e in zip(starts, ends):
                d = e-s
                u = d/np.linalg.norm(d)
                P = np.eye(dim)-np.outer(u, u)
                total = np.zeros((dim, dim))
                for v, F in enumerate(tensors):
                    cell = np.array(np.unravel_index(v, grid.image_shape))[::-1]
                    low = np.array(grid.origin)+cell*grid.spacing
                    high = low+grid.spacing
                    t = np.stack(((low-s)/d, (high-s)/d))
                    length = max(0, min(1, t.max(axis=0).min())-max(0, t.min(axis=0).max()))*np.linalg.norm(d)
                    total += length*(P@F@P)
                expected.append(pack(total))
            assert_allclose(op.direct(x), expected, atol=2e-13)
            assert_allclose(np.vdot(op.direct(x), y), np.vdot(x, op.adjoint(y)), atol=2e-13)
            assert_allclose(op.as_scipy()@x.ravel(), op.direct(x).ravel(), atol=1e-14)
            assert_allclose(op.as_scipy().rmatvec(y.ravel()), op.adjoint(y).ravel(), atol=1e-14)
            reverse = TransverseRayTransform(grid, ends, starts)
            assert_allclose(reverse.direct(x), op.direct(x), atol=2e-13)
            # tr(P f P) = tr(f) - u^T f u, after integration.
            trace_image = np.trace(tensors, axis1=1, axis2=2).reshape(grid.image_shape)
            lrt = TensorRayTransform(grid, starts, ends, rank=2)
            result = unpack(op.direct(x), dim)
            assert_allclose(np.trace(result, axis1=1, axis2=2),
                            op.scalar.direct(trace_image)-lrt.direct(x), atol=2e-13)

    def test_covariance_under_axis_permutation(self):
        rng = np.random.default_rng(3)
        grid = Grid((4, 4, 4), origin=(-2, -2, -2))
        starts, ends = rng.uniform(-3, 3, (2, 25, 3))
        op = TransverseRayTransform(grid, starts, ends)
        F = rng.normal(size=(4, 4, 4, 3, 3))
        F = (F+F.swapaxes(-1, -2))/2
        x = np.moveaxis(pack(F), -1, 0)
        # Swap physical x and z: transform both the grid field and its indices.
        perm = [2, 1, 0]
        rotated = F.transpose(2, 1, 0, 3, 4)[..., perm, :][..., :, perm]
        other = TransverseRayTransform(grid, starts[:, perm], ends[:, perm])
        result = unpack(op.direct(x), 3)
        assert_allclose(unpack(other.direct(np.moveaxis(pack(rotated), -1, 0)), 3),
                        result[:, perm, :][:, :, perm], atol=1e-13)

    def test_empty_miss_degenerate_and_invalid(self):
        grid = Grid((2, 3, 4))
        for starts, ends in [(np.empty((0, 3)), np.empty((0, 3))),
                             ([[10, 10, 10], [1, 1, 1]], [[11, 11, 11], [1, 1, 1]])]:
            op = TransverseRayTransform(grid, starts, ends)
            assert_array_equal(op.direct(np.ones(op.domain_shape)), np.zeros(op.range_shape))
            assert_array_equal(op.adjoint(np.ones(op.range_shape)), np.zeros(op.domain_shape))
            self.assertEqual(op.as_scipy().shape, op.shape)
        op = TransverseRayTransform(grid, [0, 0, 0], [2, 3, 4])
        with self.assertRaises(ValueError): op.direct(np.ones(grid.image_shape))
        with self.assertRaises(ValueError): op.direct(np.ones(op.domain_shape, complex))
        with self.assertRaises(ValueError): op.adjoint(np.ones(6))
        with self.assertRaises(ValueError): op.adjoint(np.ones(op.range_shape, complex))
        with self.assertRaises(ValueError): TransverseRayTransform(grid, [np.nan, 0, 0], [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
