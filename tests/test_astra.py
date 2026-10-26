# SPDX-License-Identifier: Apache-2.0
"""Independent scalar X-ray checks against ASTRA's 2D CPU line projectors.

ASTRA is a test dependency only. No reference geometry or weights are taken
from jacobsrays. See docs/validation.md for conventions and scope.
"""
import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_allclose

from jacobsrays import Grid, RayTransform


HAS_ASTRA = importlib.util.find_spec('astra') is not None
HAS_CIL = importlib.util.find_spec('cil') is not None
# ASTRA uses float32 geometry/weights; near-axis division amplifies roundoff.
# Fixed bounds: 3e-4 pixel widths per coefficient, 2e-5 relative L2. No fitted
# normalization, omitted coefficients, or adjustment of the physical geometry.
COEFFICIENT_ATOL = 3e-4
RELATIVE_L2_TOL = 2e-5


def relative_error(actual, reference):
    return np.linalg.norm(actual-reference) / np.linalg.norm(reference)


@unittest.skipUnless(HAS_ASTRA, 'ASTRA is not installed')
class ASTRATests(unittest.TestCase):
    def compare_matrix(self, kind):
        import astra

        rng = np.random.default_rng(51)
        # Rectangular images, translated origins, non-unit physical lengths.
        # Square pixels: ASTRA's line kernels are not our anisotropic oracle.
        for spacing in (1., .375, 2.5):
            with self.subTest(kind=kind, spacing=spacing):
                grid = Grid((11, 8), (spacing, spacing),
                            (-1.73*spacing, -2.19*spacing))
                low = np.array(grid.origin)
                size = np.array(grid.shape)*grid.spacing
                high = low+size
                centre = (low+high)/2
                radius = 3*np.linalg.norm(size)
                angles = rng.uniform(-np.pi, np.pi, 29)
                direction = np.column_stack((np.sin(angles), -np.cos(angles)))
                pixel_step = np.column_stack((np.cos(angles), np.sin(angles)))*(.83*spacing)
                source = centre+radius*direction
                detector = centre+pixel_step*.137
                if kind == 'fanflat_vec':
                    detector = detector-radius*direction
                vectors = np.column_stack((direction if kind == 'parallel_vec' else source,
                                           detector, pixel_step))
                count = 41
                pixels = detector[:, None, :] + (np.arange(count)-(count-1)/2)[None, :, None]*pixel_step[:, None, :]
                if kind == 'parallel_vec':
                    starts = pixels-radius*direction[:, None, :]
                    ends = pixels+radius*direction[:, None, :]
                else:
                    starts = np.broadcast_to(source[:, None, :], pixels.shape)
                    ends = pixels
                ours = RayTransform(grid, starts.reshape(-1, 2), ends.reshape(-1, 2))
                vol = astra.create_vol_geom(grid.shape[1], grid.shape[0],
                                           low[0], high[0], low[1], high[1])
                proj = astra.create_proj_geom(kind, count, vectors)
                kernel = 'line' if kind == 'parallel_vec' else 'line_fanflat'
                projector_id = astra.create_projector(kernel, proj, vol)
                self.addCleanup(astra.projector.delete, projector_id)
                self.assertFalse(astra.projector.is_cuda(projector_id))
                matrix_id = astra.projector.matrix(projector_id)
                self.addCleanup(astra.matrix.delete, matrix_id)
                matrix = astra.matrix.get(matrix_id)
                # Native ASTRA rows descend in physical y. Our rows ascend.
                permutation = np.arange(np.prod(grid.shape)).reshape(grid.image_shape)[::-1].ravel()
                matrix = matrix[:, permutation]
                expected = matrix.toarray()
                actual = ours.paths().to_csr().toarray()
                assert_allclose(actual, expected, rtol=0,
                                atol=COEFFICIENT_ATOL*spacing)
                self.assertLess(relative_error(actual, expected), RELATIVE_L2_TOL)
                # Full matrix comparison checks every individual voxel impulse;
                # also exercise our matrix-free forward and adjoint kernels.
                x = rng.normal(size=grid.image_shape)
                y = rng.normal(size=matrix.shape[0])
                forward_error = relative_error(ours.direct(x), matrix@x.ravel())
                adjoint_error = relative_error(ours.adjoint(y).ravel(), matrix.T@y)
                self.assertLess(forward_error, RELATIVE_L2_TOL)
                self.assertLess(adjoint_error, RELATIVE_L2_TOL)
                # Confirm that a missing y flip would fail for these geometries.
                wrong_order = matrix[:, permutation].toarray()
                self.assertGreater(relative_error(actual, wrong_order), .1)
                print(f'ASTRA {kind}, pixel={spacing:g}: '
                      f'max coefficient error/pixel={np.max(np.abs(actual-expected))/spacing:.3g}, '
                      f'forward rel L2={forward_error:.3g}, adjoint rel L2={adjoint_error:.3g}')

    def test_parallel_matrix(self):
        self.compare_matrix('parallel_vec')

    def test_fan_matrix(self):
        self.compare_matrix('fanflat_vec')

    def test_axis_lengths_and_orientation(self):
        import astra

        # Two horizontal rays through cell centres. This has exactly known
        # physical weights and independently anchors the ASTRA y convention.
        grid = Grid((3, 2), (.5, .5), (1., 2.))
        starts = np.array([[0., 2.25], [0., 2.75]])
        ends = np.array([[4., 2.25], [4., 2.75]])
        vectors = np.array([[1., 0., 0., 2.25, 0., 1.],
                            [1., 0., 0., 2.75, 0., 1.]])
        proj = astra.create_proj_geom('parallel_vec', 1, vectors)
        vol = astra.create_vol_geom(2, 3, 1., 2.5, 2., 3.)
        projector_id = astra.create_projector('line', proj, vol)
        self.addCleanup(astra.projector.delete, projector_id)
        matrix_id = astra.projector.matrix(projector_id)
        self.addCleanup(astra.matrix.delete, matrix_id)
        native = astra.matrix.get(matrix_id).toarray()
        expected = np.array([[.5, .5, .5, 0, 0, 0], [0, 0, 0, .5, .5, .5]])
        assert_allclose(native[:, [3, 4, 5, 0, 1, 2]], expected, atol=1e-7, rtol=0)
        assert_allclose(RayTransform(grid, starts, ends).paths().to_csr().toarray(),
                        expected, atol=1e-14, rtol=0)


@unittest.skipUnless(HAS_ASTRA and HAS_CIL, 'ASTRA and CIL are required')
class CILASTRATests(unittest.TestCase):
    def test_cil_forward_and_adjoint(self):
        import astra
        from cil.framework import AcquisitionGeometry, ImageGeometry
        from cil.plugins.astra import ProjectionOperator
        from jacobsrays.cil import JacobProjector

        rng = np.random.default_rng(52)
        # Centred CIL image; native tests above exercise translated windows.
        ig = ImageGeometry(11, 8, voxel_size_x=.7, voxel_size_y=.7)
        for kind in ('parallel', 'fan'):
            for origin in ('bottom-left', 'bottom-right'):
                with self.subTest(kind=kind, origin=origin):
                    if kind == 'parallel':
                        ag = AcquisitionGeometry.create_Parallel2D()
                    else:
                        ag = AcquisitionGeometry.create_Cone2D([0, -30], [0, 30])
                    ag.set_angles([7.3, 39.7, 86.1, 128.2, 178.9, 241.6], initial_angle=2.4)
                    ag.set_panel(41, pixel_size=.581, origin=origin)
                    # Construct rays independently of CIL's ASTRA converter.
                    # Positive object rotation is clockwise source/detector
                    # rotation in the object's coordinate frame.
                    theta = ag.get_angles('radian', apply_offset=True).astype(float)
                    c, s = np.cos(theta), np.sin(theta)
                    horizontal = np.column_stack((c, -s))
                    direction = np.column_stack((s, c))
                    offsets = (np.arange(41)-20)*.581
                    if origin == 'bottom-right':
                        offsets = -offsets
                    points = offsets[None, :, None]*horizontal[:, None, :]
                    ends = points+30*direction[:, None, :]
                    if kind == 'parallel':
                        starts = points-30*direction[:, None, :]
                    else:
                        starts = np.broadcast_to(-30*direction[:, None, :], points.shape)
                    ours = JacobProjector(ig, starts.reshape(-1, 2), ends.reshape(-1, 2))
                    reference = ProjectionOperator(ig, ag, device='cpu')
                    self.addCleanup(astra.projector.delete,
                                    [reference.operator.fp.proj_id, reference.operator.bp.proj_id])
                    x = ig.allocate()
                    x.fill(rng.normal(size=ig.shape))
                    y = ag.allocate()
                    y.fill(rng.normal(size=ag.shape))
                    flat_y = ours.range_geometry().allocate()
                    flat_y.fill(y.as_array().ravel())
                    forward_error = relative_error(ours.direct(x).as_array(),
                                                   reference.direct(x).as_array().ravel())
                    adjoint_error = relative_error(ours.adjoint(flat_y).as_array(),
                                                   reference.adjoint(y).as_array())
                    self.assertLess(forward_error, RELATIVE_L2_TOL)
                    self.assertLess(adjoint_error, RELATIVE_L2_TOL)
                    print(f'CIL/ASTRA {kind}, {origin}: forward rel L2={forward_error:.3g}, '
                          f'adjoint rel L2={adjoint_error:.3g}')


if __name__ == '__main__':
    unittest.main()
