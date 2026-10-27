# SPDX-License-Identifier: Apache-2.0
"""Optional CIL operators for explicit xyz rays and scalar/tensor images."""
import numpy as np
from cil.framework import VectorGeometry, BlockGeometry
from cil.optimisation.operators import LinearOperator
from . import Grid, RayTransform, TensorRayTransform, TransverseRayTransform


def _image_grid(image_geometry):
    ig = image_geometry.copy()
    ig.dtype = np.float64
    if ig.channels != 1:
        raise ValueError("only single-channel ImageGeometry is supported")
    dim = 3 if ig.voxel_num_z > 0 else 2
    expected = ("vertical", "horizontal_y", "horizontal_x") if dim == 3 else ("horizontal_y", "horizontal_x")
    if tuple(ig.dimension_labels) != expected:
        raise ValueError(f"reorder image geometry to {expected}; singleton spatial axes are not supported")
    n = (ig.voxel_num_x, ig.voxel_num_y, ig.voxel_num_z)[:dim]
    h = (ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_z)[:dim]
    c = (ig.center_x, ig.center_y, ig.center_z)[:dim]
    origin = tuple(ci-ni*hi/2 for ci, ni, hi in zip(c, n, h))
    grid = Grid(n, h, origin)
    if tuple(ig.shape) != grid.image_shape:
        raise ValueError("CIL and grid image shapes disagree")
    return ig, grid


class JacobProjector(LinearOperator):
    """Matrix-free scalar ImageGeometry -> VectorGeometry projector.

    Supports single-channel 2D/3D ImageGeometry in default spatial label
    order. No AcquisitionGeometry-to-ray conversion is inferred. Geometry
    is copied and set to float64. Nonlinear transport needs its own Jacobian.
    adjoint_mode="atomic" opts into the OpenMP scalar adjoint; "serial"
    preserves the default fixed summation order. See RayTransform.
    """
    def __init__(self, image_geometry, starts, ends, *, adjoint_mode="serial"):
        ig, grid = _image_grid(image_geometry)
        self.projector = RayTransform(grid, starts, ends, adjoint_mode=adjoint_mode)
        super().__init__(domain_geometry=ig,
                         range_geometry=VectorGeometry(self.projector.shape[0], dtype=np.float64))

    def direct(self, x, out=None):
        result = self.projector.direct(x.as_array())
        if out is None:
            out = self.range_geometry().allocate()
        out.fill(result)
        return out

    def adjoint(self, x, out=None):
        result = self.projector.adjoint(x.as_array())
        if out is None:
            out = self.domain_geometry().allocate()
        out.fill(result)
        return out


class TensorJacobProjector(LinearOperator):
    """Symmetric rank-m LRT from BlockGeometry to VectorGeometry.

    Each block is one Frobenius-orthonormal tensor component. The order is
    projector.exponents (rank 2, 2D: xx, sqrt(2)xy, yy). Scalar intersections
    are cached. Componentwise regularizers can operate on the image blocks.
    The LRT's potential-field nullspace remains; inversion requires care.
    """
    def __init__(self, image_geometry, starts, ends, rank=2):
        ig, grid = _image_grid(image_geometry)
        self.projector = TensorRayTransform(grid, starts, ends, rank)
        self.ncomponents = len(self.projector.exponents)
        super().__init__(domain_geometry=BlockGeometry(*[ig.copy() for _ in range(self.ncomponents)]),
                         range_geometry=VectorGeometry(self.projector.shape[0], dtype=np.float64))

    def direct(self, x, out=None):
        if x.shape != (self.ncomponents, 1):
            raise ValueError("input block shape does not match tensor components")
        values = np.stack([x.get_item(j).as_array() for j in range(self.ncomponents)])
        result = self.projector.direct(values)
        if out is None:
            out = self.range_geometry().allocate()
        out.fill(result)
        return out

    def adjoint(self, x, out=None):
        result = self.projector.adjoint(x.as_array())
        if out is None:
            out = self.domain_geometry().allocate()
        if out.shape != (self.ncomponents, 1):
            raise ValueError("output block shape does not match tensor components")
        for j in range(self.ncomponents):
            out.get_item(j).fill(result[j])
        return out


class TransverseJacobProjector(LinearOperator):
    """Full rank-two TRT, from image blocks to ray-vector blocks.

    Both spaces have six blocks in 3D (three in 2D), ordered as the packed
    Frobenius-orthonormal Cartesian components. Each range block contains
    one component for every ray. See TransverseRayTransform for conventions.
    """
    def __init__(self, image_geometry, starts, ends):
        ig, grid = _image_grid(image_geometry)
        self.projector = TransverseRayTransform(grid, starts, ends)
        self.ncomponents = self.projector.ncomponents
        vg = VectorGeometry(self.projector.range_shape[0], dtype=np.float64)
        super().__init__(domain_geometry=BlockGeometry(*[ig.copy() for _ in range(self.ncomponents)]),
                         range_geometry=BlockGeometry(*[vg.copy() for _ in range(self.ncomponents)]))

    def direct(self, x, out=None):
        if x.shape != (self.ncomponents, 1):
            raise ValueError("input block shape does not match tensor components")
        values = np.stack([x.get_item(j).as_array() for j in range(self.ncomponents)])
        result = self.projector.direct(values)
        if out is None:
            out = self.range_geometry().allocate()
        if out.shape != (self.ncomponents, 1):
            raise ValueError("output block shape does not match tensor components")
        for j in range(self.ncomponents):
            out.get_item(j).fill(result[:, j])
        return out

    def adjoint(self, x, out=None):
        if x.shape != (self.ncomponents, 1):
            raise ValueError("input block shape does not match tensor components")
        values = np.column_stack([x.get_item(j).as_array() for j in range(self.ncomponents)])
        result = self.projector.adjoint(values)
        if out is None:
            out = self.domain_geometry().allocate()
        if out.shape != (self.ncomponents, 1):
            raise ValueError("output block shape does not match tensor components")
        for j in range(self.ncomponents):
            out.get_item(j).fill(result[j])
        return out
