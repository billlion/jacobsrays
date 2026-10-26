# SPDX-License-Identifier: Apache-2.0
"""Ordered ray traversal. Geometry is xyz; NumPy images are (..., y, x)."""
from dataclasses import dataclass
import math
import numpy as np
from . import _native

__version__ = "0.1.0"
__all__ = ["Grid", "Paths", "trace", "RayTransform", "TensorRayTransform", "TransverseRayTransform"]


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class Grid:
    """Cartesian cells with shape/spacing/origin in (x,y[,z]) order.

    origin is the lower corner, not the centre. The default is the zero
    corner. NumPy image_shape reverses shape; MATLAB uses shape directly.
    """
    shape: tuple
    spacing: tuple = None
    origin: tuple = None

    def __post_init__(self):
        n = _real(self.shape, "shape")
        if n.ndim != 1 or n.size not in (2, 3) or not np.isfinite(n).all() or np.any(n < 1) or np.any(n > 2**52) or np.any(n != np.floor(n)):
            raise ValueError("shape must contain two or three positive integers")
        shape = tuple(int(x) for x in n)
        if math.prod(shape) > np.iinfo(np.int64).max:
            raise ValueError("grid size overflows int64")
        h = np.ones(n.size) if self.spacing is None else _real(self.spacing, "spacing")
        o = np.zeros(n.size) if self.origin is None else _real(self.origin, "origin")
        if h.shape != n.shape or o.shape != n.shape or not np.isfinite(h).all() or not np.isfinite(o).all() or np.any(h <= 0):
            raise ValueError("spacing/origin must match shape; spacing must be positive and finite")
        with np.errstate(over="ignore", invalid="ignore"):
            hi = o + n*h
        if not np.isfinite(hi).all() or np.any(o+h == o) or np.any(hi-h == hi):
            raise ValueError("grid spacing/extent is not representable")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", tuple(h))
        object.__setattr__(self, "origin", tuple(o))

    @property
    def image_shape(self):
        return self.shape[::-1]

    @property
    def ndim(self):
        return len(self.shape)

    def _vectors(self):
        if self.ndim == 2:
            return (np.array((*self.shape, 1), dtype=np.int64),
                    np.array((*self.spacing, 1.)), np.array((*self.origin, 0.)))
        return np.array(self.shape, dtype=np.int64), np.array(self.spacing), np.array(self.origin)


def _rays(grid, starts, ends):
    s, e = _real(starts, "starts"), _real(ends, "ends")
    if s.ndim == 1:
        s = s[None, :]
    if e.ndim == 1:
        e = e[None, :]
    if s.ndim != 2 or s.shape != e.shape or s.shape[1] != grid.ndim:
        raise ValueError("starts and ends must have matching (number_of_rays, dimension) shapes")
    if not np.isfinite(s).all() or not np.isfinite(e).all():
        raise ValueError("ray endpoints must be finite")
    if grid.ndim == 2:
        s = np.column_stack((s, np.full(len(s), .5)))
        e = np.column_stack((e, np.full(len(e), .5)))
    return np.ascontiguousarray(s), np.ascontiguousarray(e)


@dataclass(frozen=True)
class Paths:
    """Ragged ordered intervals. offsets[r]:offsets[r+1] selects ray r.

    indices are zero-based x-fastest; lengths are physical lengths.
    t_enter/t_exit parameterise start+t*(end-start), not arclength.
    Sparse conversion may reorder columns: use these arrays for ODEs.
    """
    grid: Grid
    offsets: np.ndarray
    indices: np.ndarray
    lengths: np.ndarray
    t_enter: np.ndarray
    t_exit: np.ndarray

    @property
    def nrays(self):
        return len(self.offsets)-1

    def ray(self, r):
        if not isinstance(r, (int, np.integer)) or r < 0 or r >= self.nrays:
            raise IndexError("ray index out of range")
        return slice(int(self.offsets[r]), int(self.offsets[r+1]))

    def to_csr(self):
        """SciPy scalar system matrix, with exact shape including misses."""
        from scipy.sparse import csr_matrix
        return csr_matrix((self.lengths.copy(), self.indices.copy(), self.offsets.copy()),
                          shape=(self.nrays, math.prod(self.grid.shape)))


def trace(grid, starts, ends):
    """Intersect finite segments with the grid, in start-to-end order."""
    s, e = _rays(grid, starts, ends)
    return Paths(grid, *_native.trace(s, e, *grid._vectors()))


class RayTransform:
    """Real scalar, matrix-free projector and Euclidean transpose.

    No intersections are retained. Each application retraces rays. Use
    paths().to_csr() when caching is preferable for repeated solves.
    """
    def __init__(self, grid, starts, ends):
        self.grid = grid
        self._starts, self._ends = (a.copy() for a in _rays(grid, starts, ends))
        self._geometry = grid._vectors()
        self.shape = (len(self._starts), math.prod(grid.shape))
        for a in (self._starts, self._ends, *self._geometry):
            a.flags.writeable = False

    def paths(self):
        return Paths(self.grid, *_native.trace(self._starts, self._ends, *self._geometry))

    def direct(self, image):
        x = _real(image, "image")
        if x.shape != self.grid.image_shape:
            raise ValueError(f"image must have shape {self.grid.image_shape}")
        return _native.project(self._starts, self._ends, *self._geometry, np.ascontiguousarray(x).ravel())

    def adjoint(self, data):
        y = _real(data, "data")
        if y.shape != (self.shape[0],):
            raise ValueError(f"data must have shape {(self.shape[0],)}")
        return _native.backproject(self._starts, self._ends, *self._geometry, y).reshape(self.grid.image_shape)

    def as_scipy(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.shape, dtype=np.float64,
            matvec=lambda x: self.direct(np.asarray(x).reshape(self.grid.image_shape)),
            rmatvec=lambda y: self.adjoint(np.asarray(y).reshape(-1)).ravel())


def _powers(rank, dimension):
    if dimension == 1:
        yield (rank,)
    else:
        for k in range(rank, -1, -1):
            for rest in _powers(rank-k, dimension-1):
                yield (k, *rest)


class TensorRayTransform:
    """Cached symmetric rank-m LRT in an orthonormal component basis.

    Multi-indices alpha are listed in exponents. Store sqrt(m!/alpha!)
    times each physical symmetric component; then Euclidean component dot
    products equal Frobenius products. Rank two in 2D: (xx, sqrt(2)xy, yy).
    Input shape: (number_of_components, *grid.image_shape).
    Requires SciPy; caches scalar paths, not a full tensor system matrix.
    """
    def __init__(self, grid, starts, ends, rank=2):
        if not isinstance(rank, (int, np.integer)) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.grid = grid
        self.scalar = RayTransform(grid, starts, ends)
        self.matrix = self.scalar.paths().to_csr()
        self.exponents = tuple(_powers(rank, grid.ndim))
        d = self.scalar._ends[:, :grid.ndim] - self.scalar._starts[:, :grid.ndim]
        norm = np.hypot.reduce(d, axis=1)
        u = np.divide(d, norm[:, None], out=np.zeros_like(d), where=norm[:, None] != 0)
        self.weights = np.stack([math.sqrt(math.factorial(rank)/math.prod(math.factorial(k) for k in alpha)) *
                                 np.prod(u**np.array(alpha), axis=1) for alpha in self.exponents], axis=1)
        self.domain_shape = (len(self.exponents), *grid.image_shape)
        self.shape = (self.scalar.shape[0], math.prod(self.domain_shape))

    def direct(self, image):
        x = _real(image, "tensor")
        if x.shape != self.domain_shape:
            raise ValueError(f"tensor must have shape {self.domain_shape}")
        values = self.matrix @ x.reshape(len(self.exponents), -1).T
        return np.sum(values*self.weights, axis=1)

    def adjoint(self, data):
        y = _real(data, "data")
        if y.shape != (self.shape[0],):
            raise ValueError("data shape does not match ray count")
        return (self.matrix.T @ (y[:, None]*self.weights)).T.reshape(self.domain_shape)

    def as_scipy(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.shape, dtype=np.float64,
            matvec=lambda x: self.direct(np.asarray(x).reshape(self.domain_shape)),
            rmatvec=lambda y: self.adjoint(np.asarray(y).reshape(-1)).ravel())


class TransverseRayTransform:
    """Full rank-two symmetric TRT: integral of P_u f P_u along each ray.

    P_u = I - u u.T for the unit ray direction u. Input and output use the
    same Frobenius-orthonormal Cartesian basis as the rank-two LRT:
    (xx, sqrt(2)xy, sqrt(2)xz, yy, sqrt(2)yz, zz) in 3D;
    (xx, sqrt(2)xy, yy) in 2D. This retains the transverse trace.

    Input shape: (components, *grid.image_shape).
    Output shape: (rays, components); each decoded tensor annihilates u.
    In 3D the six output entries represent three independent transverse
    components, without choosing a detector frame. Adjoint uses Frobenius
    products in both spaces, with no spatial/angular quadrature weights.

    Requires SciPy. Caches one scalar CSR matrix and small per-ray
    projections; does not assemble the full tensor system matrix.
    """
    def __init__(self, grid, starts, ends):
        self.grid = grid
        self.scalar = RayTransform(grid, starts, ends)
        self.matrix = self.scalar.paths().to_csr()
        self.exponents = tuple(_powers(2, grid.ndim))
        self.ncomponents = len(self.exponents)
        self.domain_shape = (self.ncomponents, *grid.image_shape)
        self.range_shape = (self.scalar.shape[0], self.ncomponents)
        self.shape = (math.prod(self.range_shape), math.prod(self.domain_shape))
        d = self.scalar._ends[:, :grid.ndim]-self.scalar._starts[:, :grid.ndim]
        norm = np.hypot.reduce(d, axis=1)
        self.directions = np.divide(d, norm[:, None], out=np.zeros_like(d),
                                    where=norm[:, None] != 0)
        self._projections = (np.eye(grid.ndim)[None, :, :]
                             - self.directions[:, :, None]*self.directions[:, None, :])
        self._projections[norm == 0] = 0
        self._basis = np.zeros((self.ncomponents, grid.ndim, grid.ndim))
        for k, (i, j) in enumerate((i, j) for i in range(grid.ndim) for j in range(i, grid.ndim)):
            self._basis[k, i, j] = self._basis[k, j, i] = 1 if i == j else 1/math.sqrt(2)
        for a in (self.directions, self._projections, self._basis):
            a.flags.writeable = False

    def _project(self, values):
        tensors = np.einsum('rc,cij->rij', values, self._basis)
        tensors = self._projections @ tensors @ self._projections
        return np.einsum('rij,cij->rc', tensors, self._basis)

    def direct(self, image):
        x = _real(image, "tensor")
        if x.shape != self.domain_shape:
            raise ValueError(f"tensor must have shape {self.domain_shape}")
        values = self.matrix @ x.reshape(self.ncomponents, self.scalar.shape[1]).T
        return self._project(values)

    def adjoint(self, data):
        y = _real(data, "data")
        if y.shape != self.range_shape:
            raise ValueError(f"data must have shape {self.range_shape}")
        # P f P is self-adjoint for the Frobenius product, including arbitrary
        # (not necessarily transverse) input data. Then backproject each entry.
        return (self.matrix.T @ self._project(y)).T.reshape(self.domain_shape)

    def as_scipy(self):
        """LinearOperator: component-major images, ray-major data (C flatten)."""
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.shape, dtype=np.float64,
            matvec=lambda x: self.direct(np.asarray(x).reshape(self.domain_shape)).ravel(),
            rmatvec=lambda y: self.adjoint(np.asarray(y).reshape(self.range_shape)).ravel())
