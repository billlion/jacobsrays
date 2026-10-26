"""Symmetric rank-two LRT, using Frobenius-orthonormal components."""
import numpy as np
from jacobsrays import Grid, TensorRayTransform


def main():
    grid = Grid((32, 24), spacing=(1/16, 1/12), origin=(-1, -1))
    angles = np.linspace(0, np.pi, 40, endpoint=False)
    offsets = np.linspace(-1.3, 1.3, 45)
    u = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    normal = np.stack((-u[:, 1], u[:, 0]), axis=1)
    centres = normal[:, None, :]*offsets[None, :, None]
    starts = (centres-3*u[:, None, :]).reshape(-1, 2)
    ends = (centres+3*u[:, None, :]).reshape(-1, 2)
    op = TensorRayTransform(grid, starts, ends, rank=2)
    f = np.zeros(op.domain_shape)
    # Components in this basis: xx, sqrt(2)*xy, yy. Constant tensor identity.
    f[0] = f[2] = 1
    data = op.direct(f)
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=op.domain_shape), rng.normal(size=len(starts))
    lhs, rhs = np.vdot(op.direct(x), y), np.vdot(x, op.adjoint(y))
    print('exponents:', op.exponents)
    print('data shape:', data.reshape(len(angles), len(offsets)).shape)
    print('adjoint relative error:', abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1))
    # op.as_scipy() can be passed to lsqr/lsmr. The tensor LRT has a nullspace:
    # regularisation/constraints are needed to select a meaningful reconstruction.


if __name__ == '__main__':
    main()
