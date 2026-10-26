# SPDX-License-Identifier: Apache-2.0
"""Full 3D rank-two TRT, detector components, and optional CIL CGLS.

Run: python examples/tensor_trt.py
     python examples/tensor_trt.py --cil  # also requires CIL
"""
import argparse
import numpy as np
from jacobsrays import Grid, TransverseRayTransform


def make_problem():
    grid = Grid((8, 7, 6), (.25, .3, .35), (-1, -1.05, -1.05))
    rng = np.random.default_rng(17)
    directions = rng.normal(size=(24, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # A detector basis for each view. The TRT operator itself needs no basis.
    axis = np.eye(3)[np.argmin(np.abs(directions), axis=1)]
    a = np.cross(directions, axis)
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(directions, a)
    p, q = np.meshgrid(np.linspace(-1.4, 1.4, 9), np.linspace(-1.4, 1.4, 9))
    centres = a[:, None, :]*p.ravel()[None, :, None] + b[:, None, :]*q.ravel()[None, :, None]
    starts = (centres-3*directions[:, None, :]).reshape(-1, 3)
    ends = (centres+3*directions[:, None, :]).reshape(-1, 3)
    z, y, x = np.meshgrid(*(grid.origin[k]+(np.arange(grid.shape[k])+.5)*grid.spacing[k]
                            for k in (2, 1, 0)), indexing='ij')
    mask = (x/.8)**2+(y/.7)**2+(z/.9)**2 < 1
    # Spatially varying symmetric positive tensor: mask * (I + .4 v v.T).
    tensor = np.stack((1+.4*x*x, np.sqrt(2)*.4*x*y, np.sqrt(2)*.4*x*z,
                       1+.4*y*y, np.sqrt(2)*.4*y*z, 1+.4*z*z))*mask
    return grid, starts, ends, tensor, a, b


def main(use_cil=False):
    grid, starts, ends, tensor, a, b = make_problem()
    op = TransverseRayTransform(grid, starts, ends)
    data = op.direct(tensor)
    print('Input shape (components,z,y,x):', tensor.shape)
    print('Output shape (rays,components):', data.shape)
    # The central ray in the first view: convert its six stored components
    # to a physical matrix, then read the three independent detector entries.
    c = data[40]
    T = np.array([[c[0], c[1]/np.sqrt(2), c[2]/np.sqrt(2)],
                  [c[1]/np.sqrt(2), c[3], c[4]/np.sqrt(2)],
                  [c[2]/np.sqrt(2), c[4]/np.sqrt(2), c[5]]])
    Q = np.column_stack((a[0], b[0]))
    H = Q.T@T@Q
    print('Transverse components (aa,sqrt(2)ab,bb):', H[0, 0], np.sqrt(2)*H[0, 1], H[1, 1])
    print('Transversality error ||T u||:', np.linalg.norm(T@op.directions[40]))
    # For any transverse polarization q, the scalar measurement is q.T T q.
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=op.domain_shape), rng.normal(size=op.range_shape)
    lhs, rhs = np.vdot(op.direct(x), y), np.vdot(x, op.adjoint(y))
    print('Adjoint relative error:', abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1))
    # op.as_scipy() also works with lsqr/lsmr: images and data use C flattening.
    if use_cil:
        from cil.framework import ImageGeometry
        from cil.optimisation.algorithms import CGLS
        from jacobsrays.cil import TransverseJacobProjector
        ig = ImageGeometry(voxel_num_x=8, voxel_num_y=7, voxel_num_z=6,
                           voxel_size_x=.25, voxel_size_y=.3, voxel_size_z=.35,
                           dtype=np.float64)
        operator = TransverseJacobProjector(ig, starts, ends)
        phantom = operator.domain_geometry().allocate(0)
        for j in range(6):
            phantom.get_item(j).fill(tensor[j])
        observations = operator.direct(phantom)
        solver = CGLS(initial=operator.domain_geometry().allocate(0),
                      operator=operator, data=observations)
        solver.run(40, verbose=0)
        print('CIL CGLS relative data residual:',
              (operator.direct(solver.solution)-observations).norm()/observations.norm())
        print('CIL CGLS relative tensor error:',
              (solver.solution-phantom).norm()/phantom.norm())
        # A finite, noisy or restricted scan can still need regularization.
        # This noiseless demonstration is not a general uniqueness guarantee.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cil', action='store_true', help='also run CIL CGLS')
    main(parser.parse_args().cil)
