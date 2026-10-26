"""Explicit parallel rays with CIL CGLS; run in an environment containing CIL."""
import numpy as np
from cil.framework import ImageGeometry
from cil.optimisation.algorithms import CGLS
from jacobsrays.cil import JacobProjector


def main():
    ig = ImageGeometry(voxel_num_x=32, voxel_num_y=24,
                       voxel_size_x=1/16, voxel_size_y=1/12, dtype=np.float64)
    angles = np.linspace(0, np.pi, 50, endpoint=False)
    offsets = np.linspace(-1.4, 1.4, 50)
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    centres = normal[:, None, :]*offsets[None, :, None]
    starts = (centres-3*direction[:, None, :]).reshape(-1, 2)
    ends = (centres+3*direction[:, None, :]).reshape(-1, 2)
    operator = JacobProjector(ig, starts, ends)
    phantom = ig.allocate(0, dtype=np.float64)
    yy, xx = np.mgrid[-1:1:24j, -1:1:32j]
    phantom.fill(((xx/.6)**2+(yy/.8)**2 < 1).astype(float))
    data = operator.direct(phantom)
    solver = CGLS(initial=ig.allocate(0, dtype=np.float64),
                  operator=operator, data=data)
    solver.run(30, verbose=0)
    residual = np.linalg.norm(operator.direct(solver.solution).as_array()-data.as_array())
    print('CGLS relative data residual:', residual/np.linalg.norm(data.as_array()))


if __name__ == '__main__':
    main()
