"""Voxelwise constant U'(s)=A(x(s))U(s); U(entry)=I; s is arclength.

For A(x,u) from a connection, contract with each ray's oriented unit direction
before forming the generators. Reversing orientation alone does not invert
transport for an orientation-independent A(x); a connection changes sign too.
"""
import numpy as np
from scipy.linalg import expm
from jacobsrays import Grid, trace


def ordered_transport(paths, generators, ray=0):
    """generators has shape (*grid.image_shape, matrix_size, matrix_size)."""
    generators = np.asarray(generators)
    if generators.ndim != paths.grid.ndim+2 or generators.shape[:-2] != paths.grid.image_shape or generators.shape[-1] != generators.shape[-2]:
        raise ValueError('generators must have grid image shape followed by a square matrix')
    k = generators.shape[-1]
    values = generators.reshape(-1, k, k)
    result = np.eye(k, dtype=np.result_type(generators.dtype, np.float64))
    take = paths.ray(ray)
    for voxel, ds in zip(paths.indices[take], paths.lengths[take]):
        # Left multiplication is essential for U'=A U: the latest factor is leftmost.
        result = expm(ds*values[voxel]) @ result
    return result


def main():
    grid = Grid((2,1))
    paths = trace(grid, [-1,.5], [3,.5])
    A = np.array([[0.,1.],[0.,0.]])
    B = np.array([[0.,0.],[1.,0.]])
    generators = np.array([[A,B]])
    result = ordered_transport(paths, generators)
    np.testing.assert_allclose(result, expm(B) @ expm(A))
    print('ordered product:\n', result)
    print('exponential of integrated generator (generally different):\n', expm(A+B))


if __name__ == '__main__':
    main()
