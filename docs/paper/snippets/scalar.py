"""Complete scalar example printed in the user guide."""
import numpy as np
from jacobsrays import Grid, RayTransform, trace

grid = Grid((3, 2), spacing=(1, 1), origin=(0, 0))
starts, ends = [[-1, .5]], [[4, .5]]
paths = trace(grid, starts, ends)
k = paths.ray(0)
print(paths.indices[k], paths.lengths[k])

A = RayTransform(grid, starts, ends)
x = np.ones(grid.image_shape)  # (ny, nx)
y = A.direct(x)
xadj = A.adjoint(y)
matrix = paths.to_csr()
np.testing.assert_allclose(y, matrix @ x.ravel())
np.testing.assert_allclose(xadj.ravel(), matrix.T @ y)
print(y)
