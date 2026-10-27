"""Complete full-TRT trace example printed in the user guide."""
import numpy as np
from jacobsrays import Grid, RayTransform, TransverseRayTransform

grid = Grid((4, 3, 2))
starts = [[-1, .5, .5], [.5, -1, .5]]
ends = [[5, .5, .5], [.5, 4, .5]]
phi = np.ones(grid.image_shape)
F = np.zeros((6, *grid.image_shape))
F[[0, 3, 5]] = phi  # physical tensor phi*I

J = TransverseRayTransform(grid, starts, ends)
Y = J.direct(F)
X = RayTransform(grid, starts, ends).direct(phi)
np.testing.assert_allclose(Y[:, [0, 3, 5]].sum(axis=1), 2*X)
print("TRT trace:", Y[:, [0, 3, 5]].sum(axis=1))
