# jacobsrays

Ordered Cartesian voxel intersections for scalar and tensor tomography, and
transport ODEs along rays. A small C++11 kernel is shared by Python and MATLAB.
This is an initial implementation; see [validation and limitations](docs/validation.md).

```sh
python -m pip install '.[scipy]'
```

A C++ compiler is required. NumPy is the only mandatory runtime dependency.
SciPy is optional for sparse matrices, tensor operators and the ODE example.

```python
from jacobsrays import Grid, trace, RayTransform
import numpy as np

grid = Grid((3, 2), spacing=(1, 1), origin=(0, 0))
starts, ends = [[-1, .5]], [[4, .5]]
p = trace(grid, starts, ends)
k = p.ray(0)
print(p.indices[k], p.lengths[k])  # [0, 1, 2], [1., 1., 1.]
print(p.t_enter[k], p.t_exit[k])  # positions along start+t*(end-start)
A = RayTransform(grid, starts, ends)
data = A.direct(np.ones(grid.image_shape))  # [3.]
backprojection = A.adjoint(data)
# Optional: scipy.sparse.linalg solvers or a cached sparse matrix
operator = A.as_scipy()
matrix = p.to_csr()
```

## Coordinate contract

- Geometry and endpoints use **x, y, z** order; 2D uses x, y.
- `origin` is the **lower corner**, with zero as the default. To centre a grid,
  use `origin = -np.array(shape)*spacing/2`.
- Python images use `(ny,nx)` or `(nz,ny,nx)`; MATLAB arrays use `[nx,ny,nz]`.
  Both have x-fastest linear indexing. Python indices/offsets are zero-based;
  MATLAB indices/offsets are one-based.
- Inputs are **finite segments**, clipped to the grid. Choose exterior
  endpoints if the entire line through the grid is wanted.
- Lengths are physical lengths. Intervals remain in **start-to-end order**;
  sparse matrix conversion must not be used to recover ODE travel order.
- Cells are half-open. A parallel ray exactly on an internal plane belongs
  to the higher-index cell. Lower outer faces are included; upper faces are
  excluded. Point tangencies, misses and zero-length rays have no entries.
- Simultaneous plane crossings advance all tied axes. There is no arbitrary
  absolute direction cutoff. Numerically near-tied crossings can retain tiny
  positive intervals; do not expect bitwise reversal symmetry.

## MATLAB

```matlab
addpath('matlab');
build_jacobsrays;
test_jacobsrays;
p = jacobsrays([3 2], [-1 .5], [4 .5]);
A = jacobsrays_matrix(p);
image = ones(3,2);
y = A*image(:);
x_adjoint = reshape(A'*y, [3 2]);
k = p.offsets(1):p.offsets(2)-1; % ordered intervals for the first ray
```

`mod_jacobs_ray3d_multiple` provides the old call signature (including ordered
`'internal'` triples). Its offset is the lower corner, matching the old code's
behaviour rather than its contradictory help text. Add this `matlab` directory
before the old one on your MATLAB path. Sparse sizing preserves the final
coefficient and empty/missing rays. MEX inputs are full real double arrays.

## Tensor tomography and ray ODEs

`TensorRayTransform(grid, starts, ends, rank=m)` supplies the symmetric rank-m
LRT and its adjoint. It caches the scalar sparse matrix and uses an orthonormal
symmetric-tensor basis; see [the tensor example](examples/tensor_lrt.py).
For rank two in 2D the component array is `(f_xx, sqrt(2)*f_xy, f_yy)`.
The ray direction is the unit vector from start to end. Reversing a ray changes
the LRT by `(-1)**m`. This discretization does not remove the continuum LRT's
potential-field nullspace or guarantee unique tensor reconstruction.

`TransverseRayTransform(grid, starts, ends)` supplies the full rank-two TRT
`integral P_u f P_u ds`, where `P_u = I - u u.T`, and its adjoint in 2D/3D.
In 3D both input and output use `(xx, sqrt(2)xy, sqrt(2)xz, yy, sqrt(2)yz, zz)`;
the six output entries encode a tensor with three independent transverse
components. The transverse trace is retained. See [the Python example](examples/tensor_trt.py),
[the MATLAB example](matlab/example_trt.m), and [the TRT conventions](docs/trt.md).

The [slice-by-slice example](examples/trt_slices.py) reconstructs the published
Desai--Lionheart Gaussian `f33` component with a 2D scalar solver. It compares
TRT data, scalar voxel projections and analytic Gaussian integrals; see
[the experiment and its limitations](docs/trt_slices.md).

[The transport example](examples/nonabelian_transport.py) solves `U'=A(x)U`
with an ordered product of voxelwise matrix exponentials. The matrices need
not commute. For a connection use `A(x,u)` contracted with the oriented ray
direction. Smooth within-voxel coefficients can instead use the returned
entry/exit points with an ODE solver on each interval. This is a forward-model
building block, not a nonlinear inverse solver.

## CIL

The optional `jacobsrays.cil.JacobProjector` implements CIL's `LinearOperator`
interface (`direct`, `adjoint`, and `out`) for single-channel `ImageGeometry`
and explicit rays. See [the CGLS example](examples/cil_cgls.py). CIL is installed
separately using [its installation instructions](https://tomographicimaging.github.io/CIL/).
No ASTRA/TIGRE backend is needed. This first adapter returns a `VectorGeometry`;
conversion from CIL acquisition geometries is a future extension.
`TensorJacobProjector` supplies a `BlockGeometry` domain, with one image per
orthonormal tensor component, and can be used with CIL CGLS in the same way. The package does not import CIL unless the adapter is used.
`TransverseJacobProjector` uses component blocks in both domain and range;
run `python examples/tensor_trt.py --cil` for a 3D TRT reconstruction with CGLS.

## Performance and development

The streaming traversal takes O(number of crossed grid planes) time and O(1)
working memory per ray. Ordered batches use two passes to allocate exactly
sized output arrays. Python scalar projection and backprojection avoid storing
intersections altogether; MATLAB uses cached sparse matrices. All adjoints
are **Euclidean discrete transposes**, without voxel-volume/data quadrature
weights. Supply such weights explicitly when approximating other inner products.

Optional OpenMP parallelizes Python tracing and scalar forward projection:

```sh
JACOBSRAYS_OPENMP=1 python -m pip install --no-cache-dir '.[scipy]'
python -m unittest discover -s tests -v
python benchmarks/benchmark.py
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Use a compiler with OpenMP support (on macOS, install/configure libomp first).
The default build is serial. Backprojection deliberately uses deterministic
serial accumulation; it does not allocate a full image per thread. Chunk ray
batches when path storage would be too large. See [the performance notes](docs/performance.md).

The CI integration job also compares 2D scalar projections against ASTRA's
independent CPU line projectors, directly and through CIL. It checks full
system matrices, forward projections and adjoints without requiring a GPU.
See [validation](docs/validation.md#independent-x-ray-comparison) for reproduction,
coordinate conventions, precision tolerances and the square-pixel scope.

## Provenance and references

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). This implementation was
written afresh, informed by an audit of the interface and behaviour in
[Bill Lionheart's LRTinMatlab](https://github.com/billlion/LRTinMatlab), which
contains David Szotten's 2005–2008 Manchester implementation. No legacy source
is bundled or relicensed here. See [the audit](docs/legacy-audit.md).

F. Jacobs, E. Sundermann, B. De Sutter, M. Christiaens and I. Lemahieu,
“A Fast Algorithm to Calculate the Exact Radiological Path through a Pixel or
Voxel Space”, CIT 6(1), 89–94 (1998),
[author institution record](https://biblio.ugent.be/publication/281913).
The traversal is in the Siddon/Jacobs family: slab clipping followed by merging
three monotonically ordered plane-crossing sequences, without a global sort.
