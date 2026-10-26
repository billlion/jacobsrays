# Rank-two transverse ray transform

For a unit ray direction `u`, let `P = I - u u.T`. The implemented full TRT is

```text
Jf(ray) = integral P f(x) P ds.
```

The field is constant in each Cartesian voxel and zero outside the grid;
we integrate over the supplied finite segment. Use endpoints outside the
object to model a complete line. The projection is constant along a straight
ray, so we first integrate each Cartesian tensor component with the scalar
ray matrix, then apply `P f P`.

This is the full transverse tensor, including its trace. It is not the
trace-free/truncated TRT used in some polarization models, nor just one
scalar transverse contraction. In 3D each output is a symmetric 3-by-3 matrix
`G` satisfying `G u = 0`, hence it has three independent entries. Reversing
start/end leaves the rank-two TRT unchanged. A zero-length ray gives zero.

The definition follows Section 5 of
[W. R. B. Lionheart and A. M. Korsunsky, *Diffraction tomography inversion and
the transverse ray transform* (2021)](https://eprints.maths.manchester.ac.uk/2828/1/fully3d_LionheartKorsunsky_v2.pdf).
The implementation is a generic tensor transform; a specific experiment may
measure only selected contractions or require additional physical factors.

## Components, norms and adjoint

In 3D we store

```text
[ f_xx, sqrt(2)*f_xy, sqrt(2)*f_xz, f_yy, sqrt(2)*f_yz, f_zz ].
```

In 2D the order is `[f_xx, sqrt(2)*f_xy, f_yy]`. It agrees with the existing
rank-two LRT. The off-diagonal factors make the ordinary component dot
product equal to the matrix Frobenius product. Input and output use this
same convention; do not apply an additional factor of two to data entries.

The discrete adjoint projects each ray tensor by `P Y P`, then backprojects
its components using the transpose of the scalar matrix. It also accepts
non-transverse data, as can arise from noise. Spatial volume, angular and
line-offset quadrature weights are not included. Add these explicitly when
approximating a chosen continuum inner product.

A full 3D TRT output has six stored components and three independent
components. This redundant Cartesian representation avoids selecting a
transverse frame and preserves Frobenius norms. In 2D there is only one
independent transverse component. The finite scan used in an example does
not establish uniqueness or good conditioning for another acquisition.

## Python and SciPy

```python
from jacobsrays import Grid, TransverseRayTransform

grid = Grid((8, 7, 6), spacing=(.25, .3, .35), origin=(-1, -1.05, -1.05))
op = TransverseRayTransform(grid, starts, ends)
# f.shape == (6, 6, 7, 8): components, z, y, x
Y = op.direct(f)              # shape (number_of_rays, 6)
f_adjoint = op.adjoint(Y)     # same shape as f
A = op.as_scipy()             # use with scipy.sparse.linalg.lsqr / lsmr
# A @ f.ravel() == Y.ravel(), both using NumPy's default C order.
```

`as_scipy()` flattens images component first and data ray first. The operator
caches a scalar CSR matrix, a small projection matrix per ray and constant
basis tensors. It applies the six scalar components together and does not
expand a six-by-six block sparse system. It still needs storage proportional
to the number of scalar ray/voxel intersections; batch scans if necessary.

Run the spatially varying 3D phantom demonstration with:

```sh
python examples/tensor_trt.py
python examples/tensor_trt.py --cil
```

The second command additionally runs CIL CGLS and reports both data residual
and tensor error. It requires a separate CIL installation.
`jacobsrays.cil.TransverseJacobProjector` has a `BlockGeometry` domain (one
image per component) and a `BlockGeometry` range (one ray vector per
component), both in the order above. It supports `direct`, `adjoint`, and
caller-provided `out` buffers. Image geometry restrictions match the other
CIL adapters: single channel, default spatial labels, no singleton axes.

## Detector components and scalar measurements

Choose orthonormal transverse vectors `a,b` and form `Q = [a b]`. From the
physical Cartesian output `G`, the detector tensor is `H = Q.T @ G @ Q`.
Its orthonormal packed components are

```text
[ H_aa, sqrt(2)*H_ab, H_bb ].
```

Changing the detector frame rotates this 2-by-2 tensor. It preserves its
Frobenius norm. For a transverse polarization `q = a*cos(phi) + b*sin(phi)`,
the scalar measurement is

```text
q.T @ G @ q = H_aa*cos(phi)^2 + 2*H_ab*sin(phi)*cos(phi) + H_bb*sin(phi)^2.
```

The examples show this frame conversion. Sampling several polarizations
would define a further measurement operator with its own adjoint and
sampling weights. No particular polarization sampling is assumed here.

## MATLAB

After the existing MEX build, run:

```matlab
addpath('matlab');
test_trt;
example_trt;

T = jacobsrays_trt_matrix(shape, starts, ends, spacing, origin);
% F is prod(shape)-by-6; each column is one tensor component image(:).
Y = reshape(T*F(:), size(starts,1), 6);
Fadj = reshape(T'*Y(:), prod(shape), 6);
```

In 2D use three columns instead of six. MATLAB uses `[nx ny nz]` images,
with x fastest, whereas Python uses `[nz ny nx]` images. The packed component
order agrees across wrappers. Both MATLAB arrays `F(:)` and `Y(:)` are
component first because MATLAB uses column-major flattening; in particular,
MATLAB `Y(:)` and Python `Y.ravel()` have different orderings. Compare the
ray-by-component arrays, or use `Y.ravel(order='F')` when transferring a
Python array to the MATLAB sparse-matrix convention.

This MATLAB helper assembles the full sparse tensor system for clarity and
small numerical experiments. It can have up to 36 times the scalar matrix's
nonzeros in 3D, and allocates temporary block matrices and ray weights.
For larger problems, the Python operator avoids that block expansion.
The shared C++ traversal and MEX gateway are unchanged.

## Checks

The tests compare to analytic constant tensors and to an independent oracle
that slab-clips every voxel and accumulates ordinary physical matrices.
They check transversality, detector contractions, retained transverse trace,
longitudinal and mixed tensors annihilated along a ray, reversal, coordinate
permutation, the identity `tr(Jf) = X(tr(f)) - LRT(f)`, arbitrary-data adjoints,
SciPy/CIL layouts and empty/missing rays. ASTRA validation remains a separate
scalar check; no independent third-party TRT implementation has been used.
