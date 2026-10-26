# Validation and current limits

Initial local validation: 2026-10-26, Linux x86-64, GCC 13.3, Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0.

## Executed

- Serial and OpenMP builds of the Python extension.
- 4,000 random 2D/3D rays against an independent oracle that enumerates all
  grid planes, sorts the crossings, and assigns cells by interval midpoints.
  Includes anisotropic, offset, non-cubic grids and reversal checks.
- Explicit axis, corner, boundary, internal endpoint, short-ray, miss,
  zero-length, empty-batch and invalid-input cases.
- Scalar matrix-free operations versus sparse multiplication; scalar and
  symmetric tensor dot-product adjoint tests for ranks 0 through 3 in 2D/3D.
- Noncommuting complex matrix transport, unitarity for skew-Hermitian
  generators, and inverse transport after reversing the ray AND the sign
  of the connection generator.
- Native C++ CMake/CTest test and an AddressSanitizer/UndefinedBehaviorSanitizer
  build. LeakSanitizer was disabled because the container cannot inspect
  process threads; no claim of leak-sanitizer validation is made.
- Actual CIL built from upstream commit
  `b39f95799bf1af891a896fc4be704986284b593e` (2026-10-23 checkout).
  Scalar and tensor block adapters passed 2D/3D adjoint and `out` checks.
  CGLS ran with scalar and tensor operators. The scalar example reached a
  relative data residual of 0.00016764 after 30 iterations. A small data
  residual does not demonstrate uniqueness or accurate tensor recovery.
- A compiled reproduction of the historical partial-ray bug and a
  head-to-head legacy traversal benchmark; see the audit/performance notes.

The full local Python suite contains 13 test methods, including two tests
against actual CIL. Without CIL, only those two tests are explicitly skipped.
The GitHub workflow adds Windows/macOS builds, Python 3.10 with OpenMP,
and an integration job pinned to the tested CIL source. Remote CI results
must be assessed separately from the local checks above.

## Not yet validated / deliberately limited

- **MATLAB is unavailable in the development environment.** The MEX source,
  build script, compatibility wrapper and `test_jacobsrays.m` are included,
  but the MEX gateway has not been compiled or executed in MATLAB. Run
  `build_jacobsrays; test_jacobsrays` before relying on it.
- This is straight-line traversal on uniform Cartesian grids. Rotated grids
  require transforming endpoints first; curved rays, adaptive grids and
  finite-width beams are not implemented. There is no GPU backend.
- Coordinates and arithmetic use double precision, with checked finite
  ranges and representable grid extents. Geometry on scales too disparate
  for double precision can be inaccurate; translate/rescale the geometry.
  Near-coincident planes can create tiny positive intervals. Do not assert
  exact bitwise reversal equality or silently drop small segments in ODEs.
- Paths store O(number of intersections) output. Use finite batches for large
  scans or matrix-free scalar operations when path caching is unaffordable.
- Core Python scalar and tensor projectors accept real arrays. Ordered paths
  can drive complex-valued ODEs, as in the transport example. The CIL bridge
  uses float64 geometries; user-supplied lower-precision `out` buffers round
  the results in the usual way.
- CIL uses explicit rays, default spatial dimension labels and no channel
  axis. Singleton spatial axes are rejected by this first adapter. There
  is no automatic AcquisitionGeometry conversion or nonlinear Jacobian.
- Tensor projection caches a scalar matrix, plus direction weights and
  temporary component projections. It does not assemble a dense tensor
  matrix. High tensor ranks increase the number of components substantially.
- Adjoint means the discrete Euclidean transpose (with orthonormal tensor
  components). Other image/data inner products need explicit quadrature
  weights. Tensor nullspaces and inverse-problem regularization remain the
  responsibility of the reconstruction model.

## Reproduction

```sh
python -m pip install '.[scipy]'
python -m unittest discover -s tests -v
python examples/tensor_lrt.py
python examples/nonabelian_transport.py
cmake -S . -B build/core
cmake --build build/core
ctest --test-dir build/core --output-on-failure
# In a CIL environment:
python examples/cil_cgls.py
```

The benchmark scripts do not enforce timing thresholds: timing is hardware
and workload dependent. Scalar/tensor CIL integration tests are genuine
upstream integration checks, not substitutes implemented with mocks.
