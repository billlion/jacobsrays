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
- Independent ASTRA 2.5.0 CPU line-projector comparisons, both directly and
  through the pinned CIL ASTRA plugin; see the details below.
- MATLAB MEX build and `test_jacobsrays` passed on GitHub's Ubuntu runner
  with MATLAB R2026b and g++. Bill also confirmed the corrected suite passes
  on his own MATLAB installation. Inspect CI for the particular commit used.
- Rank-two TRT in 2D/3D: physical-matrix formulas and independent per-voxel
  slab clipping, full transverse trace, longitudinal/mixed components
  annihilated, detector polarizations, reversal, axis-permutation covariance,
  the trace/LRT identity, arbitrary-data adjoint checks, SciPy flattening,
  empty/missing/degenerate rays, and invalid input. The CIL block adapter
  passes `out` and adjoint checks and a CGLS solve. The 3D example reaches
  relative data residual 0.00165152 and relative tensor error 0.0349916 after
  40 CGLS iterations. This is one noiseless finite-scan experiment.

- A [slice-by-slice Gaussian example](trt_slices.md) compares the TRT's `33`
  output with scalar 2D projections and reconstructions. Its independent
  analytic finite-segment Gaussian integrals are checked against adaptive
  quadrature, including clipping, axis-parallel rays, misses and reversal.

On 2026-10-27, the optional OpenMP atomic scalar adjoint was checked with
1, 2, 4 and 8 threads. New tests cover 16,384 identical rays accumulating
exactly representable integer contributions into four shared voxels, signed
random 2D/3D projections on anisotropic grids against the sparse transpose,
dot-product identities, empty/missing/degenerate rays, SciPy `rmatvec`, and
the scalar CIL adapter's `out` handling and CGLS. Serial builds reject an
atomic request explicitly. The full suite passed in both builds: all 29
methods in the OpenMP build, and 25 with four OpenMP-only skips in the serial
build. See [the atomic benchmarks](performance.md#optional-atomic-adjoint).

The full local Python suite contains 29 test methods: 11 core/transport,
four TRT, two slice-example, three CIL integration, three native ASTRA
comparison, one CIL/ASTRA comparison, and five atomic-adjoint methods.
Tests for optional packages are explicitly skipped when those packages
are absent; the `cil` CI job installs both pinned references and
runs all 29 methods in an OpenMP build and the full slice example.
The GitHub workflow adds Windows/macOS serial builds, Python 3.10 with OpenMP
at 1/2/4 threads, an integration job pinned to the tested CIL source, and a
MATLAB job that builds the MEX gateway and runs `test_jacobsrays`, `test_trt`
and `example_trt`.
The new TRT MATLAB tests include independent voxel clipping, packed tensor
normalization, adjoint/reversal and empty/miss cases. Remote CI results must be
assessed separately from the local checks above.

## Independent X-ray comparison

`tests/test_astra.py` uses ASTRA's CPU `line` and `line_fanflat` projectors.
Their weights are zero-width ray/pixel intersection lengths, which match
our piecewise-constant scalar model. The comparison covers:

- Full system matrices for parallel and fan beams: 29 views, 41 detector
  elements, and an 11-by-8 image, at pixel widths 1, 0.375 and 2.5. These six
  matrices cover 7,134 rays and check every voxel impulse coefficient,
  including zero coefficients and rays that miss the image.
- Translated image windows, physical length scaling, and matrix-free forward
  and adjoint results for signed random inputs.
- Exactly known axis-aligned lengths and an asymmetric row pattern to anchor
  orientation. Native ASTRA image rows descend in physical y; jacobsrays rows
  ascend. The test permutes matrix columns accordingly and verifies that
  omitting this permutation would fail.
- CIL's `ProjectionOperator(..., device='cpu')` against `JacobProjector`, for
  parallel/fan geometries, left/right detector ordering, and an initial angle
  offset. The rays are constructed independently of CIL's ASTRA converter.
  These wrapped comparisons use centred images and compare both forward
  projection and backprojection. CIL's 2D adapter already handles ASTRA's
  y convention, so no extra image flip is applied there.

ASTRA's line kernels use float32 internally; jacobsrays uses float64.
Tolerances are `3e-4 * pixel_width` absolute per matrix coefficient and
`2e-5` relative L2 for matrices, forward projections and adjoints. Near-axis
rays amplify float32 roundoff. During validation, the largest discrepant
coefficients were also checked by direct ray/voxel slab clipping and agreed
with jacobsrays to double precision. No fitted scale factor or exclusion of
small coefficients is used. The test prints measured errors in the CI log.

Scope matters:

- The external comparison uses **square pixels**. ASTRA's parallel line
  kernel explicitly states this limitation; an anisotropic probe disagreed
  substantially for both line kernels. Anisotropic grids remain covered by
  our independent plane-enumeration oracle, not claimed as ASTRA-validated.
- This validates 2D scalar projection coefficients. It does not independently
  certify 3D traversal, tensor normalization, or ordered non-abelian transport.
  A scalar system matrix cannot establish the order of intersections along
  a ray; the existing ordered-path and noncommuting-transport tests remain.
- ASTRA's CUDA projector uses the Joseph/interpolated model, so equality to
  our intersection-length matrix is not expected. For a future 3D GPU
  comparison, CIL/TIGRE with `direct_method='Siddon'` is a suitable candidate.
  CIL documents TIGRE's `matched` backprojection as approximately matching
  the interpolated projector; it should not be assumed to be the exact
  transpose of the Siddon forward projector.

No GPU is needed for the current comparison. The ASTRA pip wheel installs
CUDA runtime libraries, but the tests explicitly select CPU line projectors
and ran locally with `astra.use_cuda()` false. GPU validation would require
a separately provisioned runner: GitHub's paid GPU larger runners require
an organisation on Team/Enterprise Cloud, or a self-hosted GPU machine can
be used. No GPU runner or paid service is configured by this project.

References: [ASTRA projector models](https://astra-toolbox.com/docs/proj2d.html),
[ASTRA geometry conventions](https://astra-toolbox.com/docs/geom2d.html),
[parallel line kernel limitations](https://github.com/astra-toolbox/astra-toolbox/blob/v2.5.0/include/astra/ParallelBeamLineKernelProjector2D.inl),
[CIL's TIGRE interface](https://github.com/TomographicImaging/CIL/blob/b39f95799bf1af891a896fc4be704986284b593e/Wrappers/Python/cil/plugins/tigre/ProjectionOperator.py),
and [GitHub GPU runners](https://docs.github.com/en/actions/concepts/runners/larger-runners).

## Not yet validated / deliberately limited

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
- Tensor projection caches a scalar matrix, plus direction weights or
  transverse projections and
  temporary component projections. It does not assemble a dense tensor
  matrix. The MATLAB TRT example explicitly assembles a sparse tensor matrix
  and is intended for small problems; see [TRT memory use](trt.md).
  High LRT tensor ranks increase the number of components substantially.
- Adjoint means the discrete Euclidean transpose (with orthonormal tensor
  components). Other image/data inner products need explicit quadrature
  weights. Tensor nullspaces and inverse-problem regularization remain the
  responsibility of the reconstruction model.

## Reproduction

```sh
python -m pip install '.[scipy]'
python -m unittest discover -s tests -v
python examples/tensor_lrt.py
python examples/tensor_trt.py
python examples/trt_slices.py
python examples/nonabelian_transport.py
cmake -S . -B build/core
cmake --build build/core
ctest --test-dir build/core --output-on-failure
# In a CIL environment:
python examples/cil_cgls.py
python examples/tensor_trt.py --cil
# Independent CPU reference (Linux/Windows Python environment):
python -m pip install 'astra-toolbox==2.5.0'
python -m unittest discover -s tests -p test_astra.py -v
# Also install the pinned CIL source above to include the wrapped comparison.
```

The benchmark scripts do not enforce timing thresholds: timing is hardware
and workload dependent. Scalar/tensor CIL integration tests are genuine
upstream integration checks, not substitutes implemented with mocks.
