# Slice-by-slice TRT example

Run from the repository root after installing `.[scipy]`:

```sh
python examples/trt_slices.py
# A smaller run:
python examples/trt_slices.py --sizes 16 --views 30 --detectors 49 --maxiter 100
# All errors and solver diagnostics as JSON:
python examples/trt_slices.py --json
```

This is a Python/SciPy example. It uses the published **f33 component only**
from Table 1 of N. M. Desai and W. R. B. Lionheart, *An explicit reconstruction
algorithm for the transverse ray transform of a second rank tensor field
from three axis data*, Inverse Problems **32** (2016), 115009:
[DOI](https://doi.org/10.1088/0266-5611/32/11/115009),
[open manuscript, Section 8.2.1](https://arxiv.org/pdf/1511.05305).
It does not implement the paper's full three-axis reconstruction algorithm.

## Phantom and slice identity

On the cube `[-1,1)^3`, the physical component is

```text
f33(x) =  exp(-50 |x - (0.5,-0.5,-0.5)|^2)
        - exp(-50 |x - (0.5,-0.5, 0.5)|^2)
        + exp(-50 |x - (0.5, 0.5, 0.5)|^2).
```

It is zero outside the cube. Other tensor components are zero in this
example. We do not use or modify the ambiguous cancelling entries listed
for f22 in Table 1. The packed TRT component index for f33 is 5 in Python.

For a ray direction `u` with `u3=0`, the transverse projector satisfies
`P_u e3=e3`, so

```text
(Jf)33(ray) = integral f33(x) ds.
```

The experiment uses planes `x3=-0.5` and `x3=+0.5`, represented as the centres
of two voxel layers. Each slice has an `n`-by-`n` grid on `[-1,1)^2`, with
voxel-centre samples of the Gaussian field. Rays stay within their layer;
there is no inter-slice interpolation. These two planes test the slice
identity, not accuracy of a two-layer approximation to the entire 3D field.

Default settings are `n=32,64`, 90 angles over a half-turn (offset by 0.013
radians), and 129 detector offsets from -1.45 to +1.45. Endpoints extend three
length units in either direction from each line's reference point.

## Three data sets

For each resolution, the example constructs:

1. The `33` component of the 3D TRT of the voxel field.
2. Scalar 2D ray projections of the same voxel samples.
3. Analytic integrals of the continuous Gaussian sum, clipped to the cube
   and finite ray segment, using error functions.

The first two should agree to numerical precision. This checks consistency
of the tensor and scalar operations; they share the same traversal core.
The third is independent of the voxel projector. Its difference from the
first two measures the piecewise-constant approximation, not an expected
roundoff-level identity.

For each Gaussian with centre `a`, amplitude `alpha` and width parameter
`beta=50`, parameterize the ray by `s + t*u`, with unit `u` and height `z`.
Let `mu=(a_xy-s).u`, let `rho^2` be squared perpendicular distance to `a`
(including `(z-a3)^2`), and let `[t0,t1]` be the portion inside the cube and
segment. Its integral is

```text
alpha * exp(-beta*rho^2) * sqrt(pi)/(2*sqrt(beta))
      * [erf(sqrt(beta)*(t1-mu)) - erf(sqrt(beta)*(t0-mu))].
```

Misses and zero-length segments return zero. The analytic formula is itself
regression-tested against adaptive numerical quadrature of the phantom,
including partial segments, reversed rays, and axis-parallel boundaries.

## Reconstruction and observed errors

All three data sets are reconstructed slice by slice with the same scalar
matrix and **unregularised LSMR**, using `atol=btol=1e-10` and at most 300
iterations. The reference image is the sampled continuous phantom, rather
than cell averages. `--json` includes the relative data residual, image
error, iteration count and LSMR stopping code for every solve. Stop code 7
means the iteration limit was reached.

In the original local run (Linux, NumPy 2.3.5, SciPy 1.17.0):

| Slice grid | Voxel vs analytic projection error | Image error from analytic data, z=-0.5 / +0.5 |
| --- | ---: | ---: |
| 32 by 32 | 4.13% | 2.79% / 2.77% |
| 64 by 64 | 1.48% | 4.62% / 5.32% |

TRT and scalar voxel data agreed exactly in that run, as did their
reconstructions. Reconstruction from these matched voxel data had relative
image errors around `2e-9` and `2e-8`, respectively. These small errors are
an algebraic consistency check, not evidence of continuum reconstruction
accuracy. Solver rounding/stopping can vary slightly across environments.
All default solves terminated on tolerance before the 300-iteration limit:
voxel-data solves used stop code 1 and analytic-data solves used code 2.

The analytic-data reconstruction errors **do not decrease monotonically**
with voxel refinement. Refining the image while fixing acquisition sampling
changes the inverse problem, and inversion can amplify the remaining model
discrepancy. The experiment does not distinguish conditioning, sampling,
and model error. Regularisation and controlled acquisition
refinement are needed before making a convergence claim.

The small CI regression checks the slice identity and matched-data solve;
it does not enforce an artificial monotonic convergence threshold for the
analytic-data reconstruction. The integration job runs the full example
and prints its diagnostics.

This published phantom can also serve as a reference for future
reconstruction experiments. The acquisition and associated quadrature
weights must be specified for each method; a three-axis data set does not
have the same normal operator as complete directional data. No complete-data
Fourier inversion is implemented in this example.
