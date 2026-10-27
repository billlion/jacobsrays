# Performance: measured tradeoffs

Measured on 2026-10-26 in a Linux x86-64 container on an AMD EPYC 9V74, with
GCC 13.3, Python 3.12.14 and NumPy 2.3.5. These are local measurements, not
portable performance guarantees. Re-run the scripts on your machine.

## End-to-end Python operations

`python benchmarks/benchmark.py`: 20,000 rays through a 64×64×64 unit-voxel
grid, each starting at x=-1 and ending at x=65 with independently random
interior y/z coordinates. Seed 2026; 2,107,166 intersections. Best of five timed
calls after a warm-up; array allocation and native input validation included.

| Operation | Serial build | OpenMP build, 4 threads |
|---|---:|---:|
| Ordered tracing, including entry/exit parameters | 46.42 ms | 9.90 ms |
| Matrix-free forward projection | 23.33 ms | 5.48 ms |
| Matrix-free adjoint (serial in both builds) | 24.45 ms | 24.48 ms |
| Precomputed SciPy CSR forward | 1.63 ms | 1.62 ms |
| Precomputed SciPy CSR adjoint | 1.94 ms | 2.08 ms |

The cached figures **exclude traversal and sparse construction**. They show
why repeated iterative solves should consider caching when memory allows.
These historical measurements predate the optional atomic adjoint below.
The default adjoint remains serial, with a fixed summation order and no
per-thread images. OpenMP builds can now opt into atomic accumulation.

Ordered paths occupy 67,589,320 bytes: int64 indices, three double arrays
(length, entry, exit), and int64 offsets. The SciPy CSR matrix occupies
25,365,996 bytes in this case (SciPy selects 32-bit sparse indices). It does
not retain interval parameters or the traversal-order contract.

The old wrapper's worst-case temporary triplets for this same grid and ray
count would occupy 92,160,000 bytes, followed by about 50,571,984 bytes for
its trimmed triplets, before sparse conversion. This is a calculation from
its allocation formulas, **not a measurement of peak MATLAB RSS**. New
ordered output contains more information than those triplets. Matrix-free
operations store no path arrays; their native traversal workspace is constant
per thread, plus the input and output arrays.

## Legacy core comparison

```sh
python benchmarks/compare_legacy.py /path/to/LRTinMatlab/jacobsrays
```

This separately compiles the historical C core without modifying or bundling
it. Both kernels produce indices and lengths into per-ray stack arrays;
results are consumed in a checksum. Both run serially, with 200,000
exterior-to-exterior rays through a 64³ grid; the new kernel includes ray
range validation. Best of five after a warm-up. Random points intentionally
avoid known defective legacy partial-ray cases.

| Kernel | Time | Intersections | Weighted length checksum |
|---|---:|---:|---:|
| Historical C | 137.70 ms | 21,068,823 | 16,510,838.6159 |
| New C++ | 164.20 ms | 21,068,823 | 16,510,838.6159 |

The new one-pass kernel was **about 19% slower** here. There is no claim of a
serial traversal speedup. This comparison excludes MATLAB/Python wrappers,
two-pass batch allocation, entry/exit output, sparse conversion and OpenMP.
The practical improvements are correctness, explicit ordered geometry,
checked 64-bit indexing, shared wrappers, efficient batch allocation,
optional parallel processing, streaming projections and reusable matrices.

Avoid `-ffast-math`: infinities and finite-value validation are intentional.
The implementation recomputes crossing parameters from integer plane
numbers to avoid cumulative stepping drift; optimizing this further should
preserve the independent geometric oracle and the adjoint tests.

## Optional atomic adjoint

`RayTransform(..., adjoint_mode="atomic")` and the scalar CIL
`JacobProjector(..., adjoint_mode="atomic")` use an OpenMP loop over rays.
Each intersection calls the same traversal as the serial implementation,
with an atomic addition to a single shared output image. No volume copies
or cached intersection matrix are needed. Input validation and output
zeroing are still serial. The implicit loop barrier completes all updates
before the array is returned.

The default mode is `"serial"`, even in an OpenMP build. Atomic mode raises
an error if the extension was compiled without OpenMP. It is still a matched
Euclidean transpose, but the order of floating-point additions may change
between calls. Contention and cache traffic can limit speedup; there is no
claim that atomic mode is faster for every geometry or thread count.
This mode does not change the cached tensor or MATLAB sparse operators.

Reproduce with an OpenMP build (see the README):

```sh
python benchmarks/atomic_adjoint.py --threads 1 2 4 8
```

The script starts a fresh process per thread count with `OMP_DYNAMIC=FALSE`
and one BLAS thread. It compares serial and atomic adjoints in the same
OpenMP build, alternates their timing order, and reports the minimum and
median of five calls after warm-up. Allocation and validation are included.
It checks every atomic result against the serial result and checks the
dot-product identity. Timing results are informational, not CI pass/fail
thresholds. Requested threads need not all correspond to physical cores.

Local results on 2026-10-27: Linux, GCC 13.3, AMD EPYC 9V74, with nine CPUs
available to the process, Python 3.12.14 and NumPy 2.3.5. Each case has
20,000 rays through a 64-cubed unit-voxel grid. Medians in milliseconds:

| Ray geometry | Threads | Serial adjoint | Atomic adjoint | Speedup |
|---|---:|---:|---:|---:|
| Distributed endpoints on opposite x faces | 2 | 28.22 | 14.70 | 1.92x |
| Distributed endpoints on opposite x faces | 4 | 28.56 | 9.51 | 3.00x |
| Distributed endpoints on opposite x faces | 8 | 27.35 | 6.07 | 4.51x |
| All rays through the centre | 4 | 30.10 | 12.15 | 2.48x |
| All rays through the centre | 8 | 31.24 | 9.62 | 3.25x |
| Identical axis-aligned rays | 4 | 9.51 | 5.04 | 1.89x |
| Identical axis-aligned rays | 8 | 9.69 | 7.62 | 1.27x |

Identical rays intentionally stress repeated updates to the same voxels;
eight threads were slower than four for that case. The serial measurements
also vary between processes, so these ratios describe this run, not portable
guarantees. The older table used minima and a separate build/run and should
not be directly used as a baseline for these medians.

Across all cases and 1/2/4/8 threads, the largest relative image difference
from serial was 7.1e-15. The largest dot-product discrepancy, divided by
`norm(Ax)*norm(y) + norm(x)*norm(Aty)`, was 1.1e-17. These are observed values,
not universal roundoff bounds. Full timings, including forward projection,
are in [the recorded JSON](../benchmarks/atomic_results_2026-10-27.json).
