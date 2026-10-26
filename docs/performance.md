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
The adjoint is deliberately serial to avoid nondeterministic scatter races,
atomics, or one full image per worker. It is the next obvious parallel
performance target, requiring an explicit memory/determinism tradeoff.

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
