# Optional oneMKL Sparse BLAS experiment

This experiment calls oneMKL's CPU Sparse BLAS on cached ray matrices. It
measures scalar products, six-component products, and complete rank-two LRT
and TRT applications. The experimental adapter lives in
[`benchmarks/mkl_probe.py`](../benchmarks/mkl_probe.py); it is **not a supported
package backend**. Normal installation, MATLAB, CIL and the matrix-free C++
traversal are unchanged. Installing MKL alone does not redirect these operators
to MKL.

## What is compared

The same voxel-intersection matrix and random input data are used throughout:

- `scipy`: the current cached CSR matrix, in ray-intersection order.
- `scipy_sorted`: a copy with sorted column indices, exactly the CSR arrays
  supplied to MKL. This controls for sorting alone.
- `scipy_cached_transpose`: sorted CSR forward, plus a separately cached
  `A.T.tocsr()` for the adjoint. This controls for a simpler improvement that
  requires no additional library. Its construction time and storage are recorded.
- `mkl`: double-precision `mkl_sparse_d_mv` / `mkl_sparse_d_mm`, with forward
  and transpose hints and `mkl_sparse_optimize` performed once. Six tensor
  components are processed together in row-major dense storage.

For LRT/TRT, only the existing operator's `matrix` attribute is substituted
inside the benchmark. Component packing, tensor contractions and transverse
projection use the existing Python implementation. Input conversion, output
allocation and those tensor operations are inside the timed application.
Ray tracing, CSR creation, MKL handle preparation and transpose-cache creation
are reported separately. MKL preparation includes copying/sorting the CSR and
inspector hints/optimization, but excludes loading the shared library and
Python startup. No complete iterative solver is timed.

Each thread count runs in a fresh process. GNU OpenMP matches the GCC build;
LP64 uses 32-bit sparse indices. OpenBLAS/BLIS are restricted to one thread,
MKL dynamic threading is disabled, and `OMP_WAIT_POLICY=PASSIVE` prevents idle
MKL workers from spinning during SciPy measurements. After warm-up and accuracy
checks, backend order is rotated/reversed and all timing samples are retained.
The tables use medians, not best times. A configured MKL thread count is a
maximum; MKL may choose fewer workers for an operation. SciPy's sparse kernels
are the current single-threaded reference.

## Measurements on 27 October 2026

Measured on an AMD EPYC 9V74 virtual machine, with eight cores of CPU quota
and 8 GiB memory limit; Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
oneMKL 2026.2 (build 20260915), GCC 13.3 and the OpenMP extension.
The package code is main commit `2cc1e2c`. These are observations on one
shared host, not promised speedups; variation between samples is visible in
[the complete raw results](../benchmarks/mkl_results_2026-10-27.json).

The following uses **four MKL threads**. Each entry is the sum of the
separately measured median forward and adjoint times, in milliseconds.
This approximates the operator work in an iteration, not its other solver
operations. The small case uses nine repetitions, the large case seven.

| Grid / rays | Operator | Current SciPy | SciPy + cached CSR transpose | MKL | Speedup vs current | Speedup vs cached transpose |
|---|---|---:|---:|---:|---:|---:|
| 64³ / 20,000 | Scalar | 5.8 | 5.8 | 3.1 | 1.87× | 1.87× |
| 64³ / 20,000 | LRT | 27.5 | 23.6 | 8.6 | 3.20× | 2.75× |
| 64³ / 20,000 | TRT | 42.7 | 34.9 | 18.6 | 2.30× | 1.88× |
| 128³ / 100,000 | Scalar | 193.3 | 122.7 | 39.9 | 4.85× | 3.08× |
| 128³ / 100,000 | LRT | 968.2 | 596.6 | 199.5 | 4.85× | 2.99× |
| 128³ / 100,000 | TRT | 867.8 | 586.3 | 212.3 | 4.09× | 2.76× |

The transpose control is significant: on the large case it reduces the
SciPy LRT adjoint from about 422 ms to 100 ms without MKL. MKL reduces it
further to about 31 ms. Thus the largest headline adjoint gain is partly
a storage-layout improvement, not purely a library advantage. MKL remains
faster than this control in the complete tensor application pairs.

With **one MKL thread**, the LRT/TRT pair gains over current SciPy were
about 1.5×/1.4× on the small case and 2.1×/2.0× on the large case.
Against the cached-transpose control they were about 1.6×/1.4× and
1.4×/1.4× respectively. The small scalar pair was essentially unchanged;
MKL is not a universal win for individual scalar calls.

| Four-thread setup/storage observation | 64³ / 20,000 rays | 128³ / 100,000 rays |
|---|---:|---:|
| Original CSR storage (MiB) | 24.1 | 243.0 |
| MKL preparation (ms) | 66 | 886 |
| Process RSS increase during MKL preparation (MiB) | 124 | 1074 |
| SciPy transpose creation (ms) | 26 | 467 |
| Additional SciPy transpose storage (MiB) | 25.0 | 250.6 |

RSS is a process observation, including the owned CSR copy, touched library
pages and MKL internal allocations; it is not an isolated allocator count or
a universal per-matrix cost. The benchmark also retains its SciPy references
and multiple tensor buffers. Peak RSS in the JSON is for the whole experiment.
The roughly 1.05 GiB increase for the large MKL setup is a material cost.

Using the four-thread medians, MKL preparation is recovered after roughly
3–4 LRT/TRT forward-plus-adjoint pairs on the small case and two pairs on the
large case, relative to current SciPy. This estimate excludes common geometry
construction, library loading and interpreter startup; short runs can still
lose overall. The simpler transpose cache is a useful alternative when memory
or extra dependencies matter.

Maximum observed relative discrepancy from SciPy was 5.5e-16; the largest
scaled LRT/TRT adjoint-test error was 6.2e-18.

**Decision:** the measured gains justify retaining an opt-in experiment and
testing it on other machines. They do not justify making MKL mandatory or
silently enabling it. A supported backend would need explicit resource
ownership, memory policy, platform coverage and integration tests beyond this
benchmark. No public Python or MATLAB interface has changed.

## Correctness and scope

Every timed operator is compared to the original SciPy result outside the timed
region, with relative L2 tolerance `5e-12`. Full LRT/TRT also pass an adjoint
inner-product test, scaled by `||Ax|| ||y|| + ||x|| ||A*y||`, at `5e-14`.
The separate adapter tests check nonsquare matrices, unsorted columns,
duplicate entries, empty rows, no rays, all-missed rays, zero dense columns,
C/Fortran/strided input, ownership of CSR storage, and invalid input rejection
against small independent dense products. They are outside the standard test
suite because MKL is optional.

These are CPU cached-matrix experiments on distributed rays crossing the x
faces of a cube; they are not a general geometry or hardware survey. They do
not accelerate the streaming traversal or its atomic adjoint. Retaining a
large sparse matrix is already a different memory trade-off from streaming.
The adapter deliberately supports only real double CSR and LP64-sized matrices
in a fresh Linux x86-64 process. Sorting is confined to the adapter's own CSR
copy: ordered intersection paths used by ODE/non-abelian examples are untouched.

## Reproduce

Use a fresh checkout/build directory and Python 3.12 on Linux x86-64:

```sh
python3.12 -m venv .venv
JACOBSRAYS_OPENMP=1 .venv/bin/python -m pip install '.[scipy]' \
    'numpy==2.3.5' 'scipy==1.17.0' 'mkl==2026.2.0'
OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE MKL_NUM_THREADS=2 \
    MKL_THREADING_LAYER=GNU MKL_INTERFACE_LAYER=LP64 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -m unittest discover -s benchmarks -p 'test_mkl_probe.py' -v
.venv/bin/python benchmarks/mkl_probe.py \
    --size 64 --rays 20000 --threads 1 4 --repeat 9 > small.json
.venv/bin/python benchmarks/mkl_probe.py \
    --size 128 --rays 100000 --threads 1 4 --repeat 7 > large.json
```

The benchmark sets the threading environment separately for each child process.
It expects the PyPI runtime at `.venv/lib/libmkl_rt.so.3`; use `--library` for
another location. No Intel compiler or MKL header package is needed for this
ctypes experiment. Do not benchmark concurrently with other heavy jobs.

[The optional workflow](../.github/workflows/mkl.yml) runs on manual dispatch
and on PRs changing the experiment code or its workflow. It installs MKL in its
own job and uploads raw JSON. Unrelated PRs keep the existing validation cost;
there is no performance threshold or required-check rule added. Hosted-runner
timings should be interpreted as observations, not stable regression gates.

## Library references

- [Intel oneMKL Python runtime package](https://pypi.org/project/mkl/).
- [Sparse matrix/multiple-vector multiplication](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/mkl-sparse-mm.html).
- [Inspector hints](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/mkl-sparse-set-mm-hint.html)
  and [optimization](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/mkl-sparse-optimize.html).
