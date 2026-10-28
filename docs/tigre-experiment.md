# Experimental TIGRE tensor projector

This is a small **benchmark-only** Python prototype, not a supported GPU
backend. It adds no dependency or import to `jacobsrays`, and does not alter
the CPU operators or ordered traversal used for nonabelian transport.
No GPU performance or accuracy result has yet been measured for this prototype.

The prototype is in [`benchmarks/tigre_tensor.py`](../benchmarks/tigre_tensor.py).
It applies TIGRE's scalar `Ax(..., projection_type="Siddon")` to the six
orthonormal symmetric-tensor components, then forms the rank-two LRT or full
TRT with NumPy contractions. Backprojection applies the corresponding tensor
weights before six `Atb(..., backprojection_type="matched")` calls.

The component order is `(xx, sqrt(2)xy, sqrt(2)xz, yy, sqrt(2)yz, zz)`.
Images are `(component, z, y, x)`. Scalar TIGRE sinograms are `(view, v, u)`;
the wrapper flattens rays in this order, returning LRT `(ray,)` and TRT
`(ray, component)` data. TRT retains the transverse trace.

## Scope and limitations

- Only centred 3D parallel-beam scans rotating around z, with untilted,
  unshifted detectors. At angle zero the ray direction is -x, horizontal
  detector position increases along +y, and vertical position along +z.
  Geometry conversion follows the pinned TIGRE Python interface, including
  its internal detector-v reversal. Cone beam, arbitrary endpoints and
  general scan trajectories are outside this prototype.
- These directions are a controlled performance/validation case, **not
  sufficient sampling for general 3D tensor inversion**. No reconstruction
  quality or invertibility claim is made.
- TIGRE uses float32. CPU cached-CSR controls are measured in float32 and
  float64; optional MKL uses float64. All receive the same representable
  input values. CPU tensor products use all six columns in one sparse
  matrix multiplication, matching the existing cached tensor approach.
- This wrapper passes host NumPy arrays to TIGRE. Six calls, transfers,
  host tensor contractions, input checks and output allocations are included
  in timings. It is not a fused tensor CUDA kernel or a GPU-resident solver.
- `backproject` is intentionally not named `adjoint`: TIGRE's `matched`
  weighting is not a promise of an exact transpose of its Siddon projector.
  No CIL `LinearOperator` is exposed by this experiment.

## What validation measures

Before any benchmark, two tiny non-cubic volumes (unit and non-unit
anisotropic voxel sizes) are tested. Every column of the GPU forward matrix
is extracted using voxel impulses and compared with the independently
generated `jacobsrays` CSR weights. There is no fitted scale or axis search.
The provisional geometry checks require relative matrix error <= 5e-4 and
maximum coefficient error <= 5e-3 times the smallest voxel width. Failure
saves a diagnostic JSON, exits nonzero, and prevents performance timings.

For scalar, LRT and TRT, three input pairs include a positive trial and signed
random trials. The report contains:

- Forward error against the CPU reference.
- Backprojection error against the CPU adjoint.
- Backprojection error against the transpose of **TIGRE's own extracted
  forward matrix**. This separates an adjoint mismatch from a different
  geometric discretisation.
- Both dot products and the scaled defect
  `abs(<Ax,y>-<x,By>)/(norm(Ax)*norm(y)+norm(x)*norm(By))`.

A relative-error field is `null` if its reference is zero and the result is
nonzero, since a finite relative error is then undefined.

Backprojection discrepancies are **diagnostics**, not pass/fail checks with a
permissive tolerance. A status of `completed_with_backprojection_diagnostics`
means the experiment ran and its forward matrix check passed; it does not
mean the backprojector passed an adjoint test. Missing TIGRE, a missing GPU,
runtime errors and non-finite results fail the command, with an error JSON.

The ordinary CPU test suite checks tensor algebra against the existing
operators, physical constant-tensor identities, geometry ordering, both
precisions and optional-dependency isolation. These tests are not CUDA tests.

## Run on CSF3

From the repository root, after checking out the prototype branch:

```bash
sbatch benchmarks/csf3/tigre-tensor.slurm
```

The script requests one A100 80 GB GPU, eight CPU cores and 64 GB host RAM.
It uses a separate `~/conda-envs/jacobsrays-tigre312` environment and calls its
Python by absolute path, avoiding the earlier central-Python activation issue.
It loads GCC 14.2.0, CUDA 12.8.1 and the previously used Miniforge module.
On the first run it builds CERN TIGRE from the pinned source commit
`9195fd57faa1eba47dedfd45d95a9957d3910dbd` (version 3.1.3); later runs reuse it.
The two-hour limit allows for this initial source build. The script and build
have not yet been exercised on CSF3; any setup failure will be in the job log.

The default benchmark uses a 64^3 volume, 32 views, five repetitions, and an
eight-thread MKL control alongside the two SciPy controls. It is a **different
ray set from the earlier random-ray MKL experiments**: compare the backends
within the new JSON, not their absolute timings with the earlier files.
All timings follow warm-up and rotate/reverse backend order. TIGRE returns
completed host results, so wall-clock timing includes the CUDA work and
transfers. Geometry and backend setup are recorded separately. Scalar CSR
storage is reported; peak GPU memory is not measured.

Return both `tigre-JOBID.log` and `tigre-JOBID.json`. Inspect `validation`
first; timing samples are in `benchmark.operations`. The JSON records package
versions, the TIGRE install source, repository commit, thread settings and
`nvidia-smi` output. A clean checkout makes the commit provenance easiest to
interpret. CPU SciPy sparse products are single-threaded; MKL uses the recorded
thread count.

## Other CUDA machines

Install this project with its SciPy extra and install CERN TIGRE using its
upstream instructions (a compatible CUDA toolkit and C++ compiler are needed
for a source build). The exact source used here can be installed with:

```bash
python -m pip install 'setuptools>=77' wheel Cython numpy scipy
python -m pip install --no-build-isolation \
  'tigre @ git+https://github.com/CERN/TIGRE.git@9195fd57faa1eba47dedfd45d95a9957d3910dbd'
python -m pip install '.[scipy]'
python benchmarks/tigre_tensor.py --validation-only --output tigre-validation.json
python benchmarks/tigre_tensor.py --size 64 --views 32 --repeat 5 --output tigre-results.json
```

The command without `--mkl-library` needs no MKL installation. It always uses
one visible GPU and respects `CUDA_VISIBLE_DEVICES`; run only inside an
appropriate GPU allocation on a cluster. There is no automatic CPU fallback.
The existing GitHub CPU runners exercise the algebra tests. No GPU runner or
paid Actions service is configured by this change.

## Sources

- [TIGRE Python Ax](https://github.com/CERN/TIGRE/blob/9195fd57faa1eba47dedfd45d95a9957d3910dbd/Python/tigre/utilities/Ax.py),
  [Atb](https://github.com/CERN/TIGRE/blob/9195fd57faa1eba47dedfd45d95a9957d3910dbd/Python/tigre/utilities/Atb.py),
  [parallel Siddon implementation](https://github.com/CERN/TIGRE/blob/9195fd57faa1eba47dedfd45d95a9957d3910dbd/Common/CUDA/Siddon_projection_parallel.cu).
- [TIGRE installation](https://github.com/CERN/TIGRE/blob/9195fd57faa1eba47dedfd45d95a9957d3910dbd/Frontispiece/python_installation.md).
- [CIL TIGRE documentation](https://tomographicimaging.github.io/CIL/v24.2.0/plugins/)
  describes `matched` as approximately matching the interpolated projector.
- [CSF3 GPU jobs](https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/gpu-jobs-slurm/)
  and [CUDA modules](https://ri.itservices.manchester.ac.uk/csf3/software/libraries/cuda/).
- [CUDA 12.8 host compiler support](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)
  includes GCC 14.
