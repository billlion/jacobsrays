# CSF3 cached oneMKL measurements

These are the unmodified JSON records supplied from two submissions of the
same Slurm job script on the University of Manchester CSF3 cluster on
27 October 2026:

| Guide label | Job | Raw record |
| --- | --- | --- |
| Run A | 23867575 | [mkl-23867575.json](mkl-23867575.json) |
| Run B | 23867786 | [mkl-23867786.json](mkl-23867786.json) |

Each job requested 16 CPUs on one node in the `multicore` partition and
tested 1, 2, 4, 8 and 16 MKL threads in separate worker processes. Both
records report an AMD EPYC 9634 CPU, Python 3.12.16, NumPy 2.3.5,
SciPy 1.17.0 and oneMKL 2026.2 (build 20260915). The GNU threading layer,
LP64 interface, passive OpenMP waiting and disabled dynamic thread
adjustment were used. The jobs had different permitted CPU sets, recorded
in each JSON's `affinity` field. Those numbers alone do not identify
physical topology, worker-thread placement or the cause of timing changes.

The benchmark command, from the repository root in the prepared environment,
was:

```sh
python benchmarks/mkl_probe.py \
    --threads 1 2 4 8 16 \
    --size 128 --rays 100000 --repeat 9 \
    > "mkl-${SLURM_JOB_ID}.json"
```

Use the Python executable belonging to the environment containing the
package, SciPy and MKL. The adapter defaults to loading
`lib/libmkl_rt.so.3` under that interpreter's `sys.prefix`.
See [the experiment notes](../../docs/mkl-experiment.md) for the adapter and
installation details. The JSON records retain all per-call samples and
correctness checks; benchmarks are not CI timing gates.

## Interpretation and provenance

The [user guide](../../docs/paper/jacobsrays.tex) reports the sum of the
separate forward and adjoint medians, converted from seconds to milliseconds.
These sums are not directly timed solver iterations. Setup is excluded from
the product timings and reported separately. LRT/TRT timings include their
surrounding tensor operations. For the scalar cached-transpose control, the
pair cost combines the `scipy_sorted` forward median and
`scipy_cached_transpose` adjoint median; the latter backend is not separately
timed for scalar forward projection.

The observed product and adjoint errors are discrete consistency checks,
not reconstruction errors. Memory figures describe this comparison process,
which retains several representations and work arrays, rather than a minimal
MKL-only implementation. A 16-thread run is not always faster than an
8-thread run; these two jobs do not establish a universal optimum.

The benchmark implementation inspected when preparing this documentation is
`benchmarks/mkl_probe.py` at repository revision
`59f7fd402dc29f408b917e4aa53e0e3cf5b2994f` (blob
`2a9cdf4c6212217c1a20e6189a82b4f31a5078c8`). The supplied JSON does not record
the checkout commit or node hostname, and the corresponding job logs were
not supplied with these two results. Consequently that inspected revision
is not asserted to be the exact revision executed on CSF3. The environment
and results stated above come from the raw records and the submitted-job
configuration supplied with the experiment.
