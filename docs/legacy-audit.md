# Audit of the LRTinMatlab ray implementation

Reviewed on 2026-10-26 at commit
`4e61af83e3d9cfe58305cb1b0fe95532ca1aea94` in
[billlion/LRTinMatlab](https://github.com/billlion/LRTinMatlab).
The historical files credit David Szotten (2005–2008) and carry a University
of Manchester copyright notice. No licence file was present in the inspected
tree. The new repository retains its existing Apache-2.0 licence and contains
new source rather than a copy of those files. Acknowledgement is in NOTICE.

## Correctness

In `jacobsrays/mod_jacobs_rays.c`, the final partial-voxel block:

- Uses `j*im_size_y` where x-fastest flattening needs `j*im_size_x`.
- Stores a parameter interval without multiplying by the physical ray length.

A compiled reproduction with shape `(3,2,1)`, unit spacing, zero lower corner,
start `(.2,1.2,.5)` and end `(.8,1.2,.5)` returned zero-based index **2** and
length **1**, instead of index **3** and length **0.6**. This is covered by a
regression test in both the C++ and Python suites and in the MATLAB test file.

The MATLAB wrapper sets its matrix size by assigning zero to its final element,
which can overwrite a genuine intersection. The replacement constructs a sparse
matrix with explicit dimensions. It also preserves all-miss and empty batches.

The old absolute `1e-8` direction tolerance makes behaviour depend on coordinate
units and can discard short nonzero segments. The replacement treats only an
exactly zero component as parallel. Corner ties and roundoff still require
care: exact ties advance all matching axes; near ties may produce very short
positive intervals rather than being merged by a fixed threshold.

## Interface and memory

The old MATLAB help describes a grid-centre offset and fractional weights.
The supplied arguments actually reach the core as a **lower-corner offset**,
and the main loop produces **physical lengths**. The replacement documents
those actual conventions and makes defaults explicit.

The MEX gateway allocates a worst-case double triplet buffer for all rays,
then allocates a second output and copies into it. Index and size arithmetic
uses `int` in several places; input checks do not fully validate shapes or
classes. The replacement uses checked 64-bit indexing, validates input before
traversal, and counts intersections before allocating tightly sized arrays.
It outputs interval parameters as well as lengths, so its total storage must
be compared on equal output requirements (see performance notes).

The old core does produce traversal order in its internal triples. Converting
them to a sparse matrix loses that useful ordering contract. The new primary
API preserves order independently of sparse storage.

`MatrixLRT.m` constructs a dense matrix by probing each pixel/component with
repeated forward calls. The new tensor operator builds scalar intersections
once and reuses directional contractions for all tensor components.

## Scope

No files in LRTinMatlab were modified. The new MATLAB compatibility function
preserves the old call signature and core lower-corner convention; it does
not reproduce known wrong coefficients or the erroneous help text. The MEX
wrappers in the new repository still require execution in MATLAB before a
release can be considered validated on that platform.
