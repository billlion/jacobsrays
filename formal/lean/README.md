# A first Lean proof for jacobsrays

This optional Lean 4 project proves the **one-coordinate initialisation and
next-crossing lemmas in exact real arithmetic**. It is the first small part of
the traversal proof. It does not yet prove clipping, the multi-coordinate loop,
termination, or any property of the C++ floating-point execution or wrappers.
Lean is not a dependency of the Python, MATLAB or C++ packages.

The notation follows [Appendix A of the written exact-arithmetic proof](https://github.com/billlion/jacobsrays/blob/21d6b536054308d5576307be32dbcb20c8f62748/docs/paper/exact-arithmetic.tex),
introduced in documentation PR #12.

## What is proved?

At entry, let `x` be one coordinate in voxel units and let `v` be its velocity
with respect to the ray parameter. After a parameter increase `u`, the
coordinate is `x + u*v`. The voxel with integer index `c` is `[c, c+1)`.

For positive motion, define

```text
c = floor(x)
delta = (c + 1 - x) / v
```

For negative motion, define

```text
c = ceil(x) - 1
delta = (c - x) / v
```

Here `delta` is the increase in parameter to the next plane, not a physical
distance. In the appendix notation, `x = q_i + alpha*v_i` and the absolute
crossing parameter is `alpha + delta`. The final ray endpoint may occur before
that crossing; the full traversal will take the minimum with that endpoint.

[Coordinate.lean](JacobsRays/Coordinate.lean) proves:

| Theorem | Assumptions | Conclusion |
| --- | --- | --- |
| `forward_correct` | `0 <= x < n`, `v > 0` | Valid integer cell index, `delta > 0`, membership for every `0 < u < delta`, arrival at the upper face |
| `reverse_correct` | `0 < x <= n`, `v < 0` | Valid integer cell index, `delta > 0`, membership for every `0 < u < delta`, arrival at the lower face |
| `stationary_correct` | `0 <= x < n`, velocity zero | Valid integer cell index and unchanged membership for every real `u` |

`n` is an integer voxel count. The displayed input bounds already imply `n > 0`,
so a separate positivity hypothesis is unnecessary. These are universal theorems
over mathematical real numbers and integers, not checks of selected inputs.

The open condition `0 < u` deliberately excludes the entry point. At `x = 2`,
positive motion starts in cell 2 and negative motion in cell 1, although the
point 2 itself belongs to cell 2. Both examples are also checked in the file.
The stationary case uses floor, matching the half-open boundary convention.

## Running it

Install VS Code and its **Lean 4** extension using the
[official installation instructions](https://lean-lang.org/install/).
Open this `formal/lean` directory as the project folder. The extension uses
`elan`, Lean's version manager, to select the version in `lean-toolchain`.

From this directory, run:

```sh
lake exe cache get Mathlib.Data.Real.Archimedean Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
lake build --wfail
lake env lean -DwarningAsError=true Audit.lean
```

`lake` is Lean's build tool, analogous to `make`. The first command obtains
compiled Mathlib dependencies; the initial download is much larger than this
proof. Subsequent builds reuse them. The project pins Lean 4.24.0 and the
matching Mathlib release by commit, with transitive dependencies recorded in
`lake-manifest.json`. It does not track the moving Mathlib development branch.

GitHub's **Lean proofs** workflow runs the same checks when this project or its
workflow changes. Ordinary numerical builds do not install Lean.

## Reading the first proof

Start with `forward_index_valid`. Mathematically it says

```text
0 <= x < n  implies  0 <= floor(x) < n.
```

Its Lean statement is:

```lean
theorem forward_index_valid {n : ℤ} {x : ℝ}
    (hx0 : 0 ≤ x) (hxn : x < (n : ℝ)) :
    0 ≤ forwardCell x ∧ forwardCell x < n := by
  constructor
  · exact Int.floor_nonneg.mpr hx0
  · exact Int.floor_lt.mpr hxn
```

- `{n : ℤ}` and `{x : ℝ}` declare an integer and a real variable. Braces let
  Lean infer these arguments from the hypotheses when the theorem is used.
- `hx0` and `hxn` name the two assumptions. A name is useful because a later
  proof step can refer to that particular assumption.
- `(n : ℝ)` explicitly views the integer `n` as a real number for comparison
  with `x`. The cell index itself stays an integer.
- The colon before the conclusion introduces what we promise to prove.
  `∧` means “and”. `:= by` starts the proof.
- `constructor` splits the conjunction into two goals; the two `·` blocks
  prove them separately.
- `Int.floor_nonneg` is the existing equivalence between nonnegative floor
  and nonnegative argument. `.mpr` uses that equivalence from right to left.
  `Int.floor_lt` similarly connects `floor(x) < n` with `x < n`.
- `exact` supplies a proof of the current goal. Lean unfolds the definition
  `forwardCell x = floor(x)` where needed.

In VS Code, place the cursor after each step and watch **Lean Infoview** show
the remaining goals. For example, after `constructor` there are two; after
the first `exact`, only the upper-bound goal remains.

Next read `forward_step_pos` and `forward_inside`. They use the familiar
inequalities `floor(x) <= x < floor(x)+1`. `have` names an intermediate fact;
`linarith` finishes a consequence of linear inequalities. The reversed motion
proof uses ceiling and reverses the inequality when multiplying by `v < 0`.
`omega` handles integer inequalities, and `ring` checks a polynomial identity.
These tactics construct proofs checked by Lean's kernel.

The definitions use `noncomputable` because they describe exact real-number
operations in a mathematical model. They are not executable replacements for
the C++ kernel. This does not weaken the theorems about those definitions.

## What counts as a completed proof?

Successful elaboration alone can include an explicitly admitted gap such as
`sorry`. [Audit.lean](Audit.lean) checks the dependencies of the three combined
correctness theorems, including their supporting lemmas, and fails if any axiom
other than `propext`, `Classical.choice` or `Quot.sound` occurs. Those are Lean's
standard logical axioms. No additional axiom is introduced by this project.

For an individual theorem, the standard command is:

```lean
#print axioms JacobsRays.forward_correct
```

See Lean's [proof-validation guide](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
Human review still has to establish that our theorem statements describe the
algorithm and conventions we intend. The next proof work is combining the
coordinate results, treating simultaneous crossings, and establishing the
loop invariant and termination. Relating that model to C++ machine arithmetic
is a separate subsequent task.
