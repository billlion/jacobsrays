import JacobsRays
import Lean.Util.CollectAxioms

/-
Check dependencies transitively, including imported lemmas. Allow only Lean's
three standard logical axioms: no unfinished proofs or additional assumptions.
-/
open Lean Elab Command in
run_cmd do
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  for decl in #[``JacobsRays.forward_correct, ``JacobsRays.reverse_correct,
      ``JacobsRays.stationary_correct] do
    let axioms ← Lean.collectAxioms decl
    for ax in axioms do
      unless allowed.contains ax do
        throwError "{decl} depends on unapproved axiom {ax}"
    logInfo m!"{decl}: axiom audit passed ({axioms})"
