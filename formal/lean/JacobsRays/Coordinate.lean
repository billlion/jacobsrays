/-
Copyright (c) 2026 jacobsrays contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Real.Archimedean
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# One-coordinate initialisation and next crossing

This is an exact-real model of the floor/ceiling rules in `traverse.hpp`.
It does not verify floating-point execution or the full traversal loop.

`x` is the coordinate at entry in voxel units, `v` its velocity with respect
to the ray parameter, and `u` the increase in that parameter since entry.
The voxel with integer index `c` is `[c, c + 1)`.
-/

namespace JacobsRays

/-- Initial voxel index when moving in the positive direction. -/
noncomputable def forwardCell (x : ℝ) : ℤ := ⌊x⌋

/-- Initial voxel index when moving in the negative direction. -/
noncomputable def reverseCell (x : ℝ) : ℤ := ⌈x⌉ - 1

/-- Parameter increase to the upper face, used only when `0 < v`. -/
noncomputable def forwardStep (x v : ℝ) : ℝ :=
  ((forwardCell x : ℝ) + 1 - x) / v

/-- Parameter increase to the lower face, used only when `v < 0`. -/
noncomputable def reverseStep (x v : ℝ) : ℝ :=
  ((reverseCell x : ℝ) - x) / v

/-- The positive-direction initial index is in the grid. -/
theorem forward_index_valid {n : ℤ} {x : ℝ}
    (hx0 : 0 ≤ x) (hxn : x < (n : ℝ)) :
    0 ≤ forwardCell x ∧ forwardCell x < n := by
  constructor
  · exact Int.floor_nonneg.mpr hx0
  · exact Int.floor_lt.mpr hxn

/-- The negative-direction initial index is in the grid. -/
theorem reverse_index_valid {n : ℤ} {x : ℝ}
    (hx0 : 0 < x) (hxn : x ≤ (n : ℝ)) :
    0 ≤ reverseCell x ∧ reverseCell x < n := by
  have hpos : 0 < ⌈x⌉ := Int.ceil_pos.mpr hx0
  have hle : ⌈x⌉ ≤ n := Int.ceil_le.mpr hxn
  dsimp [reverseCell]
  constructor <;> omega

/-- For positive motion, the next crossing lies strictly ahead. -/
theorem forward_step_pos {x v : ℝ} (hv : 0 < v) :
    0 < forwardStep x v := by
  have hface : x < (forwardCell x : ℝ) + 1 := Int.lt_floor_add_one x
  exact div_pos (sub_pos.mpr hface) hv

/-- Positive motion stays in the initial voxel before its next crossing. -/
theorem forward_inside {x v u : ℝ} (hv : 0 < v)
    (hu : 0 < u) (hnext : u < forwardStep x v) :
    (forwardCell x : ℝ) ≤ x + u * v ∧
      x + u * v < (forwardCell x : ℝ) + 1 := by
  have hfloor : (forwardCell x : ℝ) ≤ x := Int.floor_le x
  have hmove : 0 < u * v := mul_pos hu hv
  have hbefore : u * v < (forwardCell x : ℝ) + 1 - x :=
    (lt_div_iff₀ hv).mp hnext
  constructor <;> linarith

/-- At the positive-direction crossing the ray reaches the upper face. -/
theorem forward_at_face {x v : ℝ} (hv : 0 < v) :
    x + forwardStep x v * v = (forwardCell x : ℝ) + 1 := by
  unfold forwardStep
  rw [div_mul_cancel₀ _ (ne_of_gt hv)]
  ring

/-- For negative motion, the next crossing also lies strictly ahead. -/
theorem reverse_step_pos {x v : ℝ} (hv : v < 0) :
    0 < reverseStep x v := by
  have hceil := Int.ceil_lt_add_one x
  have hface : (reverseCell x : ℝ) < x := by
    dsimp [reverseCell]
    push_cast
    linarith
  exact div_pos_of_neg_of_neg (sub_neg.mpr hface) hv

/-- Negative motion stays in the initial voxel before its next crossing.

The open condition `0 < u` matters: if entry is on an integer plane, that
point itself belongs to the higher-index voxel under the half-open rule.
-/
theorem reverse_inside {x v u : ℝ} (hv : v < 0)
    (hu : 0 < u) (hnext : u < reverseStep x v) :
    (reverseCell x : ℝ) ≤ x + u * v ∧
      x + u * v < (reverseCell x : ℝ) + 1 := by
  have hceil : x ≤ (reverseCell x : ℝ) + 1 := by
    dsimp [reverseCell]
    push_cast
    linarith [Int.le_ceil x]
  have hmove : u * v < 0 := mul_neg_of_pos_of_neg hu hv
  have hbefore : (reverseCell x : ℝ) - x < u * v :=
    (lt_div_iff_of_neg hv).mp hnext
  constructor <;> linarith

/-- At the negative-direction crossing the ray reaches the lower face. -/
theorem reverse_at_face {x v : ℝ} (hv : v < 0) :
    x + reverseStep x v * v = (reverseCell x : ℝ) := by
  unfold reverseStep
  rw [div_mul_cancel₀ _ (ne_of_lt hv)]
  ring

/-- A stationary coordinate keeps the half-open voxel selected by floor. -/
theorem stationary_inside (x u : ℝ) :
    (forwardCell x : ℝ) ≤ x + u * 0 ∧
      x + u * 0 < (forwardCell x : ℝ) + 1 := by
  simpa only [forwardCell, mul_zero, add_zero] using
    And.intro (Int.floor_le x) (Int.lt_floor_add_one x)

/-- All four positive-direction guarantees, packaged as one theorem. -/
theorem forward_correct {n : ℤ} {x v : ℝ}
    (hx0 : 0 ≤ x) (hxn : x < (n : ℝ)) (hv : 0 < v) :
    (0 ≤ forwardCell x ∧ forwardCell x < n) ∧
    0 < forwardStep x v ∧
    (∀ u : ℝ, 0 < u → u < forwardStep x v →
      (forwardCell x : ℝ) ≤ x + u * v ∧
        x + u * v < (forwardCell x : ℝ) + 1) ∧
    x + forwardStep x v * v = (forwardCell x : ℝ) + 1 := by
  exact ⟨forward_index_valid hx0 hxn, forward_step_pos hv,
    fun _ hu hnext => forward_inside hv hu hnext, forward_at_face hv⟩

/-- All four negative-direction guarantees, packaged as one theorem. -/
theorem reverse_correct {n : ℤ} {x v : ℝ}
    (hx0 : 0 < x) (hxn : x ≤ (n : ℝ)) (hv : v < 0) :
    (0 ≤ reverseCell x ∧ reverseCell x < n) ∧
    0 < reverseStep x v ∧
    (∀ u : ℝ, 0 < u → u < reverseStep x v →
      (reverseCell x : ℝ) ≤ x + u * v ∧
        x + u * v < (reverseCell x : ℝ) + 1) ∧
    x + reverseStep x v * v = (reverseCell x : ℝ) := by
  exact ⟨reverse_index_valid hx0 hxn, reverse_step_pos hv,
    fun _ hu hnext => reverse_inside hv hu hnext, reverse_at_face hv⟩

/-- A stationary coordinate has a valid, unchanged voxel for every parameter. -/
theorem stationary_correct {n : ℤ} {x : ℝ}
    (hx0 : 0 ≤ x) (hxn : x < (n : ℝ)) :
    (0 ≤ forwardCell x ∧ forwardCell x < n) ∧
    (∀ u : ℝ, (forwardCell x : ℝ) ≤ x + u * 0 ∧
      x + u * 0 < (forwardCell x : ℝ) + 1) := by
  exact ⟨forward_index_valid hx0 hxn, stationary_inside x⟩

-- The boundary example: motion to the right starts in cell 2; to the left, cell 1.
example : forwardCell 2 = 2 := by norm_num [forwardCell]
example : reverseCell 2 = 1 := by norm_num [reverseCell]

end JacobsRays
