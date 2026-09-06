import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# Ziren determinism prelude

The AIRs are stated over the KoalaBear prime field.  `picus_det` is the closing tactic used by
the generated theorems: it tries the cheap closers and otherwise leaves a `sorry`, so generated
files always elaborate and the remaining obligations show up as warnings.
-/

namespace ZirenDet

/-- The KoalaBear prime `2^31 - 2^24 + 1`. -/
abbrev KB : ℕ := 2130706433

-- `KB` is prime; the `norm_num` certificate for a 31-bit prime needs a deeper kernel recursion
-- limit than the default (it checks in about 3 s).
set_option maxRecDepth 100000 in
instance : Fact (Nat.Prime KB) := ⟨by norm_num⟩

/-- The base field of every Ziren AIR: `ZMod KB` is a `Field` through the instance above.
Division in extracted constraints is rendered as multiplication by the inverse. -/
abbrev F := ZMod KB

end ZirenDet

/-- Closing tactic for generated determinism / postcondition theorems.

Strategy: split both witness records into their fields, unfold the generated definitions,
turn every `a - b = 0` into `a = b`, split conjunctions, substitute every variable that is
defined by an equation (`subst_vars` propagates the linear definitions and the input
equalities), and close what is left by `rfl` / `ring`.  Anything that needs case analysis on
guarded bits is left as `sorry` so the file still elaborates and the obligation is visible. -/
syntax "picus_det" : tactic
-- Hygiene off: the tactic must see the use-site `constraints`, `inputs`, `W`, `w`, … of the
-- module it closes, not names resolved at this definition site.
set_option hygiene false in
macro_rules
  | `(tactic| picus_det) =>
    `(tactic| first
        | (intros; rfl)
        | (intros
           all_goals (try (rename_i w w' hw hw' hin hassume))
           all_goals (try cases w)
           all_goals (try cases w')
           simp only [constraints, inputs, outputs, assumed, rel, List.cons.injEq, List.nil_eq,
             and_true, true_and, sub_eq_zero, W.mk.injEq] at *
           all_goals (try casesm* _ ∧ _)
           all_goals (try subst_vars)
           all_goals (try constructorm* _ ∧ _)
           all_goals (first | rfl | ring | (simp only [mul_comm, mul_assoc, mul_left_comm, add_comm, add_assoc, add_left_comm]; done) | sorry))
        | sorry)
