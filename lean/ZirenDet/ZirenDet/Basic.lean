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

/-- Closing tactic for generated determinism / postcondition theorems. -/
syntax "picus_det" : tactic
macro_rules
  | `(tactic| picus_det) =>
    `(tactic| first
        | (intros; rfl)
        | (intros; simp_all only [List.cons.injEq, List.nil_eq, and_true, true_and]; done)
        | (intros; simp_all; done)
        | sorry)
