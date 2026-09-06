import ZirenDet.Chips.AddSub
import ZirenDet.Isa

/-!
# Bridge: the AddSub chip computes the ISA's `alu`

The generated determinism theorems say the chip's outputs are a function of its inputs.  This
file says *which* function: the one the executor implements, as modelled (and oracle-checked)
in `ZirenDet.Isa`.  The ports are the provenance-named accessors of the generated module: the
register operands are the values the row reads (`in_mem_read_0_val` = `$op_b`,
`in_mem_read_1_val` = `$op_c`) and the result is the value it writes (`out_mem_write_2_val` =
`$op_a`), all as four little-endian byte limbs.
-/

namespace ZirenDet.Bridge.AddSub
open ZirenDet ZirenDet.Chips.AddSub

/-- Little-endian byte limbs to a machine word. -/
def toWord (l : List F) : Isa.W :=
  BitVec.ofNat 32 (l.foldr (fun x acc => x.val + 256 * acc) 0)

/-- On a real `ADD` row that writes a register other than `$zero` (`op_a_0 = 0`), the written
word is the ISA sum of the two register operands. -/
theorem add_computes_alu (w : AddSub_is_add.W) (hw : AddSub_is_add.constraints w)
    (h_op_a_0 : AddSub_is_add.in_program_op_a_0 w = 0) :
    toWord (AddSub_is_add.out_mem_write_2_val w) =
      Isa.alu Isa.Opc.ADD (toWord (AddSub_is_add.in_mem_read_0_val w))
        (toWord (AddSub_is_add.in_mem_read_1_val w)) := by
  sorry

/-- Same for `SUB`. -/
theorem sub_computes_alu (w : AddSub_is_sub.W) (hw : AddSub_is_sub.constraints w)
    (h_op_a_0 : AddSub_is_sub.in_program_op_a_0 w = 0) :
    toWord (AddSub_is_sub.out_mem_write_2_val w) =
      Isa.alu Isa.Opc.SUB (toWord (AddSub_is_sub.in_mem_read_0_val w))
        (toWord (AddSub_is_sub.in_mem_read_1_val w)) := by
  sorry

end ZirenDet.Bridge.AddSub
