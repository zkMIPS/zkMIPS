use crate::memory::RegisterCols;
use std::borrow::Borrow;
use zkm_pcs::air::BaseAirBuilder;

use p3_air::{Air, AirBuilder, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use zkm_core_executor::Opcode;
use zkm_pcs::air::ZKMAirBuilder;

use crate::air::WordAirBuilder;

use crate::operations::KoalaBearWordRangeChecker;

use super::{JumpChip, JumpColumns};

impl<AB> Air<AB> for JumpChip
where
    AB: ZKMAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &JumpColumns<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_jump`, `is_jumpi`, `is_jumpdirect`  are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_jump + is_jumpi + is_jumpdirect` is boolean.
        // Therefore, the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_jump);
        builder.assert_bool(local.is_jumpi);
        builder.assert_bool(local.is_jumpdirect);
        let is_real = local.is_jump + local.is_jumpi + local.is_jumpdirect;
        builder.assert_bool(is_real.clone());

        let opcode = local.is_jump * Opcode::Jump.as_field::<AB::F>()
            + local.is_jumpi * Opcode::Jumpi.as_field::<AB::F>()
            + local.is_jumpdirect * Opcode::JumpDirect.as_field::<AB::F>();

        let _ = opcode;

        // A real instruction carries its own program fetch, register access and
        // `(clk, pc)` chaining.  A jump's next_next_pc is the TARGET; jumps
        // WRITE the link register, so op_a_immutable stays 0; never halt.
        crate::frame::eval_instruction_frame(
            builder,
            &local.frame,
            local.is_jump * Opcode::Jump.as_field::<AB::F>()
                + local.is_jumpi * Opcode::Jumpi.as_field::<AB::F>()
                + local.is_jumpdirect * Opcode::JumpDirect.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            AB::Expr::ZERO,
            is_real.clone(),
        );

        // The link value lives directly in the frame's committed `op_a`
        // register access — there is no separate link column any more.  A
        // no-link jump discards the write and the frame pins the commit to
        // ZERO, so the link equation is gated by `op_a_0`; on those rows the
        // committed zero still passes the word range check below.
        let link = *local.frame.op_a_access.value();
        builder
            .when(is_real.clone())
            .when_not(local.frame.instruction.op_a_0)
            .assert_eq(link.reduce::<AB>(), local.next_pc.reduce::<AB>() + AB::F::from_u32(4));

        // Range check the link, next_pc, and next_next_pc.
        // SAFETY: `is_real` is already checked to be boolean.
        // The frame range checks the committed word's BYTES; the KoalaBear
        // check on top makes the `reduce()` binding above canonical.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            link,
            local.op_a_range_checker,
            is_real.clone(),
        );
        // SAFETY: `is_real` is already checked to be boolean.
        // `local.next_pc`, `local.next_next_pc` are checked to a valid word when relevant.
        // This is due to the ADD ALU table checking all inputs and outputs are valid words.
        // This is done when the `AddOperation` is invoked in the ADD ALU table.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            is_real.clone(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_next_pc,
            local.next_next_pc_range_checker,
            is_real.clone(),
        );

        // We now constrain `next_next_pc` for J/JR/JALR.
        builder
            .when(local.is_jump + local.is_jumpi)
            .assert_word_eq(local.next_next_pc, local.frame.op_b_val());

        // Verify that the next_next_pc is calculated correctly for BAL
        // instructions, IN-ROW (the AddSub request row is gone).
        // SAFETY: `is_jumpdirect` is boolean, and zero for padding rows.
        crate::operations::AddOperation::<AB::F>::eval(
            builder,
            local.next_pc,
            local.frame.op_b_val(),
            local.target_add,
            local.is_jumpdirect.into(),
        );
        builder
            .when(local.is_jumpdirect)
            .assert_word_eq(local.target_add.value, local.next_next_pc);
    }
}
