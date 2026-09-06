use crate::memory::RegisterCols;
use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use zkm_core_executor::Opcode;
use zkm_pcs::air::{BaseAirBuilder, ZKMAirBuilder};

use crate::{
    air::WordAirBuilder,
    operations::{AddOperation, KoalaBearWordRangeChecker},
};

use super::{BranchChip, BranchColumns};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the helper bool columns (a_eq_b,
///    a_gt_b, a_lt_b).
/// 3. It verifies the correct values of the helper bool columns based on op_a and op_b.
///
impl<AB> Air<AB> for BranchChip
where
    AB: ZKMAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &BranchColumns<AB::Var> = (*local).borrow();

        // SAFETY: All selectors `is_beq`, `is_bne`, `is_bltz`, `is_bgez`, `is_blez`, `is_bgtz` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the six selectors, is boolean.
        // Therefore, the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_beq);
        builder.assert_bool(local.is_bne);
        builder.assert_bool(local.is_bltz);
        builder.assert_bool(local.is_bgez);
        builder.assert_bool(local.is_blez);
        builder.assert_bool(local.is_bgtz);
        let is_real = local.is_beq
            + local.is_bne
            + local.is_bltz
            + local.is_bgez
            + local.is_blez
            + local.is_bgtz;
        builder.assert_bool(is_real.clone());

        let opcode = local.is_beq * Opcode::BEQ.as_field::<AB::F>()
            + local.is_bne * Opcode::BNE.as_field::<AB::F>()
            + local.is_bltz * Opcode::BLTZ.as_field::<AB::F>()
            + local.is_bgez * Opcode::BGEZ.as_field::<AB::F>()
            + local.is_blez * Opcode::BLEZ.as_field::<AB::F>()
            + local.is_bgtz * Opcode::BGTZ.as_field::<AB::F>();

        let _ = opcode;

        // A real instruction carries its own program fetch, register access and
        // `(clk, pc)` chaining.  A branch's next_next_pc is the TARGET (or the
        // fallthrough), already a constrained Word column; branches never halt.
        crate::frame::eval_i_type_frame(
            builder,
            &local.frame,
            local.is_beq * Opcode::BEQ.as_field::<AB::F>()
                + local.is_bne * Opcode::BNE.as_field::<AB::F>()
                + local.is_bltz * Opcode::BLTZ.as_field::<AB::F>()
                + local.is_bgez * Opcode::BGEZ.as_field::<AB::F>()
                + local.is_blez * Opcode::BLEZ.as_field::<AB::F>()
                + local.is_bgtz * Opcode::BGTZ.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            AB::Expr::ZERO,
            is_real.clone(),
        );
        // A branch READS op_a immutably: the register write carries the
        // previous value through unchanged.
        builder
            .when(is_real.clone())
            .assert_word_eq(*local.frame.op_a_access.value(), local.frame.op_a_access.prev_value);

        // Evaluate program counter constraints.
        {
            // Range check local.next_pc, local.next_next_pc and local.target_pc, .
            // SAFETY: `is_real` is already checked to be boolean.
            // The `KoalaBearWordRangeChecker` assumes that the value is checked to be a valid word.
            // This is done when the word form is relevant, i.e. when `pc` and `next_pc` are sent to the ADD ALU table.
            // The ADD ALU table checks the inputs are valid words, when it invokes `AddOperation`.
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

            // When we are branching, prove target = next_pc + c IN-ROW (the
            // AddSub request row is gone; the memory chips' inlined address
            // add set the precedent).
            AddOperation::<AB::F>::eval(
                builder,
                local.next_pc,
                local.frame.op_c_val(),
                local.target_add,
                local.is_branching.into(),
            );

            // When we are not branching, assert that local.next_pc + 4 <==> next.next_next_pc.
            builder.when(is_real.clone()).when_not(local.is_branching).assert_eq(
                local.next_pc.reduce::<AB>() + AB::Expr::from_u32(4),
                local.next_next_pc.reduce::<AB>(),
            );

            // check local.next_pc/next_next_pc to be valid word when we are not branching.
            // they are checked as valid value by the ADD ALU table when we are branching.
            builder.slice_range_check_u8(&local.next_pc.0, is_real.clone() - local.is_branching);
            builder
                .slice_range_check_u8(&local.next_next_pc.0, is_real.clone() - local.is_branching);

            // When we are branching, assert that next_next_pc is the target.
            builder
                .when(is_real.clone())
                .when(local.is_branching)
                .assert_word_eq(local.target_add.value, local.next_next_pc);

            // To prevent the ALU send above to be non-zero when the row is a padding row.
            builder.when_not(is_real.clone()).assert_zero(local.is_branching);

            // Assert the branching or not branching when the instruction is a
            builder.when(is_real.clone()).assert_bool(local.is_branching);
        }

        // ── The comparison helpers ───────────────────────────────────────
        //
        // A branch needs only EQUALITY (BEQ/BNE — and, since the zero-compare
        // opcodes read register 0 as `op_b`, `a_eq_b` doubles as `a == 0`
        // there) and the SIGN BIT of `op_a`.  Equality is two `IsZero`s over
        // the 16-bit limb differences: with byte-shaped words each difference
        // lies in `[-65535, 65535]`, so it vanishes in the field iff both of
        // its byte differences do.  (`op_a`'s bytes are range checked by the
        // frame; `op_b`'s value inherits byte shape from its writer through
        // the register file, exactly as the old `LtOperation` assumed.)
        let av = *local.frame.op_a_access.value();
        let bv = local.frame.op_b_val();
        let two_pow_8 = AB::Expr::from_u32(1 << 8);
        let d_lo = (av[0] - bv[0]) + (av[1] - bv[1]) * two_pow_8.clone();
        let d_hi = (av[2] - bv[2]) + (av[3] - bv[3]) * two_pow_8;

        // The standard IsZero pattern, guarded on real rows.
        builder.when(is_real.clone()).assert_zero(local.eq_lo * d_lo.clone());
        builder
            .when(is_real.clone())
            .assert_eq(local.eq_lo, AB::Expr::ONE - d_lo * local.eq_lo_inv);
        builder.when(is_real.clone()).assert_zero(local.eq_hi * d_hi.clone());
        builder
            .when(is_real.clone())
            .assert_eq(local.eq_hi, AB::Expr::ONE - d_hi * local.eq_hi_inv);
        builder.when(is_real.clone()).assert_eq(local.a_eq_b, local.eq_lo * local.eq_hi);

        // The sign bit, bound by one MSB byte lookup on exactly the rows that
        // consult it (each row has at most one selector on, so the
        // multiplicity is boolean).
        let zero_ops = local.is_bltz + local.is_blez + local.is_bgtz + local.is_bgez;
        builder.send_byte(
            AB::Expr::from_u8(zkm_core_executor::ByteOpcode::MSB as u8),
            local.msb_a,
            av[3],
            AB::Expr::ZERO,
            zero_ops.clone(),
        );
        // `a > 0` signed on a zero-compare row: not negative and not zero.
        builder.when(zero_ops).assert_eq(
            local.a_gt_0,
            (AB::Expr::ONE - local.msb_a) * (AB::Expr::ONE - local.a_eq_b),
        );

        // ── The branching decision, per opcode ───────────────────────────
        // BEQ branches iff a == b.
        builder.when(local.is_beq * local.is_branching).assert_one(local.a_eq_b);
        builder.when(local.is_beq).when_not(local.is_branching).assert_zero(local.a_eq_b);
        // BNE branches iff a != b.
        builder.when(local.is_bne * local.is_branching).assert_zero(local.a_eq_b);
        builder.when(local.is_bne).when_not(local.is_branching).assert_one(local.a_eq_b);
        // BLTZ branches iff a < 0, i.e. the sign bit.
        builder.when(local.is_bltz * local.is_branching).assert_one(local.msb_a);
        builder.when(local.is_bltz).when_not(local.is_branching).assert_zero(local.msb_a);
        // BLEZ branches iff a <= 0, i.e. NOT (a > 0).
        builder.when(local.is_blez * local.is_branching).assert_zero(local.a_gt_0);
        builder.when(local.is_blez).when_not(local.is_branching).assert_one(local.a_gt_0);
        // BGTZ branches iff a > 0.
        builder.when(local.is_bgtz * local.is_branching).assert_one(local.a_gt_0);
        builder.when(local.is_bgtz).when_not(local.is_branching).assert_zero(local.a_gt_0);
        // BGEZ branches iff a >= 0, i.e. NOT the sign bit.
        builder.when(local.is_bgez * local.is_branching).assert_zero(local.msb_a);
        builder.when(local.is_bgez).when_not(local.is_branching).assert_one(local.msb_a);
    }
}
