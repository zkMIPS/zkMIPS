use crate::memory::RegisterCols;
use std::borrow::Borrow;
use zkm_pcs::air::BaseAirBuilder;

use crate::{memory::MemoryCols, operations::IsEqualWordOperation};
use p3_air::{Air, AirBuilder, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use zkm_core_executor::{events::MemoryAccessPosition, ByteOpcode, Opcode};
use zkm_pcs::{air::ZKMAirBuilder, Word};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    operations::AddDoubleOperation,
};

use super::{columns::MiscInstrColumns, MiscInstrsChip};

impl<AB> Air<AB> for MiscInstrsChip
where
    AB: ZKMAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &MiscInstrColumns<AB::Var> = (*local).borrow();

        let is_real = local.is_sext
            + local.is_ins
            + local.is_ext
            + local.is_maddu
            + local.is_msubu
            + local.is_madd
            + local.is_msub
            + local.is_teq;

        builder.assert_bool(local.is_sext);
        builder.assert_bool(local.is_ins);
        builder.assert_bool(local.is_ext);
        builder.assert_bool(local.is_maddu);
        builder.assert_bool(local.is_msubu);
        builder.assert_bool(local.is_madd);
        builder.assert_bool(local.is_msub);
        builder.assert_bool(local.is_teq);
        builder.assert_bool(is_real.clone());

        let is_check_memory = local.is_maddu + local.is_msubu + local.is_madd + local.is_msub;

        // The Instruction-bus receives are gone: every row is a real
        // instruction serving itself via the frame.  Misc instructions are
        // sequential, never halt.
        crate::frame::eval_instruction_frame(
            builder,
            &local.frame,
            local.is_sext * Opcode::SEXT.as_field::<AB::F>()
                + local.is_ext * Opcode::EXT.as_field::<AB::F>()
                + local.is_ins * Opcode::INS.as_field::<AB::F>()
                + local.is_maddu * Opcode::MADDU.as_field::<AB::F>()
                + local.is_msubu * Opcode::MSUBU.as_field::<AB::F>()
                + local.is_madd * Opcode::MADD.as_field::<AB::F>()
                + local.is_msub * Opcode::MSUB.as_field::<AB::F>()
                + local.is_teq * Opcode::TEQ.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            is_real.clone(),
        );
        // TEQ reads op_a immutably: the register write carries the previous
        // value through unchanged.
        builder
            .when(local.is_teq)
            .assert_word_eq(*local.frame.op_a_access.value(), local.frame.op_a_access.prev_value);
        // Bind `prev_a_value` to the access's previous value EXACTLY on the
        // rows that use it (the read-write group: MADD family + INS).  SEXT /
        // EXT / TEQ pin the column to zero below while the register's actual
        // old value is arbitrary — binding them would be unsatisfiable.
        builder
            .when(is_check_memory.clone() + local.is_ins)
            .assert_word_eq(local.prev_a_value, local.frame.op_a_access.prev_value);
        // Bind this chip's operand columns to the frame's register-file view:
        // the chip must compute on exactly the values the register accesses
        // commit.  Writes are gated by op_a_0 (discarded); a TEQ READ of
        // register 0 must see 0.
        builder
            .when(is_real.clone())
            .when_not(local.frame.instruction.op_a_0)
            .assert_word_eq(local.op_a_value, *local.frame.op_a_access.value());
        builder
            .when(local.is_teq)
            .when(local.frame.instruction.op_a_0)
            .assert_word_zero(local.op_a_value);

        self.eval_ext(builder, local);
        self.eval_ins(builder, local);
        self.eval_maddsub(builder, local);
        self.eval_sext(builder, local);

        builder
            .when(local.is_sext + local.is_ext + local.is_teq)
            .assert_word_zero(local.prev_a_value);
        builder.when(local.is_ins + local.is_ext).assert_zero(local.frame.op_c_val()[2]);
        builder.when(local.is_ins + local.is_ext).assert_zero(local.frame.op_c_val()[3]);
    }
}

impl MiscInstrsChip {
    pub(crate) fn eval_sext<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MiscInstrColumns<AB::Var>,
    ) {
        let sext_cols = local.misc_specific_columns.sext();

        // Check that a != b when `is_teq` is enabled
        IsEqualWordOperation::<AB::F>::eval(
            builder,
            local.op_a_value.map(|x| x.into()),
            local.frame.op_b_val().map(|x| x.into()),
            sext_cols.a_eq_b,
            local.is_teq.into(),
        );
        let a_eq_b = sext_cols.a_eq_b.is_diff_zero.result;
        builder.when(local.is_teq).assert_zero(a_eq_b);

        // most_sig_bit is bit 7 of sig_byte.
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            sext_cols.most_sig_bit,
            sext_cols.sig_byte,
            AB::Expr::zero(),
            local.is_sext,
        );

        // op_c can be 0 (for seb) and 1(for seh).
        builder.when(local.is_sext).assert_bool(local.frame.op_c_val()[0]);
        builder.when(local.is_sext).assert_bool(sext_cols.is_seb);
        builder.when(local.is_sext).assert_bool(sext_cols.is_seh);
        builder.when(local.is_sext).assert_one(sext_cols.is_seh + sext_cols.is_seb);

        builder.when(local.is_sext).when(sext_cols.is_seb).assert_zero(local.frame.op_c_val()[0]);
        builder.when(local.is_sext).when(sext_cols.is_seh).assert_one(local.frame.op_c_val()[0]);

        // For seb, sig_byte is byte 0 of op_a.
        // For seh, sig_byte is byte 1 of op_a.
        {
            builder
                .when(local.is_sext)
                .when(sext_cols.is_seb)
                .assert_eq(local.frame.op_b_val()[0], sext_cols.sig_byte);

            builder
                .when(local.is_sext)
                .when(sext_cols.is_seh)
                .assert_eq(local.frame.op_b_val()[1], sext_cols.sig_byte);
        }

        // Constraints for result value:
        // For both seb and seh, bytes lower than sig_byte(contain) equal op_b,
        // bytes upper than sig_byte equal sign byte(0xff when sig_bit is 1, otherwise 0).
        {
            let sign_byte = AB::Expr::from_u8(0xFF) * sext_cols.most_sig_bit;

            builder.when(local.is_sext).assert_eq(local.op_a_value[0], local.frame.op_b_val()[0]);

            builder
                .when(local.is_sext)
                .when(sext_cols.is_seb)
                .assert_eq(local.op_a_value[1], sign_byte.clone());

            builder
                .when(local.is_sext)
                .when(sext_cols.is_seh)
                .assert_eq(local.op_a_value[1], local.frame.op_b_val()[1]);

            builder.when(local.is_sext).assert_eq(local.op_a_value[2], sign_byte.clone());

            builder.when(local.is_sext).assert_eq(local.op_a_value[3], sign_byte);
        }
    }

    pub(crate) fn eval_maddsub<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MiscInstrColumns<AB::Var>,
    ) {
        let maddsub_cols = local.misc_specific_columns.maddsub();
        let is_real = local.is_maddu + local.is_msubu + local.is_madd + local.is_msub;
        let is_sign = local.is_madd + local.is_msub;
        let is_unsign = local.is_maddu + local.is_msubu;
        let is_add = local.is_maddu + local.is_madd;
        let is_sub = local.is_msubu + local.is_msub;

        // Prove op_b * op_c IN-ROW (the MULT/MULTU request row is gone).
        crate::operations::MulOperation::<AB::F>::eval(
            builder,
            local.frame.op_b_val(),
            local.frame.op_c_val(),
            &local.maddsub_mul,
            is_sign,
            is_unsign,
        );
        let mul_lo = local.maddsub_mul.lo();
        let mul_hi = local.maddsub_mul.hi();

        for i in 0..WORD_SIZE {
            builder.when(is_real.clone()).assert_eq(
                maddsub_cols.src2_hi[i],
                maddsub_cols.op_hi_access.prev_value[i] * is_add.clone()
                    + (*maddsub_cols.op_hi_access.value())[i] * is_sub.clone(),
            );
            builder.when(is_real.clone()).assert_eq(
                maddsub_cols.src2_lo[i],
                local.prev_a_value[i] * is_add.clone() + local.op_a_value[i] * is_sub.clone(),
            );
        }

        AddDoubleOperation::<AB::F>::eval(
            builder,
            mul_lo,
            mul_hi,
            maddsub_cols.src2_lo,
            maddsub_cols.src2_hi,
            maddsub_cols.add_operation,
            is_real.clone(),
        );

        builder
            .when(is_add.clone())
            .assert_word_eq(local.op_a_value, maddsub_cols.add_operation.value);

        builder.when(is_add).assert_word_eq(
            *maddsub_cols.op_hi_access.value(),
            maddsub_cols.add_operation.value_hi,
        );

        builder
            .when(is_sub.clone())
            .assert_word_eq(local.prev_a_value, maddsub_cols.add_operation.value);

        builder.when(is_sub).assert_word_eq(
            maddsub_cols.op_hi_access.prev_value,
            maddsub_cols.add_operation.value_hi,
        );

        builder.eval_memory_access(
            local.frame.shard,
            crate::frame::clk_from_frame::<AB>(&local.frame)
                + AB::F::from_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_u32(33),
            &maddsub_cols.op_hi_access,
            is_real.clone(),
        );
    }

    pub(crate) fn eval_ins<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MiscInstrColumns<AB::Var>,
    ) {
        let ins_cols = local.misc_specific_columns.ins();

        // Ins is decomposed into 6 sub-operations, each proven IN-ROW by a
        // dedicated gadget (the Instruction-bus request rows are gone):
        //    ror_val  = rotate_right(prev_a, lsb)            [shift: lsb ∈ 0..31]
        //    srl1_val = ror_val >> 1                          [shift: 1]
        //    srl_val  = srl1_val >> (msb - lsb)               [shift: msb-lsb ∈ 0..31]
        //    sll_val  = op_b << (31 - msb + lsb)              [shift: ∈ 0..31]
        //    add_val  = srl_val + sll_val
        //    result   = rotate_right(add_val, 31 - msb)       [shift: ∈ 0..31]
        //
        // The original single SRL by `width = msb - lsb + 1` is split into two
        // steps (`>> 1` then `>> (msb - lsb)`) so that each shift amount is
        // always in [0, 31], staying inside the shift logic's range when
        // width = 32.
        {
            use crate::operations::{AddOperation, ShiftLeftOperation, ShiftRightOperation};
            let zero = || AB::Expr::zero();
            let shift_word = |amount: AB::Expr| Word([amount, zero(), zero(), zero()]);

            ShiftRightOperation::<AB::F>::eval(
                builder,
                local.prev_a_value.map(|x| x.into()),
                shift_word(ins_cols.lsb.into()),
                &local.ins_ror,
                zero(),
                zero(),
                local.is_ins.into(),
            );

            // SRL step 1: shift right by 1 (always in range).
            ShiftRightOperation::<AB::F>::eval(
                builder,
                local.ins_ror.value().map(|x| x.into()),
                shift_word(AB::Expr::one()),
                &local.ins_srl1,
                local.is_ins.into(),
                zero(),
                zero(),
            );

            // SRL step 2: shift right by msb - lsb (range [0, 31]).
            ShiftRightOperation::<AB::F>::eval(
                builder,
                local.ins_srl1.value().map(|x| x.into()),
                shift_word(ins_cols.msb - ins_cols.lsb),
                &local.ins_srl,
                local.is_ins.into(),
                zero(),
                zero(),
            );

            ShiftLeftOperation::<AB::F>::eval(
                builder,
                ins_cols.sll_val.map(|x| x.into()),
                local.frame.op_b_val().map(|x| x.into()),
                shift_word(AB::Expr::from_u32(31) - ins_cols.msb + ins_cols.lsb),
                &local.ins_sll,
                local.is_ins.into(),
            );

            AddOperation::<AB::F>::eval(
                builder,
                local.ins_srl.value(),
                ins_cols.sll_val,
                local.ins_add,
                local.is_ins.into(),
            );

            ShiftRightOperation::<AB::F>::eval(
                builder,
                local.ins_add.value.map(|x| x.into()),
                shift_word(AB::Expr::from_u32(31) - ins_cols.msb),
                &local.ins_ror2,
                zero(),
                zero(),
                local.is_ins.into(),
            );
            builder.when(local.is_ins).assert_word_eq(local.op_a_value, local.ins_ror2.value());
        }
        // op_c = (msb << 5) + lsb
        builder.when(local.is_ins).assert_eq(
            local.frame.op_c_val().reduce::<AB>(),
            ins_cols.lsb + ins_cols.msb * AB::Expr::from_u32(32),
        );

        // 32 > msb >= lsb >=0.
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            ins_cols.lsb,
            ins_cols.msb,
            local.is_ins,
        );

        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            ins_cols.lsb,
            ins_cols.msb + AB::Expr::one(),
            local.is_ins,
        );

        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            ins_cols.msb,
            AB::Expr::from_u32(32),
            local.is_ins,
        );
    }

    pub(crate) fn eval_ext<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MiscInstrColumns<AB::Var>,
    ) {
        let ext_cols = local.misc_specific_columns.ext();

        // Ext can be divided into 2 operations, each proven IN-ROW by a
        // dedicated gadget (the Instruction-bus request rows are gone):
        //    sll_val = op_b << (31 - lsb - msbd)
        //    result = sll_val >> (31 - msbd)
        {
            use crate::operations::{ShiftLeftOperation, ShiftRightOperation};
            let zero = || AB::Expr::zero();

            ShiftLeftOperation::<AB::F>::eval(
                builder,
                ext_cols.sll_val.map(|x| x.into()),
                local.frame.op_b_val().map(|x| x.into()),
                Word([
                    AB::Expr::from_u32(31) - ext_cols.lsb - ext_cols.msbd,
                    zero(),
                    zero(),
                    zero(),
                ]),
                &local.ext_sll,
                local.is_ext.into(),
            );

            ShiftRightOperation::<AB::F>::eval(
                builder,
                ext_cols.sll_val.map(|x| x.into()),
                Word([AB::Expr::from_u32(31) - ext_cols.msbd, zero(), zero(), zero()]),
                &local.ext_srl,
                local.is_ext.into(),
                zero(),
                zero(),
            );
            builder.when(local.is_ext).assert_word_eq(local.op_a_value, local.ext_srl.value());
        }

        // op_c = (msbd << 5) + lsb
        builder.when(local.is_ext).assert_eq(
            local.frame.op_c_val().reduce::<AB>(),
            ext_cols.lsb + ext_cols.msbd * AB::Expr::from_u32(32),
        );

        // 0=< lsb/msbd < 32 , lsb + msbd < 32.
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            ext_cols.lsb,
            ext_cols.msbd,
            local.is_ext,
        );

        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            ext_cols.lsb + ext_cols.msbd,
            AB::Expr::from_u32(32),
            local.is_ext,
        );
    }
}
