//! CLO and CLZ verification.
//!
//! This module implements the verification logic for clz and clo operations. It ensures
//! that for any given input b and outputs the leading zero/one count.
//!
//! First, we prove the CLZ.
//! if b == 0, then clz(b) = 32
//! if b > 0, then b >> (32 - (result + 1)) == 1 && b >> (32 - result) == 0
//!
//! Second, we prove the CLO.
//! we use clo(b) = clz(0xffffffff - b)

use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::air::BaseAirBuilder;
use zkm_pcs::{air::MachineAir, PicusInfo, Word};

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    frame::{eval_i_type_frame, ITypeFrameCols},
    operations::ShiftRightOperation,
    utils::{next_multiple_of_32, pad_rows_mult32},
    CoreChipError,
};

/// The number of main trace columns for `CloClzChip`.
pub const NUM_CLOCLZ_COLS: usize = size_of::<CloClzCols<u8>>();

/// A chip that implements addition for the opcodes CLO/CLZ.
#[derive(Default)]
pub struct CloClzChip;

/// The column layout for the chip.
///
/// Optimized: `sr1` removed (hardcoded as 1 in SRL lookup since we always verify sr1 == 1),
/// `is_clo` removed (derived as `is_real - is_clz`).
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CloClzCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The result
    pub a: Word<T>,

    /// if clo, bb == 0xffffffff - b
    /// if clz, bb == b
    pub bb: Word<T>,

    /// whether the `bb` is zero.
    pub is_bb_zero: T,

    /// The inlined shift proving `bb >> (31 - a) == 1` when `bb != 0` (the
    /// SRL request row the chip used to push onto ShiftRight).
    pub srl: ShiftRightOperation<T>,

    /// Flag to indicate whether the opcode is CLZ.
    #[picus(selector)]
    pub is_clz: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every CloClz row is an instruction).
    pub frame: ITypeFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for CloClzChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "CloClz".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        CloClzCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.cloclz_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <CloClzChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let mut rows: Vec<[F; NUM_CLOCLZ_COLS]> = vec![];
        let cloclz_events = input.cloclz_events.clone();
        for event in cloclz_events.iter() {
            assert!(event.opcode == Opcode::CLZ || event.opcode == Opcode::CLO);
            let mut row = [F::ZERO; NUM_CLOCLZ_COLS];
            let cols: &mut CloClzCols<F> = row.as_mut_slice().borrow_mut();

            cols.a = Word::from(event.a);
            cols.pc = F::from_u32(event.pc);
            cols.next_pc = F::from_u32(event.next_pc);
            cols.is_real = F::ONE;
            cols.is_clz = F::from_bool(event.opcode == Opcode::CLZ);
            // Every CloClz row is a real instruction (no chip outsources to
            // CloClz, and its SRL sub-operation is inlined).
            cols.frame.populate_from_alu(
                event,
                &input.program,
                input.public_values.execution_shard,
                output,
            );

            let bb = if event.opcode == Opcode::CLZ { event.b } else { 0xffffffff - event.b };
            cols.bb = Word::from(bb);

            // if bb == 0, then result is 32.
            cols.is_bb_zero = F::from_bool(bb == 0);

            // The inlined shift (the SRL request row): bb >> (31 - a) == 1.
            if bb != 0 {
                cols.srl.populate(output, Opcode::SRL, bb, 31 - event.a);
            }

            // Range check.
            output.add_u8_range_checks(&bb.to_le_bytes());
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: event.a as u8,
                c: 33,
            });

            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        // The inlined shift is gated on `is_real - is_bb_zero`, which is zero
        // on all-zero padding rows, so no fake-row template is needed — only
        // the frame must be neutralised (or its register-access
        // multiplicities break the Memory bus).
        pad_rows_mult32(
            &mut rows,
            || {
                let mut row = [F::ZERO; NUM_CLOCLZ_COLS];
                let cols: &mut CloClzCols<F> = row.as_mut_slice().borrow_mut();
                row
            },
            input.fixed_log2_rows::<F, _>(self),
            <CloClzChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        let trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_CLOCLZ_COLS);

        Ok(trace)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.cloclz_events.is_empty()
        }
    }
}

impl<F> BaseAir<F> for CloClzChip {
    fn width(&self) -> usize {
        NUM_CLOCLZ_COLS
    }
}

impl<AB> Air<AB> for CloClzChip
where
    AB: ZKMCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &CloClzCols<AB::Var> = (*local).borrow();
        let one: AB::Expr = AB::F::ONE.into();
        let zero: AB::Expr = AB::F::ZERO.into();

        // Derive is_clo from is_real and is_clz.
        let is_clo: AB::Expr = local.is_real.into() - local.is_clz.into();

        // if clz, bb == b, else bb = !b
        {
            local.frame.op_b_val().0.iter().zip_eq(local.bb.0.iter()).for_each(|(a, b)| {
                builder.when(is_clo.clone()).assert_eq(*a + *b, AB::Expr::from_u32(255));
                builder.when(local.is_clz).assert_eq(*a, *b);
            });

            builder.slice_range_check_u8(&local.bb.0, local.is_real);
        }

        // ensure result < 33
        // Send the comparison lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::F::ONE,
            local.a[0],
            AB::Expr::from_u8(33),
            local.is_real,
        );

        builder.when(local.is_real).assert_zero(local.a[1]);
        builder.when(local.is_real).assert_zero(local.a[2]);
        builder.when(local.is_real).assert_zero(local.a[3]);

        // The Instruction-bus receive is gone: every row is a real
        // instruction serving itself via the frame.

        // A real instruction carries its own program fetch, register access and
        // `(clk, pc)` chaining.  CLZ/CLO are sequential and can never halt.
        // Bind this chip's operand columns to the frame's register-file view:
        // the chip must compute on exactly the values the register accesses
        // commit (the Instruction bus that used to carry them is gone).
        builder
            .when(local.is_real)
            .when_not(local.frame.op_a_0)
            .assert_word_eq(local.a, *local.frame.op_a_access.value());

        eval_i_type_frame(
            builder,
            &local.frame,
            local.is_clz * Opcode::CLZ.as_field::<AB::F>()
                + (local.is_real - local.is_clz) * Opcode::CLO.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            local.is_real.into(),
        );

        // if is_bb_zero == 1, bb == 0, and result is 32
        {
            builder.assert_bool(local.is_bb_zero);

            builder.when(local.is_bb_zero).assert_zero(local.bb.reduce::<AB>());
            builder.when(local.is_bb_zero).assert_zero(local.bb[3]);

            builder.when(local.is_bb_zero).assert_eq(local.a[0], AB::Expr::from_u32(32));
        }

        {
            // Verify bb >> (31 - result) == 1 IN-ROW (the SRL request row is
            // gone).  The shift is gated on `is_real - is_bb_zero`: live
            // exactly on real rows with bb != 0, zero on padding.
            let is_srl = local.is_real - local.is_bb_zero;
            ShiftRightOperation::<AB::F>::eval(
                builder,
                local.bb.map(|x| x.into()),
                Word([
                    AB::Expr::from_u32(31) - local.a[0],
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                ]),
                &local.srl,
                is_srl.clone(),
                zero.clone(),
                zero.clone(),
            );
            let shifted = local.srl.value();
            builder.when(is_srl.clone()).assert_eq(shifted[0], one.clone());
            builder.when(is_srl.clone()).assert_zero(shifted[1]);
            builder.when(is_srl.clone()).assert_zero(shifted[2]);
            builder.when(is_srl).assert_zero(shifted[3]);
        }

        // is_clz and is_real are boolean; is_clo = is_real - is_clz must also be boolean,
        // which is equivalent to: is_clz = 1 implies is_real = 1.
        builder.assert_bool(local.is_clz);
        builder.assert_bool(local.is_real);
        builder.when(local.is_clz).assert_one(local.is_real);
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{uni_stark_prove, uni_stark_verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Executor, Instruction, Opcode, Program};
    use zkm_pcs::{
        air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig, ZKMCoreOpts,
    };

    use super::CloClzChip;

    /// Real CloClz rows carry an instruction frame (program fetch + register
    /// records), so the tests execute a small program instead of hand-writing
    /// events.
    fn cloclz_record() -> ExecutionRecord {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x00800000, false, true),
            Instruction::new(Opcode::CLZ, 30, 29, 0, false, false),
            Instruction::new(Opcode::CLO, 31, 29, 0, false, false),
            Instruction::new(Opcode::ADD, 28, 0, 0, false, true),
            Instruction::new(Opcode::CLZ, 27, 28, 0, false, false),
            Instruction::new(Opcode::CLO, 26, 28, 0, false, false),
            Instruction::new(Opcode::ADD, 25, 0, 0xffffffff, false, true),
            Instruction::new(Opcode::CLZ, 24, 25, 0, false, false),
            Instruction::new(Opcode::CLO, 23, 25, 0, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        runtime.records[0].clone()
    }

    #[test]
    fn generate_trace() {
        let shard = cloclz_record();
        assert!(!shard.cloclz_events.is_empty());
        let chip = CloClzChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn prove_koalabear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        let shard = cloclz_record();
        let chip = CloClzChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof =
            uni_stark_prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
