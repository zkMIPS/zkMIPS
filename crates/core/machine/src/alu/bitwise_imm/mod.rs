use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::air::{MachineAir, PicusInfo, ZKMAirBuilder};

use crate::{
    frame::{eval_i_type_frame, ITypeFrameCols},
    utils::{next_multiple_of_32, pad_rows_mult32},
    CoreChipError,
};

/// The number of main trace columns for `BitwiseImmChip`.
pub const NUM_BITWISE_IMM_COLS: usize = size_of::<BitwiseImmCols<u8>>();

/// A chip that implements bitwise operations for the immediate-form opcodes
/// XORI, ORI and ANDI: `op_b` is a register, `op_c` is an immediate.  The
/// register forms (and NOR, which has no immediate form) prove in
/// [`super::BitwiseChip`].
#[derive(Default)]
pub struct BitwiseImmChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct BitwiseImmCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The output operand.
    ///
    /// Witnessed because it is what the chip COMPUTES: on a row whose
    /// destination is register 0 the write is discarded, so it legitimately
    /// differs from the value the register access commits.  The two INPUTS are
    /// not columns -- the register read and the immediate come off the frame.
    /// `(is_xor + is_or + is_and) * (1 - op_a_0)` — the byte-lookup
    /// multiplicity; see the register-form chip.
    pub lookup_gate: T,

    /// If the opcode is XOR.
    #[picus(selector)]
    pub is_xor: T,

    // If the opcode is OR.
    #[picus(selector)]
    pub is_or: T,

    /// If the opcode is AND.
    #[picus(selector)]
    pub is_and: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row.  Immediate-form only, so the I-type frame carries the
    /// immediate itself in `op_c` and needs no register access for it.
    pub frame: ITypeFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for BitwiseImmChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "BitwiseImm".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        BitwiseImmCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.bitwise_imm_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <BitwiseImmChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows = input
            .bitwise_imm_events
            .par_iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_BITWISE_IMM_COLS];
                let cols: &mut BitwiseImmCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(
                    event,
                    cols,
                    &mut blu,
                    &input.program,
                    input.public_values.execution_shard,
                );
                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two.
        pad_rows_mult32(
            &mut rows,
            // A padding row needs no neutralising: the I-type frame's
            // register-access multiplicities are `is_real`, which an all-zero
            // row leaves at zero.
            || [F::ZERO; NUM_BITWISE_IMM_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <BitwiseImmChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_BITWISE_IMM_COLS,
        ))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.bitwise_imm_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bitwise_imm_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_BITWISE_IMM_COLS];
                    let cols: &mut BitwiseImmCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(
                        event,
                        cols,
                        &mut blu,
                        &input.program,
                        input.public_values.execution_shard,
                    );
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.bitwise_imm_events.is_empty()
        }
    }
}

impl BitwiseImmChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut BitwiseImmCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // NOR has no immediate form, so it can never reach this chip.
        debug_assert_ne!(event.opcode, Opcode::NOR, "NOR has no immediate form");

        // Every BitwiseImm row is a real instruction owning its frame.
        cols.frame.populate_from_alu(event, program, shard, blu);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();

        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);

        // The lookups run against the committed result; a discarded
        // register-0 write sends none (mirrors the gated multiplicity).
        if cols.frame.op_a_0 == F::ZERO {
            cols.lookup_gate = F::ONE;
            for ((b_a, b_b), b_c) in a.into_iter().zip(b).zip(c) {
                let byte_event = ByteLookupEvent {
                    opcode: ByteOpcode::from(event.opcode),
                    a1: b_a as u16,
                    a2: 0,
                    b: b_b,
                    c: b_c,
                };
                blu.add_byte_lookup_event(byte_event);
            }
        }
    }
}

impl<F> BaseAir<F> for BitwiseImmChip {
    fn width(&self) -> usize {
        NUM_BITWISE_IMM_COLS
    }
}

impl<AB> Air<AB> for BitwiseImmChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &BitwiseImmCols<AB::Var> = (*local).borrow();

        // Get the opcode for the operation.
        let opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>()
            + local.is_or * ByteOpcode::OR.as_field::<AB::F>()
            + local.is_and * ByteOpcode::AND.as_field::<AB::F>();

        // Get a multiplicity of `1` only for a true row.
        let is_real_g = local.is_xor + local.is_or + local.is_and;
        builder.assert_eq(local.lookup_gate, is_real_g * (AB::Expr::ONE - local.frame.op_a_0));
        let av = *local.frame.op_a_access.value();
        for ((a, b), c) in av.into_iter().zip(local.frame.op_b_val()).zip(local.frame.op_c_val()) {
            builder.send_byte(opcode.clone(), a, b, c, local.lookup_gate);
        }

        let is_real = local.is_xor + local.is_or + local.is_and;
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(is_real.clone());

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining.  Bitwise ops are
        // sequential and can never halt.
        eval_i_type_frame(
            builder,
            &local.frame,
            local.is_xor * Opcode::XOR.as_field::<AB::F>()
                + local.is_or * Opcode::OR.as_field::<AB::F>()
                + local.is_and * Opcode::AND.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Instruction, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use crate::utils::{uni_stark_prove, uni_stark_verify};

    use super::BitwiseImmChip;
    use crate::programs::tests::run_instructions;

    /// A register seed plus immediate-form bitwise ops on it.
    fn bitwise_imm_instructions(reps: usize) -> Vec<Instruction> {
        let mut instructions = vec![Instruction::new(Opcode::ADD, 29, 0, 0b1100, false, true)];
        for _ in 0..reps {
            instructions.push(Instruction::new(Opcode::XOR, 28, 29, 0b0011, false, true));
            instructions.push(Instruction::new(Opcode::OR, 27, 29, 0b0101, false, true));
            instructions.push(Instruction::new(Opcode::AND, 26, 29, 0b0110, false, true));
        }
        instructions
    }

    #[test]
    fn generate_trace() {
        let shard = run_instructions(bitwise_imm_instructions(4));
        assert!(!shard.bitwise_imm_events.is_empty());
        assert!(shard.bitwise_events.is_empty());
        let chip = BitwiseImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn measure_bitwiseimm_degree() {
        let chip = zkm_pcs::Chip::<KoalaBear, _>::new(BitwiseImmChip::default());
        println!("BITWISEIMM_LOG_QUOTIENT_DEGREE={}", chip.log_quotient_degree());
    }

    #[test]
    fn prove_koalabear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        // `p3_uni_stark::prove` requires a power-of-two height;
        // `generate_trace` pads to `next_multiple_of_32` only, so keep the
        // immediate-form event count a power of two: 3 ops x 341 reps + one
        // extra = 1024 rows.
        let mut instructions = bitwise_imm_instructions(341);
        instructions.push(Instruction::new(Opcode::XOR, 25, 29, 0b1010, false, true));
        let shard = run_instructions(instructions);
        assert_eq!(shard.bitwise_imm_events.len(), 1024);
        let chip = BitwiseImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof =
            uni_stark_prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
