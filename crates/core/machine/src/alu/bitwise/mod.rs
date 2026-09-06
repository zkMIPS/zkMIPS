use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
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
    frame::{eval_r_type_frame, RTypeFrameCols},
    utils::{next_multiple_of_32, pad_rows_mult32},
    CoreChipError,
};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the register-form opcodes
/// XOR, OR, AND and NOR.  The immediate forms (XORI, ORI, ANDI — NOR has
/// none) prove on the narrower I-type frame in [`super::BitwiseImmChip`].
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct BitwiseCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The output operand.
    ///
    /// Witnessed because it is what the chip COMPUTES: on a row whose
    /// destination is register 0 the write is discarded, so it legitimately
    /// differs from the value the register access commits.  The two INPUTS are
    /// not columns -- they are the frame's register reads, read directly.
    /// `(is_xor + is_or + is_and + is_nor) * (1 - op_a_0)` — the byte-lookup
    /// multiplicity.  The result word is the frame's committed `op_a` access
    /// directly; a register-0 destination is discarded (frame-pinned to
    /// zero), so its rows send NO lookups and verify nothing — exactly the
    /// old behaviour, minus the 4-column result mirror.
    pub lookup_gate: T,

    /// If the opcode is NOR.
    #[picus(selector)]
    pub is_nor: T,

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
    /// real row (every Bitwise row is an instruction — the Instruction bus
    /// and its dependency rows are gone).  Register-form only, so the R-type
    /// frame carries bare register indices instead of operand words.
    pub frame: RTypeFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for BitwiseChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Bitwise".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        BitwiseCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.bitwise_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <BitwiseChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows = input
            .bitwise_events
            .par_iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
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
            // A padding row needs no neutralising: the R-type frame's
            // register-access multiplicities are `is_real`, which an all-zero
            // row leaves at zero.
            || [F::ZERO; NUM_BITWISE_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <BitwiseChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_BITWISE_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bitwise_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.bitwise_events.is_empty()
        }
    }
}

impl BitwiseChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut BitwiseCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // Every Bitwise row is a real instruction owning its frame.
        cols.frame.populate_from_alu(event, program, shard, blu);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();

        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        cols.is_nor = F::from_bool(event.opcode == Opcode::NOR);
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

impl<F> BaseAir<F> for BitwiseChip {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}

impl<AB> Air<AB> for BitwiseChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &BitwiseCols<AB::Var> = (*local).borrow();

        // Get the opcode for the operation.
        let opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>()
            + local.is_or * ByteOpcode::OR.as_field::<AB::F>()
            + local.is_and * ByteOpcode::AND.as_field::<AB::F>()
            + local.is_nor * ByteOpcode::NOR.as_field::<AB::F>();

        let is_real = local.is_xor + local.is_or + local.is_and + local.is_nor;
        // The lookup multiplicity: real rows whose result write is NOT a
        // discarded register-0 write (a zero-multiplicity entry contributes
        // nothing regardless of its tuple values, so the pinned-zero result
        // word on those rows is harmless).
        builder
            .assert_eq(local.lookup_gate, is_real.clone() * (AB::Expr::ONE - local.frame.op_a_0));
        let av = *local.frame.op_a_access.value();
        for ((a, b), c) in av.into_iter().zip(local.frame.op_b_val()).zip(local.frame.op_c_val()) {
            builder.send_byte(opcode.clone(), a, b, c, local.lookup_gate);
        }

        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(local.is_nor);
        builder.assert_bool(is_real.clone());

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining (the Instruction bus and
        // its dependency rows are gone).  Bitwise ops are sequential and can
        // never halt.
        eval_r_type_frame(
            builder,
            &local.frame,
            local.is_xor * Opcode::XOR.as_field::<AB::F>()
                + local.is_or * Opcode::OR.as_field::<AB::F>()
                + local.is_and * Opcode::AND.as_field::<AB::F>()
                + local.is_nor * Opcode::NOR.as_field::<AB::F>(),
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
    use zkm_core_executor::{ExecutionRecord, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use crate::utils::{uni_stark_prove, uni_stark_verify};

    use super::BitwiseChip;
    use crate::programs::tests::{alu_op, run_instructions};

    fn bitwise_record() -> ExecutionRecord {
        let mut instructions = Vec::new();
        for opcode in [Opcode::XOR, Opcode::OR, Opcode::AND, Opcode::NOR] {
            instructions.extend(alu_op(opcode, 10, 19));
        }
        run_instructions(instructions)
    }

    #[test]
    fn generate_trace() {
        let shard = bitwise_record();
        assert!(!shard.bitwise_events.is_empty());
        let chip = BitwiseChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    /// Build a 1024-row bitwise trace of `OR`s whose operands share no set
    /// bits, optionally swapping the chip's opcode SELECTOR to XOR first.
    ///
    /// `OR` and `XOR` agree exactly on disjoint operands, so the swap leaves
    /// every VALUE in the trace untouched -- the result and the byte lookup
    /// both stay valid.  And the frame takes its opcode from
    /// `program.fetch(pc)` while the selectors come from the EVENT, so flipping
    /// the event alone IS the malicious assignment: there is no other column to
    /// fix up, and the row still claims the `OR` the program holds at that pc.
    fn or_trace(swap_selector_to_xor: bool) -> RowMajorMatrix<KoalaBear> {
        let (b, c) = (0b1100u32, 0b0011u32);
        assert_eq!(b | c, b ^ c, "the forgery needs OR and XOR to agree here");

        // `p3_uni_stark::prove` needs a power-of-two height.
        let mut instructions = Vec::new();
        for _ in 0..1024 {
            instructions.extend(alu_op(Opcode::OR, b, c));
        }
        let mut shard = run_instructions(instructions);
        assert_eq!(shard.bitwise_events.len(), 1024);

        if swap_selector_to_xor {
            for event in &mut shard.bitwise_events {
                assert_eq!(event.opcode, Opcode::OR);
                event.opcode = Opcode::XOR;
            }
        }

        BitwiseChip::default().generate_trace(&shard, &mut ExecutionRecord::default()).unwrap()
    }

    /// The control.  Without it the forgery test below could pass for the wrong
    /// reason -- any harness mistake also fails.
    #[test]
    fn honest_or_trace_proves() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();
        let chip = BitwiseChip::default();
        let proof = uni_stark_prove::<KoalaBearPoseidon2, _>(
            &config,
            &chip,
            &mut challenger,
            or_trace(false),
        );
        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    /// A chip's opcode SELECTORS must be the instruction the program holds at
    /// `pc`.  Nothing enforced that until the frame began binding them:
    /// `frame.instruction.opcode` is pinned to the program by the `Program`
    /// lookup, but the selectors -- which are what actually drive the result --
    /// floated free of it, so a row could compute XOR while claiming OR.
    #[test]
    #[should_panic]
    fn opcode_substitution_is_rejected() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();
        let chip = BitwiseChip::default();
        let proof = uni_stark_prove::<KoalaBearPoseidon2, _>(
            &config,
            &chip,
            &mut challenger,
            or_trace(true),
        );
        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn prove_koalabear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        // `p3_uni_stark::prove` requires a power-of-two height;
        // `generate_trace` pads to `next_multiple_of_32` only, so keep the
        // bitwise event count a power of two: 4 ops x 256 reps = 1024 rows.
        let mut instructions = Vec::new();
        for _ in 0..256 {
            for opcode in [Opcode::XOR, Opcode::OR, Opcode::AND, Opcode::NOR] {
                instructions.extend(alu_op(opcode, 10, 19));
            }
        }
        let shard = run_instructions(instructions);
        assert_eq!(shard.bitwise_events.len(), 1024);
        let chip = BitwiseChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof =
            uni_stark_prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
