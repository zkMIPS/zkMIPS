use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use zkm_core_executor::{
    events::{AluEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::air::{MachineAir, PicusInfo, ZKMAirBuilder};

use crate::{
    frame::{eval_i_type_frame, ITypeFrameCols},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AddSubImmChip`.
pub const NUM_ADD_SUB_IMM_COLS: usize = size_of::<AddSubImmCols<u8>>();

/// A chip that implements addition for the immediate-form opcodes ADDI and
/// ADDIU: `op_b` is a register, `op_c` is an immediate.  The register form
/// proves in [`super::AddSubChip`]; splitting by operand form lets each half
/// carry the narrower frame its shape implies instead of the universal one.
///
/// SUB keeps the same re-arranged verification as the register chip: given
/// `a` = `b` OP `c`, ADD verifies `a` = `b` + `c` and SUB verifies
/// `b` = `a` + `c`.  (MIPS has no SUBI — subtraction of an immediate arrives
/// as ADDI of its negation — but the executor routes by opcode and form, so
/// the chip keeps both selectors rather than baking that assumption in.)
#[derive(Default)]
pub struct AddSubImmChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct AddSubImmCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// `is_add * (1 - op_a_0)`; a discarded register-0 result ungates the
    /// equation (the frame pins the commit to zero).
    pub add_gate: T,
    /// `is_sub * (1 - op_a_0)`.
    pub sub_gate: T,

    /// Flag indicating whether the opcode is `ADD`.
    #[picus(selector)]
    pub is_add: T,

    /// Flag indicating whether the opcode is `SUB`.
    #[picus(selector)]
    pub is_sub: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row.  Immediate-form only, so the I-type frame carries the
    /// immediate itself in `op_c` and needs no register access for it.
    pub frame: ITypeFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for AddSubImmChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "AddSubImm".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.add_sub_imm_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <AddSubImmChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn picus_info(&self) -> PicusInfo {
        AddSubImmCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.add_sub_imm_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AddSubImmChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD_SUB_IMM_COLS);

        values.chunks_mut(chunk_size * NUM_ADD_SUB_IMM_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD_SUB_IMM_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddSubImmCols<F> = row.borrow_mut();

                    if idx < input.add_sub_imm_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.add_sub_imm_events[idx];
                        self.event_to_row(
                            event,
                            cols,
                            &mut byte_lookup_events,
                            &input.program,
                            input.public_values.execution_shard,
                        );
                    }
                    // A PADDING row needs no neutralising: the I-type frame's
                    // register-access multiplicities are `is_real`, which an
                    // all-zero row leaves at zero.
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_ADD_SUB_IMM_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.add_sub_imm_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .add_sub_imm_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ADD_SUB_IMM_COLS];
                    let cols: &mut AddSubImmCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.add_sub_imm_events.is_empty()
        }
    }
}

impl AddSubImmChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AddSubImmCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        // Every AddSubImm row is a real instruction owning its frame — program
        // fetch, register access, `(clk, pc)` chaining.
        cols.frame.populate_from_alu(event, program, shard, blu);

        cols.is_add = F::from_bool(event.opcode == Opcode::ADD);
        cols.is_sub = F::from_bool(event.opcode == Opcode::SUB);

        let not_a0 = F::ONE - cols.frame.op_a_0;
        cols.add_gate = cols.is_add * not_a0;
        cols.sub_gate = cols.is_sub * not_a0;
        let _ = blu;
    }
}

impl<F> BaseAir<F> for AddSubImmChip {
    fn width(&self) -> usize {
        NUM_ADD_SUB_IMM_COLS
    }
}

impl<AB> Air<AB> for AddSubImmChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &AddSubImmCols<AB::Var> = (*local).borrow();

        let is_real = local.is_add + local.is_sub;
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_sub);
        builder.assert_bool(is_real.clone());

        // The addition runs DIRECTLY on the frame's words — the second
        // operand is the frame's IMMEDIATE rather than a register read; see
        // the register-form chip for the byte-shape argument.
        builder.assert_eq(local.add_gate, local.is_add * (AB::Expr::ONE - local.frame.op_a_0));
        builder.assert_eq(local.sub_gate, local.is_sub * (AB::Expr::ONE - local.frame.op_a_0));
        let av = *local.frame.op_a_access.value();
        let bv = local.frame.op_b_val();
        let cv = local.frame.op_c_val();
        // The carries are RECOVERED linear expressions, boolean-asserted under
        // the case gate (no carry columns): `256*c_out = x_i + y_i - z_i + c_in`
        // with all words byte-shaped has a unique boolean solution.
        let base_inv = AB::F::from_u32(256).inverse();
        // ADD: `a = b + c`.
        let mut carry = AB::Expr::ZERO;
        for i in 0..4 {
            carry = (bv[i] + cv[i] - av[i] + carry) * base_inv;
            builder
                .when(local.add_gate)
                .assert_zero(carry.clone() * (carry.clone() - AB::Expr::ONE));
        }
        // SUB: `a = b - c`, verified as `b = a + c`.
        let mut carry = AB::Expr::ZERO;
        for i in 0..4 {
            carry = (av[i] + cv[i] - bv[i] + carry) * base_inv;
            builder
                .when(local.sub_gate)
                .assert_zero(carry.clone() * (carry.clone() - AB::Expr::ONE));
        }

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining.  ADD/SUB are sequential,
        // so `next_next_pc` is `next_pc + 4`.
        eval_i_type_frame(
            builder,
            &local.frame,
            local.is_add * Opcode::ADD.as_field::<AB::F>()
                + local.is_sub * Opcode::SUB.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            // ADD/SUB can never halt: the received continuation is `next_pc`.
            local.next_pc.into(),
            AB::Expr::ZERO,
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "sys")]
    use std::borrow::BorrowMut;
    #[cfg(feature = "sys")]
    use std::sync::LazyLock;

    #[cfg(feature = "sys")]
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    #[cfg(feature = "sys")]
    use p3_maybe_rayon::prelude::ParallelIterator;
    use rand::{thread_rng, Rng};
    use zkm_core_executor::{ExecutionRecord, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use super::AddSubImmChip;
    #[cfg(feature = "sys")]
    use super::{AddSubImmCols, NUM_ADD_SUB_IMM_COLS};
    use crate::programs::tests::{alu_op, run_instructions};
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};

    #[test]
    fn generate_trace() {
        let shard = run_instructions(alu_op(Opcode::ADD, 8, 6));
        // The two immediate loads of every `alu_op` triple land here.
        assert!(!shard.add_sub_imm_events.is_empty());
        let chip = AddSubImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn measure_addsubimm_degree() {
        let chip = zkm_pcs::Chip::<KoalaBear, _>::new(AddSubImmChip::default());
        // log_quotient_degree = log2_ceil(max_constraint_degree - 1):
        //   1 => degree 3 ; 2 => degree 4 or 5.
        println!("ADDSUBIMM_LOG_QUOTIENT_DEGREE={}", chip.log_quotient_degree());
    }

    #[test]
    fn prove_koala_bear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        // `p3_uni_stark::prove` needs a power-of-two height and
        // `generate_trace` pads to next_multiple_of_32 only, so make the
        // IMMEDIATE-form event count exactly 1024: 511 alu_op triples (two
        // immediate loads each) plus two bare immediate ADDs.
        let mut instructions = Vec::new();
        for _ in 0..511 {
            let operand_1 = thread_rng().gen_range(0..u32::MAX);
            let operand_2 = thread_rng().gen_range(0..u32::MAX);
            instructions.extend(alu_op(Opcode::SUB, operand_1, operand_2));
        }
        instructions.push(zkm_core_executor::Instruction::new(Opcode::ADD, 28, 29, 5, false, true));
        instructions.push(zkm_core_executor::Instruction::new(Opcode::ADD, 27, 29, 7, false, true));
        let shard = run_instructions(instructions);
        assert_eq!(shard.add_sub_imm_events.len(), 1024);

        let chip = AddSubImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    /// Lazily initialized record for use across multiple tests.
    /// Consists of executor-driven `ADD` and `SUB` instructions.
    #[cfg(feature = "sys")]
    static SHARD: LazyLock<ExecutionRecord> = LazyLock::new(|| {
        let mut instructions = Vec::new();
        instructions.extend(alu_op(Opcode::ADD, 1, 2));
        for _ in 0..255 {
            let operand_1 = thread_rng().gen_range(0..u32::MAX);
            let operand_2 = thread_rng().gen_range(0..u32::MAX);
            instructions.extend(alu_op(Opcode::SUB, operand_1, operand_2));
        }
        run_instructions(instructions)
    });

    #[cfg(feature = "sys")]
    #[test]
    fn test_generate_trace_ffi_eq_rust() {
        let shard = LazyLock::force(&SHARD);

        let chip = AddSubImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(shard, &mut ExecutionRecord::default()).unwrap();
        let trace_ffi = generate_trace_ffi(shard);

        assert_eq!(trace_ffi, trace);
    }

    #[cfg(feature = "sys")]
    fn generate_trace_ffi(input: &ExecutionRecord) -> RowMajorMatrix<KoalaBear> {
        use rayon::slice::ParallelSlice;

        use crate::utils::pad_rows_mult32;

        type F = KoalaBear;

        let chunk_size = std::cmp::max(input.add_sub_imm_events.len() / num_cpus::get(), 1);

        let row_batches = input
            .add_sub_imm_events
            .par_chunks(chunk_size)
            .map(|events| {
                let rows = events
                    .iter()
                    .map(|event| {
                        let mut row = [F::ZERO; NUM_ADD_SUB_IMM_COLS];
                        let cols: &mut AddSubImmCols<F> = row.as_mut_slice().borrow_mut();
                        // Every event is a real instruction, fetched from the
                        // program by pc exactly as the Rust `event_to_row`
                        // does.
                        let instruction: zkm_core_executor::InstructionFfi =
                            input.program.fetch(event.pc).into();
                        unsafe {
                            crate::sys::add_sub_imm_event_to_row_koalabear(
                                event,
                                cols,
                                instruction,
                                input.public_values.execution_shard,
                            );
                        }
                        row
                    })
                    .collect::<Vec<_>>();
                rows
            })
            .collect::<Vec<_>>();

        let mut rows: Vec<[F; NUM_ADD_SUB_IMM_COLS]> = vec![];
        for row_batch in row_batches {
            rows.extend(row_batch);
        }

        pad_rows_mult32(
            &mut rows,
            // Mirror `generate_trace`'s padding: the I-type frame needs no
            // neutralising, so a padding row is simply zero.
            || [F::ZERO; NUM_ADD_SUB_IMM_COLS],
            None,
            "AddSubImm",
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_ADD_SUB_IMM_COLS)
    }
}
