use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::{izip, Itertools};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{
    air::{BaseAirBuilder, MachineAir, ZKMAirBuilder},
    PicusInfo, Word,
};

use crate::{
    frame::{eval_r_type_frame, RTypeFrameCols},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `LtChip`.
pub const NUM_LT_COLS: usize = size_of::<LtCols<u8>>();

/// A chip that implements bitwise operations for the opcodes SLT and SLTU.
#[derive(Default)]
pub struct LtChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct LtCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// If the opcode is SLT.
    #[picus(selector)]
    pub is_slt: T,

    /// If the opcode is SLTU.
    #[picus(selector)]
    pub is_sltu: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every Lt row is an instruction — DivRem's comparison is
    /// inlined and the Instruction bus is gone).
    pub frame: RTypeFrameCols<T>,

    /// The output operand.

    /// Boolean flag to indicate which byte pair differs if the operands are not equal.
    pub byte_flags: [T; 4],

    /// The masking b[3] & 0x7F.
    pub b_masked: T,
    /// The masking c[3] & 0x7F.
    pub c_masked: T,
    /// An inverse of differing byte if c_comp != b_comp.
    pub not_eq_inv: T,

    /// The most significant bit of operand b.
    pub msb_b: T,
    /// The most significant bit of operand c.
    pub msb_c: T,
    /// The multiplication msb_b * is_slt.
    pub bit_b: T,
    /// The multiplication msb_c * is_slt.
    pub bit_c: T,

    /// The result of the intermediate SLTU operation `b_comp < c_comp`.
    pub sltu: T,
    /// A boolean flag for an intermediate comparison.
    pub is_comp_eq: T,
    /// A boolean flag for comparing the sign bits.
    pub is_sign_eq: T,
    /// The comparison bytes to be looked up.
    pub comparison_bytes: [T; 2],
}

impl<F: PrimeField32> MachineAir<F> for LtChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Lt".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.lt_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <LtChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn picus_info(&self) -> PicusInfo {
        LtCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let padded_nb_rows = <LtChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LT_COLS);
        let chunk_size = std::cmp::max((input.lt_events.len() + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_LT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_LT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LtCols<F> = row.borrow_mut();

                    if idx < input.lt_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.lt_events[idx];
                        self.event_to_row(
                            event,
                            cols,
                            &mut byte_lookup_events,
                            &input.program,
                            input.public_values.execution_shard,
                        );
                    } else {
                        // A padding row's frame needs no neutralising: the
                        // typed R-type frame's register-access multiplicities
                        // are `is_real`.
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.

        Ok(RowMajorMatrix::new(values, NUM_LT_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.lt_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .lt_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_LT_COLS];
                    let cols: &mut LtCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.lt_events.is_empty()
        }
    }
}

impl LtChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut LtCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // Every Lt row is a real instruction owning its frame.
        cols.frame.populate_from_alu(event, program, shard, blu);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();

        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        // If this is SLT, mask the MSB of b & c before computing cols.bits.
        let masked_b = b[3] & 0x7f;
        let masked_c = c[3] & 0x7f;
        cols.b_masked = F::from_u8(masked_b);
        cols.c_masked = F::from_u8(masked_c);

        // Send the masked lookup.
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_b as u16,
            a2: 0,
            b: b[3],
            c: 0x7f,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_c as u16,
            a2: 0,
            b: c[3],
            c: 0x7f,
        });

        let mut b_comp = b;
        let mut c_comp = c;
        if event.opcode == Opcode::SLT {
            b_comp[3] = masked_b;
            c_comp[3] = masked_c;
        }
        cols.sltu = F::from_bool(b_comp < c_comp);
        cols.is_comp_eq = F::from_bool(b_comp == c_comp);

        // Set the byte equality flags.
        for (b_byte, c_byte, flag) in
            izip!(b_comp.iter().rev(), c_comp.iter().rev(), cols.byte_flags.iter_mut().rev())
        {
            if c_byte != b_byte {
                *flag = F::ONE;
                cols.sltu = F::from_bool(b_byte < c_byte);
                let b_byte = F::from_u8(*b_byte);
                let c_byte = F::from_u8(*c_byte);
                cols.not_eq_inv = (b_byte - c_byte).inverse();
                cols.comparison_bytes = [b_byte, c_byte];
                break;
            }
        }

        cols.msb_b = F::from_u8((b[3] >> 7) & 1);
        cols.msb_c = F::from_u8((c[3] >> 7) & 1);
        cols.is_sign_eq = if event.opcode == Opcode::SLT {
            F::from_bool((b[3] >> 7) == (c[3] >> 7))
        } else {
            F::ONE
        };

        cols.is_slt = F::from_bool(event.opcode == Opcode::SLT);
        cols.is_sltu = F::from_bool(event.opcode == Opcode::SLTU);

        cols.bit_b = cols.msb_b * cols.is_slt;
        cols.bit_c = cols.msb_c * cols.is_slt;

        debug_assert_eq!(
            F::from_bool(event.a == 1),
            cols.bit_b * (F::ONE - cols.bit_c) + cols.is_sign_eq * cols.sltu
        );

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: cols.sltu.as_canonical_u32() as u16,
            a2: 0,
            b: cols.comparison_bytes[0].as_canonical_u32() as u8,
            c: cols.comparison_bytes[1].as_canonical_u32() as u8,
        });
    }
}

impl<F> BaseAir<F> for LtChip {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }
}

impl<AB> Air<AB> for LtChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &LtCols<AB::Var> = (*local).borrow();

        let is_real = local.is_slt + local.is_sltu;
        // The inputs live in the frame's register reads, not in columns here.
        let op_b = local.frame.op_b_val();
        let op_c = local.frame.op_c_val();

        // We can compute the signed set-less-than as follows:
        // SLT (signed) = b_s * (1 - c_s) + (b_s == c_s) * SLTU(b_<s, c_<s)
        // Source: Jolt 5.3: Set Less Than (https://people.cs.georgetown.edu/jthaler/Jolt-paper.pdf)

        // We will compute SLTU(b_comp, c_comp) where `b_comp` and `c_comp` where:
        // * if the operation is `SLTU`, `b_comp = b` and `c_comp = c`
        // * if the operation is `SLT`, `b_comp = b & 0x7FFFFFFF` and `c_comp = c & 0x7FFFFFFF``
        //
        // We will set booleans `b_bit` and `c_bit` so that:
        // * If the operation is `SLTU`, then `b_bit = 0` and `c_bit = 0`.
        // * If the operation is `SLT`, then `b_bit`, `c_bit` are the most significant bits of `b`
        //   and `c` respectively.
        //
        // Then, we will compute the answer as:
        // SLT = b_bit * (1 - c_bit) + (b_bit == c_bit) * SLTU(b_comp, c_comp)

        // First, we set up the values of `b_comp` and `c_comp`.
        let mut b_comp: Word<AB::Expr> = op_b.map(|x| x.into());
        let mut c_comp: Word<AB::Expr> = op_c.map(|x| x.into());

        b_comp[3] = op_b[3] * local.is_sltu + local.b_masked * local.is_slt;
        c_comp[3] = op_c[3] * local.is_sltu + local.c_masked * local.is_slt;

        // Constrain the `masked_b` and `masked_c` values via lookup.
        //
        // The values are given by `b_masked = b[3] & 0x7F` and `c_masked = c[3] & 0x7F`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.b_masked,
            op_b[3],
            AB::F::from_u8(0x7f),
            is_real.clone(),
        );
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            local.c_masked,
            op_c[3],
            AB::F::from_u8(0x7f),
            is_real.clone(),
        );

        // Set the values of `b_bit` and `c_bit`.
        builder.assert_eq(local.bit_b, local.msb_b * local.is_slt);
        builder.assert_eq(local.bit_c, local.msb_c * local.is_slt);

        // Assert the correctness of `local.msb_b` and `local.msb_c` using the mask.
        let inv_128 = AB::F::from_u32(128).inverse();
        builder.assert_eq(local.msb_b, (op_b[3] - local.b_masked) * inv_128);
        builder.assert_eq(local.msb_c, (op_c[3] - local.c_masked) * inv_128);

        // Constrain that when is_sign_eq = (bit_b == bit_c).

        // assert the flag is a boolean.
        builder.assert_bool(local.is_sign_eq);

        // assert the correction of the comparison.
        builder.when(local.is_sign_eq).assert_eq(local.bit_b, local.bit_c);
        builder
            .when(is_real.clone())
            .when_not(local.is_sign_eq)
            .assert_one(local.bit_b + local.bit_c);

        // Assert the final result is correct, directly on the frame's
        // committed `op_a` register access — there is no result mirror
        // column.  The frame pins the commit to ZERO when `op_a` is
        // register 0 (the write is discarded), so the low byte binds through
        // a `(1 - op_a_0)` factor; the three high bytes are zero in BOTH
        // cases and bind directly.
        let av = *local.frame.op_a_access.value();
        builder.assert_eq(
            av[0],
            (AB::Expr::ONE - local.frame.op_a_0)
                * (local.bit_b * (AB::Expr::ONE - local.bit_c) + local.is_sign_eq * local.sltu),
        );
        builder.assert_zero(av[1]);
        builder.assert_zero(av[2]);
        builder.assert_zero(av[3]);

        // Verify that the byte equality flags are set correctly, i.e. all are boolean and only
        // at most a single byte flag is set.
        let sum_flags =
            local.byte_flags[0] + local.byte_flags[1] + local.byte_flags[2] + local.byte_flags[3];
        builder.assert_bool(local.byte_flags[0]);
        builder.assert_bool(local.byte_flags[1]);
        builder.assert_bool(local.byte_flags[2]);
        builder.assert_bool(local.byte_flags[3]);
        builder.assert_bool(sum_flags.clone());
        builder.when(is_real.clone()).assert_eq(AB::Expr::ONE - local.is_comp_eq, sum_flags);

        // Constrain `local.sltu == SLTU(b_comp, c_comp)`.
        //
        // We define bytes `b_comp_byte` and `c_comp_byte` as follows: If `b_comp == c_comp`, then
        // `b_comp_byte = c_comp_byte = 0`. Otherwise, we set `b_comp_byte` and `c_comp_byte` to
        // the first differing byte (in most significant order). We will use the `local.is_comp_eq`
        // flag to indicate whether the bytes are equal.

        // Check the equality flag is boolean.
        builder.assert_bool(local.is_comp_eq);

        // Find the differing byte if `b_comp != c_comp` and assert equality in case the flag
        // `local.is_comp_eq` is set to `1`.

        // A flag to indicate whether an equality check is necessary (this is for all bytes from
        // most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::ZERO;

        // Expressions for computing the comparison bytes.
        let mut b_comparison_byte = AB::Expr::ZERO;
        let mut c_comparison_byte = AB::Expr::ZERO;
        // Iterate over the bytes in reverse order and select the differing bytes using the byte
        // flag columns values.
        for (b_byte, c_byte, &flag) in
            izip!(b_comp.0.iter().rev(), c_comp.0.iter().rev(), local.byte_flags.iter().rev())
        {
            // Once the byte flag was set to one, we turn off the quality check flag.
            // We can do this by calculating the sum of the flags since only `1` is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            b_comparison_byte = b_comparison_byte.clone() + b_byte.clone() * flag;
            c_comparison_byte = c_comparison_byte.clone() + c_byte.clone() * flag;

            // If inequality is not visited, assert that the bytes are equal.
            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_byte.clone(), c_byte.clone());
            // If the numbers are assumed equal, inequality should not be visited.
            builder.when(local.is_comp_eq).assert_zero(is_inequality_visited.clone());
        }
        // We need to verify that the comparison bytes are set correctly. This is only relevant in
        // the case where the bytes are not equal.

        // Constrain the row comparison byte values to be equal to the calculated ones.
        let (b_comp_byte, c_comp_byte) = (local.comparison_bytes[0], local.comparison_bytes[1]);
        builder.assert_eq(b_comp_byte, b_comparison_byte);
        builder.assert_eq(c_comp_byte, c_comparison_byte);

        // Using the values above, we can constrain the `local.is_comp_eq` flag. We already asserted
        // in the loop that when `local.is_comp_eq == 1` then all bytes are equal. It is left to
        // verify that when `local.is_comp_eq == 0` the comparison bytes are indeed not equal.
        // This is done using the inverse hint `not_eq_inv`.
        builder
            .when_not(local.is_comp_eq)
            .assert_eq(local.not_eq_inv * (b_comp_byte - c_comp_byte), is_real.clone());

        // Now the value of `local.sltu` is equal to the same value for the comparison bytes.
        //
        // Set `local.sltu = SLTU(b_comp_byte, c_comp_byte)` via a lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            local.sltu,
            b_comp_byte,
            c_comp_byte,
            is_real.clone(),
        );

        // Constrain the operation flags.

        // Check that the operation flags are boolean.
        builder.assert_bool(local.is_slt);
        builder.assert_bool(local.is_sltu);
        // Check that at most one of the operation flags is set.
        //
        // *remark*: this is not strictly necessary since it's also covered by the bus multiplicity
        // but this is included here to make sure the condition is met.
        builder.assert_bool(local.is_slt + local.is_sltu);

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining (the Instruction bus and
        // its dependency rows are gone).  SLT/SLTU are sequential and can
        // never halt.
        eval_r_type_frame(
            builder,
            &local.frame,
            local.is_slt * Opcode::SLT.as_field::<AB::F>()
                + local.is_sltu * Opcode::SLTU.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            is_real.clone(),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use super::LtChip;
    use crate::programs::tests::{alu_op, run_instructions};

    #[test]
    fn generate_trace() {
        let shard = run_instructions(alu_op(Opcode::SLT, 3, 2));
        assert!(!shard.lt_events.is_empty());
        let chip = LtChip::default();
        let generate_trace = chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let trace: RowMajorMatrix<KoalaBear> = generate_trace;
        println!("{:?}", trace.values)
    }

    fn prove_koalabear_template(shard: &ExecutionRecord) {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        let chip = LtChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(shard, &mut ExecutionRecord::default()).unwrap();
        let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }

    #[test]
    fn prove_koalabear_slt() {
        const NEG_3: u32 = 0b11111111111111111111111111111101;
        const NEG_4: u32 = 0b11111111111111111111111111111100;
        let mut instructions = Vec::new();
        for (b, c) in [
            (3, 2),
            (2, 3),
            (5, NEG_3),
            (NEG_3, 5),
            (NEG_3, NEG_4),
            (NEG_4, NEG_3),
            (3, 3),
            (NEG_3, NEG_3),
        ] {
            instructions.extend(alu_op(Opcode::SLT, b, c));
        }
        let shard = run_instructions(instructions);
        assert!(!shard.lt_events.is_empty());

        prove_koalabear_template(&shard);
    }

    #[test]
    fn prove_koalabear_sltu() {
        const LARGE: u32 = 0b11111111111111111111111111111101;
        let mut instructions = Vec::new();
        for (b, c) in [(3, 2), (2, 3), (LARGE, 5), (5, LARGE), (0, 0), (LARGE, LARGE)] {
            instructions.extend(alu_op(Opcode::SLTU, b, c));
        }
        let shard = run_instructions(instructions);
        assert!(!shard.lt_events.is_empty());

        prove_koalabear_template(&shard);
    }
}
