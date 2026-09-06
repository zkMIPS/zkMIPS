//! Logical And Arithmetic Right Shift Verification.
//!
//! Implements verification for a = b >> c, decomposing the shift into bit and byte components:
//!
//! 1. num_bits_to_shift = c % 8: Bit-level shift, achieved by using ShrCarry.
//! 2. num_bytes_to_shift = c // 8: Byte-level shift, shifting entire bytes or words in b.
//!
//! The right shift is verified by reformulating it as (b >> c) = (b >> (num_bytes_to_shift * 8)) >>
//! num_bits_to_shift.
//!
//! The correct leading bits of logical and arithmetic right shifts are verified by sign extending b
//! to 64 bits.
//!
//! c = take the least significant 5 bits of c
//! num_bytes_to_shift = c // 8
//! num_bits_to_shift = c % 8
//!
//! # Sign extend b to 64 bits if SRA.
//! if opcode == SRA:
//!    b = sign_extend_32_bits_to_64_bits(b)
//! else:
//!    b = zero_extend_32_bits_to_64_bits(b)
//!
//!
//! # Byte shift. Leave the num_bytes_to_shift most significant bytes of b 0 for simplicity as it
//! # doesn't affect the correctness of the result.
//! result = [0; LONG_WORD_SIZE]
//! for i in range(LONG_WORD_SIZE - num_bytes_to_shift):
//!     result[i] = b[i + num_bytes_to_shift]
//!
//! # Bit shift.
//! carry_multiplier = 1 << (8 - num_bits_to_shift)
//! last_carry = 0
//! for i in reversed(range(LONG_WORD_SIZE)):
//!     # Shifts a byte to the right and returns both the shifted byte and the bits that carried.
//!     (shifted_byte[i], carry) = shr_carry(result[i], num_bits_to_shift)
//!     result[i] = shifted_byte[i] + last_carry * carry_multiplier
//!     last_carry = carry
//!
//! # The 4 least significant bytes must match a. The 4 most significant bytes of result may be
//! # inaccurate.
//! assert a = result[0..WORD_SIZE]

use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::air::BaseAirBuilder;
use zkm_pcs::{air::MachineAir, PicusInfo, Word};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    alu::sr::utils::{nb_bits_to_shift, nb_bytes_to_shift},
    bytes::utils::shr_carry,
    frame::{eval_shamt_frame, ShamtFrameCols},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `ShiftRightImmChip`.
pub const NUM_SHIFT_RIGHT_IMM_COLS: usize = size_of::<ShiftRightImmCols<u8>>();

/// The number of bytes necessary to represent a 64-bit integer.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise operations for the opcodes SRL and SRA.
#[derive(Default)]
pub struct ShiftRightImmChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftRightImmCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    pub shift_by_n_bits: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    /// The result of "byte-shifting" the input operand `b` by `num_bytes_to_shift`.
    pub byte_shift_result: [T; LONG_WORD_SIZE],

    /// The result of "bit-shifting" the byte-shifted input by `num_bits_to_shift`.
    pub bit_shift_result: [T; LONG_WORD_SIZE],

    /// The carry output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_carry: [T; LONG_WORD_SIZE],

    /// The shift byte output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_shifted_byte: [T; LONG_WORD_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and `shift_by_n_bytes`.
    pub c_least_sig_byte: [T; BYTE_SIZE],

    /// If the opcode is SRL.
    #[picus(selector)]
    pub is_srl: T,

    /// If the opcode is ROR.
    #[picus(selector)]
    pub is_ror: T,

    /// If the opcode is SRA.
    #[picus(selector)]
    pub is_sra: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row.  Shamt form: the immediate is one scalar column (every ShiftRight row is an instruction — the Instruction bus
    /// and its dependency rows are gone).
    pub frame: ShamtFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for ShiftRightImmChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShiftRightImm".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        ShiftRightImmCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.shift_right_imm_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <ShiftRightImmChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let nb_rows = input.shift_right_imm_events.len();
        let padded_nb_rows = <ShiftRightImmChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SHIFT_RIGHT_IMM_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SHIFT_RIGHT_IMM_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_RIGHT_IMM_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ShiftRightImmCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.shift_right_imm_events[idx];
                        self.event_to_row(
                            event,
                            cols,
                            &mut byte_lookup_events,
                            &input.program,
                            input.public_values.execution_shard,
                        );
                    } else {
                        cols.shift_by_n_bits[0] = F::ONE;
                        cols.shift_by_n_bytes[0] = F::ONE;
                        // A padding row's frame needs no neutralising: the
                        // typed I-type frame's register-access multiplicities
                        // are `is_real`.
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_SHIFT_RIGHT_IMM_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.shift_right_imm_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_right_imm_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SHIFT_RIGHT_IMM_COLS];
                    let cols: &mut ShiftRightImmCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.shift_right_imm_events.is_empty()
        }
    }
}

impl ShiftRightImmChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftRightImmCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // Every ShiftRight row is a real instruction owning its frame.
        cols.frame.populate_from_alu(event, program, shard, blu);

        // Initialize cols with basic operands and flags derived from the current event.
        {
            cols.pc = F::from_u32(event.pc);
            cols.next_pc = F::from_u32(event.next_pc);

            cols.b_msb = F::from_u32((event.b >> 31) & 1);

            cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
            cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
            cols.is_ror = F::from_bool(event.opcode == Opcode::ROR);

            cols.is_real = F::ONE;

            for i in 0..BYTE_SIZE {
                cols.c_least_sig_byte[i] = F::from_u32((event.c >> i) & 1);
            }

            // Insert the MSB lookup event.
            let most_significant_byte = event.b.to_le_bytes()[WORD_SIZE - 1];
            blu.add_byte_lookup_events(vec![ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: ((most_significant_byte >> 7) & 1) as u16,
                a2: 0,
                b: most_significant_byte,
                c: 0,
            }]);
        }

        let num_bytes_to_shift = nb_bytes_to_shift(event.c);
        let num_bits_to_shift = nb_bits_to_shift(event.c);

        // Byte shifting.
        let mut byte_shift_result = [0u8; LONG_WORD_SIZE];
        {
            for i in 0..WORD_SIZE {
                cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
            }
            let sign_extended_b = {
                if event.opcode == Opcode::SRA {
                    // Sign extension is necessary only for arithmetic right shift.
                    ((event.b as i32) as i64).to_le_bytes()
                } else if event.opcode == Opcode::ROR {
                    (((event.b as u64) << 32) | (event.b as u64)).to_le_bytes()
                } else {
                    (event.b as u64).to_le_bytes()
                }
            };

            for i in 0..LONG_WORD_SIZE {
                if i + num_bytes_to_shift < LONG_WORD_SIZE {
                    byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift];
                }
            }
            cols.byte_shift_result = byte_shift_result.map(F::from_u8);
        }

        // Bit shifting.
        {
            for i in 0..BYTE_SIZE {
                cols.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
            }
            let carry_multiplier = 1 << (8 - num_bits_to_shift);
            let mut last_carry = 0u32;
            let mut bit_shift_result = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_carry = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_shifted_byte = [0u8; LONG_WORD_SIZE];
            for i in (0..LONG_WORD_SIZE).rev() {
                let (shift, carry) = shr_carry(byte_shift_result[i], num_bits_to_shift as u8);

                let byte_event = ByteLookupEvent {
                    opcode: ByteOpcode::ShrCarry,
                    a1: shift as u16,
                    a2: carry,
                    b: byte_shift_result[i],
                    c: num_bits_to_shift as u8,
                };
                blu.add_byte_lookup_event(byte_event);

                shr_carry_output_carry[i] = carry;
                shr_carry_output_shifted_byte[i] = shift;
                bit_shift_result[i] = ((shift as u32 + last_carry * carry_multiplier) & 0xff) as u8;
                last_carry = carry as u32;
            }
            cols.bit_shift_result = bit_shift_result.map(F::from_u8);
            cols.shr_carry_output_carry = shr_carry_output_carry.map(F::from_u8);
            cols.shr_carry_output_shifted_byte = shr_carry_output_shifted_byte.map(F::from_u8);
            for i in 0..WORD_SIZE {
                debug_assert_eq!(cols.bit_shift_result[i], F::from_u8(event.a.to_le_bytes()[i]));
            }
            // Range checks.
            blu.add_u8_range_checks(&byte_shift_result);
            blu.add_u8_range_checks(&bit_shift_result);
            blu.add_u8_range_checks(&shr_carry_output_carry);
            blu.add_u8_range_checks(&shr_carry_output_shifted_byte);
        }
    }
}

impl<F> BaseAir<F> for ShiftRightImmChip {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_IMM_COLS
    }
}

impl<AB> Air<AB> for ShiftRightImmChip
where
    AB: ZKMCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &ShiftRightImmCols<AB::Var> = (*local).borrow();
        // The inputs are the frame's register reads, not columns of this chip.
        let op_b = local.frame.op_b_val();
        let op_c = local.frame.op_c;
        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();

        // Check that the MSB of most_significant_byte matches local.b_msb using lookup.
        {
            let byte = op_b[WORD_SIZE - 1];
            let opcode = AB::F::from_u32(ByteOpcode::MSB as u32);
            let msb = local.b_msb;
            builder.send_byte(opcode, msb, byte, zero.clone(), local.is_real);
        }

        // Calculate the number of bits and bytes to shift by from c.
        {
            // The sum of c_least_sig_byte[i] * 2^i must match c[0].
            let mut c_byte_sum = AB::Expr::zero();
            for i in 0..BYTE_SIZE {
                let val: AB::Expr = AB::F::from_u32(1 << i).into();
                c_byte_sum = c_byte_sum.clone() + val * local.c_least_sig_byte[i];
            }
            builder.assert_eq(c_byte_sum, op_c);

            // Number of bits to shift.

            // The 3-bit number represented by the 3 least significant bits of c equals the number
            // of bits to shift.
            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift =
                    num_bits_to_shift.clone() + local.c_least_sig_byte[i] * AB::F::from_u32(1 << i);
            }
            for i in 0..BYTE_SIZE {
                builder
                    .when(local.shift_by_n_bits[i])
                    .assert_eq(num_bits_to_shift.clone(), AB::F::from_usize(i));
            }

            // Exactly one of the shift_by_n_bits must be 1.
            builder.assert_eq(
                local.shift_by_n_bits.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );

            // The 2-bit number represented by the 3rd and 4th least significant bits of c is the
            // number of bytes to shift.
            let num_bytes_to_shift =
                local.c_least_sig_byte[3] + local.c_least_sig_byte[4] * AB::F::from_u32(2);

            // If shift_by_n_bytes[i] = 1, then i = num_bytes_to_shift.
            for i in 0..WORD_SIZE {
                builder
                    .when(local.shift_by_n_bytes[i])
                    .assert_eq(num_bytes_to_shift.clone(), AB::F::from_usize(i));
            }

            // Exactly one of the shift_by_n_bytes must be 1.
            builder.assert_eq(
                local.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );
        }

        // Byte shift the sign-extended b.
        {
            // The leading bytes of b should be 0xff if b's MSB is 1 & opcode = SRA, 0 otherwise.
            let mut sign_extended_b: Vec<AB::Expr> = vec![];
            for i in 0..WORD_SIZE {
                sign_extended_b.push(op_b[i].into());
            }
            for i in 0..WORD_SIZE {
                let leading_byte = local.is_sra * local.b_msb * AB::Expr::from_u8(0xff)
                    + local.is_ror * op_b[i].into();
                sign_extended_b.push(leading_byte.clone());
            }

            // Shift the bytes of sign_extended_b by num_bytes_to_shift.
            for num_bytes_to_shift in 0..WORD_SIZE {
                for i in 0..(LONG_WORD_SIZE - num_bytes_to_shift) {
                    builder.when(local.shift_by_n_bytes[num_bytes_to_shift]).assert_eq(
                        local.byte_shift_result[i],
                        sign_extended_b[i + num_bytes_to_shift].clone(),
                    );
                }
            }
        }

        // Bit shift the byte_shift_result using ShrCarry, and compare the result to a.
        {
            // The carry multiplier is 2^(8 - num_bits_to_shift).
            let mut carry_multiplier = AB::Expr::from_u8(0);
            for i in 0..BYTE_SIZE {
                carry_multiplier = carry_multiplier.clone()
                    + AB::Expr::from_u32(1u32 << (8 - i)) * local.shift_by_n_bits[i];
            }

            // The 3-bit number represented by the 3 least significant bits of c equals the number
            // of bits to shift.
            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift =
                    num_bits_to_shift.clone() + local.c_least_sig_byte[i] * AB::F::from_u32(1 << i);
            }

            // Calculate ShrCarry.
            for i in (0..LONG_WORD_SIZE).rev() {
                builder.send_byte_pair(
                    AB::F::from_u32(ByteOpcode::ShrCarry as u32),
                    local.shr_carry_output_shifted_byte[i],
                    local.shr_carry_output_carry[i],
                    local.byte_shift_result[i],
                    num_bits_to_shift.clone(),
                    local.is_real,
                );
            }

            // Use the results of ShrCarry to calculate the bit shift result.
            for i in (0..LONG_WORD_SIZE).rev() {
                let mut v: AB::Expr = local.shr_carry_output_shifted_byte[i].into();
                if i + 1 < LONG_WORD_SIZE {
                    v = v.clone() + local.shr_carry_output_carry[i + 1] * carry_multiplier.clone();
                }
                builder.assert_eq(v, local.bit_shift_result[i]);
            }
        }

        // Check that the flags are indeed boolean.
        {
            let flags = [local.is_srl, local.is_sra, local.is_ror, local.is_real, local.b_msb];
            for flag in flags.iter() {
                builder.assert_bool(*flag);
            }
            for shift_by_n_byte in local.shift_by_n_bytes.iter() {
                builder.assert_bool(*shift_by_n_byte);
            }
            for shift_by_n_bit in local.shift_by_n_bits.iter() {
                builder.assert_bool(*shift_by_n_bit);
            }
            for bit in local.c_least_sig_byte.iter() {
                builder.assert_bool(*bit);
            }
        }

        // Range check bytes.
        {
            let long_words = [
                local.byte_shift_result,
                local.bit_shift_result,
                local.shr_carry_output_carry,
                local.shr_carry_output_shifted_byte,
            ];

            for long_word in long_words.iter() {
                builder.slice_range_check_u8(long_word, local.is_real);
            }
        }

        // Check that is_real is the sum of the operation flags.
        builder.assert_eq(local.is_srl + local.is_sra + local.is_ror, local.is_real);

        // Receive the arguments.
        // Use bit_shift_result[0..4] directly as the output operand `a`, eliminating the
        // redundant `a` column since a[i] == bit_shift_result[i] is always true.
        let a_word = Word([
            local.bit_shift_result[0],
            local.bit_shift_result[1],
            local.bit_shift_result[2],
            local.bit_shift_result[3],
        ]);
        // Bind this chip's operand columns to the frame's register-file view:
        // the chip must compute on exactly the values the register accesses
        // commit (the Instruction bus that used to carry them is gone).
        builder
            .when(local.is_real)
            .when_not(local.frame.op_a_0)
            .assert_word_eq(a_word, *local.frame.op_a_access.value());

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining (the Instruction bus and
        // its dependency rows are gone).  SRL/SRA/ROR are sequential and can
        // never halt.
        eval_shamt_frame(
            builder,
            &local.frame,
            local.is_srl * Opcode::SRL.as_field::<AB::F>()
                + local.is_sra * Opcode::SRA.as_field::<AB::F>()
                + local.is_ror * Opcode::ROR.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            local.is_real.into(),
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Instruction, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use super::ShiftRightImmChip;
    use crate::programs::tests::run_instructions;
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};

    /// A register seed plus immediate-form (shamt) right shifts on it.
    fn sr_imm_instructions(reps: usize) -> Vec<Instruction> {
        let mut instructions = vec![Instruction::new(Opcode::ADD, 29, 0, 0x87654321, false, true)];
        for i in 0..reps {
            let sa = (i % 32) as u32;
            instructions.push(Instruction::new(Opcode::SRL, 31, 29, sa, false, true));
            instructions.push(Instruction::new(Opcode::SRA, 30, 29, sa, false, true));
            instructions.push(Instruction::new(Opcode::ROR, 28, 29, sa, false, true));
        }
        instructions
    }

    #[test]
    fn generate_trace() {
        let shard = run_instructions(sr_imm_instructions(13));
        assert!(!shard.shift_right_imm_events.is_empty());
        assert!(shard.shift_right_events.is_empty());
        let chip = ShiftRightImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn prove_koala_bear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        // `p3_uni_stark::prove` needs a power-of-two height and
        // `generate_trace` pads to next_multiple_of_32 only: 341 triples plus
        // one extra = 1024 immediate-form events.
        let mut instructions = sr_imm_instructions(341);
        instructions.push(Instruction::new(Opcode::SRL, 27, 29, 7, false, true));
        let shard = run_instructions(instructions);
        assert_eq!(shard.shift_right_imm_events.len(), 1024);

        let chip = ShiftRightImmChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
