//! Verifies left shift.
//!
//! This module implements left shift (b << c) as a combination of bit and byte shifts.
//!
//! The shift amount c is decomposed into two components:
//!
//! - num_bits_to_shift = c % 8: Represents the fine-grained bit-level shift.
//! - num_bytes_to_shift = c // 8: Represents the coarser byte-level shift.
//!
//! Bit shifting is done by multiplying b by 2^num_bits_to_shift. Byte shifting is done by shifting
//! words. The logic looks as follows:
//!
//! c = take the least significant 5 bits of c
//! num_bytes_to_shift = c // 8
//! num_bits_to_shift = c % 8
//!
//! # "Bit shift"
//! bit_shift_multiplier = pow(2, num_bits_to_shift)
//! bit_shift_result = bit_shift_multiplier * b
//!
//! # "Byte shift"
//! for i in range(WORD_SIZE):
//!     if i < num_bytes_to_shift:
//!         assert(a[i] == 0)
//!     else:
//!         assert(a[i] == bit_shift_result[i - num_bytes_to_shift])
//!
//! Notes:
//!
//! - Ideally, we would calculate b * pow(2, c), but pow(2, c) could overflow in F.
//! - Shifting by a multiple of 8 bits is easy (=num_bytes_to_shift) since we just shift words.

use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice};
use zkm_core_executor::{
    events::{AluEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{air::MachineAir, PicusInfo};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    frame::{eval_shamt_frame, ShamtFrameCols},
    utils::{next_multiple_of_32, pad_rows_mult32},
    CoreChipError,
};

/// The number of main trace columns for `ShiftLeftImm`.
/// Width of a MIPS shift amount.
pub const SHAMT_BITS: usize = 5;

pub const NUM_SHIFT_LEFT_IMM_COLS: usize = size_of::<ShiftLeftImmCols<u8>>();

/// The number of bits in a byte.
pub const BYTE_SIZE: usize = 8;

/// A chip that implements the immediate-form (shamt) SLL; the
/// variable-register form proves in [`super::ShiftLeft`].
#[derive(Default)]
pub struct ShiftLeftImm;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftLeftImmCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The output operand.



    /// The 5 bits of the shift amount `c` (a shamt, `< 32`): bits 0..3 select the bit shift,
    /// bits 3..5 the byte shift.
    pub c_least_sig_byte: [T; SHAMT_BITS],

    /// `2^num_bits_to_shift`, pinned as the product `(1 + c0)(1 + 3 c1)(1 + 15 c2)` — no one-hot
    /// selector array is needed for it.
    pub bit_shift_multiplier: T,

    /// The result of multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result: [T; WORD_SIZE],

    /// The carry propagated when multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result_carry: [T; WORD_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    pub is_real: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row.  Shamt form: the immediate is one scalar column (every SLL row is an instruction — the Instruction bus and
    /// its dependency rows are gone).
    pub frame: ShamtFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for ShiftLeftImm {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShiftLeftImm".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        ShiftLeftImmCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.shift_left_imm_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <ShiftLeftImm as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let mut rows: Vec<[F; NUM_SHIFT_LEFT_IMM_COLS]> = vec![];
        let shift_left_imm_events = input.shift_left_imm_events.clone();
        for event in shift_left_imm_events.iter() {
            let mut row = [F::ZERO; NUM_SHIFT_LEFT_IMM_COLS];
            let cols: &mut ShiftLeftImmCols<F> = row.as_mut_slice().borrow_mut();
            let mut blu = Vec::new();
            self.event_to_row(
                event,
                cols,
                &mut blu,
                &input.program,
                input.public_values.execution_shard,
            );
            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_mult32(
            &mut rows,
            || [F::ZERO; NUM_SHIFT_LEFT_IMM_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <ShiftLeftImm as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        let mut trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHIFT_LEFT_IMM_COLS,
        );

        // Create the template for the padded rows. These are fake rows that don't fail on some
        // sanity checks.
        let padded_row_template = {
            let mut row = [F::ZERO; NUM_SHIFT_LEFT_IMM_COLS];
            let cols: &mut ShiftLeftImmCols<F> = row.as_mut_slice().borrow_mut();
            cols.shift_by_n_bytes[0] = F::ONE;
            cols.bit_shift_multiplier = F::ONE;
            // A padding row's frame needs no neutralising: the typed frame's
            // register-access multiplicities are `is_real`.
            row
        };
        debug_assert!(padded_row_template.len() == NUM_SHIFT_LEFT_IMM_COLS);
        for i in input.shift_left_imm_events.len() * NUM_SHIFT_LEFT_IMM_COLS..trace.values.len() {
            trace.values[i] = padded_row_template[i % NUM_SHIFT_LEFT_IMM_COLS];
        }

        Ok(trace)
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.shift_left_imm_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_left_imm_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SHIFT_LEFT_IMM_COLS];
                    let cols: &mut ShiftLeftImmCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.shift_left_imm_events.is_empty()
        }
    }
}

impl ShiftLeftImm {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftLeftImmCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // Every SLL row is a real instruction owning its frame.
        cols.frame.populate_from_alu(event, program, shard, blu);

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();
        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);
        cols.is_real = F::ONE;
        debug_assert!(event.c < 32, "SLL shamt must be < 32");
        for i in 0..SHAMT_BITS {
            cols.c_least_sig_byte[i] = F::from_u32((event.c >> i) & 1);
        }

        // Variables for bit shifting.
        let num_bits_to_shift = event.c as usize % BYTE_SIZE;

        let bit_shift_multiplier = 1u32 << num_bits_to_shift;
        cols.bit_shift_multiplier = F::from_u32(bit_shift_multiplier);

        let mut carry = 0u32;
        let base = 1u32 << BYTE_SIZE;
        let mut bit_shift_result = [0u8; WORD_SIZE];
        let mut bit_shift_result_carry = [0u8; WORD_SIZE];
        for i in 0..WORD_SIZE {
            let v = b[i] as u32 * bit_shift_multiplier + carry;
            carry = v / base;
            bit_shift_result[i] = (v % base) as u8;
            bit_shift_result_carry[i] = carry as u8;
        }
        cols.bit_shift_result = bit_shift_result.map(F::from_u8);
        cols.bit_shift_result_carry = bit_shift_result_carry.map(F::from_u8);

        // Variables for byte shifting.
        let num_bytes_to_shift = (event.c & 0b11111) as usize / BYTE_SIZE;
        for i in 0..WORD_SIZE {
            cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
        }

        // Range checks.
        {
            blu.add_u8_range_checks(&bit_shift_result);
            blu.add_u8_range_checks(&bit_shift_result_carry);
        }

        // Sanity check.
        for i in num_bytes_to_shift..WORD_SIZE {
            debug_assert_eq!(cols.bit_shift_result[i - num_bytes_to_shift], F::from_u8(a[i]));
        }
    }
}

impl<F> BaseAir<F> for ShiftLeftImm {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_IMM_COLS
    }
}

impl<AB> Air<AB> for ShiftLeftImm
where
    AB: ZKMCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &ShiftLeftImmCols<AB::Var> = (*local).borrow();
        // The inputs are the frame's register reads, not columns of this chip.
        let op_b = local.frame.op_b_val();
        let op_c = local.frame.op_c;

        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();
        let base: AB::Expr = AB::F::from_u32(1 << BYTE_SIZE).into();

        // We first "bit shift" and next we "byte shift". Then we compare the results with a.
        // Finally, we perform some misc checks.

        // Step 1: Perform the fine-grained bit shift (i.e., shifting b by c % 8 bits).

        // Check the sum of c_least_sig_byte[i] * 2^i equals c (a shamt, so 5 bits bind it fully
        // and force c < 32).
        let mut c_byte_sum = zero.clone();
        for i in 0..SHAMT_BITS {
            let val: AB::Expr = AB::F::from_u32(1 << i).into();
            c_byte_sum = c_byte_sum.clone() + val * local.c_least_sig_byte[i];
        }
        builder.assert_eq(c_byte_sum, op_c);

        // Check bit_shift_multiplier = 2^(c mod 8) = (1 + c0)(1 + 3 c1)(1 + 15 c2) (degree 3, the
        // bits being boolean).
        let multiplier = (one.clone() + local.c_least_sig_byte[0])
            * (one.clone() + local.c_least_sig_byte[1] * AB::F::from_u32(3))
            * (one.clone() + local.c_least_sig_byte[2] * AB::F::from_u32(15));
        builder.assert_eq(local.bit_shift_multiplier, multiplier);

        // Check bit_shift_result = b * bit_shift_multiplier by using bit_shift_result_carry to
        // carry-propagate.
        for i in 0..WORD_SIZE {
            let mut v = op_b[i] * local.bit_shift_multiplier
                - local.bit_shift_result_carry[i] * base.clone();
            if i > 0 {
                v = v.clone() + local.bit_shift_result_carry[i - 1].into();
            }
            builder.assert_eq(local.bit_shift_result[i], v);
        }

        // Step 2: Perform the coarser bit shift (i.e., shifting b by c // 8 bits).

        // The two-bit number represented by the 3rd and 4th least significant bits of c is the
        // number of bytes to shift.
        let num_bytes_to_shift =
            local.c_least_sig_byte[3] + local.c_least_sig_byte[4] * AB::F::from_u32(2);

        // Verify that shift_by_n_bytes[i] = 1 if and only if i = num_bytes_to_shift.
        for i in 0..WORD_SIZE {
            builder
                .when(local.shift_by_n_bytes[i])
                .assert_eq(num_bytes_to_shift.clone(), AB::F::from_usize(i));
        }

        // The result binds DIRECTLY to the frame's committed `op_a` register
        // access, taking the byte shifting into account — there is no result
        // mirror column.  The frame pins the commit to ZERO for a register-0
        // destination (the write is discarded), so the value-carrying bytes
        // bind through a `(1 - op_a_0)` factor; the shifted-in zero bytes are
        // zero in BOTH cases and bind directly.
        let av = *local.frame.op_a_access.value();
        let not_a0 = AB::Expr::ONE - local.frame.op_a_0;
        for num_bytes_to_shift in 0..WORD_SIZE {
            let mut shifting = builder.when(local.shift_by_n_bytes[num_bytes_to_shift]);
            for i in 0..WORD_SIZE {
                if i < num_bytes_to_shift {
                    // The first num_bytes_to_shift bytes must be zero.
                    shifting.assert_eq(av[i], zero.clone());
                } else {
                    shifting.assert_eq(
                        av[i],
                        not_a0.clone() * local.bit_shift_result[i - num_bytes_to_shift],
                    );
                }
            }
        }

        // Step 3: Misc checks such as range checks & bool checks.
        for bit in local.c_least_sig_byte.iter() {
            builder.assert_bool(*bit);
        }

        // Range check.
        {
            builder.slice_range_check_u8(&local.bit_shift_result, local.is_real);
            builder.slice_range_check_u8(&local.bit_shift_result_carry, local.is_real);
        }

        for shift in local.shift_by_n_bytes.iter() {
            builder.assert_bool(*shift);
        }

        builder.assert_eq(
            local.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
            one.clone(),
        );

        builder.assert_bool(local.is_real);

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining (the Instruction bus and
        // its dependency rows are gone).  SLL is sequential and can never
        // halt.
        eval_shamt_frame(
            builder,
            &local.frame,
            local.is_real * Opcode::SLL.as_field::<AB::F>(),
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

    use super::ShiftLeftImm;
    use crate::programs::tests::run_instructions;
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};

    /// A register seed plus immediate-form (shamt) shifts on it.
    fn sll_imm_instructions(count: usize) -> Vec<Instruction> {
        let mut instructions =
            vec![Instruction::new(Opcode::ADD, 29, 0, 0x12345678, false, true)];
        for i in 0..count {
            instructions.push(Instruction::new(
                Opcode::SLL,
                31,
                29,
                (i % 32) as u32,
                false,
                true,
            ));
        }
        instructions
    }

    #[test]
    fn generate_trace() {
        let shard = run_instructions(sll_imm_instructions(37));
        assert!(!shard.shift_left_imm_events.is_empty());
        assert!(shard.shift_left_events.is_empty());
        let chip = ShiftLeftImm::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn prove_koala_bear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        // `p3_uni_stark::prove` needs a power-of-two height and
        // `generate_trace` pads to next_multiple_of_32 only.
        let shard = run_instructions(sll_imm_instructions(1024));
        assert_eq!(shard.shift_left_imm_events.len(), 1024);

        let chip = ShiftLeftImm::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
