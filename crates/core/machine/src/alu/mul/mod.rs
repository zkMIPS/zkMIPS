//! Implementation to check that b * c = product.
//!
//! We first extend the operands to 64 bits. We sign-extend them if the op code is signed. Then we
//! calculate the un-carried product and propagate the carry. Finally, we check that the appropriate
//! bits of the product match the result.
//!
//! b_64 = sign_extend(b) if signed operation else b
//! c_64 = sign_extend(c) if signed operation else c
//!
//! m = []
//! # 64-bit integers have 8 limbs.
//! # Calculate un-carried product.
//! for i in 0..8:
//!     for j in 0..8:
//!         if i + j < 8:
//!             m[i + j] += b_64[i] * c_64[j]
//!
//! # Propagate carry
//! for i in 0..8:
//!     x = m[i]
//!     if i > 0:
//!         x += carry[i - 1]
//!     carry[i] = x / 256
//!     m[i] = x % 256
//!
//! assert_eq(a, m[0..4])
//!
//! if mult or multu:
//!     assert_eq(hi, m[4..8])

mod utils;

use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use zkm_pcs::air::BaseAirBuilder;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, CompAluEvent, MemoryAccessPosition, MemoryRecordEnum},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{air::MachineAir, PicusInfo, Word};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    alu::mul::utils::get_msb,
    frame::{eval_r_type_frame, RTypeFrameCols},
    memory::{MemoryCols, MemoryReadWriteCols},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `MulChip`.
pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

/// The number of digits in the product is at most the sum of the number of digits in the
/// multiplicands.
pub const PRODUCT_SIZE: usize = 8;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// The mask for a byte.
pub const BYTE_MASK: u8 = 0xff;

/// A chip that implements multiplication for the opcode MUL, MULT and MULTU.
#[derive(Default)]
pub struct MulChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulCols<T> {
    /// The current/next pc, used for instruction lookup table.
    #[picus(input)]
    pub pc: T,
    pub next_pc: T,

    /// The upper bits of the output operand.
    pub hi: Word<T>,

    /// The output operand.

    /// Trace.
    pub carry: [T; PRODUCT_SIZE],

    /// An array storing the product of `b * c` after the carry propagation.
    pub product: [T; PRODUCT_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The most significant bit of `c`.
    pub c_msb: T,

    /// The sign extension of `b`.
    pub b_sign_extend: T,

    /// The sign extension of `c`.
    pub c_sign_extend: T,

    /// Flag indicating whether the opcode is `MUL`.
    #[picus(selector)]
    pub is_mul: T,

    /// Flag indicating whether the opcode is `MULT`.
    #[picus(selector)]
    pub is_mult: T,

    /// Flag indicating whether the opcode is `MULTU`.
    #[picus(selector)]
    pub is_multu: T,

    pub is_real: T,

    /// Access to hi register
    pub op_hi_access: MemoryReadWriteCols<T>,

    /// Flag indicating whether the hi_access record is real.
    pub hi_record_is_real: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every Mul row is an instruction — the Instruction bus and
    /// its dependency rows are gone).
    pub frame: RTypeFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for MulChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Mul".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        MulCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.mul_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <MulChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let padded_nb_rows = <MulChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL_COLS);
        let nb_rows = input.mul_events.len();
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.mul_events[idx];
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
        Ok(RowMajorMatrix::new(values, NUM_MUL_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
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

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.mul_events.is_empty()
        }
    }
}

impl MulChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CompAluEvent,
        cols: &mut MulCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
        shard: u32,
    ) {
        // Every Mul row is a real instruction owning its frame.
        cols.frame.populate_from_comp_alu(event, program, shard, blu);

        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        cols.hi_record_is_real = F::from_bool(event.hi_record_is_real);
        if event.hi_record_is_real {
            // For madd[u]/msub[u] instructions, pass in a dummy byte lookup vector.  This madd[u]/msub[u]
            // instruction chip also has a op_hi_access field that will be populated and that will contribute
            // to the byte lookup dependencies.
            cols.op_hi_access.populate(MemoryRecordEnum::Write(event.hi_record), blu);
        }

        let hi_word = event.hi.to_le_bytes();
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        let mut b = b_word.to_vec();
        let mut c = c_word.to_vec();

        // Handle b and c's signs.
        {
            let b_msb = get_msb(b_word);
            cols.b_msb = F::from_u8(b_msb);
            let c_msb = get_msb(c_word);
            cols.c_msb = F::from_u8(c_msb);

            // If b is signed and it is negative, sign extend b.
            if event.opcode == Opcode::MULT && b_msb == 1 {
                cols.b_sign_extend = F::ONE;
                b.resize(PRODUCT_SIZE, BYTE_MASK);
            }

            // If c is signed and it is negative, sign extend c.
            if event.opcode == Opcode::MULT && c_msb == 1 {
                cols.c_sign_extend = F::ONE;
                c.resize(PRODUCT_SIZE, BYTE_MASK);
            }

            // Insert the MSB lookup events.
            {
                let words = [b_word, c_word];
                let mut blu_events: Vec<ByteLookupEvent> = vec![];
                for word in words.iter() {
                    let most_significant_byte = word[WORD_SIZE - 1];
                    blu_events.push(ByteLookupEvent {
                        opcode: ByteOpcode::MSB,
                        a1: get_msb(*word) as u16,
                        a2: 0,
                        b: most_significant_byte,
                        c: 0,
                    });
                }
                blu.add_byte_lookup_events(blu_events);
            }
        }

        let mut product = [0u32; PRODUCT_SIZE];
        for i in 0..b.len() {
            for j in 0..c.len() {
                if i + j < PRODUCT_SIZE {
                    product[i + j] += (b[i] as u32) * (c[j] as u32);
                }
            }
        }

        // Calculate the correct product using the `product` array. We store the
        // correct carry value for verification.
        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < PRODUCT_SIZE {
                product[i + 1] += carry[i];
            }
            cols.carry[i] = F::from_u32(carry[i]);
        }

        cols.product = product.map(F::from_u32);
        cols.hi = Word(hi_word.map(F::from_u8));
        cols.is_real = F::ONE;
        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_mult = F::from_bool(event.opcode == Opcode::MULT);
        cols.is_multu = F::from_bool(event.opcode == Opcode::MULTU);

        // Range check.
        {
            blu.add_u16_range_checks(&carry.map(|x| x as u16));
            blu.add_u8_range_checks(&product.map(|x| x as u8));
        }
    }
}

impl<F> BaseAir<F> for MulChip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<AB> Air<AB> for MulChip
where
    AB: ZKMCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &MulCols<AB::Var> = (*local).borrow();
        // The inputs are the frame's register reads, not columns of this chip.
        let op_b = local.frame.op_b_val();
        let op_c = local.frame.op_c_val();
        let base = AB::F::from_u32(1 << 8);

        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();
        let byte_mask = AB::F::from_u8(BYTE_MASK);

        // Calculate the MSBs.
        let (b_msb, c_msb) = {
            let msb_pairs =
                [(local.b_msb, op_b[WORD_SIZE - 1]), (local.c_msb, op_c[WORD_SIZE - 1])];
            let opcode = AB::F::from_u32(ByteOpcode::MSB as u32);
            for msb_pair in msb_pairs.iter() {
                let msb = msb_pair.0;
                let byte = msb_pair.1;
                builder.send_byte(opcode, msb, byte, zero.clone(), local.is_real);
            }
            (local.b_msb, local.c_msb)
        };

        // Calculate whether to extend b and c's sign.
        let (b_sign_extend, c_sign_extend) = {
            let is_b_i32 = local.is_mult;
            let is_c_i32 = local.is_mult;

            builder.assert_eq(local.b_sign_extend, is_b_i32 * b_msb);
            builder.assert_eq(local.c_sign_extend, is_c_i32 * c_msb);
            (local.b_sign_extend, local.c_sign_extend)
        };

        // Sign extend op_b and op_c whenever appropriate.
        let (b, c) = {
            let mut b: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
            let mut c: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
            for i in 0..PRODUCT_SIZE {
                if i < WORD_SIZE {
                    b[i] = op_b[i].into();
                    c[i] = op_c[i].into();
                } else {
                    b[i] = b_sign_extend * byte_mask;
                    c[i] = c_sign_extend * byte_mask;
                }
            }
            (b, c)
        };

        // Compute the uncarried product b(x) * c(x) = m(x).
        let mut m: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            for j in 0..PRODUCT_SIZE {
                if i + j < PRODUCT_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * c[j].clone();
                }
            }
        }

        // Propagate carry.
        let product = {
            for i in 0..PRODUCT_SIZE {
                if i == 0 {
                    builder.assert_eq(m[i].clone(), local.carry[i] * base + local.product[i]);
                } else {
                    builder.assert_eq(
                        local.product[i] - local.carry[i - 1] + local.carry[i] * base,
                        m[i].clone(),
                    );
                }
            }
            local.product
        };

        // Compare the product's appropriate bytes with that of the result.
        {
            let has_hi = local.is_mult + local.is_multu;
            for i in 0..WORD_SIZE {
                builder.when(has_hi.clone()).assert_eq(product[i + WORD_SIZE], local.hi[i]);
            }
        }

        // Check that the boolean values are indeed boolean values.
        {
            let booleans = [
                local.b_msb,
                local.c_msb,
                local.b_sign_extend,
                local.c_sign_extend,
                local.is_mul,
                local.is_mult,
                local.is_multu,
                local.is_real,
                local.hi_record_is_real,
            ];
            for boolean in booleans.iter() {
                builder.assert_bool(*boolean);
            }
        }

        // If signed extended, the MSB better be 1.
        builder.when(local.b_sign_extend).assert_eq(local.b_msb, one.clone());
        builder.when(local.c_sign_extend).assert_eq(local.c_msb, one.clone());

        // Calculate the opcode.
        let opcode = {
            // Exactly one of the op codes must be on.
            builder.when(local.is_real).assert_one(local.is_mul + local.is_mult + local.is_multu);

            let mul: AB::Expr = AB::F::from_u32(Opcode::MUL as u32).into();
            let mult: AB::Expr = AB::F::from_u32(Opcode::MULT as u32).into();
            let multu: AB::Expr = AB::F::from_u32(Opcode::MULTU as u32).into();
            local.is_mul * mul + local.is_mult * mult + local.is_multu * multu
        };

        // Range check.
        {
            // Ensure that the carry is at most 2^16. This ensures that
            // product_before_carry_propagation - carry * base + last_carry never overflows or
            // underflows enough to "wrap" around to create a second solution.
            builder.slice_range_check_u16(&local.carry, local.is_real);

            builder.slice_range_check_u8(&local.product, local.is_real);
        }

        let _ = opcode;

        // Bind the product's LOW WORD to the frame's register-file view
        // directly — the old `a` column was a pure mirror of `product[0..4]`.
        // A discarded register-0 write is frame-pinned to zero, so the bind
        // gates on `op_a_0` exactly as before.
        builder.when(local.is_real).when_not(local.frame.op_a_0).assert_word_eq(
            Word([local.product[0], local.product[1], local.product[2], local.product[3]]),
            *local.frame.op_a_access.value(),
        );

        // Every real row is an instruction carrying its own program fetch,
        // register access and `(clk, pc)` chaining.  MUL/MULT/MULTU are
        // sequential and never halt.
        eval_r_type_frame(
            builder,
            &local.frame,
            local.is_mul * Opcode::MUL.as_field::<AB::F>()
                + local.is_mult * Opcode::MULT.as_field::<AB::F>()
                + local.is_multu * Opcode::MULTU.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            local.is_real.into(),
        );
        // The HI-register write below rides the frame's shard/clk directly.  Mul
        // used to keep private copies, tied to the frame on hi-writing rows and
        // forced zero elsewhere; the access is gated by `hi_record_is_real`, so
        // the value on a non-writing row was never read in the first place.
        builder.eval_memory_access(
            local.frame.shard,
            crate::frame::clk_from_r_type_frame::<AB>(&local.frame)
                + AB::Expr::from_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_u32(33),
            &local.op_hi_access,
            local.hi_record_is_real,
        );

        // Check hi_record_is_real.
        // hi_record_is_real can only be set for MULT and MULTU instruction when is_real = 1.
        // if hi_record_is_real = 0, both clk and shard should be zero.
        builder.when_not(local.is_real).assert_zero(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_one(local.is_mult + local.is_multu);
        // Every MULT/MULTU row writes HI (there are no dependency-only
        // multiply rows any more).
        builder.when(local.is_mult + local.is_multu).assert_one(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_word_eq(local.hi, *local.op_hi_access.value());
        builder.when(local.is_mul).assert_word_zero(local.hi);
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Opcode};
    use zkm_pcs::{air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use super::MulChip;
    use crate::programs::tests::{alu_op, run_instructions};

    #[test]
    fn generate_trace_mul() {
        let mut instructions = Vec::new();
        for _ in 0..10 {
            instructions.extend(alu_op(Opcode::MUL, 0x80000000, 0xffff8000));
        }
        let shard = run_instructions(instructions);
        assert!(!shard.mul_events.is_empty());
        let chip = MulChip::default();
        let _trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
    }

    #[cfg(feature = "sys")]
    #[test]
    fn test_mul_generate_trace_ffi_eq_rust() {
        // Every Mul row carries an instruction frame, so drive the record
        // through the executor.
        let shard = run_instructions(alu_op(Opcode::MULT, 274417, 3776743705));
        assert!(!shard.mul_events.is_empty());

        let chip = MulChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let trace_ffi = generate_trace_ffi(&shard);

        assert_eq!(trace_ffi, trace);
    }

    #[cfg(feature = "sys")]
    fn generate_trace_ffi(input: &ExecutionRecord) -> RowMajorMatrix<KoalaBear> {
        use super::{MulCols, NUM_MUL_COLS};
        use crate::utils::next_multiple_of_32;
        use crate::utils::zeroed_f_vec;
        use p3_koala_bear::KoalaBear;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
        use std::borrow::BorrowMut;

        type F = KoalaBear;

        let padded_nb_rows = next_multiple_of_32(input.mul_events.len(), None, "Mul");
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL_COLS);
        let nb_rows = input.mul_events.len();
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let event = &input.mul_events[idx];
                        let instruction: zkm_core_executor::InstructionFfi =
                            input.program.fetch(event.pc).into();
                        unsafe {
                            crate::sys::mul_event_to_row_koalabear(
                                event,
                                cols,
                                instruction,
                                input.public_values.execution_shard,
                            );
                        }
                    } else {
                        // Typed R-type frame: padding rows stay zero.
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MUL_COLS)
    }

    #[test]
    fn prove_koalabear() {
        let config = KoalaBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mul_instructions: Vec<(Opcode, u32, u32)> = vec![
            (Opcode::MUL, 0x00007e00, 0xb6db6db7),
            (Opcode::MUL, 0x00007fc0, 0xb6db6db7),
            (Opcode::MUL, 0x00000000, 0x00000000),
            (Opcode::MUL, 0x00000001, 0x00000001),
            (Opcode::MUL, 0x00000003, 0x00000007),
            (Opcode::MUL, 0x00000000, 0xffff8000),
            (Opcode::MUL, 0x80000000, 0x00000000),
            (Opcode::MUL, 0x80000000, 0xffff8000),
            (Opcode::MUL, 0xaaaaaaab, 0x0002fe7d),
            (Opcode::MUL, 0x0002fe7d, 0xaaaaaaab),
            (Opcode::MUL, 0xff000000, 0xff000000),
            (Opcode::MUL, 0xffffffff, 0xffffffff),
            (Opcode::MUL, 0xffffffff, 0x00000001),
            (Opcode::MUL, 0x00000001, 0xffffffff),
            (Opcode::MULT, 0x00000001, 0xffffffff),
            (Opcode::MULTU, 0xffffffff, 0xffffffff),
        ];
        let mut instructions = Vec::new();
        for &(opcode, b, c) in mul_instructions.iter() {
            instructions.extend(alu_op(opcode, b, c));
        }

        // Append more events until we have ~1000 mul rows.
        for _ in 0..(1000 - mul_instructions.len()) {
            instructions.extend(alu_op(Opcode::MUL, 1, 1));
        }

        let shard = run_instructions(instructions);
        let chip = MulChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
