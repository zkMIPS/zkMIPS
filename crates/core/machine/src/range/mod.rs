use core::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{ByteOpcode, ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_pcs::air::{MachineAir, ZKMAirBuilder};

use crate::{utils::zeroed_f_vec, CoreChipError};

/// 1 + sum over bits of 2^bits slots, laid out at row index `2^bits + a`.
/// Sized for `bits <= MAX_RANGE_BITS` — the only widths the machine emits
/// (the 10-bit clk/diff high limbs, `TIMESTAMP_HIGH_LIMB_BITS`); 16-bit checks
/// already have `U16Range`.  2^11 rows instead of SP1's 2^17: the table rides in
/// EVERY shard, so its height is a per-shard fixed cost worth keeping tiny.
pub const MAX_RANGE_BITS: usize = 10;
pub const NUM_RANGE_ROWS: usize = 1 << (MAX_RANGE_BITS + 1);

pub const NUM_RANGE_PREPROCESSED_COLS: usize = size_of::<RangePreprocessedCols<u8>>();
pub const NUM_RANGE_MULT_COLS: usize = size_of::<RangeMultCols<u8>>();

/// The parametric bit-width range table: `(a, bits)` for every `a < 2^bits`,
/// `bits <= 16`, at row index `2^bits + a` (row 0 is `(0, 0)`).  Serves the
/// `ByteOpcode::Range` lookups the byte table cannot (its grid is keyed on a
/// `(b, c)` byte pair, not on a value/width pair).
#[derive(Debug, Clone, Copy, Default)]
pub struct RangeChip<F>(PhantomData<F>);

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct RangePreprocessedCols<T> {
    /// The value to range check.
    pub a: T,
    /// The number of bits.
    pub bits: T,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct RangeMultCols<T> {
    /// How many times this `(a, bits)` pair was checked.
    pub multiplicity: T,
}

impl<F: Field> RangeChip<F> {
    fn preprocessed_trace() -> RowMajorMatrix<F> {
        let mut values = zeroed_f_vec::<F>(NUM_RANGE_PREPROCESSED_COLS * NUM_RANGE_ROWS);
        // Row 0 is (0, 0); rows [2^bits, 2^{bits+1}) hold (a, bits).
        for bits in 0..=MAX_RANGE_BITS {
            for a in 0..(1usize << bits) {
                let row = (1usize << bits) + a;
                let cols: &mut RangePreprocessedCols<F> = values
                    [row * NUM_RANGE_PREPROCESSED_COLS..(row + 1) * NUM_RANGE_PREPROCESSED_COLS]
                    .borrow_mut();
                cols.a = F::from_usize(a);
                cols.bits = F::from_usize(bits);
            }
        }
        RowMajorMatrix::new(values, NUM_RANGE_PREPROCESSED_COLS)
    }
}

impl<F: PrimeField32> MachineAir<F> for RangeChip<F> {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "Range".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_RANGE_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        Some(Self::preprocessed_trace())
    }

    fn generate_dependencies(
        &self,
        _input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut trace = RowMajorMatrix::new(
            zeroed_f_vec(NUM_RANGE_MULT_COLS * NUM_RANGE_ROWS),
            NUM_RANGE_MULT_COLS,
        );
        for (lookup, mult) in input.byte_lookups.iter() {
            if lookup.opcode != ByteOpcode::Range {
                continue;
            }
            let row = (1usize << lookup.b) + lookup.a1 as usize;
            let cols: &mut RangeMultCols<F> = trace.row_mut(row).borrow_mut();
            cols.multiplicity += F::from_usize(*mult);
        }
        Ok(trace)
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }
}

impl<F: Field> BaseAir<F> for RangeChip<F> {
    fn width(&self) -> usize {
        NUM_RANGE_MULT_COLS
    }
}

impl<AB: ZKMAirBuilder<F: Field>> Air<AB> for RangeChip<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_mult = main.current_slice();
        let local_mult: &RangeMultCols<AB::Var> = (*local_mult).borrow();

        let prep = builder.preprocessed().clone();
        let prep = prep.current_slice();
        let local: &RangePreprocessedCols<AB::Var> = (*prep).borrow();

        let field_op = ByteOpcode::Range.as_field::<AB::F>();
        builder.receive_byte(
            field_op,
            local.a,
            local.bits,
            AB::Expr::ZERO,
            local_mult.multiplicity,
        );
    }
}
