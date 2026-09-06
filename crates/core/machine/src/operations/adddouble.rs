use zkm_core_executor::events::ByteRecord;
use zkm_pcs::{air::ZKMAirBuilder, Word};

use p3_air::AirBuilder;
use p3_field::{Field, PrimeCharacteristicRing};
use zkm_derive::AlignedBorrow;

use crate::air::WordAirBuilder;

/// A set of columns needed to compute the add of two double words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddDoubleOperation<T> {
    /// The result of `a + b`.  The carries are recovered in the AIR as
    /// linear expressions and asserted boolean (see `AddOperation`).
    pub value: Word<T>,
    pub value_hi: Word<T>,
}

impl<F: Field> AddDoubleOperation<F> {
    #[allow(unused_assignments)]
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u64: u64, b_u64: u64) -> u64 {
        let expected = a_u64.wrapping_add(b_u64);
        self.value = Word::from(expected as u32);
        self.value_hi = Word::from((expected >> 32) as u32);

        // Range check
        {
            record.add_u8_range_checks(&a_u64.to_le_bytes());
            record.add_u8_range_checks(&b_u64.to_le_bytes());
            record.add_u8_range_checks(&expected.to_le_bytes());
        }
        expected
    }

    pub fn eval<AB: ZKMAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        a_hi: Word<AB::Var>,
        b: Word<AB::Var>,
        b_hi: Word<AB::Var>,
        cols: AddDoubleOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let one = AB::Expr::ONE;
        let base = AB::F::from_u32(256);

        let mut builder_is_real = builder.when(is_real.clone());

        // Recover each carry as a LINEAR expression across the 8 limbs of the
        // 64-bit sum and force it boolean (see `AddOperation::eval`).
        let base_inv = AB::F::from_u32(256).inverse();
        let lo = [a[0], a[1], a[2], a[3]];
        let hi = [a_hi[0], a_hi[1], a_hi[2], a_hi[3]];
        let blo = [b[0], b[1], b[2], b[3]];
        let bhi = [b_hi[0], b_hi[1], b_hi[2], b_hi[3]];
        let vlo = [cols.value[0], cols.value[1], cols.value[2], cols.value[3]];
        let vhi = [cols.value_hi[0], cols.value_hi[1], cols.value_hi[2], cols.value_hi[3]];
        let mut carry = AB::Expr::ZERO;
        for i in 0..8 {
            let (x, y, v) =
                if i < 4 { (lo[i], blo[i], vlo[i]) } else { (hi[i - 4], bhi[i - 4], vhi[i - 4]) };
            carry = (x + y - v + carry) * base_inv;
            builder_is_real.assert_bool(carry.clone());
        }
        builder_is_real.assert_bool(is_real.clone());

        // Range check each byte.
        {
            builder.slice_range_check_u8(&a.0, is_real.clone());
            builder.slice_range_check_u8(&a_hi.0, is_real.clone());
            builder.slice_range_check_u8(&b.0, is_real.clone());
            builder.slice_range_check_u8(&b_hi.0, is_real.clone());
            builder.slice_range_check_u8(&cols.value.0, is_real.clone());
            builder.slice_range_check_u8(&cols.value_hi.0, is_real);
        }
    }
}
