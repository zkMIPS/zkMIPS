use zkm_core_executor::events::ByteRecord;
use zkm_pcs::{air::ZKMAirBuilder, Word};

use p3_air::AirBuilder;
use p3_field::{Field, PrimeCharacteristicRing};
use zkm_derive::AlignedBorrow;

use crate::air::WordAirBuilder;

/// A set of columns needed to compute the add of two words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddOperation<T> {
    /// The result of `a + b`.  The ONLY column: the carries are recovered in
    /// the AIR as the linear expressions `(a_i + b_i - value_i + carry_in) / 256`
    /// and asserted boolean (SP1's AddOperation shape).
    pub value: Word<T>,
}

impl<F: Field> AddOperation<F> {
    #[allow(unused_assignments)]
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32) -> u32 {
        let expected = self.populate_carries(a_u32, b_u32);
        // Range check
        {
            record.add_u8_range_checks(&a_u32.to_le_bytes());
            record.add_u8_range_checks(&b_u32.to_le_bytes());
            record.add_u8_range_checks(&expected.to_le_bytes());
        }
        expected
    }

    /// [`Self::populate`] for a site whose OPERANDS are already byte-shaped
    /// (register-file reads, program immediates): only the fresh result word
    /// is range checked, matching [`Self::eval_check_value_only`].
    pub fn populate_check_value_only(
        &mut self,
        record: &mut impl ByteRecord,
        a_u32: u32,
        b_u32: u32,
    ) -> u32 {
        let expected = self.populate_carries(a_u32, b_u32);
        record.add_u8_range_checks(&expected.to_le_bytes());
        expected
    }

    fn populate_carries(&mut self, a_u32: u32, b_u32: u32) -> u32 {
        let expected = a_u32.wrapping_add(b_u32);
        self.value = Word::from(expected);
        expected
    }

    pub fn eval<AB: ZKMAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base_inv = AB::F::from_u32(256).inverse();

        let mut builder_is_real = builder.when(is_real.clone());

        // Recover each carry as a LINEAR expression and force it boolean:
        //   256 * carry_out = a_i + b_i - value_i + carry_in
        // With a, b, value byte-shaped and carry_in boolean, the equation has
        // a unique boolean solution per limb; the final carry-out is the
        // discarded bit-32 overflow of the wrapping add.
        let mut carry = AB::Expr::ZERO;
        for i in 0..zkm_primitives::consts::WORD_SIZE {
            carry = (a[i] + b[i] - cols.value[i] + carry) * base_inv;
            builder_is_real.assert_bool(carry.clone());
        }
        builder_is_real.assert_bool(is_real.clone());

        // Range check each byte.
        {
            builder.slice_range_check_u8(&a.0, is_real.clone());
            builder.slice_range_check_u8(&b.0, is_real.clone());
            builder.slice_range_check_u8(&cols.value.0, is_real);
        }
    }

    /// [`Self::eval`] for a site whose OPERANDS are already byte-shaped and
    /// need no re-check — a register-file read (every write into the file is
    /// range checked, so the multiset argument carries byte shape to every
    /// read) or a program-table immediate (committed in the vk).  Only the
    /// fresh RESULT word is range checked: the carry argument needs all three
    /// words byte-shaped, and the result is the one this row creates.
    ///
    /// Pair with [`Self::populate_check_value_only`] — the byte-event
    /// emission must mirror the sends exactly.
    pub fn eval_check_value_only<AB: ZKMAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base_inv = AB::F::from_u32(256).inverse();

        let mut builder_is_real = builder.when(is_real.clone());

        // Recovered linear carries, exactly as in [`Self::eval`]; only the
        // fresh RESULT word is range checked (operands already byte-shaped).
        let mut carry = AB::Expr::ZERO;
        for i in 0..zkm_primitives::consts::WORD_SIZE {
            carry = (a[i] + b[i] - cols.value[i] + carry) * base_inv;
            builder_is_real.assert_bool(carry.clone());
        }
        builder_is_real.assert_bool(is_real.clone());

        builder.slice_range_check_u8(&cols.value.0, is_real);
    }
}
