use p3_air::AirBuilder;
use p3_field::BasedVectorSpace;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use zkm_core_executor::ByteOpcode;
use zkm_derive::AlignedBorrow;
use zkm_pcs::ZKMAirBuilder;
use zkm_pcs::{
    septic_curve::{SepticCurve, CURVE_WITNESS_DUMMY_POINT_X, CURVE_WITNESS_DUMMY_POINT_Y},
    septic_extension::{SepticBlock, SepticExtension, RECEIVE_Y6_MAX, SEND_Y6_MIN},
};

/// Upper bound (exclusive) of the witnessed `y6_value`; equals `RECEIVE_Y6_MAX`
/// (spelled as a literal: cbindgen copies the initialiser into the C++ header).
pub const Y6_RANGE_BOUND: u32 = 1056964608;
/// `Y6_RANGE_BOUND >> 24` — the bound on the top limb checked by `LTU`.
pub const Y6_TOP_BOUND: u8 = 63;
const _: () = assert!(Y6_RANGE_BOUND == RECEIVE_Y6_MAX);
const _: () = assert!((Y6_RANGE_BOUND >> 24) == Y6_TOP_BOUND as u32);
const _: () = assert!(Y6_RANGE_BOUND == 63 << 24);

/// A set of columns needed to compute the global interaction elliptic curve digest.
///
/// The digest sign is fixed by the top limb of `y`: a receive has
/// `y6 = 1 + y6_value`, a send `y6 = SEND_Y6_MIN + y6_value`, with
/// `y6_value < 63 * 2^24` proven through three byte-table lookups on the limbs
/// `y6_value = y6_lo16 + y6_mid8 * 2^16 + y6_top * 2^24` (the last one is
/// `LTU(y6_top, 63)`).  The two `y6` ranges are disjoint and mirror images of
/// each other, so exactly one of `±P` is provable for any point outside the
/// exception band (`lift_x` skips that band).  This replaces a 30-bit boolean
/// decomposition plus an inverse witness, and the 8 offset bits collapse to
/// one byte range checked together with `y6_mid8`: 53 -> 18 columns.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct GlobalLookupOperation<T: Copy> {
    /// The `lift_x` offset (a byte).
    pub offset: T,
    pub x_coordinate: SepticBlock<T>,
    pub y_coordinate: SepticBlock<T>,
    /// Low 16 bits of `y6_value`.
    pub y6_lo16: T,
    /// Bits 16..24 of `y6_value`.
    pub y6_mid8: T,
    /// Bits 24.. of `y6_value`; `< Y6_TOP_BOUND`.
    pub y6_top: T,
}

/// The per-event digest data (point, offset, `y6_value` limbs); the layout lives in
/// the executor crate so the record can carry it.  See `GlobalDigestCell`.
pub type GlobalDigestRow = zkm_core_executor::GlobalDigestRowRaw;

impl<F: PrimeField32> GlobalLookupOperation<F> {
    pub fn get_digest(
        values: SepticBlock<u32>,
        is_receive: bool,
        kind: u8,
    ) -> (SepticCurve<F>, u8) {
        let x_start =
            SepticExtension::<F>::from_basis_coefficients_fn(|i| F::from_u32(values.0[i]))
                + SepticExtension::from(F::from_u32((kind as u32) << 16));
        let (point, offset) = SepticCurve::<F>::lift_x(x_start);
        if !is_receive {
            return (point.neg(), offset);
        }
        (point, offset)
    }

    /// Compute the digest row for one event.
    pub fn digest_row(values: SepticBlock<u32>, is_receive: bool, kind: u8) -> GlobalDigestRow {
        let (point, offset) = Self::get_digest(values, is_receive, kind);
        let y6 = point.y.0[6].as_canonical_u32();
        let y6_value = if is_receive { y6 - 1 } else { y6 - SEND_Y6_MIN };
        debug_assert!(y6_value < Y6_RANGE_BOUND);
        GlobalDigestRow {
            x: point.x.0.map(|v| v.as_canonical_u32()),
            y: point.y.0.map(|v| v.as_canonical_u32()),
            offset,
            y6_top: (y6_value >> 24) as u8,
            y6_mid8: ((y6_value >> 16) & 0xFF) as u8,
            y6_lo16: (y6_value & 0xFFFF) as u16,
        }
    }

    pub fn populate(
        &mut self,
        values: SepticBlock<u32>,
        is_receive: bool,
        is_real: bool,
        kind: u8,
    ) {
        if is_real {
            self.populate_from_row(&Self::digest_row(values, is_receive, kind));
        } else {
            self.populate_dummy();
        }
    }

    /// Populate from a precomputed digest row.
    pub fn populate_from_row(&mut self, row: &GlobalDigestRow) {
        self.offset = F::from_u8(row.offset);
        self.x_coordinate = SepticBlock(row.x.map(F::from_u32));
        self.y_coordinate = SepticBlock(row.y.map(F::from_u32));
        self.y6_lo16 = F::from_u16(row.y6_lo16);
        self.y6_mid8 = F::from_u8(row.y6_mid8);
        self.y6_top = F::from_u8(row.y6_top);
    }

    pub fn populate_dummy(&mut self) {
        self.offset = F::ZERO;
        self.x_coordinate =
            SepticBlock::<F>::from_base_fn(|i| F::from_u32(CURVE_WITNESS_DUMMY_POINT_X[i]));
        self.y_coordinate =
            SepticBlock::<F>::from_base_fn(|i| F::from_u32(CURVE_WITNESS_DUMMY_POINT_Y[i]));
        self.y6_lo16 = F::ZERO;
        self.y6_mid8 = F::ZERO;
        self.y6_top = F::ZERO;
    }
}

impl<F: Field> GlobalLookupOperation<F> {
    /// Constrain that the elliptic curve point for the global interaction is correctly derived.
    pub fn eval_single_digest<AB: ZKMAirBuilder>(
        builder: &mut AB,
        values: [AB::Expr; 7],
        cols: GlobalLookupOperation<AB::Var>,
        is_receive: AB::Expr,
        is_send: AB::Expr,
        is_real: AB::Var,
        kind: AB::Var,
    ) {
        // Constrain that the `is_real` is boolean.
        builder.assert_bool(is_real);

        // Range check the first element in the message to be a u16 so that we can encode the
        // interaction kind in the upper 8 bits.
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            values[0].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            is_real,
        );

        // `y6_value = y6_lo16 + y6_mid8 * 2^16 + y6_top * 2^24 < 63 * 2^24`, and the offset is
        // a byte: one U16Range, one U8Range (pairing `y6_mid8` with `offset`) and one LTU.
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            cols.y6_lo16,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            is_real,
        );
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U8Range as u8),
            AB::Expr::ZERO,
            cols.y6_mid8,
            cols.offset,
            is_real,
        );
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::LTU as u8),
            AB::Expr::ONE,
            cols.y6_top,
            AB::Expr::from_u8(Y6_TOP_BOUND),
            is_real,
        );

        let x = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.x_coordinate[i].into());
        let y = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.y_coordinate[i].into());

        // Constrain that x_coordinate is derived from (values, kind, offset) via the
        // map-to-curve function. This is the critical link between the tuple columns
        // (which participate in the cross-table lookup) and the witness curve point
        // (which is accumulated into the global digest).
        //
        // The map-to-curve computes:
        //   x[0] = values[0] + kind * 65536
        //   x[i] = values[i]              for i in 1..6
        //   x[6] = values[6] * 256 + offset
        builder
            .when(is_real)
            .assert_eq(x.0[0].clone(), values[0].clone() + kind.into() * AB::Expr::from_u32(65536));
        for i in 1..6 {
            builder.when(is_real).assert_eq(x.0[i].clone(), values[i].clone());
        }
        builder
            .when(is_real)
            .assert_eq(x.0[6].clone(), values[6].clone() * AB::Expr::from_u32(256) + cols.offset);

        // Constrain that `(x, y)` is a valid point on the curve.
        let y2 = y.square();
        let x3_3zx_m3 = SepticCurve::<AB::Expr>::curve_formula(x);
        builder.assert_septic_ext_eq(y2, x3_3zx_m3);

        let y6_value = cols.y6_lo16
            + cols.y6_mid8 * AB::F::from_u32(1 << 16)
            + cols.y6_top * AB::F::from_u32(1 << 24);

        // Constrain that y has correct sign.
        // If it's a receive: `1 <= y_6 <= RECEIVE_Y6_MAX`, so `y_6 - 1 = y6_value < RECEIVE_Y6_MAX`.
        // If it's a send: `SEND_Y6_MIN <= y_6 <= p - 1`, so `y_6 - SEND_Y6_MIN = y6_value < RECEIVE_Y6_MAX`
        // (`p - SEND_Y6_MIN == RECEIVE_Y6_MAX`).  The two ranges are disjoint.
        builder.when(is_receive).assert_eq(y.0[6].clone(), AB::Expr::ONE + y6_value.clone());
        builder.when(is_send).assert_eq(y.0[6].clone(), AB::Expr::from_u32(SEND_Y6_MIN) + y6_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_koala_bear::KoalaBear;

    #[test]
    fn digest_row_limbs_reassemble_and_bound() {
        for i in 0..2000u32 {
            let values = SepticBlock([i, i * 7 + 1, 3, i & 0xFFFF, 5, 6, i ^ 0x5A5A]);
            for is_receive in [true, false] {
                let row = GlobalLookupOperation::<KoalaBear>::digest_row(values, is_receive, 3);
                let v =
                    row.y6_lo16 as u32 + ((row.y6_mid8 as u32) << 16) + ((row.y6_top as u32) << 24);
                assert!(v < Y6_RANGE_BOUND);
                assert!(row.y6_top < Y6_TOP_BOUND);
                let y6 = row.y[6];
                if is_receive {
                    assert_eq!(y6, 1 + v);
                } else {
                    assert_eq!(y6, SEND_Y6_MIN + v);
                }
                let mut cols = GlobalLookupOperation::<KoalaBear> {
                    offset: KoalaBear::ZERO,
                    x_coordinate: SepticBlock([KoalaBear::ZERO; 7]),
                    y_coordinate: SepticBlock([KoalaBear::ZERO; 7]),
                    y6_lo16: KoalaBear::ZERO,
                    y6_mid8: KoalaBear::ZERO,
                    y6_top: KoalaBear::ZERO,
                };
                cols.populate(values, is_receive, true, 3);
                assert_eq!(cols.offset, KoalaBear::from_u8(row.offset));
                assert_eq!(cols.x_coordinate.0[6], KoalaBear::from_u32(row.x[6]));
            }
        }
    }
}
