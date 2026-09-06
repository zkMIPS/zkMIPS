//! An embeddable signed/unsigned less-than gadget — the `Lt` CHIP's compare
//! logic, extracted so instruction chips can prove their own comparisons
//! instead of pushing SLT/SLTU request rows onto the `Lt` chip over the
//! Instruction bus.
//!
//! Semantics are EXACTLY the MIPS `SLT`/`SLTU` the `Lt` chip proves:
//!
//! > SLT (signed) = b_bit·(1 − c_bit) + (b_bit == c_bit)·SLTU(b_comp, c_comp)
//!
//! (Jolt 5.3), where for `SLT` the top bytes are masked (`& 0x7f`) and the
//! sign bits are handled through `b_bit`/`c_bit`; for `SLTU` the raw bytes
//! compare directly.  On top of the chip logic this gadget also exposes TRUE
//! equality (`eq = is_comp_eq · is_sign_eq`): masked-bytes-equal AND
//! same-sign-bits is full 32-bit equality in both modes.

use itertools::izip;
use p3_air::AirBuilder;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use zkm_pcs::air::BaseAirBuilder;

use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_pcs::{air::ZKMAirBuilder, Word};

/// Columns for one signed/unsigned less-than comparison of two words.
///
/// The caller passes `is_slt` / `is_sltu` selector EXPRESSIONS (boolean, at
/// most one set on a real row, both zero on padding); `is_real` is their sum.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LtOperation<T> {
    /// `b[3] & 0x7f` — the masked top byte for the signed compare.
    pub b_masked: T,
    /// `c[3] & 0x7f`.
    pub c_masked: T,
    /// `SLTU(b_comp, c_comp)` over the (masked) byte arrays.
    pub sltu: T,
    /// Whether `b_comp == c_comp` (masked equality).
    pub is_comp_eq: T,
    /// At most one flag marks the most significant differing byte.
    pub byte_flags: [T; 4],
    /// Inverse hint proving the comparison bytes differ when `!is_comp_eq`.
    pub not_eq_inv: T,
    /// The differing byte pair fed to the `LTU` byte lookup.
    pub comparison_bytes: [T; 2],
    /// The sign bits of `b` / `c` (zero in `SLTU` mode via `bit_*`).
    pub msb_b: T,
    pub msb_c: T,
    /// Whether the effective sign bits agree.
    pub is_sign_eq: T,
    /// `msb_b · is_slt` / `msb_c · is_slt` (materialised to keep degrees low).
    pub bit_b: T,
    pub bit_c: T,
    /// The comparison result: `b < c` under the selected signedness.
    pub lt: T,
}

impl<F: PrimeField32> LtOperation<F> {
    /// Populate for a real comparison row, emitting the byte events the
    /// constraints request (2×AND for the masks, 1×LTU for the comparison
    /// bytes) — the mirror of `LtChip::event_to_row`.
    pub fn populate(&mut self, record: &mut impl ByteRecord, b: u32, c: u32, signed: bool) {
        let b_bytes = b.to_le_bytes();
        let c_bytes = c.to_le_bytes();

        let masked_b = b_bytes[3] & 0x7f;
        let masked_c = c_bytes[3] & 0x7f;
        self.b_masked = F::from_u8(masked_b);
        self.c_masked = F::from_u8(masked_c);

        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_b as u16,
            a2: 0,
            b: b_bytes[3],
            c: 0x7f,
        });
        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_c as u16,
            a2: 0,
            b: c_bytes[3],
            c: 0x7f,
        });

        let mut b_comp = b_bytes;
        let mut c_comp = c_bytes;
        if signed {
            b_comp[3] = masked_b;
            c_comp[3] = masked_c;
        }
        self.sltu = F::from_bool(b_comp < c_comp);
        self.is_comp_eq = F::from_bool(b_comp == c_comp);

        for (b_byte, c_byte, flag) in
            izip!(b_comp.iter().rev(), c_comp.iter().rev(), self.byte_flags.iter_mut().rev())
        {
            if b_byte != c_byte {
                *flag = F::ONE;
                self.sltu = F::from_bool(b_byte < c_byte);
                let b_byte_f = F::from_u8(*b_byte);
                let c_byte_f = F::from_u8(*c_byte);
                self.not_eq_inv = (b_byte_f - c_byte_f).inverse();
                self.comparison_bytes = [b_byte_f, c_byte_f];
                break;
            }
        }

        self.msb_b = F::from_u8((b_bytes[3] >> 7) & 1);
        self.msb_c = F::from_u8((c_bytes[3] >> 7) & 1);
        self.is_sign_eq =
            if signed { F::from_bool((b_bytes[3] >> 7) == (c_bytes[3] >> 7)) } else { F::ONE };
        if signed {
            self.bit_b = self.msb_b;
            self.bit_c = self.msb_c;
        }
        self.lt = self.bit_b * (F::ONE - self.bit_c) + self.is_sign_eq * self.sltu;

        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: self.sltu.as_canonical_u32() as u16,
            a2: 0,
            b: self.comparison_bytes[0].as_canonical_u32() as u8,
            c: self.comparison_bytes[1].as_canonical_u32() as u8,
        });
    }
}

impl<F: Field> LtOperation<F> {
    /// The constraint mirror of `LtChip::eval` minus the chip plumbing
    /// (selectors, frame, Instruction-bus receive).  The caller guarantees
    /// `is_slt` / `is_sltu` are boolean with at most one set, and both zero on
    /// non-real rows; `b` / `c` must be valid byte-words.
    pub fn eval<AB: ZKMAirBuilder>(
        builder: &mut AB,
        b: Word<AB::Var>,
        c: Word<AB::Var>,
        cols: &LtOperation<AB::Var>,
        is_slt: AB::Expr,
        is_sltu: AB::Expr,
    ) {
        let is_real = is_slt.clone() + is_sltu.clone();

        // Masked comparison operands: raw bytes for SLTU, `& 0x7fffffff` for SLT.
        let mut b_comp: Word<AB::Expr> = b.map(|x| x.into());
        let mut c_comp: Word<AB::Expr> = c.map(|x| x.into());
        b_comp[3] = b[3] * is_sltu.clone() + cols.b_masked * is_slt.clone();
        c_comp[3] = c[3] * is_sltu.clone() + cols.c_masked * is_slt.clone();

        // The masks hold: `b_masked = b[3] & 0x7f`, `c_masked = c[3] & 0x7f`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            cols.b_masked,
            b[3],
            AB::F::from_u8(0x7f),
            is_real.clone(),
        );
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            cols.c_masked,
            c[3],
            AB::F::from_u8(0x7f),
            is_real.clone(),
        );

        // Effective sign bits.
        builder.assert_eq(cols.bit_b, cols.msb_b * is_slt.clone());
        builder.assert_eq(cols.bit_c, cols.msb_c * is_slt.clone());
        let inv_128 = AB::F::from_u32(128).inverse();
        builder.assert_eq(cols.msb_b, (b[3] - cols.b_masked) * inv_128);
        builder.assert_eq(cols.msb_c, (c[3] - cols.c_masked) * inv_128);

        // `is_sign_eq <=> bit_b == bit_c`.
        builder.assert_bool(cols.is_sign_eq);
        builder.when(cols.is_sign_eq).assert_eq(cols.bit_b, cols.bit_c);
        builder.when(is_real.clone()).when_not(cols.is_sign_eq).assert_one(cols.bit_b + cols.bit_c);

        // The result: `lt = bit_b·(1 − bit_c) + is_sign_eq·sltu`.
        builder.assert_eq(
            cols.lt,
            cols.bit_b * (AB::Expr::ONE - cols.bit_c) + cols.is_sign_eq * cols.sltu,
        );

        // Byte flags: boolean, at most one set, none set iff masked equality.
        let sum_flags =
            cols.byte_flags[0] + cols.byte_flags[1] + cols.byte_flags[2] + cols.byte_flags[3];
        builder.assert_bool(cols.byte_flags[0]);
        builder.assert_bool(cols.byte_flags[1]);
        builder.assert_bool(cols.byte_flags[2]);
        builder.assert_bool(cols.byte_flags[3]);
        builder.assert_bool(sum_flags.clone());
        builder.when(is_real.clone()).assert_eq(AB::Expr::ONE - cols.is_comp_eq, sum_flags);
        builder.assert_bool(cols.is_comp_eq);

        // Walk bytes most-significant first: everything above the flagged
        // byte must be equal, and the flagged pair feeds the LTU lookup.
        let mut is_inequality_visited = AB::Expr::ZERO;
        let mut b_comparison_byte = AB::Expr::ZERO;
        let mut c_comparison_byte = AB::Expr::ZERO;
        for (b_byte, c_byte, &flag) in
            izip!(b_comp.0.iter().rev(), c_comp.0.iter().rev(), cols.byte_flags.iter().rev())
        {
            is_inequality_visited = is_inequality_visited.clone() + flag.into();
            b_comparison_byte = b_comparison_byte.clone() + b_byte.clone() * flag;
            c_comparison_byte = c_comparison_byte.clone() + c_byte.clone() * flag;
            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_byte.clone(), c_byte.clone());
            builder.when(cols.is_comp_eq).assert_zero(is_inequality_visited.clone());
        }
        let (b_comp_byte, c_comp_byte) = (cols.comparison_bytes[0], cols.comparison_bytes[1]);
        builder.assert_eq(b_comp_byte, b_comparison_byte);
        builder.assert_eq(c_comp_byte, c_comparison_byte);

        // When not equal, the comparison bytes genuinely differ.
        builder
            .when_not(cols.is_comp_eq)
            .assert_eq(cols.not_eq_inv * (b_comp_byte - c_comp_byte), is_real.clone());

        // `sltu = LTU(b_comp_byte, c_comp_byte)` via lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            cols.sltu,
            b_comp_byte,
            c_comp_byte,
            is_real.clone(),
        );
    }
}
