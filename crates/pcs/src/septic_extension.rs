//! A septic extension with an irreducible polynomial `z^7 + 2z - 8`.
use num_bigint::BigUint;
use num_traits::One;
use p3_field::PrimeField32;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, Packable, PackedFieldExtension,
    PrimeCharacteristicRing,
};
use serde::{Deserialize, Serialize};
use std::array;
use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::air::SepticExtensionAirBuilder;

/// A septic extension with an irreducible polynomial `z^7 + 2z - 8`.
///
/// The field can be constructed as `F_{p^7} = F_p[z]/(z^7 + 2z - 8)`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SepticExtension<F>(pub [F; 7]);

impl<F: PrimeCharacteristicRing + Copy> PrimeCharacteristicRing for SepticExtension<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self =
        SepticExtension([F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    const ONE: Self =
        SepticExtension([F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    const TWO: Self =
        SepticExtension([F::TWO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    const NEG_ONE: Self =
        SepticExtension([F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        SepticExtension([
            F::from_prime_subfield(f),
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ])
    }
}

impl<F: Field> Field for SepticExtension<F> {
    type Packing = Self;

    const GENERATOR: Self =
        SepticExtension([F::TWO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(self.inv())
    }

    fn order() -> BigUint {
        F::order().pow(7)
    }
}

impl<F: PrimeCharacteristicRing> BasedVectorSpace<F> for SepticExtension<F> {
    const DIMENSION: usize = 7;

    fn as_basis_coefficients_slice(&self) -> &[F] {
        self.0.as_slice()
    }

    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> F>(f: Fn) -> Self {
        Self(array::from_fn(f))
    }

    fn from_basis_coefficients_slice(bs: &[F]) -> Option<Self> {
        if bs.len() < 7 {
            return None;
        }
        Some(SepticExtension([
            bs[0].clone(),
            bs[1].clone(),
            bs[2].clone(),
            bs[3].clone(),
            bs[4].clone(),
            bs[5].clone(),
            bs[6].clone(),
        ]))
    }

    fn from_basis_coefficients_iter<I: Iterator<Item = F>>(mut iter: I) -> Option<Self> {
        let mut arr = [F::ZERO; 7];
        #[allow(clippy::needless_range_loop)]
        for i in 0..7 {
            arr[i] = iter.next()?;
        }
        Some(Self(arr))
    }
}

impl<F: PrimeCharacteristicRing + Copy> Algebra<F> for SepticExtension<F> {}

impl<F: Field<Packing = F>> PackedFieldExtension<F, SepticExtension<F>> for SepticExtension<F> {
    fn from_ext_slice(ext_slice: &[SepticExtension<F>]) -> Self {
        // Since SepticExtension is its own packing (WIDTH=1), just take the first element.
        ext_slice[0]
    }

    fn packed_ext_powers(base: SepticExtension<F>) -> p3_field::Powers<SepticExtension<F>> {
        base.powers()
    }
}

impl<F: Field<Packing = F>> ExtensionField<F> for SepticExtension<F> {
    type ExtensionPacking = Self;

    fn is_in_basefield(&self) -> bool {
        self.0[1..].iter().all(|x| x.is_zero())
    }

    fn as_base(&self) -> Option<F> {
        if <Self as ExtensionField<F>>::is_in_basefield(self) {
            Some(self.0[0])
        } else {
            None
        }
    }
}

impl<F: Field> Packable for SepticExtension<F> {}

impl<F: PrimeCharacteristicRing> Add for SepticExtension<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = self.0;
        for (r, rhs_val) in res.iter_mut().zip(rhs.0) {
            *r = (*r).clone() + rhs_val;
        }
        Self(res)
    }
}

impl<F: PrimeCharacteristicRing> AddAssign for SepticExtension<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0].clone();
        self.0[1] += rhs.0[1].clone();
        self.0[2] += rhs.0[2].clone();
        self.0[3] += rhs.0[3].clone();
        self.0[4] += rhs.0[4].clone();
        self.0[5] += rhs.0[5].clone();
        self.0[6] += rhs.0[6].clone();
    }
}

impl<F: PrimeCharacteristicRing> Sub for SepticExtension<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut res = self.0;
        for (r, rhs_val) in res.iter_mut().zip(rhs.0) {
            *r = (*r).clone() - rhs_val;
        }
        Self(res)
    }
}

impl<F: PrimeCharacteristicRing> SubAssign for SepticExtension<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0].clone();
        self.0[1] -= rhs.0[1].clone();
        self.0[2] -= rhs.0[2].clone();
        self.0[3] -= rhs.0[3].clone();
        self.0[4] -= rhs.0[4].clone();
        self.0[5] -= rhs.0[5].clone();
        self.0[6] -= rhs.0[6].clone();
    }
}

impl<F: PrimeCharacteristicRing> Neg for SepticExtension<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut res = self.0;
        for r in res.iter_mut() {
            *r = -r.clone();
        }
        Self(res)
    }
}

impl<F: PrimeCharacteristicRing> Mul for SepticExtension<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res: [F; 13] = core::array::from_fn(|_| F::ZERO);
        for i in 0..7 {
            for j in 0..7 {
                res[i + j] = res[i + j].clone() + self.0[i].clone() * rhs.0[j].clone();
            }
        }
        let mut ret: [F; 7] = core::array::from_fn(|i| res[i].clone());
        for i in 7..13 {
            ret[i - 7] = ret[i - 7].clone() + res[i].clone() * F::from_u32(8);
            ret[i - 6] = ret[i - 6].clone() - res[i].clone() * F::from_u32(2);
        }
        Self(ret)
    }
}

impl<F: PrimeCharacteristicRing> MulAssign for SepticExtension<F> {
    fn mul_assign(&mut self, rhs: Self) {
        let res = self.clone() * rhs;
        *self = res;
    }
}

impl<F: PrimeCharacteristicRing + Copy> Product for SepticExtension<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let one = Self::ONE;
        iter.fold(one, |acc, x| acc * x)
    }
}

impl<F: PrimeCharacteristicRing + Copy> Sum for SepticExtension<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = Self::ZERO;
        iter.fold(zero, |acc, x| acc + x)
    }
}

impl<F: PrimeCharacteristicRing> From<F> for SepticExtension<F> {
    fn from(f: F) -> Self {
        SepticExtension([f, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO])
    }
}

impl<F: PrimeCharacteristicRing> Add<F> for SepticExtension<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        SepticExtension([
            self.0[0].clone() + rhs,
            self.0[1].clone(),
            self.0[2].clone(),
            self.0[3].clone(),
            self.0[4].clone(),
            self.0[5].clone(),
            self.0[6].clone(),
        ])
    }
}

impl<F: PrimeCharacteristicRing> AddAssign<F> for SepticExtension<F> {
    fn add_assign(&mut self, rhs: F) {
        self.0[0] += rhs;
    }
}

impl<F: PrimeCharacteristicRing> Sub<F> for SepticExtension<F> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self + (-rhs)
    }
}

impl<F: PrimeCharacteristicRing> SubAssign<F> for SepticExtension<F> {
    fn sub_assign(&mut self, rhs: F) {
        self.0[0] -= rhs;
    }
}

impl<F: PrimeCharacteristicRing> Mul<F> for SepticExtension<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        SepticExtension([
            self.0[0].clone() * rhs.clone(),
            self.0[1].clone() * rhs.clone(),
            self.0[2].clone() * rhs.clone(),
            self.0[3].clone() * rhs.clone(),
            self.0[4].clone() * rhs.clone(),
            self.0[5].clone() * rhs.clone(),
            self.0[6].clone() * rhs.clone(),
        ])
    }
}

impl<F: PrimeCharacteristicRing> MulAssign<F> for SepticExtension<F> {
    fn mul_assign(&mut self, rhs: F) {
        for i in 0..7 {
            self.0[i] *= rhs.clone();
        }
    }
}

impl<F: Field> Div for SepticExtension<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: Field> DivAssign for SepticExtension<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Field> p3_field::RawDataSerializable for SepticExtension<F> {
    const NUM_BYTES: usize = F::NUM_BYTES * 7;

    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.0.into_iter().flat_map(|f| f.into_bytes().into_iter().collect::<Vec<_>>())
    }
}

impl<F: PrimeCharacteristicRing> Display for SepticExtension<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<F: PrimeCharacteristicRing> SepticExtension<F> {
    /// A function similar to `core::array::from_fn`.
    pub fn from_base_fn<G: FnMut(usize) -> F>(f: G) -> Self {
        Self(array::from_fn(f))
    }

    /// Creates a `SepticExtension` from a slice of base field elements.
    /// Panics if the slice has fewer than 7 elements.
    pub fn from_base_slice(bs: &[F]) -> Self {
        SepticExtension([
            bs[0].clone(),
            bs[1].clone(),
            bs[2].clone(),
            bs[3].clone(),
            bs[4].clone(),
            bs[5].clone(),
            bs[6].clone(),
        ])
    }

    /// Squares the extension element. Unlike `PrimeCharacteristicRing::square()`,
    /// this only requires `Clone` (not `Copy`).
    pub fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    /// Cubes the extension element. Unlike `PrimeCharacteristicRing::cube()`,
    /// this only requires `Clone` (not `Copy`).
    pub fn cube(&self) -> Self {
        self.square() * self.clone()
    }
}

impl<F: Field> SepticExtension<F> {
    /// Returns the value of z^{index * p} in the [`SepticExtension`] field.
    fn z_pow_p(index: u32) -> Self {
        // The constants written below are specifically for the KoalaBear field.
        debug_assert_eq!(F::order(), BigUint::from(2130706433u32));
        if index == 0 {
            return Self::ONE;
        }
        if index == 1 {
            return SepticExtension([
                F::from_u32(587483156),
                F::from_u32(843070426),
                F::from_u32(856916903),
                F::from_u32(802055410),
                F::from_u32(1274370027),
                F::from_u32(839777993),
                F::from_u32(1763169463),
            ]);
        }
        if index == 2 {
            return SepticExtension([
                F::from_u32(1211185764),
                F::from_u32(536911287),
                F::from_u32(1786731555),
                F::from_u32(1891857573),
                F::from_u32(591969516),
                F::from_u32(550155966),
                F::from_u32(706525029),
            ]);
        }
        if index == 3 {
            return SepticExtension([
                F::from_u32(926148950),
                F::from_u32(97341948),
                F::from_u32(1328592391),
                F::from_u32(2024338901),
                F::from_u32(1053611575),
                F::from_u32(858809194),
                F::from_u32(895371293),
            ]);
        }
        if index == 4 {
            return SepticExtension([
                F::from_u32(1525385643),
                F::from_u32(1541060576),
                F::from_u32(1544460289),
                F::from_u32(1695665723),
                F::from_u32(1260084848),
                F::from_u32(209013872),
                F::from_u32(1422484900),
            ]);
        }
        if index == 5 {
            return SepticExtension([
                F::from_u32(636881039),
                F::from_u32(1369380874),
                F::from_u32(1823056783),
                F::from_u32(411001166),
                F::from_u32(474370133),
                F::from_u32(1991878855),
                F::from_u32(193955070),
            ]);
        }
        if index == 6 {
            return SepticExtension([
                F::from_u32(448462982),
                F::from_u32(1809047550),
                F::from_u32(1873051132),
                F::from_u32(1563342685),
                F::from_u32(638206204),
                F::from_u32(1034022669),
                F::from_u32(616721146),
            ]);
        }
        unreachable!();
    }

    /// Returns the value of z^{index * p^2} in the [`SepticExtension`] field.
    fn z_pow_p2(index: u32) -> Self {
        // The constants written below are specifically for the KoalaBear field.
        debug_assert_eq!(F::order(), BigUint::from(2130706433u32));
        if index == 0 {
            return Self::ONE;
        }
        if index == 1 {
            return SepticExtension([
                F::from_u32(850855402),
                F::from_u32(83752463),
                F::from_u32(578907183),
                F::from_u32(1077461187),
                F::from_u32(841195559),
                F::from_u32(707516819),
                F::from_u32(141214579),
            ]);
        }
        if index == 2 {
            return SepticExtension([
                F::from_u32(836146895),
                F::from_u32(2043859405),
                F::from_u32(2072756292),
                F::from_u32(685210173),
                F::from_u32(510761813),
                F::from_u32(193547797),
                F::from_u32(310193486),
            ]);
        }
        if index == 3 {
            return SepticExtension([
                F::from_u32(1605797233),
                F::from_u32(989471584),
                F::from_u32(1210699680),
                F::from_u32(1003960530),
                F::from_u32(1444517609),
                F::from_u32(759580625),
                F::from_u32(1114273922),
            ]);
        }
        if index == 4 {
            return SepticExtension([
                F::from_u32(1181931158),
                F::from_u32(511865135),
                F::from_u32(172170608),
                F::from_u32(1549372938),
                F::from_u32(153489079),
                F::from_u32(1246252776),
                F::from_u32(1044577581),
            ]);
        }
        if index == 5 {
            return SepticExtension([
                F::from_u32(682248311),
                F::from_u32(1022876955),
                F::from_u32(1873346400),
                F::from_u32(850875418),
                F::from_u32(605656029),
                F::from_u32(190509635),
                F::from_u32(220419312),
            ]);
        }
        if index == 6 {
            return SepticExtension([
                F::from_u32(688846502),
                F::from_u32(1836380477),
                F::from_u32(172054673),
                F::from_u32(688169080),
                F::from_u32(187745906),
                F::from_u32(414105003),
                F::from_u32(756944866),
            ]);
        }
        unreachable!();
    }

    #[must_use]
    fn frobenius(&self) -> Self {
        let mut result = Self::ZERO;
        result += self.0[0];
        result += Self::z_pow_p(1) * self.0[1];
        result += Self::z_pow_p(2) * self.0[2];
        result += Self::z_pow_p(3) * self.0[3];
        result += Self::z_pow_p(4) * self.0[4];
        result += Self::z_pow_p(5) * self.0[5];
        result += Self::z_pow_p(6) * self.0[6];
        result
    }

    #[must_use]
    fn double_frobenius(&self) -> Self {
        let mut result = Self::ZERO;
        result += self.0[0];
        result += Self::z_pow_p2(1) * self.0[1];
        result += Self::z_pow_p2(2) * self.0[2];
        result += Self::z_pow_p2(3) * self.0[3];
        result += Self::z_pow_p2(4) * self.0[4];
        result += Self::z_pow_p2(5) * self.0[5];
        result += Self::z_pow_p2(6) * self.0[6];
        result
    }

    #[must_use]
    fn pow_r_1(&self) -> Self {
        let base = self.frobenius() * self.double_frobenius();
        let base_p2 = base.double_frobenius();
        let base_p4 = base_p2.double_frobenius();
        base * base_p2 * base_p4
    }

    #[must_use]
    fn inv(&self) -> Self {
        let pow_r_1 = self.pow_r_1();
        let pow_r = pow_r_1 * *self;
        pow_r_1 * pow_r.0[0].inverse()
    }

    fn is_square(&self) -> (F, bool) {
        let pow_r_1 = self.pow_r_1();
        let pow_r = pow_r_1 * *self;
        let exp = (F::order() - BigUint::one()) / BigUint::from(2u8);
        let exp = exp.to_u64_digits()[0];

        (pow_r.0[0], pow_r.0[0].exp_u64(exp) == F::ONE)
    }

    /// Computes the square root of the septic field extension element.
    /// Returns None if the element is not a square, and Some(result) if it is a square.
    pub fn sqrt(&self) -> Option<Self> {
        let n = *self;

        if n == Self::ZERO || n == Self::ONE {
            return Some(n);
        }

        let (numerator, is_square) = n.is_square();

        if !is_square {
            return None;
        }

        let mut n_iter = n;
        let mut n_power = n;
        for i in 1..30 {
            n_iter *= n_iter;
            if i >= 23 {
                n_power *= n_iter;
            }
        }

        let mut n_frobenius = n_power.frobenius();
        let mut denominator = n_frobenius;

        n_frobenius = n_frobenius.double_frobenius();
        denominator *= n_frobenius;
        n_frobenius = n_frobenius.double_frobenius();
        denominator *= n_frobenius;
        denominator *= n;

        let base = numerator.inverse();
        let g = F::GENERATOR;
        let mut a = F::ONE;
        let mut nonresidue = F::ONE - base;
        let legendre_exp = (F::order() - BigUint::one()) / BigUint::from(2u8);

        while nonresidue.exp_u64(legendre_exp.to_u64_digits()[0]) == F::ONE {
            a *= g;
            nonresidue = a.square() - base;
        }

        let order = F::order();
        let cipolla_pow = (&order + BigUint::one()) / BigUint::from(2u8);
        let mut x = CipollaExtension::new(a, F::ONE);
        x = x.pow(&cipolla_pow, nonresidue);

        Some(denominator * x.real)
    }
}

/// Digest sign convention (shared with the `GlobalLookupOperation` AIR and the
/// CUDA `kb31_septic_extension_t` header — keep the three in sync).
///
/// A point whose top `y` limb lies in `1..=RECEIVE_Y6_MAX` encodes a receive,
/// one in `SEND_Y6_MIN..=p-1` a send; `y6 == 0` and the band
/// `RECEIVE_Y6_MAX+1 ..= SEND_Y6_MIN-1` are exceptions and `lift_x` moves on
/// to the next offset.  The bands are mirror images (`p - RECEIVE_Y6_MAX ==
/// SEND_Y6_MIN`), so negating a send gives a receive and vice versa.
///
/// `RECEIVE_Y6_MAX = 63 * 2^24` lets the AIR prove `y6 - 1 < 63 * 2^24` with a
/// 16-bit limb, an 8-bit limb and one `LTU(top, 63)` byte lookup (3 lookups,
/// 3 columns) instead of a 30-bit boolean decomposition.  The band costs one
/// extra `lift_x` retry in ~2^-7 of the cases.
pub const RECEIVE_Y6_MAX: u32 = 63 << 24;
pub const SEND_Y6_MIN: u32 = (1 << 30) + 1;

impl<F: PrimeField32> SepticExtension<F> {
    /// Returns whether the extension field element viewed as an y-coordinate of a digest represents a receive lookup.
    pub fn is_receive(&self) -> bool {
        let limb = self.0[6].as_canonical_u32();
        1 <= limb && limb <= RECEIVE_Y6_MAX
    }

    /// Returns whether the extension field element viewed as an y-coordinate of a digest represents a send lookup.
    pub fn is_send(&self) -> bool {
        let limb = self.0[6].as_canonical_u32();
        SEND_Y6_MIN <= limb && limb <= (F::ORDER_U32 - 1)
    }

    /// Returns whether the extension field element viewed as an y-coordinate of a digest cannot represent anything.
    pub fn is_exception(&self) -> bool {
        !self.is_receive() && !self.is_send()
    }
}

/// Extension field for Cipolla's algorithm, taken from <https://github.com/Plonky3/Plonky3/pull/439/files>.
#[derive(Clone, Copy, Debug)]
struct CipollaExtension<F: Field> {
    real: F,
    imag: F,
}

impl<F: Field> CipollaExtension<F> {
    fn new(real: F, imag: F) -> Self {
        Self { real, imag }
    }

    fn one() -> Self {
        Self::new(F::ONE, F::ZERO)
    }

    fn mul_ext(&self, other: Self, nonresidue: F) -> Self {
        Self::new(
            self.real * other.real + nonresidue * self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    fn pow(&self, exp: &BigUint, nonresidue: F) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        let bits = exp.bits();

        for i in 0..bits {
            if exp.bit(i) {
                result = result.mul_ext(base, nonresidue);
            }
            base = base.mul_ext(base, nonresidue);
        }
        result
    }
}

/// A block of columns for septic extension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct SepticBlock<T>(pub [T; 7]);

impl<T> SepticBlock<T> {
    /// Maps a `SepticBlock<T>` to `SepticBlock<U>` based on a map from `T` to `U`.
    pub fn map<F, U>(self, f: F) -> SepticBlock<U>
    where
        F: FnMut(T) -> U,
    {
        SepticBlock(self.0.map(f))
    }

    /// A function similar to `core:array::from_fn`.
    pub fn from_base_fn<G: FnMut(usize) -> T>(f: G) -> Self {
        Self(array::from_fn(f))
    }
}

impl<T: Clone> SepticBlock<T> {
    /// Takes a `SepticBlock` into a `SepticExtension` of expressions.
    pub fn as_extension<AB: SepticExtensionAirBuilder<Var = T>>(
        &self,
    ) -> SepticExtension<AB::Expr> {
        let arr: [AB::Expr; 7] = self.0.clone().map(|x| AB::Expr::ZERO + x);
        SepticExtension(arr)
    }
}

impl<T> From<[T; 7]> for SepticBlock<T> {
    fn from(arr: [T; 7]) -> Self {
        Self(arr)
    }
}

impl<T: PrimeCharacteristicRing> From<T> for SepticBlock<T> {
    fn from(value: T) -> Self {
        Self([value, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO])
    }
}

impl<T: Copy> From<&[T]> for SepticBlock<T> {
    fn from(slice: &[T]) -> Self {
        let arr: [T; 7] = slice.try_into().unwrap();
        Self(arr)
    }
}

impl<T, I> Index<I> for SepticBlock<T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.0, index)
    }
}

impl<T, I> IndexMut<I> for SepticBlock<T>
where
    [T]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.0, index)
    }
}

impl<T> IntoIterator for SepticBlock<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, 7>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;

    use super::*;

    #[test]
    fn test_mul() {
        let a: SepticExtension<KoalaBear> = SepticExtension::from_u32(1);
        let b: SepticExtension<KoalaBear> = SepticExtension::from_u32(2);
        let c = a * b;
        println!("{c}");

        let i = 0;
        let a: SepticExtension<KoalaBear> = SepticExtension([
            KoalaBear::from_u32(i + 3),
            KoalaBear::from_u32(2 * i + 6),
            KoalaBear::from_u32(5 * i + 17),
            KoalaBear::from_u32(6 * i + 91),
            KoalaBear::from_u32(8 * i + 37),
            KoalaBear::from_u32(11 * i + 35),
            KoalaBear::from_u32(14 * i + 33),
        ]);
        let i = 1;
        let b: SepticExtension<KoalaBear> = SepticExtension([
            KoalaBear::from_u32(i + 3),
            KoalaBear::from_u32(2 * i + 6),
            KoalaBear::from_u32(5 * i + 17),
            KoalaBear::from_u32(6 * i + 91),
            KoalaBear::from_u32(8 * i + 37),
            KoalaBear::from_u32(11 * i + 35),
            KoalaBear::from_u32(14 * i + 33),
        ]);
        let c = a * b;
        println!("{c}");
    }

    #[test]
    fn test_inv() {
        for i in 0..256 {
            let a: SepticExtension<KoalaBear> = SepticExtension([
                KoalaBear::from_u32(i + 3),
                KoalaBear::from_u32(2 * i + 6),
                KoalaBear::from_u32(5 * i + 17),
                KoalaBear::from_u32(6 * i + 91),
                KoalaBear::from_u32(8 * i + 37),
                KoalaBear::from_u32(11 * i + 35),
                KoalaBear::from_u32(14 * i + 33),
            ]);
            let b = a.inv();
            assert_eq!(a * b, SepticExtension::<KoalaBear>::ONE);
        }
    }

    #[test]
    fn test_legendre() {
        let a: SepticExtension<KoalaBear> = SepticExtension::GENERATOR;
        let mut b = SepticExtension::<KoalaBear>::ONE;
        for i in 1..256 {
            b *= a;
            let (_, c) = b.is_square();
            assert!(c == (i % 2 == 0));
        }
    }

    #[test]
    fn test_sqrt() {
        for i in 0..256 {
            let a: SepticExtension<KoalaBear> = SepticExtension([
                KoalaBear::from_u32(i + 3),
                KoalaBear::from_u32(2 * i + 6),
                KoalaBear::from_u32(5 * i + 17),
                KoalaBear::from_u32(6 * i + 91),
                KoalaBear::from_u32(8 * i + 37),
                KoalaBear::from_u32(11 * i + 35),
                KoalaBear::from_u32(14 * i + 33),
            ]);
            let b = a * a;
            let recovered_a = b.sqrt().unwrap();
            assert_eq!(recovered_a * recovered_a, b);
        }
        let mut b = SepticExtension::<KoalaBear>::ONE;
        for i in 1..256 {
            let a: SepticExtension<KoalaBear> = SepticExtension::GENERATOR;
            b *= a;
            let c = b.sqrt();
            if i % 2 == 1 {
                assert!(c.is_none());
            } else {
                let c = c.unwrap();
                assert_eq!(c * c, b);
            }
        }
    }
}
