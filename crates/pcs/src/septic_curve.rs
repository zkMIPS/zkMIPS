//! Elliptic Curve `y^2 = x^3 + 3z*x - 3` over the `F_{p^7} = F_p[z]/(z^7 + 2z - 8)` extension field.
use crate::septic_extension::SepticExtension;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};
use serde::{Deserialize, Serialize};
use std::ops::Add;

/// A septic elliptic curve point on y^2 = x^3 + 3z*x - 3 over field `F_{p^7} = F_p[z]/(z^7 + 2z - 8)`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SepticCurve<F> {
    /// The x-coordinate of an elliptic curve point.
    pub x: SepticExtension<F>,
    /// The y-coordinate of an elliptic curve point.
    pub y: SepticExtension<F>,
}

/// The x-coordinate for a curve point used as a witness for padding lookups, derived from `e`.
pub const CURVE_WITNESS_DUMMY_POINT_X: [u32; 7] =
    // [0x65B0D64E, 0x4E8C0BFD, 0x8D4B5E6, 0x19A5AE9, 0x6932D4A4, 0x61F6B89C, 0x78D8D5D8];=
    [1706420302, 1319108093, 148224806, 26874985, 1766171812, 1645633948, 2028659224];

/// The y-coordinate for a curve point used as a witness for padding lookups, derived from `e`.
pub const CURVE_WITNESS_DUMMY_POINT_Y: [u32; 7] =
    [942390502, 1239997438, 458866455, 1843332012, 1309764648, 572807436, 74267719];

impl<F: Field> SepticCurve<F> {
    /// Returns the dummy point.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            x: SepticExtension::from_basis_coefficients_fn(|i| {
                F::from_u32(CURVE_WITNESS_DUMMY_POINT_X[i])
            }),
            y: SepticExtension::from_basis_coefficients_fn(|i| {
                F::from_u32(CURVE_WITNESS_DUMMY_POINT_Y[i])
            }),
        }
    }

    /// Check if a `SepticCurve` struct is on the elliptic curve.
    pub fn check_on_point(&self) -> bool {
        self.y.square() == Self::curve_formula(self.x)
    }

    /// Negates a `SepticCurve` point.
    #[must_use]
    pub fn neg(&self) -> Self {
        SepticCurve { x: self.x, y: -self.y }
    }

    #[must_use]
    /// Adds two elliptic curve points, assuming that the addition doesn't lead to the exception cases of weierstrass addition.
    pub fn add_incomplete(&self, other: SepticCurve<F>) -> Self {
        let slope = (other.y - self.y) / (other.x - self.x);
        let result_x = slope.square() - self.x - other.x;
        let result_y = slope * (self.x - result_x) - self.y;
        Self { x: result_x, y: result_y }
    }

    /// Add assigns an elliptic curve point, assuming that the addition doesn't lead to the exception cases of weierstrass addition.
    pub fn add_assign(&mut self, other: SepticCurve<F>) {
        let result = self.add_incomplete(other);
        self.x = result.x;
        self.y = result.y;
    }

    #[must_use]
    /// Double the elliptic curve point.
    pub fn double(&self) -> Self {
        let slope = (self.x * self.x * F::from_u8(3u8)
            + SepticExtension::from_base_slice(&[
                F::ZERO,
                F::from_u32(3),
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ]))
            / (self.y * F::TWO);
        let result_x = slope.square() - self.x * F::TWO;
        let result_y = slope * (self.x - result_x) - self.y;
        Self { x: result_x, y: result_y }
    }

    /// Subtracts two elliptic curve points, assuming that the subtraction doesn't lead to the exception cases of weierstrass addition.
    #[must_use]
    pub fn sub_incomplete(&self, other: SepticCurve<F>) -> Self {
        self.add_incomplete(other.neg())
    }

    /// Subtract assigns an elliptic curve point, assuming that the subtraction doesn't lead to the exception cases of weierstrass addition.
    pub fn sub_assign(&mut self, other: SepticCurve<F>) {
        let result = self.add_incomplete(other.neg());
        self.x = result.x;
        self.y = result.y;
    }
}

impl<F: PrimeCharacteristicRing> SepticCurve<F> {
    /// Evaluates the curve formula y^2 = x^3 + 3z*x -3
    pub fn curve_formula(x: SepticExtension<F>) -> SepticExtension<F> {
        x.cube()
            + x * SepticExtension::from_base_slice(&[
                F::ZERO,
                F::from_u32(3),
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ])
            - SepticExtension::from_base_slice(&[
                F::from_u32(3),
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ])
    }
}

impl<F: PrimeField32> SepticCurve<F> {
    /// Lift an x coordinate into an elliptic curve.
    /// As an x-coordinate may not be a valid one, we allow an additional value in `[0, 256)` to the hash input.
    /// Also, we always return the curve point with y-coordinate within `[1, (p-1)/2]`, where p is the characteristic.
    /// The returned values are the curve point and the offset used.
    pub fn lift_x(m: SepticExtension<F>) -> (Self, u8) {
        for offset in 0..=255 {
            let x_trial = SepticExtension::from_base_slice(&[
                m.0[0],
                m.0[1],
                m.0[2],
                m.0[3],
                m.0[4],
                m.0[5],
                m.0[6] * F::from_u16(256) + F::from_u8(offset),
            ]);

            let y_sq = Self::curve_formula(x_trial);
            if let Some(y) = y_sq.sqrt() {
                if y.is_exception() {
                    continue;
                }
                if y.is_send() {
                    return (Self { x: x_trial, y: -y }, offset);
                }
                return (Self { x: x_trial, y }, offset);
            }
        }
        panic!("curve point couldn't be found after 256 attempts");
    }
}

impl<F: PrimeCharacteristicRing> SepticCurve<F> {
    /// Given three points p1, p2, p3, the function is zero if and only if p3.x == (p1 + p2).x assuming that no weierstrass edge cases occur.
    pub fn sum_checker_x(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        (p1.x.clone() + p2.x.clone() + p3.x) * (p2.x.clone() - p1.x.clone()).square()
            - (p2.y - p1.y).square()
    }

    /// Given three points p1, p2, p3, the function is zero if and only if p3.y == (p1 + p2).y assuming that no weierstrass edge cases occur.
    pub fn sum_checker_y(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        (p1.y.clone() + p3.y.clone()) * (p2.x.clone() - p1.x.clone())
            - (p2.y - p1.y.clone()) * (p1.x - p3.x)
    }
}

impl<T> SepticCurve<T> {
    /// Convert a `SepticCurve<S>` into `SepticCurve<T>`, with a map that implements `FnMut(S) -> T`.
    pub fn convert<S: Copy, G: FnMut(S) -> T>(point: SepticCurve<S>, mut f: G) -> Self {
        SepticCurve {
            x: SepticExtension(point.x.0.map(&mut f)),
            y: SepticExtension(point.y.0.map(&mut f)),
        }
    }
}

/// A septic elliptic curve point on y^2 = x^3 + 3z*x - 3 over field `F_{p^7} = F_p[z]/(z^7 + 2z - 8)`, including the point at infinity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SepticCurveComplete<T> {
    /// The point at infinity.
    Infinity,
    /// The affine point which can be represented with a `SepticCurve<T>` structure.
    Affine(SepticCurve<T>),
}

impl<F: Field> Add for SepticCurveComplete<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_infinity() {
            return rhs;
        }
        if rhs.is_infinity() {
            return self;
        }
        let point1 = self.point();
        let point2 = rhs.point();
        if point1.x != point2.x {
            return Self::Affine(point1.add_incomplete(point2));
        }
        if point1.y == point2.y {
            return Self::Affine(point1.double());
        }
        Self::Infinity
    }
}

impl<F: Field> SepticCurveComplete<F> {
    /// Returns whether or not the point is a point at infinity.
    pub fn is_infinity(&self) -> bool {
        match self {
            Self::Infinity => true,
            Self::Affine(_) => false,
        }
    }

    /// Asserts that the point is not a point at infinity, and returns the `SepticCurve` value.
    pub fn point(&self) -> SepticCurve<F> {
        match self {
            Self::Infinity => panic!("point() called for point at infinity"),
            Self::Affine(point) => *point,
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_maybe_rayon::prelude::ParallelIterator;
    use p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
    use rayon_scan::ScanParallelIterator;
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_lift_x1() {
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
            KoalaBear::from_u32(1511106837),
            KoalaBear::from_u32(0),
            KoalaBear::from_u32(0),
            KoalaBear::from_u32(0),
            KoalaBear::from_u32(0),
            KoalaBear::from_u32(0),
            KoalaBear::from_u32(0),
        ]);
        let (curve_point, _) = SepticCurve::<KoalaBear>::lift_x(x);
        assert!(curve_point.check_on_point());
        println!("{curve_point:?}");
    }

    #[test]
    fn test_lift_x() {
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
            KoalaBear::from_u32(0x2013),
            KoalaBear::from_u32(0x2015),
            KoalaBear::from_u32(0x2016),
            KoalaBear::from_u32(0x2023),
            KoalaBear::from_u32(0x2024),
            KoalaBear::from_u32(0x2016),
            KoalaBear::from_u32(0x2017),
        ]);
        let (curve_point, _) = SepticCurve::<KoalaBear>::lift_x(x);
        assert!(curve_point.check_on_point());
    }

    #[test]
    fn test_double() {
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
            KoalaBear::from_u32(0x2013),
            KoalaBear::from_u32(0x2015),
            KoalaBear::from_u32(0x2016),
            KoalaBear::from_u32(0x2023),
            KoalaBear::from_u32(0x2024),
            KoalaBear::from_u32(0x2016),
            KoalaBear::from_u32(0x2017),
        ]);
        let (curve_point, _) = SepticCurve::<KoalaBear>::lift_x(x);
        let double_point = curve_point.double();
        assert!(double_point.check_on_point());
    }

    #[test]
    #[ignore]
    fn test_simple_bench() {
        const D: u32 = 1 << 16;
        let mut vec = Vec::with_capacity(D as usize);
        let mut sum = Vec::with_capacity(D as usize);
        let start = Instant::now();
        for i in 0..D {
            let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
                KoalaBear::from_u32(i + 25),
                KoalaBear::from_u32(2 * i + 376),
                KoalaBear::from_u32(4 * i + 23),
                KoalaBear::from_u32(8 * i + 531),
                KoalaBear::from_u32(16 * i + 542),
                KoalaBear::from_u32(32 * i + 196),
                KoalaBear::from_u32(64 * i + 667),
            ]);
            let (curve_point, _) = SepticCurve::<KoalaBear>::lift_x(x);
            vec.push(curve_point);
        }
        println!("Time elapsed: {:?}", start.elapsed());
        let start = Instant::now();
        for i in 0..D {
            sum.push(vec[i as usize].add_incomplete(vec[((i + 1) % D) as usize]));
        }
        println!("Time elapsed: {:?}", start.elapsed());
        let start = Instant::now();
        for i in 0..(D as usize) {
            assert!(
                SepticCurve::<KoalaBear>::sum_checker_x(vec[i], vec[(i + 1) % D as usize], sum[i])
                    == SepticExtension::<KoalaBear>::ZERO
            );
            assert!(
                SepticCurve::<KoalaBear>::sum_checker_y(vec[i], vec[(i + 1) % D as usize], sum[i])
                    == SepticExtension::<KoalaBear>::ZERO
            );
        }
        println!("Time elapsed: {:?}", start.elapsed());
    }

    #[test]
    #[ignore]
    fn test_parallel_bench() {
        const D: u32 = 1 << 20;
        let mut vec = Vec::with_capacity(D as usize);
        let start = Instant::now();
        for i in 0..D {
            let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
                KoalaBear::from_u32(i + 25),
                KoalaBear::from_u32(2 * i + 376),
                KoalaBear::from_u32(4 * i + 23),
                KoalaBear::from_u32(8 * i + 531),
                KoalaBear::from_u32(16 * i + 542),
                KoalaBear::from_u32(32 * i + 196),
                KoalaBear::from_u32(64 * i + 667),
            ]);
            let (curve_point, _) = SepticCurve::<KoalaBear>::lift_x(x);
            vec.push(SepticCurveComplete::Affine(curve_point));
        }
        println!("Time elapsed: {:?}", start.elapsed());

        let mut cum_sum = SepticCurveComplete::Infinity;
        let start = Instant::now();
        for point in &vec {
            cum_sum = cum_sum + *point;
        }
        println!("Time elapsed: {:?}", start.elapsed());
        let start = Instant::now();
        let par_sum = vec
            .into_par_iter()
            .with_min_len(1 << 16)
            .scan(|a, b| *a + *b, SepticCurveComplete::Infinity)
            .collect::<Vec<SepticCurveComplete<KoalaBear>>>();
        println!("Time elapsed: {:?}", start.elapsed());
        assert_eq!(cum_sum, *par_sum.last().unwrap());
    }
}

#[cfg(test)]
mod lift_x_bench {
    use super::*;
    use p3_koala_bear::KoalaBear;

    #[test]
    fn lift_x_cost() {
        let n = 20_000u32;
        let t = std::time::Instant::now();
        let mut acc = 0u64;
        for i in 0..n {
            let m = SepticExtension::<KoalaBear>::from_basis_coefficients_fn(|j| {
                KoalaBear::from_u32(i.wrapping_mul(2654435761).wrapping_add(j as u32 * 40503) & 0xFFFF)
            });
            let (p, off) = SepticCurve::<KoalaBear>::lift_x(m);
            acc += off as u64 + p.x.0[0].as_canonical_u32() as u64;
        }
        let dt = t.elapsed();
        println!("LIFT_X_BENCH n={n} total={:.3}s per={:.2}us acc={acc}", dt.as_secs_f64(), dt.as_secs_f64() * 1e6 / n as f64);
    }
}
