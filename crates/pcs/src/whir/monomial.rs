//! Monomial-basis primitives for WHIR — the basis the STIR query phase needs.
//!
//! WHIR commits the polynomial as **monomial coefficients** (the codeword is a
//! DFT of coefficients), and answers OOD / STIR constraints with the monomial
//! evaluation, not the Lagrange one.  The distinction is exactly what an earlier
//! set of probes tripped over: Ziren's [`crate::basefold::mle::Mle::eval_at`] is
//! the Lagrange multilinear extension (`lo+α(hi−lo)`), whereas the codeword is a
//! coefficient-basis univariate.  These helpers provide the monomial side.
//!
//! Conventions mirror upstream `slop_multilinear`:
//!   * [`monomial_partial_eq`] builds `v[i] = Π_j point_j^{bit_j(i)}` with
//!     **big-endian** bits (`point[0]` is the most-significant bit of `i`),
//!     round rule `[val, val·coord]` — the monomial analogue of the Lagrange
//!     `[val·(1−coord), val·coord]`.
//!   * [`mono_eval`] evaluates a coefficient vector: `Σ_i coeff_i · v[i]`.
//!   * [`map_to_pow`] maps a domain element `x` to `[x^{2^{k-1}}, …, x², x]`,
//!     the point at which a monomial multilinear eval equals the univariate
//!     `Σ_i coeff_i x^i` (the keystone that makes STIR's codeword-fold a
//!     multilinear constraint — see the `keystone_*` tests).
//!   * [`full_monomial_basis_eq`] is `Π_j (a_j·b_j + 1 − b_j)`.

use alloc::vec::Vec;

use p3_field::Field;

/// `v[i] = Π_j point_j^{bit_j(i)}`, big-endian (`point[0]` = MSB of `i`).
pub fn monomial_partial_eq<EF: Field>(point: &[EF]) -> Vec<EF> {
    let mut evals = alloc::vec![EF::ONE];
    for &coord in point {
        let mut next = Vec::with_capacity(evals.len() * 2);
        for v in &evals {
            next.push(*v);
            next.push(*v * coord);
        }
        evals = next;
    }
    evals
}

/// Monomial multilinear evaluation of a coefficient vector at `point`.
pub fn mono_eval<F: Field, EF: p3_field::ExtensionField<F>>(coeffs: &[F], point: &[EF]) -> EF {
    let w = monomial_partial_eq(point);
    debug_assert_eq!(w.len(), coeffs.len());
    coeffs.iter().zip(&w).map(|(&c, &wi)| wi * c).sum()
}

/// `[x^{2^{len-1}}, …, x², x]` — the point where a monomial multilinear eval
/// equals the univariate `Σ_i coeff_i x^i`.
pub fn map_to_pow<EF: Field>(x: EF, len: usize) -> Vec<EF> {
    let mut res = Vec::with_capacity(len);
    let mut e = x;
    for _ in 0..len {
        res.push(e);
        e = e.square();
    }
    res.reverse();
    res
}

/// `Π_j (a_j·b_j + 1 − b_j)` — the monomial analogue of the Lagrange `eq`.
pub fn full_monomial_basis_eq<EF: Field>(a: &[EF], b: &[EF]) -> EF {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y + EF::ONE - y).product()
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;

    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::{full_monomial_basis_eq, map_to_pow, mono_eval, monomial_partial_eq};
    use crate::kb31_poseidon2::{InnerChallenge, InnerVal};

    type F = InnerVal;
    type EF = InnerChallenge;

    fn clamp(r: &mut StdRng) -> F {
        F::from_u32(r.gen::<u32>() & 0x3FFF_FFFF)
    }
    fn rand_ef(r: &mut StdRng) -> EF {
        use p3_field::BasedVectorSpace;
        <EF as BasedVectorSpace<F>>::from_basis_coefficients_iter((0..4).map(|_| clamp(r))).unwrap()
    }

    /// The keystone: a monomial multilinear eval at `map_to_pow(x)` equals the
    /// univariate `Σ_i coeff_i x^i`.  This is what turns a codeword value (a
    /// univariate evaluation) into a multilinear constraint for the sumcheck.
    #[test]
    fn keystone_mono_eval_is_univariate() {
        let m = 5usize;
        let mut rng = StdRng::seed_from_u64(0x1);
        let coeffs: Vec<F> = (0..(1usize << m)).map(|_| clamp(&mut rng)).collect();
        for _ in 0..8 {
            let x = rand_ef(&mut rng);
            // Direct univariate Σ coeff_i x^i.
            let mut uni = EF::ZERO;
            let mut xp = EF::ONE;
            for &c in &coeffs {
                uni += xp * c;
                xp *= x;
            }
            assert_eq!(mono_eval(&coeffs, &map_to_pow(x, m)), uni);
        }
    }

    /// The codeword bridge: `dft_batch` of the coefficients gives, at domain
    /// point `g^j`, exactly `mono_eval(coeffs, map_to_pow(g^j))`.  So a STIR
    /// codeword opening IS a monomial multilinear evaluation.
    #[test]
    fn keystone_codeword_is_mono_eval() {
        let m = 4usize;
        let log_blowup = 2usize;
        let mut rng = StdRng::seed_from_u64(0x2);
        let coeffs: Vec<F> = (0..(1usize << m)).map(|_| clamp(&mut rng)).collect();

        let dft = Radix2DitParallel::<F>::default();
        // dft_batch on coefficients (padded) gives evals in natural order.
        let mut padded = coeffs.clone();
        padded.resize(padded.len() << log_blowup, F::ZERO);
        let evals = dft.dft_batch(RowMajorMatrix::new(padded, 1)).to_row_major_matrix();
        let log_h = m + log_blowup;
        let g = F::two_adic_generator(log_h);

        for j in [0usize, 1, 5, 11, 30] {
            let cj: EF = EF::from(evals.values[j]);
            let x: EF = EF::from(g.exp_u64(j as u64));
            assert_eq!(cj, mono_eval(&coeffs, &map_to_pow(x, m)), "codeword[{j}] != mono_eval");
        }
    }

    /// `full_monomial_basis_eq(a, b)` equals `monomial_partial_eq(a)` inner-
    /// producted against `monomial_partial_eq(b)`'s Lagrange dual — here we just
    /// check the linear-time closed form matches the `2^n` expansion via
    /// `mono_eval(partial_eq(a), b)`-style consistency for the eq weight itself.
    #[test]
    fn full_monomial_eq_matches_partial() {
        let m = 4usize;
        let mut rng = StdRng::seed_from_u64(0x3);
        let a: Vec<EF> = (0..m).map(|_| rand_ef(&mut rng)).collect();
        let b: Vec<EF> = (0..m).map(|_| rand_ef(&mut rng)).collect();
        // full_monomial_basis_eq(a,b) = Σ_i monomial_partial_eq(a)[i] · lagrange?
        // Direct check: the closed form Π(a_j b_j + 1 - b_j) equals evaluating
        // the monomial partial-eq table of `a` in the Lagrange basis at `b`.
        let wa = monomial_partial_eq(&a);
        // Σ_i wa[i] · eq_lagrange(b)[i]  where eq_lagrange(b)[i] = Π (bit? b_j : 1-b_j)
        let mut lag = alloc::vec![EF::ONE; 1usize << m];
        for (i, slot) in lag.iter_mut().enumerate() {
            let mut p = EF::ONE;
            // big-endian bits to match monomial_partial_eq
            for (j, &bj) in b.iter().enumerate() {
                let bit = (i >> (m - 1 - j)) & 1;
                p *= if bit == 1 { bj } else { EF::ONE - bj };
            }
            *slot = p;
        }
        let sum: EF = wa.iter().zip(&lag).map(|(&w, &l)| w * l).sum();
        assert_eq!(full_monomial_basis_eq(&a, &b), sum);
    }
}
