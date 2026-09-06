//! WHIR folding sumcheck — phase 2's core.
//!
//! WHIR folds the committed polynomial with an eq-weighted sumcheck.  The
//! weight combines the evaluation point's eq table with the out-of-domain (OOD)
//! points' eq tables, batched by powers of a challenger-drawn coefficient; the
//! claimed sum batches the evaluation claim with the OOD answers the same way.
//! Each round sends a degree-2 univariate `g(X) = c0 + c1·X + c2·X²`, grinds a
//! proof of work, draws a folding challenge `r`, reduces the claim to `g(r)`,
//! and folds the last variable of both the polynomial and the weight at `r`.
//! This mirrors the upstream `SumcheckProver::compute_sumcheck_polynomials`.
//!
//! Reducing ALL variables leaves a single field element on each side; the
//! reduced claim then equals `weight(r) · f(r)`, which the verifier (phase 3)
//! re-derives.  The `folds_reduce_the_claim` test checks that identity on the
//! prover side.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use crate::basefold::mle::Mle;
use crate::whir::proof::{ProofOfWork, SumcheckPoly};

/// The eq table `eq(point, ·)` over the `2^n` hypercube: index `x`'s bit `j`
/// is variable `j` (LSB = variable 0), matching [`Mle::fix_last_variable`]'s
/// adjacent-pair fold, so `eq(point)[x] = Π_j (bit_j(x) ? point_j : 1-point_j)`.
pub(crate) fn eq_table<EF: Field>(n: usize, point: &[EF]) -> Vec<EF> {
    let mut v = alloc::vec![EF::ONE; 1usize << n];
    for (x, slot) in v.iter_mut().enumerate() {
        let mut prod = EF::ONE;
        for (j, &zj) in point.iter().enumerate() {
            prod *= if (x >> j) & 1 == 1 { zj } else { EF::ONE - zj };
        }
        *slot = prod;
    }
    v
}

/// The batched eq weight over the `2^n` hypercube: `eq(query, ·)` plus
/// `Σ_i batch^{i+1} · eq(ood_i, ·)`.
pub(crate) fn batched_eq_weight<EF: Field>(
    n: usize,
    query_point: &[EF],
    ood_points: &[Vec<EF>],
    batch: EF,
) -> Vec<EF> {
    let mut weight = eq_table(n, query_point);
    let mut coeff = batch;
    for ood in ood_points {
        let e = eq_table(n, ood);
        for (w, ei) in weight.iter_mut().zip(&e) {
            *w += coeff * *ei;
        }
        coeff *= batch;
    }
    weight
}

/// A stateful WHIR folder: holds the (lifted) polynomial, the batched eq
/// weight, and the running claim, and folds a chosen number of variables at a
/// time.  The multi-round prover folds `folding_factor` variables, re-encodes,
/// then folds again; a single call folding all variables is the flat sumcheck.
pub struct WhirFolder<EF> {
    /// The polynomial, lifted to EF, over the remaining variables.
    pub f_vec: Vec<EF>,
    /// The batched eq weight over the remaining variables.
    pub weight: Vec<EF>,
    /// The running (batched) claim.
    pub claimed_sum: EF,
}

impl<EF: Field> WhirFolder<EF> {
    /// Fold a fresh round's OOD constraints into the running weight and claim.
    ///
    /// After a re-commit, WHIR draws `ood_points` on the FOLDED polynomial and
    /// answers them; those constraints join the sumcheck via a fresh batching
    /// coefficient.  Because `answer_i = Σ_x eq(ood_i)[x]·f[x]`, adding
    /// `batch^{i+1}·eq(ood_i)` to the weight and `batch^{i+1}·answer_i` to the
    /// claim preserves the invariant `claim = Σ_x weight[x]·f[x]`.
    pub fn add_ood_constraints(&mut self, ood_points: &[Vec<EF>], ood_answers: &[EF], batch: EF) {
        let coeffs = self.ood_coeffs(ood_answers, batch);
        self.absorb_eq_tables(ood_points, &coeffs);
    }

    /// The transcript half of [`Self::add_ood_constraints`]: fold the OOD
    /// ANSWERS into the claimed sum and return the per-constraint batching
    /// coefficients (`batch, batch^2, ..`).  Split out so a device backend
    /// can take over the eq-table absorption.
    pub fn ood_coeffs(&mut self, ood_answers: &[EF], batch: EF) -> Vec<EF> {
        let mut coeffs = Vec::with_capacity(ood_answers.len());
        let mut coeff = batch;
        for &ans in ood_answers {
            coeffs.push(coeff);
            self.claimed_sum += coeff * ans;
            coeff *= batch;
        }
        coeffs
    }

    /// The weight half of [`Self::add_ood_constraints`]: absorb the batched
    /// eq tables into the weight.
    pub fn absorb_eq_tables(&mut self, points: &[Vec<EF>], coeffs: &[EF]) {
        let n = self.f_vec.len().trailing_zeros() as usize;
        debug_assert_eq!(1usize << n, self.f_vec.len());
        for (pt, &coeff) in points.iter().zip(coeffs) {
            let e = eq_table(n, pt);
            for (w, ei) in self.weight.iter_mut().zip(&e) {
                *w += coeff * *ei;
            }
        }
    }

    /// Fold `k` variables, appending one degree-2 message per variable.  Draws
    /// a challenge per variable and reduces the claim through each.
    pub fn fold_variables<F, Challenger>(
        &mut self,
        k: usize,
        pow_bits: &[usize],
        challenger: &mut Challenger,
        randomness_out: &mut Vec<EF>,
    ) -> Vec<(SumcheckPoly<EF>, ProofOfWork<F>)>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let mut polys = Vec::with_capacity(k);
        for round in 0..k {
            let half = self.f_vec.len() / 2;
            let mut c0 = EF::ZERO;
            let mut c2 = EF::ZERO;
            for i in 0..half {
                let (flo, fhi) = (self.f_vec[2 * i], self.f_vec[2 * i + 1]);
                let (wlo, whi) = (self.weight[2 * i], self.weight[2 * i + 1]);
                c0 += wlo * flo;
                c2 += (fhi - flo) * (whi - wlo);
            }
            let c1 = self.claimed_sum - c0.double() - c2;
            let g = SumcheckPoly(alloc::vec![c0, c1, c2]);

            challenger.observe_algebra_element(c0);
            challenger.observe_algebra_element(c1);
            challenger.observe_algebra_element(c2);
            let pow = challenger.grind(pow_bits.get(round).copied().unwrap_or(0));
            let r: EF = challenger.sample_algebra_element();

            self.claimed_sum = c0 + c1 * r + c2 * r * r;
            for i in 0..half {
                self.f_vec[i] = self.f_vec[2 * i] + r * (self.f_vec[2 * i + 1] - self.f_vec[2 * i]);
                self.weight[i] =
                    self.weight[2 * i] + r * (self.weight[2 * i + 1] - self.weight[2 * i]);
            }
            self.f_vec.truncate(half);
            self.weight.truncate(half);

            polys.push((g, ProofOfWork(pow)));
            randomness_out.push(r);
        }
        polys
    }
}

/// The output of the folding sumcheck.
pub struct WhirFold<F, EF> {
    /// Per-round `(g, pow)`: the degree-2 message and its grinding witness.
    pub round_polys: Vec<(SumcheckPoly<EF>, ProofOfWork<F>)>,
    /// Folding challenges, in sample order (variable 0 first).
    pub folding_randomness: Vec<EF>,
    /// The claim reduced through every round.
    pub final_claim: EF,
    /// The polynomial folded to its single value = `f(folding_randomness)`.
    pub folded_f: EF,
    /// The weight folded to its single value = `weight(folding_randomness)`.
    pub folded_weight: EF,
}

/// Run the WHIR folding sumcheck to completion (all `n` variables), proving
/// `Σ_x weight(x)·f(x) = claim + Σ_i batch^{i+1}·ood_answer_i`.
///
/// `claim` is `f(query_point)`; `ood_answers[i]` is `f(ood_points[i])`.
pub fn prove_fold<F, EF, Challenger>(
    f: &Mle<F>,
    query_point: &[EF],
    ood_points: &[Vec<EF>],
    ood_answers: &[EF],
    claim: EF,
    challenger: &mut Challenger,
) -> WhirFold<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let n = f.num_variables() as usize;
    debug_assert_eq!(query_point.len(), n);

    // Batch the claim and the OOD answers by powers of a drawn coefficient.
    let batch: EF = challenger.sample_algebra_element();
    let mut claimed_sum = claim;
    let mut coeff = batch;
    for &a in ood_answers {
        claimed_sum += coeff * a;
        coeff *= batch;
    }

    // Lift f and the batched weight to EF, then fold every variable.
    let mut folder = WhirFolder {
        f_vec: f.guts().as_slice().iter().map(|&v| EF::from(v)).collect(),
        weight: batched_eq_weight(n, query_point, ood_points, batch),
        claimed_sum,
    };
    let mut folding_randomness = Vec::with_capacity(n);
    let round_polys = folder.fold_variables::<F, _>(n, &[], challenger, &mut folding_randomness);

    WhirFold {
        round_polys,
        folding_randomness,
        final_claim: folder.claimed_sum,
        folded_f: folder.f_vec[0],
        folded_weight: folder.weight[0],
    }
}
