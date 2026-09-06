//! WHIR verifier — phase 3 (sumcheck + OOD + terminal identity).
//!
//! [`WhirVerifier::verify_rounds`] validates the folding tower ([`crate::whir::
//! round_prover`]) end to end by replaying the Fiat–Shamir transcript: it
//! observes each committed root, re-samples the OOD points and batching
//! coefficients, checks every sumcheck round message (`g(0)+g(1) == claim`,
//! reduce to `g(r)`), verifies the per-fold proof-of-work, and finally checks
//! the terminal identity
//!
//!   threaded_claim == Σ_constraints c · eq(p[..k], cfr_suffix) · final_poly(p[k..])
//!
//! which is the verifier-side reconstruction of `Σ_x weight[x]·final_poly[x]`
//! from the transcript-derived constraint points alone (no `2^n` weight table).
//!
//! What this does NOT yet check is the STIR query authentication — opening each
//! committed codeword at the sampled indices, verifying its Merkle path, and
//! folding the opened coset into a `stir_value` constraint.  That is the
//! remaining phase-3 work (it needs the interleaved-encode + monomial
//! point-map); the full prover ([`crate::whir::full_prover`]) already produces
//! those openings.

use alloc::string::String;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::basefold::mle::Mle;
use crate::whir::config::WhirConfig;
use crate::whir::round_prover::RoundedProof;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirVerifierError {
    /// A sumcheck round message failed `g(0)+g(1) == claim`.
    SumcheckMismatch { round: usize, var: usize },
    /// A per-fold proof-of-work witness did not pass `check_witness`.
    PowMismatch { round: usize, var: usize },
    /// A re-sampled OOD point disagreed with the one in the proof.
    OodPointMismatch { round: usize, sample: usize },
    /// The terminal identity did not hold.
    TerminalMismatch,
    /// The proof's shape (message counts, final-poly length) is wrong.
    IncorrectShape(String),
}

pub struct WhirVerifier<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    pub mmcs: MT,
    pub config: WhirConfig,
    _ef: core::marker::PhantomData<(F, EF)>,
}

impl<F, EF, MT> WhirVerifier<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone>,
{
    pub fn new(mmcs: MT, config: WhirConfig) -> Self {
        Self { mmcs, config, _ef: core::marker::PhantomData }
    }

    /// `eq(a, b) = Π_i (a_i·b_i + (1-a_i)(1-b_i))` over equal-length points.
    fn eq(a: &[EF], b: &[EF]) -> EF {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| x * y + (EF::ONE - x) * (EF::ONE - y)).product()
    }

    /// Verify the tower proof against the evaluation claim `f(point) = eval`.
    pub fn verify_rounds<Challenger>(
        &self,
        challenger: &mut Challenger,
        point: &[EF],
        eval: EF,
        proof: &RoundedProof<F, EF, MT>,
    ) -> Result<(), WhirVerifierError>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = point.len();
        let num_rounds = self.config.round_parameters.len();
        let folds: alloc::vec::Vec<usize> =
            self.config.round_parameters.iter().map(|rp| rp.folding_factor).collect();
        let final_log = n - folds.iter().sum::<usize>();
        if proof.final_poly.len() != 1usize << final_log {
            return Err(WhirVerifierError::IncorrectShape(alloc::format!(
                "final_poly len {} != 2^{final_log}",
                proof.final_poly.len()
            )));
        }

        // ---- Replay the starting commit + OOD (commit_with_ood order). ----
        challenger.observe(proof.starting.commitment[0].clone());
        for (k, ans) in proof.starting.ood_answers.iter().enumerate() {
            let pt: Vec<EF> = (0..n).map(|_| challenger.sample_algebra_element()).collect();
            if pt != proof.starting.ood_points[k] {
                return Err(WhirVerifierError::OodPointMismatch { round: 0, sample: k });
            }
            challenger.observe_algebra_element(*ans);
        }
        let batch: EF = challenger.sample_algebra_element();

        // Running claim, and the constraint accumulator (point, coeff, v-vars).
        let mut claim = eval;
        let mut constraints: Vec<(Vec<EF>, EF, usize)> = alloc::vec![(point.to_vec(), EF::ONE, n)];
        let mut coeff = batch;
        for (a, p) in proof.starting.ood_answers.iter().zip(&proof.starting.ood_points) {
            claim += coeff * *a;
            constraints.push((p.clone(), coeff, n));
            coeff *= batch;
        }

        // ---- Per-round sumcheck + OOD replay. ----
        let mut flat = 0usize;
        let mut all_fr: Vec<EF> = Vec::with_capacity(n - final_log);
        let mut folded_vars = 0usize;
        for (r, round_cfg) in self.config.round_parameters.iter().enumerate() {
            for var in 0..round_cfg.folding_factor {
                let (poly, pow) = &proof.round_polys[flat];
                flat += 1;
                let c = &poly.0;
                if c.len() != 3 {
                    return Err(WhirVerifierError::IncorrectShape("degree-2 message".into()));
                }
                let (c0, c1, c2) = (c[0], c[1], c[2]);
                // g(0) + g(1) == claim.
                if c0 + (c0 + c1 + c2) != claim {
                    return Err(WhirVerifierError::SumcheckMismatch { round: r, var });
                }
                challenger.observe_algebra_element(c0);
                challenger.observe_algebra_element(c1);
                challenger.observe_algebra_element(c2);
                if !challenger
                    .check_witness(round_cfg.pow_bits.get(var).copied().unwrap_or(0), pow.0)
                {
                    return Err(WhirVerifierError::PowMismatch { round: r, var });
                }
                let rc: EF = challenger.sample_algebra_element();
                claim = c0 + c1 * rc + c2 * rc * rc;
                all_fr.push(rc);
            }
            folded_vars += round_cfg.folding_factor;

            if r + 1 == num_rounds {
                break;
            }

            // Round commitment + OOD (the tower does NOT commit the last round).
            let round = &proof.rounds[r];
            challenger.observe(round.parsed.commitment[0].clone());
            let rem = n - folded_vars;
            for (k, ans) in round.parsed.ood_answers.iter().enumerate() {
                let pt: Vec<EF> = (0..rem).map(|_| challenger.sample_algebra_element()).collect();
                if pt != round.parsed.ood_points[k] {
                    return Err(WhirVerifierError::OodPointMismatch { round: r + 1, sample: k });
                }
                challenger.observe_algebra_element(*ans);
            }
            let round_batch: EF = challenger.sample_algebra_element();
            let mut cc = round_batch;
            for (a, p) in round.parsed.ood_answers.iter().zip(&round.parsed.ood_points) {
                claim += cc * *a;
                constraints.push((p.clone(), cc, rem));
                cc *= round_batch;
            }
        }
        if flat != proof.round_polys.len() {
            return Err(WhirVerifierError::IncorrectShape("leftover sumcheck messages".into()));
        }

        // ---- Terminal identity. ----
        let final_mle =
            Mle::from_row_major(p3_matrix::dense::RowMajorMatrix::new(proof.final_poly.clone(), 1));
        let mut total = EF::ZERO;
        for (p, c, v) in &constraints {
            let k = v - final_log;
            let eq_part = Self::eq(&p[..k], &all_fr[(n - v)..(n - v) + k]);
            let f_part = final_mle.eval_at::<EF>(&p[k..])[0];
            total += *c * eq_part * f_part;
        }
        if total != claim {
            return Err(WhirVerifierError::TerminalMismatch);
        }
        Ok(())
    }
}
