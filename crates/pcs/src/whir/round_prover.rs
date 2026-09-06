//! WHIR round orchestration — phase 2b: the folding tower.
//!
//! Phase 1 committed the polynomial and drew starting OOD samples; phase 2a
//! built the eq-weighted folding sumcheck engine ([`WhirFolder`]).  This phase
//! chains them into WHIR's multi-round structure:
//!
//!   starting commit (phase 1)
//!   for each round:
//!     · fold `folding_factor` variables      (phase 2a sumcheck)
//!     · re-encode the folded polynomial at the round's rate, Merkle-commit it
//!     · draw fresh OOD on the folded polynomial and answer it
//!     · fold those OOD constraints into the running claim/weight
//!   reveal the final small polynomial in the clear
//!
//! The claim threads through every fold and every re-batch by the invariant
//! `claim = Σ_x weight[x]·f[x]`, so the whole tower is internally consistent:
//! the [`test`] module checks that master identity end to end, plus per-round
//! OOD correctness and that the tower folds the *original* polynomial.
//!
//! What phase 2b does NOT do is the STIR query openings — sampling query indices
//! into each committed codeword, opening them, and folding the opened values in
//! as extra constraints.  Those are only meaningfully checked by the verifier
//! re-deriving them, so they are co-developed with phase 3 (see `mod.rs`).

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::basefold::config::FriConfig;
use crate::basefold::encoder::DftEncoder;
use crate::basefold::fri::codeword_from_ef;
use crate::basefold::mle::Mle;
use crate::whir::proof::{ParsedCommitment, ProofOfWork, SumcheckPoly};
use crate::whir::prover::WhirProver;
use crate::whir::sumcheck::{batched_eq_weight, WhirFolder};

/// One committed round of the tower: its parsed commitment (Merkle root + OOD)
/// alongside the prover-side Merkle data, retained for the phase-2c/3 query
/// openings.
pub struct RoundCommitment<F: p3_field::Field, EF, MT: Mmcs<F>> {
    pub parsed: ParsedCommitment<F, EF, MT::Commitment>,
    pub prover_data: MT::ProverData<RowMajorMatrix<F>>,
}

/// The output of the folding tower.
pub struct RoundedProof<F: p3_field::Field, EF, MT: Mmcs<F>> {
    /// The starting commitment + its OOD (phase 1).
    pub starting: ParsedCommitment<F, EF, MT::Commitment>,
    /// The starting codeword's Merkle data, for the first round's query opening.
    pub starting_prover_data: MT::ProverData<RowMajorMatrix<F>>,
    /// The intermediate round commitments (re-encoded folded codewords + OOD).
    pub rounds: Vec<RoundCommitment<F, EF, MT>>,
    /// Every fold message across every round, in order.
    pub round_polys: Vec<(SumcheckPoly<EF>, ProofOfWork<F>)>,
    /// The folding challenge drawn for each folded variable, in order.
    pub folding_randomness: Vec<EF>,
    /// The final small polynomial, revealed in the clear (its `2^k` evals).
    pub final_poly: Vec<EF>,
    /// The reduced batched eq weight over the final polynomial's variables.
    pub final_weight: Vec<EF>,
    /// The reduced claim after all folds — should equal `Σ_x weight[x]·f[x]`.
    pub final_claim: EF,
}

impl<F, EF, MT, D> WhirProver<F, EF, MT, D>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone>,
    D: TwoAdicSubgroupDft<F>,
{
    /// Run the folding tower over `mle`, opening the claim `f(point) = eval`.
    ///
    /// `ef_dft` is the DFT used to re-encode the folded (EF-valued) polynomial
    /// each round; the starting commit reuses `self.encoder`'s base DFT.
    pub fn prove_rounds<EFDft, Challenger>(
        &self,
        ef_dft: Arc<EFDft>,
        challenger: &mut Challenger,
        mle: Arc<Mle<F>>,
        point: Vec<EF>,
        eval: EF,
    ) -> RoundedProof<F, EF, MT>
    where
        EFDft: TwoAdicSubgroupDft<EF>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = mle.num_variables() as usize;
        debug_assert_eq!(point.len(), n);

        // Phase 1: commit + starting OOD.
        let start = self.commit_with_ood(challenger, Arc::clone(&mle));

        // Batch the starting claim with the starting OOD answers, and build the
        // batched eq weight, then seat the folder.
        let batch: EF = challenger.sample_algebra_element();
        let mut claimed_sum = eval;
        let mut coeff = batch;
        for &a in &start.parsed.ood_answers {
            claimed_sum += coeff * a;
            coeff *= batch;
        }
        let mut folder = WhirFolder {
            f_vec: mle.guts().as_slice().iter().map(|&v| EF::from(v)).collect(),
            weight: batched_eq_weight(n, &point, &start.parsed.ood_points, batch),
            claimed_sum,
        };

        let num_rounds = self.config.round_parameters.len();

        let mut round_polys: Vec<(SumcheckPoly<EF>, ProofOfWork<F>)> = Vec::new();
        let mut folding_randomness: Vec<EF> = Vec::new();
        let mut rounds: Vec<RoundCommitment<F, EF, MT>> = Vec::new();

        for (r, round_cfg) in self.config.round_parameters.iter().enumerate() {
            // (1) Fold this round's variables.
            let polys = folder.fold_variables::<F, _>(
                round_cfg.folding_factor,
                &round_cfg.pow_bits,
                challenger,
                &mut folding_randomness,
            );
            round_polys.extend(polys);

            // The final round's folded polynomial is revealed in the clear, so
            // it is neither re-committed nor OOD-constrained.
            if r + 1 == num_rounds {
                break;
            }

            // (2) Re-encode the folded polynomial at this round's rate and
            //     Merkle-commit it.  The folded polynomial is EF-valued, so it
            //     is encoded by the EF DFT then flattened to base storage and
            //     committed with the same Merkle scheme as the base codewords.
            let rem = folder.f_vec.len().trailing_zeros() as usize;
            let folded_mle =
                Arc::new(Mle::<EF>::from_row_major(RowMajorMatrix::new(folder.f_vec.clone(), 1)));
            let ef_encoder = DftEncoder::new(
                FriConfig::<EF>::new(round_cfg.log_inv_rate, 0, 0),
                Arc::clone(&ef_dft),
            );
            let ef_codewords = ef_encoder.encode_batch(alloc::vec![folded_mle]);
            let base_codeword = codeword_from_ef::<F, EF>(ef_codewords[0].data.values.clone());
            let (commitment, prover_data) = self.mmcs.commit(alloc::vec![base_codeword.data]);
            challenger.observe(commitment.clone());

            // (3) Draw fresh OOD on the folded polynomial (over `rem` vars).
            let folded = Mle::<EF>::from_row_major(RowMajorMatrix::new(folder.f_vec.clone(), 1));
            let mut ood_points: Vec<Vec<EF>> = Vec::with_capacity(round_cfg.ood_samples);
            let mut ood_answers: Vec<EF> = Vec::with_capacity(round_cfg.ood_samples);
            for _ in 0..round_cfg.ood_samples {
                let pt: Vec<EF> = (0..rem).map(|_| challenger.sample_algebra_element()).collect();
                let ans = folded.eval_at::<EF>(&pt)[0];
                challenger.observe_algebra_element(ans);
                ood_points.push(pt);
                ood_answers.push(ans);
            }

            // (4) Fold the OOD constraints into the running claim/weight.
            let round_batch: EF = challenger.sample_algebra_element();
            folder.add_ood_constraints(&ood_points, &ood_answers, round_batch);

            rounds.push(RoundCommitment {
                parsed: ParsedCommitment {
                    commitment: alloc::vec![commitment],
                    ood_points,
                    ood_answers,
                    _f: core::marker::PhantomData,
                },
                prover_data,
            });
        }

        RoundedProof {
            starting: start.parsed,
            starting_prover_data: start.prover_data,
            rounds,
            round_polys,
            folding_randomness,
            final_poly: folder.f_vec,
            final_weight: folder.weight,
            final_claim: folder.claimed_sum,
        }
    }
}
