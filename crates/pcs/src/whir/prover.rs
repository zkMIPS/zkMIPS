//! WHIR prover — phase 1: the OOD commitment.
//!
//! `commit_with_ood` is WHIR's commit: RS-encode the MLE and Merkle-commit it
//! exactly as BaseFold does (reusing [`DftEncoder`] + the `Mmcs`), then draw
//! `starting_ood_samples` out-of-domain points from the transcript and evaluate
//! the committed polynomial at them.  The (points, answers) go into the
//! transcript; the folding sumcheck later constrains them.  Phases 2/3 add the
//! folding prover and verifier — see `mod.rs`.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};

use crate::basefold::config::FriConfig;
use crate::basefold::encoder::DftEncoder;
use crate::basefold::mle::Mle;
use crate::whir::config::WhirConfig;
use crate::whir::proof::ParsedCommitment;

/// A WHIR prover.  Holds the same primitives BaseFold's does (an RS encoder
/// over a DFT, and a Merkle `Mmcs`) plus the WHIR schedule.
pub struct WhirProver<F: p3_field::Field, EF, MT: Mmcs<F>, D> {
    pub encoder: DftEncoder<F, D>,
    pub mmcs: MT,
    pub config: WhirConfig,
    _ef: core::marker::PhantomData<EF>,
}

/// What `commit_with_ood` retains prover-side to continue the protocol: the
/// committed Merkle data and the codeword, alongside the parsed commitment.
pub struct WhirCommitment<F: p3_field::Field, EF, MT: Mmcs<F>> {
    pub parsed: ParsedCommitment<F, EF, MT::Commitment>,
    pub prover_data: MT::ProverData<p3_matrix::dense::RowMajorMatrix<F>>,
}

impl<F, EF, MT, D> WhirProver<F, EF, MT, D>
where
    F: TwoAdicField + p3_field::PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone>,
    D: TwoAdicSubgroupDft<F>,
{
    pub fn new(dft: Arc<D>, mmcs: MT, config: WhirConfig) -> Self {
        // The starting RS rate is the WHIR schedule's; the query count / pow at
        // commit time are unused (folding sets them per round), so a nominal
        // FriConfig carries only the blowup the encoder needs.
        let fri = FriConfig::new(config.starting_log_inv_rate, 0, 0);
        Self { encoder: DftEncoder::new(fri, dft), mmcs, config, _ef: core::marker::PhantomData }
    }

    /// WHIR commit: encode + Merkle-commit the MLE, then draw and answer the
    /// starting OOD samples.  The challenger observes the root before the OOD
    /// points are sampled and the answers before anything downstream, matching
    /// the upstream `parse_commitment_data` transcript order.
    pub fn commit_with_ood<Challenger>(
        &self,
        challenger: &mut Challenger,
        mle: Arc<Mle<F>>,
    ) -> WhirCommitment<F, EF, MT>
    where
        Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
    {
        let num_variables = mle.num_variables() as usize;

        // Encode + Merkle-commit, exactly BaseFold's commit path.
        let codewords = self.encoder.encode_batch(alloc::vec![Arc::clone(&mle)]);
        let mats: Vec<p3_matrix::dense::RowMajorMatrix<F>> =
            codewords.iter().map(|c| c.data.clone()).collect();
        let (commitment, prover_data) = self.mmcs.commit(mats);
        challenger.observe(commitment.clone());

        // Draw `starting_ood_samples` OOD points and answer them.  Each point
        // is a full `num_variables`-coordinate evaluation point in EF, drawn
        // from the transcript so the verifier redraws the same points.
        let mut ood_points: Vec<Vec<EF>> = Vec::with_capacity(self.config.starting_ood_samples);
        let mut ood_answers: Vec<EF> = Vec::with_capacity(self.config.starting_ood_samples);
        for _ in 0..self.config.starting_ood_samples {
            let point: Vec<EF> =
                (0..num_variables).map(|_| challenger.sample_algebra_element()).collect();
            // The committed polynomial evaluated at the OOD point.
            let answer = mle.eval_at::<EF>(&point)[0];
            challenger.observe_algebra_element(answer);
            ood_points.push(point);
            ood_answers.push(answer);
        }

        WhirCommitment {
            parsed: ParsedCommitment {
                commitment: alloc::vec![commitment],
                ood_points,
                ood_answers,
                _f: core::marker::PhantomData,
            },
            prover_data,
        }
    }
}
