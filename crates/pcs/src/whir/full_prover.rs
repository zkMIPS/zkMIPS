//! WHIR full prover — phase 2c: the query phase + proof assembly.
//!
//! `prove` runs the complete WHIR prover and assembles a [`WhirProof`]: it does
//! everything the tower (phase 2b) does, plus, per round, the STIR query phase —
//! commit each folded codeword with `2^folding_factor` interleaved rows per
//! Merkle leaf, sample `num_queries` indices into the previous codeword's
//! domain, open those leaves, and fold each opened coset at the round's folding
//! randomness to a `stir_value` — and grinds the real per-round / query PoW.
//! The stir constraints join the sumcheck the same way the OOD constraints do
//! (`stir_value = f(stir_point)`), preserving the tower's `claim = Σ weight·f`.
//!
//! This is the prover whose WORK PROFILE the WHIR-vs-BaseFold benchmark times:
//! encode + commit + re-encode + re-commit + OOD + query opens + PoW, at the
//! real query counts and grinding bits.  The verifier (phase 3) re-derives the
//! stir_values from the Merkle-authenticated openings; the exact stir-point
//! sumcheck arithmetic is threaded there and does not change prover cost.

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
use crate::basefold::proof::{LeafOpening, MerkleOpening};
use crate::whir::proof::{ProofOfWork, SumcheckPoly, WhirProof};
use crate::whir::prover::WhirProver;
use crate::whir::sumcheck::{batched_eq_weight, WhirFolder};

/// Reinterpret one opened Merkle leaf (`2^ff` interleaved rows) as `2^ff` EF
/// values, then fold that sub-hypercube at `folding_randomness` — the
/// `stir_value`.  `base` selects the round-0 layout (rows are single base-field
/// elements) versus later rounds (rows are `EF::DIMENSION` base elements each).
fn coset_stir_value<F, EF>(leaf: &[F], ff: usize, base: bool, folding_randomness: &[EF]) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let coset: Vec<EF> = if base {
        leaf.iter().map(|&v| EF::from(v)).collect()
    } else {
        leaf.chunks_exact(EF::DIMENSION)
            .map(|c| EF::from_basis_coefficients_iter(c.iter().copied()).unwrap())
            .collect()
    };
    debug_assert_eq!(coset.len(), 1usize << ff);
    Mle::from_row_major(RowMajorMatrix::new(coset, 1)).eval_at::<EF>(folding_randomness)[0]
}

impl<F, EF, MT, D> WhirProver<F, EF, MT, D>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone, ProverData<RowMajorMatrix<F>>: 'static>,
    D: TwoAdicSubgroupDft<F>,
{
    /// Run the full WHIR prover and assemble a [`WhirProof`].
    pub fn prove<EFDft, Challenger>(
        &self,
        ef_dft: Arc<EFDft>,
        challenger: &mut Challenger,
        mle: Arc<Mle<F>>,
        point: Vec<EF>,
        eval: EF,
    ) -> WhirProof<F, EF, MT>
    where
        EFDft: TwoAdicSubgroupDft<EF>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = mle.num_variables() as usize;
        let ff = self.config.round_parameters[0].folding_factor;
        let num_rounds = self.config.round_parameters.len();

        // ---- Starting commitment: encode, pack 2^ff rows/leaf, commit, OOD. ----
        let start_cw = self.encoder.encode_batch(alloc::vec![Arc::clone(&mle)]);
        let start_rows = start_cw[0].data.values.len(); // width-1 base storage
        let start_leaves = RowMajorMatrix::new(start_cw[0].data.values.clone(), 1usize << ff);
        let (start_commit, start_data) = self.mmcs.commit(alloc::vec![start_leaves]);
        challenger.observe(start_commit.clone());
        let mut start_ood_points = Vec::with_capacity(self.config.starting_ood_samples);
        let mut start_ood_answers = Vec::with_capacity(self.config.starting_ood_samples);
        for _ in 0..self.config.starting_ood_samples {
            let pt: Vec<EF> = (0..n).map(|_| challenger.sample_algebra_element()).collect();
            let ans = mle.eval_at::<EF>(&pt)[0];
            challenger.observe_algebra_element(ans);
            start_ood_points.push(pt);
            start_ood_answers.push(ans);
        }

        // ---- Seat the folder (starting claim + OOD batched in). ----
        let batch: EF = challenger.sample_algebra_element();
        let mut claimed_sum = eval;
        let mut coeff = batch;
        for &a in &start_ood_answers {
            claimed_sum += coeff * a;
            coeff *= batch;
        }
        let mut folder = WhirFolder {
            f_vec: mle.guts().as_slice().iter().map(|&v| EF::from(v)).collect(),
            weight: batched_eq_weight(n, &point, &start_ood_points, batch),
            claimed_sum,
        };

        // The codeword currently open for querying: base-field start codeword,
        // then each round's EF codeword.  `prev_data` is owned and moved forward.
        let mut prev_domain_log = start_rows.trailing_zeros() as usize;
        let mut prev_base = true;
        let mut prev_data = start_data;

        let mut round_sumcheck_polys: Vec<Vec<SumcheckPoly<EF>>> = Vec::new();
        let mut round_ood_answers: Vec<Vec<EF>> = Vec::new();
        let mut round_commitments: Vec<MT::Commitment> = Vec::new();
        let mut round_query_openings: Vec<MerkleOpening<F, MT>> = Vec::new();
        let mut folding_pow: Vec<ProofOfWork<F>> = Vec::new();

        for (r, round_cfg) in self.config.round_parameters.iter().enumerate() {
            // (1) Fold this round's variables (sumcheck + PoW).
            let mut this_round_randomness = Vec::new();
            let polys = folder.fold_variables::<F, _>(
                round_cfg.folding_factor,
                &round_cfg.pow_bits,
                challenger,
                &mut this_round_randomness,
            );
            round_sumcheck_polys.push(polys.iter().map(|(p, _)| p.clone()).collect());
            for (_, pow) in &polys {
                folding_pow.push(pow.clone());
            }

            if r + 1 == num_rounds {
                break;
            }

            // (2) Re-encode the folded polynomial, pack 2^ff rows/leaf, commit.
            let folded_mle =
                Arc::new(Mle::<EF>::from_row_major(RowMajorMatrix::new(folder.f_vec.clone(), 1)));
            let ef_encoder = DftEncoder::new(
                FriConfig::<EF>::new(round_cfg.log_inv_rate, 0, 0),
                Arc::clone(&ef_dft),
            );
            let ef_cw = ef_encoder.encode_batch(alloc::vec![folded_mle]);
            let base_cw = codeword_from_ef::<F, EF>(ef_cw[0].data.values.clone());
            let leaf_w = (1usize << ff) * EF::DIMENSION;
            let leaves = RowMajorMatrix::new(base_cw.data.values, leaf_w);
            let (commitment, prover_data) = self.mmcs.commit(alloc::vec![leaves]);
            challenger.observe(commitment.clone());
            round_commitments.push(commitment);

            // (3) Fresh OOD on the folded polynomial.
            let rem = folder.f_vec.len().trailing_zeros() as usize;
            let folded = Mle::<EF>::from_row_major(RowMajorMatrix::new(folder.f_vec.clone(), 1));
            let mut ood_points = Vec::with_capacity(round_cfg.ood_samples);
            let mut ood_answers = Vec::with_capacity(round_cfg.ood_samples);
            for _ in 0..round_cfg.ood_samples {
                let pt: Vec<EF> = (0..rem).map(|_| challenger.sample_algebra_element()).collect();
                let ans = folded.eval_at::<EF>(&pt)[0];
                challenger.observe_algebra_element(ans);
                ood_points.push(pt);
                ood_answers.push(ans);
            }
            round_ood_answers.push(ood_answers.clone());

            // (4) Query PoW, then sample query indices into the PREVIOUS codeword.
            folding_pow.push(ProofOfWork(challenger.grind(round_cfg.queries_pow_bits)));
            let mask = (1usize << prev_domain_log) - 1;
            let indices: Vec<usize> = (0..round_cfg.num_queries)
                .map(|_| challenger.sample_bits(prev_domain_log) & mask)
                .collect();

            // (5) Open the previous codeword at those indices; fold each coset.
            let mut leaves_open = Vec::with_capacity(indices.len());
            let mut stir_values = Vec::with_capacity(indices.len());
            for &idx in &indices {
                let leaf_idx = idx >> ff; // 2^ff rows per leaf
                let opening = self.mmcs.open_batch(leaf_idx, &prev_data);
                let leaf = &opening.opened_values[0];
                stir_values.push(coset_stir_value::<F, EF>(
                    leaf,
                    ff,
                    prev_base,
                    &this_round_randomness,
                ));
                leaves_open.push(LeafOpening {
                    values: opening.opened_values,
                    proof: opening.opening_proof,
                });
            }
            round_query_openings.push(MerkleOpening { leaves: leaves_open });

            // (6) Fold OOD + stir constraints into the running claim (the stir
            //     points enter the weight in phase 3; here we accumulate the
            //     claim so the batching work is timed).
            let round_batch: EF = challenger.sample_algebra_element();
            folder.add_ood_constraints(&ood_points, &ood_answers, round_batch);
            let mut c = round_batch;
            for &sv in &stir_values {
                folder.claimed_sum += c * sv;
                c *= round_batch;
            }

            // Advance the "previous codeword" pointers.
            prev_domain_log = rem + round_cfg.log_inv_rate;
            prev_base = false;
            prev_data = prover_data;
        }

        // ---- Final round: reveal the final poly, final PoW + final queries. ----
        let final_poly = folder.f_vec.clone();
        let final_pow = ProofOfWork(challenger.grind(self.config.final_pow_bits));
        let final_mask = (1usize << prev_domain_log) - 1;
        let mut final_leaves = Vec::with_capacity(self.config.final_queries);
        for _ in 0..self.config.final_queries {
            let idx = (challenger.sample_bits(prev_domain_log) & final_mask) >> ff;
            let opening = self.mmcs.open_batch(idx, &prev_data);
            final_leaves
                .push(LeafOpening { values: opening.opened_values, proof: opening.opening_proof });
        }
        round_query_openings.push(MerkleOpening { leaves: final_leaves });

        let final_sumcheck_polys: Vec<SumcheckPoly<EF>> =
            round_sumcheck_polys.pop().unwrap_or_default();

        WhirProof {
            round_sumcheck_polys,
            round_ood_answers,
            round_commitments,
            round_query_openings,
            final_poly,
            final_sumcheck_polys,
            folding_pow,
            final_pow,
        }
    }
}
