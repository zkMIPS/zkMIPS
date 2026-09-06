//! BaseFold prover.
//!
//! Two entry points:
//!   * [`BasefoldProver::commit_mles`] — per-round commitment.  For
//!     each MLE in the batch, RS-encode and commit the codeword to
//!     its own Merkle tree.  Caller observes the digest before
//!     starting the next round.
//!   * [`BasefoldProver::prove_trusted_mle_evaluations`] — produces
//!     the full `BasefoldProof` after all rounds have committed.
//!
//! The "trusted" flavor assumes the verifier already has the
//! evaluation claims observed; the "untrusted" flavor observes them
//! transcript-side first.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use super::code::RsCodeWord;
use super::config::{FriConfig, BATCH_GRINDING_BITS};
use super::encoder::DftEncoder;
use super::fri::{codeword_from_ef, commit_round_leaves, final_poly, fold_codeword_once};
use super::mle::Mle;
use super::proof::{BasefoldProof, LeafOpening, MerkleOpening};

/// An accelerated proof-of-work search, registered for ONE concrete challenger
/// type.
///
/// The accelerator this exists for is a CUDA kernel, and it lives in a crate
/// that DEPENDS on this one.  It cannot be attached through the type system:
/// both the trait it would implement and the challenger it would implement it
/// for would be foreign to that crate, which the orphan rule forbids.  So the
/// accelerator registers itself once, at prover construction, and
/// [`deterministic_grind`] looks up its own `C`.  The lookup is `downcast_ref`,
/// i.e. type-exact — a registration made for a different challenger is simply
/// not found and the host search below runs unchanged.
///
/// The contract an accelerator MUST honour is `deterministic_grind`'s own: it
/// returns the SMALLEST witness index that passes `check_witness`, and it
/// leaves the challenger advanced exactly as one `check_witness(bits, witness)`
/// call would.  Any other witness is still a VALID proof, but a different one —
/// the witness is part of the proof byte stream.
struct GrindAccelerator<C: GrindingChallenger>(fn(&mut C, usize) -> C::Witness);

/// One entry per challenger type, so registering a second challenger cannot
/// silently evict the first.  Written once per challenger at prover
/// construction and read (never written) from the proving threads, so the
/// lock is uncontended; the guard is dropped BEFORE the accelerator runs,
/// because the accelerator is a seconds-long device call.
static GRIND_ACCELERATORS: std::sync::RwLock<
    alloc::vec::Vec<alloc::boxed::Box<dyn core::any::Any + Send + Sync>>,
> = std::sync::RwLock::new(alloc::vec::Vec::new());

/// Register `accelerator` as the proof-of-work search for challenger `C`.
///
/// Idempotent per challenger and first-writer-wins: re-registering `C` is a
/// no-op, so a process cannot switch grinds half way through a proof.  See
/// [`GrindAccelerator`] for the contract an accelerator must honour.
pub fn register_grind_accelerator<C>(accelerator: fn(&mut C, usize) -> C::Witness)
where
    C: GrindingChallenger + 'static,
{
    let mut registered = GRIND_ACCELERATORS.write().expect("grind accelerator registry poisoned");
    if registered.iter().any(|a| a.is::<GrindAccelerator<C>>()) {
        return;
    }
    registered.push(alloc::boxed::Box::new(GrindAccelerator::<C>(accelerator)));
}

/// The registered accelerator for `C`, if one was registered for exactly `C`.
fn grind_accelerator_for<C>() -> Option<fn(&mut C, usize) -> C::Witness>
where
    C: GrindingChallenger + 'static,
{
    GRIND_ACCELERATORS
        .read()
        .expect("grind accelerator registry poisoned")
        .iter()
        .find_map(|a| a.downcast_ref::<GrindAccelerator<C>>())
        .map(|a| a.0)
}

/// Deterministic counterpart to `<C as GrindingChallenger>::grind(bits)`.
///
/// plonky3's [`p3_challenger::DuplexChallenger::grind`] is
/// implemented with `(0..num_batches).into_par_iter().find_map_any(...)`,
/// which returns the witness from whichever rayon worker finishes first.
/// On multi-GPU runs the result is therefore non-deterministic across
/// re-invocations of the same shard, even though the proof is honest.
///
/// That non-determinism cascades into the recursion compress program:
/// the basefold proof's `pow_witness` field is included in the
/// `evaluation_proof_bytes: Vec<u8>` (msgpack-encoded) carried by the
/// next layer's witness.  A different `pow_witness` value yields a
/// different msgpack byte length (msgpack uses variable-length integer
/// encoding), which shifts the witness-read instruction count, which
/// can flip the `RecursionShapeConfig::fix_shape` selection, which
/// changes the compress program's preprocessed traces, which changes
/// `compressed_proof.vk.hash_koalabear()`.  Concretely measured: 3
/// distinct vk hashes across runs v8/v9/v10 of `bench_8x5090.sh
/// tendermint compress`.
///
/// This helper provides a deterministic substitute that returns the
/// *smallest-index* canonical-u64 witness satisfying the PoW
/// condition.  The output is reproducible across runs, machines, and
/// thread-pool sizes, eliminating the cascade.
///
/// Implementation: parallel rayon `find_first` (NOT `find_any`).  A
/// sequential walk was tried first — at ~65 ms per grind for 16-bit
/// PoW it back-pressured the per-shard host-pool worker enough that
/// shards backed up and the host RAM grew unbounded (Linux OOM-killer
/// terminated the perf binary at ~850 GB anon-rss).  `find_first`
/// keeps the CPU work parallel so per-shard wall stays close to
/// plonky3's parallel grind, while its left-of-match cancellation
/// rule guarantees the smallest-index witness wins regardless of
/// thread scheduling.
///
/// Validated: 3 back-to-back tendermint compress runs (v1/v2b/v3 of
/// the May 6 fix session) produce IDENTICAL `compressed_proof.vk
/// .hash_koalabear()`; baseline (without fix) produced 3 distinct
/// hashes (v8/v9/v10).
pub(crate) fn deterministic_grind<F, C>(challenger: &mut C, bits: usize) -> F
where
    F: p3_field::PrimeField64 + p3_field::integers::QuotientMap<u64> + Send + Sync,
    C: GrindingChallenger<Witness = F> + 'static,
{
    use p3_maybe_rayon::prelude::*;
    if bits == 0 {
        return F::ZERO;
    }
    // An accelerator registered for THIS challenger type replaces the search
    // below.  See `register_grind_accelerator` for the contract and for why the
    // dispatch is a registration rather than a trait impl.
    if let Some(accelerated) = grind_accelerator_for::<C>() {
        return accelerated(challenger, bits);
    }
    let order = F::ORDER_U64;
    // Parallel search with `find_first` semantics — returns the
    // smallest-index witness, NOT the first-thread-wins witness as
    // plonky3's `find_any` does.  See the docstring for why parallel
    // is required (sequential OOM'd the host).
    let witness = (0..order)
        .into_par_iter()
        .map(|i| {
            // SAFETY: i < F::ORDER_U64 by iterator bound, so this is
            // a valid canonical field element.
            unsafe { <F as p3_field::integers::QuotientMap<u64>>::from_canonical_unchecked(i) }
        })
        .find_first(|&w| {
            let mut probe = challenger.clone();
            probe.check_witness(bits, w)
        })
        .expect("deterministic_grind: failed to find a PoW witness");
    // Replay on the real challenger to commit its state update
    // (observe(witness) + sample_bits(bits)).  Mirrors plonky3's
    // post-find `assert!(check_fn(self, witness))`.
    let ok = challenger.check_witness(bits, witness);
    debug_assert!(ok);
    let _ = ok;
    witness
}

/// Prover-side state for one committed round.
///
/// Holds the mmcs ProverData (needed to open at query indices later)
/// plus the encoded codewords (one per Mle in this round's batch).
pub struct BasefoldProverData<F: Field, MT: Mmcs<F>> {
    pub prover_data: MT::ProverData<RowMajorMatrix<F>>,
    pub encoded_codewords: Vec<Arc<RsCodeWord<F>>>,
    /// The tree's digest layers, first above the leaves to the root, kept
    /// beside it for a commit whose `prover_data` is LEAFLESS.
    ///
    /// A commit that never materialised its codewords on host cannot be opened
    /// through the MMCS — `open_batch` reads leaf rows — so its opener supplies
    /// the leaf rows itself and walks these layers for the sibling paths,
    /// `digest_layers[l][idx ^ 1]` per level, exactly as the MMCS would.  The
    /// layers are a small fraction of the codewords they describe, which is why
    /// keeping them is worth what dropping the codewords saves.
    ///
    /// EMPTY whenever `prover_data` holds its leaves, which is every commit the
    /// host prover builds — it opens through the MMCS as before.
    pub digest_layers: Vec<Vec<[F; 8]>>,
}

pub struct BasefoldProver<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>, D> {
    pub encoder: DftEncoder<F, D>,
    pub mmcs: MT,
    pub num_expected_commitments: usize,
    _ef: core::marker::PhantomData<EF>,
}

impl<F, EF, MT, D> BasefoldProver<F, EF, MT, D>
where
    F: TwoAdicField + p3_field::PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone>,
    D: TwoAdicSubgroupDft<F>,
{
    pub fn new(
        fri_config: FriConfig<F>,
        dft: Arc<D>,
        mmcs: MT,
        num_expected_commitments: usize,
    ) -> Self {
        let encoder = DftEncoder::new(fri_config, dft);
        Self { encoder, mmcs, num_expected_commitments, _ef: core::marker::PhantomData }
    }

    pub fn config(&self) -> &FriConfig<F> {
        self.encoder.config()
    }

    /// Commit a batch of MLEs (one round of the protocol).
    ///
    /// Each MLE gets its own RS codeword, and all codewords for this
    /// round are committed under a single Merkle digest by stacking
    /// their (already-bit-reversed) values column-wise.
    pub fn commit_mles(&self, mles: Vec<Arc<Mle<F>>>) -> (MT::Commitment, BasefoldProverData<F, MT>)
    where
        F: Send + Sync,
        D: Send + Sync,
    {
        let codewords = self.encoder.encode_batch(mles);
        self.commit_codewords(codewords)
    }

    /// Commit a batch of *already-encoded* RS codewords for one round
    /// of the protocol.
    ///
    /// Mirrors [`Self::commit_mles`] but skips the host
    /// [`DftEncoder::encode_batch`] step — used by the GPU dispatch
    /// path where codewords are produced on
    /// device by `FriCudaProver::encode_and_commit` and pulled back to
    /// host before this step.
    ///
    /// The returned `BasefoldProverData` is byte-equivalent to what
    /// `commit_mles` returns when the codewords are byte-identical to
    /// the ones host encode would have produced (validated by
    /// `ziren-gpu/basefold/tests/cpu_vs_gpu_commit.rs`).
    pub fn commit_codewords(
        &self,
        codewords: Vec<Arc<RsCodeWord<F>>>,
    ) -> (MT::Commitment, BasefoldProverData<F, MT>)
    where
        F: Send + Sync,
        D: Send + Sync,
    {
        // For commitment: stack each codeword as one matrix in the
        // `mmcs.commit` call.  Layout matches what
        // `query_openings_at_indices` will read back.
        let mats: Vec<RowMajorMatrix<F>> = codewords.iter().map(|c| c.data.clone()).collect();

        let (commitment, prover_data) = self.mmcs.commit(mats);
        (
            commitment,
            BasefoldProverData {
                prover_data,
                encoded_codewords: codewords,
                // The host commit keeps its leaves, so its opener uses the MMCS.
                digest_layers: Vec::new(),
            },
        )
    }

    /// Build the partial-Lagrange evaluation vector at `point`.
    ///
    /// Returns length `2^point.len()` where entry `i` is the
    /// evaluation of the indicator polynomial `eq(point, i_bits)`.
    /// Used both to sample batching coefficients and to weight the
    /// evaluation claims into a single batched claim.
    fn partial_lagrange(point: &[EF]) -> Vec<EF> {
        let mut acc = vec![EF::ONE];
        for &r in point {
            let mut next = Vec::with_capacity(acc.len() * 2);
            for v in &acc {
                next.push(*v * (EF::ONE - r));
                next.push(*v * r);
            }
            acc = next;
        }
        acc
    }

    /// Random linear combination of all per-round MLEs and codewords
    /// using `batching_coefficients`.  Returns the batched MLE (in
    /// EF), the batched codeword (still stored as F packed in width
    /// `EF::DIMENSION` per row), and the batched evaluation claim.
    #[allow(clippy::type_complexity)]
    fn batch(
        &self,
        batching_coefficients: &[EF],
        mle_rounds: &[Vec<Arc<Mle<F>>>],
        codeword_rounds: &[Vec<Arc<RsCodeWord<F>>>],
        evaluation_claims_rounds: &[Vec<EF>],
    ) -> (Mle<EF>, RsCodeWord<F>, EF) {
        let num_variables = mle_rounds[0][0].num_variables() as usize;
        let hyp_size = 1usize << num_variables;
        let codeword_height = mle_rounds[0][0].hypercube_size() << self.config().log_blowup();

        // Single linear pass: walk every base-MLE and codeword and
        // accumulate into the batched buffers.
        let mut batched_mle = vec![EF::ZERO; hyp_size];
        let mut batched_codeword_ef = vec![EF::ZERO; codeword_height];
        let mut batched_eval = EF::ZERO;
        let mut coeff_idx = 0usize;

        // Parallelize the per-row inner products across both the MLE
        // and codeword loops.  For codeword_height = 2^{N + log_blowup}
        // (e.g. N=22, log_blowup=4 → 2^26 ≈ 67M sequential row mul-adds
        // per (mle, codeword) pair) the codeword loop dominates.  Each
        // row writes into a distinct accumulator slot, so par_iter_mut
        // is safe.
        use p3_maybe_rayon::prelude::*;
        for ((mles, codewords), evals) in
            mle_rounds.iter().zip(codeword_rounds.iter()).zip(evaluation_claims_rounds.iter())
        {
            let mut eval_in_round = 0usize;
            for (mle, codeword) in mles.iter().zip(codewords.iter()) {
                let n_polys = mle.num_polynomials();
                let coeffs = &batching_coefficients[coeff_idx..coeff_idx + n_polys];

                debug_assert_eq!(mle.hypercube_size(), hyp_size);
                let mle_vals = mle.guts().as_slice();
                batched_mle.par_iter_mut().enumerate().for_each(|(row, acc)| {
                    let row_start = row * n_polys;
                    let mut row_sum = EF::ZERO;
                    for k in 0..n_polys {
                        row_sum += coeffs[k] * mle_vals[row_start + k];
                    }
                    *acc += row_sum;
                });

                let cw_row_width = codeword.data.width();
                let cw_vals = &codeword.data.values;
                debug_assert_eq!(cw_row_width, n_polys);
                debug_assert_eq!(codeword.data.height(), codeword_height);
                batched_codeword_ef.par_iter_mut().enumerate().for_each(|(row, acc)| {
                    let row_start = row * cw_row_width;
                    let mut row_sum = EF::ZERO;
                    for k in 0..n_polys {
                        row_sum += coeffs[k] * cw_vals[row_start + k];
                    }
                    *acc += row_sum;
                });

                for k in 0..n_polys {
                    batched_eval += coeffs[k] * evals[eval_in_round + k];
                }
                eval_in_round += n_polys;
                coeff_idx += n_polys;
            }
        }

        // Pack the batched EF codeword back into F-storage (width = EF::DIMENSION).
        let batched_codeword_storage =
            <EF as p3_field::BasedVectorSpace<F>>::flatten_to_base(batched_codeword_ef);
        let batched_codeword =
            RsCodeWord::new(RowMajorMatrix::new(batched_codeword_storage, EF::DIMENSION));

        let batched_mle = Mle::from_row_major(RowMajorMatrix::new(batched_mle, 1));

        (batched_mle, batched_codeword, batched_eval)
    }

    /// Produce a `BasefoldProof` over the given evaluation claims
    /// for the multilinear polynomials committed in `prover_data`.
    ///
    /// `eval_point` has dimension equal to the underlying number of
    /// variables of every committed MLE (all MLEs in a single proof
    /// must share the same `num_variables`).
    #[allow(clippy::type_complexity)]
    pub fn prove_trusted_mle_evaluations<Challenger: 'static>(
        &self,
        eval_point: Vec<EF>,
        mle_rounds: Vec<Vec<Arc<Mle<F>>>>,
        evaluation_claims: Vec<Vec<EF>>,
        // BORROWED: every use below is a read (`iter()`, and
        // `mmcs.open_batch(idx, &data.prover_data)`), so the committed Merkle
        // tree never has to be owned — or deep-copied — to be opened.  That is
        // what lets a long-lived commit (the preprocessed one, built once at
        // setup and held in the proving key) be opened by every shard.
        prover_data: &[&BasefoldProverData<F, MT>],
        challenger: &mut Challenger,
    ) -> BasefoldProof<F, EF, MT>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let num_variables = eval_point.len();

        // (1) Batch grinding witness (forces verifier-prover to share
        // a transcript prefix before sampling batching coefficients).
        // Use deterministic_grind to keep the witness
        // value reproducible across runs (plonky3's parallel grind
        // uses `find_any` which is non-deterministic).
        let batch_grinding_witness = deterministic_grind(challenger, BATCH_GRINDING_BITS);

        // (2) Sample batching coefficients via partial-Lagrange basis.
        let total_polys: usize =
            mle_rounds.iter().flat_map(|r| r.iter()).map(|m| m.num_polynomials()).sum();
        let num_batching_vars = total_polys.next_power_of_two().trailing_zeros() as usize;
        let batching_point: Vec<EF> =
            (0..num_batching_vars).map(|_| challenger.sample_algebra_element()).collect();
        let batching_coefficients = Self::partial_lagrange(&batching_point);

        // (3) Build the batched MLE + codeword + claim.
        let codeword_rounds: Vec<Vec<Arc<RsCodeWord<F>>>> =
            prover_data.iter().map(|d| d.encoded_codewords.clone()).collect();
        let (mut current_mle, mut current_codeword, batched_eval) =
            self.batch(&batching_coefficients, &mle_rounds, &codeword_rounds, &evaluation_claims);

        // (4) Observe number of FRI rounds.
        challenger.observe(F::from_usize(num_variables));

        // (5) Commit phase: emit one univariate poly + one Merkle
        // commitment per round.  Last-coordinate-first folding.
        let mut univariate_messages: Vec<[EF; 2]> = Vec::with_capacity(num_variables);
        let mut fri_commitments: Vec<MT::Commitment> = Vec::with_capacity(num_variables);
        let mut commit_phase_data: Vec<<MT as Mmcs<F>>::ProverData<RowMajorMatrix<F>>> =
            Vec::with_capacity(num_variables);
        let mut current_eval = batched_eval;
        // One FRI round folds `log_folding_arity` variables: it commits ONCE,
        // with `2^k` codeword rows per Merkle leaf, then runs `k` sumcheck
        // rounds and `k` folds against that single commitment.  A query
        // therefore walks `num_variables / k` Merkle paths instead of
        // `num_variables`, and each is `k` levels shallower, which is where
        // the recursion verifier's cost lives.
        //
        // At `k == 1` the transcript is byte-identical to the classic shape:
        // univariate message, then commitment, then beta, then fold.
        let log_folding_arity = self.config().log_folding_arity();
        // How many variables each FRI round folded, so the query phase knows
        // how far to shift the index per round.
        let mut round_arities: Vec<usize> = Vec::new();
        let mut var = 0usize;
        while var < num_variables {
            // A trailing group shorter than the arity folds (and commits) at
            // its own width, so `num_variables` need not divide the arity.
            let group = core::cmp::min(log_folding_arity, num_variables - var);

            let mut codeword_ef: Option<Vec<EF>> = None;
            for j in 0..group {
                // Sumcheck round on the *first* remaining variable (matches
                // `Mle::fold`'s even/odd pairing — the same beta is used as
                // both sumcheck point and FRI fold parameter).
                //
                //   claim = (1 - r) * g(0) + r * g(1)
                //   => g(1) = (claim - g(0)) / r + g(0)
                let r = eval_point[var + j];
                let zero_val = {
                    let mut p: Vec<EF> = Vec::with_capacity(num_variables - var - j);
                    p.push(EF::ZERO);
                    p.extend_from_slice(&eval_point[var + j + 1..]);
                    current_mle.eval_at(&p)[0]
                };
                let one_val =
                    if r == EF::ZERO { EF::ZERO } else { (current_eval - zero_val) / r + zero_val };
                let uni_poly = [zero_val, one_val];
                univariate_messages.push(uni_poly);
                for &elem in &uni_poly {
                    challenger.observe_algebra_element(elem);
                }

                // The commitment for the whole group is made after the FIRST
                // univariate message and before the first beta, exactly as the
                // one-variable-per-round protocol does.
                if j == 0 {
                    let (ef, commitment, data) = commit_round_leaves::<F, EF, MT, _>(
                        current_codeword,
                        group,
                        &self.mmcs,
                        challenger,
                    );
                    fri_commitments.push(commitment);
                    commit_phase_data.push(data);
                    codeword_ef = Some(ef);
                    current_codeword = RsCodeWord::new(RowMajorMatrix::new(Vec::new(), 1));
                }

                let beta: EF = challenger.sample_algebra_element();
                codeword_ef = Some(fold_codeword_once::<F, EF>(
                    codeword_ef.take().expect("codeword committed at j == 0"),
                    beta,
                ));
                current_mle = current_mle.fold(beta);
                current_eval = uni_poly[0] + beta * uni_poly[1];
            }
            // One flatten per group, not per fold.
            current_codeword =
                codeword_from_ef::<F, EF>(codeword_ef.expect("group folds at least once"));
            round_arities.push(group);
            var += group;
        }

        // (6) Final poly + grinding witness + observe transcript.
        // Invariant (BaseFold key identity): `current_eval` (sumcheck
        // chain), `fp` (codeword K-fold), and `current_mle.guts.values[0]`
        // (MLE K-fold) are all equal in an honest proof.
        let fp = final_poly::<F, EF>(current_codeword);
        challenger.observe_algebra_element(fp);

        let pow_bits = self.config().proof_of_work_bits;
        // See `deterministic_grind` for why this call must
        // not delegate to plonky3's parallel `challenger.grind`.
        let pow_witness = deterministic_grind(challenger, pow_bits);

        // (7) Sample query indices.
        let log_codeword_size = num_variables + self.config().log_blowup();
        let num_queries = self.config().num_queries;
        let query_indices: Vec<usize> =
            (0..num_queries).map(|_| challenger.sample_bits(log_codeword_size)).collect();

        // (8) Open the original (per-round) component-poly commitments.
        // Each query index yields one Merkle path; the leaf at that
        // index is the row across every encoded codeword for the round.
        // (Tried par_iter — `MT::Proof: !Send` blocks at the trait
        // layer; same root cause as the STIR-loop revert in the
        // now-removed WHIR prototype.  Would need
        // upstream Send+Sync on `Mmcs::Proof`; deferred.)
        let mut component_polynomials_query_openings_and_proofs =
            Vec::with_capacity(prover_data.len());
        for data in prover_data.iter() {
            let mut leaves = Vec::with_capacity(num_queries);
            for &idx in &query_indices {
                let opening = self.mmcs.open_batch(idx, &data.prover_data);
                leaves.push(LeafOpening {
                    values: opening.opened_values,
                    proof: opening.opening_proof,
                });
            }
            component_polynomials_query_openings_and_proofs.push(MerkleOpening { leaves });
        }

        // (9) Open commit-phase round commitments at the (shifted) indices.
        let mut query_phase_openings_and_proofs = Vec::with_capacity(num_variables);
        // Round r's Merkle tree has one leaf per `2^arity_r` codeword rows, and
        // each earlier round already shrank the codeword by its own arity, so
        // the shift ACCUMULATES.  At arity 1 throughout this is the classic
        // `>>= 1` per round.
        let mut indices = query_indices;
        for (data, &arity) in commit_phase_data.iter().zip(round_arities.iter()) {
            for ix in indices.iter_mut() {
                *ix >>= arity;
            }
            let mut leaves = Vec::with_capacity(num_queries);
            for &idx in &indices {
                let opening = self.mmcs.open_batch(idx, data);
                leaves.push(LeafOpening {
                    values: opening.opened_values,
                    proof: opening.opening_proof,
                });
            }
            query_phase_openings_and_proofs.push(MerkleOpening { leaves });
        }

        BasefoldProof {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings_and_proofs,
            query_phase_openings_and_proofs,
            final_poly: fp,
            pow_witness,
            batch_grinding_witness,
        }
    }
}
