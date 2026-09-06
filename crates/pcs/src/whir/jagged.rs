//! Jagged-WHIR wiring — the WHIR siblings of `jagged_pcs`'s commit / open /
//! verify entry points, sharing the jagged layer's dense-packing, stacking
//! interleave, and claim-binding conventions so the shard prover can swap the
//! inner PCS without touching the jagged reduction above it.
//!
//! Contract parity with the BaseFold path:
//!   * commit consumes the same `chip_traces` (in production a single width-1
//!     dense polynomial), runs the same `chips_to_mles_owned` +
//!     `interleave_multilinears_with_fixed_rate` stacking, and reports the
//!     same `(chip_dims, area, log_stacking_height)` metadata;
//!   * the interleaved stripes (width `DEFAULT_BATCH_SIZE`) are split into
//!     width-1 polynomials, so a round's polynomial count is exactly
//!     `area >> log_stacking_height` — the count the stacked verifier derives
//!     from `round_areas`, and the order matches BaseFold's flat
//!     `batch_evaluations` (stripe-major, then column);
//!   * verify first checks `evaluation_claim == interpolation of the echoed
//!     evaluations at the batch coordinates` (the `StackingMismatch` bind),
//!     then runs the stacked WHIR verifier on the stack coordinates.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::basefold::mle::Mle;
use crate::basefold::stacked::interleave_multilinears_with_fixed_rate;
use crate::jagged_pcs::{
    chips_to_mles_owned, pick_log_stacking_height, JaggedChallenge, JaggedCommitGeneric, JaggedVal,
    DEFAULT_BATCH_SIZE,
};
use crate::whir::config::{RoundConfig, WhirConfig};
use crate::whir::stacked::{
    StackedWhirProof, StackedWhirProver, StackedWhirProverData, StackedWhirVerifier,
};
use crate::whir::verifier::WhirVerifierError;

/// Prover-side state kept after a jagged-WHIR commit.
pub struct JaggedWhirProverDataGeneric<MT: Mmcs<JaggedVal>> {
    pub stacked_data: StackedWhirProverData<JaggedVal, MT>,
    pub chip_dims: Vec<(usize, u32)>,
    pub area: usize,
    pub log_stacking_height: u32,
}

/// The WHIR configuration for a given stacking height: fold `ff` variables a
/// round until `final_log` remain, with the upstream escalating-rate
/// schedule (round r's folded codeword commits at `1 + 3(r+1)` bits of
/// blowup — the poly shrinks `2^ff`-fold per round, so the deeper, smaller
/// codewords afford lower rates and correspondingly fewer queries).
/// Query/PoW budgets here are the TEST shape; the production budget is
/// [`core_whir_config`].
pub fn whir_config_for_stack(lsh: usize, ff: usize, final_log: usize) -> WhirConfig {
    assert!(lsh > final_log && (lsh - final_log) % ff == 0, "lsh must fold evenly");
    let num_rounds = (lsh - final_log) / ff;
    whir_config_for_fold_schedule(lsh, &alloc::vec![ff; num_rounds], final_log)
}

/// The general (per-round) fold schedule: round `r` folds `folds[r]`
/// variables; `final_log` remain for the revealed final polynomial.  Each
/// round's committed codeword packs `2^folds[r+1]` positions per Merkle leaf
/// (the NEXT round's stir fold consumes one leaf per query), so a SMALLER
/// round-0 factor shrinks round-0 query leaves — which for the stacked form
/// span EVERY stripe's coset row — without touching the round count, the
/// rate escalation, or the query budgets (all round-indexed).
pub fn whir_config_for_fold_schedule(lsh: usize, folds: &[usize], final_log: usize) -> WhirConfig {
    assert!(!folds.is_empty() && folds.iter().all(|&f| f > 0));
    assert_eq!(
        folds.iter().sum::<usize>() + final_log,
        lsh,
        "fold schedule must consume lsh exactly"
    );
    let mut config = WhirConfig::default_whir_config();
    config.starting_ood_samples = 0; // stacked WHIR: OOD rides in round constraints
    config.starting_log_inv_rate = 1;
    config.round_parameters = folds
        .iter()
        .enumerate()
        .map(|(r, &ff)| RoundConfig {
            folding_factor: ff,
            evaluation_domain_log_size: 0,
            queries_pow_bits: 0,
            pow_bits: alloc::vec![0usize; ff],
            num_queries: 4,
            ood_samples: 1,
            log_inv_rate: 1 + 3 * (r + 1),
        })
        .collect();
    config.final_poly_log_degree = final_log;
    config.final_queries = 4;
    config.final_pow_bits = 0;
    config
}

/// The PRODUCTION jagged-WHIR budget for a core-shard stack of height
/// `2^lsh`: **100 bits PROVABLE in the unique-decoding regime**, per round
/// (ethereum/soundcalc, `docs/soundness/ziren.soundcalc.toml`).  Under the
/// unique-decoding bound a query is worth `-log2((1+rho)/2)` bits — at most
/// ~1 bit at ANY rate — so every round must clear ~84 bits of queries on its
/// own plus the 16-bit query PoW:
///
///   round 0: 124 queries into the rate-2^-2 stripe trees  (124·0.678 + 16 = 100)
///   round 1:  88 queries into the rate-2^-5 codeword      ( 88·0.978 + 16 = 102)
///   final  :  85 queries into the rate-2^-8 codeword      ( 85·0.997 + 16 = 100)
///
/// The previous schedule (rate 1/2, 84/21/12 queries, folds [4,7,7]) counted
/// log-inv-rate bits per query — the capacity accounting — and was 27 bits
/// provable (final round 12·0.99 + 16), 53 under the Johnson bound.
///
/// Later rounds fold 6 (not 7) so the recursion leaf's Merkle-leaf hashing
/// (queries x opened felts) stays near the old budget: 124·40·2^3 + 88·4·2^6
/// + 85·4·2^6 ≈ 84 K felts vs 71 K before.  Wider queries at rate 1/4 double
/// the round-0 codeword; the round-0 fold drops 4 -> 3 so a query leaf
/// (`stripes x 2^ff0`) halves.  OOD samples 2 per committed round; folding
/// PoW 0 (soundness rides on the query PoW).
pub fn core_whir_config(lsh: usize) -> WhirConfig {
    // Round-0 folds FEWER variables than the later rounds.  A round-0 query
    // authenticates one coset row from EVERY stripe of every round — `chunks
    // x 2^ff0` felts — and re-hashing those rows is the recursion leaf's
    // dominant cost (measured: at reth areas the uniform ff=7 schedule gives
    // ~28K felts/query x 84 queries ≈ 2.4M felts ⇒ a ~640M-cell leaf that
    // cannot fit a 32GB card).  ff0=4 cuts that term 8x; later rounds query
    // a single folded poly (leaf = 2^7 felts, chunk-independent) so their
    // factor stays 7.  Query counts, rates, and PoW are round-indexed and
    // unchanged.  lsh=21: folds [4,7,7], final poly 2^3 coefficients.
    // Provable (unique-decoding) 64-bit schedule — see docs/soundness/.
    // Per round: queries x (-log2((1+rho)/2)) + PoW = 71x0.678+16, 51x0.956+16,
    // 49x0.994+16 ~ 64.  (The Johnson regime is capped at 65 bits by the
    // field's fold terms whatever the query count, so 64 is what both
    // accountings agree on.)  Round-0 queries drive the compress proof size
    // (79% of its bytes) and the recursion verifier's work; 124 -> 71 was the
    // 100 -> 64 bit decision of Sep 5.
    const ROUND0_FF: usize = 3;
    const START_LOG_INV_RATE: usize = 2;
    let mut rem =
        lsh.checked_sub(ROUND0_FF).expect("stacking height must exceed the round-0 folding factor");
    let mut folds = alloc::vec![ROUND0_FF];
    while rem > 6 {
        folds.push(6);
        rem -= 6;
    }
    let mut config = whir_config_for_fold_schedule(lsh, &folds, rem);
    // Rate 1/4 at the start, escalating by 3 bits per committed round
    // (`START + 3(r+1)`): 21 + 2 = 23 <= KoalaBear's two-adicity of 24.
    config.starting_log_inv_rate = START_LOG_INV_RATE;
    for (r, rp) in config.round_parameters.iter_mut().enumerate() {
        rp.log_inv_rate = START_LOG_INV_RATE + 3 * (r + 1);
    }
    let num_rounds = config.round_parameters.len();
    let queries = [71usize, 51, 49, 49, 49, 49, 49];
    for (r, rp) in config.round_parameters.iter_mut().enumerate() {
        rp.num_queries = queries[r.min(queries.len() - 1)];
        rp.queries_pow_bits = 16;
        rp.ood_samples = 2;
    }
    // The final queries open the LAST committed codeword (committed by
    // round num_rounds-2); its rate is that round's log_inv_rate.
    config.final_queries = queries[(num_rounds - 1).min(queries.len() - 1)];
    config.final_pow_bits = 16;
    config
}

/// Split the stacking interleave's width-`batch` stripes into width-1
/// polynomials, in the SAME flat order BaseFold's `round_batch_evaluations`
/// reports (stripe-major, then column: `eval_at` returns per-column evals).
fn split_stripes_to_polys(stripes: Vec<Arc<Mle<JaggedVal>>>) -> Vec<Arc<Mle<JaggedVal>>> {
    let mut polys = Vec::new();
    for stripe in stripes {
        let width = stripe.num_polynomials();
        let vals = stripe.guts().as_slice();
        if width <= 1 {
            polys.push(stripe.clone());
            continue;
        }
        let height = vals.len() / width;
        for col in 0..width {
            let column: Vec<JaggedVal> = (0..height).map(|r| vals[r * width + col]).collect();
            polys.push(Arc::new(Mle::from_row_major(RowMajorMatrix::new(column, 1))));
        }
    }
    polys
}

/// Commit chip traces under jagged-WHIR.  Transcript-silent, exactly like
/// `commit_jagged_pcs_generic`: the caller owns the commitment observe.
#[allow(clippy::type_complexity)]
pub fn commit_jagged_whir_generic<MT, D>(
    chip_traces: Vec<(String, RowMajorMatrix<JaggedVal>)>,
    mmcs: MT,
    dft: Arc<D>,
    config: WhirConfig,
) -> (JaggedCommitGeneric<MT>, JaggedWhirProverDataGeneric<MT>)
where
    MT: Mmcs<JaggedVal, Commitment: Clone, ProverData<RowMajorMatrix<JaggedVal>>: 'static> + Clone,
    D: TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
{
    let (mles, chip_dims) = chips_to_mles_owned(chip_traces);
    let total_entries: usize = mles.iter().map(|m| m.guts().total_len()).sum();
    let log_stacking_height = pick_log_stacking_height(total_entries);
    let area = total_entries.next_multiple_of(1usize << log_stacking_height);

    let stripes =
        interleave_multilinears_with_fixed_rate(DEFAULT_BATCH_SIZE, mles, log_stacking_height);
    let polys = split_stripes_to_polys(stripes);
    debug_assert_eq!(polys.len(), area >> log_stacking_height);

    let prover = StackedWhirProver::<JaggedVal, JaggedChallenge, MT, D>::new(
        mmcs,
        dft,
        config,
        log_stacking_height,
    );
    let stacked_data = prover.commit_stripes(polys);

    let commit = JaggedCommitGeneric::<MT> {
        original_commitment: stacked_data.commitment.clone(),
        chip_dims: chip_dims.clone(),
        area,
        log_stacking_height,
    };
    let prover_data =
        JaggedWhirProverDataGeneric::<MT> { stacked_data, chip_dims, area, log_stacking_height };
    (commit, prover_data)
}

/// ONE batched open across every round's committed data — the WHIR sibling of
/// `open_jagged_pcs_rounds_generic`.
pub fn open_jagged_whir_rounds_generic<Challenger, MT, D, EFD>(
    rounds: &[&JaggedWhirProverDataGeneric<MT>],
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut Challenger,
    mmcs: MT,
    dft: Arc<D>,
    ef_dft: Arc<EFD>,
    config: WhirConfig,
) -> StackedWhirProof<JaggedVal, JaggedChallenge, MT>
where
    MT: Mmcs<JaggedVal, Commitment: Clone, ProverData<RowMajorMatrix<JaggedVal>>: 'static> + Clone,
    D: TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
    EFD: TwoAdicSubgroupDft<JaggedChallenge>,
    Challenger: FieldChallenger<JaggedVal>
        + GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as Mmcs<JaggedVal>>::Commitment>
        + 'static,
{
    open_jagged_whir_rounds_generic_with_engine::<Challenger, MT, D, EFD>(
        rounds, eval_point, challenger, mmcs, dft, ef_dft, config, None,
    )
}

/// [`open_jagged_whir_rounds_generic`] with an optional
/// [`crate::whir::stacked::WhirRound0Engine`] carrying the
/// stacking-height-sized work (see the trait docs); `None` is the plain
/// host path.
#[allow(clippy::too_many_arguments)]
pub fn open_jagged_whir_rounds_generic_with_engine<Challenger, MT, D, EFD>(
    rounds: &[&JaggedWhirProverDataGeneric<MT>],
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut Challenger,
    mmcs: MT,
    dft: Arc<D>,
    ef_dft: Arc<EFD>,
    config: WhirConfig,
    engine: Option<&mut dyn crate::whir::stacked::WhirRound0Engine<JaggedVal, JaggedChallenge, MT>>,
) -> StackedWhirProof<JaggedVal, JaggedChallenge, MT>
where
    MT: Mmcs<JaggedVal, Commitment: Clone, ProverData<RowMajorMatrix<JaggedVal>>: 'static> + Clone,
    D: TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
    EFD: TwoAdicSubgroupDft<JaggedChallenge>,
    Challenger: FieldChallenger<JaggedVal>
        + GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as Mmcs<JaggedVal>>::Commitment>
        + 'static,
{
    let log_stacking_height = rounds[0].log_stacking_height;
    let prover = StackedWhirProver::<JaggedVal, JaggedChallenge, MT, D>::new(
        mmcs,
        dft,
        config,
        log_stacking_height,
    );
    let stack_point: Vec<JaggedChallenge> = eval_point[..log_stacking_height as usize].to_vec();
    let stacked: Vec<&_> = rounds.iter().map(|r| &r.stacked_data).collect();
    prover.prove_trusted_evaluation_with_engine(ef_dft, stack_point, &stacked, challenger, engine)
}

/// Verify a jagged-WHIR batched open: bind the claim by interpolating the
/// echoed per-polynomial evaluations at the batch coordinates (the
/// `StackingMismatch` check), then run the stacked WHIR verifier on the stack
/// coordinates.
#[allow(clippy::too_many_arguments)]
pub fn verify_jagged_whir_rounds<Challenger, MT>(
    mmcs: MT,
    config: WhirConfig,
    log_stacking_height: u32,
    commitments: &[<MT as Mmcs<JaggedVal>>::Commitment],
    round_areas: &[usize],
    point: &[JaggedChallenge],
    proof: &StackedWhirProof<JaggedVal, JaggedChallenge, MT>,
    evaluation_claim: JaggedChallenge,
    challenger: &mut Challenger,
) -> Result<(), WhirVerifierError>
where
    MT: Mmcs<JaggedVal, Commitment: Clone> + Clone,
    Challenger: FieldChallenger<JaggedVal>
        + GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as Mmcs<JaggedVal>>::Commitment>
        + 'static,
{
    let lsh = log_stacking_height as usize;
    if point.len() < lsh {
        return Err(WhirVerifierError::IncorrectShape("point too short".into()));
    }
    let stack_point = &point[..lsh];
    let batch_point = &point[lsh..];

    // Round stripe counts from the areas (the jagged layer's own metadata).
    let mut stripe_counts = Vec::with_capacity(round_areas.len());
    for &area in round_areas {
        if !area.is_multiple_of(1usize << lsh) {
            return Err(WhirVerifierError::IncorrectShape("area alignment".into()));
        }
        stripe_counts.push(area >> lsh);
    }

    // The StackingMismatch bind: the claim must equal the interpolation of the
    // flat echoed evaluations at the batch coordinates.
    let flat: Vec<JaggedChallenge> = proof.batch_evaluations.iter().flatten().copied().collect();
    let mut current = flat;
    current.resize(1usize << batch_point.len(), JaggedChallenge::ZERO);
    for &r in batch_point {
        let half = current.len() / 2;
        for i in 0..half {
            let lo = current[2 * i];
            let hi = current[2 * i + 1];
            current[i] = lo + r * (hi - lo);
        }
        current.truncate(half);
    }
    if current[0] != evaluation_claim {
        return Err(WhirVerifierError::IncorrectShape("stacking mismatch".into()));
    }

    let verifier = StackedWhirVerifier::<JaggedVal, JaggedChallenge, MT>::new(
        mmcs,
        config,
        log_stacking_height,
    );
    verifier.verify_trusted_evaluation(commitments, &stripe_counts, stack_point, proof, challenger)
}
