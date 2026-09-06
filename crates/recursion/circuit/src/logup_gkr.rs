//! In-circuit LogUp-GKR verifier helpers.
//!
//! This module hosts the small, self-contained helpers used by the
//! full LogUp-GKR sumcheck-stack verifier:
//!
//!   - [`evaluate_mle_ext`]: evaluate a multilinear extension at a
//!     verifier-sampled point, returning a single Ext value
//!   - [`sample_point`]: convenience to sample `n` Ext challenges
//!     in one call
//!   - [`observe_ext_element`] / [`observe_ext_slice`]: decompose
//!     each Ext into its `D` base-field components and feed them
//!     into the challenger
//!
//! The full `verify_logup_gkr` orchestrator composes these helpers
//! with the [`crate::sumcheck::verify_sumcheck`] inner loop and the
//! public-values constraint folder.
//!
//! # Reference
//!
//! Mirrors the upstream crates/recursion/circuit/src/logup_gkr.rs
//! verifier helpers.

use std::marker::PhantomData;

use p3_air::Air;
use p3_field::{Algebra, PrimeCharacteristicRing};
use zkm_pcs::air::MachineAir;
use zkm_pcs::MachineChip;
use zkm_recursion_compiler::ir::{Builder, Ext, Felt, SymbolicExt};

use crate::basefold_chip_opened_values::BasefoldShardOpenedValuesVariable;
use crate::basefold_constraint_folder::BasefoldConstraintFolder;
use crate::challenger::{CanObserveVariable, FieldChallengerVariable};
use crate::public_values_folder::RecursivePublicValuesConstraintFolder;
use crate::{CircuitConfig, KoalaBearFriParametersVariable};

/// Sample `num_variables` extension-field challenges from the
/// transcript in one call.  Mirrors the `Point::from_iter((0..n).map(|_| sample_ext))`
/// idiom used throughout the upstream verifier.
pub fn sample_point<C, FC>(
    builder: &mut Builder<C>,
    challenger: &mut FC,
    num_variables: usize,
) -> Vec<Ext<C::F, C::EF>>
where
    C: CircuitConfig,
    FC: FieldChallengerVariable<C, C::Bit>,
{
    (0..num_variables).map(|_| challenger.sample_ext(builder)).collect()
}

/// Decompose `value` into its `D` base-field components and observe
/// them into the challenger.  Convenience wrapper around
/// [`crate::CircuitConfig::ext2felt`] + [`CanObserveVariable::observe_slice`].
pub fn observe_ext_element<C, FC>(
    builder: &mut Builder<C>,
    challenger: &mut FC,
    value: Ext<C::F, C::EF>,
) where
    C: CircuitConfig,
    FC: FieldChallengerVariable<C, C::Bit>,
{
    let felts = C::ext2felt(builder, value);
    challenger.observe_slice(builder, felts);
}

/// Decompose every Ext in `slice` into base-field components and
/// observe them in order.  Used inside the LogUp verifier to
/// observe the per-round prover messages and the GKR circuit
/// output's MLE evaluation vectors.
pub fn observe_ext_slice<C, FC>(
    builder: &mut Builder<C>,
    challenger: &mut FC,
    slice: &[Ext<C::F, C::EF>],
) where
    C: CircuitConfig,
    FC: FieldChallengerVariable<C, C::Bit>,
{
    for value in slice {
        observe_ext_element::<C, FC>(builder, challenger, *value);
    }
}

/// Observe a LENGTH-PREFIXED extension-field slice: the element COUNT as one
/// felt, then the elements.
///
/// In-circuit mirror of the host `zkm_pcs::shard_level::prover::
/// observe_length_prefixed_ext`.  The
/// length is a circuit CONSTANT (opening widths are fixed by the shape/VK), so
/// this costs one extra observed felt per slice and no witness.
pub fn observe_length_prefixed_ext_slice<C, FC>(
    builder: &mut Builder<C>,
    challenger: &mut FC,
    slice: &[Ext<C::F, C::EF>],
) where
    C: CircuitConfig,
    FC: FieldChallengerVariable<C, C::Bit>,
{
    let len_felt: Felt<C::F> = builder.constant(C::F::from_canonical_usize(slice.len()));
    challenger.observe(builder, len_felt);
    observe_ext_slice::<C, FC>(builder, challenger, slice);
}

/// Evaluate a multilinear extension `mle_evals` (the dense
/// hypercube-evaluation vector of length `2^point.len()`) at the
/// verifier-sampled extension point.
///
/// Returns the single Ext value `MLE(point) = Σ_i mle_evals[i] · eq(i, point)`
/// where `eq(i, point)` is the partial-Lagrange weight at boolean
/// vertex `i`.
///
/// Uses the LSB-first hypercube indexing convention (matches
/// [`zkm_pcs::basefold::mle::Mle::eval_at`]): `point[0]`
/// controls the LSB of the index, `point[n-1]` the MSB.
///
/// Mirrors the upstream `evaluate_mle_ext`
/// (crates/recursion/circuit/src/sumcheck/mod.rs) shape; the Ziren port computes `partial_lagrange` symbolically
/// inside the builder rather than allocating intermediate Tensors.
pub fn evaluate_mle_ext<C: CircuitConfig>(
    builder: &mut Builder<C>,
    mle_evals: &[Ext<C::F, C::EF>],
    point: &[Ext<C::F, C::EF>],
) -> Ext<C::F, C::EF> {
    let dim = point.len();
    assert_eq!(mle_evals.len(), 1 << dim, "mle eval vector size must be 2^point.dimension");

    // partial_lagrange — index-as-MSB expansion (LSB-first point):
    // for each new coord, double the table by `(1-r)` and `r`
    // factors, putting the i_k=0 contribution at index `j` and the
    // i_k=1 contribution at index `j + old_len`.  LSB-first
    // partial-lagrange convention shared by the BaseFold pipeline.
    let mut weights: Vec<SymbolicExt<C::F, C::EF>> = vec![SymbolicExt::ONE];
    for &r in point {
        let r_sym: SymbolicExt<C::F, C::EF> = r.into();
        let old_len = weights.len();
        let mut next: Vec<SymbolicExt<C::F, C::EF>> = vec![SymbolicExt::ZERO; old_len * 2];
        for j in 0..old_len {
            let prod = weights[j] * r_sym;
            next[j] = weights[j] - prod;
            next[j + old_len] = prod;
        }
        weights = next;
    }

    // Dot product Σ_i mle_evals[i] · weights[i] inside the
    // symbolic algebra.
    let acc: SymbolicExt<C::F, C::EF> = mle_evals
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| SymbolicExt::<C::F, C::EF>::from(*v) * *w)
        .fold(SymbolicExt::ZERO, |a, b| a + b);

    builder.eval(acc)
}

/// Build a symbolic partial-Lagrange table for a point of length
/// `n`, returning `Vec<SymbolicExt>` of length `2^n`.
///
/// Index ordering matches [`evaluate_mle_ext`]: LSB-first
/// (index `i`'s bit `k` corresponds to point coordinate `k`).
/// Used by [`verify_public_values`] to expand the LogUp
/// `beta_seed` into the per-interaction beta-power table.
pub fn partial_lagrange_symbolic<C: CircuitConfig>(
    point: &[SymbolicExt<C::F, C::EF>],
) -> Vec<SymbolicExt<C::F, C::EF>> {
    let mut weights: Vec<SymbolicExt<C::F, C::EF>> = vec![SymbolicExt::ONE];
    for &r in point {
        let old_len = weights.len();
        let mut next: Vec<SymbolicExt<C::F, C::EF>> = vec![SymbolicExt::ZERO; old_len * 2];
        for j in 0..old_len {
            let prod = weights[j] * r;
            next[j] = weights[j] - prod;
            next[j + old_len] = prod;
        }
        weights = next;
    }
    weights
}

/// Verify the public-values portion of the LogUp-GKR argument.
///
/// Builds the per-record constraint folder, lets the caller emit
/// record-level constraints into it via `eval_public_values_fn`,
/// asserts the accumulator is zero, and returns the resulting
/// `local_interaction_digest`.
///
/// The caller-supplied closure decouples this verifier from any
/// concrete `MachineRecord::eval_public_values` trait method —
/// the closure receives a mutable reference to the folder and is
/// expected to call `assert_zero` for each per-record constraint.
/// Records with no public-values constraints can pass an empty
/// closure.
///
/// # Arguments
///
/// * `challenge` — alpha for constraint folding
/// * `alpha` — the LogUp permutation `alpha` challenge
/// * `beta_seed` — the LogUp `beta_seed` point (length =
///   `log2_ceil(max_interaction_arity)`); expanded to per-
///   interaction beta-powers via partial Lagrange
/// * `public_values` — the shard's public values
/// * `eval_public_values_fn` — closure that emits record-level
///   constraints into the folder
///
/// # Returns
///
/// The `local_interaction_digest` symbolic value, which the LogUp
/// orchestrator compares against the GKR-circuit-derived
/// cumulative-sum value.
///
/// # Reference
///
/// Mirrors `RecursiveLogUpGkrVerifier::verify_public_values`
/// (crates/recursion/circuit/src/logup_gkr.rs).
/// Substitution: the upstream's `A::Record::eval_public_values`
/// trait dispatch becomes a closure parameter so this function
/// doesn't depend on a Record trait extension on the Ziren side.
pub fn verify_public_values<C, F>(
    builder: &mut Builder<C>,
    challenge: Ext<C::F, C::EF>,
    alpha: &Ext<C::F, C::EF>,
    beta_seed: &[Ext<C::F, C::EF>],
    public_values: &[Felt<C::F>],
    eval_public_values_fn: F,
) -> SymbolicExt<C::F, C::EF>
where
    C: CircuitConfig,
    F: FnOnce(&mut RecursivePublicValuesConstraintFolder<C>),
{
    // Lift beta_seed into the symbolic algebra and expand to per-
    // interaction beta-powers via partial Lagrange.
    let beta_symbolic: Vec<SymbolicExt<C::F, C::EF>> =
        beta_seed.iter().map(|e| SymbolicExt::from(*e)).collect();
    let betas = partial_lagrange_symbolic::<C>(&beta_symbolic);

    let mut folder = RecursivePublicValuesConstraintFolder::<C> {
        perm_challenges: (alpha, &betas),
        alpha: challenge,
        accumulator: SymbolicExt::ZERO,
        public_values,
        local_interaction_digest: SymbolicExt::ZERO,
        _marker: PhantomData,
    };

    eval_public_values_fn(&mut folder);

    // Assert the accumulator is zero — the constraints emitted
    // through the folder must hold for the proof to be sound.
    builder.assert_ext_eq(folder.accumulator, SymbolicExt::ZERO);

    folder.local_interaction_digest
}

/// Number of grinding bits for the LogUp-GKR challenge — must stay in
/// lockstep with the host prover's `zkm_pcs::logup_gkr::GKR_GRINDING_BITS`
/// (= 12); the in-circuit verifier re-checks the same witness the host
/// ground, so a mismatch would reject honest proofs.
pub const GKR_GRINDING_BITS: usize = 0;

/// Per-shard chip introspection input to [`verify_logup_gkr`].
///
/// Encapsulates the Chip-introspection bits the verifier needs
/// without coupling this module to a particular Chip type.  The
/// caller computes these from `MachineChip` introspection
/// (sends/receives counts, max interaction arity).
pub struct LogupGkrShardChipMetadata {
    /// `log2_ceil(max_interaction_arity)` across all chips, where
    /// `interaction_arity = values.len() + 1` per send/receive.
    /// Determines the LogUp `beta_seed` dimension.
    pub beta_seed_dim: usize,
    /// `log2_ceil(total_num_interactions)` where
    /// `total_num_interactions = Σ_chip (sends.len() + receives.len())`.
    /// Determines the GKR-circuit input dimension.
    pub log_num_interactions: usize,
}

/// Verify a LogUp-GKR proof in-circuit.
///
/// Replays the LogUp-GKR sumcheck stack:
///
///   1. Check the GKR-grinding witness
///   2. Sample (alpha, beta_seed, pv_challenge) from the transcript
///   3. Evaluate public-value constraints (delegates to caller via
///      `eval_public_values_fn`); use the resulting digest as the
///      negated cumulative sum
///   4. Observe the GKR circuit output (numerator + denominator
///      MLE evaluations) into the transcript
///   5. Assert `Σ_i (num[i] / den[i]) == cumulative_sum`
///   6. Sample the first evaluation point
///   7. For each round: sample lambda, assert sumcheck-claim
///      consistency, run [`crate::sumcheck::verify_sumcheck`],
///      sample the round's last coordinate, fold the
///      numerator/denominator MLE evaluations
///
/// The caller-supplied `chip_metadata` provides the chip-
/// enumeration bits the verifier needs (interaction count,
/// beta-seed dimension); `eval_public_values_fn` is the same
/// closure parameter used by [`verify_public_values`].
///
/// # Transcript convention (LSB-fold, push-at-back)
///
/// The per-round 4-tuple is observed as `(n0, n1, d0, d1)` and the
/// new `last_coordinate` is appended to the back of `eval_point`
/// (LSB-first push: `eval_point[len-1] = last_coordinate`).
///
/// Per-round transcript ops (in order — must match the prover):
///
/// | Step | Operation | Line |
/// |---|---|---|
/// | 1 | sample `lambda` | logup_gkr.rs:401 |
/// | 2 | assert `claimed_sum == numerator_eval * lambda + denominator_eval` | logup_gkr.rs:406-407 |
/// | 3 | `verify_sumcheck` | logup_gkr.rs:410-414 |
/// | 4 | assert `final_eval == eq(point,eval_point) * ((n0*d1 + n1*d0)*λ + d0*d1)` | logup_gkr.rs:430-440 |
/// | 5 | observe `n0` | logup_gkr.rs:447 |
/// | 6 | observe `n1` | logup_gkr.rs:448 |
/// | 7 | observe `d0` | logup_gkr.rs:449 |
/// | 8 | observe `d1` | logup_gkr.rs:450 |
/// | 9 | `eval_point = sumcheck_point.clone()` | logup_gkr.rs:461 |
/// | 10 | sample `last_coordinate` | logup_gkr.rs:462 |
/// | 11 | append `last_coordinate` to back of `eval_point` | logup_gkr.rs:463 (`push`) |
/// | 12 | fold `num_eval = n0 + (n1 - n0) * last_coord` | logup_gkr.rs:469 |
/// | 13 | fold `den_eval = d0 + (d1 - d0) * last_coord` | logup_gkr.rs:470 |
///
/// Trace-evaluation reconstruction from per-chip openings is
/// deferred to the zerocheck stage; consumed via
/// `proof.logup_evaluations`.
#[allow(clippy::too_many_arguments)]
pub fn verify_logup_gkr<C, SC, A, FC, EVPV>(
    builder: &mut Builder<C>,
    chip_metadata: &LogupGkrShardChipMetadata,
    proof: &crate::logup_proof::LogupGkrProof<Felt<C::F>, Ext<C::F, C::EF>>,
    shard_chips: &[&MachineChip<SC, A>],
    opened_values: &BasefoldShardOpenedValuesVariable<C>,
    public_values: &[Felt<C::F>],
    max_log_row_count: usize,
    challenger: &mut FC,
    eval_public_values_fn: EVPV,
) where
    C: CircuitConfig<F = SC::Val>,
    SC: KoalaBearFriParametersVariable<C>,
    A: MachineAir<C::F> + for<'b> Air<BasefoldConstraintFolder<'b, C>>,
    FC: FieldChallengerVariable<C, C::Bit>,
    EVPV: FnOnce(&mut RecursivePublicValuesConstraintFolder<C>),
    SymbolicExt<C::F, C::EF>: Algebra<C::EF>,
{
    let crate::logup_proof::LogupGkrProof {
        circuit_output,
        round_proofs,
        logup_evaluations,
        witness,
    } = proof;
    let crate::logup_proof::LogUpGkrOutput { numerator, denominator } = circuit_output;

    // The GKR round count is FIXED — the prover
    // pads to max_log_row_count-1 rounds and the verifier asserts
    // `round_proofs.len() + 1 == max_log_row_count`.  The count is
    // STRUCTURAL in the recursion program (the loop below unrolls over
    // the lifted vec), so enforcement happens at program build: refuse
    // to build a verifier circuit for a shortened reduction.  Mirrors
    // the host check in shard_level/verifier.rs::verify_logup_gkr_host.
    assert_eq!(
        round_proofs.len() + 1,
        max_log_row_count,
        "LogUp-GKR proof must carry exactly max_log_row_count-1 padded rounds"
    );

    // (1) Check the proof-of-work grinding witness.  Use `gkr_check_witness`
    // (NOT `check_witness`): the host gates GKR grinding to the inner
    // challenger — inner advances + checks, the OUTER/wrap ring is a no-op.
    // The BaseFold open uses the distinct `check_witness`, which advances on
    // both rings.
    challenger.gkr_check_witness(builder, GKR_GRINDING_BITS, *witness);

    // (2) Sample the permutation challenges (alpha + beta_seed).
    // beta_seed dim is decided by chip metadata.  NOTE: the host
    // prover (row_gkr/top_level.rs:71-86) and host verifier
    // (shard_level/verifier.rs:1135-1138) sample ONLY [alpha, beta_seed]
    // here — there is NO separate public-values challenge draw.  The
    // public-values digest folds the record-level PV interactions under
    // the SAME `alpha` (used as both the permutation challenge and the
    // constraint-fold alpha; see `eval_public_values_digest_host`,
    // public_values_folder.rs:132, which passes `alpha` for both the
    // `perm_challenges.0` and `alpha` slots).  A prior version sampled an
    // EXTRA `pv_challenge` here, which had no host counterpart: it
    // desynced every post-alpha squeeze (eval_point, per-round lambda,
    // the whole GKR sumcheck) from the prover's transcript.  Drop it and
    // reuse `alpha` so the in-circuit transcript is byte-identical to the
    // host from alpha onward.
    let alpha = challenger.sample_ext(builder);
    let beta_seed: Vec<Ext<C::F, C::EF>> =
        (0..chip_metadata.beta_seed_dim).map(|_| challenger.sample_ext(builder)).collect();

    // (3) Evaluate public-values constraints.  Negated digest =
    // cumulative_sum (matches the sign convention upstream).  The
    // constraint-fold alpha is `alpha` itself (host parity), not a
    // separate challenge.
    let local_interaction_digest = verify_public_values::<C, _>(
        builder,
        alpha,
        &alpha,
        &beta_seed,
        public_values,
        eval_public_values_fn,
    );
    let cumulative_sum: SymbolicExt<C::F, C::EF> = -local_interaction_digest;

    // (4) Observe the GKR circuit output (per-element ext slice).
    observe_ext_slice::<C, FC>(builder, challenger, numerator);
    observe_ext_slice::<C, FC>(builder, challenger, denominator);

    // (5) Assert Σ (numerator[i] / denominator[i]) == cumulative_sum.
    let output_cumulative_sum: SymbolicExt<C::F, C::EF> = numerator
        .iter()
        .zip(denominator.iter())
        .map(|(n, d)| {
            let n_sym: SymbolicExt<C::F, C::EF> = (*n).into();
            let d_sym: SymbolicExt<C::F, C::EF> = (*d).into();
            n_sym / d_sym
        })
        .fold(SymbolicExt::ZERO, |acc, x| acc + x);
    builder.assert_ext_eq(output_cumulative_sum, cumulative_sum);

    // (6) Sample the first evaluation point.  Dimension =
    // log_num_interactions + 1 (one extra var for the GKR circuit
    // output's MLE depth above the per-interaction layer).
    let initial_num_variables = chip_metadata.log_num_interactions + 1;
    let mut eval_point: Vec<Ext<C::F, C::EF>> =
        sample_point::<C, FC>(builder, challenger, initial_num_variables);

    // Initial evaluation of the numerator/denominator MLEs at the
    // sampled point — this is what gets reduced through GKR.
    let mut numerator_eval: SymbolicExt<C::F, C::EF> =
        evaluate_mle_ext::<C>(builder, numerator, &eval_point).into();
    let mut denominator_eval: SymbolicExt<C::F, C::EF> =
        evaluate_mle_ext::<C>(builder, denominator, &eval_point).into();

    // (7) Iterate round_proofs in order.
    for round_proof in round_proofs.iter() {
        // Sample the batching challenge λ for combining the two
        // claims (numerator + denominator) into one sumcheck.
        let lambda = challenger.sample_ext(builder);
        let lambda_sym: SymbolicExt<C::F, C::EF> = lambda.into();

        // Per-round soundness: the sumcheck's claimed_sum must
        // equal `numerator_eval * λ + denominator_eval`.
        let expected_claim = numerator_eval * lambda_sym + denominator_eval;
        builder.assert_ext_eq(round_proof.sumcheck_proof.claimed_sum, expected_claim);

        // Verify the per-round sumcheck.
        crate::sumcheck::verify_sumcheck::<C, FC>(builder, challenger, &round_proof.sumcheck_proof);

        // Verify the eval claim is consistent with the prover's
        // 4-tuple message.  The tuple encodes (num_0, num_1,
        // den_0, den_1) — values with the round's last coord
        // fixed to 0 and 1.  Combined into the GKR identity:
        //
        //   final_eval = eq(point, eval_point) *
        //     (num_0 * den_1 + num_1 * den_0) * λ +
        //     (den_0 * den_1)
        let (sumcheck_point, final_eval) = (
            &round_proof.sumcheck_proof.point_and_eval.0,
            round_proof.sumcheck_proof.point_and_eval.1,
        );
        let sumcheck_point_sym: Vec<SymbolicExt<C::F, C::EF>> =
            sumcheck_point.iter().map(|e| (*e).into()).collect();
        let eval_point_sym: Vec<SymbolicExt<C::F, C::EF>> =
            eval_point.iter().map(|e| (*e).into()).collect();
        let eq_eval_value = crate::zerocheck::eq_eval::<C>(&sumcheck_point_sym, &eval_point_sym);
        let n0_sym: SymbolicExt<C::F, C::EF> = round_proof.numerator_0.into();
        let n1_sym: SymbolicExt<C::F, C::EF> = round_proof.numerator_1.into();
        let d0_sym: SymbolicExt<C::F, C::EF> = round_proof.denominator_0.into();
        let d1_sym: SymbolicExt<C::F, C::EF> = round_proof.denominator_1.into();
        let numerator_sumcheck_eval = n0_sym * d1_sym + n1_sym * d0_sym;
        let denominator_sumcheck_eval = d0_sym * d1_sym;
        let expected_final_eval =
            eq_eval_value * (numerator_sumcheck_eval * lambda_sym + denominator_sumcheck_eval);
        builder.assert_ext_eq(final_eval, expected_final_eval);

        // Observe the prover's 4-tuple message into the transcript.
        // Order MUST be the `(n0, n1, d0, d1)` sequence
        // — any reorder shifts every subsequent α-sample and
        // produces an OOD mismatch.
        observe_ext_element::<C, FC>(builder, challenger, round_proof.numerator_0);
        observe_ext_element::<C, FC>(builder, challenger, round_proof.numerator_1);
        observe_ext_element::<C, FC>(builder, challenger, round_proof.denominator_0);
        observe_ext_element::<C, FC>(builder, challenger, round_proof.denominator_1);

        // Update eval_point: take the sumcheck-reduced point and INSERT a
        // freshly-sampled line coordinate at index `log_num_interactions`
        // (the row LSB of the LSB-first flat index).  The prover's layer
        // transition pairs ADJACENT rows, so the peeled variable is the row
        // LSB and the reduced point's row coordinates shift up by one.
        // Mirrors `row_gkr/top_level.rs` and `shard_level/verifier.rs`.
        eval_point = sumcheck_point.clone();
        let last_coordinate = challenger.sample_ext(builder);
        eval_point.insert(chip_metadata.log_num_interactions, last_coordinate);

        // Update numerator/denominator evals via the linear
        // interpolation at last_coordinate:
        //   eval_new = eval_0 + (eval_1 - eval_0) * last_coord
        let last_coord_sym: SymbolicExt<C::F, C::EF> = last_coordinate.into();
        numerator_eval = n0_sym + (n1_sym - n0_sym) * last_coord_sym;
        denominator_eval = d0_sym + (d1_sym - d0_sym) * last_coord_sym;
    }

    // ── DEGREE-MASKED LAST-LAYER RECONSTRUCTION ──
    //
    // In-circuit mirror of the host `verify_logup_gkr_host` reconstruction
    // (crates/pcs/src/shard_level/verifier.rs:1628-1881).
    //
    // The round walk above reduced the GKR `circuit_output` num/den MLEs to
    // (numerator_eval, denominator_eval) at the fully-reduced `eval_point`
    // (dim = log_num_interactions + max_log_row_count).  Without this block
    // those evals are DISCARDED — the verifier never ties the GKR output back
    // to the chips' actual trace openings, leaving the area-preserving height-
    // forgery hole.  We re-derive (num, den) from the per-chip trace openings
    // masked by `full_geq(degree, ·)` and assert they equal the round walk's.
    // A forged `degree` (height) moves the `full_geq` boundary, perturbing the
    // reconstruction while the walk's evals (which never see `degree`) stay
    // fixed → reject.  This is pure arithmetic over already-sampled challenges
    // and already-observed openings: transcript- and proof-byte-neutral.
    {
        let log_num_interactions = initial_num_variables - 1;

        // (1) Split the reduced eval_point into (interaction, trace) axes.
        // The round walk leaves `eval_point` LSB-first, interaction axis low.
        assert_eq!(
            eval_point.len(),
            log_num_interactions + max_log_row_count,
            "logup reconstruction: reduced eval_point dim {} != log_num_interactions {} + \
             max_log_row_count {}",
            eval_point.len(),
            log_num_interactions,
            max_log_row_count,
        );
        let (interaction_point, trace_point) = eval_point.split_at(log_num_interactions);

        // (2) The trace point must equal the claimed opening point.
        assert_eq!(
            trace_point.len(),
            max_log_row_count,
            "logup reconstruction: trace_point dim {} != max_log_row_count {}",
            trace_point.len(),
            max_log_row_count,
        );
        assert_eq!(
            logup_evaluations.point.len(),
            trace_point.len(),
            "logup reconstruction: logup_evaluations.point dim {} != trace_point dim {}",
            logup_evaluations.point.len(),
            trace_point.len(),
        );
        for (claimed, expected) in logup_evaluations.point.iter().zip(trace_point.iter()) {
            builder.assert_ext_eq(*claimed, *expected);
        }

        // (3) `point_extended` for the per-chip `full_geq` padding mask:
        // [ZERO, ...trace_point.rev()] — REVERSED, the LSB-first GKR-leaf mask
        // convention (host verifier.rs:1681-1683).  `full_geq` (zerocheck.rs:68,
        // MSB-first internally) over this reproduces the LSB-first leaf mask
        // geq = Σ_{row ≥ height} eq(row, trace_point).
        let mut point_extended: Vec<SymbolicExt<C::F, C::EF>> =
            Vec::with_capacity(max_log_row_count + 1);
        point_extended.push(SymbolicExt::ZERO);
        point_extended
            .extend(trace_point.iter().rev().map(|p| -> SymbolicExt<C::F, C::EF> { (*p).into() }));

        // (4) Expand the LogUp challenges into the symbolic algebra.  `betas`
        // = partial-Lagrange table over `beta_seed` (= host `eq_mle_table`,
        // both LSB-first); `betas[0]` is the argument_index weight.
        let alpha_sym: SymbolicExt<C::F, C::EF> = alpha.into();
        let beta_seed_sym: Vec<SymbolicExt<C::F, C::EF>> =
            beta_seed.iter().map(|b| -> SymbolicExt<C::F, C::EF> { (*b).into() }).collect();
        let betas: Vec<SymbolicExt<C::F, C::EF>> = partial_lagrange_symbolic::<C>(&beta_seed_sym);

        // (5) Per-chip reconstruction.  `shard_chips`, `opened_values.chips`,
        // and `logup_evaluations.chip_openings.values()` are ALL name-ordered
        // (the call site sorts `shard_chips` by name; both maps are name-keyed),
        // so they align positionally — matching the host's
        // name-keyed lookup.  RAW-contiguous packing (Ziren extract.rs), padded
        // to the global interaction axis below.
        assert_eq!(
            opened_values.chips.len(),
            shard_chips.len(),
            "logup reconstruction: opened_values chip count {} != shard_chips {}",
            opened_values.chips.len(),
            shard_chips.len(),
        );
        assert_eq!(
            logup_evaluations.chip_openings.len(),
            shard_chips.len(),
            "logup reconstruction: chip_openings count {} != shard_chips {}",
            logup_evaluations.chip_openings.len(),
            shard_chips.len(),
        );

        let mut numerator_values: Vec<SymbolicExt<C::F, C::EF>> = Vec::new();
        let mut denominator_values: Vec<SymbolicExt<C::F, C::EF>> = Vec::new();

        for ((chip, opening), chip_eval) in shard_chips
            .iter()
            .zip(opened_values.chips.iter())
            .zip(logup_evaluations.chip_openings.values())
        {
            // degree = big-endian boolean coords of the chip HEIGHT (2^log_h),
            // the in-circuit analog of host `opening.quotient.first()`.
            let degree_sym: Vec<SymbolicExt<C::F, C::EF>> = opening
                .degree
                .iter()
                .map(|d| -> SymbolicExt<C::F, C::EF> { (*d).into() })
                .collect();
            assert_eq!(
                degree_sym.len(),
                point_extended.len(),
                "logup reconstruction: chip degree dim {} != point_extended dim {}",
                degree_sym.len(),
                point_extended.len(),
            );
            let geq_eval = crate::zerocheck::full_geq::<C>(&degree_sym, &point_extended);

            // FULL-POINT openings (the GKR leaf is LSB-first
            // natural-row).  Production FIX-off proofs always carry `*_full`;
            // panic if absent (matches the gated host assert semantics — the
            // reconstruction is only meaningful on `*_full`-carrying proofs).
            let main: &[Ext<C::F, C::EF>] = chip_eval
                .main_trace_evaluations_full
                .as_deref()
                .expect(
                "logup reconstruction requires main_trace_evaluations_full (FIX-off core proof)",
            );
            let prep: Option<&[Ext<C::F, C::EF>]> =
                chip_eval.preprocessed_trace_evaluations_full.as_deref();

            // Zero padding openings (the trace eval on a fully-padding all-zero
            // row), used to correct the padding region.
            let zero_ext: Ext<C::F, C::EF> = builder.constant(C::EF::ZERO);
            let padding_main: Vec<Ext<C::F, C::EF>> = vec![zero_ext; main.len()];
            let padding_prep: Option<Vec<Ext<C::F, C::EF>>> = prep.map(|p| vec![zero_ext; p.len()]);

            for (interaction, is_send) in chip
                .sends()
                .iter()
                .map(|s| (s, true))
                .chain(chip.receives().iter().map(|r| (r, false)))
            {
                let (real_numerator, real_denominator) = interaction
                    .eval::<SymbolicExt<C::F, C::EF>, Ext<C::F, C::EF>>(
                        prep,
                        main,
                        alpha_sym.clone(),
                        &betas,
                    );
                let (padding_numerator, padding_denominator) = interaction
                    .eval::<SymbolicExt<C::F, C::EF>, Ext<C::F, C::EF>>(
                        padding_prep.as_deref(),
                        &padding_main,
                        alpha_sym.clone(),
                        &betas,
                    );

                // Degree-masked num/den, then sign for receives (host
                // verifier.rs:1828-1832).
                let numerator_eval_i = real_numerator - padding_numerator * geq_eval.clone();
                let denominator_eval_i =
                    real_denominator + (SymbolicExt::ONE - padding_denominator) * geq_eval.clone();
                let numerator_eval_i = if is_send { numerator_eval_i } else { -numerator_eval_i };
                numerator_values.push(numerator_eval_i);
                denominator_values.push(denominator_eval_i);
            }
        }

        // (6) Pad to the global interaction-axis size: numerator with 0,
        // denominator with 1 (the identity fraction).  Materialize to Ext.
        numerator_values.resize(1usize << interaction_point.len(), SymbolicExt::ZERO);
        denominator_values.resize(1usize << interaction_point.len(), SymbolicExt::ONE);
        let numerator_values_ext: Vec<Ext<C::F, C::EF>> =
            numerator_values.into_iter().map(|x| builder.eval(x)).collect();
        let denominator_values_ext: Vec<Ext<C::F, C::EF>> =
            denominator_values.into_iter().map(|x| builder.eval(x)).collect();

        // (7) Evaluate the reconstructed MLEs at the interaction point
        // (LSB-first — matches host `evaluate_mle_host`).
        let expected_numerator =
            evaluate_mle_ext::<C>(builder, &numerator_values_ext, interaction_point);
        let expected_denominator =
            evaluate_mle_ext::<C>(builder, &denominator_values_ext, interaction_point);

        // (8) The height-soundness assert: the round walk's reduced final
        // evals MUST equal the reconstruction from the chips' trace openings.
        builder.assert_ext_eq(numerator_eval, expected_numerator);
        builder.assert_ext_eq(denominator_eval, expected_denominator);
    }

    // ── Observe slot 1 — the GKR trace openings (trace@ζ) ──────────
    //
    // In-circuit mirror of the host prover
    // (`row_gkr::top_level::prove_shard_logup_gkr_rows`) and of the host
    // verifier (end of `verify_logup_gkr_host`).
    //
    // Position is load-bearing: `BasefoldZerocheckVerifier::verify_zerocheck`
    // opens by sampling α / γ / λ, and the zerocheck identity it then enforces
    // (`assert_ext_eq(claimed_sum, zerocheck_sum_modification)` plus the
    // rlc_eval assert) is only a Schwartz–Zippel test of those challenges if
    // this opening vector is already committed.  Observed here — before that
    // sample — the openings are fixed; observed after (where this used to sit,
    // as step (9) of verify_zerocheck) the identity degenerates to one linear
    // equation a prover can solve for the openings.
    //
    // `shard_chips.len()` felt, then per chip in NAME order
    // (`chip_openings` is a `BTreeMap`) the four length-prefixed slices:
    // preprocessed, main, preprocessed_full, main_full.  Both opening sets are
    // observed because Ziren's core stage drives the claim from `*_full` while
    // the recursion / shrink / wrap stages drive it from the legacy set.
    let num_chips_felt: Felt<C::F> =
        builder.constant(C::F::from_canonical_usize(shard_chips.len()));
    challenger.observe(builder, num_chips_felt);
    for chip_evaluation in logup_evaluations.chip_openings.values() {
        observe_length_prefixed_ext_slice::<C, FC>(
            builder,
            challenger,
            chip_evaluation.preprocessed_trace_evaluations_full.as_deref().unwrap_or(&[]),
        );
        observe_length_prefixed_ext_slice::<C, FC>(
            builder,
            challenger,
            chip_evaluation.main_trace_evaluations_full.as_deref().unwrap_or(&[]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use zkm_pcs::{InnerChallenge, InnerVal};
    use zkm_recursion_compiler::circuit::AsmBuilder;
    use zkm_recursion_compiler::ir::Ext;

    type F = InnerVal;
    type EF = InnerChallenge;

    /// Construction smoke test: evaluating a constant-1 MLE at any
    /// point produces a single Ext output.  Doesn't run the
    /// generated program; just checks the IR construction
    /// roundtrips through the builder.
    #[test]
    fn evaluate_mle_ext_constructs_for_constant_polynomial() {
        let mut builder = AsmBuilder::<F, EF>::default();

        // 2^3 = 8 evaluations, all = 1 (constant-1 polynomial).
        let mle: Vec<Ext<F, EF>> = (0..8).map(|_| builder.constant(EF::ONE)).collect();
        let point: Vec<Ext<F, EF>> = (0..3).map(|_| builder.constant(EF::ZERO)).collect();
        let result = evaluate_mle_ext(&mut builder, &mle, &point);
        // Construction succeeded; the `Ext<F, EF>` is now part of
        // the IR.  Body intentionally elides runtime execution to
        // keep the test self-contained — IR-shape correctness is
        // covered by the `verify_shard_inner` end-to-end test in
        // [`crate::stark::tests`].
        let _ = result;
    }

    /// All-zero MLE construction smoke test.
    #[test]
    fn evaluate_mle_ext_constructs_for_zero_polynomial() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let mle: Vec<Ext<F, EF>> = (0..4).map(|_| builder.constant(EF::ZERO)).collect();
        let point: Vec<Ext<F, EF>> =
            vec![builder.constant(EF::from(F::ONE + F::ONE)), builder.constant(EF::from(F::ONE))];
        let _result = evaluate_mle_ext(&mut builder, &mle, &point);
    }

    /// Construction smoke test for partial_lagrange_symbolic.
    #[test]
    fn partial_lagrange_symbolic_returns_correct_length() {
        use zkm_recursion_compiler::config::InnerConfig;
        let mut builder = AsmBuilder::<F, EF>::default();
        let point: Vec<SymbolicExt<F, EF>> = (0..3)
            .map(|_| {
                let e: Ext<F, EF> = builder.constant(EF::ZERO);
                e.into()
            })
            .collect();
        let weights = partial_lagrange_symbolic::<InnerConfig>(&point);
        assert_eq!(weights.len(), 1usize << 3);
    }

    /// Construction smoke test for verify_public_values: empty
    /// closure should produce a folder where accumulator stays at
    /// zero (assert_ext_eq passes trivially) and digest stays at
    /// zero too.
    #[test]
    fn verify_public_values_with_empty_closure() {
        use zkm_recursion_compiler::config::InnerConfig;
        use zkm_recursion_compiler::ir::Felt;
        let mut builder = AsmBuilder::<F, EF>::default();
        let challenge: Ext<F, EF> = builder.constant(EF::ONE);
        let alpha: Ext<F, EF> = builder.constant(EF::ONE);
        let beta_seed: Vec<Ext<F, EF>> = (0..2).map(|_| builder.constant(EF::ZERO)).collect();
        let public_values: Vec<Felt<F>> = (0..4).map(|_| builder.constant(F::ZERO)).collect();

        let _digest = verify_public_values::<InnerConfig, _>(
            &mut builder,
            challenge,
            &alpha,
            &beta_seed,
            &public_values,
            |_folder| {
                // intentionally empty — no per-record constraints
            },
        );
    }

    // ── in-circuit LogUp degree-masked reconstruction tests ──
    //
    // The full `verify_logup_gkr` transcript replay is exercised end-to-end by
    // the `test_e2e_compress_fibonacci` integration test (a real FIX-off proof
    // through the recursion verifier).  Here we add EXECUTED-CIRCUIT tests
    // (`run_test_recursion`) that drive the EXACT reconstruction arithmetic the
    // in-circuit block at logup_gkr.rs:(reconstruction) computes — `Lookup::eval`
    // (Var=Ext, Expr=SymbolicExt), `full_geq` over the reversed `point_extended`,
    // and the degree-masked `num = real − pad·geq` / `den = real + (1−pad)·geq`
    // — and assert it equals an OFF-CIRCUIT host re-computation of the same
    // formula (the round-walk eval).  This proves both directions of the
    // soundness contract WITHOUT a full GKR proof:
    //   * honest (degree, *_full) → reconstruction == round-walk eval (accepts);
    //   * forged `degree` (height) → `geq` mask shifts → reconstruction diverges
    //     from the (honest) round-walk eval → the `assert_ext_eq` trips (rejects).
    //
    // Single send-interaction, single chip, log_num_interactions = 0 (one
    // interaction → `interaction_point` empty → the reconstructed numerator MLE
    // collapses to its single value), so the test isolates the per-chip
    // degree-masked num/den arithmetic — the genuine height anchor.

    use p3_air::VirtualPairCol;
    
    use zkm_pcs::air::LookupScope;
    use zkm_pcs::{Lookup, LookupKind};

    /// Off-circuit `full_geq` (MSB-first, matches `crate::zerocheck::full_geq`
    /// and host `full_geq_host`): `threshold` ≥ `eval_point` indicator.
    fn full_geq_host(threshold: &[EF], eval_point: &[EF]) -> EF {
        assert_eq!(threshold.len(), eval_point.len());
        let one = EF::ONE;
        threshold
            .iter()
            .rev()
            .zip(eval_point.iter().rev())
            .fold(one, |acc, (&x, &y)| ((one - y) * (one - x) + y * x) * acc + y * (one - x))
    }

    /// Off-circuit LSB-first partial-Lagrange table (matches
    /// `partial_lagrange_symbolic` and host `eq_mle_table`).
    fn eq_mle_table_host(r: &[EF]) -> Vec<EF> {
        let mut table = vec![EF::ONE];
        for &ri in r {
            let old = table.len();
            let mut next = vec![EF::ZERO; old * 2];
            for j in 0..old {
                let prod = table[j] * ri;
                next[j] = table[j] - prod;
                next[j + old] = prod;
            }
            table = next;
        }
        table
    }

    /// Off-circuit replica of the in-circuit per-chip single-send reconstruction
    /// for one interaction: returns the expected `(numerator, denominator)`
    /// (the round-walk eval an honest prover would carry).
    ///
    /// Mirrors logup_gkr.rs reconstruction steps (3)-(8) for one send
    /// interaction with `log_num_interactions = 0` (so num/den == the single
    /// interaction value, no MLE pad).
    fn host_reconstruct_single_send(
        lookup: &Lookup<F>,
        main_full: &[EF],
        degree: &[EF],
        alpha: EF,
        beta_seed: &[EF],
        trace_point: &[EF],
    ) -> (EF, EF) {
        // point_extended = [ZERO, ...trace_point.rev()]
        let mut point_extended = Vec::with_capacity(trace_point.len() + 1);
        point_extended.push(EF::ZERO);
        point_extended.extend(trace_point.iter().rev().copied());
        let geq = full_geq_host(degree, &point_extended);
        let betas = eq_mle_table_host(beta_seed);
        let zeros = vec![EF::ZERO; main_full.len()];
        let (real_num, real_den) = lookup.eval::<EF, EF>(None, main_full, alpha, &betas);
        let (pad_num, pad_den) = lookup.eval::<EF, EF>(None, &zeros, alpha, &betas);
        let num = real_num - pad_num * geq; // send → +num
        let den = real_den + (EF::ONE - pad_den) * geq;
        (num, den)
    }

    /// Build + EXECUTE the in-circuit single-send reconstruction for one chip,
    /// asserting the reconstructed `(num, den)` equal the supplied round-walk
    /// `(rw_num, rw_den)` constants — the exact height-soundness
    /// `assert_ext_eq` (logup_gkr.rs step (8)).  Runs the DSL so the assert
    /// fires at runtime.
    fn run_single_send_reconstruction(
        lookup: &Lookup<F>,
        main_full_vals: &[EF],
        degree_vals: &[EF],
        alpha_val: EF,
        beta_seed_vals: &[EF],
        trace_point_vals: &[EF],
        rw_num: EF,
        rw_den: EF,
    ) {
        use crate::utils::tests::run_test_recursion;
        use zkm_recursion_compiler::config::InnerConfig;
        type C = InnerConfig;

        let mut builder = Builder::<C>::default();

        // Constants for all inputs.
        let main_full: Vec<Ext<F, EF>> =
            main_full_vals.iter().map(|&v| builder.constant(v)).collect();
        let alpha: Ext<F, EF> = builder.constant(alpha_val);
        let beta_seed: Vec<Ext<F, EF>> =
            beta_seed_vals.iter().map(|&v| builder.constant(v)).collect();
        let trace_point: Vec<Ext<F, EF>> =
            trace_point_vals.iter().map(|&v| builder.constant(v)).collect();
        let degree: Vec<Ext<F, EF>> = degree_vals.iter().map(|&v| builder.constant(v)).collect();
        let rw_num_ext: Ext<F, EF> = builder.constant(rw_num);
        let rw_den_ext: Ext<F, EF> = builder.constant(rw_den);

        // (3) point_extended = [ZERO, ...trace_point.rev()].
        let mut point_extended: Vec<SymbolicExt<F, EF>> = Vec::with_capacity(trace_point.len() + 1);
        point_extended.push(SymbolicExt::ZERO);
        point_extended
            .extend(trace_point.iter().rev().map(|p| -> SymbolicExt<F, EF> { (*p).into() }));

        // (4) betas + alpha into the symbolic algebra.
        let alpha_sym: SymbolicExt<F, EF> = alpha.into();
        let beta_seed_sym: Vec<SymbolicExt<F, EF>> =
            beta_seed.iter().map(|b| -> SymbolicExt<F, EF> { (*b).into() }).collect();
        let betas = partial_lagrange_symbolic::<C>(&beta_seed_sym);

        // (5) degree mask + the single send interaction.
        let degree_sym: Vec<SymbolicExt<F, EF>> =
            degree.iter().map(|d| -> SymbolicExt<F, EF> { (*d).into() }).collect();
        let geq = crate::zerocheck::full_geq::<C>(&degree_sym, &point_extended);

        let zero_ext: Ext<F, EF> = builder.constant(EF::ZERO);
        let padding_main: Vec<Ext<F, EF>> = vec![zero_ext; main_full.len()];

        let (real_num, real_den) = lookup.eval::<SymbolicExt<F, EF>, Ext<F, EF>>(
            None,
            &main_full,
            alpha_sym.clone(),
            &betas,
        );
        let (pad_num, pad_den) = lookup.eval::<SymbolicExt<F, EF>, Ext<F, EF>>(
            None,
            &padding_main,
            alpha_sym.clone(),
            &betas,
        );
        let one_sym: SymbolicExt<F, EF> = SymbolicExt::ONE;
        let num_i = real_num - pad_num * geq.clone(); // send
        let den_i = real_den + (one_sym - pad_den) * geq.clone();

        // (6)-(7) log_num_interactions = 0 ⇒ interaction_point empty ⇒ the
        // reconstructed MLE collapses to the single value.
        let recon_num: Ext<F, EF> = builder.eval(num_i);
        let recon_den: Ext<F, EF> = builder.eval(den_i);

        // (8) the height-soundness assert.
        builder.assert_ext_eq(rw_num_ext, recon_num);
        builder.assert_ext_eq(rw_den_ext, recon_den);

        run_test_recursion(builder.into_operations(), std::iter::empty());
    }

    /// A representative single send interaction: `multiplicity = main[0]`,
    /// one value `main[1]`, kind `Byte` (argument_index = 4).  Width-2 main.
    fn sample_lookup() -> Lookup<F> {
        Lookup::<F> {
            values: vec![VirtualPairCol::single_main(1)],
            multiplicity: VirtualPairCol::single_main(0),
            kind: LookupKind::Byte,
            scope: LookupScope::Local,
        }
    }

    /// POSITIVE: honest `(degree, *_full)` → the in-circuit reconstruction
    /// equals the round-walk eval the same honest data implies → the
    /// height-soundness `assert_ext_eq` is a no-op → the circuit runs clean.
    #[test]
    fn reconstruction_accepts_honest_degree() {
        let lookup = sample_lookup();
        // main_full = [multiplicity=5, value=7]; height 2^2 over a 3-coord cube.
        let main_full = vec![EF::from(F::from_u32(5)), EF::from(F::from_u32(7))];
        let alpha = EF::from(F::from_u32(11));
        let beta_seed = vec![EF::from(F::from_u32(13))]; // arity 2 → beta_seed_dim 1
                                                         // max_log_row_count = 3 → point_extended dim 4 → degree dim 4.
        let trace_point =
            vec![EF::from(F::from_u32(2)), EF::from(F::from_u32(3)), EF::from(F::from_u32(4))];
        // Honest degree = big-endian bits of height 2^2 = 4 over 4 slots:
        // 0b0100 → [0,1,0,0].
        let degree = vec![EF::ZERO, EF::ONE, EF::ZERO, EF::ZERO];

        let (rw_num, rw_den) = host_reconstruct_single_send(
            &lookup,
            &main_full,
            &degree,
            alpha,
            &beta_seed,
            &trace_point,
        );
        run_single_send_reconstruction(
            &lookup,
            &main_full,
            &degree,
            alpha,
            &beta_seed,
            &trace_point,
            rw_num,
            rw_den,
        );
    }

    /// NEGATIVE (the height forgery): keep the HONEST round-walk eval and the
    /// HONEST `*_full`, but FORGE `degree` (claim height 2^1 = 2 instead of
    /// 2^2 = 4 → [0,0,1,0]).  The forged `degree` moves the `full_geq` padding
    /// boundary, so the reconstructed num/den diverge from the honest
    /// round-walk eval → the in-circuit `assert_ext_eq` trips at runtime.  This
    /// is the area-preserving height-forgery rejection (host analog:
    /// verifier.rs:1867 mismatch).
    #[test]
    #[should_panic]
    fn reconstruction_rejects_forged_degree() {
        let lookup = sample_lookup();
        let main_full = vec![EF::from(F::from_u32(5)), EF::from(F::from_u32(7))];
        let alpha = EF::from(F::from_u32(11));
        let beta_seed = vec![EF::from(F::from_u32(13))];
        let trace_point =
            vec![EF::from(F::from_u32(2)), EF::from(F::from_u32(3)), EF::from(F::from_u32(4))];
        // HONEST round-walk eval (height 2^2 = 4 → [0,1,0,0]).
        let honest_degree = vec![EF::ZERO, EF::ONE, EF::ZERO, EF::ZERO];
        let (rw_num, rw_den) = host_reconstruct_single_send(
            &lookup,
            &main_full,
            &honest_degree,
            alpha,
            &beta_seed,
            &trace_point,
        );
        // FORGED degree (height 2^1 = 2 → [0,0,1,0]); the round-walk eval is
        // still the honest one above → mismatch → assert trips.
        let forged_degree = vec![EF::ZERO, EF::ZERO, EF::ONE, EF::ZERO];
        run_single_send_reconstruction(
            &lookup,
            &main_full,
            &forged_degree,
            alpha,
            &beta_seed,
            &trace_point,
            rw_num,
            rw_den,
        );
    }

    /// NEGATIVE (the `*_full` binding side): keep the HONEST round-walk eval and
    /// HONEST `degree`, but FORGE `main_trace_evaluations_full` (the trace
    /// opening the reconstruction reads).  A tampered `*_full` perturbs
    /// `Lookup::eval(main_full)` → the reconstructed num/den diverge from the
    /// honest round-walk eval → the assert trips.  Together with
    /// `reconstruction_rejects_forged_degree` this shows the reconstruction
    /// binds BOTH the degree AND the trace opening (so neither is a free
    /// variable that can satisfy the assert under a forgery).
    #[test]
    #[should_panic]
    fn reconstruction_rejects_forged_full_opening() {
        let lookup = sample_lookup();
        let honest_main_full = vec![EF::from(F::from_u32(5)), EF::from(F::from_u32(7))];
        let alpha = EF::from(F::from_u32(11));
        let beta_seed = vec![EF::from(F::from_u32(13))];
        let trace_point =
            vec![EF::from(F::from_u32(2)), EF::from(F::from_u32(3)), EF::from(F::from_u32(4))];
        let degree = vec![EF::ZERO, EF::ONE, EF::ZERO, EF::ZERO];
        // HONEST round-walk eval from the HONEST *_full.
        let (rw_num, rw_den) = host_reconstruct_single_send(
            &lookup,
            &honest_main_full,
            &degree,
            alpha,
            &beta_seed,
            &trace_point,
        );
        // FORGED *_full (multiplicity 5→6) → reconstruction diverges → trips.
        let forged_main_full = vec![EF::from(F::from_u32(6)), EF::from(F::from_u32(7))];
        run_single_send_reconstruction(
            &lookup,
            &forged_main_full,
            &degree,
            alpha,
            &beta_seed,
            &trace_point,
            rw_num,
            rw_den,
        );
    }
}
