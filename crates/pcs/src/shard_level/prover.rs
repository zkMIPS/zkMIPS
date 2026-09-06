//! Shard-level prover assembly: transcript prologue → LogUp-GKR →
//! zerocheck → bridge observe → jagged-PCS → assemble.

use p3_challenger::CanObserve;
use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::RowMajorMatrix;

use super::shard_proof::{BasefoldShardProof, FoldOrientation};
use crate::air::MachineAir;
use crate::prover::ShardData;
use crate::shard_level::row_gkr::top_level::prove_shard_logup_gkr_rows;
use crate::shard_level::zerocheck_prover::prove_shard_zerocheck;
use crate::{Challenge, Chip, ShardOpenedValues, StarkGenericConfig, Val};

/// Build the shard's BaseFold jagged-PCS commit during the prove pass.
///
/// Runs the BaseFold pre-commit on the supplied (already-materialized)
/// `main_traces`, returns the 8-felt BaseFold digest as the new
/// `main_commitment`, and returns the precomputed commit so the caller
/// threads it into the jagged-PCS opening.  The opening then skips its own
/// commit step and the in-band commit observe, matching the verifier
/// counterpart (`verify_jagged_basefold_no_observe`).  The `main_traces`
/// views are BORROWED and only
/// relabeled to `InnerVal` for the commit build (a zero-copy slice
/// reinterpret) — no trace data is copied or moved, and no ownership
/// round-trips through the return.
pub fn commit_traces<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    // BORROWED views over the shard prover's shared
    // `Arc<Mle>` store (no owned deep copy).  On the device / in-dispatch
    // commit path they are zero-copy relabeled to InnerVal views for the
    // commit hook / host fallback.
    main_traces: &[crate::multilinear::PaddedMle<Val<SC>>],
    // The per-shard rev(zeta) orientation
    // (from `StarkMachine::core_rev()`).  Threaded to the host-fallback
    // precompute (dense materialize) and FORCED onto the built
    // `PrecomputedJaggedCommit.rev` so the reduction stays in lockstep — covers
    // BOTH the device-hook and host-fallback build branches.
    use_rev: bool,
) -> (
    [Val<SC>; 8],
    crate::jagged_pcs::jagged::PrecomputedJaggedCommitGeneric<<SC as crate::BasefoldRing>::BfMmcs>,
)
where
    SC: StarkGenericConfig + crate::BasefoldRing,
    A: MachineAir<Val<SC>>,
    Val<SC>: PrimeField + 'static,
    Challenge<SC>: ExtensionField<Val<SC>> + 'static,
    SC::Challenger: 'static,
{
    use crate::{BasefoldRing, InnerChallenge, InnerVal};
    use core::any::TypeId;

    // The BaseFold commit is built HERE, during the prove pass, and is returned
    // UNCONDITIONALLY.  Do not reintroduce an early return: the verifier always
    // uses `verify_jagged_basefold_no_observe`, so a path that skipped the commit
    // would leave the prover observing it in-band -- a transcript desync a green
    // test suite cannot see.
    //
    // Both rings have Val == InnerVal (KoalaBear) and Challenge == InnerChallenge
    // (KoalaBear^4) -- the identities the `named_inner` relabel below relies on.
    // This is a REAL assert, not a `debug_assert!`: it is the only thing standing
    // between a non-KoalaBear config and the `from_raw_parts` / `transmute_copy`
    // reinterprets below, and a `debug_assert!` compiles out in release, which is
    // exactly where that would be UB.  Cost is one TypeId compare per shard.
    assert!(
        TypeId::of::<Val<SC>>() == TypeId::of::<InnerVal>()
            && TypeId::of::<Challenge<SC>>() == TypeId::of::<InnerChallenge>(),
        "commit_traces: requires Val==KoalaBear / \
         Challenge==KoalaBear^4 (shared by inner + outer rings)",
    );
    // Ring discriminator: the INNER ring (core/compress/shrink) uses the
    // Poseidon2-KoalaBear `JaggedChallenger`; the OUTER/wrap ring uses the BN254
    // `OuterChallenger` (and `BfMmcs = OuterValMmcs`).  The inner ring routes the
    // commit through the `commit_multilinears` device seam (so a `StarkGpuProver`
    // override is picked up); the outer ring — always host (the wrap never runs
    // on the GPU) — builds via the `BasefoldRing::commit_multilinears`
    // trait method (which encapsulates the ring-generic `SC::BfMmcs` bounds).
    let is_inner =
        TypeId::of::<SC::Challenger>() == TypeId::of::<crate::jagged_pcs::JaggedChallenger>();

    // Build named InnerVal VIEWS by a zero-copy slice relabel of each borrowed
    // Val<SC> view (Val<SC> == InnerVal under the TypeId gate; identical
    // layout, no copy).  These views borrow the same shared `Arc<Mle>` cells
    // as `main_traces`, so they live as long as the `'t` borrow.
    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace.
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(chips.len(), main_traces.len(), "commit_traces: chips/main_traces must be parallel",);
    let named_inner: alloc::vec::Vec<crate::jagged_pcs::jagged::ChipTraceView> = chips
        .iter()
        .zip(main_traces.iter())
        .map(|(chip, pm)| {
            let name = chip.name().to_string();
            // SAFETY: `Val<SC> == InnerVal` under the assert above, so
            // `PaddedMle<Val<SC>>` and `PaddedMle<InnerVal>` are the SAME type
            // and this is a no-op relabel.  The clone is an `Arc` refcount
            // bump, not a copy of the trace.
            let pm_inner: crate::multilinear::PaddedMle<InnerVal> = unsafe {
                core::mem::transmute_copy::<
                    crate::multilinear::PaddedMle<Val<SC>>,
                    crate::multilinear::PaddedMle<InnerVal>,
                >(&core::mem::ManuallyDrop::new(pm.clone()))
            };
            (name, pm_inner)
        })
        .collect();

    let (main_commitment, precomputed_generic): (
        [Val<SC>; 8],
        crate::jagged_pcs::jagged::PrecomputedJaggedCommitGeneric<
            <SC as crate::BasefoldRing>::BfMmcs,
        >,
    ) = if is_inner {
        // ── INNER ring ───────────────────────────────────────────────────
        // Single shard-wide commit buffer, built by the host precompute over
        // the inner ring's `BfMmcs`.
        let mut precomputed =
            <crate::koala_bear_poseidon2::KoalaBearPoseidon2 as BasefoldRing>::commit_multilinears(
                &named_inner,
                use_rev,
            );
        // Record the per-shard orientation on the built commit.  The producer
        // builds its dense under this SAME `use_rev` but may not stamp the field,
        // and an unstamped `false` is indistinguishable from a deliberate
        // `false`, so this is an unconditional overwrite rather than a check.
        // NOTE the cost of that: if a producer ever built under a DIFFERENT
        // orientation, this would stamp the expected value over the actual one
        // and turn a detectable mismatch into a wrong proof.  Making the producer
        // stamp it (and asserting here) is the fix if that ever becomes possible.
        precomputed.rev = use_rev;
        // FORCE the recursion AREA PIN onto the
        // built commit (the device hook pins `log_dense_size` device-side under
        // the SAME value, but may not stamp the field) so the OPEN-path
        // jagged-eval half reads it back in lockstep.
        let raw_root_inner: [InnerVal; 8] =
            crate::jagged_pcs::basefold_commit_digest(&precomputed.commit);

        // ── jagged HASH-BIND (inner ring only) ────────────
        // Tie the per-chip (row_count, column_count) geometry to the commitment:
        //   modified = compress([raw_root, hash(once(len) ++ row_counts ++ col_counts)])
        // The Fiat-Shamir transcript observes `modified` (set as `main_commitment`
        // below); the BaseFold opening still binds against `raw_root`, carried to
        // the recursion lift via `BasefoldShardProof::jagged_original_commitment`.
        let digest_inner: [InnerVal; 8] = crate::jagged_pcs::jagged_hash_bind_from_jagged_packing(
            raw_root_inner,
            &precomputed.packing,
        );
        // SAFETY: [InnerVal; 8] == [Val<SC>; 8] under the TypeId gate.
        let main_commitment: [Val<SC>; 8] =
            unsafe { core::mem::transmute_copy::<[InnerVal; 8], [Val<SC>; 8]>(&digest_inner) };

        // Inner build path: SC::BfMmcs == JaggedMmcs, so the concrete
        // PrecomputedJaggedCommit IS PrecomputedJaggedCommitGeneric<SC::BfMmcs>.
        let precomputed_generic: crate::jagged_pcs::jagged::PrecomputedJaggedCommitGeneric<
            <SC as crate::BasefoldRing>::BfMmcs,
        > = {
            let any: Box<dyn core::any::Any> = Box::new(precomputed);
            *any.downcast().unwrap_or_else(|_| {
                panic!(
                    "commit_traces: inner build path produces a \
                     JaggedMmcs precompute == SC::BfMmcs"
                )
            })
        };
        (main_commitment, precomputed_generic)
    } else {
        // ── OUTER/wrap ring (BN254 OuterValMmcs) ───────────────────────
        // Build the ring-native BaseFold precompute via the `BasefoldRing`
        // trait method, INLINE during the prove pass.  The returned commit
        // already stamps `rev`.
        let precomputed_generic = <SC as BasefoldRing>::commit_multilinears(&named_inner, use_rev);
        // Ring-generic digest: NO jagged hash-bind on the outer ring (the
        // BN254 wrap re-binds in its registered hook).
        let digest_jv: [crate::jagged_pcs::JaggedVal; 8] =
            <SC as BasefoldRing>::digest_felts(&precomputed_generic.commit.original_commitment);
        // SAFETY: [JaggedVal; 8] == [Val<SC>; 8] (JaggedVal == KoalaBear == Val<SC>).
        let main_commitment: [Val<SC>; 8] = unsafe {
            core::mem::transmute_copy::<[crate::jagged_pcs::JaggedVal; 8], [Val<SC>; 8]>(&digest_jv)
        };
        (main_commitment, precomputed_generic)
    };

    // The borrowed `main_traces` views stay with the caller (`named_inner`
    // only relabeled them to InnerVal for the commit build).  They still
    // borrow the shared `Arc<Mle>` store for the open.
    drop(named_inner);

    (main_commitment, precomputed_generic)
}

/// The shard-level BaseFold producer: transcript prologue -> LogUp-GKR ->
/// zerocheck -> jagged-PCS open -> assemble, over host-resident traces.
///
/// `machine` supplies the two per-stage discriminators the body needs — the
/// per-shard rev(zeta) orientation (`core_rev`) and the recursion-layer area
/// pin — and nothing else; the shard's chips, traces, public values and
/// precomputed commits all ride on `data`.
///
/// A device-native driver reproduces this same sequence against its own
/// resident traces; the two must stay in lockstep, since both emit the same
/// proof bytes for the same shard.
#[allow(clippy::too_many_arguments)]
pub fn prove_shard_with_data<SC, A>(
    machine: &crate::StarkMachine<SC, A>,
    data: crate::prover::ShardData<'_, SC, A>,
    challenger: &mut SC::Challenger,
) -> BasefoldShardProof<Val<SC>, Challenge<SC>>
where
    SC: StarkGenericConfig + crate::BasefoldRing,
    A: MachineAir<Val<SC>> + crate::shard_level::basefold_constraint_folder::ShardProvableAir<SC>,
    Val<SC>: PrimeField + 'static,
    Challenge<SC>: ExtensionField<Val<SC>> + 'static,
    SC::Challenger:
        'static
            + p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + CanObserve<
                <<SC as crate::BasefoldRing>::BfMmcs as p3_commit::Mmcs<
                    crate::jagged_pcs::JaggedVal,
                >>::Commitment,
            >,
{
    let ShardData {
        chips,
        preprocessed_traces,
        preprocessed_commit_data,
        main_traces,
        public_values,
        commit_data,
    } = data;
    // Sourced from `self`/traces:
    //   * `orientation` — CpuProver default emits MSB-folded proofs (it ONLY
    //     sets the proof envelope's `fold_orientation` field; no transcript
    //     effect).  A `StarkGpuProver` overrides this whole method and
    //     supplies its own orientation.
    //   * `dense_rev` — the per-shard rev(zeta) orientation, from the
    //     per-stage source of truth `StarkMachine::core_rev()`.
    //   * `max_log_row_count` — the FIXED config cube.  The
    //     construction site padded every entry to it, so
    //     `num_variables()` on any entry must agree — asserted below.
    let orientation = crate::shard_level::shard_proof::FoldOrientation::Msb;
    let dense_rev = machine.core_rev();
    // The FIXED config cube.  Every `PaddedMle` in the map was built AT
    // this constant (both the `padded_with_zeros` host chips and the
    // `dummy` width-0 chips), so each entry must report it — asserted in
    // debug builds.
    let max_log_row_count =
        crate::shard_level::verifier::BasefoldShardVerifier::production_default().max_log_row_count;
    debug_assert!(
        main_traces.values().all(|pm| pm.num_variables() as usize == max_log_row_count),
        "prove_shard_with_data: main_traces padded to a cube != the fixed \
         max_log_row_count {max_log_row_count}",
    );
    // The shared analytic trace-MLE store — the SINGLE authoritative host
    // main-trace store — is built ONCE at the construction site and handed
    // over ready-made on `data.main_traces`.
    // Re-key the name-ordered map onto the chip-INDEX order the loader and
    // every downstream stage expect (they zip `chips` with this slice).
    // `chips` is itself in name order — it comes from
    // `shard_chips_ordered(chip_ordering)` and `chip_ordering` is built
    // from the name-order-sorted commit — so this lookup is
    // order-preserving.  Cloning a `PaddedMle` clones an `Arc<Mle>` + a
    // small `Padding`, so the trace cells are never deep-copied.
    let shared_trace_mles_vec: Vec<crate::multilinear::PaddedMle<Val<SC>>> = chips
        .iter()
        .map(|chip| {
            let name = chip.name();
            match main_traces.get(&name) {
                Some(pm) => pm.clone(),
                None => panic!("prove_shard_with_data: chip {name} missing from main_traces",),
            }
        })
        .collect();
    let shared_trace_mles: &[crate::multilinear::PaddedMle<Val<SC>>] =
        shared_trace_mles_vec.as_slice();
    // ── The shard body, single-body form (the stage helpers live in
    // shard_level).
    debug_assert_eq!(
        chips.len(),
        shared_trace_mles.len(),
        "chips and shared_trace_mles must be parallel arrays",
    );

    // `shared_trace_mles` is the single authoritative host main-trace store
    // (all chips, chip-index order); every stage below reads it directly, so
    // handing the slice down costs a refcount, not a copy.
    //
    // Commit: consume-or-build.  This body is the host CpuProver path ONLY —
    // the GPU pipeline assembles the shard stages device-natively in
    // ziren-gpu and overrides this method.  When `commit_data` is `None`,
    // `commit_traces` builds the BaseFold commit here and the jagged open
    // consumes it with the in-band commit observe SKIPPED.  That skip is
    // load-bearing: the verifier always uses
    // `verify_jagged_basefold_no_observe`, so an in-band observe on the
    // prover side would be a transcript desync.
    //
    // Every chip is host-resident here, so the device-residency parameters
    // the shared helpers accept are inert (`chip_cum_tails` all-`None` —
    // cumulative sums read raw host cells); the live device-remat logic
    // lives in ziren-gpu's `shard_helpers` feeding these same helpers.
    //
    // HEIGHT-AGNOSTIC RECURSION: present chips commit at their NATURAL raw
    // height, so packing offsets == degree heights == the in-circuit raw
    // col_prefix_sums reconstruction; missing (injected) chips pack at band
    // height (see the injection in `CpuProver::commit`) to preserve the
    // chip-SET and hence the vk.
    let trace_views: Vec<crate::multilinear::PaddedMle<Val<SC>>> = shared_trace_mles.to_vec();
    let chip_cum_tails: Vec<Option<Vec<Val<SC>>>> = chips.iter().map(|_| None).collect();
    let n_chips = chips.len();
    let _shard_span = tracing::info_span!("prove shard with data", chips = n_chips).entered();

    let (main_commitment, precomputed_commit) = {
        let _span = tracing::info_span!("commit traces").entered();
        match commit_data {
            // `commit()` already built and retained the jagged commitment —
            // consume it.  The digest and precompute are the identical values
            // that build would have produced (same seam, same inputs, one
            // shard-phase earlier).
            Some(retained) => (retained.main_commitment, retained.precomputed),
            None => commit_traces::<SC, A>(chips, &trace_views, dense_rev),
        }
    };
    // `trace_views` is kept OWNED (no reborrow): the dims sites below
    // borrow it, and the jagged open MOVES it in so its per-chip
    // cells become the open's `chip_traces` with NO clone.

    // Transcript prologue. Chip metadata observe (count +
    // per-chip RAW height + name length + name bytes) binds post-
    // commit challenges to the shard's chip-set identity AND each
    // chip's row count.
    //
    // The per-chip height felt is the RAW `num_real_entries`
    // (0 allowed) — the value the recursion verifier binds in
    // this slot via the `chip_height_bits` Horner recompose.  The host
    // verifier mirror in `shard_level::verifier::verify_shard_basefold`
    // observes the same value sourced from `proof.chip_heights`.
    //
    // Observe order (the verifiers replay it exactly):
    //   public_values → main_commitment → num_chips →
    //   per-chip { height_felt, name_len, name_bytes }
    {
        // The prologue observes live in a pub helper so the
        // device-native drivers reproduce the EXACT Fiat-Shamir prologue
        // (order unchanged).
        observe_transcript_prologue::<SC, A>(
            challenger,
            &public_values,
            &main_commitment,
            chips,
            shared_trace_mles,
        );
    }
    // LogUp-GKR.
    let _t_logup_gkr = std::time::Instant::now();
    let logup_gkr_proof = {
        let _span = tracing::info_span!("logup gkr proof").entered();
        prove_shard_logup_gkr_rows::<Val<SC>, Challenge<SC>, A, SC::Challenger>(
            chips,
            preprocessed_traces,
            max_log_row_count,
            challenger,
            // The shared per-chip trace-MLE built once above (covers ALL
            // chips) — the SOLE host main-trace source for this stage.
            shared_trace_mles,
        )
    };
    tracing::info!(
        elapsed_ms = _t_logup_gkr.elapsed().as_millis() as u64,
        chips = n_chips,
        phase = "logup_gkr",
        "shard phase done"
    );

    // Per-chip zerocheck.  Takes the LogUp-GKR
    // evaluations so each chip's sumcheck claim chains to its GKR
    // openings (`claimed_sum = λ-RLC(Σ openings·β^k)`), eq-anchored at
    // the shared GKR point.
    let _t_zerocheck = std::time::Instant::now();
    // The per-chip constraint-batching challenge and the GKR-opening batch
    // challenge are squeezed here, between the two arguments, so the
    // zerocheck span times the argument and not the transcript draws.
    // Order is load-bearing: alpha -> gkr_batch_open, then `lambda` inside.
    let (alpha, gkr_batch_open) =
        crate::shard_level::zerocheck_prover::sample_zerocheck_batching_challenges::<SC>(
            challenger,
        );

    let (zerocheck_proof, trace_at_z) = {
        let _span = tracing::info_span!("zerocheck").entered();
        let (zerocheck_proof, trace_at_z) = prove_shard_zerocheck::<SC, A>(
            chips,
            preprocessed_traces,
            &public_values,
            alpha,
            gkr_batch_open,
            &logup_gkr_proof.logup_evaluations,
            max_log_row_count,
            challenger,
            // The shared per-chip trace-MLE built once above (covers ALL
            // chips) — the SOLE host main-trace source for this stage.
            shared_trace_mles,
            // The per-shard rev(zeta) orientation.
            dense_rev,
        );

        // Observe slot 2 — the zerocheck openings (trace@z*), observed after
        // the zerocheck sumcheck and BEFORE the jagged phase.  Slot 1 (the
        // GKR openings, trace@ζ) is emitted at the end of the GKR phase
        // (`row_gkr::top_level::prove_shard_logup_gkr_rows`); see
        // `observe_logup_gkr_openings` for why the ordering is load-bearing.
        //
        // `num_chips` felt, then per chip the length-prefixed
        // preprocessed-then-main openings in chip-NAME order — the order the
        // recursion verifier and the host verifier replay.
        observe_zerocheck_openings_from_residual::<SC, A>(challenger, chips, &trace_at_z);

        (zerocheck_proof, trace_at_z)
    };
    tracing::info!(
        elapsed_ms = _t_zerocheck.elapsed().as_millis() as u64,
        chips = n_chips,
        phase = "zerocheck",
        "shard phase done"
    );

    // ── Openings-for-free: reuse the zerocheck residual as the
    // jagged step-3 y_per_chip ────────────────────────────────────────────
    // `trace_at_z[name]` is the zerocheck reduction's component_poly_evals
    // (prep-then-main per chip, = padded-MLE_BE(bitrev(trace)) @ z) — exactly
    // the per-column values jagged step (3) would recompute from the trace.
    // Passing the main slice as pre_y_per_chip skips the host triple-nested
    // step-3 reduction; the proof bytes are unchanged (identical values, and
    // step 3 is transcript-silent).
    // Per-chip metadata HEIGHT for the two jagged-open sites that branch on an
    // EMPTY commit trace (`compute_residual_y_openings` + the jagged-eval
    // producer) and so cannot reach `shared_trace_mles` directly.  A
    // device-resident chip (dummy, `inner` None) carries its baked height
    // here; a host chip maps to `None` (its height comes from the non-empty
    // trace, so this slot is never read).
    let open_heights: Vec<Option<usize>> = shared_trace_mles
        .iter()
        .map(|pm| if pm.inner().is_none() { pm.metadata_height() } else { None })
        .collect();

    // ── The PREPROCESSED round (the first opening round) ──────────────────
    //
    // Its chip set, ORDER and dims come from the commit itself
    // (`packing.chip_infos`), which is authoritative: `setup` sorted the
    // preprocessed traces by NAME and committed them in that
    // order.  Reading the order off
    // the commit means the round can never disagree with what was committed.
    //
    // A machine with no preprocessed traces yields an empty round set and a
    // single (main-only) round downstream.
    let prep_chip_infos = &preprocessed_commit_data.packing.chip_infos;
    let mut preprocessed_named: Vec<(String, crate::multilinear::PaddedMle<Val<SC>>)> =
        Vec::with_capacity(prep_chip_infos.len());
    let mut preprocessed_claims: Vec<Vec<Challenge<SC>>> =
        Vec::with_capacity(prep_chip_infos.len());
    for info in prep_chip_infos.iter() {
        let idx = chips
            .iter()
            .position(|c| MachineAir::<Val<SC>>::name(*c) == info.name)
            .unwrap_or_else(|| {
                panic!(
                    "preprocessed round: committed chip {} is absent from the shard's \
                 chip set — the proving key and the shard disagree",
                    info.name,
                )
            });
        preprocessed_named.push((info.name.clone(), preprocessed_traces[idx].clone()));
        // This chip's PREPROCESSED columns at z are the PREFIX of its zerocheck
        // residual (`preprocessed.local ++ main.local`, split by
        // `preprocessed_width` — see the opened-values builder).  They are
        // already computed; the round proves them against the vk's commitment.
        let evals = trace_at_z.get(&info.name).unwrap_or_else(|| {
            panic!("preprocessed round: chip {} has no zerocheck residual", info.name)
        });
        assert!(
            evals.len() >= info.column_count,
            "preprocessed round: chip {} residual is {} wide but the commit has {} \
         preprocessed columns",
            info.name,
            evals.len(),
            info.column_count,
        );
        preprocessed_claims.push(evals[..info.column_count].to_vec());
    }

    let residual_y: Vec<Vec<Challenge<SC>>> = compute_residual_y_openings::<SC, A>(
        chips,
        &trace_views,
        preprocessed_traces,
        &trace_at_z,
        &logup_gkr_proof.logup_evaluations,
        &open_heights,
        dense_rev,
    );

    // Jagged-PCS opening (prove evaluation claims). Per-chip `r_row` is the trailing
    // log(chip_height) coords of the LogUp-GKR final eval_point.
    let _t_prove_eval_claims = std::time::Instant::now();
    let evaluation_proof = {
        let _span = tracing::info_span!("prove evaluation claims").entered();
        crate::shard_level::prover::prove_trusted_evaluations::<SC, A>(
            chips,
            // The PREPROCESSED round: its traces (in the order `setup`
            // committed them), its claims, and the proving key's commit.
            &preprocessed_named,
            preprocessed_claims,
            preprocessed_commit_data,
            // Commit-coverage trace set (BORROWED views over the shared
            // `Arc<Mle>` store) — MUST be the same traces the precompute
            // committed, or the openings won't bind.
            &trace_views,
            // Open jagged at the zerocheck-reduced z*.
            &zerocheck_proof.point_and_eval.0,
            challenger,
            precomputed_commit,
            residual_y,
            // Every chip is host-resident on this body, so no chip needs a
            // metadata height: each one's height comes from its non-empty
            // commit trace.
            &[],
        )
    };
    tracing::info!(
        elapsed_ms = _t_prove_eval_claims.elapsed().as_millis() as u64,
        chips = n_chips,
        phase = "prove_evaluation_claims",
        "shard phase done"
    );

    // Shard-proof assembly.

    // Per-chip RAW-height map (usize), device-residency aware.  Stored on
    // the proof as `chip_heights` (the felt the prologue observed) AND
    // feeds the `opened_values` degree-bit decomposition below.
    // MUST agree with the prologue observe + the verifier.
    let chip_heights = build_chip_heights::<SC, A>(chips, shared_trace_mles);

    // Populate `opened_values` with the per-chip trace@z openings from the
    // zerocheck reduction (the values the recursion zerocheck verifier
    // batches/constrains at the reduced point z and asserts equal
    // `point_and_eval.1`).  `trace_at_z` is keyed by chip name and is
    // prep-then-main per chip; split at the chip's `preprocessed_width` to
    // recover `preprocessed.local` / `main.local`.  Chips are emitted in NAME
    // order to match the recursion `opened_values.chips` BTreeMap key-order
    // iteration.  The REAL-height big-endian degree bits ride in the
    // `quotient` slot.
    let opened_values =
        build_opened_values::<SC, A>(chips, trace_at_z, &chip_heights, max_log_row_count);

    // Per-chip (local, global) cumulative sums.  `local` is ZERO (the
    // basefold path doesn't materialize the permutation trace); `global`
    // reads the RAW per-chip cells (device chips use the early TAIL).
    let chip_cumulative_sums =
        build_chip_cumulative_sums::<SC, A>(chips, shared_trace_mles, &chip_cum_tails);

    // The final `BasefoldShardProof` construction — including the witnessed
    // row/padding-column counts + the raw BaseFold root
    // (`jagged_original_commitment`), both derived from `evaluation_proof`.
    let proof = assemble_basefold_shard_proof::<SC>(
        public_values,
        main_commitment,
        logup_gkr_proof,
        zerocheck_proof,
        opened_values,
        chip_heights,
        chip_cumulative_sums,
        evaluation_proof,
        orientation,
    );
    proof
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared NON-DEVICE shard-driver orchestration helpers, `pub` so the
// ziren-gpu device-native drivers reuse them instead of duplicating.  Each
// helper's operation order + observe/sample sequence must stay in lockstep
// with the inline driver above (the drivers must emit byte-identical proofs).
// ═══════════════════════════════════════════════════════════════════════════

/// Per-chip RAW height for the transcript prologue + the proof's
/// `chip_heights` map (the observed felt is the raw
/// `num_real_entries`, 0 allowed for an unexercised chip — NOT its
/// ceil-log2).  Device-residency aware: a device chip's REAL height is
/// baked into its dummy MLE (`metadata_height()`), floored at 1 — the
/// device dummy's VirtualGeq floor.  This is the SINGLE source for the
/// observe, the proof map and the degree-bit decomposition; deriving any
/// of them separately risks a transcript-vs-proof desync.
#[inline]
pub fn raw_chip_height<F: p3_field::Field>(pm: &crate::multilinear::PaddedMle<F>) -> usize {
    if pm.inner().is_some() {
        pm.num_real_entries()
    } else {
        pm.metadata_height().unwrap_or(0).max(1)
    }
}

/// ceil(log2(h)) with `ceil_log2(0) == 0` — the shared geometry
/// derivation for consumers that need a LOG height (shape lookups,
/// `log_degree`) from the RAW `chip_heights` value.  Kept separate from
/// the transcript felt: the prologue observes the RAW height, never this.
#[inline]
pub fn ceil_log2(h: usize) -> usize {
    if h <= 1 {
        0
    } else if h.is_power_of_two() {
        h.trailing_zeros() as usize
    } else {
        (usize::BITS - h.leading_zeros()) as usize
    }
}

/// Stage-1 transcript prologue.  Observes, in order:
/// `public_values → main_commitment → num_chips → per-chip {height_felt,
/// name_len, name_bytes}`.  These are the ONLY challenger writes of the
/// prologue; the device-native drivers call this to reproduce the exact
/// Fiat-Shamir binding of the shard's chip-set identity + per-chip row count.
pub fn observe_transcript_prologue<SC, A>(
    challenger: &mut SC::Challenger,
    public_values: &[Val<SC>],
    main_commitment: &[Val<SC>; 8],
    chips: &[&Chip<Val<SC>, A>],
    shared_trace_mles: &[crate::multilinear::PaddedMle<Val<SC>>],
) where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
{
    for &pv in public_values.iter() {
        challenger.observe(pv);
    }
    for &c in main_commitment.iter() {
        challenger.observe(c);
    }
    let num_chips = Val::<SC>::from_u64(chips.len() as u64);
    challenger.observe(num_chips);
    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace.
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(
        chips.len(),
        shared_trace_mles.len(),
        "observe_transcript_prologue: chips/shared_trace_mles must be parallel",
    );
    for (chip, pm) in chips.iter().zip(shared_trace_mles.iter()) {
        // Per-chip RAW-height observe (the observed felt is
        // `num_real_entries` directly, a true 0 for an unexercised chip;
        // the previous ceil-log2 felt with a `.max(1)` floor is retired).
        // Source matches the proof's `chip_heights` map
        // (`build_chip_heights`) + the verifier re-observe + the recursion
        // circuit's Horner recompose of the witnessed degree bits — the
        // FOUR mirrors observe this exact value.
        let h = raw_chip_height(pm);
        challenger.observe(Val::<SC>::from_u64(h as u64));

        // Name length + name bytes.
        let name_bytes = chip.name();
        let len_felt = Val::<SC>::from_u64(name_bytes.len() as u64);
        challenger.observe(len_felt);
        for byte in name_bytes.bytes() {
            challenger.observe(Val::<SC>::from_u64(byte as u64));
        }
    }
}

/// Observe a LENGTH-PREFIXED extension-field slice.
///
/// Observes the element COUNT as one felt, then each element's basis
/// coefficients.  The prefix is what removes the parsing ambiguity between
/// two adjacent opening slices — without it, a prover free to move a column
/// between two chips (or between the preprocessed and main halves of one
/// chip) reaches the same transcript state from different opening vectors.
pub fn observe_length_prefixed_ext<F, EF, Challenger>(challenger: &mut Challenger, data: &[EF])
where
    F: p3_field::PrimeField,
    EF: BasedVectorSpace<F>,
    Challenger: p3_challenger::FieldChallenger<F>,
{
    challenger.observe(F::from_u64(data.len() as u64));
    for v in data.iter() {
        for basis in v.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }
}

/// Observe slot **1** — the LogUp-GKR trace openings (trace@ζ).
///
/// Observed INSIDE the GKR phase, immediately after the round walk produces
/// the terminal evaluation point and before the shard driver samples any
/// zerocheck challenge (α/γ/λ): the `chips.len()` felt, then per chip the
/// length-prefixed preprocessed and main openings.
///
/// # Why the position is load-bearing
///
/// The zerocheck identity the verifier enforces is
///
/// ```text
///   Σ_i λ^i Σ_k γ^k O_{i,k}  =  Σ_i λ^i [ C̃_{α,i}(anchor) + Σ_k γ^k T̃_{i,k}(anchor) ]
/// ```
///
/// (left side: `claimed_sum` re-derived from these openings, host
/// `verifier.rs` step G2-b; right side: what the zerocheck sumcheck forces).
/// With `O` fixed BEFORE α/γ/λ this is a Schwartz–Zippel test of a nonzero
/// polynomial in those challenges, so it forces both `O = T̃(anchor)` and a
/// vanishing constraint sum.  With α/γ/λ sampled first it collapses to ONE
/// linear equation in `|O|` unknowns, which a prover can solve for `O` —
/// absorbing a constraint violation `Σ_i λ^i C̃_{α,i}` into the opening vector.
/// The only other binding on `O` is the LogUp last-layer reconstruction, which
/// is two scalar equations (`verifier.rs`, the numerator/denominator
/// mismatch returns) and touches only the columns that appear in an
/// interaction expression.
///
/// # What is observed
///
/// TWO opening sets per chip: the legacy trailing-`log_h`
/// `main_trace_evaluations` (which drives the claim on the recursion /
/// shrink / wrap stages) and the full-point `main_trace_evaluations_full`
/// (which drives it on the core stage).  Both are observed unconditionally so
/// the binding does not depend on which stage's convention is in force; each
/// is length-prefixed, so the four slices stay unambiguous.  Chip order is
/// NAME order (`chip_openings` is a `BTreeMap`).
pub fn observe_logup_gkr_openings<F, EF, Challenger>(
    challenger: &mut Challenger,
    num_chips: usize,
    logup_evaluations: &crate::shard_level::types::LogUpEvaluations<EF>,
) where
    F: p3_field::PrimeField,
    EF: BasedVectorSpace<F>,
    Challenger: p3_challenger::FieldChallenger<F>,
{
    challenger.observe(F::from_u64(num_chips as u64));
    for (_name, opening) in logup_evaluations.chip_openings.iter() {
        observe_length_prefixed_ext::<F, EF, Challenger>(
            challenger,
            opening.preprocessed_trace_evaluations_full.as_deref().unwrap_or(&[]),
        );
        observe_length_prefixed_ext::<F, EF, Challenger>(
            challenger,
            opening.main_trace_evaluations_full.as_deref().unwrap_or(&[]),
        );
    }
}

/// Observe slot **2** — the zerocheck openings (trace@z\*).
///
/// Observes the sumcheck's `component_poly_evals` right after the zerocheck
/// sumcheck returns and before the jagged phase — the `airs.len()` felt, then
/// per chip the length-prefixed preprocessed and main openings — so the
/// jagged phase's challenges are sampled with the openings they are meant to
/// be opening already bound.
///
/// `per_chip` yields `(preprocessed@z*, main@z*)` per chip in NAME order —
/// on the prover from the zerocheck residual `trace_at_z` split at each chip's
/// `preprocessed_width`, on the verifier from `opened_values.chips` (which
/// `build_opened_values` emits name-sorted with exactly that split).
pub fn observe_zerocheck_openings<'a, F, EF, Challenger, I>(
    challenger: &mut Challenger,
    num_chips: usize,
    per_chip: I,
) where
    F: p3_field::PrimeField,
    EF: BasedVectorSpace<F> + 'a,
    Challenger: p3_challenger::FieldChallenger<F>,
    I: IntoIterator<Item = (&'a [EF], &'a [EF])>,
{
    challenger.observe(F::from_u64(num_chips as u64));
    for (prep, main) in per_chip {
        observe_length_prefixed_ext::<F, EF, Challenger>(challenger, prep);
        observe_length_prefixed_ext::<F, EF, Challenger>(challenger, main);
    }
}

/// Prover-side adapter for [`observe_zerocheck_openings`]: split the zerocheck
/// residual `trace_at_z` (prep-then-main concatenated per chip) at each chip's
/// `preprocessed_width` and feed the pairs in NAME order — the same split and
/// the same order [`build_opened_values`] uses to build the `opened_values` the
/// verifier observes.
pub fn observe_zerocheck_openings_from_residual<SC, A>(
    challenger: &mut SC::Challenger,
    chips: &[&Chip<Val<SC>, A>],
    trace_at_z: &std::collections::BTreeMap<String, Vec<Challenge<SC>>>,
) where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
    Val<SC>: p3_field::PrimeField,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    let mut name_sorted: Vec<&&Chip<Val<SC>, A>> = chips.iter().collect();
    name_sorted
        .sort_by(|a, b| MachineAir::<Val<SC>>::name(**a).cmp(&MachineAir::<Val<SC>>::name(**b)));
    observe_zerocheck_openings::<Val<SC>, Challenge<SC>, SC::Challenger, _>(
        challenger,
        chips.len(),
        name_sorted.iter().map(|chip| {
            let name = MachineAir::<Val<SC>>::name(**chip);
            let prep_width = MachineAir::<Val<SC>>::preprocessed_width(**chip);
            // The borrow is of `trace_at_z` (a parameter), not of the local
            // `name`, so it outlives the closure body.
            let evals: &[Challenge<SC>] =
                trace_at_z.get(&name).map(|v| v.as_slice()).unwrap_or(&[]);
            let split = prep_width.min(evals.len());
            evals.split_at(split)
        }),
    );
}

/// Splice D2H-rematerialized device traces into the shared host trace store.
///
/// A device-resident chip has an EMPTY host `PaddedMle` (`inner() == None`);
/// when `eager_device_remat` carries a matrix for one, rebuild its `Mle` from
/// those cells.  Every other chip is an `Arc` bump.
///
/// The remat slots are almost always `None`: `compute_skip_device_d2h` is
/// `prospective_log_dense >= ZIREN_GPU_JAGGED_PCS_MIN_LOG_SIZE`, which
/// defaults to 0 and so is always true, leaving the eager D2H for the one case
/// where a device chip has no provider height.  On the pure-host driver every
/// slot is `None` by construction, making this `shared_trace_mles.to_vec()`.
///
/// This function exists only because a device trace cannot yet BE a
/// `PaddedMle<F, CudaBackend>` — see the `MleBaseBackend` work; once it can,
/// there is nothing left to splice.
pub fn splice_device_remat_traces<SC>(
    shared_trace_mles: &[crate::multilinear::PaddedMle<Val<SC>>],
    eager_device_remat: &[Option<RowMajorMatrix<Val<SC>>>],
) -> Vec<crate::multilinear::PaddedMle<Val<SC>>>
where
    SC: StarkGenericConfig,
{
    shared_trace_mles
        .iter()
        .zip(eager_device_remat.iter())
        .map(|(pm, remat)| {
            if pm.inner().is_none() {
                // Device-resident / unexercised chip: wrap the rematerialized
                // side-storage when there is one, else hand back the dummy
                // (which projects to zero area, as the width-0 view did).
                if let Some(m) = remat {
                    let h = if m.width == 0 { 0 } else { m.values.len() / m.width };
                    let log_h = if h <= 1 { 0 } else { h.next_power_of_two().ilog2() };
                    let mle = std::sync::Arc::new(crate::basefold::Mle::from_row_major(
                        RowMajorMatrix::new(m.values.clone(), m.width),
                    ));
                    return crate::multilinear::PaddedMle::padded_with_zeros(mle, log_h);
                }
                return pm.clone();
            }
            // Host chip: an `Arc` refcount bump, no cells touched.
            pm.clone()
        })
        .collect()
}

/// Residual-y reuse: the zerocheck reduction residual (`trace_at_z` main
/// slice) IS the jagged step-3 `y_per_chip`, so the host triple-nested
/// recompute is skipped.  Step 3 is transcript-silent, so the proof bytes are
/// unchanged.  Panics when any chip's residual is missing or
/// shape-mismatched, or on a non-pow2 height under the LEGACY (`!use_rev`)
/// bitrev convention.
pub fn compute_residual_y_openings<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    commit_traces: &[crate::multilinear::PaddedMle<Val<SC>>],
    preprocessed_traces: &[crate::multilinear::PaddedMle<Val<SC>>],
    trace_at_z: &std::collections::BTreeMap<String, Vec<Challenge<SC>>>,
    logup_evaluations: &crate::shard_level::types::LogUpEvaluations<Challenge<SC>>,
    // Per-chip metadata heights, parallel to `chips` (device dummies carry a
    // baked height; host chips `None`).  The sole empty-commit-trace height
    // source.  An empty / short slice (host callers that don't precompute it)
    // tolerates `.get` → falls back to 0 (unexercised).
    heights: &[Option<usize>],
    // The shard's rev(zeta) orientation (`dense_rev`).  Under `use_rev` BOTH
    // the zerocheck residual and the jagged `y_per_chip` read NATURAL rows, so
    // the reuse is valid for ANY height; only the LEGACY (`!use_rev`) bitrev
    // convention needs a power-of-two height.
    use_rev: bool,
) -> Vec<Vec<Challenge<SC>>>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
{
    // The zerocheck residual IS the jagged round's column claims; there is no
    // silent recompute fallback.  Each failure mode is a named panic, because
    // each would be a REAL bug:
    //
    //   * missing full openings — structurally impossible: both producers
    //     (`row_gkr::top_level` and ziren-gpu's `device_logup_gkr`) return
    //     `Some` on every branch, zero-filling a device-only or width-0 chip;
    //   * a non-power-of-two height under the LEGACY (`!use_rev`) bitrev
    //     convention, where the residual's row order would not match the
    //     jagged one;
    //   * a residual whose width does not match the chip — a desync between
    //     the zerocheck and the commit.
    assert!(
        !logup_evaluations.chip_openings.is_empty(),
        "compute_residual_y_openings: LogUp-GKR produced no chip openings",
    );
    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace (a 2N
    // [prep | main] commit set would pair chip[i] with its PREPROCESSED trace
    // where its MAIN trace is expected).
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(
        chips.len(),
        commit_traces.len(),
        "compute_residual_y_openings: chips/commit_traces must be parallel",
    );
    assert_eq!(
        chips.len(),
        preprocessed_traces.len(),
        "compute_residual_y_openings: chips/preprocessed_traces must be parallel",
    );
    let mut out: Vec<Vec<Challenge<SC>>> = Vec::with_capacity(chips.len());
    for (idx, ((chip, ctrace), ptrace)) in
        chips.iter().zip(commit_traces.iter()).zip(preprocessed_traces.iter()).enumerate()
    {
        let name = MachineAir::<Val<SC>>::name(*chip);
        // A device-resident chip carries an EMPTY commit trace; resolve its
        // REAL dims so the residual openings still cover it: height from the
        // dummy's baked metadata (else the provider), width from the residual
        // itself.
        let (ctrace_values, ctrace_width) = crate::jagged::real_cells(ctrace);
        let (w, h) = if ctrace_width == 0 {
            let dev_h = heights.get(idx).copied().flatten().unwrap_or(0);
            let dev_w = trace_at_z
                .get(&name)
                .map(|evals| evals.len().saturating_sub(ptrace.num_polynomials()))
                .unwrap_or(0);
            (dev_w, dev_h)
        } else {
            let w = ctrace_width;
            (w, ctrace_values.len() / w)
        };
        // Mirror the `y_per_chip` guard in jagged_pcs.rs.  A genuine
        // HEIGHT-0 but FULL-WIDTH missing chip must still emit ONE zero column
        // claim PER COLUMN (the verifier k-walk advances through every
        // committed column); a truly width-0 chip skips.
        if w == 0 {
            out.push(Vec::new());
            continue;
        }
        if h == 0 {
            out.push(vec![Challenge::<SC>::ZERO; w]);
            continue;
        }
        assert!(
            use_rev || h.is_power_of_two(),
            "compute_residual_y_openings: chip {name} has height {h}, which is not a \
             power of two, under the LEGACY (use_rev = false) bitrev convention — the \
             zerocheck residual's row order would not match the jagged one",
        );
        // Strict shape check: prep-then-main, main slice is the last `w` values
        // (zerocheck num_main_cols == trace width).
        let prep_cols = ptrace.num_polynomials();
        let evals = trace_at_z.get(&name).unwrap_or_else(|| {
            panic!(
                "compute_residual_y_openings: chip {name} has no zerocheck residual — \
                 the zerocheck and the commit disagree on the chip set",
            )
        });
        assert_eq!(
            evals.len(),
            prep_cols + w,
            "compute_residual_y_openings: chip {name} residual is {} wide but the chip \
             has {prep_cols} preprocessed + {w} main columns",
            evals.len(),
        );
        out.push(evals[prep_cols..].to_vec());
    }
    out
}

/// Per-chip RAW height map (the value stored on the proof as
/// `chip_heights`, observed in the Phase-1 prologue AND the VirtualGeq
/// threshold feeding the `opened_values` degree-bit decomposition),
/// device-residency aware.  MUST agree with the Phase-1 prologue observe +
/// the verifier re-observe — all three read [`raw_chip_height`].
pub fn build_chip_heights<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    shared_trace_mles: &[crate::multilinear::PaddedMle<Val<SC>>],
) -> std::collections::BTreeMap<String, usize>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
{
    let mut chip_heights = std::collections::BTreeMap::new();
    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace.
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(
        chips.len(),
        shared_trace_mles.len(),
        "build_chip_heights: chips/shared_trace_mles must be parallel",
    );
    for (chip, pm) in chips.iter().zip(shared_trace_mles.iter()) {
        // Device residency: a device chip's REAL height is baked into its
        // dummy MLE, read back via `metadata_height()` with the `.max(1)`
        // dummy floor; a MISSING canonical-cluster HOST chip is a genuine
        // 0-row matrix (raw 0 => all-zero degree bits).
        let h = raw_chip_height(pm);
        let name = MachineAir::<Val<SC>>::name(*chip);
        chip_heights.insert(name, h);
    }
    chip_heights
}

/// Per-chip trace@z opened values, emitted in chip-NAME order (matching the
/// recursion `opened_values.chips` BTreeMap key order).
/// `trace_at_z` is prep-then-main per chip; the REAL height's big-endian bit
/// decomposition is carried via the `quotient` slot for the recursion
/// `full_geq` degree.
pub fn build_opened_values<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    mut trace_at_z: std::collections::BTreeMap<String, Vec<Challenge<SC>>>,
    chip_heights: &std::collections::BTreeMap<String, usize>,
    max_log_row_count: usize,
) -> ShardOpenedValues<Val<SC>, Challenge<SC>>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
{
    let mut name_sorted: Vec<&&Chip<Val<SC>, A>> = chips.iter().collect();
    name_sorted
        .sort_by(|a, b| MachineAir::<Val<SC>>::name(**a).cmp(&MachineAir::<Val<SC>>::name(**b)));
    let chip_opened: Vec<crate::types::ChipOpenedValues<Val<SC>, Challenge<SC>>> = name_sorted
        .iter()
        .map(|chip| {
            let name = MachineAir::<Val<SC>>::name(**chip);
            let prep_width = MachineAir::<Val<SC>>::preprocessed_width(**chip);
            // MOVE the chip's residual out of the map and split it IN
            // PLACE: `remove` + `split_off` transfer ownership, so neither
            // the preprocessed nor the main opening copies its cells.
            let mut prep_local: Vec<Challenge<SC>> = trace_at_z.remove(&name).unwrap_or_default();
            let split = prep_width.min(prep_local.len());
            let main_local = prep_local.split_off(split);
            // big-endian bit decomposition of the REAL height (the
            // VirtualGeq threshold).  bit_len = max_log_row_count + 1.
            // `log_degree` is DERIVED geometry (ceil-log2 of the raw
            // height) — the transcript observes the RAW height, not this.
            let height = *chip_heights.get(&name).unwrap_or(&0);
            let log_degree = ceil_log2(height);
            let bit_len = max_log_row_count + 1;
            let degree_bits: Vec<Challenge<SC>> = (0..bit_len)
                .map(|i| {
                    // BIG-ENDIAN (MSB at index 0): the verifier shape
                    // asserts degree[0] = MSB.
                    let shift = bit_len - 1 - i;
                    let bit = if shift < usize::BITS as usize { (height >> shift) & 1 } else { 0 };
                    if bit == 1 {
                        Challenge::<SC>::ONE
                    } else {
                        Challenge::<SC>::ZERO
                    }
                })
                .collect();
            crate::types::ChipOpenedValues {
                preprocessed: crate::types::AirOpenedValues { local: prep_local, next: Vec::new() },
                main: crate::types::AirOpenedValues { local: main_local, next: Vec::new() },
                permutation: crate::types::AirOpenedValues { local: Vec::new(), next: Vec::new() },
                quotient: vec![degree_bits],
                global_cumulative_sum: crate::septic_digest::SepticDigest::<Val<SC>>::zero(),
                local_cumulative_sum: Challenge::<SC>::ZERO,
                log_degree,
            }
        })
        .collect();
    ShardOpenedValues { chips: chip_opened }
}

/// Per-chip (local, global) cumulative sums.  `local` is ZERO
/// (the basefold path doesn't materialize the permutation trace); `global`
/// reads the RAW per-chip cells at their raw heights (device chips use the
/// early-captured provider TAIL, `chip_cum_tails`).
pub fn build_chip_cumulative_sums<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    shared_trace_mles: &[crate::multilinear::PaddedMle<Val<SC>>],
    chip_cum_tails: &[Option<Vec<Val<SC>>>],
) -> std::collections::BTreeMap<
    String,
    crate::shard_level::shard_proof::ChipCumulativeSums<Val<SC>, Challenge<SC>>,
>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
{
    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace.
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(
        chips.len(),
        shared_trace_mles.len(),
        "build_chip_cumulative_sums: chips/shared_trace_mles must be parallel",
    );
    assert_eq!(
        chips.len(),
        chip_cum_tails.len(),
        "build_chip_cumulative_sums: chips/chip_cum_tails must be parallel",
    );
    chips
        .iter()
        .zip(shared_trace_mles.iter())
        .zip(chip_cum_tails.iter())
        .map(|((chip, pm), tail)| {
            let name = MachineAir::<Val<SC>>::name(*chip);
            let global = if let Some(tail14) = tail {
                crate::shard_level::zerocheck_prover::chip_global_cumulative_sum_from_tail(
                    *chip, tail14,
                )
            } else {
                // Host chip: the raw row-major cells (last 14 read); a width-0
                // dummy yields an empty slice (sz<14 → zero digest).
                let vals: &[Val<SC>] = pm.real_trace_ref().map(|tr| tr.values).unwrap_or(&[]);
                crate::shard_level::zerocheck_prover::chip_global_cumulative_sum_from_values(
                    *chip, vals,
                )
            };
            let local = Challenge::<SC>::ZERO;
            (name, crate::shard_level::shard_proof::ChipCumulativeSums { local, global })
        })
        .collect()
}

/// The final `BasefoldShardProof` construction.  Derives
/// the witnessed per-round row/padding-column counts and the RAW
/// BaseFold root (`jagged_original_commitment`) from `evaluation_proof`, then
/// moves every piece into the proof.  PURE DATA — no transcript.
#[allow(clippy::too_many_arguments)]
pub fn assemble_basefold_shard_proof<SC>(
    public_values: Vec<Val<SC>>,
    main_commitment: [Val<SC>; 8],
    logup_gkr_proof: crate::shard_level::types::LogupGkrProof<Val<SC>, Challenge<SC>>,
    zerocheck_proof: crate::shard_level::types::PartialSumcheckProof<Challenge<SC>>,
    opened_values: ShardOpenedValues<Val<SC>, Challenge<SC>>,
    chip_heights: std::collections::BTreeMap<String, usize>,
    chip_cumulative_sums: std::collections::BTreeMap<
        String,
        crate::shard_level::shard_proof::ChipCumulativeSums<Val<SC>, Challenge<SC>>,
    >,
    evaluation_proof: crate::shard_level::shard_proof::EvaluationProof,
    orientation: FoldOrientation,
) -> BasefoldShardProof<Val<SC>, Challenge<SC>>
where
    SC: StarkGenericConfig,
{
    // Witnessed per-round per-chip row_counts + per-round padding_column_count,
    // derived from the host jagged packing (single-stacked main commit = ONE
    // round).  PURE DATA: nothing branches on these.
    let (row_counts, padding_column_counts): (Vec<Vec<usize>>, Vec<usize>) = match &evaluation_proof
    {
        crate::shard_level::shard_proof::EvaluationProof::Bundle(bundle) => {
            let (rc, pcc) = crate::jagged::derive_row_and_padding_counts(
                &bundle.packing.column_counts,
                &bundle.packing.offsets,
                bundle.packing.total_values,
            );
            (vec![rc], vec![pcc])
        }
        _ => (Vec::new(), Vec::new()),
    };

    // Jagged hash-bind: carry the RAW BaseFold root (the value the
    // BaseFold opening binds against) while the FS-observed `main_commitment`
    // is the MODIFIED digest.  Fall back to `main_commitment` on the
    // hash-bind-off path / non-bundle proofs.
    let jagged_original_commitment: [Val<SC>; 8] = match &evaluation_proof {
        crate::shard_level::shard_proof::EvaluationProof::Bundle(bundle) => {
            let raw_inner = crate::jagged_pcs::basefold_commit_digest(&bundle.commit);
            // SAFETY: [InnerVal; 8] == [Val<SC>; 8] under the inner-ring
            // TypeId identity (the only ring that produces a Bundle).
            unsafe { core::mem::transmute_copy::<[crate::InnerVal; 8], [Val<SC>; 8]>(&raw_inner) }
        }
        _ => main_commitment,
    };

    // The PREPROCESSED round, for a verifier that cannot see the key's chip
    // metadata: its RAW root (the key holds the hash-bound digest) and its per
    // chip row counts followed by its single padding column's height.  Heights
    // are the one part of that round's geometry the machine does not already
    // give a verifier, and the hash-bind pins them.
    let (preprocessed_original_commitment, preprocessed_row_counts): ([Val<SC>; 8], Vec<Val<SC>>) =
        match &evaluation_proof {
            crate::shard_level::shard_proof::EvaluationProof::Bundle(bundle)
                if !bundle.preceding_commits.is_empty()
                    && bundle.packing.round_counts.len() >= 2 =>
            {
                let raw_inner =
                    crate::jagged_pcs::basefold_commit_digest_felts(&bundle.preceding_commits[0]);
                // SAFETY: [InnerVal; 8] == [Val<SC>; 8] under the inner-ring TypeId
                // identity (the only ring that produces a multi-round Bundle).
                let raw = unsafe {
                    core::mem::transmute_copy::<[crate::InnerVal; 8], [Val<SC>; 8]>(&raw_inner)
                };
                let heights: Vec<Val<SC>> = bundle.packing.round_counts[0]
                    .iter()
                    .map(|(h, _w)| Val::<SC>::from_usize(*h))
                    .collect();
                (raw, heights)
            }
            _ => ([Val::<SC>::ZERO; 8], Vec::new()),
        };

    // Each round's single stacking-padding column height — what closes that
    // round out to its committed area.
    // Straight from the packing: the height the prover actually gave each
    // round's padding column.  Re-deriving it as
    // `real.next_multiple_of(1 << log_stacking_height) - real` is WRONG for a
    // round whose cells already fill whole stripes — the commitment still
    // covers one more stripe than that, so the derived height is a full stripe
    // short and the recursion's reconstructed final offset (and with it the
    // last column's jagged evaluation) misses by `1 << log_stacking_height`.
    let padding_row_heights: Vec<Vec<Val<SC>>> = match &evaluation_proof {
        crate::shard_level::shard_proof::EvaluationProof::Bundle(bundle) => bundle
            .packing
            .padding_heights
            .iter()
            .map(|round| round.iter().map(|h| Val::<SC>::from_usize(*h)).collect())
            .collect(),
        _ => Vec::new(),
    };

    BasefoldShardProof {
        public_values,
        main_commitment,
        padding_row_heights,
        logup_gkr_proof,
        zerocheck_proof,
        opened_values,
        chip_heights,
        chip_cumulative_sums,
        evaluation_proof,
        fold_orientation: orientation,
        row_counts,
        padding_column_counts,
        jagged_original_commitment,
        preprocessed_original_commitment,
        preprocessed_row_counts,
    }
}

/// Prove the shard's **trusted evaluations**: that the per-chip main-column
/// openings at `z_row` (`pre_y_per_chip` — these ARE the
/// `opened_values.chips[].main.local` values that zerocheck + LogUp-GKR
/// constrain) are the committed columns' values at `z_row`.
///
/// The emitted proof is the FULL chain — not just the F(r) opening: the real
/// jagged-eval sumcheck reducing the trusted-eval claims to
/// `sumcheck_final = F(r)·J(r)` (`prove_jagged_reduction_owned` +
/// `prove_jagged_evaluation`), PLUS the jagged-PCS opening proving `F(r)` is
/// the committed polynomial at the reduced point. The recursion verifier
/// binds this exact chain over the SAME `opened_values` Vec zerocheck
/// constrains, so the trusted evals cannot diverge from the committed trace.
///
/// The BaseFold commit arrives precomputed (`precomputed_commit`), so the
/// jagged-PCS pipeline skips its own commit step and the in-band commit
/// observe — the commit's 8-felt digest was already observed in the
/// transcript prologue as `main_commitment`.
///
/// The per-ring jagged open is dispatched through
/// [`crate::BasefoldRing::prove_jagged_open`], so the concrete `BfMmcs` /
/// `Challenger` are supplied by the impl rather than recovered at runtime.
///
/// `pub` so the host shard body reaches it directly.  A device driver has its
/// own body reading its own provider; this stays the host one.
pub fn prove_trusted_evaluations<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    // The FIRST opening round — see the trait method.
    preprocessed_named: &[(String, crate::multilinear::PaddedMle<Val<SC>>)],
    preprocessed_claims: Vec<Vec<Challenge<SC>>>,
    preprocessed_commit: &crate::jagged_pcs::jagged::PrecomputedJaggedCommitGeneric<
        <SC as crate::BasefoldRing>::BfMmcs,
    >,
    // BORROWED views over the shard prover's shared
    // `Arc<Mle>` store; `chip_traces` is built by a zero-copy slice relabel of
    // these views (no clone / move).
    main_traces: &[crate::multilinear::PaddedMle<Val<SC>>],
    shared_eval_point: &[Challenge<SC>],
    challenger: &mut SC::Challenger,
    precomputed_commit: crate::jagged_pcs::jagged::PrecomputedJaggedCommitGeneric<
        <SC as crate::BasefoldRing>::BfMmcs,
    >,
    // Per-chip main-column openings at z from the zerocheck residual
    // (`trace_at_z` main slice), parallel to `chips`; empty Vec per empty
    // chip.  UNCONDITIONAL column claims: the jagged layer still accepts
    // `Option` for synthetic callers that genuinely have no claims, but the
    // production path always supplies them.
    pre_y_per_chip: Vec<Vec<Challenge<SC>>>,
    // Per-chip metadata heights, parallel to `chips` (device dummies carry a
    // baked height; host chips `None`).  Consulted before `_device_traces` for
    // an empty (width-0) commit trace's REAL height in `r_row_per_chip` below.
    // An empty / short slice tolerates `.get` → provider fallback (the
    // CpuProver trait-method path passes `&[]`).
    heights: &[Option<usize>],
) -> crate::shard_level::shard_proof::EvaluationProof
where
    SC: StarkGenericConfig + crate::BasefoldRing,
    A: MachineAir<Val<SC>>,
    Val<SC>: PrimeField + 'static,
    Challenge<SC>: ExtensionField<Val<SC>> + 'static,
    // `SC::Challenger` drives the generic jagged BaseFold prover
    // directly on the OUTER (wrap) branch — the capability bounds
    // `prove_jagged_basefold_rounds_generic` requires. Both rings satisfy them
    // (inner `JaggedChallenger`, wrap `OuterChallenger`); NOT expressible as a
    // `BasefoldRing` implied bound, so threaded down the call chain.
    SC::Challenger:
        'static
            + p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + p3_challenger::CanObserve<
                <<SC as crate::BasefoldRing>::BfMmcs as p3_commit::Mmcs<
                    crate::jagged_pcs::JaggedVal,
                >>::Commitment,
            >,
{
    use crate::{BasefoldRing, InnerChallenge, InnerVal};
    use core::any::TypeId;

    // A REAL assert, not a `debug_assert!`: it is the only thing standing
    // between a non-KoalaBear config and the transmutes below, and
    // `debug_assert!` compiles out in release, which is exactly where that
    // would be UB.  One TypeId compare per shard.
    assert!(
        TypeId::of::<Val<SC>>() == TypeId::of::<InnerVal>()
            && TypeId::of::<Challenge<SC>>() == TypeId::of::<InnerChallenge>(),
        "prove_trusted_evaluations requires Val==KoalaBear /          Challenge==KoalaBear^4 (shared by inner + outer rings) for the trace/point          transmutes below",
    );

    // One reviewed reinterpret for the KoalaBear Val/Challenge `Vec`
    // transmutes below (per-chip `r_row` and the zerocheck-residual column
    // claims).  Under the TypeId gate asserted above, `Val<SC> == InnerVal`
    // and `Challenge<SC> == InnerChallenge`, so each conversion is a
    // zero-copy relabel with identical layout.
    //
    // SAFETY: every caller passes `A`/`B` that are the SAME KoalaBear type
    // (the TypeId gate). `ManuallyDrop` forbids the source double-free; the
    // (ptr, len, cap) triple is reused verbatim under an identical layout, so
    // the produced `Vec<B>` is byte-for-byte the reinterpreted `Vec<A>`.
    unsafe fn reinterpret_vec<A, B>(v: alloc::vec::Vec<A>) -> alloc::vec::Vec<B> {
        let mut v = core::mem::ManuallyDrop::new(v);
        alloc::vec::Vec::from_raw_parts(v.as_mut_ptr() as *mut B, v.len(), v.capacity())
    }

    // PARALLEL-ARRAY PRECONDITION.  The pairings below are POSITIONAL (`zip`),
    // and `zip` TRUNCATES on a length mismatch rather than failing — so a
    // mismatch would silently pair a chip with a DIFFERENT chip's trace.
    // `assert_eq!`, not `debug_assert_eq!`: release is where that matters.
    assert_eq!(
        chips.len(),
        main_traces.len(),
        "prove_trusted_evaluations: chips/main_traces must be parallel",
    );

    // Per-chip `r_row` = trailing log(chip_height) coords of the
    // shared eval_point.  Width-0 (device-resident, un-materialized) chips
    // resolve their REAL height via `heights` — the NORMAL device-resident
    // case: the dense commit packed them D2D and the reduction reads the
    // device handle.
    let r_row_per_chip: Vec<Vec<InnerChallenge>> = chips
        .iter()
        .zip(main_traces.iter())
        .enumerate()
        .map(|(i, (_chip, pm))| {
            let (tvals, twidth) = crate::jagged::real_cells(pm);
            let main_height = if twidth == 0 {
                heights.get(i).copied().flatten().unwrap_or(1)
            } else {
                tvals.len() / twidth
            };
            let log_h = main_height.max(1).next_power_of_two().trailing_zeros() as usize;
            let slice: &[Challenge<SC>] = if shared_eval_point.len() >= log_h {
                &shared_eval_point[shared_eval_point.len() - log_h..]
            } else {
                shared_eval_point
            };
            // SAFETY: Challenge<SC> == InnerChallenge (TypeId gate above).
            unsafe { reinterpret_vec::<Challenge<SC>, InnerChallenge>(slice.to_vec()) }
        })
        .collect();

    // Send `trace.width` directly; the verifier reads each chip's
    // `column_count` from `PackingMeta` so padding to `chip.width()`
    // would just inflate jagged-PCS data on sparse chips.
    // Each `chip_traces` entry is a BORROWED InnerVal view via a zero-copy
    // relabel of the borrowed Val<SC> view (Val<SC> == InnerVal under the
    // TypeId gate).  The views borrow the shard prover's shared `Arc<Mle>`
    // store for the duration of this open.
    let chip_traces: Vec<crate::jagged_pcs::jagged::ChipTraceView> = chips
        .iter()
        .zip(main_traces.iter())
        .map(|(chip, pm)| {
            let name = chip.name().to_string();
            // SAFETY: `Val<SC> == InnerVal` under the assert in this module, so
            // `PaddedMle<Val<SC>>` and `PaddedMle<InnerVal>` are the SAME type
            // and this is a no-op relabel.  The clone is an `Arc` refcount bump.
            let pm_inner: crate::multilinear::PaddedMle<InnerVal> = unsafe {
                core::mem::transmute_copy::<
                    crate::multilinear::PaddedMle<Val<SC>>,
                    crate::multilinear::PaddedMle<InnerVal>,
                >(&core::mem::ManuallyDrop::new(pm.clone()))
            };
            (name, pm_inner)
        })
        .collect();

    // z_row for the branching-program jagged-eval is the full shared
    // zerocheck point (the recursion verifier uses
    // `zerocheck_proof.point_and_eval.0`).  SAFETY: Challenge<SC> ==
    // InnerChallenge under the TypeId gate asserted above.
    let z_row: &[InnerChallenge] = unsafe {
        core::slice::from_raw_parts(
            shared_eval_point.as_ptr() as *const InnerChallenge,
            shared_eval_point.len(),
        )
    };

    // Reinterpret the residual openings to InnerChallenge (Challenge<SC> ==
    // InnerChallenge under the TypeId gate — the same relabel
    // `r_row_per_chip` and `chip_traces` already went through).  The wrap
    // ring's impl ignores these and keeps its own step-3 recompute —
    // identical values either way.
    let pre_y_inner: Vec<Vec<InnerChallenge>> = pre_y_per_chip
        .into_iter()
        // SAFETY: Challenge<SC> == InnerChallenge (TypeId gate).
        .map(|v| unsafe { reinterpret_vec::<Challenge<SC>, InnerChallenge>(v) })
        .collect();

    // Per-ring jagged open.  Each `BasefoldRing` impl supplies its own concrete
    // `BfMmcs` + `Challenger`, so `precomputed_commit` — typed
    // `PrecomputedJaggedCommitGeneric<SC::BfMmcs>` all the way down — is handed
    // over WITHOUT a `Box<dyn Any>` downcast, and the challenger without a
    // `downcast_mut`.
    //
    // The inner rings return `EvaluationProof::Bundle`; the wrap ring returns
    // `Bytes` (rmp-serialized `JaggedBasefoldBundleGeneric<OuterValMmcs>`) and
    // passes `pre_y_per_chip = None`.
    // ── The PREPROCESSED round's views, mirroring the main round's ────────
    //
    // Its heights come from the trace itself; every preprocessed trace is
    // host-resident (it was committed once at setup), so there is no
    // device-dummy height to resolve as there is for main.
    let prep_r_row_per_chip: Vec<Vec<InnerChallenge>> = preprocessed_named
        .iter()
        .map(|(_name, pm)| {
            let (tvals, twidth) = crate::jagged::real_cells(pm);
            let h = if twidth == 0 { 1 } else { tvals.len() / twidth };
            let log_h = h.max(1).next_power_of_two().trailing_zeros() as usize;
            let slice: &[Challenge<SC>] = if shared_eval_point.len() >= log_h {
                &shared_eval_point[shared_eval_point.len() - log_h..]
            } else {
                shared_eval_point
            };
            // SAFETY: Challenge<SC> == InnerChallenge (TypeId gate above).
            unsafe { reinterpret_vec::<Challenge<SC>, InnerChallenge>(slice.to_vec()) }
        })
        .collect();
    let prep_chip_traces: Vec<crate::jagged_pcs::jagged::ChipTraceView> = preprocessed_named
        .iter()
        .map(|(name, pm)| {
            // SAFETY: same no-op relabel as the main round above.
            let pm_inner: crate::multilinear::PaddedMle<InnerVal> = unsafe {
                core::mem::transmute_copy::<
                    crate::multilinear::PaddedMle<Val<SC>>,
                    crate::multilinear::PaddedMle<InnerVal>,
                >(&core::mem::ManuallyDrop::new(pm.clone()))
            };
            (name.clone(), pm_inner)
        })
        .collect();
    let prep_claims_inner: Vec<Vec<InnerChallenge>> = preprocessed_claims
        .into_iter()
        // SAFETY: Challenge<SC> == InnerChallenge (TypeId gate).
        .map(|v| unsafe { reinterpret_vec::<Challenge<SC>, InnerChallenge>(v) })
        .collect();

    // Round order: [preprocessed, main].  The preprocessed round comes FIRST
    // because the verifier samples each round's z_col from the shared
    // challenger in round order.  A machine with no preprocessed traces emits
    // the single main round.
    let mut rounds: Vec<crate::jagged_pcs::jagged::JaggedOpenRound<'_, _>> = Vec::with_capacity(2);
    if !prep_chip_traces.is_empty() {
        rounds.push(crate::jagged_pcs::jagged::JaggedOpenRound {
            chip_traces: &prep_chip_traces,
            r_row_per_chip: &prep_r_row_per_chip,
            claims: prep_claims_inner,
            precomputed: preprocessed_commit,
        });
    }
    rounds.push(crate::jagged_pcs::jagged::JaggedOpenRound {
        chip_traces: &chip_traces,
        r_row_per_chip: &r_row_per_chip,
        claims: pre_y_inner,
        precomputed: &precomputed_commit,
    });
    <SC as BasefoldRing>::prove_jagged_open(z_row, rounds, challenger)
}
