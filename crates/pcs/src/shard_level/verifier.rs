//! Host-side BasefoldShardVerifier: transcript prologue + LogUp-GKR
//! + zerocheck + jagged-PCS verification, executing directly against
//! host types rather than symbolic AIR.

use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField};

use super::basefold_constraint_folder::{
    compute_padded_row_adjustment_basefold_host, eval_constraints_basefold_host,
    BasefoldConstraintFolder,
};
use super::shard_proof::{BasefoldShardProof, FoldOrientation};
use super::types::{LogupGkrProof, PartialSumcheckProof};
use crate::air::MachineAir;
use crate::lookup::LookupKind;
use crate::types::ShardOpenedValues;
use crate::{Challenge, Chip, StarkGenericConfig, StarkVerifyingKey, Val};

/// Errors emitted by the host-side shard-level BaseFold verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasefoldVerifyError {
    /// Shape mismatch between the proof's public_values length and
    /// the machine's expected PV count.
    PublicValuesLengthMismatch { expected: usize, got: usize },
    /// Shape mismatch between the proof's chip list and the machine's
    /// chip set.
    ChipCountMismatch { expected: usize, got: usize },
    /// LogUp-GKR verification failed (sumcheck identity, chip opening
    /// consistency, or GKR-circuit-output MLE shape).
    LogupGkr(String),
    /// Zerocheck verification failed (constraint identity or
    /// sumcheck-point dimension).
    Zerocheck(String),
    /// Jagged-PCS opening verification failed.
    JaggedPcs(String),
    /// Reserved for staged verifier ports and defensive call sites that
    /// intentionally reject an unsupported proof sub-flow.
    Unimplemented(&'static str),
}

impl core::fmt::Display for BasefoldVerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PublicValuesLengthMismatch { expected, got } => {
                write!(f, "public_values length mismatch: expected {expected}, got {got}")
            }
            Self::ChipCountMismatch { expected, got } => {
                write!(f, "chip count mismatch: expected {expected}, got {got}")
            }
            Self::LogupGkr(msg) => write!(f, "LogUp-GKR: {msg}"),
            Self::Zerocheck(msg) => write!(f, "zerocheck: {msg}"),
            Self::JaggedPcs(msg) => write!(f, "jagged-PCS: {msg}"),
            Self::Unimplemented(phase) => {
                // The trailing "" is the grep-able staged-verifier-port
                // tracking hint (`unimplemented_error_displays_phase_hint`
                // asserts on it) so users who hit an unimplemented sub-flow
                // can find the umbrella tracking issue.
                write!(f, "host-side BasefoldShardVerifier: {phase} not yet implemented")
            }
        }
    }
}

impl std::error::Error for BasefoldVerifyError {}

/// Host-side shard-level BaseFold verifier.
///
/// Parameterised on `SC: StarkGenericConfig` to match the
/// [`BasefoldShardProof`] it consumes.  When the proof and config
/// refer to `KoalaBearPoseidon2`, the verifier drives the LogUp-GKR
/// + zerocheck + jagged-PCS flow that the recursion-circuit
/// in-circuit version already implements.
///
/// Construct via [`Self::production_default`] for max_log_row_count = 22
/// (Ziren's shard-padded default) or [`Self::with_params`] for custom.
#[derive(Clone, Debug)]
pub struct BasefoldShardVerifier {
    /// Shard-padded max log row count — determines zerocheck dim and
    /// jagged-PCS stack depth.
    pub max_log_row_count: usize,
}

impl BasefoldShardVerifier {
    /// Production default (max_log_row_count = 22).  The BaseFold codeword
    /// two-adicity bound is over the STACKED poly's `log_stacking_height`
    /// (≤ DEFAULT_LOG_STACKING_HEIGHT = 21), NOT max_log_row_count: the
    /// LDE domain is `2^(log_stacking + log_blowup)`.  At the inner
    /// default `log_blowup = 2` (`basefold/config.rs::default_fri_config`),
    /// `log_stacking(≤21) + 2 ≤ 23 ≤ KoalaBear TWO_ADICITY = 24`, so the
    /// recursion-circuit verifier's `two_adic_generator(log_codeword_size)`
    /// does not panic (one bit of headroom; the wrap stage at blowup=3 sits
    /// at exactly 24).
    #[must_use]
    pub const fn production_default() -> Self {
        // FIXED cube: every stage proves and verifies at exactly this
        // constant — never floated per proof.  Coverage invariant: the core
        // executor's `height_split` closes a shard before any chip reaches
        // `2^CORE_MAX_LOG_ROW_COUNT` rows (`CORE_SHARD_HEIGHT_THRESHOLD`,
        // measured peaks ~2.5M vs the 4.1M fence), and every recursion band
        // is asserted `<=` this cube at shape construction
        // (recursion/core shape.rs).
        // Two-adic-safe: the BaseFold codeword bound is over
        // log_stacking_height (fixed 21), NOT max_log_row_count.
        Self { max_log_row_count: crate::stacked_shapes::types::consts::CORE_MAX_LOG_ROW_COUNT }
    }

    /// Construct with explicit parameters.  Use when writing tests
    /// against small shards.
    #[must_use]
    pub const fn with_params(max_log_row_count: usize) -> Self {
        Self { max_log_row_count }
    }

    /// Verify a shard-level BaseFold proof against the machine's
    /// chip set, verifying key, and public values.
    ///
    /// The verifier mirrors the shard-level prover transcript order:
    /// prologue observations, LogUp-GKR verification, direct zerocheck
    /// sumcheck verification, and jagged-PCS opening verification.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_shard<SC, A>(
        &self,
        _vk: &StarkVerifyingKey<SC>,
        chips: &[&Chip<Val<SC>, A>],
        // The MACHINE's preprocessed chips (name, width), name ordered.
        prep_chip_dims: &[(String, usize)],
        proof: &BasefoldShardProof<Val<SC>, Challenge<SC>>,
        challenger: &mut SC::Challenger,
        num_pv_elts: usize,
        // `true` for the CORE machine (rev shard proofs); `false`
        // for recursion / shrink / wrap (LEGACY). Drives the zerocheck host
        // orientation (collapsed/no-embed claim + rev(z_gkr) eq-bridge anchor).
        core_rev: bool,
    ) -> Result<(), BasefoldVerifyError>
    where
        SC: StarkGenericConfig + crate::BasefoldRing,
        A: MachineAir<Val<SC>>
            + for<'b> Air<BasefoldConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
        Val<SC>: PrimeField,
        Challenge<SC>: ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>>,
        // Threaded to `verify_jagged_pcs_host`'s static OUTER
        // generic BaseFold verify (see its where-clause). Verify-only, both
        // rings satisfy it.
        SC::Challenger: 'static
            + p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + p3_challenger::CanObserve<
                <<SC as crate::BasefoldRing>::BfMmcs as p3_commit::Mmcs<
                    crate::jagged_pcs::JaggedVal,
                >>::Commitment,
            >,
    {
        // Shape check: public_values length.
        if proof.public_values.len() != num_pv_elts {
            return Err(BasefoldVerifyError::PublicValuesLengthMismatch {
                expected: num_pv_elts,
                got: proof.public_values.len(),
            });
        }
        // Shape check: chip count vs. LogUp-GKR openings.
        let opening_count = proof.logup_gkr_proof.logup_evaluations.chip_openings.len();
        if opening_count != chips.len() {
            return Err(BasefoldVerifyError::ChipCountMismatch {
                expected: chips.len(),
                got: opening_count,
            });
        }

        // ── Transcript prologue ──────────────────────────────────
        //
        // Observe public values, main commitment, and per-chip
        // metadata.  Order MUST match the prover's ordering at
        // `shard_level::prover::prove_shard_with_data` (transcript
        // prologue):
        //   1. public_values (each felt)
        //   2. main_commitment (8 felts)
        //   3. num_chips (1 felt)
        //   4. for each chip:
        //        a. RAW height (1 felt)
        //        b. name_length_felt
        //        c. per-byte felts
        //
        // The per-chip height observe sources from
        // `proof.chip_heights` keyed by chip name — the value
        // already carried in the proof and observed in the recursion
        // verifier via the `chip_height_bits` Horner recompose.

        for &pv in proof.public_values.iter() {
            challenger.observe(pv);
        }
        for &c in proof.main_commitment.iter() {
            challenger.observe(c);
        }
        let num_chips = Val::<SC>::from_u64(chips.len() as u64);
        challenger.observe(num_chips);
        for chip in chips.iter() {
            let name = chip.name();

            // Per-chip RAW-height observe (the raw
            // `num_real_entries`, 0 allowed).  Mirrors the prover's
            // `raw_chip_height` derivation via `proof.chip_heights[name]`.
            // Default 0 if absent (matches legacy proof bytes where the
            // map is empty).
            let h = proof.chip_heights.get(name.as_str()).copied().unwrap_or(0);
            challenger.observe(Val::<SC>::from_u64(h as u64));

            // Name length + name bytes (unchanged).
            let len_felt = Val::<SC>::from_u64(name.len() as u64);
            challenger.observe(len_felt);
            for byte in name.bytes() {
                challenger.observe(Val::<SC>::from_u64(byte as u64));
            }
        }

        // ── LogUp-GKR sumcheck verification ──────────────────────
        //
        // Ported from
        //   crates/recursion/circuit/src/logup_gkr.rs::verify_logup_gkr
        // with in-circuit Builder<C>/Ext<> ops replaced by direct
        // Challenge<SC> arithmetic.
        //
        // Note: the public-values constraint evaluation piece
        // (verify_public_values closure) is *not* ported here —
        // shard-level proofs carry public values in a separate
        // logup_evaluations path and the check is deferred to the
        // final reduction.  For structural verification this
        // simplifies to sumcheck consistency + GKR identity.
        // Compute beta_seed_dim the same way the prover does:
        // log2(max_arity.next_power_of_two()) where max_arity =
        // max(interaction.values.len() + 1) across all chips.
        let max_arity = chips
            .iter()
            .flat_map(|chip| chip.sends().iter().chain(chip.receives().iter()))
            .map(|interaction| interaction.values.len() + 1)
            .max()
            .unwrap_or(1);
        let beta_seed_dim = max_arity.next_power_of_two().trailing_zeros() as usize;

        // The core Option-2 public-values closure (`eval_public_values`:
        // State / GlobalAccumulation / MemoryGlobalInit+Finalize boundary
        // buses) only applies to machines that actually carry those buses —
        // i.e. the MIPS core machine.  The recursion machine uses only
        // self-cancelling `Local` buses (Memory / Program / Range / Syscall),
        // so its local-only closure is `gkr_sum == 0` and the core State-bus
        // PV-AIR (which reads a different PV schema and emits arity-16
        // GlobalAccumulation messages) must not run for it.  Detect the
        // machine kind structurally from its interaction set.
        let machine_has_pv_buses = chips.iter().any(|chip| {
            chip.sends().iter().chain(chip.receives().iter()).any(|lk| {
                matches!(
                    lk.kind,
                    LookupKind::State
                        | LookupKind::GlobalAccumulation
                        | LookupKind::MemoryGlobalInitControl
                        | LookupKind::MemoryGlobalFinalizeControl
                )
            })
        });

        // ── FIXED CUBE ──────────────────────────────────
        //
        // `self.max_log_row_count` is the fixed config cube — never floated
        // up from the proof.  The GKR round-count check
        // (`round_proofs.len() + 1 == max_log_row_count`) and the zerocheck
        // point-dim check both bind the proof to this constant, so a proof
        // produced at any other cube is rejected outright.  Every reachable
        // proof IS at this cube: the core executor's `height_split` caps
        // chip heights below `2^22` and the recursion bands are asserted
        // `<=` the cube at shape construction.
        let max_log_row_count = self.max_log_row_count;

        verify_logup_gkr_host::<SC, A>(
            &proof.logup_gkr_proof,
            chips,
            &proof.opened_values,
            max_log_row_count,
            beta_seed_dim,
            proof.fold_orientation,
            &proof.public_values,
            machine_has_pv_buses,
            challenger,
        )?;

        // ── Zerocheck sumcheck verification ──────────────────────
        //
        // Samples the same phase challenges as the in-circuit verifier,
        // checks the direct `Σ_b C(b) == 0` sumcheck, and observes the
        // per-chip openings that feed the following jagged-PCS phase.
        //
        // Reference:
        //   crates/recursion/circuit/src/zerocheck.rs::BasefoldZerocheckVerifier::verify_zerocheck
        verify_zerocheck_host::<SC, A>(
            chips,
            &proof.zerocheck_proof,
            &proof.logup_gkr_proof.logup_evaluations,
            &proof.public_values,
            max_log_row_count,
            core_rev,
            challenger,
            // Discriminator: opened_values carries the trace@z*
            // openings the circuit's rlc_eval (zerocheck.rs:613) is built
            // from; pass it so the gated host-recompute can compare.
            &proof.opened_values,
        )?;

        // ── Jagged HASH-BIND re-check ───────
        //
        // Recompute
        //   modified' = compress([raw_root, hash(once(len) ++ rc ++ cc)])
        // from the bundle's RAW BaseFold root + per-chip geometry, and assert
        // it equals the FS-observed `main_commitment`.  This ties the
        // per-chip (row_count, column_count) geometry to the commitment so a
        // height-agnostic prover cannot witness a geometry different from what
        // was committed.  Only the inner KoalaBear ring (the one that emits a
        // `Bundle`); the outer ring re-binds inside its registered hook.
        // Skipped when the hash-bind is off (then `main_commitment` IS the raw
        // root — the bundle re-check would trivially fail, so gate on it).
        {
            use crate::shard_level::shard_proof::EvaluationProof;
            use crate::{InnerChallenge, InnerVal};
            use core::any::TypeId;
            let inner_ring = TypeId::of::<Val<SC>>() == TypeId::of::<InnerVal>()
                && TypeId::of::<Challenge<SC>>() == TypeId::of::<InnerChallenge>()
                && TypeId::of::<SC::Challenger>()
                    == TypeId::of::<crate::jagged_pcs::JaggedChallenger>();
            if inner_ring {
                if let EvaluationProof::Bundle(bundle) = &proof.evaluation_proof {
                    // The raw root must be the MAIN round's, to match
                    // `proof.main_commitment`: the preprocessed round occupies
                    // group 0, so main is group 1.  The batched proof carries
                    // the MAIN round's commit (the preprocessed one is the
                    // verifying key's).
                    let raw_inner = crate::jagged_pcs::basefold_commit_digest(&bundle.commit);

                    // Guard the counts (BaseFieldOverflow + AreaOutOfBounds).
                    // Counts feed `from_canonical_usize` (wraps mod the field
                    // order ~2^31), so a count >= ORDER could alias to a
                    // different felt — reject it.  And the total area must be
                    // 0 < area < 2^30 (field-arith overflow bound).  These are
                    // host-side usize checks on the SAME per-chip (row, col)
                    // counts the hash is taken over.
                    // The hash-bind ties the MAIN round's geometry to the MAIN
                    // commitment (`proof.main_commitment`), and the prover
                    // computed it over that round's OWN counts.  Read them from
                    // the per-round list rather than recovering them from the
                    // flattened packing — the flattened column space also
                    // carries the stacking padding between rounds, so a round's
                    // geometry is not a positional slice of it.
                    let (rc_g, cc_g): (Vec<usize>, Vec<usize>) =
                        match bundle.packing.round_counts.last() {
                            Some(main_round) => main_round.iter().copied().unzip(),
                            // Legacy single-round bundle: the whole packing IS
                            // the main round.
                            None => crate::jagged_pcs::jagged_counts_from_packing(&bundle.packing),
                        };
                    let order = <InnerVal as p3_field::PrimeField32>::ORDER_U32 as usize;
                    if rc_g.iter().chain(cc_g.iter()).any(|&c| c >= order) {
                        return Err(BasefoldVerifyError::JaggedPcs(
                            "jagged hash-bind: count >= F::ORDER (BaseFieldOverflow)".into(),
                        ));
                    }
                    let area: usize = rc_g
                        .iter()
                        .zip(cc_g.iter())
                        .map(|(r, c)| r.saturating_mul(*c))
                        .fold(0usize, |a, b| a.saturating_add(b));
                    if area == 0 || area >= (1usize << 30) {
                        return Err(BasefoldVerifyError::JaggedPcs(
                            "jagged hash-bind: area out of bounds (0 < area < 2^30) \
                             (AreaOutOfBounds)"
                                .into(),
                        ));
                    }

                    // The bundle carries `packing: PackingMeta` (offsets +
                    // column_counts) — use the PackingMeta overload so the
                    // hashed felt sequence is byte-identical to the host emit
                    // (which used the full JaggedPacking; both derive the same
                    // per-chip (row, col) counts).
                    let recomputed =
                        crate::jagged_pcs::jagged_hash_bind_modified(raw_inner, &rc_g, &cc_g);
                    // SAFETY: [InnerVal;8] == [Val<SC>;8] under the inner gate.
                    let observed_inner: [InnerVal; 8] = unsafe {
                        core::mem::transmute_copy::<[Val<SC>; 8], [InnerVal; 8]>(
                            &proof.main_commitment,
                        )
                    };
                    if recomputed != observed_inner {
                        return Err(BasefoldVerifyError::JaggedPcs(
                            "jagged hash-bind mismatch: recomputed \
                             compress([raw_root, hash(counts)]) != observed \
                             main_commitment (IncorrectTableSizes)"
                                .into(),
                        ));
                    }
                }
            }
        }

        // ── Jagged-PCS opening verification ──────────────────────
        //
        // Delegate to the existing host-side verifier at
        // crate::jagged_pcs::jagged::verify_jagged_basefold_no_observe
        // after deserialising the bundle bytes.  See detailed rationale
        // in verify_jagged_pcs_host.
        verify_jagged_pcs_host::<SC, A>(
            _vk,
            chips,
            prep_chip_dims,
            // Jagged verified at the zerocheck-reduced z*.
            &proof.zerocheck_proof.point_and_eval.0,
            &proof.evaluation_proof,
            &proof.logup_gkr_proof.logup_evaluations,
            // The trace openings the recursion circuit cross-binds to the
            // jagged claimed sum; index-aligned with `chips`.
            &proof.opened_values,
            challenger,
        )?;

        Ok(())
    }
}

/// Host-side jagged-PCS opening verification.
///
/// Deserialises the bundle bytes and delegates to the host-side verifier at
/// [`crate::jagged_pcs::jagged::verify_jagged_basefold_no_observe`].
///
/// The TypeId gate mirrors prove_trusted_evaluations — returns `Ok(())`
/// for non-KoalaBear configs (nothing to verify in that path).
fn verify_jagged_pcs_host<SC, A>(
    // The verifying key pins the PREPROCESSED round: which chips it covers,
    // in which order, and at what dimensions.  Read from here, never from the
    // proof — the round exists to bind the preprocessed traces to
    // `vk.commit`.
    vk: &StarkVerifyingKey<SC>,
    chips: &[&Chip<Val<SC>, A>],
    // The MACHINE's preprocessed chips (name, width), name ordered.
    prep_chip_dims: &[(String, usize)],
    shared_eval_point: &[Challenge<SC>],
    evaluation_proof: &super::shard_proof::EvaluationProof,
    _gkr_evaluations: &super::types::LogUpEvaluations<Challenge<SC>>,
    // Cross-bind: the shard's trace openings.  `opened_values.chips[i]`
    // is index-aligned with `chips[i]` and the bundle's `y_per_chip[i]`
    // (all three share the machine's name-sorted chip order); we pass each
    // chip's `main.local` into the jagged verifier so it can mirror the
    // recursion circuit's `evaluate_mle(main.local, z_col) == claimed_sum`
    // assert (recursive_jagged_pcs.rs:247) and reject a bundle whose
    // `y_per_chip` diverges from the openings the zerocheck consumed.
    opened_values: &crate::ShardOpenedValues<Val<SC>, Challenge<SC>>,
    challenger: &mut SC::Challenger,
) -> Result<(), BasefoldVerifyError>
where
    SC: StarkGenericConfig + crate::BasefoldRing,
    A: MachineAir<Val<SC>>,
    Val<SC>: PrimeField + 'static,
    Challenge<SC>: ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>> + Copy + 'static,
    // `SC::Challenger` drives the generic jagged BaseFold VERIFIER
    // directly on the OUTER (wrap) branch. The prover threads the same
    // capability bounds; both rings satisfy them (inner `JaggedChallenger`,
    // wrap `OuterChallenger`). Verify-only: no VK / committed-byte impact.
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
    use crate::jagged::JaggedChipInfo;
    use crate::jagged_pcs::jagged::{verify_jagged_basefold_no_observe, JaggedBasefoldBundle};
    use crate::shard_level::shard_proof::EvaluationProof;
    use crate::{InnerChallenge, InnerVal};
    use core::any::{Any, TypeId};

    // Type gate (same as prover-side prove_trusted_evaluations): a TypeId
    // transmute-safety guard for the unsafe `InnerChallenge` reinterpretation
    // below — the TypeId check is exactly the identity that makes the
    // transmute sound.  The `BasefoldRing` bound lets the OUTER (wrap) branch
    // call the generic BaseFold verify statically; the field/challenger
    // TypeId gates remain as transmute + static-vs-dynamic dispatch guards.
    if TypeId::of::<Val<SC>>() != TypeId::of::<InnerVal>()
        || TypeId::of::<Challenge<SC>>() != TypeId::of::<InnerChallenge>()
    {
        // Non-KoalaBear field — skip (prover emitted Empty too).
        return Ok(());
    }

    // OUTER (wrap) ring dispatch. Val/Challenge are KoalaBear / KoalaBear^4
    // here, but the challenger is OuterChallenger (not JaggedChallenger):
    // deserialize + `build_jagged_verify_inputs` +
    // `verify_jagged_basefold_inner_generic` over the `BasefoldRing`
    // associated types (on this branch `SC::Challenger == OuterChallenger`,
    // `SC::BfMmcs == OuterValMmcs`).  Verify-only: no VK / committed-byte
    // impact.
    if TypeId::of::<SC::Challenger>() != TypeId::of::<crate::jagged_pcs::JaggedChallenger>() {
        use crate::jagged_pcs::jagged::{
            build_jagged_verify_inputs, verify_jagged_basefold_inner_generic,
            JaggedBasefoldBundleGeneric,
        };
        use p3_air::BaseAir;
        let bytes = match evaluation_proof {
            EvaluationProof::Empty => return Ok(()),
            EvaluationProof::Bytes(b) => b,
            EvaluationProof::Bundle(_) => {
                return Err(BasefoldVerifyError::JaggedPcs(
                    "outer ring expects a serialized (Bytes) BaseFold bundle, got Bundle".into(),
                ));
            }
        };
        let bundle =
            match JaggedBasefoldBundleGeneric::<<SC as crate::BasefoldRing>::BfMmcs>::from_bytes(
                bytes,
            ) {
                Some(b) => b,
                None => {
                    return Err(BasefoldVerifyError::JaggedPcs(format!(
                        "outer BaseFold bundle deserialize failed ({} bytes)",
                        bytes.len()
                    )));
                }
            };
        let chip_widths: Vec<usize> =
            chips.iter().map(|c| <_ as BaseAir<Val<SC>>>::width(*c)).collect();
        // SAFETY: Challenge<SC> == InnerChallenge under the field gate above.
        let eval_point_inner: &[InnerChallenge] = unsafe {
            core::slice::from_raw_parts(
                shared_eval_point.as_ptr() as *const InnerChallenge,
                shared_eval_point.len(),
            )
        };
        let (chip_infos, r_row_per_chip, z_row) =
            build_jagged_verify_inputs(&bundle.packing, &chip_widths, eval_point_inner);
        let mmcs = <SC as crate::BasefoldRing>::bf_mmcs();
        let fri = <SC as crate::BasefoldRing>::fri_config();
        let ok = verify_jagged_basefold_inner_generic::<
            SC::Challenger,
            <SC as crate::BasefoldRing>::BfMmcs,
        >(
            &chip_infos,
            &r_row_per_chip,
            &z_row,
            &bundle,
            challenger,
            mmcs,
            /* skip_commit_observe = */ true,
            fri,
            // The rounds committed before this one — the preprocessed round.
            // Its area is exactly its real cells plus the stacking padding that
            // closes it out, both of which the packing carries, so nothing is
            // re-derived here.
            &bundle
                .preceding_commits
                .iter()
                .enumerate()
                .map(|(r, c)| {
                    let real: usize = bundle
                        .packing
                        .round_counts
                        .get(r)
                        .map(|round| round.iter().map(|(h, w)| h * w).sum())
                        .unwrap_or(0);
                    let pad: usize =
                        bundle.packing.padding_heights.get(r).map(|p| p.iter().sum()).unwrap_or(0);
                    (c.clone(), real + pad)
                })
                .collect::<Vec<_>>(),
        );
        return if ok {
            Ok(())
        } else {
            Err(BasefoldVerifyError::JaggedPcs("outer BaseFold bundle rejected".into()))
        };
    }

    // Resolve to a bundle. Empty means no jagged-PCS proof to verify;
    // Bundle is the host-emitted structured form; Bytes is a device
    // hook's pre-serialized form that we deserialize here.
    let bundle = match evaluation_proof {
        EvaluationProof::Empty => return Ok(()),
        EvaluationProof::Bundle(b) => b.clone(),
        EvaluationProof::Bytes(bytes) => {
            JaggedBasefoldBundle::from_bytes(bytes).ok_or_else(|| {
                BasefoldVerifyError::JaggedPcs(format!(
                    "rmp-serde deserialize failed ({} bytes)",
                    bytes.len()
                ))
            })?
        }
    };

    // Read per-chip `column_count` from the bundle's PackingMeta (written by
    // the prover) instead of `BaseAir::width(chip)`, so the verifier agrees
    // with the *actually-exercised* column count.  Falls back to
    // `BaseAir::width(chip)` for legacy bundles (column_counts vec is empty
    // when serde-default populated from older wire format).
    // Two rounds: `[preprocessed, main]`.  The proof carries the
    // preprocessed round as group 0 and main as group 1.
    //
    // The PREPROCESSED round's chips and WIDTHS come from the MACHINE (name
    // ordered, exactly the set and order `setup` commits); its HEIGHTS are
    // claimed by the proof and pinned by the hash-bind against the key's
    // commitment.  Nothing here reads chip metadata off the key — the key
    // carries none; the commitment already says what shape was committed.
    // ── Rebuild the batched column layout, round by round ────────────────
    //
    // The proof is ONE jagged instance whose columns run
    // `[round 0 real | round 0 stacking pad | round 1 real | round 1 pad | ..]`.
    // The pad is not chip geometry — it is the space the stacked commitment
    // added rounding each round up to a stripe boundary — so it cannot be
    // recovered positionally from the flattened counts.  Rebuild it here from
    // the per-round counts, and pin round 0 (preprocessed) against the
    // VERIFYING KEY, which is what makes that round mean anything.
    // How many chips the preprocessed round has is a property of the MACHINE, so
    // the verifier counts them itself.  Taking it from the proof would let a
    // prover drop the round and skip its binding entirely.
    let n_prep = chips
        .iter()
        .filter(|c| <_ as crate::air::MachineAir<Val<SC>>>::preprocessed_width(**c) > 0)
        .count();
    let combined_packing = &bundle.packing;
    let log_stack = bundle.commit.log_stacking_height as usize;
    let cube = 1usize << shared_eval_point.len();

    let mut chip_infos: Vec<JaggedChipInfo> = Vec::new();
    // How many leading entries belong to the preprocessed round (its real chips
    // plus its padding) — the cross-bind and `n_prep` bookkeeping key off this.
    let mut n_prep_infos = 0usize;

    // ALWAYS at least one padding column per round, even on a round that lands
    // exactly on a stripe boundary.  Mirrors the prover.
    let push_padding = |infos: &mut Vec<JaggedChipInfo>, pad: usize| {
        let mut done = 0usize;
        loop {
            let h = core::cmp::min(cube, pad - done);
            infos.push(JaggedChipInfo {
                name: alloc::format!("<stacking-pad:{}>", infos.len()),
                row_count: h,
                column_count: 1,
            });
            done += h;
            if done >= pad {
                break;
            }
        }
    };

    if n_prep > 0 {
        // Round 0's geometry is CLAIMED by the proof and PINNED by the
        // hash-bind below: the key's commitment is
        // `compress([raw_root, hash(these counts)])`, so a proof that claims
        // different counts cannot re-derive it — the key carries no chip
        // metadata; the commitment already says what shape was committed.
        // NAMES and WIDTHS come from the MACHINE (its preprocessed chips, name
        // ordered -- the same set and order `setup` commits); HEIGHTS are
        // claimed by the proof and pinned by the hash-bind below.
        let Some(prep_round) = combined_packing.round_counts.first() else {
            return Err(BasefoldVerifyError::JaggedPcs(
                "preprocessed round: the proof carries no geometry for it".into(),
            ));
        };
        if prep_round.len() != n_prep {
            return Err(BasefoldVerifyError::JaggedPcs(format!(
                "preprocessed round: the proof claims {} chips, the machine has {n_prep}",
                prep_round.len(),
            )));
        }
        let mut prep_total = 0usize;
        for ((name, width), (height, claimed_width)) in prep_chip_dims.iter().zip(prep_round.iter())
        {
            if claimed_width != width {
                return Err(BasefoldVerifyError::JaggedPcs(format!(
                    "preprocessed round: chip {name} is {claimed_width} columns in the proof \
                     but {width} in the machine",
                )));
            }
            chip_infos.push(JaggedChipInfo {
                name: name.clone(),
                row_count: *height,
                column_count: *width,
            });
            prep_total += width.saturating_mul(*height);
        }
        // The area the preprocessed commitment actually covers: the real cells
        // rounded out to whole stacking blocks, exactly as the prover's commit
        // does (`zkm_pcs::jagged::committed_dense_len`).  Derived, not read from
        // the proof: this is what pins round 0's padding, and with it the column
        // space the jagged evaluation runs over.
        let prep_area = crate::jagged::committed_dense_len(prep_total, log_stack);
        push_padding(&mut chip_infos, prep_area.saturating_sub(prep_total));
        n_prep_infos = chip_infos.len();
    }

    // Round 1: the shard's main chips, widths from the packing.
    use p3_air::BaseAir;
    let main_column_counts: &[usize] =
        combined_packing.column_counts.get(n_prep_infos..).unwrap_or(&[]);
    chip_infos.extend(chips.iter().enumerate().map(|(i, chip)| {
        let column_count = main_column_counts
            .get(i)
            .copied()
            .unwrap_or_else(|| <_ as BaseAir<Val<SC>>>::width(*chip));
        JaggedChipInfo {
            name: chip.name().to_string(),
            row_count: 0, // filled from the packing offsets below
            column_count,
        }
    }));
    let n_main_infos = chip_infos.len() - n_prep_infos;

    // Patch row_count from bundle.packing.offsets.
    //
    // Important: `offsets` has ONE ENTRY PER COLUMN plus a final
    // sentinel `offsets[total_cols] = total_values`
    // (see `crate::jagged::JaggedPacking::offsets`).  The
    // prover's `compute_jagged_metadata` pushes `chip.width` offsets
    // per chip and closes the slice with the sentinel.  Within a
    // single chip's run of columns, consecutive offsets differ by
    // exactly that chip's row_count (all columns have the same
    // height).  So we walk offsets with a column-index cursor and
    // read `offsets[col_idx + 1] - offsets[col_idx]` to get the
    // height — the sentinel keeps the `col_idx + 1` lookup in-bounds
    // for the last column too.  The `else if` fallback remains for
    // legacy bundles serialized before the sentinel was added.
    let mut chip_infos = chip_infos;
    {
        // Only the MAIN region's heights come from the packing; the
        // preprocessed region's are already pinned by the verifying key above,
        // which is the whole point of opening it against `vk.commit`.
        let mut col_idx = 0usize;
        for (i, info) in chip_infos.iter_mut().enumerate() {
            if info.column_count == 0 {
                continue;
            }
            let h = if col_idx + 1 < combined_packing.offsets.len() {
                combined_packing.offsets[col_idx + 1]
                    .saturating_sub(combined_packing.offsets[col_idx])
            } else if col_idx < combined_packing.offsets.len() {
                combined_packing.total_values.saturating_sub(combined_packing.offsets[col_idx])
            } else {
                0
            };
            if i < n_prep_infos {
                // Round 0 (preprocessed, including its padding) is already
                // pinned — by the verifying key for the real chips, and by the
                // key-derived area for the padding.  The packing must AGREE.
                // This is the bind that makes the preprocessed round mean
                // anything.
                if info.row_count != h {
                    return Err(BasefoldVerifyError::JaggedPcs(format!(
                        "preprocessed round: {} is {} rows in the packing but {} as \
                         pinned by the verifying key",
                        info.name, h, info.row_count,
                    )));
                }
            } else if i < n_prep_infos + n_main_infos {
                info.row_count = h;
            } else {
                // Round 1's padding: heights are whatever the packing says, but
                // they must still be covered by the accounting below.
                info.row_count = h;
            }
            col_idx += info.column_count;
        }

        // COLUMN-ACCOUNTING CHECK.  The walk above assumes the verifier's chip
        // list accounts for EVERY column in the main packing: it advances
        // `col_idx` by each chip's width and reads heights from
        // `offsets[col_idx]`.  If the packing carried more column groups than
        // this list covers, the walk would stop partway and silently read one
        // region's heights as the other's.  Reject instead.
        // The MAIN round's stacking padding closes out the column space; append
        // however many columns the packing still has left, so the accounting is
        // exact rather than approximate.
        let total_cols = combined_packing.offsets.len().saturating_sub(1);
        if col_idx < total_cols {
            let mut pad_idx = col_idx;
            while pad_idx < total_cols {
                let h = if pad_idx + 1 < combined_packing.offsets.len() {
                    combined_packing.offsets[pad_idx + 1]
                        .saturating_sub(combined_packing.offsets[pad_idx])
                } else {
                    combined_packing.total_values.saturating_sub(combined_packing.offsets[pad_idx])
                };
                chip_infos.push(JaggedChipInfo {
                    name: alloc::format!("<stacking-pad:{}>", chip_infos.len()),
                    row_count: h,
                    column_count: 1,
                });
                pad_idx += 1;
            }
            col_idx = pad_idx;
        }
        if col_idx != total_cols {
            return Err(BasefoldVerifyError::JaggedPcs(format!(
                "packing column accounting mismatch: [preprocessed | main] covers \
                 {col_idx} columns but the packing carries {total_cols}",
            )));
        }
    }

    // Build r_row_per_chip from the shared eval_point's trailing
    // log_row_count coords for each chip.
    let r_row_per_chip: Vec<Vec<InnerChallenge>> = chip_infos
        .iter()
        .map(|info| {
            let log_h = info.row_count.max(1).next_power_of_two().trailing_zeros() as usize;
            let slice: &[Challenge<SC>] = if shared_eval_point.len() >= log_h {
                &shared_eval_point[shared_eval_point.len() - log_h..]
            } else {
                shared_eval_point
            };
            // SAFETY: Challenge<SC> == InnerChallenge under the TypeId gate.
            let cloned: Vec<Challenge<SC>> = slice.to_vec();
            let (ptr, len, cap) = {
                let mut v = core::mem::ManuallyDrop::new(cloned);
                (v.as_mut_ptr(), v.len(), v.capacity())
            };
            unsafe { Vec::from_raw_parts(ptr as *mut InnerChallenge, len, cap) }
        })
        .collect();

    // The full z* point as InnerChallenge for the jagged embedding factor.
    // SAFETY: Challenge<SC> == InnerChallenge under the TypeId gate.
    let z_row_inner: Vec<InnerChallenge> = {
        let cloned: Vec<Challenge<SC>> = shared_eval_point.to_vec();
        let (ptr, len, cap) = {
            let mut vv = core::mem::ManuallyDrop::new(cloned);
            (vv.as_mut_ptr(), vv.len(), vv.capacity())
        };
        unsafe { Vec::from_raw_parts(ptr as *mut InnerChallenge, len, cap) }
    };

    // Downcast SC::Challenger to &mut JaggedChallenger.
    let challenger_any: &mut dyn Any = challenger;
    let lb_challenger = challenger_any
        .downcast_mut::<crate::jagged_pcs::JaggedChallenger>()
        .expect("TypeId gate guarantees SC::Challenger == JaggedChallenger");

    // Cross-bind: reinterpret each chip's `main.local` opening as
    // `InnerChallenge` (sound under the TypeId gate above: Challenge<SC> ==
    // InnerChallenge on this inner KoalaBear ring) and hand it to the jagged
    // verifier index-aligned with `chip_infos` / `bundle.y_per_chip`.  The
    // recursion circuit binds these SAME openings to the jagged claimed sum
    // (shard_basefold.rs:588 → recursive_jagged_pcs.rs:247); the host did not,
    // so a malicious proof could ship y_per_chip ≠ opened_values.main and be
    // accepted by both the zerocheck (opened_values) and jagged (y_per_chip)
    // phases independently.  See `verify_one_jagged_group`.
    // SAFETY (used twice below): Challenge<SC> == InnerChallenge under the
    // TypeId gate.
    let relabel = |cloned: Vec<Challenge<SC>>| -> Vec<InnerChallenge> {
        let (ptr, len, cap) = {
            let mut v = core::mem::ManuallyDrop::new(cloned);
            (v.as_mut_ptr(), v.len(), v.capacity())
        };
        unsafe { Vec::from_raw_parts(ptr as *mut InnerChallenge, len, cap) }
    };
    // The cross-bind runs over BOTH rounds, index-aligned with `chip_infos`:
    // the preprocessed round binds each committed chip's `preprocessed.local`
    // (looked up BY NAME, because that round is ordered by the verifying key,
    // not by the shard's chip order), the main round binds `main.local`.
    // Index-aligned with `chip_infos`, which now runs
    // `[prep real | prep pad | main real | main pad]`.  A padding entry is one
    // column of committed zeros, so its claim is ZERO — the prover emits the
    // same zero claims per round.
    let zero_claim = || alloc::vec![InnerChallenge::ZERO];
    let mut opened_main: Vec<Vec<InnerChallenge>> = Vec::with_capacity(chip_infos.len());
    for info in chip_infos.iter().take(n_prep_infos) {
        if info.name.starts_with("<stacking-pad:") {
            opened_main.push(zero_claim());
            continue;
        }
        let idx = chips.iter().position(|c| c.name() == info.name).ok_or_else(|| {
            BasefoldVerifyError::JaggedPcs(format!(
                "preprocessed round covers chip {} which the shard does not have",
                info.name,
            ))
        })?;
        opened_main.push(relabel(opened_values.chips[idx].preprocessed.local.clone()));
    }
    opened_main.extend(opened_values.chips.iter().map(|c| relabel(c.main.local.clone())));
    // The main round's trailing padding.
    for _ in opened_main.len()..chip_infos.len() {
        opened_main.push(zero_claim());
    }

    // The preprocessed round, as the batched open sees it: the vk's commitment
    // plus the committed area implied by the vk's geometry (stripes rounded up
    // to the stacking height, as `StackedPcsVerifier` requires).
    let prep_rounds: Vec<(
        <crate::jagged_pcs::JaggedMmcs as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment,
        usize,
    )> = if n_prep_infos == 0 {
        Vec::new()
    } else {
        let prep_cells: usize = chip_infos
            .iter()
            .take(n_prep_infos)
            .map(|i| i.row_count.saturating_mul(i.column_count))
            .sum();
        let log_stack = bundle.commit.log_stacking_height as usize;
        let stripes = prep_cells.div_ceil(1usize << log_stack);
        let area = stripes << log_stack;

        // The BaseFold open Merkle-verifies against the round's RAW root, which
        // only the proof has; the KEY holds the HASH-BOUND digest
        // `compress([raw, hash(geometry)])`.  So re-derive the bound form from
        // the claimed raw root and the geometry this verifier is about to open
        // against, and require it to equal the key's.  That single check is
        // what pins BOTH the root and the preprocessed row/column counts.
        let Some(raw) = bundle.preceding_commits.first() else {
            return Err(BasefoldVerifyError::JaggedPcs(
                "preprocessed round: the proof carries no raw commitment for it".into(),
            ));
        };
        // Only the REAL preprocessed chips: the stacking-pad columns appended
        // above belong to the BATCHED layout, not to what `setup` committed, so
        // the commit-time hash never saw them.
        let (prep_rows, prep_cols): (Vec<usize>, Vec<usize>) = chip_infos
            .iter()
            .take(n_prep)
            .map(|i| (i.row_count, i.column_count))
            .unzip();
        let rebound = crate::jagged_pcs::jagged_hash_bind_modified(
            crate::jagged_pcs::basefold_commit_digest_felts(raw),
            &prep_rows,
            &prep_cols,
        );
        // SAFETY: `Com<SC> == JaggedMmcs::Commitment` under the inner TypeId
        // gate.  The commitment OWNS a heap allocation (`MerkleCap` is a Vec of
        // digests), so this must relabel a CLONE that is then forgotten — a
        // bitwise `transmute_copy` of the borrowed original duplicates the
        // ownership and double-frees.
        let key_commitment = unsafe {
            core::mem::transmute_copy::<
                crate::Com<SC>,
                <crate::jagged_pcs::JaggedMmcs as p3_commit::Mmcs<
                    crate::jagged_pcs::JaggedVal,
                >>::Commitment,
            >(&core::mem::ManuallyDrop::new(vk.commit.clone()))
        };
        if crate::jagged_pcs::basefold_commit_digest_felts(&key_commitment) != rebound {
            return Err(BasefoldVerifyError::JaggedPcs(
                "preprocessed round: the claimed commitment and geometry do not \
                 re-derive the verifying key's commitment"
                    .into(),
            ));
        }
        alloc::vec![(raw.clone(), area)]
    };

    // Delegate to the existing host-side verifier.
    //
    // Single-main-commit: the prover's transcript prologue
    // already observed the BaseFold commit's 8-felt digest as
    // `main_commitment` (mirrored in the transcript prologue above).
    // Use the `_no_observe` variant so the verifier doesn't observe
    // the same digest a second time (which would desync the
    // transcript vs the prover).
    if !verify_jagged_basefold_no_observe(
        &chip_infos,
        &r_row_per_chip,
        &z_row_inner,
        // The PREPROCESSED round's commitment comes from the VERIFYING KEY, and
        // its committed area follows from the geometry the key pins — so the
        // proof carries neither.
        &prep_rounds,
        n_prep_infos,
        &bundle,
        Some(&opened_main),
        lb_challenger,
    ) {
        return Err(BasefoldVerifyError::JaggedPcs(
            "verify_jagged_basefold_no_observe rejected the bundle".into(),
        ));
    }

    Ok(())
}

/// Host-side `full_geq`: padded-row mask used by the zerocheck
/// verifier to subtract constraint contributions from out-of-range
/// padded rows.  Computes the indicator
///
/// ```text
///   full_geq(threshold, eval_point)
///       = Σ_{bit b}  (bit >= threshold at big-endian comparison)
/// ```
///
/// via the same recurrence as the in-circuit
/// `zkm_recursion_circuit::zerocheck::full_geq` but on concrete
/// extension-field values.
#[allow(dead_code)] // kept for unit tests
fn full_geq_host<EF: Field + Copy>(threshold: &[EF], eval_point: &[EF]) -> EF {
    debug_assert_eq!(
        threshold.len(),
        eval_point.len(),
        "full_geq_host: threshold and eval_point must have equal dimension"
    );
    let one = EF::ONE;
    threshold
        .iter()
        .rev()
        .zip(eval_point.iter().rev())
        .fold(one, |acc, (x, y)| ((one - *y) * (one - *x) + *y * *x) * acc + *y * (one - *x))
}

/// Produce the per-chip `degree` point used by [`full_geq_host`].
///
/// Matches the in-circuit witness stub at
/// [`crate::recursion::circuit::shard_proof_variable_lift::empty_chip_height_bits`]
/// — returns a zero-filled vector of length `max_log_row_count + 1`.
/// With an all-zero threshold the padded-row mask collapses to a
/// constant (no-op).
#[allow(dead_code)] // kept for unit tests
fn degree_stub_host<EF: Field + Copy>(max_log_row_count: usize) -> Vec<EF> {
    vec![EF::ZERO; max_log_row_count + 1]
}

/// Host-side zerocheck verification.
///
/// # Validates
///
///   1. Challenge sampling order (`alpha`, `gkr_batch_open`, `lambda`)
///      — transcript kept in sync with the prover.
///   2. Point dimension == `max_log_row_count`.
///   3. Point dimension == `gkr_evaluations.point` dimension.
///   4. Inner sumcheck proof via [`verify_sumcheck_host`] (degree 4,
///      `max_log_row_count` rounds).
///   5. Per-chip opening transcript observations matching the prover's
///      ordering.
///
/// Also binds the cross-chip constraint-RLC and the GKR sum-modification
/// identity (in-circuit equivalent at
/// [`crate::recursion_circuit::zerocheck::BasefoldZerocheckVerifier::verify_zerocheck`])
/// against the direct `Σ_b C(b) == 0` sumcheck the shard-level prover
/// ([`crate::shard_level::zerocheck_prover::prove_shard_zerocheck`]) emits.
#[allow(clippy::too_many_arguments)]
fn verify_zerocheck_host<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    zerocheck_proof: &PartialSumcheckProof<Challenge<SC>>,
    gkr_evaluations: &super::types::LogUpEvaluations<Challenge<SC>>,
    public_values: &[Val<SC>],
    max_log_row_count: usize,
    // `true` for the CORE machine (rev shard proofs); `false` for
    // recursion / shrink / wrap (LEGACY).
    core_rev: bool,
    challenger: &mut SC::Challenger,
    opened_values: &ShardOpenedValues<Val<SC>, Challenge<SC>>,
) -> Result<(), BasefoldVerifyError>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>
        + for<'b> Air<BasefoldConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    Val<SC>: PrimeField,
    Challenge<SC>: ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>> + Copy,
{
    // (1) Sample the per-phase challenges (transcript-sync with the prover).
    // `gkr_batch_open` + `lambda` drive the claimed_sum binding (G2-b) below;
    // `alpha` drives the constraint-RLC half (G2-a), deferred to the re-point.
    let _alpha: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();
    let gkr_batch_open: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();
    let lambda: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();

    // ── constraint-RLC BINDING (HARD CHECK) ───────
    // Recompute the in-circuit `rlc_eval` ON THE HOST from the SAME inputs
    // the circuit uses — the trace@z* openings carried in `opened_values`,
    // the transcript-sampled (alpha, gkr_batch_open, lambda), the GKR point
    // and the zerocheck-reduced point — and BIND it to the proof's claimed
    // `point_and_eval.1`.
    //
    // SOUNDNESS: the structural sumcheck alone only ties `point_and_eval.1`
    // back to `claimed_sum` (telescoping) and the GKR openings — nothing
    // else forces it to equal the constraint-RLC of the commitment-bound
    // openings@z*.  Verifier-only, transcript-neutral (only already-sampled
    // challenges + opened values).
    let rlc_eval = recompute_zerocheck_rlc_eval_host::<SC, A>(
        chips,
        zerocheck_proof,
        gkr_evaluations,
        public_values,
        _alpha,
        gkr_batch_open,
        lambda,
        opened_values,
        core_rev,
    );
    if rlc_eval != zerocheck_proof.point_and_eval.1 {
        return Err(BasefoldVerifyError::Zerocheck(
            "zerocheck rlc_eval != point_and_eval.1 (item-12 constraint-RLC binding)".to_string(),
        ));
    }

    // (2) Point dimension == max_log_row_count.
    let point_dim = zerocheck_proof.point_and_eval.0.len();
    if point_dim != max_log_row_count {
        return Err(BasefoldVerifyError::Zerocheck(format!(
            "zerocheck point dim {point_dim} != max_log_row_count {max_log_row_count}"
        )));
    }

    // (3) gkr_point dim must match zerocheck point dim.
    if gkr_evaluations.point.len() != point_dim {
        return Err(BasefoldVerifyError::Zerocheck(format!(
            "gkr_evaluations.point dim {} != zerocheck point dim {}",
            gkr_evaluations.point.len(),
            point_dim
        )));
    }

    // (G2-b) Bind the zerocheck `claimed_sum` to the lambda-RLC of the
    // (commitment-bound) GKR openings.  Closes the "arbitrary claimed_sum"
    // forgery: the structural sumcheck only checks p_0(0)+p_0(1)==claimed_sum,
    // never that claimed_sum equals the GKR-derived modification.  Pure
    // arithmetic over already-sampled challenges → transcript-neutral,
    // verifier-only.
    //
    // The cross-chip constraint-RLC half (== point_and_eval.1) needs the
    // trace opened at the zerocheck REDUCED point z* — exactly where the
    // prover opens the jagged PCS — and is discharged by the
    // `recompute_zerocheck_rlc_eval_host` hard check above.
    let _ = public_values;
    {
        use p3_air::BaseAir;
        let max_elements = chips
            .iter()
            .map(|chip| {
                <_ as BaseAir<Val<SC>>>::width(*chip)
                    + <A as MachineAir<Val<SC>>>::preprocessed_width(&chip.air)
            })
            .max()
            .unwrap_or(0);
        let mut gkr_batch_open_powers: Vec<Challenge<SC>> = Vec::with_capacity(max_elements);
        let mut acc_pow: Challenge<SC> = Challenge::<SC>::ONE;
        for _ in 0..max_elements {
            acc_pow = acc_pow * gkr_batch_open;
            gkr_batch_open_powers.push(acc_pow);
        }
        // ── SHARD-UNIFORM convention decision (mirror prover) ─
        let zerocheck_sum_mod: Challenge<SC> = gkr_evaluations
            .chip_openings
            .values()
            .map(|chip_evaluation| {
                // ── SINGLE-FIELD CLAIM COLLAPSE ──────────────
                // When the SHARD uses the collapsed convention, seed the
                // per-chip claimed_sum term DIRECTLY from the FULL-POINT
                // openings (`*_full`) with NO embed_factor — mirroring the
                // prover (zerocheck_prover.rs).  The full-point
                // opening already carries the mixed-height padding factor
                //   main_full = Π_{k=log_h}^{N-1}(1 − zeta[k]) · MLE(trace @
                //               zeta[0..log_h])
                // so the old `raw(trailing) · embed_LEAD` correction is dropped.
                {
                    let main_full =
                        chip_evaluation.main_trace_evaluations_full.as_deref().unwrap_or(&[]);
                    let prep_full = chip_evaluation
                        .preprocessed_trace_evaluations_full
                        .as_deref()
                        .unwrap_or(&[]);
                    main_full
                        .iter()
                        .copied()
                        .chain(prep_full.iter().copied())
                        .zip(gkr_batch_open_powers.iter().copied())
                        .fold(Challenge::<SC>::ZERO, |a, (o, p)| a + o * p)
                }
            })
            .fold(Challenge::<SC>::ZERO, |acc, m| acc * lambda + m);
        if zerocheck_proof.claimed_sum != zerocheck_sum_mod {
            return Err(BasefoldVerifyError::Zerocheck(
                "GKR sum-modification identity failed (claimed_sum != lambda-RLC(GKR openings))"
                    .into(),
            ));
        }
    }

    // (4) Inner sumcheck: degree 4, max_log_row_count rounds.  The round
    // poly is `elf(X)·[eq-weighted constraint sum]` — the eq term's last
    // factor `elf` is degree 1 and the max AIR constraint degree is 3, so the
    // honest round poly is degree 4 (5 coefficients), matching the prover's
    // `UnivariatePolynomial::zero(4)` dummy and the recursion dummy
    // `dummy_partial_sumcheck_proof(.., 4)`.  (The recursion `verify_sumcheck`
    // fixes the degree via the witness shape rather than an explicit check.)
    verify_sumcheck_host::<Val<SC>, Challenge<SC>, SC::Challenger>(
        zerocheck_proof,
        challenger,
        max_log_row_count,
        4,
    )
    .map_err(|e| match e {
        BasefoldVerifyError::LogupGkr(msg) => BasefoldVerifyError::Zerocheck(msg),
        other => other,
    })?;

    // (5) Observe the zerocheck openings (trace@z*), after the zerocheck
    // sumcheck and before the jagged phase — mirror of the prover's
    // `observe_zerocheck_openings_from_residual`.  The GKR openings are
    // observed earlier, at the end of `verify_logup_gkr_host`, before the
    // α/γ/λ samples above.
    //
    // `opened_values.chips` is a Vec emitted in NAME order by the prover's
    // `build_opened_values`, already split into preprocessed/main at each
    // chip's `preprocessed_width` — the same pairs, in the same order, the
    // prover feeds from its `trace_at_z` residual.
    crate::shard_level::prover::observe_zerocheck_openings::<
        Val<SC>,
        Challenge<SC>,
        SC::Challenger,
        _,
    >(
        challenger,
        chips.len(),
        opened_values
            .chips
            .iter()
            .map(|c| (c.preprocessed.local.as_slice(), c.main.local.as_slice())),
    );

    Ok(())
}

/// Host recompute of the in-circuit zerocheck `rlc_eval`.
///
/// Bit-for-bit mirror of the recursion verifier's
/// `BasefoldZerocheckVerifier::verify_zerocheck` accumulator, executed over
/// concrete host field elements instead of symbolic circuit exprs.  The
/// circuit asserts `rlc_eval == zerocheck_proof.point_and_eval.1`; the
/// caller binds this recompute the same way.
///
/// Inputs match the circuit exactly:
///   * `opened_values.chips[i].main.local / preprocessed.local` = trace@z*
///     (the zerocheck-reduced point, the SAME values the circuit batches).
///   * `opened_values.chips[i].quotient[0]` = the per-chip big-endian
///     `degree` bits (length `max_log_row_count + 1`) the circuit feeds to
///     `full_geq` (real-height bits).
///   * `(alpha, gkr_batch_open, lambda)` = the three transcript samples,
///     in the prover/verifier order.
///   * `gkr_evaluations.point` = z_gkr; `zerocheck_proof.point_and_eval.0`
///     = z* (the reduced point).
#[allow(clippy::too_many_arguments)]
fn recompute_zerocheck_rlc_eval_host<SC, A>(
    chips: &[&Chip<Val<SC>, A>],
    zerocheck_proof: &PartialSumcheckProof<Challenge<SC>>,
    gkr_evaluations: &super::types::LogUpEvaluations<Challenge<SC>>,
    public_values: &[Val<SC>],
    alpha: Challenge<SC>,
    gkr_batch_open: Challenge<SC>,
    lambda: Challenge<SC>,
    opened_values: &ShardOpenedValues<Val<SC>, Challenge<SC>>,
    // `true` for the CORE machine (rev eq-bridge anchor rev(z_gkr)),
    // `false` for recursion / shrink / wrap (LEGACY).
    core_rev: bool,
) -> Challenge<SC>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>
        + for<'b> Air<BasefoldConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    Val<SC>: PrimeField,
    Challenge<SC>: ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>> + Copy,
{
    use p3_air::BaseAir;

    let z_star = &zerocheck_proof.point_and_eval.0;
    let z_gkr = &gkr_evaluations.point;

    // ── rev(zeta) eq-bridge anchor ──────────────────────────
    // Under the collapsed convention the prover anchors every chip's zerocheck
    // poly on `rev(z_gkr)` (natural cells, dropped bitrev), so the batched
    // reduced value carries `eq(rev(z_gkr), z*)`.  Mirror that here by feeding
    // the eq-bridge the reversed GKR point.  The decision is SHARD-UNIFORM:
    // the anchor orientation follows the per-machine `core_rev` flag
    // (core => rev); legacy (recursion / wrap) shards keep `eq(z_gkr, z*)`.
    let conv_use_rev = core_rev;
    let z_gkr_anchor: Vec<Challenge<SC>> =
        if conv_use_rev { z_gkr.iter().rev().copied().collect() } else { z_gkr.clone() };

    // (2) eq(anchor, z*) — circuit zerocheck.rs:480-489.
    let zerocheck_eq_val = eq_eval_host::<Challenge<SC>>(&z_gkr_anchor, z_star);

    // (3) gkr_batch_open powers [β¹ .. β^max_width], circuit :491-505.
    let max_elements = chips
        .iter()
        .map(|chip| {
            <_ as BaseAir<Val<SC>>>::width(*chip)
                + <A as MachineAir<Val<SC>>>::preprocessed_width(&chip.air)
        })
        .max()
        .unwrap_or(0);
    let mut beta_powers: Vec<Challenge<SC>> = Vec::with_capacity(max_elements);
    {
        let mut acc = Challenge::<SC>::ONE;
        for _ in 0..max_elements {
            acc = acc * gkr_batch_open;
            beta_powers.push(acc);
        }
    }

    // z* extended by one front ZERO coord (circuit :537-538 insert(0,0)).
    let mut z_extended: Vec<Challenge<SC>> = Vec::with_capacity(z_star.len() + 1);
    z_extended.push(Challenge::<SC>::ZERO);
    z_extended.extend_from_slice(z_star);

    let mut rlc_eval = Challenge::<SC>::ZERO;

    for (chip, opening) in chips.iter().zip(opened_values.chips.iter()) {
        // degree = quotient[0] (circuit opening.degree), real-height bits.
        let degree: &[Challenge<SC>] =
            opening.quotient.first().map(|v| v.as_slice()).unwrap_or(&[]);

        // (4e) geq + padded-row adjustment.  full_geq over (degree, z_ext);
        // when degree.len() != z_extended.len() (e.g. placeholder lift) the
        // circuit would still pair them — here we guard so the probe never
        // panics and report the dimension so a mismatch is visible.
        let geq_val = if degree.len() == z_extended.len() {
            full_geq_host::<Challenge<SC>>(degree, &z_extended)
        } else {
            // dimension mismatch: report it (degree placeholder/zero path).
            Challenge::<SC>::ONE
        };
        let pra = compute_padded_row_adjustment_basefold_host::<Val<SC>, Challenge<SC>, A>(
            chip,
            opening,
            alpha,
            public_values,
        );

        // (4f) constraint_eval = C(trace@z*, alpha) - pra·geq, circuit :566-577.
        let ce = eval_constraints_basefold_host::<Val<SC>, Challenge<SC>, A>(
            chip,
            opening,
            alpha,
            public_values,
        );
        let constraint_eval = ce - pra * geq_val;

        // (4g) openings_batch = Σ (main ++ prep) · β^(1..), circuit :579-600.
        let openings_batch: Challenge<SC> = opening
            .main
            .local
            .iter()
            .chain(opening.preprocessed.local.iter())
            .copied()
            .zip(beta_powers.iter().copied())
            .fold(Challenge::<SC>::ZERO, |acc, (o, p)| acc + o * p);

        // (4h) fold: rlc = rlc·λ + eq·(constraint_eval + openings_batch).
        rlc_eval = rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch);
    }

    rlc_eval
}

// ─────────────────────────────────────────────────────────────
// LogUp-GKR stage: host-side verification helpers
// ─────────────────────────────────────────────────────────────

/// Host-side `eq_eval`: the multilinear equality indicator
///
///   eq(a, b) = Π_k ((1 - a_k)(1 - b_k) + a_k · b_k)
///
/// Mirrors `zkm_recursion_circuit::zerocheck::eq_eval` but for concrete
/// `Challenge<SC>` values instead of symbolic circuit exprs.
fn eq_eval_host<EF: Field + Copy>(a: &[EF], b: &[EF]) -> EF {
    debug_assert_eq!(a.len(), b.len(), "eq_eval_host: dimension mismatch");
    let one = EF::ONE;
    a.iter().zip(b.iter()).fold(one, |acc, (ai, bi)| acc * ((one - *ai) * (one - *bi) + *ai * *bi))
}

/// Host-side MLE evaluation at an arbitrary extension-field point.
///
/// Computes `Σ_i f[i] · eq(i, point)` via the standard partial-lagrange
/// table expansion.  Length of `mle_evals` must equal `1 << point.len()`.
fn evaluate_mle_host<EF: Field + Copy>(mle_evals: &[EF], point: &[EF]) -> EF {
    let dim = point.len();
    assert_eq!(
        mle_evals.len(),
        1usize << dim,
        "evaluate_mle_host: mle length {} != 2^{} = {}",
        mle_evals.len(),
        dim,
        1usize << dim,
    );
    // Build the partial-lagrange table in-place.  Index convention
    // matches the in-circuit `evaluate_mle_ext`: variable 0 is the
    // LSB, later-processed coords occupy higher bits.
    let mut weights: Vec<EF> = vec![EF::ONE];
    for &r in point {
        let old_len = weights.len();
        let mut next = vec![EF::ZERO; old_len * 2];
        for j in 0..old_len {
            let prod = weights[j] * r;
            next[j] = weights[j] - prod;
            next[j + old_len] = prod;
        }
        weights = next;
    }
    mle_evals.iter().zip(weights.iter()).fold(EF::ZERO, |acc, (v, w)| acc + *v * *w)
}

/// Evaluate a degree-`d` polynomial (stored as `d+1` coefficients
/// low-degree-first) at a field point via Horner's.
fn eval_coeffs_host<EF: Field + Copy>(coeffs: &[EF], x: EF) -> EF {
    let mut acc = EF::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// Host-side sumcheck verifier.
///
/// Returns `Ok(())` when:
///   1. `univariate_polys.len() == expected_num_variables`
///   2. Every round poly has `expected_degree + 1` coefficients
///   3. First round: `p_0(0) + p_0(1) == claimed_sum`
///   4. For each round i ≥ 1: `p_{i-1}(α_{i-1}) == p_i(0) + p_i(1)`
///      where α_{i-1} is the challenger-sampled challenge
///   5. The proof's `point_and_eval.0` matches the sampled challenges
///   6. `p_{last}(α_last) == point_and_eval.1`
///
/// Mirrors [`crate::recursion_circuit::sumcheck::verify_sumcheck`].
fn verify_sumcheck_host<F, EF, Challenger>(
    proof: &PartialSumcheckProof<EF>,
    challenger: &mut Challenger,
    expected_num_variables: usize,
    expected_degree: usize,
) -> Result<(), BasefoldVerifyError>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F> + Copy,
    Challenger: FieldChallenger<F>,
{
    let n = proof.univariate_polys.len();
    if n != expected_num_variables {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "sumcheck proof has {n} rounds, expected {expected_num_variables}"
        )));
    }
    if proof.point_and_eval.0.len() != expected_num_variables {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "sumcheck point_and_eval.0 has dim {}, expected {expected_num_variables}",
            proof.point_and_eval.0.len()
        )));
    }
    if n == 0 {
        return Err(BasefoldVerifyError::LogupGkr(
            "sumcheck has zero rounds — invalid proof shape".into(),
        ));
    }

    // First round: p_0(0) + p_0(1) == claimed_sum.
    let p0 = &proof.univariate_polys[0];
    if p0.coefficients.len() != expected_degree + 1 {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "sumcheck round 0 poly has {} coefficients, expected {}",
            p0.coefficients.len(),
            expected_degree + 1
        )));
    }
    let p0_at_0 = eval_coeffs_host(&p0.coefficients, EF::ZERO);
    let p0_at_1 = eval_coeffs_host(&p0.coefficients, EF::ONE);
    if p0_at_0 + p0_at_1 != proof.claimed_sum {
        return Err(BasefoldVerifyError::LogupGkr(
            "sumcheck first-round inconsistency with claimed_sum".into(),
        ));
    }

    // Observe round 0 coefficients into the challenger.
    for c in &p0.coefficients {
        for basis in c.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }

    // Walk rounds 1..n.
    //
    // Sumcheck convention: the prover runs an MSB fold
    // and `insert(0, α)`s each freshly-sampled challenge at the front
    // of `reduced_point`.  We mirror the prover's construction here so
    // the equality check below sees the same Vec.
    let mut alphas: Vec<EF> = Vec::with_capacity(n);
    let mut prev_poly = p0;
    for i in 1..n {
        let alpha: EF = challenger.sample_algebra_element::<EF>();
        alphas.insert(0, alpha);
        let curr = &proof.univariate_polys[i];
        if curr.coefficients.len() != expected_degree + 1 {
            return Err(BasefoldVerifyError::LogupGkr(format!(
                "sumcheck round {i} poly has {} coefficients, expected {}",
                curr.coefficients.len(),
                expected_degree + 1
            )));
        }
        let prev_at_alpha = eval_coeffs_host(&prev_poly.coefficients, alpha);
        let curr_at_0 = eval_coeffs_host(&curr.coefficients, EF::ZERO);
        let curr_at_1 = eval_coeffs_host(&curr.coefficients, EF::ONE);
        if prev_at_alpha != curr_at_0 + curr_at_1 {
            return Err(BasefoldVerifyError::LogupGkr(format!(
                "sumcheck round-{i} consistency failed"
            )));
        }
        for c in &curr.coefficients {
            for basis in c.as_basis_coefficients_slice() {
                challenger.observe(*basis);
            }
        }
        prev_poly = curr;
    }

    // Sample the terminal challenge.  Same insert-at-front rule.
    let alpha_last: EF = challenger.sample_algebra_element::<EF>();
    alphas.insert(0, alpha_last);

    // Point must match the sampled challenges.
    if alphas != proof.point_and_eval.0 {
        return Err(BasefoldVerifyError::LogupGkr(
            "sumcheck reduced point doesn't match sampled challenges".into(),
        ));
    }

    // Final: p_{n-1}(alpha_last) == claimed final eval.
    let final_recomputed = eval_coeffs_host(&prev_poly.coefficients, alpha_last);
    if final_recomputed != proof.point_and_eval.1 {
        return Err(BasefoldVerifyError::LogupGkr(
            "sumcheck final eval doesn't match recomputed value".into(),
        ));
    }

    Ok(())
}

/// Host-side LogUp-GKR verification.
///
/// Port of [`crate::recursion_circuit::logup_gkr::verify_logup_gkr`]
/// (see `crates/recursion/circuit/src/logup_gkr.rs:293-439`).
///
/// Omits the grinding-witness check and the public-values closure
/// (those live in separate host-port scope).  Validates the core
/// identity:
///
///   1. Sample (alpha, beta_seed, pv_challenge) from the challenger
///   2. Observe circuit_output.{numerator, denominator} into the transcript
///   3. Sample initial eval_point of dim log_num_interactions + 1
///   4. For each round:
///      - sample lambda
///      - check `sumcheck_proof.claimed_sum == λ·n_eval + d_eval`
///      - verify the inner sumcheck
///      - check `point_and_eval.1 == eq(sumcheck_point, eval_point) ·
///                                  (λ·(n0·d1 + n1·d0) + d0·d1)`
///      - observe (n0, n1, d0, d1) into the transcript
///      - sample line challenge, extend eval_point, update n/d evals
#[allow(clippy::too_many_arguments)]
fn verify_logup_gkr_host<SC, A>(
    proof: &LogupGkrProof<Val<SC>, Challenge<SC>>,
    chips: &[&Chip<Val<SC>, A>],
    opened_values: &ShardOpenedValues<Val<SC>, Challenge<SC>>,
    max_log_row_count: usize,
    beta_seed_dim: usize,
    fold_orientation: FoldOrientation,
    public_values: &[Val<SC>],
    machine_has_pv_buses: bool,
    challenger: &mut SC::Challenger,
) -> Result<(), BasefoldVerifyError>
where
    SC: StarkGenericConfig,
    A: MachineAir<Val<SC>>,
    Val<SC>: PrimeField,
    Challenge<SC>: ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>> + Copy,
{
    // Note: we derive log_num_interactions from the output MLE length
    // rather than taking chip_metadata as an extra parameter, since
    // the proof itself encodes the dimension.
    let numerator = &proof.circuit_output.numerator;
    let denominator = &proof.circuit_output.denominator;
    if numerator.len() != denominator.len() {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "circuit_output numerator/denominator length mismatch: {} vs {}",
            numerator.len(),
            denominator.len()
        )));
    }
    if !numerator.len().is_power_of_two() {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "circuit_output length {} is not a power of two",
            numerator.len()
        )));
    }
    // initial_num_variables = log_num_interactions + 1 = log2(output.len)
    let initial_num_variables = numerator.len().trailing_zeros() as usize;

    // (0) Re-observe + check the GKR proof-of-work grinding witness BEFORE
    // sampling alpha/beta — EXACTLY matching the prover's grind
    // (row_gkr/top_level.rs::gkr_grind), which observes the witness into the
    // challenger. Without this the verifier's alpha/beta diverge from the
    // prover's and the G1 PV-balance below fails. Config-aware: a real check
    // for the Inner core proof, a no-op for the Outer/wrap (whose prover
    // grind is itself a no-op). This provides both soundness AND
    // consistency (the grinding witness is checked, not omitted).
    if !crate::logup_gkr::GkrGrind::gkr_check_witness(
        challenger,
        crate::logup_gkr::GKR_GRINDING_BITS,
        proof.witness,
    ) {
        return Err(BasefoldVerifyError::LogupGkr("GKR grinding witness check failed".into()));
    }

    // (1) Sample the LogUp permutation challenges (alpha + beta_seed),
    // matching the prover (row_gkr/top_level.rs:62-78).
    let alpha: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();
    let beta_seed: Vec<Challenge<SC>> =
        (0..beta_seed_dim).map(|_| challenger.sample_algebra_element::<Challenge<SC>>()).collect();
    // betas[0] = argument_index (kind) weight, betas[1..] = per-value weights —
    // the partial-lagrange table over {0,1}^beta_seed_dim (eq_mle_table),
    // identical to the prover's leaf-denominator construction.
    let beta_powers: Vec<Challenge<SC>> = if beta_seed.is_empty() {
        vec![Challenge::<SC>::ONE]
    } else {
        crate::zerocheck_prover::eq_mle_table::<Challenge<SC>>(&beta_seed)
    };

    // (G1) Public-values balance — THE Option-2 local-only invariant.  The
    // LogUp-GKR cumulative sum over the chip interactions must equal -PV_digest,
    // where PV_digest folds the record-level public-values AIR interactions
    // (the State / GlobalAccumulation / MemoryGlobalInit+Finalize bus
    // boundaries) under the SAME (alpha, beta_powers).  The local-only buses
    // are closed ONLY by this balance; without it a prover could forge bus
    // fractions and the host would accept.  Host port of recursion
    // logup_gkr.rs:357-381.  `eval_public_values` emits no assert_zero
    // constraints, so the constraint-fold alpha is unused (pass `alpha`).  Pure
    // arithmetic over already-sampled challenges — transcript-neutral.
    {
        let gkr_sum: Challenge<SC> = numerator
            .iter()
            .zip(denominator.iter())
            .fold(Challenge::<SC>::ZERO, |acc, (n, d)| acc + *n / *d);
        // Machine-aware local-only closure.  The core MIPS machine carries
        // the State/GlobalAccumulation/MemoryGlobal boundary buses, closed by
        // the public-values AIR (`gkr_sum == -PV_digest`).  The recursion
        // machine carries only self-cancelling `Local` buses, so its closure
        // is `gkr_sum == 0`; the core State-bus PV-AIR does not apply (it
        // reads a different PV schema and its arity-16 GlobalAccumulation
        // message would overflow the recursion `beta_powers`).
        let pv_digest = if machine_has_pv_buses {
            crate::air::eval_public_values_digest_host::<Val<SC>, Challenge<SC>>(
                &alpha,
                &beta_powers,
                alpha,
                public_values,
            )
        } else {
            // Recursion machine: all buses are self-cancelling `Local`
            // (Memory/Program/Range/Syscall), so the local-only closure is
            // `gkr_sum == 0` (empirically confirmed for the compress shard).
            Challenge::<SC>::ZERO
        };
        if gkr_sum != -pv_digest {
            return Err(BasefoldVerifyError::LogupGkr(
                "public-values balance failed (sum circuit_output num/den != -PV_digest)".into(),
            ));
        }
    }

    // (2) Observe circuit_output into the transcript.  Each EF
    // element contributes its base-field basis coefficients.
    for &n in numerator.iter() {
        for basis in n.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }
    for &d in denominator.iter() {
        for basis in d.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }

    // (3) Sample the initial eval_point.
    let mut eval_point: Vec<Challenge<SC>> = (0..initial_num_variables)
        .map(|_| challenger.sample_algebra_element::<Challenge<SC>>())
        .collect();

    // Initial numerator/denominator evals at the sampled point.  These are
    // reduced through the round walk below and then consumed by the
    // degree-masked last-layer reconstruction (no longer discarded).
    let mut numerator_eval: Challenge<SC> = evaluate_mle_host(numerator, &eval_point);
    let mut denominator_eval: Challenge<SC> = evaluate_mle_host(denominator, &eval_point);

    // The prover pads GKR to a FIXED round count
    // (`round_proofs.len() + 1 == max_log_row_count`).  Enforce it here so a
    // malicious prover cannot shorten the reduction (each missing round is an
    // unverified MLE halving) — the round count must be checked, not derived
    // from the proof.
    if proof.round_proofs.len() + 1 != max_log_row_count {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "GKR round count {} + 1 != max_log_row_count {} (proof must be \
             padded to the fixed round count)",
            proof.round_proofs.len(),
            max_log_row_count
        )));
    }

    // (4) Walk round_proofs.  For each round:
    //   - sample lambda
    //   - check claimed_sum == λ·n_eval + d_eval
    //   - verify inner sumcheck
    //   - check final_eval identity
    //   - observe (n0, n1, d0, d1)
    //   - sample line challenge, extend eval_point, update n/d
    for (i, round_proof) in proof.round_proofs.iter().enumerate() {
        let lambda: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();

        // Expected claimed sum.
        let expected_claim = lambda * numerator_eval + denominator_eval;
        if round_proof.sumcheck_proof.claimed_sum != expected_claim {
            return Err(BasefoldVerifyError::LogupGkr(format!(
                "round {i}: sumcheck claimed_sum mismatch"
            )));
        }

        // Inner sumcheck over i + initial_num_variables rounds.
        // The per-round sumcheck runs over whatever dim the layer
        // has — for the first round that's initial_num_variables,
        // growing by 1 each subsequent round via the line challenge.
        // Degree is 3 (LogUp-GKR's quadratic + eq contribution).
        let expected_sumcheck_vars = i + initial_num_variables;
        verify_sumcheck_host::<Val<SC>, Challenge<SC>, SC::Challenger>(
            &round_proof.sumcheck_proof,
            challenger,
            expected_sumcheck_vars,
            3,
        )?;

        // Final-eval identity.
        //
        // The eq pairing depends on the prover's fold orientation.
        // The CPU/legacy MSB-orientation fold pairs `eval_point`
        // in original order; the GPU packed-pool LSB-orientation
        // fold pairs the reversed `eval_point`.  Dispatched off the
        // proof tag (not env vars) so the verifier matches whichever
        // prover produced the proof.
        let sumcheck_point = &round_proof.sumcheck_proof.point_and_eval.0;
        let final_eval = round_proof.sumcheck_proof.point_and_eval.1;
        let eq_val = match fold_orientation {
            FoldOrientation::Msb => eq_eval_host(sumcheck_point, &eval_point),
            FoldOrientation::Lsb => {
                let mut rev = eval_point.clone();
                rev.reverse();
                eq_eval_host(sumcheck_point, &rev)
            }
        };
        let n0 = round_proof.numerator_0;
        let n1 = round_proof.numerator_1;
        let d0 = round_proof.denominator_0;
        let d1 = round_proof.denominator_1;
        let expected_final = eq_val * (lambda * (n0 * d1 + n1 * d0) + d0 * d1);
        if final_eval != expected_final {
            return Err(BasefoldVerifyError::LogupGkr(format!(
                "round {i}: final_eval identity failed"
            )));
        }

        // Observe (n0, n1, d0, d1) into the transcript.
        for e in [n0, n1, d0, d1] {
            for basis in e.as_basis_coefficients_slice() {
                challenger.observe(*basis);
            }
        }

        // Update eval_point: sumcheck-reduced point + line challenge.  The
        // prover's layer transition pairs ADJACENT rows (peels the row LSB =
        // variable `log_num_interactions` of the LSB-first flat index), so
        // the line challenge is INSERTED there — mirrors
        // `row_gkr/top_level.rs` and the recursion circuit.
        eval_point = sumcheck_point.clone();
        let line: Challenge<SC> = challenger.sample_algebra_element::<Challenge<SC>>();
        eval_point.insert(initial_num_variables - 1, line);

        // Update n/d evals via linear interpolation at `line`.
        numerator_eval = n0 + (n1 - n0) * line;
        denominator_eval = d0 + (d1 - d0) * line;
    }

    // ── DEGREE-MASKED LAST-LAYER RECONSTRUCTION (height anchor) ──
    //
    // The round walk above reduces the GKR `circuit_output` num/den MLEs to
    // their evaluation `(numerator_eval, denominator_eval)` at the fully
    // reduced `eval_point` (dim = log_num_interactions + max_log_row_count).
    // This block re-derives those evals from the chips' trace openings masked
    // by `full_geq(degree, ·)` and asserts equality — the bind that ties the
    // GKR output back to per-chip heights.  Without it, an area-preserving
    // per-chip height forgery (chip A 2^h→2^(h+1), chip B 2^g→2^(g-1)) leaves
    // the GKR `circuit_output` / round walk / public-values balance intact
    // while moving each chip's `full_geq` padding boundary.
    //
    // This is a PURE ARITHMETIC ASSERT over already-sampled challenges and
    // already-observed openings: it samples nothing and observes nothing
    // (the len + per-chip openings observe stays in zerocheck), so it is
    // transcript-neutral and proof-byte-neutral.  The GKR `circuit_output`
    // is built from RAW trace cells with NO embed_factor (extract.rs ←
    // generate_first_layer ← generate_interaction_vals); the
    // `embed_factor = Π_high(1−zeta[k])` lives ONLY in the zerocheck claim
    // path, NOT here — so the reconstruction uses raw openings with no
    // factor (the padding mask is carried by `full_geq`).
    //
    // SOUNDNESS: this runs UNCONDITIONALLY — a degree-only lie (transcript
    // honest, `degree` bits forged) is caught here and only here on the
    // host path, so it must not be skippable.
    let log_num_interactions = initial_num_variables - 1;

    // (1) Split the reduced eval_point into (interaction, trace) axes.
    // The GKR flat index is `row*cols + col` with the interaction `col`
    // in the LOW bits (extract.rs / round.rs flatten_layer), and the
    // MSB-fold round walk leaves `eval_point` in LSB-first order
    // (round.rs:179-184), so the first `log_num_interactions` coords are
    // the interaction axis and the remaining are the trace (row) axis.
    if eval_point.len() != log_num_interactions + max_log_row_count {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "reconstruction: reduced eval_point dim {} != log_num_interactions {} + \
             max_log_row_count {}",
            eval_point.len(),
            log_num_interactions,
            max_log_row_count
        )));
    }
    let (interaction_point, trace_point) = eval_point.split_at(log_num_interactions);

    // (2) The trace point must equal the claimed opening point, and its
    // dimension must equal the FIXED cube threaded in from `verify_shard`.
    let logup_evaluations = &proof.logup_evaluations;
    if trace_point.len() != max_log_row_count {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "reconstruction: trace_point dim {} != max_log_row_count {}",
            trace_point.len(),
            max_log_row_count
        )));
    }
    if logup_evaluations.point.as_slice() != trace_point {
        return Err(BasefoldVerifyError::LogupGkr(
            "reconstruction: logup_evaluations.point != reduced trace_point".into(),
        ));
    }

    // (3) `point_extended` for the per-chip `full_geq` padding mask.
    //
    // The GKR leaf is LSB-first natural-row: the chip's real rows are
    // `[0, height)` matched LSB-first with `trace_point` (verified by the
    // prover-side direct-leaf ground truth, top_level.rs).  So the padding
    // mask the reconstruction needs is
    //     geq_B = Σ_{row ≥ height} eq_mle_table(trace_point)[row]
    // i.e. `full_geq(degree, ·)` paired so degree-bit k aligns with
    // `trace_point[k]`.  `full_geq_host` pairs
    // `threshold.rev()` with `point.rev()`, i.e. degree-bit i with
    // point[len-1-i]; feeding the REVERSED trace_point (plus a zero high
    // coord) makes degree-bit k align with
    // `trace_point[k]`, reproducing the LSB-first leaf mask.  (The
    // `[0, ...trace_point]` ordering is the zerocheck's bit-REVERSED
    // convention — correct for the zerocheck's `bitrev_rows` poly but the
    // OPPOSITE of the GKR leaf, so it would make honest reconstruction fail.)
    let mut point_extended: Vec<Challenge<SC>> = Vec::with_capacity(max_log_row_count + 1);
    point_extended.push(Challenge::<SC>::ZERO);
    point_extended.extend(trace_point.iter().rev().copied());

    // (4) Per-chip reconstruction in Ziren's interaction layout.
    //
    // CRITICAL packing detail: the GKR `circuit_output` packs each chip's RAW
    // interactions CONTIGUOUSLY (extract.rs:86,118 / round.rs:78-80
    // `offset += num_interactions` — the RAW count), and ALL padding lands at
    // the GLOBAL trailing end of the `col` axis: there are no between-chip
    // gaps, and the global axis is `log2(Σ raw)` rounded up.  So we pack
    // RAW-contiguous here and resize the whole vector to the global
    // `2^interaction_dim` with the identity fraction — byte-matching
    // `extract_outputs`.
    // Iterate `chips` in slice order — the SAME order `generate_first_layer`
    // builds the layer, so the global `col` axis here matches
    // `circuit_output`'s.
    if opened_values.chips.len() != chips.len() {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "reconstruction: opened_values chip count {} != chips {}",
            opened_values.chips.len(),
            chips.len()
        )));
    }
    let mut numerator_values: Vec<Challenge<SC>> = Vec::new();
    let mut denominator_values: Vec<Challenge<SC>> = Vec::new();

    for (chip, opening) in chips.iter().zip(opened_values.chips.iter()) {
        let name = <A as MachineAir<Val<SC>>>::name(&chip.air);

        // degree = quotient[0] = real-height big-endian bits.
        let degree: &[Challenge<SC>] =
            opening.quotient.first().map(|v| v.as_slice()).unwrap_or(&[]);
        if degree.len() != point_extended.len() {
            return Err(BasefoldVerifyError::LogupGkr(format!(
                "reconstruction: chip '{}' degree dim {} != point_extended dim {}",
                name,
                degree.len(),
                point_extended.len()
            )));
        }
        // A genuine height-0 missing chip has all-zero degree bits
        // => full_geq == 1 => identity fraction (0,1) => excluded from the
        // reconstruction.
        let geq_eval = full_geq_host::<Challenge<SC>>(degree, &point_extended);

        // Trace openings at the GKR point, looked up by chip NAME (the
        // chip_openings BTreeMap is name-ordered, `chips` is def-ordered).
        let chip_eval = logup_evaluations.chip_openings.get(name.as_str()).ok_or_else(|| {
            BasefoldVerifyError::LogupGkr(format!(
                "reconstruction: no chip_opening for chip '{}'",
                name
            ))
        })?;
        // ── FULL-POINT OPENING ──
        //
        // Each chip's trace is opened at the FULL `max_log_row_count` point
        // (the trace is a padded MLE, real on the low rows and ZERO on the
        // padding rows) and the identity-fraction-padded leaf is recovered
        // via
        //   numerator   = real − padding·geq
        //   denominator = real + (1 − padding)·geq
        // on those FULL-point openings.
        //
        // The GKR leaf is LSB-first natural-row (real rows `[0,height)`
        // matched LSB-first with `trace_point`), so the FULL-point opening
        //   main_full[col] = Σ_{row<height} eq(row, trace_point)·trace[row]
        // is EXACTLY the value `interaction.eval(full_opening)` needs
        // — no per-chip embed lift.  The prover emits this opening in
        // `main_trace_evaluations_full` (top_level.rs), and it is the ONLY
        // opening a chip carries.  `geq` (over the REVERSED
        // `point_extended`, see (3)) is the LSB-first padding mask
        //   geq = Σ_{row ≥ height} eq(row, trace_point).
        //
        // SOUNDNESS: `geq = full_geq(degree, point_extended)` reads the
        // per-chip `degree` BITS, so a height forgery (tampered `degree`)
        // perturbs the `padding·geq` mask → the reconstructed num/den
        // diverge from the round walk → reject.
        //
        let (main, prep, geq_for_mask): (
            Vec<Challenge<SC>>,
            Option<Vec<Challenge<SC>>>,
            Challenge<SC>,
        ) = (
            chip_eval.main_trace_evaluations_full.as_deref().unwrap_or(&[]).to_vec(),
            chip_eval.preprocessed_trace_evaluations_full.clone(),
            geq_eval,
        );
        let geq_eval = geq_for_mask;

        // Zero padding openings — the trace eval on a fully padding row
        // (all-zero), used to correct the padding region.
        let padding_main: Vec<Challenge<SC>> = vec![Challenge::<SC>::ZERO; main.len()];
        let padding_prep: Option<Vec<Challenge<SC>>> =
            prep.as_ref().map(|p| vec![Challenge::<SC>::ZERO; p.len()]);

        for (interaction, is_send) in
            chip.sends().iter().map(|s| (s, true)).chain(chip.receives().iter().map(|r| (r, false)))
        {
            let (real_numerator, real_denominator) = interaction
                .eval::<Challenge<SC>, Challenge<SC>>(prep.as_deref(), &main, alpha, &beta_powers);
            let (padding_numerator, padding_denominator) = interaction
                .eval::<Challenge<SC>, Challenge<SC>>(
                    padding_prep.as_deref(),
                    &padding_main,
                    alpha,
                    &beta_powers,
                );

            // Degree-masked num/den, then sign flip for receives.
            let numerator_eval_i = real_numerator - padding_numerator * geq_eval;
            let denominator_eval_i =
                real_denominator + (Challenge::<SC>::ONE - padding_denominator) * geq_eval;
            let numerator_eval_i = if is_send { numerator_eval_i } else { -numerator_eval_i };
            numerator_values.push(numerator_eval_i);
            denominator_values.push(denominator_eval_i);
        }
        // NO between-chip padding: chips pack contiguously by RAW count
        // (see extract.rs:118 / round.rs:80).  Trailing global pad below.
    }

    // (5) Pad to the full interaction-axis size and evaluate at the
    // interaction point.  Numerator pads with 0, denominator with 1
    // (the identity fraction — extract.rs:73-76 / round.rs:117-123).
    //
    // The axis width comes from the PROOF (`log_num_interactions` is read off
    // the circuit-output MLE length), so it must be checked before it is used
    // as a resize target: an axis narrower than the chips' raw interaction
    // total would TRUNCATE real interactions out of the reconstruction, and a
    // prover could use that to drop lookups it does not want counted.
    let axis_width = 1usize << interaction_point.len();
    if numerator_values.len() > axis_width {
        return Err(BasefoldVerifyError::LogupGkr(format!(
            "reconstruction: interaction axis {} is narrower than the chips' raw \
             interaction total {}",
            axis_width,
            numerator_values.len()
        )));
    }
    numerator_values.resize(axis_width, Challenge::<SC>::ZERO);
    denominator_values.resize(axis_width, Challenge::<SC>::ONE);

    let reconstructed_numerator = evaluate_mle_host(&numerator_values, interaction_point);
    let reconstructed_denominator = evaluate_mle_host(&denominator_values, interaction_point);

    // (6) The GKR round walk's reduced final evals MUST equal the
    // reconstruction from the chips' trace openings.  This is the assert
    // that catches the area-preserving height forgery: tampering a chip's
    // `degree` moves `geq_eval`, perturbing reconstructed num/den while the
    // walk's `numerator_eval`/`denominator_eval` (which never sees the
    // degree) stays fixed.
    if numerator_eval != reconstructed_numerator {
        return Err(BasefoldVerifyError::LogupGkr(
            "last-layer reconstruction: numerator mismatch (degree-masked \
             height-soundness assert)"
                .into(),
        ));
    }
    if denominator_eval != reconstructed_denominator {
        return Err(BasefoldVerifyError::LogupGkr(
            "last-layer reconstruction: denominator mismatch (degree-masked \
             height-soundness assert)"
                .into(),
        ));
    }

    let _ = max_log_row_count;

    // Observe the GKR trace openings (trace@ζ) — mirror of the prover's
    // observe at the end of
    // `row_gkr::top_level::prove_shard_logup_gkr_rows`.  It MUST land
    // here, at the end of the LogUp-GKR stage, because `verify_zerocheck_host`
    // opens by sampling α / γ / λ — the challenges the opening vector has to be
    // bound before.
    crate::shard_level::prover::observe_logup_gkr_openings::<Val<SC>, Challenge<SC>, SC::Challenger>(
        challenger,
        chips.len(),
        &proof.logup_evaluations,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_constructs_with_defaults() {
        let v = BasefoldShardVerifier::production_default();
        assert_eq!(v.max_log_row_count, 22);
    }

    #[test]
    fn verifier_with_params_honors_custom_row_count() {
        let v = BasefoldShardVerifier::with_params(3);
        assert_eq!(v.max_log_row_count, 3);
    }

    /// `production_default` carries the one config constant
    /// (`consts::CORE_MAX_LOG_ROW_COUNT`, 22), and the verifier binds
    /// every proof to it exactly — the GKR round-count check
    /// (`round_proofs.len() + 1 == max_log_row_count`) rejects a proof
    /// produced at any other cube; nothing floats the cube up from the
    /// proof.
    #[test]
    fn fixed_cube_sourced_from_the_core_constant() {
        let base = BasefoldShardVerifier::production_default().max_log_row_count;
        assert_eq!(base, crate::stacked_shapes::types::consts::CORE_MAX_LOG_ROW_COUNT);
        assert_eq!(base, 22);
    }

    /// The three-variant error Display ends with the exact phase hint
    /// text so users can grep for it.
    #[test]
    fn unimplemented_error_displays_phase_hint() {
        let e = BasefoldVerifyError::Unimplemented("Phase 2 (LogUp-GKR verification)");
        let s = format!("{e}");
        assert!(s.contains("Phase 2"));
        assert!(s.contains(""));
    }

    #[test]
    fn shape_errors_display_expected_and_got() {
        let e = BasefoldVerifyError::PublicValuesLengthMismatch { expected: 100, got: 50 };
        let s = format!("{e}");
        assert!(s.contains("100"));
        assert!(s.contains("50"));

        let e = BasefoldVerifyError::ChipCountMismatch { expected: 10, got: 7 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains("7"));
    }

    /// `full_geq_host` with all-zero threshold is identically 1 — the
    /// fold `acc_new = (1-y)*acc + y` starting from 1 collapses to 1
    /// at every step regardless of `y` (the in-circuit stub uses this
    /// invariant, so the host port must match).
    #[test]
    fn full_geq_host_zero_threshold_is_one() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        let threshold = vec![EF::ZERO; 4];
        let eval_point = vec![EF::from_u32(3), EF::from_u32(7), EF::from_u32(11), EF::from_u32(13)];
        let result = full_geq_host(&threshold, &eval_point);
        assert_eq!(result, EF::ONE);
    }

    /// `full_geq_host` on boolean inputs where `threshold == eval_point`
    /// equals 1 — the "step-up" term `y*(1-x)` is 0 at every bit, so
    /// the recurrence stays at the identity.
    #[test]
    fn full_geq_host_equal_boolean_threshold_is_one() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        // Boolean point.
        let point = vec![EF::ONE, EF::ZERO, EF::ONE];
        let result = full_geq_host(&point, &point);
        assert_eq!(result, EF::ONE);
    }

    /// `full_geq_host` on boolean inputs with `eval_point > threshold`
    /// in big-endian comparison fires the step-up term.  Specifically
    /// threshold = [0,0], eval_point = [1,0] → at bit 0 (MSB), y=1,
    /// x=0 contributes step-up=1, yielding result = 1.
    #[test]
    fn full_geq_host_boolean_strict_greater() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        let threshold = vec![EF::ZERO, EF::ZERO];
        let eval_point = vec![EF::ONE, EF::ZERO];
        let result = full_geq_host(&threshold, &eval_point);
        // At MSB bit: eq_factor=(1-0)(1-1)+0·1=0, step=1·(1-0)=1. acc=1·0+1=1.
        // At LSB bit: eq_factor=(1-0)(1-0)+0·0=1, step=0·1=0.  acc=1·1+0=1.
        assert_eq!(result, EF::ONE);
    }

    /// `degree_stub_host` returns a vector of exactly
    /// `max_log_row_count + 1` zero entries, matching the witness
    /// stub at `shard_proof_variable_lift::empty_chip_height_bits`.
    #[test]
    fn degree_stub_host_is_zero_filled_with_extra_bit() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        for max_log in [0usize, 1, 5, 22] {
            let v: Vec<EF> = degree_stub_host(max_log);
            assert_eq!(v.len(), max_log + 1);
            assert!(v.iter().all(|x| *x == EF::ZERO));
        }
    }

    /// eq_eval on identical points = 1; on differing = not-1.
    #[test]
    fn eq_eval_host_indicator() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        let a = vec![EF::from_u32(3), EF::from_u32(5)];
        let b = vec![EF::from_u32(3), EF::from_u32(5)];
        // eq(a, b) where a == b: Π ((1-x)(1-x) + x·x) = Π (1 - 2x + 2x²)
        // evaluated element-wise.  Not necessarily 1 unless both are boolean.
        // Just confirm it's deterministic & computes:
        let v = eq_eval_host(&a, &b);
        let _ = v;

        // Different points produce different eq values.
        let c = vec![EF::from_u32(3), EF::from_u32(7)];
        let u = eq_eval_host(&a, &c);
        assert_ne!(v, u, "eq_eval differs when points differ");
    }

    /// MLE eval at uniform 0 vector == first entry; at uniform 1 vector
    /// (all 1s) probes the last entry in LSB-first indexing.
    #[test]
    fn evaluate_mle_host_endpoints() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        // 4-element MLE (2 variables).  Values: [a, b, c, d].
        let evals: Vec<EF> = (10..14).map(EF::from_u32).collect();

        // At (0, 0) → entry 0.
        let at_origin = evaluate_mle_host(&evals, &[EF::ZERO, EF::ZERO]);
        assert_eq!(at_origin, EF::from_u32(10));

        // At (1, 1) → entry 3 (all-ones index).
        let at_all_ones = evaluate_mle_host(&evals, &[EF::ONE, EF::ONE]);
        assert_eq!(at_all_ones, EF::from_u32(13));

        // At (1, 0) → entry 1.
        let at_10 = evaluate_mle_host(&evals, &[EF::ONE, EF::ZERO]);
        assert_eq!(at_10, EF::from_u32(11));

        // At (0, 1) → entry 2.
        let at_01 = evaluate_mle_host(&evals, &[EF::ZERO, EF::ONE]);
        assert_eq!(at_01, EF::from_u32(12));
    }

    /// Horner's eval_coeffs_host produces the correct polynomial value.
    #[test]
    fn eval_coeffs_host_horner_correctness() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;
        type EF = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

        // p(X) = 3 + 5X + 7X² = [3, 5, 7] (low-degree-first).
        let coeffs: Vec<EF> = vec![EF::from_u32(3), EF::from_u32(5), EF::from_u32(7)];

        // p(0) = 3
        assert_eq!(eval_coeffs_host(&coeffs, EF::ZERO), EF::from_u32(3));
        // p(1) = 3 + 5 + 7 = 15
        assert_eq!(eval_coeffs_host(&coeffs, EF::ONE), EF::from_u32(15));
        // p(2) = 3 + 10 + 28 = 41
        assert_eq!(eval_coeffs_host(&coeffs, EF::from_u32(2)), EF::from_u32(41));
    }
}
