//! Per-chip BaseFold jagged-PCS adapter.
//!
//! Each chip trace becomes one MLE that goes through
//! [`crate::basefold::StackedPcsProver`], so the BaseFold encoder
//! materializes one stripe at a time (`1 << log_stacking_height`
//! rows × `batch_size` polys) instead of one giant dense LDE.  No
//! `Vec<F>` of size `2^(num_vars + log_blowup)` is ever held in
//! memory at once.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_dft::Radix2DitParallel;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::basefold::{
    BasefoldProver, BasefoldVerifier, FriConfig, Mle, StackedBasefoldProof,
    StackedBasefoldProverData, StackedPcsProver, StackedPcsVerifier,
};
use crate::kb31_poseidon2::{InnerChallenge, InnerChallenger, InnerValMmcs};

pub type JaggedVal = crate::kb31_poseidon2::InnerVal;
pub type JaggedChallenge = InnerChallenge;
pub type JaggedDft = Radix2DitParallel<JaggedVal>;
pub type JaggedMmcs = InnerValMmcs;
pub type JaggedChallenger = InnerChallenger;

/// One committed batch of chip traces, plus the per-chip metadata
/// needed to recompute evaluation points on the verifier side.
///
/// Generic over the MMCS `MT` so the inner (Poseidon2-KoalaBear) and the
/// wrap (OuterSC, Poseidon2-BN254) commit paths share one struct.
/// `Val`/`Challenge` stay KoalaBear / KoalaBear⁴ for both; only the
/// commitment hash varies.  The concrete [`JaggedCommit`] alias below pins
/// `MT = JaggedMmcs` so every existing caller (incl. serde wire-format + the
/// ziren-gpu hooks) compiles unchanged.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "<MT as p3_commit::Mmcs<JaggedVal>>::Commitment: serde::Serialize",
    deserialize = "<MT as p3_commit::Mmcs<JaggedVal>>::Commitment: serde::Deserialize<'de>"
))]
pub struct JaggedCommitGeneric<MT: p3_commit::Mmcs<JaggedVal>> {
    /// The `#[serde(rename = "commitment")]` pins the *serialized* field name
    /// to the historical `commitment` so the (positional) proof wire format
    /// stays byte-identical.
    #[serde(rename = "commitment")]
    pub original_commitment: <MT as p3_commit::Mmcs<JaggedVal>>::Commitment,
    /// Per-chip `(width, log_height_padded)` so the verifier can
    /// reconstruct the same Mle shapes when checking openings.
    pub chip_dims: Vec<(usize, u32)>,
    /// Total `[batch_size << log_stacking_height]` area of the
    /// stacked PCS commit — equals the verifier's `round_areas[0]`.
    pub area: usize,
    /// Actual log_stacking_height used for this commit (clamped down
    /// for tiny commits — see [`pick_log_stacking_height`]).
    pub log_stacking_height: u32,
}

/// Concrete inner (Poseidon2-KoalaBear) commit — the type every current
/// caller uses.  Transparent alias to the generic struct so struct
/// literals / field access compile unchanged.
pub type JaggedCommit = JaggedCommitGeneric<JaggedMmcs>;

pub struct JaggedProverDataGeneric<MT: p3_commit::Mmcs<JaggedVal>> {
    pub stacked_data: StackedBasefoldProverData<JaggedVal, MT>,
    pub chip_dims: Vec<(usize, u32)>,
    pub area: usize,
    pub log_stacking_height: u32,
}

/// Concrete inner prover-data alias (`MT = JaggedMmcs`).
pub type JaggedProverData = JaggedProverDataGeneric<JaggedMmcs>;

/// Stacking height of the stacked PCS: `2^21` rows per stripe.  Never
/// clamped down for small commits (see [`pick_log_stacking_height`]).
pub const DEFAULT_LOG_STACKING_HEIGHT: u32 = 21;

/// Interleave batch size for the stacked PCS: number of MLE-column
/// streams packed into each stripe.  Purely a packing constant — no
/// soundness implication.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// FIXED stacking height: ALWAYS `DEFAULT_LOG_STACKING_HEIGHT`
/// (21), never clamped down for small commits.
///
/// Clamping to `min(21, log2(np2(total))-1)` for tiny commits would make
/// the prover's `log_stacking_height` depend on the trace AREA.  Because the
/// recursion normalize/compress program is rebuilt per-proof from
/// `bundle.commit.log_stacking_height` (the value the prover used), that
/// clamp would make the program — hence its VK — CLAMP-DEPENDENT (the VK
/// varying with chip heights).
///
/// Instead the stacking height is FIXED and each call site rounds the trace
/// AREA up to a multiple of `2^21`
/// (`area = total_entries.next_multiple_of(1 << 21)`), so every commit is
/// honestly 21-round → the per-proof verifier rebuild constant-folds to
/// `num_variables = 21` → clamp-INDEPENDENCE, with no transcript masking and
/// no Fiat-Shamir risk (the unsound verifier-side alternative).  The
/// normalize VK then depends on the chip-SET only.
///
/// `total_entries` is retained for call-site/API symmetry but does not
/// affect the height (the call-site area padding absorbs it).
pub fn pick_log_stacking_height(_total_entries: usize) -> u32 {
    DEFAULT_LOG_STACKING_HEIGHT
}

/// Public for the GPU commit-dispatch hook: the
/// device-side commit path needs to run the same MLE-construction +
/// padding logic as the host before invoking the GPU encoder.
pub fn chips_to_mles_owned(
    chip_traces: Vec<(String, RowMajorMatrix<JaggedVal>)>,
) -> (Vec<Arc<Mle<JaggedVal>>>, Vec<(usize, u32)>) {
    let mut mles = Vec::with_capacity(chip_traces.len());
    let mut dims = Vec::with_capacity(chip_traces.len());
    for (_, trace) in chip_traces.into_iter() {
        let width = trace.width.max(1);
        let raw_height = trace.values.len() / width;
        // Round out to whole stacking blocks, NOT to a power of two.  The only
        // caller hands over the single width-1 jagged dense, and the interleave
        // below re-stripes it at exactly this granularity, so a power-of-two
        // round-up buys nothing and costs up to half the committed area.
        let padded_height = raw_height.next_multiple_of(1usize << DEFAULT_LOG_STACKING_HEIGHT);
        // `log_h` is the dims' height slot and stays a LOG, so for a
        // non-power-of-two block count it is the enclosing hypercube, an upper
        // bound rather than the exact height.
        let log_h = padded_height.max(1).next_power_of_two().trailing_zeros();

        let values = if raw_height == padded_height {
            trace.values
        } else {
            let mut padded = trace.values;
            padded.resize(padded_height * width, JaggedVal::ZERO);
            padded
        };

        mles.push(Arc::new(Mle::from_row_major(RowMajorMatrix::new(values, width))));
        dims.push((width, log_h));
    }
    (mles, dims)
}

/// Commit a batch of chip traces (consumes ownership — saves the
/// `trace.values.clone()` round-trip in `chips_to_mles_owned`).
/// Returns a public commitment and prover-side state for later opening.
///
/// The shard commit runs the host BaseFold + Plonky3 MMCS commit.  The GPU
/// prover does NOT reach this free-fn — its device dense-pack + BaseFold commit
/// is the `StarkGpuProver` override of `MachineProver::commit_multilinears`
/// (unconditional device commit, no host fallback).  This free-fn is
/// the CPU-prover / unit-test path.
///
/// Does NOT observe the commitment — it takes no challenger; the CALLER owns
/// the transcript write and must observe `commit.original_commitment` at the
/// same position as the verifier.  On the single-main-commit flow that write
/// is the shard-level Phase 1 prologue's 8-felt `main_commitment` observe —
/// observing here as well would desync the prover against the verifier.  The
/// verifier counterpart is [`jagged::verify_jagged_basefold_no_observe`].
pub fn commit_jagged_pcs(
    chip_traces: Vec<(String, RowMajorMatrix<JaggedVal>)>,
) -> (JaggedCommit, JaggedProverData) {
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
    let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
    let mmcs = JaggedMmcs::new(hash, compress, 0);
    let dft = Arc::new(JaggedDft::default());
    // Delegate to the GC-generic core (inner = Poseidon2-KoalaBear Mmcs).
    commit_jagged_pcs_generic::<JaggedMmcs, JaggedDft>(
        chip_traces,
        mmcs,
        dft,
        FriConfig::<JaggedVal>::from_env_or_default(),
    )
}

/// GC-generic commit core.  Does not touch the transcript: it takes no
/// challenger, and the caller owns the `observe` of the returned commitment.
/// Parameterized over the MMCS `MT` + DFT `D`; the caller supplies the
/// concrete `mmcs`/`dft` so the inner (Poseidon2-KoalaBear) and the wrap
/// (OuterSC, Poseidon2-BN254) paths share one body.  `Val`/`Challenge` stay
/// KoalaBear / KoalaBear⁴ for both.
#[allow(clippy::type_complexity)]
pub fn commit_jagged_pcs_generic<MT, D>(
    chip_traces: Vec<(String, RowMajorMatrix<JaggedVal>)>,
    mmcs: MT,
    dft: Arc<D>,
    fri: FriConfig<JaggedVal>,
) -> (JaggedCommitGeneric<MT>, JaggedProverDataGeneric<MT>)
where
    MT: p3_commit::Mmcs<JaggedVal, Commitment: Clone> + Clone,
    D: p3_dft::TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
{
    let (mles, chip_dims) = chips_to_mles_owned(chip_traces);
    let total_entries: usize = mles.iter().map(|m| m.guts().total_len()).sum();
    let log_stacking_height = pick_log_stacking_height(total_entries);
    let area = total_entries.next_multiple_of(1usize << log_stacking_height);

    // Build only the prover: the commit never uses a paired verifier.
    let prover = StackedPcsProver::new(
        BasefoldProver::<JaggedVal, JaggedChallenge, MT, D>::new(fri, dft, mmcs, 1),
        log_stacking_height,
        DEFAULT_BATCH_SIZE,
    );
    let (commitment, stacked_data) = prover.commit_multilinears(mles);

    let commit = JaggedCommitGeneric::<MT> {
        original_commitment: commitment.clone(),
        chip_dims: chip_dims.clone(),
        area,
        log_stacking_height,
    };
    let prover_data =
        JaggedProverDataGeneric::<MT> { stacked_data, chip_dims, area, log_stacking_height };
    (commit, prover_data)
}

/// Extract the 8-felt MMCS digest from a [`JaggedCommit`].
/// The digest is the value the verifier's Phase 1 prologue observes as
/// `main_commitment` in the single-main-commit flow.
///
/// The commitment is a `MerkleCap<KoalaBear, [KoalaBear; 8]>` (the
/// Plonky3 `MerkleTreeMmcs::Commitment` for `InnerValMmcs`).  This
/// helper pulls out the first cap root — the same byte sequence
/// `DuplexChallenger::observe(MerkleCap)` consumes.
#[must_use]
/// Extract the 8-felt MerkleCap root from a JaggedMmcs commitment (the
/// inner BasefoldRing::digest_felts body).
pub fn basefold_commit_digest_felts(
    commitment: &<JaggedMmcs as p3_commit::Mmcs<JaggedVal>>::Commitment,
) -> [JaggedVal; 8] {
    let roots = commitment.roots();
    assert!(!roots.is_empty(), "JaggedCommit MerkleCap must have at least one root");
    roots[0]
}

pub fn basefold_commit_digest(commit: &JaggedCommit) -> [JaggedVal; 8] {
    let roots = commit.original_commitment.roots();
    assert!(!roots.is_empty(), "JaggedCommit MerkleCap must have at least one root",);
    roots[0]
}

// ─────────────────────────────────────────────────────────────────────
// Jagged "hash-bind" (the count ↔ commitment tie).
//
// Observing only the RAW BaseFold root would leave the per-chip
// (row_count, column_count) geometry prover-supplied and (under the
// height-agnostic recursion) forgeable: a prover could witness a different
// geometry than was actually committed and still pass the prefix-sum / area
// consistency checks (which only tie the geometry to itself).
//
// The geometry is therefore hashed and folded into the observed commitment:
//
//     hash = hash_iter( once(len) ++ row_counts ++ column_counts )
//     modified = compress([raw_root, hash])
//
// and re-checked at verify time.  The observed (Fiat-Shamir) commitment is
// `modified`; the BaseFold opening still binds against `raw_root` (carried
// as `original_commitment`).
//
// CONVENTION LOCK (host == circuit must be byte-identical):
//   * `len` is `column_counts.len()` (== `row_counts.len()`), used
//     IDENTICALLY in both places to avoid any FS desync.
//   * the geometry is the PER-CHIP `(row_count, column_count)` derived from
//     the SAME `packing.offsets` / `packing.column_counts` the recursion
//     lift reconstructs (`shard_level_witness.rs` `packing_row_counts`),
//     so the in-circuit recompute hashes the identical felt sequence.
//   * felts are `from_canonical_usize` (wraps mod the field order — the
//     in-circuit verifier guards each count `< F::ORDER` so the wrap can
//     never be exploited; see the recursion guards).
// ─────────────────────────────────────────────────────────────────────

/// Derive the per-chip `(row_counts, column_counts)` the hash-bind hashes,
/// from the host jagged `PackingMeta`.  This is the SINGLE source of truth
/// for the hash convention — both the host emit path
/// (`jagged_hash_bind_modified`) and the in-circuit recompute consume the
/// SAME per-chip vectors (the recursion lift's `packing_row_counts` /
/// `packing.column_counts`), so the hashed felt sequence is byte-identical.
///
/// `row_counts[i]` = height of chip `i` = `offsets[col_i+1] - offsets[col_i]`
/// where `col_i` is the first column index of chip `i` (a `column_count==0`
/// chip contributes height `0`).  `column_counts[i] = packing.column_counts[i]`.
#[must_use]
pub fn jagged_counts_from_packing(packing: &jagged::PackingMeta) -> (Vec<usize>, Vec<usize>) {
    let column_counts: Vec<usize> = packing.column_counts.clone();
    let offsets = &packing.offsets;
    let total_values = packing.total_values;
    let mut row_counts: Vec<usize> = Vec::with_capacity(column_counts.len());
    let mut col_idx = 0usize;
    for &cc in column_counts.iter() {
        if cc == 0 {
            row_counts.push(0);
            continue;
        }
        let h = if col_idx + 1 < offsets.len() {
            offsets[col_idx + 1].saturating_sub(offsets[col_idx])
        } else if col_idx < offsets.len() {
            total_values.saturating_sub(offsets[col_idx])
        } else {
            0
        };
        row_counts.push(h);
        col_idx += cc;
    }
    (row_counts, column_counts)
}

/// Compute the geometry hash for ONE round (one commit):
/// `hash_iter( once(len) ++ row_counts ++ column_counts )` where
/// `len = column_counts.len()`.  Uses the inner Poseidon2-KoalaBear sponge
/// (`InnerHash`) — the SAME hasher `SC::hash` resolves to in-circuit.
#[must_use]
pub fn jagged_geometry_hash(row_counts: &[usize], column_counts: &[usize]) -> [JaggedVal; 8] {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::CryptographicHasher;
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hasher = crate::kb31_poseidon2::InnerHash::new(perm);
    let len = column_counts.len();
    let iter = core::iter::once(JaggedVal::from_canonical_usize(len))
        .chain(row_counts.iter().map(|&c| JaggedVal::from_canonical_usize(c)))
        .chain(column_counts.iter().map(|&c| JaggedVal::from_canonical_usize(c)));
    hasher.hash_iter(iter)
}

/// Fold the geometry hash into the raw BaseFold root: `compress([raw, hash])`.
/// Uses `InnerCompress` (the SAME compressor `SC::compress` resolves to
/// in-circuit).  Returns the MODIFIED 8-felt digest that the Fiat-Shamir
/// transcript observes as `main_commitment`.
#[must_use]
pub fn jagged_hash_bind_modified(
    raw_root: [JaggedVal; 8],
    row_counts: &[usize],
    column_counts: &[usize],
) -> [JaggedVal; 8] {
    use p3_symmetric::PseudoCompressionFunction;
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let compressor = crate::kb31_poseidon2::InnerCompress::new(perm);
    let hash = jagged_geometry_hash(row_counts, column_counts);
    compressor.compress([raw_root, hash])
}

/// Convenience: compute the MODIFIED digest directly from the raw commit +
/// packing — the host emit-site one-liner.
#[must_use]
pub fn jagged_hash_bind_from_packing(
    raw_root: [JaggedVal; 8],
    packing: &jagged::PackingMeta,
) -> [JaggedVal; 8] {
    let (row_counts, column_counts) = jagged_counts_from_packing(packing);
    jagged_hash_bind_modified(raw_root, &row_counts, &column_counts)
}

/// Derive the per-chip `(row_counts, column_counts)` from a full
/// [`crate::jagged::JaggedPacking`] (the form the host commit prover holds in
/// `PrecomputedJaggedCommit.packing`).  Uses the OFFSETS-based derivation
/// (NOT `chip_infos[i].row_count`) so it is byte-identical to the recursion
/// lift's `packing_row_counts` (`shard_level_witness.rs`) and to
/// [`jagged_counts_from_packing`]: a `column_count == 0` chip contributes
/// height `0` regardless of its raw trace height.
#[must_use]
pub fn jagged_counts_from_jagged_packing(
    packing: &crate::jagged::JaggedPacking<JaggedVal>,
) -> (Vec<usize>, Vec<usize>) {
    let column_counts: Vec<usize> = packing.chip_infos.iter().map(|ci| ci.column_count).collect();
    let offsets = &packing.offsets;
    let total_values = packing.total_values;
    let mut row_counts: Vec<usize> = Vec::with_capacity(column_counts.len());
    let mut col_idx = 0usize;
    for &cc in column_counts.iter() {
        if cc == 0 {
            row_counts.push(0);
            continue;
        }
        let h = if col_idx + 1 < offsets.len() {
            offsets[col_idx + 1].saturating_sub(offsets[col_idx])
        } else if col_idx < offsets.len() {
            total_values.saturating_sub(offsets[col_idx])
        } else {
            0
        };
        row_counts.push(h);
        col_idx += cc;
    }
    (row_counts, column_counts)
}

/// Host emit-site one-liner: the MODIFIED digest from the raw root + the
/// full `JaggedPacking` the commit prover holds.  This is the value the
/// Fiat-Shamir transcript observes as `main_commitment`.
#[must_use]
pub fn jagged_hash_bind_from_jagged_packing(
    raw_root: [JaggedVal; 8],
    packing: &crate::jagged::JaggedPacking<JaggedVal>,
) -> [JaggedVal; 8] {
    let (row_counts, column_counts) = jagged_counts_from_jagged_packing(packing);
    jagged_hash_bind_modified(raw_root, &row_counts, &column_counts)
}

/// Host-side mirror of the in-circuit re-bind: recompute
/// `compress([raw, hash(counts)])` and check it equals the observed
/// `modified` digest.  Used by the host round-trip test to LOCK the
/// convention the circuit consumes.
#[must_use]
pub fn jagged_hash_bind_verify(
    raw_root: [JaggedVal; 8],
    modified: [JaggedVal; 8],
    packing: &jagged::PackingMeta,
) -> bool {
    let recomputed = jagged_hash_bind_from_packing(raw_root, packing);
    recomputed == modified
}

/// Production-grade FRI config used by the jagged-PCS pipeline.
/// Public so the GPU dispatch hook can construct a matching
/// device-side encoder (same `log_blowup`, same coset shift) without
/// re-creating the env-overrides logic.
pub fn lb_fri_config() -> FriConfig<JaggedVal> {
    FriConfig::<JaggedVal>::from_env_or_default()
}

// ─────────────────────────────────────────────────────────────────────
// GPU jagged-reduction sumcheck dispatch hook.
//
// Mirrors the host `crate::jagged_sumcheck::prove_jagged_reduction_owned`
// signature one-for-one — same inputs (owned `dense_q`, packing,
// `r_row_per_chip`, `y_per_chip`, challenger), same output
// (`JaggedReductionProof<InnerChallenge>`).  Wired from the jagged
// step (4) reduction when the GPU jagged-reduction hook is registered
// (GPU prover only).
// ─────────────────────────────────────────────────────────────────────

/// Borrowed-cells view of an EF row-GKR layer suitable for the GPU
/// init hook.  The four sub-MLEs are passed by slice so the upload
/// stays zero-copy on the host side; the GPU side is responsible for
/// the memcpy / pin + dma into device memory.
///
/// Lifetime borrows from the `LogUpGkrCpuLayer<JaggedChallenge, JaggedChallenge>`
/// the dispatch site holds across the call.
pub struct HostLayerView<'a> {
    pub numerator_0: &'a [crate::shard_level::row_gkr::layer::RowMajorTable<JaggedChallenge>],
    pub denominator_0: &'a [crate::shard_level::row_gkr::layer::RowMajorTable<JaggedChallenge>],
    pub numerator_1: &'a [crate::shard_level::row_gkr::layer::RowMajorTable<JaggedChallenge>],
    pub denominator_1: &'a [crate::shard_level::row_gkr::layer::RowMajorTable<JaggedChallenge>],
    pub num_row_variables: usize,
    pub num_interaction_variables: usize,
}

/// Process-wide monotonic counter for GKR-circuit IDs.  Each
/// `build_gkr_circuit` call that takes the device path allocates a
/// fresh ID via [`allocate_gpu_layer_circuit_id`] and threads it
/// through every device-layer init / transition / pull hook
/// invocation.  The GPU side keys its registry by
/// `(device_id, circuit_id)` so concurrent shards on the same GPU are
/// fully isolated.
// Backing storage uses AtomicUsize, not AtomicU64, so the file
// compiles on the zkvm-elf target (mipsel — no
// `target_has_atomic="64"`).  The GPU registry never executes on the
// zkvm-elf binary, but the symbol still has to type-check in that
// build because `row_gkr/build.rs` imports the helper unconditionally.
// Public API (`u64`) is preserved via cast.  On host (64-bit)
// `usize == u64`; on the 32-bit zkvm-elf the upper bits are always
// zero and circuit IDs grow well within `u32::MAX`.
static NEXT_GPU_LAYER_CIRCUIT_ID: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

/// Allocate a fresh process-unique GKR-circuit ID for use with the
/// GPU layer-state hooks.  Must be called once per
/// `build_gkr_circuit` device-path invocation; the returned ID is
/// passed verbatim to every init/transition/pull hook for that
/// circuit.
///
/// IDs start at 1 (0 reserved as a sentinel) and increment
/// monotonically.  Wraparound is not handled — at u64 capacity that
/// would require ~10^9 circuits/sec for centuries, which is well
/// outside the threat model.
#[must_use]
pub fn allocate_gpu_layer_circuit_id() -> u64 {
    NEXT_GPU_LAYER_CIRCUIT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u64
}

/// Batched multi-ROUND open: one BaseFold proof covering every round's
/// committed data — NOT one proof per round.  Every round's stripes are
/// `log_stacking_height` tall, so `BasefoldProver::batch` folds them into one
/// codeword regardless of how many stripes each round contributes.
///
/// Pure host-side implementation — always runs the CPU StackedPcsProver
/// `prove_trusted_evaluation` body; the GPU dispatch hook falls back to it on
/// shape-unsupported / runtime errors.
pub fn open_jagged_pcs_rounds(
    rounds: &[&JaggedProverData],
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut JaggedChallenger,
) -> StackedBasefoldProof<JaggedVal, JaggedChallenge, JaggedMmcs> {
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
    let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
    let mmcs = JaggedMmcs::new(hash, compress, 0);
    let dft = Arc::new(JaggedDft::default());
    let log_stacking_height = rounds[0].log_stacking_height;
    let prover = StackedPcsProver::new(
        BasefoldProver::<JaggedVal, JaggedChallenge, JaggedMmcs, JaggedDft>::new(
            FriConfig::<JaggedVal>::from_env_or_default(),
            dft,
            mmcs,
            // One expected commitment PER ROUND.
            rounds.len(),
        ),
        log_stacking_height,
        DEFAULT_BATCH_SIZE,
    );
    // Borrowed: the committed Merkle trees are read, never copied.
    let stacked: Vec<&_> = rounds.iter().map(|r| &r.stacked_data).collect();
    prover.prove_trusted_evaluation(eval_point, &stacked, challenger)
}

/// Ring-generic counterpart of [`open_jagged_pcs_rounds`]: ONE batched open
/// across every round's committed data, over any commitment family.  The
/// BN254 wrap ring reaches the multi-round open through this.
pub fn open_jagged_pcs_rounds_generic<Challenger, MT, D>(
    rounds: &[&JaggedProverDataGeneric<MT>],
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut Challenger,
    mmcs: MT,
    dft: Arc<D>,
    fri: FriConfig<JaggedVal>,
) -> StackedBasefoldProof<JaggedVal, JaggedChallenge, MT>
where
    MT: p3_commit::Mmcs<JaggedVal, Commitment: Clone> + Clone,
    D: p3_dft::TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
    Challenger: p3_challenger::FieldChallenger<JaggedVal>
        + p3_challenger::GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as p3_commit::Mmcs<JaggedVal>>::Commitment>
        // `'static`: `deterministic_grind` looks the challenger type up in the
        // grind-accelerator registration, which is keyed by `TypeId`.
        + 'static,
{
    let log_stacking_height = rounds[0].log_stacking_height;
    let prover = StackedPcsProver::new(
        // One expected commitment PER ROUND.
        BasefoldProver::<JaggedVal, JaggedChallenge, MT, D>::new(fri, dft, mmcs, rounds.len()),
        log_stacking_height,
        DEFAULT_BATCH_SIZE,
    );
    // Borrowed: the committed Merkle trees are read, never copied.
    let stacked: Vec<&_> = rounds.iter().map(|r| &r.stacked_data).collect();
    prover.prove_trusted_evaluation(eval_point, &stacked, challenger)
}

pub fn open_jagged_pcs(
    prover_data: &JaggedProverData,
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut JaggedChallenger,
) -> StackedBasefoldProof<JaggedVal, JaggedChallenge, JaggedMmcs> {
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
    let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
    let mmcs = JaggedMmcs::new(hash, compress, 0);
    let dft = Arc::new(JaggedDft::default());
    // Delegate to the GC-generic core (inner = Poseidon2-KoalaBear Mmcs).
    open_jagged_pcs_generic::<JaggedChallenger, JaggedMmcs, JaggedDft>(
        prover_data,
        eval_point,
        challenger,
        mmcs,
        dft,
        FriConfig::<JaggedVal>::from_env_or_default(),
    )
}

/// BaseFold-over-BN254 port: GC-generic host open core.  Parameterized
/// over the challenger `Challenger` + MMCS `MT` + DFT `D`; the caller
/// supplies the concrete `mmcs`/`dft`.  The inner path uses `JaggedChallenger`
/// + Poseidon2-KoalaBear Mmcs; the wrap (OuterSC) will pass the BN254
/// challenger + Poseidon2-BN254 Mmcs.  `Val`/`Challenge` stay KoalaBear /
/// KoalaBear⁴ for both (the eval-point is over `JaggedChallenge`).
#[allow(clippy::type_complexity)]
pub fn open_jagged_pcs_generic<Challenger, MT, D>(
    prover_data: &JaggedProverDataGeneric<MT>,
    eval_point: Vec<JaggedChallenge>,
    challenger: &mut Challenger,
    mmcs: MT,
    dft: Arc<D>,
    fri: FriConfig<JaggedVal>,
) -> StackedBasefoldProof<JaggedVal, JaggedChallenge, MT>
where
    MT: p3_commit::Mmcs<JaggedVal, Commitment: Clone> + Clone,
    D: p3_dft::TwoAdicSubgroupDft<JaggedVal> + Send + Sync,
    Challenger: p3_challenger::FieldChallenger<JaggedVal>
        + p3_challenger::GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as p3_commit::Mmcs<JaggedVal>>::Commitment>
        // `'static`: `deterministic_grind` looks the challenger type up in the
        // grind-accelerator registration, which is keyed by `TypeId`.
        + 'static,
{
    let prover = StackedPcsProver::new(
        BasefoldProver::<JaggedVal, JaggedChallenge, MT, D>::new(fri, dft, mmcs, 1),
        prover_data.log_stacking_height,
        DEFAULT_BATCH_SIZE,
    );
    prover.prove_trusted_evaluation(eval_point, &[&prover_data.stacked_data], challenger)
}

// ─────────────────────────────────────────────────────────────────────
// GPU BaseFold open dispatch.
//
// Mirror of the GPU commit override — provided statically by the prover
// (`MachineProver::gpu_basefold_open_hook`) and threaded down to the
// `open_jagged_pcs_generic` dispatch, not via a registry.  The hook receives
// the same inputs as `open_jagged_pcs_generic` and returns a byte-identical
// `StackedBasefoldProof` — the device side is responsible for:
//
//   * routing the per-stripe MLEs / codewords held in
//     `prover_data.stacked_data.pcs_batch_data` to GPU memory (or
//     reading from a device-resident cache if the commit hook installed
//     one),
//   * running `FriCudaProver::prove` (the device prove driver in
//     `ziren-gpu/basefold/src/fri.rs`),
//   * observing the per-round univariate-poly evals + Merkle commits +
//     PoW witness into the supplied `JaggedChallenger` so the transcript
//     stays in lock-step with the host path,
//   * assembling a `StackedBasefoldProof` whose `basefold_proof.*` is
//     shape-compatible with the host path consumed by
//     `verify_jagged_pcs`.
//
// The hook returns `Result<.., (prover_data, eval_point)>` so the device
// side can tunnel ownership of the host inputs back to the host fallback
// on error (mirrors the `commit_jagged_pcs` hook contract).
// ─────────────────────────────────────────────────────────────────────

// The device open lives in the `JaggedOpener` impl `DeviceJaggedOpener`
// (zkm-gpu-basefold), which calls `FriCudaProver::prove` and falls back to
// `open_jagged_pcs` on `Err` (returning `(prover_data, eval_point)`
// ownership so nothing is lost).  The open fn is threaded from the `prover`
// crate into the `prove_shard_with_data` free-fn (through the jagged-eval
// producer + `prove_trusted_evaluations` down to
// `prove_jagged_basefold_rounds`' open closure).

/// Verify the proof against a previously observed commitment.
/// Multi-ROUND verify: one BaseFold proof covering every round's commitment.
///
/// The verifier builds the commitment list itself, so a round whose
/// commitment lives in the verifying key is never taken from the proof.
pub fn verify_jagged_pcs_rounds(
    commitments: &[<JaggedMmcs as p3_commit::Mmcs<JaggedVal>>::Commitment],
    areas: &[usize],
    log_stacking_height: u32,
    eval_point: &[JaggedChallenge],
    evaluation_claim: JaggedChallenge,
    proof: &StackedBasefoldProof<JaggedVal, JaggedChallenge, JaggedMmcs>,
    challenger: &mut JaggedChallenger,
) -> Result<(), crate::basefold::StackedVerifierError> {
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
    let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
    let mmcs = JaggedMmcs::new(hash, compress, 0);
    let fri = FriConfig::<JaggedVal>::from_env_or_default();
    let verifier = crate::basefold::stacked::StackedPcsVerifier::new(
        crate::basefold::verifier::BasefoldVerifier::<JaggedVal, JaggedChallenge, JaggedMmcs>::new(
            // One expected commitment PER ROUND — the batched open covers them
            // all in a single proof.
            fri,
            mmcs,
            commitments.len(),
        ),
        log_stacking_height,
    );
    verifier.verify_trusted_evaluation(
        commitments,
        areas,
        eval_point,
        proof,
        evaluation_claim,
        challenger,
    )
}

/// Ring-generic counterpart of [`verify_jagged_pcs_rounds`]: verify ONE batched
/// BaseFold opening that covers every round.  The BN254 wrap ring reaches the
/// multi-round verify through this.
#[allow(clippy::too_many_arguments)]
pub fn verify_jagged_pcs_rounds_generic<Challenger, MT>(
    commitments: &[<MT as p3_commit::Mmcs<JaggedVal>>::Commitment],
    areas: &[usize],
    log_stacking_height: u32,
    eval_point: &[JaggedChallenge],
    evaluation_claim: JaggedChallenge,
    proof: &StackedBasefoldProof<JaggedVal, JaggedChallenge, MT>,
    challenger: &mut Challenger,
    mmcs: MT,
    fri: FriConfig<JaggedVal>,
) -> Result<(), crate::basefold::StackedVerifierError>
where
    MT: p3_commit::Mmcs<JaggedVal, Commitment: Clone> + Clone,
    Challenger: p3_challenger::FieldChallenger<JaggedVal>
        + p3_challenger::GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as p3_commit::Mmcs<JaggedVal>>::Commitment>,
{
    let verifier = crate::basefold::stacked::StackedPcsVerifier::new(
        crate::basefold::verifier::BasefoldVerifier::<JaggedVal, JaggedChallenge, MT>::new(
            // One expected commitment PER ROUND.
            fri,
            mmcs,
            commitments.len(),
        ),
        log_stacking_height,
    );
    verifier.verify_trusted_evaluation(
        commitments,
        areas,
        eval_point,
        proof,
        evaluation_claim,
        challenger,
    )
}

pub fn verify_jagged_pcs(
    commitment: &<JaggedMmcs as p3_commit::Mmcs<JaggedVal>>::Commitment,
    area: usize,
    log_stacking_height: u32,
    eval_point: &[JaggedChallenge],
    evaluation_claim: JaggedChallenge,
    proof: &StackedBasefoldProof<JaggedVal, JaggedChallenge, JaggedMmcs>,
    challenger: &mut JaggedChallenger,
) -> Result<(), crate::basefold::StackedVerifierError> {
    let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
    let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
    let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
    let mmcs = JaggedMmcs::new(hash, compress, 0);
    // Delegate to the GC-generic core (inner = Poseidon2-KoalaBear Mmcs).
    verify_jagged_pcs_generic::<JaggedChallenger, JaggedMmcs>(
        commitment,
        area,
        log_stacking_height,
        eval_point,
        evaluation_claim,
        proof,
        challenger,
        mmcs,
        FriConfig::<JaggedVal>::from_env_or_default(),
    )
}

/// BaseFold-over-BN254 port: GC-generic verify core.  Parameterized
/// over the challenger `Challenger` + MMCS `MT` + DFT `D`; the caller
/// supplies the concrete `mmcs`/`dft`.  The inner path uses `JaggedChallenger`
/// + Poseidon2-KoalaBear Mmcs; the wrap (OuterSC) will pass the BN254
/// challenger + Poseidon2-BN254 Mmcs.  `Val`/`Challenge` stay KoalaBear /
/// KoalaBear⁴ for both.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn verify_jagged_pcs_generic<Challenger, MT>(
    commitment: &<MT as p3_commit::Mmcs<JaggedVal>>::Commitment,
    area: usize,
    log_stacking_height: u32,
    eval_point: &[JaggedChallenge],
    evaluation_claim: JaggedChallenge,
    proof: &StackedBasefoldProof<JaggedVal, JaggedChallenge, MT>,
    challenger: &mut Challenger,
    mmcs: MT,
    fri: FriConfig<JaggedVal>,
) -> Result<(), crate::basefold::StackedVerifierError>
where
    MT: p3_commit::Mmcs<JaggedVal, Commitment: Clone> + Clone,
    Challenger: p3_challenger::FieldChallenger<JaggedVal>
        + p3_challenger::GrindingChallenger<Witness = JaggedVal>
        + CanObserve<<MT as p3_commit::Mmcs<JaggedVal>>::Commitment>,
{
    let verifier = StackedPcsVerifier::new(
        BasefoldVerifier::<JaggedVal, JaggedChallenge, MT>::new(fri, mmcs, 1),
        log_stacking_height,
    );
    verifier.verify_trusted_evaluation(
        core::slice::from_ref(commitment),
        &[area],
        eval_point,
        proof,
        evaluation_claim,
        challenger,
    )
}

// ─── Jagged-sumcheck integration ──────────
//
// The dense polynomial is still materialized for the sumcheck reduction (the
// memory win is in the commit phase: BaseFold streams stripes through
// dft_batch instead of blowing up the whole dense vector by 16×).
//
// Built on `jagged.rs` (data structures) and `jagged_sumcheck.rs`
// (PCS-agnostic reduction math).

pub mod jagged {
    use alloc::vec::Vec;

    use p3_challenger::{CanObserve, FieldChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::basefold::StackedBasefoldProof;
    use crate::jagged::{JaggedChipInfo, JaggedPacking};
    use crate::jagged_sumcheck::{verify_jagged_reduction, JaggedReductionProof};
    use crate::kb31_poseidon2::{InnerChallenge, InnerVal};

    /// A named per-chip trace in the form the jagged commit and open consume.
    /// Padding is virtual, so cloning one is an `Arc` bump rather than a
    /// matrix copy, and the Val<->InnerVal relabel stays a zero-copy slice
    /// reinterpret under the caller's TypeId gate.
    ///
    /// Device-resident chips carry a width-0 entry; the host fallback
    /// `rematerialize_chip_traces_via_provider` produces owned side-storage
    /// the caller re-wraps.
    pub type ChipTraceView = (alloc::string::String, crate::multilinear::PaddedMle<InnerVal>);

    use super::FriConfig;

    /// Wire-format jagged metadata: only the per-bundle quantities
    /// the verifier needs to reconstruct the same `JaggedPacking`
    /// from chip_infos it receives separately.  We don't serialize
    /// `dense_values` (that's the multi-GB vector we just committed
    /// to BaseFold).
    ///
    /// `column_counts`: per-chip *actual*
    /// column count as exercised by this shard's trace, written by
    /// the prover from `compute_jagged_metadata`.  The verifier reads
    /// this instead of `BaseAir::width(chip)` so the prover can send
    /// `trace.width` (the truly-populated columns) without any
    /// chip.width() pad.
    /// Empty vec on the wire = legacy bundle → caller falls back to
    /// `BaseAir::width(chip)` for backward compat.
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub struct PackingMeta {
        pub offsets: Vec<usize>,
        pub total_values: usize,
        pub log_dense_size: usize,
        #[serde(default)]
        pub column_counts: Vec<usize>,
        /// Per-ROUND `(row_count, column_count)` for the REAL chips of each
        /// committed round, in round order.
        ///
        /// The fields above flatten every round into ONE column space, and that
        /// space also carries the stacking padding between rounds, so a round's
        /// geometry cannot be recovered from them by position.  Each consumer
        /// that speaks about a single round — the hash-bind (which binds the
        /// MAIN round to `main_commitment`) and the preprocessed-round check
        /// (which binds round 0 to `vk.commit`) — reads it from here instead.
        ///
        /// `serde(default)` empty on a single-round bundle, which keeps the
        /// legacy wire format byte-identical.
        #[serde(default)]
        pub round_counts: Vec<Vec<(usize, usize)>>,
        /// Each round's stacking-padding column heights, in round order — the
        /// gap between the round's real cells and the area the stacked
        /// commitment actually covers, split into columns no taller than the
        /// row cube (a taller column has no eq table to be weighed against).
        ///
        /// It is NOT derivable from `round_counts`: a round whose cells already
        /// fill a whole number of stripes still gets a full extra stripe, so
        /// `next_multiple_of` under-counts it by `1 << log_stacking_height` and
        /// the reconstructed final offset lands a stripe short.  The recursion
        /// lift closes its column space with this height, so it has to be the
        /// prover's own value.
        ///
        /// `serde(default)` empty on a bundle with no padding columns, which
        /// keeps the legacy wire format byte-identical.
        #[serde(default)]
        pub padding_heights: Vec<Vec<usize>>,
    }

    // BaseFold-over-BN254: generic over the Mmcs so the wrap (OuterSC)
    // bundle holds the BN254 commitment + proof; inner alias below keeps every
    // caller + the rmp wire-format unchanged. serde(bound) mirrors the
    // JaggedCommitGeneric pattern (commitment + proof must serde).
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    #[serde(bound(
        serialize = "<MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment: serde::Serialize, <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Proof: serde::Serialize",
        deserialize = "<MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment: serde::Deserialize<'de>, <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Proof: serde::Deserialize<'de>"
    ))]
    pub struct JaggedBasefoldBundleGeneric<MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>> {
        /// Group-0 jagged-sumcheck reduction proof.  On the default (G==1)
        /// path this is the WHOLE proof and `extra_*` below are empty —
        /// byte-identical to the pre-split bundle.  On the per-round split
        /// (G≥2) path this is the FIRST group and the remaining G-1 groups
        /// live in the `extra_*` Vecs (see [`Self::reduction_g`]).
        ///
        /// "Scalar group-0 + extra Vecs" rather than a single `Vec<G>` keeps
        /// the recursion lift + in-circuit verifier
        /// (`shard_level_witness.rs`, `recursive_jagged_pcs.rs`) — which read
        /// these scalar fields — compiling unchanged, AND keeps the G==1 wire
        /// format byte-identical (the `extra_*` / `groups` fields are
        /// `serde(default)` empty so they serialize away).
        pub reduction: JaggedReductionProof<InnerChallenge>,
        /// Group-0 BaseFold open proof.
        pub basefold_proof: StackedBasefoldProof<InnerVal, InnerChallenge, MT>,
        /// Group-0 WHIR open proof — `Some` when the shard was proven under
        /// the jagged-WHIR inner PCS (the core-machine default); the
        /// `basefold_proof` slot then holds an empty placeholder.  The
        /// verifier dispatches on this field.
        #[serde(default)]
        pub whir_proof:
            Option<crate::whir::stacked::StackedWhirProof<InnerVal, InnerChallenge, MT>>,
        /// Per-chip per-column row-MLE values, FLAT in name-sorted chip order
        /// (NOT grouped) — the `groups` index map below partitions it.  Shared
        /// across all groups.
        pub y_per_chip: Vec<Vec<InnerChallenge>>,
        /// Group-0 BaseFold commit.
        pub commit: crate::jagged_pcs::JaggedCommitGeneric<MT>,
        /// Group-0 packing metadata (group-LOCAL offsets / prefix-sums).
        pub packing: PackingMeta,
        /// Group-0 jagged-eval sub-protocol proof.
        ///
        /// `serde(default)` so existing wire-format bundles deserialize.
        #[serde(default = "crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof::dummy")]
        pub jagged_eval: crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof<InnerChallenge>,

        /// The RAW BaseFold roots of the opening rounds BEFORE the last, in
        /// round order.
        ///
        /// A round's commitment as the VERIFYING KEY holds it is the HASH-BOUND
        /// digest `compress([raw, hash(geometry)])`, but the BaseFold open
        /// Merkle-verifies its leaves against the RAW root.  So the proof
        /// carries the raw root and the verifier re-derives the bound form to
        /// check it against the key — which is what pins that round's
        /// geometry.  Empty on a single-round proof.
        #[serde(default)]
        pub preceding_commits:
            Vec<<MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment>,

        // ── Per-round split extra groups (G≥2 only) ───────────────────────
        // All `serde(default)` empty so a G==1 bundle is byte-identical to the
        // pre-split wire format.  Indexed g-1 for group g≥1.
        /// Reductions for groups 1..G.
        #[serde(default)]
        pub extra_reduction: Vec<JaggedReductionProof<InnerChallenge>>,
        /// BaseFold opens for groups 1..G.
        #[serde(default)]
        pub extra_basefold_proof: Vec<StackedBasefoldProof<InnerVal, InnerChallenge, MT>>,
        /// BaseFold commits for groups 1..G.
        #[serde(default)]
        pub extra_commit: Vec<crate::jagged_pcs::JaggedCommitGeneric<MT>>,
        /// Packing metadata for groups 1..G.
        #[serde(default)]
        pub extra_packing: Vec<PackingMeta>,
        /// Jagged-eval proofs for groups 1..G.
        #[serde(default)]
        pub extra_jagged_eval:
            Vec<crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof<InnerChallenge>>,
        /// Group membership: `groups[g]` lists the indices (INTO the
        /// name-sorted chip set) committed in group `g`.  THE wire form the
        /// verifier's coverage check validates against an independent
        /// [`crate::jagged::partition_from_chip_infos`] run.  Must be an
        /// exact cover (no drop, no dup, canonical name-sorted order).
        /// `serde(default)` empty for G==1 / legacy bundles → the verifier
        /// treats an empty map as the single-group identity cover.
        #[serde(default)]
        pub groups: Vec<Vec<usize>>,
    }

    impl<MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>> JaggedBasefoldBundleGeneric<MT> {
        /// Number of independent jagged groups (G).  `1` on the default path.
        #[must_use]
        pub fn num_groups(&self) -> usize {
            1 + self.extra_reduction.len()
        }
        /// Reduction proof for group `g` (group 0 is the scalar field).
        #[must_use]
        pub fn reduction_g(&self, g: usize) -> &JaggedReductionProof<InnerChallenge> {
            if g == 0 {
                &self.reduction
            } else {
                &self.extra_reduction[g - 1]
            }
        }
        /// BaseFold open proof for group `g`.
        #[must_use]
        pub fn basefold_proof_g(
            &self,
            g: usize,
        ) -> &StackedBasefoldProof<InnerVal, InnerChallenge, MT> {
            if g == 0 {
                &self.basefold_proof
            } else {
                &self.extra_basefold_proof[g - 1]
            }
        }
        /// BaseFold commit for group `g`.
        #[must_use]
        pub fn commit_g(&self, g: usize) -> &crate::jagged_pcs::JaggedCommitGeneric<MT> {
            if g == 0 {
                &self.commit
            } else {
                &self.extra_commit[g - 1]
            }
        }
        /// Packing metadata for group `g`.
        #[must_use]
        pub fn packing_g(&self, g: usize) -> &PackingMeta {
            if g == 0 {
                &self.packing
            } else {
                &self.extra_packing[g - 1]
            }
        }
        /// Jagged-eval proof for group `g`.
        #[must_use]
        pub fn jagged_eval_g(
            &self,
            g: usize,
        ) -> &crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof<InnerChallenge> {
            if g == 0 {
                &self.jagged_eval
            } else {
                &self.extra_jagged_eval[g - 1]
            }
        }
        /// The group membership map, defaulting to the identity single-group
        /// cover when empty (G==1 / legacy bundles).
        #[must_use]
        pub fn groups_or_identity(&self, n_chips: usize) -> Vec<Vec<usize>> {
            if self.groups.is_empty() {
                alloc::vec![(0..n_chips).collect()]
            } else {
                self.groups.clone()
            }
        }
    }

    /// Concrete inner (Poseidon2-KoalaBear) bundle alias -- the type every
    /// current caller + wire-format uses.
    pub type JaggedBasefoldBundle = JaggedBasefoldBundleGeneric<crate::jagged_pcs::JaggedMmcs>;

    impl<MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>> JaggedBasefoldBundleGeneric<MT>
    where
        <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment:
            serde::Serialize + for<'d> serde::Deserialize<'d>,
        <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Proof:
            serde::Serialize + for<'d> serde::Deserialize<'d>,
    {
        /// Wire-format bytes (rmp-serde — matches the existing
        /// jagged-PCS bundle's serializer choice).
        pub fn to_bytes(&self) -> Vec<u8> {
            rmp_serde::to_vec(self).expect("JaggedBasefoldBundle serializes")
        }

        pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
            rmp_serde::from_slice(bytes).ok()
        }
    }

    /// Pre-computed jagged-PCS commit bundle for the
    /// single-main-commit flow.  Produced by
    /// [`crate::config::BasefoldRing::commit_multilinears`] before the shard-level
    /// Phase 1 prologue, then consumed as a [`JaggedOpenRound`] by
    /// [`prove_jagged_basefold_rounds`] in Phase 4.
    ///
    /// The 8-felt digest of `commit.original_commitment` (via
    /// [`crate::jagged_pcs::basefold_commit_digest`]) is the
    /// `main_commitment` that the prologue + verifier observe.
    pub struct PrecomputedJaggedCommitGeneric<MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>> {
        pub packing: crate::jagged::JaggedPacking<InnerVal>,
        pub commit: crate::jagged_pcs::JaggedCommitGeneric<MT>,
        pub prover_data: crate::jagged_pcs::JaggedProverDataGeneric<MT>,
        /// WHIR-committed data when the shard runs the jagged-WHIR inner PCS
        /// (the core-machine default); `prover_data` then carries the SAME
        /// width-1 polynomials as `interleaved_mles` (the reduction reads
        /// them) over a placeholder Merkle tree.
        pub whir_data: Option<crate::whir::jagged::JaggedWhirProverDataGeneric<MT>>,
        /// The per-shard rev(zeta) orientation the dense commit was
        /// materialized under (from the per-stage `StarkMachine::core_rev()`
        /// source of truth — `true` only on the CORE MIPS path).  Recorded on
        /// the committed data so the step-4 jagged reduction (host
        /// re-materialize + `y_per_chip`) uses the SAME orientation as the
        /// commit, in lockstep.  `false` on every recursion / shrink / wrap
        /// commit (byte-identical).
        pub rev: bool,
    }
    /// Concrete inner alias (MT = JaggedMmcs).
    pub type PrecomputedJaggedCommit =
        PrecomputedJaggedCommitGeneric<crate::jagged_pcs::JaggedMmcs>;

    // ─────────────────────────────────────────────────────────────────
    // Single shard-wide commit buffer — GPU precompute-commit hook.
    //
    // Device-side build of the precompute commit: resident chips are
    // packed D2D from the per-shard provider, host chips H2D once, the
    // stripes/encode/Merkle all run on device, and the dense buffer is
    // retained device-side (registered handle) for the step-4 jagged
    // reduction.  Output MUST be byte-identical to the host precompute
    // (commit digest, prover_data shapes, interleaved MLE bytes) — the
    // commit is transcript-critical.
    // ─────────────────────────────────────────────────────────────────

    // The device commit is built by the device prover's own `commit()`; its
    // recursion-AREA-PIN + provider-read rev(zeta) semantics match this host
    // path byte-identically.

    // The generic BaseFold precompute body now lives as the DEFAULT
    // `BasefoldRing::commit_multilinears` trait method (no free-fn
    // indirection); each ring commits with its own `bf_mmcs()` /
    // `fri_config()`.

    /// One commitment ROUND's inputs to the jagged open.
    ///
    /// The per-round prover data and evaluation claims are collapsed into one
    /// record (rather than parallel per-round collections), so a round's
    /// commit can never be paired with another round's claims.
    ///
    /// `precomputed` is BORROWED: the preprocessed round opens the proving
    /// key's commit, built once by `setup` and shared by every shard.
    pub struct JaggedOpenRound<'a, MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>> {
        pub chip_traces: &'a [ChipTraceView],
        pub r_row_per_chip: &'a [Vec<InnerChallenge>],
        /// This round's per-chip column claims.
        pub claims: Vec<Vec<InnerChallenge>>,
        pub precomputed: &'a PrecomputedJaggedCommitGeneric<MT>,
    }

    /// Build borrowed `ChipTraceView`s over an OWNED re-materialized trace
    /// set (`rematerialize_chip_traces_via_provider`), so the downstream
    /// view-taking commit/reduction consumers can read its cells with no
    /// further copy.  The returned views borrow `owned`, which must outlive
    /// them (the caller keeps it in scope).
    pub fn views_over_owned(
        owned: &[(alloc::string::String, RowMajorMatrix<InnerVal>)],
    ) -> alloc::vec::Vec<ChipTraceView> {
        owned
            .iter()
            .map(|(name, m)| {
                // The rematerialized side-storage is wrapped in a `PaddedMle`.
                // `num_variables` is the chip's own log-height: the packer
                // reads dims and cells back off the real trace, and never
                // consults the padding.
                let h = if m.width == 0 { 0 } else { m.values.len() / m.width };
                let log_h = if h <= 1 { 0 } else { h.next_power_of_two().ilog2() };
                let mle = alloc::sync::Arc::new(crate::basefold::Mle::from_row_major(
                    RowMajorMatrix::new(m.values.clone(), m.width),
                ));
                (name.clone(), crate::multilinear::PaddedMle::padded_with_zeros(mle, log_h))
            })
            .collect()
    }

    /// **Shared linear core** — the path-INDEPENDENT challenger sequence at
    /// the heart of every jagged-BaseFold prove: sample `z_col` at the
    /// verifier-matching transcript position → jagged-sumcheck reduction →
    /// jagged-eval sub-protocol → point-extension `log_dense → log2(area)`
    /// via extra Fiat-Shamir coords → open.
    ///
    /// The `reduce` and `open` closures are the ONLY per-path variation
    /// (host-owned vs device-hook reduction; concrete vs BN254 open; the
    /// pre-reduce / pre-open device-memory frees) and NEITHER may run any
    /// challenger op outside its documented reduction/open — so every caller
    /// (single-group concrete, per-group multi-group, BN254 generic) lands its
    /// `z_col` / reduction / jagged-eval / point-extend / open challenger ops
    /// in the IDENTICAL order.  This is the de-dup that stops `z_col`'s
    /// transcript position from being path-dependent.
    #[allow(clippy::type_complexity)]
    pub fn prove_jagged_basefold_linear_core<Ch, P>(
        offsets: &[usize],
        z_row: &[InnerChallenge],
        area: usize,
        challenger: &mut Ch,
        reduce: impl FnOnce(&[InnerChallenge], &mut Ch) -> JaggedReductionProof<InnerChallenge>,
        open: impl FnOnce(Vec<InnerChallenge>, &mut Ch) -> P,
    ) -> (
        JaggedReductionProof<InnerChallenge>,
        crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof<InnerChallenge>,
        P,
    )
    where
        Ch: FieldChallenger<InnerVal> + 'static,
    {
        // (4) Sample `z_col` (one challenge per column variable) at the
        // verifier-matching transcript position — after the commit observe,
        // immediately before the jagged sumcheck reduction.  Used both to
        // weight the column mix in the reduction and as the column point for
        // the branching-program jagged-eval sub-protocol.
        let num_cols = offsets.len().saturating_sub(1);
        let num_col_vars = num_cols.next_power_of_two().trailing_zeros() as usize;
        let z_col: Vec<InnerChallenge> =
            (0..num_col_vars).map(|_| challenger.sample_algebra_element()).collect();

        // Jagged sumcheck reduction.  The caller's closure supplies the
        // host-owned / device-hook / group-local body; it MUST be
        // transcript-equivalent to
        // `prove_jagged_reduction_owned(.., &z_col, z_row, ..)` (the device
        // hook is byte-equivalent + snapshot-guarded; see the concrete path).
        let reduction = reduce(&z_col, challenger);

        // (4b) Jagged-eval sub-protocol at (z_row, z_col, rev(z*)).  PHASE 2:
        // the BranchingProgram reads its z_index big-endian while the
        // reduction emits z_star little-endian, so feed rev(z_star) — matches
        // recursive_jagged_pcs.rs (verify_sumcheck → jagged_evaluator_fn).
        let z_trace_be: Vec<InnerChallenge> = reduction.eval_point.iter().rev().copied().collect();
        let jagged_eval = crate::jagged_eval_sumcheck::prove_jagged_evaluation(
            offsets,
            z_row,
            &z_col,
            &z_trace_be,
            challenger,
        );

        // (5) Point-extend: the BaseFold commit covers `area` cells
        // (num_stripes × batch_size × stack_height), which can exceed
        // 2^log_dense_size, so extend z* to log2(area) with extra Fiat-Shamir
        // coords (the verifier samples the matching coords in the same
        // transcript order), then open at z*.
        let target_dim = area.trailing_zeros() as usize;
        let mut extended_eval_point = reduction.eval_point.clone();
        while extended_eval_point.len() < target_dim {
            let r: InnerChallenge = challenger.sample_algebra_element();
            extended_eval_point.push(r);
        }
        let proof = open(extended_eval_point, challenger);

        (reduction, jagged_eval, proof)
    }

    /// Multi-ROUND prove: ONE jagged proof spanning every committed round —
    /// one jagged sumcheck over every round's stripes, ONE jagged-eval, and
    /// ONE batched BaseFold open; only the commitments are per round.
    /// Proving each round as its own jagged group instead would cost a
    /// reduction, an eval and an open PER ROUND.
    ///
    /// The rounds are concatenated into ONE column space: round r's offsets are
    /// shifted by the total cell count of the rounds before it, so the packing
    /// the reduction and the jagged-eval see is a single jagged matrix whose
    /// columns run `[round 0 | round 1 | ...]`.
    ///
    /// Round order is load-bearing — it fixes the column order the verifier
    /// reconstructs — and the preprocessed round comes first.
    /// The INNER ring's multi-round prove — the ring's own Mmcs / DFT / FRI
    /// config, forwarded to the generic body below.
    pub fn prove_jagged_basefold_rounds(
        rounds: &[JaggedOpenRound<'_, crate::jagged_pcs::JaggedMmcs>],
        z_row: &[InnerChallenge],
        challenger: &mut crate::jagged_pcs::JaggedChallenger,
    ) -> JaggedBasefoldBundle {
        let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
        let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
        let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
        let mmcs = crate::jagged_pcs::JaggedMmcs::new(hash, compress, 0);
        let dft = alloc::sync::Arc::new(crate::jagged_pcs::JaggedDft::default());
        prove_jagged_basefold_rounds_generic::<
            crate::jagged_pcs::JaggedChallenger,
            crate::jagged_pcs::JaggedMmcs,
            crate::jagged_pcs::JaggedDft,
        >(
            rounds,
            z_row,
            challenger,
            mmcs,
            dft,
            crate::basefold::FriConfig::<crate::jagged_pcs::JaggedVal>::from_env_or_default(),
        )
    }

    /// Ring-generic body.  The INNER (Poseidon2-KoalaBear) ring reaches it
    /// through [`prove_jagged_basefold_rounds`]; the BN254 wrap ring names its
    /// own commitment family, which is what lets the terminal stage open a
    /// preprocessed round like every other stage.
    #[allow(clippy::type_complexity)]
    pub fn prove_jagged_basefold_rounds_generic<Challenger, MT, D>(
        rounds: &[JaggedOpenRound<'_, MT>],
        z_row: &[InnerChallenge],
        challenger: &mut Challenger,
        mmcs: MT,
        dft: alloc::sync::Arc<D>,
        fri: crate::basefold::FriConfig<crate::jagged_pcs::JaggedVal>,
    ) -> JaggedBasefoldBundleGeneric<MT>
    where
        MT: p3_commit::Mmcs<
                crate::jagged_pcs::JaggedVal,
                Commitment: Clone,
                ProverData<p3_matrix::dense::RowMajorMatrix<crate::jagged_pcs::JaggedVal>>: 'static,
            > + Clone,
        D: p3_dft::TwoAdicSubgroupDft<crate::jagged_pcs::JaggedVal> + Send + Sync,
        Challenger: p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + p3_challenger::CanObserve<
                <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment,
            >
            // `'static`: `deterministic_grind` looks the challenger type up in
            // the grind-accelerator registration, which is keyed by `TypeId`.
            + 'static,
    {
        assert!(!rounds.is_empty(), "prove_jagged_basefold_rounds: no rounds");

        // ── Concatenate the rounds into one column space ──────────────────
        let mut chip_infos: Vec<crate::jagged::JaggedChipInfo> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        let mut round_padding_heights: Vec<Vec<usize>> = Vec::with_capacity(rounds.len());
        let mut y_per_chip: Vec<Vec<InnerChallenge>> = Vec::new();
        let mut r_row_per_chip: Vec<Vec<InnerChallenge>> = Vec::new();
        let mut base = 0usize;
        for r in rounds.iter() {
            let pk = &r.precomputed.packing;
            chip_infos.extend(pk.chip_infos.iter().cloned());
            // Drop each round's sentinel; re-base its column offsets onto the
            // running total.
            let n_cols = pk.offsets.len().saturating_sub(1);
            offsets.extend(pk.offsets.iter().take(n_cols).map(|o| o + base));
            y_per_chip.extend(r.claims.iter().cloned());
            r_row_per_chip.extend(r.r_row_per_chip.iter().cloned());

            // The rounds are NOT contiguous in the committed dense: the stacked
            // commitment rounds each one UP to a whole number of stripes
            // (`area = total_values.next_multiple_of(1 << log_stacking_height)`)
            // and the batched open indexes every round's stripes end to end.  So
            // a round FOLLOWED BY ANOTHER has a gap of real committed space in
            // the middle of the column layout, and the offsets can only stay a
            // prefix sum if a column covers it — a DUMMY column of zeros with a
            // ZERO claim.
            //
            // EVERY round pads, the last one included.  Its gap looks like it
            // could be left to the hypercube padding the point extension covers
            // — but `total_values` is what makes the reduction's hypercube equal
            // the committed area, and dropping the last round's fill shrinks it
            // below `effective_area`, so the stacked claim no longer equals the
            // interpolated batch evaluations (StackingMismatch).  The column
            // layout has to cover every committed cell.
            let area = r.precomputed.prover_data.area;
            let pad = area.saturating_sub(pk.total_values);
            {
                // Split the gap into whole COLUMNS bounded by the row cube —
                // a column taller than `2^z_row.len()` has no eq table to be
                // weighed against.
                //
                // ALWAYS at least one, even when the round happens to land on a
                // stripe boundary.  A zero-height column
                // costs nothing and is what keeps the column COUNT a function of
                // the machine rather than of how full this particular shard is,
                // which is what lets the recursion circuit carry a fixed layout.
                let cube = 1usize << z_row.len();
                let mut done = 0usize;
                let mut pad_off = base + pk.total_values;
                let mut this_round_pads: Vec<usize> = Vec::new();
                loop {
                    let h = core::cmp::min(cube, pad - done);
                    this_round_pads.push(h);
                    offsets.push(pad_off);
                    chip_infos.push(crate::jagged::JaggedChipInfo {
                        name: alloc::format!("<stacking-pad:{}>", chip_infos.len()),
                        row_count: h,
                        column_count: 1,
                    });
                    y_per_chip.push(alloc::vec![InnerChallenge::ZERO]);
                    let log_h = h.max(1).next_power_of_two().trailing_zeros() as usize;
                    r_row_per_chip.push(z_row[z_row.len() - log_h..].to_vec());
                    done += h;
                    pad_off += h;
                    if done >= pad {
                        break;
                    }
                }
                round_padding_heights.push(this_round_pads);
            }
            base += area;
        }
        let total_values = base;
        offsets.push(total_values);
        // The rounds' areas are already carried as explicit padding columns, so
        // the concatenated instance's committed length IS its column space.
        let packing = crate::jagged::JaggedPacking::<InnerVal> {
            dense_values: Vec::new(),
            chip_infos: chip_infos.clone(),
            offsets: offsets.clone(),
            total_values,
            dense_len: total_values,
        };
        let n_chips = chip_infos.len();

        // ── The reduction, over the CONCATENATED dense ────────────────────
        let reduce = |z_col: &[InnerChallenge],
                      challenger: &mut Challenger|
         -> crate::jagged_sumcheck::JaggedReductionProof<InnerChallenge> {
            let _red_span = tracing::info_span!("jagged_sumcheck_reduce").entered();
            let weights = crate::jagged_sumcheck::build_weight_table_from_z_col(
                &packing,
                &r_row_per_chip,
                z_col,
                z_row,
            );
            // Each round's dense cells are the PREFIX of its committed stripes;
            // laid end to end they are the concatenated jagged matrix, then
            // zero-padded to the combined hypercube.
            let log_dense_size = packing.log_dense_size();
            let mut dense_q: Vec<InnerVal> = Vec::with_capacity(1usize << log_dense_size);
            for r in rounds.iter() {
                // Each round contributes its FULL committed cell space — real
                // cells followed by the stacking padding — because that is what
                // the batched open indexes.
                let area = r.precomputed.prover_data.area;
                let round_dense = crate::basefold::stacked::dense_from_interleaved_mles::<InnerVal>(
                    &r.precomputed.prover_data.stacked_data.interleaved_mles,
                    area,
                );
                dense_q.extend_from_slice(&round_dense[..area]);
            }
            dense_q.resize(1usize << log_dense_size, InnerVal::ZERO);
            let hp = crate::jagged_long::HadamardProduct {
                base: crate::jagged_long::LongMle::from_components(
                    alloc::vec![crate::basefold::Mle::from_values(dense_q)],
                    log_dense_size as u32,
                ),
                ext: crate::jagged_long::LongMle::from_components(
                    alloc::vec![crate::basefold::Mle::from_values(weights)],
                    log_dense_size as u32,
                ),
            };
            crate::jagged_long::prove_jagged_reduction_hadamard_poly(hp, challenger)
        };

        // ── ONE batched open across every round's committed data ──────────
        // In WHIR mode (every round carries `whir_data`) the open runs the
        // jagged-WHIR sibling; the WHIR proof is captured into `whir_slot`
        // and the closure returns an EMPTY BaseFold placeholder so the shared
        // reduction core's return type stays fixed.
        let whir_any = rounds.iter().any(|r| r.precomputed.whir_data.is_some());
        let whir_mode = whir_any && rounds.iter().all(|r| r.precomputed.whir_data.is_some());
        if whir_any && !whir_mode {
            // A round lacks WHIR data while another carries it: the open
            // falls back to BaseFold while the caller may have observed a
            // WHIR root — an inconsistent proof.  Surface which is missing.
            let flags: alloc::vec::Vec<bool> =
                rounds.iter().map(|r| r.precomputed.whir_data.is_some()).collect();
            eprintln!(
                "[whir open] MIXED rounds, falling back to BaseFold: per-round whir_data presence = {flags:?}                  (round order: preceding/preprocessed first, main last)"
            );
        }
        let whir_slot: core::cell::RefCell<
            Option<crate::whir::stacked::StackedWhirProof<InnerVal, InnerChallenge, MT>>,
        > = core::cell::RefCell::new(None);
        let open = |extended_eval_point: Vec<InnerChallenge>, challenger: &mut Challenger| {
            let _open_span = tracing::info_span!("jagged_basefold_open").entered();
            if whir_mode {
                let wdatas: Vec<&crate::whir::jagged::JaggedWhirProverDataGeneric<MT>> = rounds
                    .iter()
                    .map(|r| r.precomputed.whir_data.as_ref().expect("whir_mode"))
                    .collect();
                let lsh = wdatas[0].log_stacking_height as usize;
                let cfg = crate::whir::jagged::core_whir_config(lsh);
                let ef_dft =
                    alloc::sync::Arc::new(p3_dft::Radix2DitParallel::<InnerChallenge>::default());
                let proof = crate::whir::jagged::open_jagged_whir_rounds_generic::<
                    Challenger,
                    MT,
                    D,
                    _,
                >(
                    &wdatas, extended_eval_point, challenger, mmcs, dft, ef_dft, cfg
                );
                *whir_slot.borrow_mut() = Some(proof);
                StackedBasefoldProof {
                    basefold_proof: crate::basefold::proof::BasefoldProof {
                        univariate_messages: Vec::new(),
                        fri_commitments: Vec::new(),
                        component_polynomials_query_openings_and_proofs: Vec::new(),
                        query_phase_openings_and_proofs: Vec::new(),
                        final_poly: InnerChallenge::ZERO,
                        pow_witness: InnerVal::ZERO,
                        batch_grinding_witness: InnerVal::ZERO,
                    },
                    batch_evaluations: Vec::new(),
                }
            } else {
                let datas: Vec<&crate::jagged_pcs::JaggedProverDataGeneric<MT>> =
                    rounds.iter().map(|r| &r.precomputed.prover_data).collect();
                crate::jagged_pcs::open_jagged_pcs_rounds_generic::<Challenger, MT, D>(
                    &datas,
                    extended_eval_point,
                    challenger,
                    mmcs,
                    dft,
                    fri,
                )
            }
        };

        // The batched open's point spans the stack coords plus enough batch
        // coords to index EVERY round's stripes end to end; the stripe total
        // need not be a power of two (the verifier zero-pads it), so the
        // dimension is the ceiling.
        let log_stacking_height = rounds[0].precomputed.prover_data.log_stacking_height as usize;
        let total_stripes: usize =
            rounds.iter().map(|r| r.precomputed.prover_data.area >> log_stacking_height).sum();
        let batch_dim = total_stripes.max(1).next_power_of_two().trailing_zeros() as usize;
        let effective_area = 1usize << (log_stacking_height + batch_dim);

        let (reduction, jagged_eval, proof) = prove_jagged_basefold_linear_core(
            &offsets,
            z_row,
            effective_area,
            challenger,
            reduce,
            open,
        );
        let _ = n_chips;

        // The bundle carries the LAST round's commit — the main one, which the
        // hash-bind ties to `main_commitment`.  An earlier round's commitment as
        // the KEY holds it is the hash-bound digest, so the proof carries only
        // its RAW root and the verifier re-derives the bound form to check it.
        let main = rounds.last().expect("non-empty");
        let preceding_commits: Vec<_> = rounds[..rounds.len() - 1]
            .iter()
            .map(|r| r.precomputed.commit.original_commitment.clone())
            .collect();
        let packing_meta = PackingMeta {
            offsets,
            total_values,
            log_dense_size: packing.log_dense_size(),
            column_counts: chip_infos.iter().map(|ci| ci.column_count).collect(),
            // Each round's REAL chip geometry, as committed — no stacking
            // padding, which is an artifact of flattening the rounds together.
            round_counts: rounds
                .iter()
                .map(|r| {
                    r.precomputed
                        .packing
                        .chip_infos
                        .iter()
                        .map(|ci| (ci.row_count, ci.column_count))
                        .collect()
                })
                .collect(),
            padding_heights: round_padding_heights,
        };
        JaggedBasefoldBundleGeneric::<MT> {
            reduction,
            basefold_proof: proof,
            whir_proof: whir_slot.into_inner(),
            y_per_chip,
            commit: main.precomputed.commit.clone(),
            packing: packing_meta,
            jagged_eval,
            extra_reduction: Vec::new(),
            extra_basefold_proof: Vec::new(),
            extra_commit: Vec::new(),
            extra_packing: Vec::new(),
            extra_jagged_eval: Vec::new(),
            groups: Vec::new(),
            preceding_commits,
        }
    }

    /// Single-main-commit variant: verifier counterpart of
    /// [`prove_jagged_basefold_rounds`].  Skips the in-band
    /// `challenger.observe(commitment)` because the orchestrator's
    /// Phase 1 prologue already observed the BaseFold commit's 8-felt
    /// digest as `main_commitment`.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_jagged_basefold_no_observe(
        chip_infos: &[JaggedChipInfo],
        r_row_per_chip: &[Vec<InnerChallenge>],
        z_row: &[InnerChallenge], // full z* for embedding factor
        // Rounds committed BEFORE the proof's own, whose commitments come from
        // the VERIFYING KEY (the preprocessed round), as (commitment, area).
        preceding_rounds: &[(
            <crate::jagged_pcs::JaggedMmcs as p3_commit::Mmcs<
                crate::jagged_pcs::JaggedVal,
            >>::Commitment,
            usize,
        )],

        // Number of leading PREPROCESSED entries in `chip_infos` — 0 for a
        // main-only proof, otherwise the two-round (preprocessed + main)
        // split.  Read off the VERIFYING KEY, never off the proof: it is what
        // the coverage check measures the proof's group map against.
        n_prep: usize,
        bundle: &JaggedBasefoldBundle,
        // Cross-bind: per-chip `opened_values.chips[].main.local` (index-
        // aligned with `chip_infos` / `bundle.y_per_chip`); `None` disables the bind.
        opened_main: Option<&[Vec<InnerChallenge>]>,
        challenger: &mut crate::jagged_pcs::JaggedChallenger,
    ) -> bool {
        verify_jagged_basefold_inner(
            chip_infos,
            r_row_per_chip,
            z_row,
            n_prep,
            preceding_rounds,
            bundle,
            opened_main,
            challenger,
            /* skip_commit_observe = */ true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_jagged_basefold_inner(
        chip_infos: &[JaggedChipInfo],
        r_row_per_chip: &[Vec<InnerChallenge>],
        z_row: &[InnerChallenge], // full z* for embedding factor
        _n_prep: usize,
        // Rounds committed BEFORE the proof's own, whose commitments come from
        // the VERIFYING KEY (the preprocessed round), as (commitment, area).
        preceding_rounds: &[(
            <crate::jagged_pcs::JaggedMmcs as p3_commit::Mmcs<
                crate::jagged_pcs::JaggedVal,
            >>::Commitment,
            usize,
        )],

        bundle: &JaggedBasefoldBundle,
        // Cross-bind: per-chip `opened_values.chips[].main.local` trace
        // openings (index-aligned with `chip_infos` / `bundle.y_per_chip`), or
        // `None` for synthetic-bundle unit tests with no shard openings.
        opened_main: Option<&[Vec<InnerChallenge>]>,
        challenger: &mut crate::jagged_pcs::JaggedChallenger,
        skip_commit_observe: bool,
    ) -> bool {
        // ── COVERAGE CHECK (the #1 soundness guard — FIRST assertion) ─────
        // Independently re-derive the round partition from the PUBLIC
        // name-sorted (name,row_count,column_count) the verifier already
        // holds, and require the proof's `groups` membership to equal it
        // EXACTLY (no chip dropped, none duplicated, canonical order).
        // Without this a malicious prover could OMIT a chip from every
        // group — that chip's trace is then never opened, and nothing
        // downstream would notice.
        let expected_groups = crate::jagged::partition_from_chip_infos(chip_infos);
        let proof_groups: Vec<Vec<usize>> = bundle.groups_or_identity(chip_infos.len());
        if proof_groups != expected_groups {
            eprintln!(
                "[basefold verify] COVERAGE CHECK FAILED: proof groups {:?} != expected {:?}",
                proof_groups, expected_groups
            );
            return false;
        }
        // Structural agreement between the group map and the per-group data.
        let g_count = proof_groups.len();
        if bundle.num_groups() != g_count {
            eprintln!(
                "[basefold verify] COVERAGE CHECK FAILED: bundle carries {} groups \
                 but the group map has G={}",
                bundle.num_groups(),
                g_count,
            );
            return false;
        }

        // STEP 1 (transcript): observe ALL G commits up-front in partition
        // order — unless the single-main-commit flow already observed them at
        // the Phase 1 prologue.
        if !skip_commit_observe {
            for g in 0..g_count {
                challenger.observe(bundle.commit_g(g).original_commitment.clone());
            }
        }

        // STEP 3: verify each independent jagged instance against the SHARED
        // z_row.  All G must accept.
        for g in 0..g_count {
            let grp = &proof_groups[g];
            // Group-LOCAL chip_infos / r_row (membership-indexed); the
            // per-group bundle metadata (offsets/total/log_dense_size) is
            // group-local too (prefix-sums restart at 0).
            let chip_infos_g: Vec<JaggedChipInfo> =
                grp.iter().map(|&i| chip_infos[i].clone()).collect();
            let r_row_g: Vec<Vec<InnerChallenge>> =
                grp.iter().map(|&i| r_row_per_chip[i].clone()).collect();
            let y_per_chip_g: Vec<Vec<InnerChallenge>> =
                grp.iter().map(|&i| bundle.y_per_chip[i].clone()).collect();
            // Slice this group's opened main.local columns in the SAME
            // membership order as `y_per_chip_g` so the cross-bind k-walk lines up.
            let opened_main_g: Option<Vec<Vec<InnerChallenge>>> =
                opened_main.map(|om| grp.iter().map(|&i| om[i].clone()).collect());
            let pkg = bundle.packing_g(g);
            let packing = JaggedPacking {
                dense_values: Vec::new(),
                chip_infos: chip_infos_g,
                offsets: pkg.offsets.clone(),
                total_values: pkg.total_values,
                // Bundle-level: the rounds' areas are already explicit padding
                // columns, so the committed length IS the column space.
                dense_len: pkg.total_values,
            };
            if !verify_one_jagged_group(
                &packing,
                &r_row_g,
                z_row,
                &y_per_chip_g,
                opened_main_g.as_deref(),
                bundle.reduction_g(g),
                bundle.jagged_eval_g(g),
                bundle.commit_g(g),
                bundle.basefold_proof_g(g),
                if g == 0 { bundle.whir_proof.as_ref() } else { None },
                challenger,
                // Only the FIRST group carries the vk-pinned rounds; with the
                // batched shape there is exactly one group.
                if g == 0 { preceding_rounds } else { &[] },
                g,
            ) {
                return false;
            }
        }
        true
    }

    /// Verify ONE independent jagged instance (group `g`) against the shared
    /// `z_row`: sample `z_col_g`, verify the reduction, replay the
    /// jagged-eval transcript, extend `z*_g`, and verify the BaseFold open
    /// of `C_g`.  Returns `false` on any rejection.
    #[allow(clippy::too_many_arguments)]
    fn verify_one_jagged_group(
        packing: &JaggedPacking<InnerVal>,
        r_row_per_chip: &[Vec<InnerChallenge>],
        z_row: &[InnerChallenge],
        y_per_chip: &[Vec<InnerChallenge>],
        // Cross-bind: this group's per-chip `opened_values.chips[].main.local`
        // trace openings (index-aligned with `y_per_chip`), or `None` for callers
        // (unit tests) that verify a synthetic bundle with no shard openings.
        opened_main: Option<&[Vec<InnerChallenge>]>,
        reduction: &crate::jagged_sumcheck::JaggedReductionProof<InnerChallenge>,
        jagged_eval: &crate::jagged_eval_sumcheck::JaggedSumcheckEvalProof<InnerChallenge>,
        commit: &crate::jagged_pcs::JaggedCommit,
        basefold_proof: &crate::basefold::StackedBasefoldProof<
            InnerVal,
            InnerChallenge,
            crate::jagged_pcs::JaggedMmcs,
        >,
        // `Some` dispatches the batched open to the jagged-WHIR verifier
        // (the shard was proven under the jagged-WHIR core PCS).
        whir_proof: Option<
            &crate::whir::stacked::StackedWhirProof<
                InnerVal,
                InnerChallenge,
                crate::jagged_pcs::JaggedMmcs,
            >,
        >,
        challenger: &mut crate::jagged_pcs::JaggedChallenger,
        // Rounds committed BEFORE this one whose commitments come from the
        // verifying key (the preprocessed round), as (commitment, area).
        preceding_rounds: &[(
            <crate::jagged_pcs::JaggedMmcs as p3_commit::Mmcs<
                crate::jagged_pcs::JaggedVal,
            >>::Commitment,
            usize,
        )],
        g: usize,
    ) -> bool {
        // Sample z_col at the matching transcript position
        // (after the commit observe, before the reduction), mirroring
        // the prover.
        let num_cols = packing.offsets.len().saturating_sub(1);
        let num_col_vars = num_cols.next_power_of_two().trailing_zeros() as usize;
        let z_col: Vec<InnerChallenge> =
            (0..num_col_vars).map(|_| challenger.sample_algebra_element()).collect();
        let red_result = verify_jagged_reduction(
            reduction,
            packing,
            r_row_per_chip,
            y_per_chip,
            &z_col,
            z_row,
            challenger,
        );
        let Some((z_star, q_at_z, _w_at_z)) = red_result else {
            eprintln!("[basefold verify] group {g}: jagged sumcheck reduction REJECTED");
            return false;
        };

        // ── CROSS-BIND (host analog of recursive_jagged_pcs.rs:247) ─────
        //
        // The recursion CIRCUIT ties the jagged sumcheck's claimed sum to the
        // TRACE OPENINGS: it forms `column_claims = opened_values.chips[].main.local`
        // (shard_basefold.rs:588 → recursive_jagged_pcs.rs:218) and asserts
        //   evaluate_mle(column_claims, z_col) == sumcheck_proof.claimed_sum   (:247).
        //
        // The host, in contrast, DERIVES the claimed sum from the bundle's
        // `y_per_chip` alone — `verify_jagged_reduction` uses
        //   t = Σ_k z_col_lagrange[k]·y_flat[k] = evaluate_mle(y_flat, z_col)
        // as its round-0 claim (jagged_sumcheck.rs:735-742) and NEVER checks
        // `y_per_chip` against the openings.  So a malicious host proof could ship
        // `y_per_chip ≠ opened_values.main.local` and be accepted by BOTH the
        // zerocheck (which consumes `opened_values`) and this jagged phase (which
        // consumes `y_per_chip`) independently — the soundness-parity gap.
        //
        // Close it exactly as the circuit does: recompute the OPENED-VALUES column
        // MLE at the SAME `z_col` and require it to equal the y-derived claimed sum
        // `t`.  We mirror the MLE form rather than a raw element-wise compare: under
        // the rev(zeta) orientation the two column vectors are NOT guaranteed
        // element-wise equal, but their `z_col`-MLEs ARE equal — that is precisely
        // the identity the circuit asserts and that passes on every honest proof.
        // We weight only the columns the sumcheck actually consumed (per-chip
        // `y_per_chip[i].len()`, i.e. the packed column_count), matching the k-walk
        // in `verify_jagged_reduction` so `sum_y` reproduces its `t` bit-for-bit.
        if let Some(opened_main_g) = opened_main {
            if opened_main_g.len() != y_per_chip.len() {
                eprintln!(
                    "[basefold verify] group {g}: cross-bind FAILED — opened-main \
                     chip count {} != y_per_chip {}",
                    opened_main_g.len(),
                    y_per_chip.len(),
                );
                return false;
            }
            let z_col_lagrange = crate::jagged_branching_program::partial_lagrange(&z_col);
            let mut sum_y = InnerChallenge::ZERO;
            let mut sum_open = InnerChallenge::ZERO;
            let mut k = 0usize;
            let mut ok = true;
            'chips: for (yc, mc) in y_per_chip.iter().zip(opened_main_g.iter()) {
                // The opened trace must expose at least the columns the sumcheck
                // consumed (column_count ≤ BaseAir::width); a proof opening fewer
                // is malformed → reject.
                if mc.len() < yc.len() {
                    ok = false;
                    break 'chips;
                }
                for j in 0..yc.len() {
                    if k >= z_col_lagrange.len() {
                        ok = false;
                        break 'chips;
                    }
                    let w = z_col_lagrange[k];
                    sum_y += w * yc[j];
                    sum_open += w * mc[j];
                    k += 1;
                }
            }
            if !ok || sum_open != sum_y {
                eprintln!(
                    "[basefold verify] group {g}: CROSS-BIND FAILED — the bundle's \
                     y_per_chip column claims are inconsistent with \
                     opened_values.main.local at z_col \
                     (evaluate_mle(opened_main, z_col) != jagged claimed_sum)"
                );
                return false;
            }
        }

        // Replay the jagged-eval sub-protocol transcript so the
        // challenger stays in sync with the prover before the BaseFold
        // open.  (Full branching-program verification is done by the
        // recursion verifier; the host self-check needs only transcript
        // fidelity here.)
        crate::jagged_eval_sumcheck::replay_jagged_evaluation_transcript(jagged_eval, challenger);

        // Extend z_star from log_dense_size to log2(area)
        // by sampling additional Fiat-Shamir coords, mirroring the
        // prover's extension in `prove_jagged_basefold_linear_core` step (5).
        // Both sides sample from the same transcript state at the same
        // point in the protocol so the coords match.
        // Capture the reduced (pre-extension) length BEFORE the extend
        // loop: that is the fixed log_stacking-equivalent height of
        // this (per-group) commit, used below to gate the sub-stripe
        // Π(1-r) claim adjustment.
        let z_star_orig_len = z_star.len();
        // The batched open covers EVERY round, so the point must index all of
        // their stripes end to end — not just this round's area.  The stripe
        // total need not be a power of two (the verifier zero-pads the
        // concatenated list), hence the CEILING.
        let stack_dim_for_target = commit.log_stacking_height as usize;
        let total_stripes: usize =
            preceding_rounds.iter().map(|(_, a)| a >> stack_dim_for_target).sum::<usize>()
                + (commit.area >> stack_dim_for_target);
        let target_dim = stack_dim_for_target
            + total_stripes.max(1).next_power_of_two().trailing_zeros() as usize;
        let mut extended_z_star = z_star;
        while extended_z_star.len() < target_dim {
            let r: InnerChallenge = challenger.sample_algebra_element();
            extended_z_star.push(r);
        }

        // Sub-stripe commits (host analog of the in-circuit `claim_adj`,
        // recursive_stacked_pcs.rs): when the reduced point is SHORTER than
        // the commit's log_stacking_height, the FS-extension coords falling
        // in the STACK portion `[z_star_orig_len, stack_dim)` correspond to
        // the ZERO high-half padding of the stripe (the dense poly of
        // `z_star_orig_len` vars is zero-padded up to `2^stack_dim`).  By the
        // MLE zero-padding identity the committed stripe's eval at
        // `stack_point` carries a Π(1 - r_k) factor over those coords that the
        // reduced-point claim lacks, so the stacked reconstruction equals
        // Π(1 - r_k) · q_at_z.  Multiply the claim to match.  NO-OP when
        // z_star_orig_len >= stack_dim ⇒ byte-identical there.  Each group is
        // verified through this function, so this covers every group.
        let stack_dim = commit.log_stacking_height as usize;
        let mut q_at_z_adj = q_at_z;
        if z_star_orig_len < stack_dim {
            for r in &extended_z_star[z_star_orig_len..stack_dim.min(target_dim)] {
                q_at_z_adj *= InnerChallenge::ONE - *r;
            }
        }

        // Verify the BaseFold opening: claim is q_at_z (sub-stripe adjusted),
        // point is the extended z*.
        // The batched open covers EVERY round: the rounds whose commitments
        // the verifying key pins come first, then this round's own.
        let mut commitments: Vec<_> = preceding_rounds.iter().map(|(c, _)| c.clone()).collect();
        commitments.push(commit.original_commitment.clone());
        let mut areas: Vec<usize> = preceding_rounds.iter().map(|(_, a)| *a).collect();
        areas.push(commit.area);
        if let Some(wp) = whir_proof {
            let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
            let hash = crate::kb31_poseidon2::InnerHash::new(perm.clone());
            let compress = crate::kb31_poseidon2::InnerCompress::new(perm);
            let mmcs = crate::jagged_pcs::JaggedMmcs::new(hash, compress, 0);
            let cfg = crate::whir::jagged::core_whir_config(commit.log_stacking_height as usize);
            let res = crate::whir::jagged::verify_jagged_whir_rounds(
                mmcs,
                cfg,
                commit.log_stacking_height,
                &commitments,
                &areas,
                &extended_z_star,
                wp,
                q_at_z_adj,
                challenger,
            );
            if let Err(e) = &res {
                eprintln!("[whir verify] whir opening REJECTED: {:?}", e);
            }
            return res.is_ok();
        }
        let res = crate::jagged_pcs::verify_jagged_pcs_rounds(
            &commitments,
            &areas,
            commit.log_stacking_height,
            &extended_z_star,
            q_at_z_adj,
            basefold_proof,
            challenger,
        );
        if let Err(e) = &res {
            eprintln!("[basefold verify] basefold opening REJECTED: {:?}", e);
        }
        res.is_ok()
    }

    /// BaseFold-over-BN254 wrap port: build the ring-agnostic verifier
    /// inputs (chip_infos / r_row_per_chip / z_row) from the bundle's PackingMeta
    /// + per-chip column widths + the shared zerocheck eval point. Mirrors the
    /// host verifier's construction (shard_level/verifier.rs) so the outer-ring
    /// verify hook reuses the exact same logic. Names are debug-only (unused in
    /// the verify math), so placeholders suffice.
    pub fn build_jagged_verify_inputs(
        packing: &PackingMeta,
        chip_widths: &[usize],
        eval_point: &[InnerChallenge],
    ) -> (Vec<crate::jagged::JaggedChipInfo>, Vec<Vec<InnerChallenge>>, Vec<InnerChallenge>) {
        use crate::jagged::JaggedChipInfo;
        let column_counts = &packing.column_counts;
        // One entry per COLUMN GROUP the prover emitted — every round's chips
        // AND the stacking-padding columns between them.  `chip_widths` (the
        // machine's main chips) is only the legacy fallback for a bundle that
        // predates `column_counts`: taking its LENGTH as the group count reads
        // a two-round packing as if the preprocessed round's widths were the
        // main chips', and stops before the main round entirely.
        let n_groups =
            if column_counts.is_empty() { chip_widths.len() } else { column_counts.len() };
        let mut chip_infos: Vec<JaggedChipInfo> = (0..n_groups)
            .map(|i| JaggedChipInfo {
                name: alloc::format!("chip{i}"),
                row_count: 0,
                column_count: column_counts
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| chip_widths.get(i).copied().unwrap_or(0)),
            })
            .collect();
        // Patch row_count from the offsets sentinel walk (same as the host verifier).
        {
            let mut col_idx = 0usize;
            for info in chip_infos.iter_mut() {
                if info.column_count == 0 {
                    continue;
                }
                let h = if col_idx + 1 < packing.offsets.len() {
                    packing.offsets[col_idx + 1].saturating_sub(packing.offsets[col_idx])
                } else if col_idx < packing.offsets.len() {
                    packing.total_values.saturating_sub(packing.offsets[col_idx])
                } else {
                    0
                };
                info.row_count = h;
                col_idx += info.column_count;
            }
        }
        let r_row_per_chip: Vec<Vec<InnerChallenge>> = chip_infos
            .iter()
            .map(|info| {
                let log_h = info.row_count.max(1).next_power_of_two().trailing_zeros() as usize;
                if eval_point.len() >= log_h {
                    eval_point[eval_point.len() - log_h..].to_vec()
                } else {
                    eval_point.to_vec()
                }
            })
            .collect();
        let z_row = eval_point.to_vec();
        (chip_infos, r_row_per_chip, z_row)
    }

    /// BaseFold-over-BN254 wrap port: verifier mirror of
    /// `prove_jagged_basefold_rounds_generic`, generic over the challenger + MMCS.
    /// The OUTER (wrap) ring drives this with OuterChallenger + OuterValMmcs via
    /// the registered verify hook; the inner ring keeps the concrete
    /// `verify_jagged_basefold_inner`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_jagged_basefold_inner_generic<Challenger, MT>(
        chip_infos: &[JaggedChipInfo],
        r_row_per_chip: &[Vec<InnerChallenge>],
        z_row: &[InnerChallenge],
        bundle: &JaggedBasefoldBundleGeneric<MT>,
        challenger: &mut Challenger,
        mmcs: MT,
        skip_commit_observe: bool,
        fri: FriConfig<crate::jagged_pcs::JaggedVal>,
        // Rounds committed BEFORE this one, as (commitment, area) — the
        // preprocessed round, whose commitment the verifying key holds.  Empty
        // for a machine with no preprocessed traces.
        preceding_rounds: &[(
            <MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment,
            usize,
        )],
    ) -> bool
    where
        MT: p3_commit::Mmcs<crate::jagged_pcs::JaggedVal, Commitment: Clone> + Clone,
        Challenger: p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + CanObserve<<MT as p3_commit::Mmcs<crate::jagged_pcs::JaggedVal>>::Commitment>,
    {
        // One jagged GROUP (the round split is inside it, as `preceding_rounds`
        // + this bundle's own commit).  The coverage check (group-map vs
        // partition) is enforced on the INNER host verifier; the wrap bundle
        // always carries the identity cover (empty `groups` / `extra_*`).
        debug_assert_eq!(bundle.num_groups(), 1, "wrap verify expects a single-GROUP bundle",);
        if !skip_commit_observe {
            challenger.observe(bundle.commit.original_commitment.clone());
        }
        let packing = JaggedPacking {
            dense_values: Vec::new(),
            chip_infos: chip_infos.to_vec(),
            offsets: bundle.packing.offsets.clone(),
            total_values: bundle.packing.total_values,
            dense_len: bundle.packing.total_values,
        };
        let num_cols = packing.offsets.len().saturating_sub(1);
        let num_col_vars = num_cols.next_power_of_two().trailing_zeros() as usize;
        let z_col: Vec<InnerChallenge> =
            (0..num_col_vars).map(|_| challenger.sample_algebra_element()).collect();
        let red_result = crate::jagged_sumcheck::verify_jagged_reduction(
            &bundle.reduction,
            &packing,
            r_row_per_chip,
            &bundle.y_per_chip,
            &z_col,
            z_row,
            challenger,
        );
        let Some((z_star, q_at_z, _w_at_z)) = red_result else {
            eprintln!("[basefold verify outer] jagged sumcheck reduction REJECTED");
            return false;
        };
        crate::jagged_eval_sumcheck::replay_jagged_evaluation_transcript(
            &bundle.jagged_eval,
            challenger,
        );
        // Capture the reduced (pre-extension) length BEFORE the extend loop
        // (see the inner verifier for the rationale).
        let z_star_orig_len = z_star.len();
        // The batched open indexes EVERY round's stripes end to end, so the
        // point spans the stack coords plus enough batch coords for the total
        // stripe count (ceiling — the verifier zero-pads the tail).
        let stack_dim_for_target = bundle.commit.log_stacking_height as usize;
        let total_stripes: usize =
            preceding_rounds.iter().map(|(_, a)| a >> stack_dim_for_target).sum::<usize>()
                + (bundle.commit.area >> stack_dim_for_target);
        let target_dim = stack_dim_for_target
            + total_stripes.max(1).next_power_of_two().trailing_zeros() as usize;
        let mut extended_z_star = z_star;
        while extended_z_star.len() < target_dim {
            let r: InnerChallenge = challenger.sample_algebra_element();
            extended_z_star.push(r);
        }
        // Sub-stripe commits: host analog of the in-circuit
        // `claim_adj` (recursive_stacked_pcs.rs).  Multiply q_at_z by the
        // Π(1-r) factor over the stack-portion extension coords when the
        // reduced point is shorter than log_stacking_height.  NO-OP (byte
        // identical) when z_star_orig_len >= stack_dim.
        let stack_dim = bundle.commit.log_stacking_height as usize;
        let mut q_at_z_adj = q_at_z;
        if z_star_orig_len < stack_dim {
            for r in &extended_z_star[z_star_orig_len..stack_dim.min(target_dim)] {
                q_at_z_adj *= InnerChallenge::ONE - *r;
            }
        }
        // The batched open covers EVERY round: the rounds the verifying key
        // pins come first, then this bundle's own.
        let mut commitments: Vec<_> = preceding_rounds.iter().map(|(c, _)| c.clone()).collect();
        commitments.push(bundle.commit.original_commitment.clone());
        let mut areas: Vec<usize> = preceding_rounds.iter().map(|(_, a)| *a).collect();
        areas.push(bundle.commit.area);
        let res = crate::jagged_pcs::verify_jagged_pcs_rounds_generic::<Challenger, MT>(
            &commitments,
            &areas,
            bundle.commit.log_stacking_height,
            &extended_z_star,
            q_at_z_adj,
            &bundle.basefold_proof,
            challenger,
            mmcs,
            fri,
        );
        if let Err(e) = &res {
            eprintln!("[basefold verify outer] basefold opening REJECTED: {:?}", e);
        }
        res.is_ok()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use p3_field::BasedVectorSpace;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_kb<R: Rng>(rng: &mut R) -> JaggedVal {
        JaggedVal::from_u32(rng.gen::<u32>() & 0x3FFF_FFFF)
    }

    fn rand_ef<R: Rng>(rng: &mut R) -> JaggedChallenge {
        <JaggedChallenge as BasedVectorSpace<JaggedVal>>::from_basis_coefficients_iter(
            (0..4).map(|_| rand_kb(rng)),
        )
        .unwrap()
    }

    fn build_challenger() -> JaggedChallenger {
        let perm: crate::kb31_poseidon2::InnerPerm = zkm_primitives::poseidon2_init();
        JaggedChallenger::new(perm)
    }

    /// End-to-end: commit a small batch of heterogeneous chip traces,
    /// open at a random point, verify.  This is the OOM-cure flow
    /// (per-chip Mles → stacked PCS → BaseFold) on a toy size.
    #[test]
    fn test_jagged_pcs_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xBA5E_F01D_5EED);

        // Two synthetic chip traces of different shapes, committed the way
        // PRODUCTION commits them: as ONE width-1 jagged dense
        // (`materialize_dense_jagged` over `committed_dense_len` cells —
        // `BasefoldRing::commit_multilinears`'s call shape).  Committing the
        // raw per-chip matrices instead would round EACH chip's height up to
        // whole 2^21 stacking blocks (28 stripes for this toy, vs the dense's
        // single stripe).
        let mk_trace = |width: usize, h: usize, rng: &mut StdRng| -> RowMajorMatrix<JaggedVal> {
            let v: Vec<JaggedVal> = (0..width * h).map(|_| rand_kb(rng)).collect();
            RowMajorMatrix::new(v, width)
        };
        let traces = vec![
            ("Cpu".into(), mk_trace(20, 100, &mut rng)),
            ("Add".into(), mk_trace(8, 50, &mut rng)),
        ];
        let views = as_chip_views(&traces);
        let packing = crate::jagged::compute_jagged_metadata::<JaggedVal>(&views);
        let dense =
            crate::jagged::materialize_dense_jagged::<JaggedVal>(&views, packing.dense_len, false);
        let dense_traces = vec![("<jagged-dense>".to_string(), RowMajorMatrix::new(dense, 1))];

        let mut p_chal = build_challenger();
        let (commit, prover_data) = commit_jagged_pcs(dense_traces);
        // The caller owns the transcript write.
        p_chal.observe(commit.original_commitment.clone());

        // Compute the eval point + claim for the stacked PCS.  Claim
        // is the multilinear-extension of the *flattened*
        // batch-evaluations vector at the batch part of the point.
        let stack_dim = commit.log_stacking_height as usize;
        let num_stripes = commit.area >> stack_dim;
        let num_batch_vars = num_stripes.next_power_of_two().trailing_zeros() as usize;
        let total_vars = num_batch_vars + stack_dim;
        let eval_point: Vec<JaggedChallenge> = (0..total_vars).map(|_| rand_ef(&mut rng)).collect();

        let stack_point: Vec<JaggedChallenge> = eval_point[..stack_dim].to_vec();
        let batch_evals_flat: Vec<JaggedChallenge> = prover_data
            .stacked_data
            .interleaved_mles
            .iter()
            .flat_map(|m| m.eval_at::<JaggedChallenge>(&stack_point))
            .collect();

        // Honest evaluation_claim = MLE of batch_evals_flat at
        // batch_point.  The verifier's `eval_multilinear_padded`
        // (basefold/stacked.rs) walks the point coords FORWARD
        // (LSB-first, `point[0]` binds var 0), so this hand-rolled fold
        // must too.
        let batch_point = &eval_point[stack_dim..];
        let evaluation_claim = {
            let target = 1usize << batch_point.len();
            let mut current: Vec<JaggedChallenge> = batch_evals_flat.clone();
            current.resize(target, JaggedChallenge::ZERO);
            for &r in batch_point.iter() {
                let half = current.len() / 2;
                for i in 0..half {
                    let lo = current[2 * i];
                    let hi = current[2 * i + 1];
                    current[i] = lo + r * (hi - lo);
                }
                current.truncate(half);
            }
            current[0]
        };

        let proof = open_jagged_pcs(&prover_data, eval_point.clone(), &mut p_chal);

        let mut v_chal = build_challenger();
        v_chal.observe(commit.original_commitment.clone());
        verify_jagged_pcs(
            &commit.original_commitment,
            commit.area,
            commit.log_stacking_height,
            &eval_point,
            evaluation_claim,
            &proof,
            &mut v_chal,
        )
        .expect("basefold jagged-PCS roundtrip");
    }

    // ════════════════════════════════════════════════════════════════
    // Jagged-BaseFold bundle tests — driven through the PRODUCTION
    // pipeline pair: `prove_jagged_basefold_rounds` (here with the
    // shard's single MAIN round) and `verify_jagged_basefold_no_observe`,
    // with the verifier inputs rebuilt by `build_jagged_verify_inputs`
    // exactly as the shard verifier rebuilds them
    // (shard_level/verifier.rs) — chip_infos carrying the EXPLICIT
    // stacking-padding columns.  The commit observe is the caller's on
    // both sides (the shard-level Phase 1 prologue analog).
    // ════════════════════════════════════════════════════════════════

    use crate::jagged_pcs::jagged::{
        build_jagged_verify_inputs, prove_jagged_basefold_rounds,
        verify_jagged_basefold_no_observe, ChipTraceView, JaggedBasefoldBundle, JaggedOpenRound,
    };
    use crate::kb31_poseidon2::koala_bear_poseidon2::KoalaBearPoseidon2;

    /// The commit/open entry points take BORROWED
    /// `ChipTraceView`s over the shard prover's shared `Arc<Mle>` store.
    /// Tests own their matrices, so relabel each owned matrix as a zero-copy
    /// view over its own cells — same cells, same width.
    fn as_chip_views(traces: &[(String, RowMajorMatrix<JaggedVal>)]) -> Vec<ChipTraceView> {
        traces
            .iter()
            .map(|(name, t)| {
                (name.clone(), {
                    let h = if t.width == 0 { 0 } else { t.values.len() / t.width };
                    let log_h = if h <= 1 { 0 } else { h.next_power_of_two().ilog2() };
                    crate::multilinear::PaddedMle::padded_with_zeros(
                        std::sync::Arc::new(crate::basefold::Mle::from_row_major(
                            p3_matrix::dense::RowMajorMatrix::new(t.values.clone(), t.width),
                        )),
                        log_h,
                    )
                })
            })
            .collect()
    }

    /// Build (traces, z_row) for a set of `(width, height)` chip shapes with
    /// deterministic random cells.  `z_row` has the production shape: ONE
    /// shared eval point of `DEFAULT_LOG_STACKING_HEIGHT` coords (the
    /// zerocheck z* the shard opens at), from which each chip's `r_row` is a
    /// trailing slice — see `r_row_suffixes`.
    fn mk_shard(
        shapes: &[(usize, usize)],
        seed: u64,
    ) -> (Vec<(String, RowMajorMatrix<JaggedVal>)>, Vec<JaggedChallenge>) {
        let mut rng = StdRng::seed_from_u64(seed);
        // Name-sorted so the partition's name-sorted-order precondition holds.
        let traces: Vec<(String, RowMajorMatrix<JaggedVal>)> = shapes
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| {
                let v: Vec<JaggedVal> = (0..w * h).map(|_| rand_kb(&mut rng)).collect();
                (alloc::format!("chip{i:03}"), RowMajorMatrix::new(v, w))
            })
            .collect();
        let z_row: Vec<JaggedChallenge> =
            (0..DEFAULT_LOG_STACKING_HEIGHT as usize).map(|_| rand_ef(&mut rng)).collect();
        (traces, z_row)
    }

    /// Per-chip `r_row` = the trailing `log2(padded height)` coords of the
    /// shared eval point — the production slice (shard_level/prover.rs).
    fn r_row_suffixes(
        views: &[ChipTraceView],
        z_row: &[JaggedChallenge],
    ) -> Vec<Vec<JaggedChallenge>> {
        views
            .iter()
            .map(|(_, pm)| {
                let (tvals, w) = crate::jagged::real_cells(pm);
                let h = if w == 0 { 0 } else { tvals.len() / w };
                let log_h = h.max(1).next_power_of_two().trailing_zeros() as usize;
                z_row[z_row.len() - log_h..].to_vec()
            })
            .collect()
    }

    /// Honest per-chip per-column claims for the MAIN round: the step-3
    /// row-MLE evaluations the production prover reads off the zerocheck
    /// residual, recomputed here from the traces.  Legacy bitrev row
    /// orientation (`use_rev = false`), in lockstep with each test's
    /// `commit_multilinears(&views, false)` commit — the pairing
    /// the shard prover carries via `PrecomputedJaggedCommit.rev`.
    fn column_claims(
        views: &[ChipTraceView],
        z_row: &[JaggedChallenge],
    ) -> Vec<Vec<JaggedChallenge>> {
        // eq_c[r] = eq(z_row, r): built over reversed z_row to undo
        // eq_mle_table's LSB-first bitrev.  The FULL row_eq subsumes the
        // height factor for any row < 2^log_h_c (the high bits of such a
        // row are 0).
        let z_row_rev: Vec<JaggedChallenge> = z_row.iter().rev().copied().collect();
        let eq_c = crate::zerocheck_prover::eq_mle_table::<JaggedChallenge>(&z_row_rev);
        views
            .iter()
            .map(|(_, pm)| {
                let (tvals, w) = crate::jagged::real_cells(pm);
                let h = if w == 0 { 0 } else { tvals.len() / w };
                if w == 0 {
                    return Vec::new();
                }
                if h == 0 {
                    return vec![JaggedChallenge::ZERO; w];
                }
                let is_pow2 = h.is_power_of_two();
                let log_h = if is_pow2 { (h as u32).trailing_zeros() } else { 0 };
                (0..w)
                    .map(|col| {
                        (0..h).fold(JaggedChallenge::ZERO, |acc, row| {
                            // Legacy orientation: the commit lays rows out
                            // bit-reversed, so the claim reads them the same
                            // way.
                            let src = if is_pow2 {
                                ((row as u32).reverse_bits() >> (32 - log_h)) as usize
                            } else {
                                row
                            };
                            acc + eq_c[row] * JaggedChallenge::from(tvals[src * w + col])
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Verify a single-main-round bundle exactly as the production shard
    /// verifier does: rebuild the verifier inputs (chip_infos WITH the
    /// explicit stacking-padding columns, per-group r_row) with
    /// `build_jagged_verify_inputs`, observe the commit, then
    /// `verify_jagged_basefold_no_observe`.  `opened_main` threads the
    /// cross-bind openings; `None` disables the bind (a caller with no
    /// shard openings).
    fn verify_main_round(
        bundle: &JaggedBasefoldBundle,
        chip_widths: &[usize],
        z_row: &[JaggedChallenge],
        opened_main: Option<&[Vec<JaggedChallenge>]>,
    ) -> bool {
        let (chip_infos, r_row_per_chip, z_row_v) =
            build_jagged_verify_inputs(&bundle.packing, chip_widths, z_row);
        let mut v_chal = build_challenger();
        v_chal.observe(bundle.commit.original_commitment.clone());
        verify_jagged_basefold_no_observe(
            &chip_infos,
            &r_row_per_chip,
            &z_row_v,
            // Main-only fixture: no preceding (preprocessed) round.
            &[],
            0,
            bundle,
            opened_main,
            &mut v_chal,
        )
    }

    /// Full jagged-sumcheck pipeline backed by BaseFold: an honest
    /// single-main-round bundle from the production rounds prover must
    /// round-trip through the production verifier.
    #[test]
    fn test_jagged_basefold_roundtrip() {
        let (traces, z_row) = mk_shard(&[(4, 16), (2, 8)], 0xC0DE_BA5E);
        let views = as_chip_views(&traces);
        let mut p_chal = build_challenger();
        // Production rounds pipeline: precompute the main round's commit
        // (inner ring, legacy `use_rev = false`, no area pin), observe it
        // (the shard-level Phase 1 prologue observe), open the single MAIN
        // round.
        let precomputed =
            <KoalaBearPoseidon2 as crate::config::BasefoldRing>::commit_multilinears(&views, false);
        p_chal.observe(precomputed.commit.original_commitment.clone());
        let r_row = r_row_suffixes(&views, &z_row);
        let rounds = [JaggedOpenRound {
            chip_traces: &views,
            r_row_per_chip: &r_row,
            claims: column_claims(&views, &z_row),
            precomputed: &precomputed,
        }];
        let bundle = prove_jagged_basefold_rounds(&rounds, &z_row, &mut p_chal);
        let widths: Vec<usize> = traces.iter().map(|(_, t)| t.width).collect();
        assert!(
            verify_main_round(&bundle, &widths, &z_row, None),
            "jagged-basefold pipeline should accept honest proof"
        );
    }

    /// **Soundness sanity** — flipping any single field of the bundle
    /// must cause the verifier to reject.  Catches whole classes of
    /// "I forgot to observe X into the challenger" bugs that pass
    /// honest-prover tests but admit forgery.
    #[test]
    fn test_jagged_basefold_rejects_tampered_proof() {
        let (traces, z_row) = mk_shard(&[(4, 16)], 0xDEAD_BEEF);
        let views = as_chip_views(&traces);
        let mut p_chal = build_challenger();
        // Production rounds pipeline: precompute the main round's commit
        // (inner ring, legacy `use_rev = false`, no area pin), observe it
        // (the shard-level Phase 1 prologue observe), open the single MAIN
        // round.
        let precomputed =
            <KoalaBearPoseidon2 as crate::config::BasefoldRing>::commit_multilinears(&views, false);
        p_chal.observe(precomputed.commit.original_commitment.clone());
        let r_row = r_row_suffixes(&views, &z_row);
        let rounds = [JaggedOpenRound {
            chip_traces: &views,
            r_row_per_chip: &r_row,
            claims: column_claims(&views, &z_row),
            precomputed: &precomputed,
        }];
        let bundle = prove_jagged_basefold_rounds(&rounds, &z_row, &mut p_chal);
        let widths: Vec<usize> = traces.iter().map(|(_, t)| t.width).collect();

        // The honest bundle must verify — otherwise the rejections below
        // are vacuous (a verifier that rejects EVERYTHING passes them).
        assert!(
            verify_main_round(&bundle, &widths, &z_row, None),
            "the untampered bundle must verify"
        );

        // Tamper #1: corrupt the sumcheck final claim `q_at_z`.
        let mut tampered = bundle.clone();
        tampered.reduction.q_at_z = tampered.reduction.q_at_z + JaggedChallenge::ONE;
        assert!(
            !verify_main_round(&tampered, &widths, &z_row, None),
            "verifier must reject q_at_z tampering"
        );

        // Tamper #2: corrupt one of the per-chip y_{c,j} column claims.
        let mut tampered = bundle.clone();
        tampered.y_per_chip[0][0] = tampered.y_per_chip[0][0] + JaggedChallenge::ONE;
        assert!(
            !verify_main_round(&tampered, &widths, &z_row, None),
            "verifier must reject y_per_chip tampering"
        );

        // Tamper #3: corrupt the BaseFold final_poly in the proof.
        let mut tampered = bundle.clone();
        tampered.basefold_proof.basefold_proof.final_poly =
            tampered.basefold_proof.basefold_proof.final_poly + JaggedChallenge::ONE;
        assert!(
            !verify_main_round(&tampered, &widths, &z_row, None),
            "verifier must reject final_poly tampering"
        );
    }

    /// The recursion circuit binds the jagged claimed sum to the trace
    /// openings (recursive_jagged_pcs.rs:247); the host verifier mirrors
    /// that bind.  This test proves the bind is load-bearing on the
    /// production pipeline:
    ///   (1) honest openings (== the bundle's column claims, the padding
    ///       columns' zero claims included) verify;
    ///   (2) openings that DIVERGE from the bundle's `y_per_chip` are
    ///       REJECTED once threaded in (`Some`);
    ///   (3) the SAME divergent case is (wrongly) ACCEPTED with the bind
    ///       disabled (`None`) — the pre-fix gap the bind closes.
    #[test]
    fn crossbind_rejects_divergent_openings() {
        let (traces, z_row) = mk_shard(&[(4, 16), (2, 8)], 0x0121_0BAD);
        let views = as_chip_views(&traces);
        let mut p_chal = build_challenger();
        // Production rounds pipeline: precompute the main round's commit
        // (inner ring, legacy `use_rev = false`, no area pin), observe it
        // (the shard-level Phase 1 prologue observe), open the single MAIN
        // round.
        let precomputed =
            <KoalaBearPoseidon2 as crate::config::BasefoldRing>::commit_multilinears(&views, false);
        p_chal.observe(precomputed.commit.original_commitment.clone());
        let r_row = r_row_suffixes(&views, &z_row);
        let rounds = [JaggedOpenRound {
            chip_traces: &views,
            r_row_per_chip: &r_row,
            claims: column_claims(&views, &z_row),
            precomputed: &precomputed,
        }];
        let bundle = prove_jagged_basefold_rounds(&rounds, &z_row, &mut p_chal);
        let widths: Vec<usize> = traces.iter().map(|(_, t)| t.width).collect();

        // The honest per-chip `main.local` openings coincide with the
        // bundle's column claims — index-aligned with the verifier's
        // chip_infos, so the stacking-padding entries ride along with
        // their zero claims.
        let opened_ok: Vec<Vec<JaggedChallenge>> = bundle.y_per_chip.clone();

        // (1) honest openings + cross-bind ON → ACCEPT.
        assert!(
            verify_main_round(&bundle, &widths, &z_row, Some(&opened_ok)),
            "honest openings must verify"
        );

        // (2) DIVERGENT openings + cross-bind ON → REJECT.
        let mut opened_bad = opened_ok.clone();
        opened_bad[0][0] += JaggedChallenge::ONE; // tamper ONE column claim
        assert!(
            !verify_main_round(&bundle, &widths, &z_row, Some(&opened_bad)),
            "y_per_chip diverging from openings MUST be rejected by the cross-bind"
        );

        // (3) SAME divergent openings but bind OFF (None) → ACCEPT.
        //     Documents the pre-fix gap the cross-bind closes.
        assert!(
            verify_main_round(&bundle, &widths, &z_row, None),
            "pre-fix baseline: with no opened-values bind the divergent proof is \
             (wrongly) accepted"
        );
    }

    /// (i) NO-OP byte-identity: with grouping OFF (the default), a G==1
    /// bundle has scalar group-0 fields, EMPTY `extra_*` + `groups`, and an
    /// honest roundtrip verifies.  The empty `serde(default)` fields make the
    /// wire bytes byte-identical to the pre-split format (a freshly
    /// deserialized bundle is bit-for-bit equal AND still verifies).
    #[test]
    fn test_cp_a_g1_noop_byte_identity() {
        // grouping OFF (no test threshold set, env unset).
        let (traces, z_row) = mk_shard(&[(4, 16), (2, 8), (6, 4)], 0xA11CE);
        let views = as_chip_views(&traces);
        let mut p_chal = build_challenger();
        // Production rounds pipeline: precompute the main round's commit
        // (inner ring, legacy `use_rev = false`, no area pin), observe it
        // (the shard-level Phase 1 prologue observe), open the single MAIN
        // round.
        let precomputed =
            <KoalaBearPoseidon2 as crate::config::BasefoldRing>::commit_multilinears(&views, false);
        p_chal.observe(precomputed.commit.original_commitment.clone());
        let r_row = r_row_suffixes(&views, &z_row);
        let rounds = [JaggedOpenRound {
            chip_traces: &views,
            r_row_per_chip: &r_row,
            claims: column_claims(&views, &z_row),
            precomputed: &precomputed,
        }];
        let bundle = prove_jagged_basefold_rounds(&rounds, &z_row, &mut p_chal);

        // Single-group invariants: scalar fields populated, extras empty.
        assert_eq!(bundle.num_groups(), 1, "grouping OFF ⇒ G==1");
        assert!(bundle.groups.is_empty(), "G==1 ⇒ empty group map");
        assert!(bundle.extra_reduction.is_empty());
        assert!(bundle.extra_basefold_proof.is_empty());
        assert!(bundle.extra_commit.is_empty());
        assert!(bundle.extra_packing.is_empty());
        assert!(bundle.extra_jagged_eval.is_empty());

        // Wire-format round-trip is bit-identical.
        let bytes = bundle.to_bytes();
        let bundle2 = JaggedBasefoldBundle::from_bytes(&bytes).expect("deserialize G==1 bundle");
        let bytes2 = bundle2.to_bytes();
        assert_eq!(bytes, bytes2, "G==1 bundle bytes must round-trip identically");

        // Honest verify: the identity cover passes coverage and the whole
        // pipeline accepts.
        let widths: Vec<usize> = traces.iter().map(|(_, t)| t.width).collect();
        assert!(verify_main_round(&bundle, &widths, &z_row, None), "G==1 bundle must verify");
        // The deserialized copy behaves identically (legacy-shape
        // equivalence).
        assert!(
            verify_main_round(&bundle2, &widths, &z_row, None),
            "deserialized G==1 bundle must verify identically"
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // G-host: LOCK THE HASH-BIND CONVENTION (jagged geometry
    // count ↔ commitment tie) with a host-only commit → verify round-trip,
    // BEFORE any circuit consumes it.  A wrong order / missing len-prefix
    // would silently desync Fiat-Shamir; this test prints the host hash and
    // asserts modified == recomputed host-side.
    // ───────────────────────────────────────────────────────────────────
    #[test]
    fn g_host_hash_bind_roundtrip() {
        use crate::jagged_pcs::jagged::PackingMeta;
        // A heterogeneous per-chip geometry (varied heights == a FIX-off
        // natural-commit shard), with the SENTINEL offset (len = total_cols+1).
        // chip heights:   3,      5,           2,        (a 0-col chip)
        // chip widths:    2,      1,           3,        0
        let column_counts: Vec<usize> = vec![2, 1, 3, 0];
        // offsets: column-major prefix sums. col widths sum = 6 columns.
        //  chip0 cols 0,1 (h=3) -> 0,3 ; chip1 col2 (h=5) -> 6 ;
        //  chip2 cols 3,4,5 (h=2) -> 11,13,15 ; sentinel 17
        let offsets: Vec<usize> = vec![0, 3, 6, 11, 13, 15, 17];
        let total_values = 17usize;
        let packing = PackingMeta {
            offsets,
            total_values,
            log_dense_size: (total_values.next_power_of_two()).trailing_zeros() as usize,
            column_counts: column_counts.clone(),
            // Synthetic single-round packing.
            round_counts: Vec::new(),
            padding_heights: Vec::new(),
        };

        // The derived per-chip (row_counts, column_counts) — the EXACT felt
        // sequence both host and circuit hash.
        let (row_counts, col_counts) = jagged_counts_from_packing(&packing);
        assert_eq!(col_counts, column_counts);
        // chip0 h = offsets[1]-offsets[0] = 3; chip1 h = offsets[3]-offsets[2] = 5;
        // chip2 h = offsets[5]-offsets[4]... col_idx walk: chip0 col_idx=0 -> 3;
        //   chip1 col_idx=2 -> offsets[3]-offsets[2]=5; chip2 col_idx=3 ->
        //   offsets[4]-offsets[3]=2; chip3 cc==0 -> 0.
        assert_eq!(row_counts, vec![3usize, 5, 2, 0], "row_counts derivation");

        // A toy raw root.
        let raw_root: [JaggedVal; 8] =
            core::array::from_fn(|i| JaggedVal::from_u32((i as u32 + 1) * 7));

        let hash = jagged_geometry_hash(&row_counts, &col_counts);
        let modified = jagged_hash_bind_modified(raw_root, &row_counts, &col_counts);
        let modified_from_packing = jagged_hash_bind_from_packing(raw_root, &packing);

        eprintln!("[G-HOST] raw_root      = {raw_root:?}");
        eprintln!("[G-HOST] geometry_hash = {hash:?}");
        eprintln!("[G-HOST] modified      = {modified:?}");
        eprintln!(
            "[G-HOST] len(=col_counts.len())={} row_counts={row_counts:?} col_counts={col_counts:?}",
            col_counts.len()
        );

        // Convention self-consistency: the packing one-liner equals the
        // explicit path.
        assert_eq!(modified, modified_from_packing, "from_packing must match explicit");

        // The host re-bind check (mirror of the in-circuit re-bind)
        // must ACCEPT the honest modified digest.
        assert!(
            jagged_hash_bind_verify(raw_root, modified, &packing),
            "G-host: host re-bind must accept the honest modified digest"
        );

        // FORGERY-SHAPED negative #1: a TAMPERED row_count must change the
        // hash -> modified, so the re-bind REJECTS it (the count↔commitment
        // tie).  (This is the host-side analog of G3b.)
        {
            let mut bad = packing.clone();
            // inflate chip0 height: offsets[1] 3 -> 4 (shifts everything)
            bad.offsets[1] = 4;
            assert!(
                !jagged_hash_bind_verify(raw_root, modified, &bad),
                "G-host: a tampered row_count MUST fail the re-bind"
            );
        }
        // FORGERY-SHAPED negative #2: a TAMPERED column_count must reject.
        {
            let mut bad = packing.clone();
            bad.column_counts[0] = 3; // was 2
            assert!(
                !jagged_hash_bind_verify(raw_root, modified, &bad),
                "G-host: a tampered column_count MUST fail the re-bind"
            );
        }
        // LEN-PREFIX guard: omitting/altering the len prefix would silently
        // collide honest geometries of different lengths.  Verify the len is
        // genuinely mixed in: a geometry with one MORE (zero-height, zero-col)
        // chip — same row/col VALUES extended by a 0 — hashes DIFFERENTLY
        // because the len prefix changes.
        {
            let mut rc2 = row_counts.clone();
            let mut cc2 = col_counts.clone();
            rc2.push(0);
            cc2.push(0);
            let h2 = jagged_geometry_hash(&rc2, &cc2);
            assert_ne!(
                hash, h2,
                "G-host: the len prefix MUST distinguish different-length geometries"
            );
        }
    }
}
