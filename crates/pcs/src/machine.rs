use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{get_symbolic_constraints, AirLayout, SymbolicAirBuilder};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::{fmt::Debug, iter::once, time::Instant};
use tracing::instrument;

use super::{debug_constraints, Dom};
use crate::PROOF_MAX_NUM_PVS;
use crate::{
    air::{LookupScope, MachineAir, MachineProgram},
    config::PrepCommitRoot,
    count_permutation_constraints,
    lookup::{debug_lookups_with_all_chips, LookupKind},
    record::MachineRecord,
    septic_curve::SepticCurve,
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    DebugConstraintBuilder, ShardProof, VerifierConstraintFolder,
};

use super::{
    Chip, Com, MachineProof, PcsProverData, StarkGenericConfig, Val, VerificationError, Verifier,
};

/// A chip in a machine.
pub type MachineChip<SC, A> = Chip<Val<SC>, A>;

/// A STARK for proving MIPS execution.
pub struct StarkMachine<SC: StarkGenericConfig, A> {
    /// The STARK settings for the MIPS STARK.
    config: SC,
    /// The chips that make up the MIPS STARK machine, in order of their execution.
    chips: Vec<Chip<Val<SC>, A>>,

    /// The number of public values elements that the machine uses
    num_pv_elts: usize,

    /// Whether this machine's shard proofs use the rev(zeta) CORE
    /// orientation (the collapsed-claim convention).  `true` ONLY
    /// for the CORE (MIPS) machine — its FIX-off/FIX-on prove path installs the
    /// `Some(true)` orientation carrier, so its shard proofs are rev.  `false`
    /// (the default) for every recursion / shrink / wrap machine — those proofs
    /// are LEGACY (the recursion prover never installs the carrier).  Threaded
    /// to the host `verify_zerocheck_host` / `recompute_zerocheck_rlc_eval_host`
    /// so a core proof is host-verified rev and a recursion/wrap proof legacy.
    core_rev: bool,
}

impl<SC: StarkGenericConfig, A> StarkMachine<SC, A> {
    /// Creates a new [`StarkMachine`] whose shard proofs use the LEGACY zerocheck
    /// orientation (every recursion / shrink / wrap machine, and test machines).
    pub const fn new(config: SC, chips: Vec<Chip<Val<SC>, A>>, num_pv_elts: usize) -> Self {
        Self { config, chips, num_pv_elts, core_rev: true }
    }

    /// Creates a CORE [`StarkMachine`] whose shard proofs use the
    /// rev(zeta) orientation (host verify picks the collapsed / no-embed claim).
    /// Used ONLY by the MIPS core machine.
    pub const fn new_core_rev(
        config: SC,
        chips: Vec<Chip<Val<SC>, A>>,
        num_pv_elts: usize,
    ) -> Self {
        Self { config, chips, num_pv_elts, core_rev: true }
    }

    /// Whether this machine's shard proofs use the rev(zeta) CORE
    /// orientation.
    #[inline]
    pub const fn core_rev(&self) -> bool {
        self.core_rev
    }
}

/// A proving key for a STARK.
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "PcsProverData<SC>: Serialize"))]
#[serde(bound(deserialize = "PcsProverData<SC>: for<'a> Deserialize<'a>"))]
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    /// The preprocessed traces, row-major.  These are the AIR-side form:
    /// serialized with the key and handed to the constraint folder / debug
    /// paths.
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    /// The same preprocessed traces in the PROVE-path form: one `Arc<Mle>`
    /// per entry of `traces`, built once on first use and shared by every
    /// shard.  The shard cube differs per stage,
    /// so the cube-dependent `PaddedMle` wrapper is applied per shard —
    /// that wrap is an `Arc` bump because `PaddedMle` padding is virtual.
    ///
    /// Not serialized: it is derived from `traces`, and rebuilding it is a
    /// one-time cost per key.  A deserialized key simply rebuilds on demand.
    #[serde(skip)]
    preprocessed_mles: std::sync::OnceLock<Vec<std::sync::Arc<crate::basefold::Mle<Val<SC>>>>>,
    /// The PRECOMPUTED preprocessed commit — the commitment plus the BaseFold
    /// prover data and jagged packing needed to OPEN the preprocessed traces,
    /// not merely observe them.  `self.commit` is the root of exactly this
    /// commit: `setup` puts the commit in the verifying key and this data in
    /// the proving key, and the preprocessed traces are opened as their own
    /// ROUND of every shard proof.
    ///
    /// Not serialized: it is a deterministic function of `traces`, so a
    /// deserialized key rebuilds it on first use — same contract as
    /// `preprocessed_mles`.
    #[serde(skip)]
    preprocessed_data: std::sync::OnceLock<std::sync::Arc<SC::PrepPrecomputed>>,
    /// The row orientation the PREPROCESSED commit was built under at `setup`
    /// (`StarkMachine::core_rev`).
    ///
    /// It has to travel WITH the key.  The preprocessed round is opened against
    /// `vk.commit`, so the prover must rebuild the commit under the exact
    /// orientation `setup` used — and a caller-supplied flag is the wrong
    /// source: a key that has been round-tripped through `pk_to_host` reaches a
    /// prover whose machine may report a different `core_rev`, and the only
    /// symptom is a Merkle `CapMismatch` on round 0, far from the cause.
    #[serde(default)]
    pub prep_rev: bool,
    /// The preprocessed chip ordering.
    pub chip_ordering: HashMap<String, usize>,
    /// The preprocessed chip local only information.
    /// The number of total constraints for each chip.
    pub constraints_map: HashMap<String, usize>,
}

impl<SC: StarkGenericConfig> Clone for StarkProvingKey<SC> {
    fn clone(&self) -> Self {
        Self {
            commit: self.commit.clone(),
            pc_start: self.pc_start,
            initial_global_cumulative_sum: self.initial_global_cumulative_sum,
            traces: self.traces.clone(),
            // Derived cache: the clone rebuilds it on first use rather than
            // deep-copying the MLEs.
            preprocessed_mles: std::sync::OnceLock::new(),
            preprocessed_data: std::sync::OnceLock::new(),
            prep_rev: self.prep_rev,
            chip_ordering: self.chip_ordering.clone(),
            constraints_map: self.constraints_map.clone(),
        }
    }
}

impl<SC: StarkGenericConfig> StarkProvingKey<SC> {
    /// Build a proving key from its parts.  The prove-path MLE cache is
    /// derived from `traces`, so it is never supplied by the caller — that
    /// keeps the two representations from drifting apart.
    pub fn from_parts(
        commit: Com<SC>,
        pc_start: Val<SC>,
        initial_global_cumulative_sum: SepticDigest<Val<SC>>,
        traces: Vec<RowMajorMatrix<Val<SC>>>,
        chip_ordering: HashMap<String, usize>,
        constraints_map: HashMap<String, usize>,
        // The orientation `setup` committed the preprocessed traces under.
        prep_rev: bool,
    ) -> Self {
        Self {
            commit,
            pc_start,
            initial_global_cumulative_sum,
            traces,
            preprocessed_mles: std::sync::OnceLock::new(),
            preprocessed_data: std::sync::OnceLock::new(),
            prep_rev,
            chip_ordering,
            constraints_map,
        }
    }

    /// Build a proving key whose preprocessed commit has already been computed.
    ///
    /// The commit is the expensive half of `setup`, and a caller that just
    /// built it has no reason to make [`Self::preprocessed_data`] build it a
    /// second time.  The supplied data MUST be the precompute of exactly these
    /// `traces` under exactly this `prep_rev` — it is what the preprocessed
    /// round of every shard proof opens against `commit`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts_with_preprocessed_data(
        commit: Com<SC>,
        pc_start: Val<SC>,
        initial_global_cumulative_sum: SepticDigest<Val<SC>>,
        traces: Vec<RowMajorMatrix<Val<SC>>>,
        chip_ordering: HashMap<String, usize>,
        constraints_map: HashMap<String, usize>,
        prep_rev: bool,
        preprocessed_data: std::sync::Arc<SC::PrepPrecomputed>,
    ) -> Self {
        let key = Self::from_parts(
            commit,
            pc_start,
            initial_global_cumulative_sum,
            traces,
            chip_ordering,
            constraints_map,
            prep_rev,
        );
        let _ = key.preprocessed_data.set(preprocessed_data);
        key
    }

    /// The preprocessed commit's prover data if it has already been built, and
    /// `None` if it has not — a peek that does NOT trigger the build.
    ///
    /// For carrying an already-paid commit onward (host key -> device key ->
    /// host key) without forcing one on a path that never opens the
    /// preprocessed round.
    pub fn preprocessed_data_if_built(&self) -> Option<&std::sync::Arc<SC::PrepPrecomputed>> {
        self.preprocessed_data.get()
    }

    /// The preprocessed traces in prove-path form, built once per key.
    ///
    /// Each entry is the zero-copy `Mle` view of the corresponding
    /// `self.traces[i]` (`Mle::from_row_major` moves the backing buffer and
    /// preserves row-major order, so `Mle::as_trace_ref()` round-trips the
    /// original cells byte-for-byte).
    pub fn preprocessed_mles(&self) -> &[std::sync::Arc<crate::basefold::Mle<Val<SC>>>] {
        self.preprocessed_mles.get_or_init(|| {
            // PARALLEL over chips: each entry deep-copies one preprocessed
            // trace (`from_row_major` takes ownership, so the clone is
            // unavoidable while the key also keeps `traces`), and the walk is
            // pure per-chip memory traffic with no shared state.  Serially
            // this was the largest un-attributed block of `setup` on a
            // pk-cache miss — ~5 s across a combined reth's 67 misses.
            use p3_maybe_rayon::prelude::*;
            self.traces
                .par_iter()
                .map(|t| std::sync::Arc::new(crate::basefold::Mle::from_row_major(t.clone())))
                .collect()
        })
    }

    /// The precomputed preprocessed commit, built on first use from `traces`
    /// in the SAME name/height order `setup` committed them in (the order
    /// `chip_ordering` records) and under the orientation recorded on the key
    /// (`prep_rev`), so the rebuilt commitment reproduces `self.commit`
    /// exactly.
    pub fn preprocessed_data(&self) -> &std::sync::Arc<SC::PrepPrecomputed> {
        let use_rev = self.prep_rev;
        self.preprocessed_data.get_or_init(|| {
            let mut names: Vec<(usize, &String)> =
                self.chip_ordering.iter().map(|(n, i)| (*i, n)).collect();
            names.sort_unstable();
            let named: Vec<(String, RowMajorMatrix<Val<SC>>)> = names
                .into_iter()
                .zip(self.traces.iter())
                .map(|((_, name), trace)| (name.clone(), trace.clone()))
                .collect();
            std::sync::Arc::new(SC::prep_precompute(&named, use_rev))
        })
    }

    /// Observes the values of the proving key into the challenger.
    pub fn observe_into(&self, challenger: &mut SC::Challenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        // Observe the padding.
        challenger.observe(Val::<SC>::ZERO);
    }
}

/// Serializable representation of a domain (shift + log_size).
/// Used in `StarkVerifyingKey` since upstream `TwoAdicMultiplicativeCoset` no longer
/// implements serde.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize"))]
#[serde(bound(deserialize = "F: DeserializeOwned"))]
pub struct SerializableDomain<F: Field> {
    pub shift: F,
    pub log_size: usize,
}

impl<F: p3_field::TwoAdicField> SerializableDomain<F> {
    /// Create from a concrete `TwoAdicMultiplicativeCoset`.
    pub fn from_coset(d: &p3_field::coset::TwoAdicMultiplicativeCoset<F>) -> Self {
        Self { shift: d.shift(), log_size: d.log_size() }
    }

    /// Reconstruct the concrete domain.
    pub fn to_coset(&self) -> p3_field::coset::TwoAdicMultiplicativeCoset<F> {
        p3_field::coset::TwoAdicMultiplicativeCoset::new(self.shift, self.log_size)
            .expect("invalid domain parameters")
    }
}

impl<F: Field> SerializableDomain<F> {
    /// Create from a PolynomialSpace-like domain (uses size to derive log_size).
    pub fn new(shift: F, log_size: usize) -> Self {
        Self { shift, log_size }
    }
}

/// A verifying key for a STARK.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = ""))]
#[serde(bound(deserialize = ""))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    /// The chip information.
    pub chip_information: Vec<(String, SerializableDomain<Val<SC>>, (usize, usize))>,
    /// The chip ordering.
    pub chip_ordering: HashMap<String, usize>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Dom<SC>: Serialize"))]
#[serde(bound(deserialize = "Dom<SC>: DeserializeOwned"))]
pub struct PartStarkVerifyingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
}

impl<SC: StarkGenericConfig> StarkVerifyingKey<SC> {
    /// Observes the values of the verifying key into the challenger.
    pub fn observe_into(&self, challenger: &mut SC::Challenger) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        // Observe the padding.
        challenger.observe(Val::<SC>::ZERO);
    }

    pub fn part_vk(&self) -> PartStarkVerifyingKey<SC> {
        PartStarkVerifyingKey { commit: self.commit.clone(), pc_start: self.pc_start }
    }
}

impl<SC: StarkGenericConfig> Debug for StarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyingKey").finish()
    }
}

impl<SC: StarkGenericConfig> Debug for PartStarkVerifyingKey<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartVerifyingKey").finish()
    }
}

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>>> StarkMachine<SC, A> {
    /// Get an array containing a `ChipRef` for all the chips of this MIPS STARK machine.
    pub fn chips(&self) -> &[MachineChip<SC, A>] {
        &self.chips
    }

    /// Returns the number of public values elements.
    pub const fn num_pv_elts(&self) -> usize {
        self.num_pv_elts
    }

    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips<'a, 'b>(
        &'a self,
        shard: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips.iter().filter(|chip| chip.included(shard))
    }

    /// Returns an iterator over the chips in the machine that are included in the given shard.
    pub fn shard_chips_ordered<'a, 'b>(
        &'a self,
        chip_ordering: &'b HashMap<String, usize>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_ordering.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip_ordering.get(&chip.name()))
    }

    /// Returns the machine chips present in a shard, keyed by the proof's
    /// per-chip log-height map, in the shard's chip order.  Shard chip
    /// order IS name order (`commit()` sorts the shard's traces by chip
    /// name), so the name-sorted `BTreeMap` keys reproduce it exactly.
    pub fn shard_chips_named<'a, 'b, V>(
        &'a self,
        chip_heights: &'b std::collections::BTreeMap<String, V>,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
    {
        self.chips
            .iter()
            .filter(|chip| chip_heights.contains_key(&chip.name()))
            .sorted_by_key(|chip| chip.name())
    }

    /// Returns the config of the machine.
    pub const fn config(&self) -> &SC {
        &self.config
    }

    /// Debugs the constraints of the given records.
    #[instrument("debug constraints", level = "debug", skip_all)]
    pub fn debug_constraints(
        &self,
        pk: &StarkProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        tracing::debug!("checking constraints for each shard");

        // Obtain the challenges used for the global permutation argument.
        let mut permutation_challenges: Vec<SC::Challenge> = Vec::new();
        for _ in 0..2 {
            permutation_challenges.push(challenger.sample_algebra_element());
        }

        // Obtain the challenges used for the local permutation argument.
        for _ in 0..2 {
            permutation_challenges.push(challenger.sample_algebra_element());
        }

        let mut global_cumulative_sums = Vec::new();
        global_cumulative_sums.push(pk.initial_global_cumulative_sum);

        for shard in records.iter() {
            // Filter the chips based on what is used.
            let chips = self.shard_chips(shard).collect::<Vec<_>>();

            // Generate the main trace for each chip.
            let pre_traces = chips
                .iter()
                .map(|chip| pk.chip_ordering.get(&chip.name()).map(|index| &pk.traces[*index]))
                .collect::<Vec<_>>();
            let mut traces = chips
                .par_iter()
                .map(|chip| chip.generate_trace(shard, &mut A::Record::default()).unwrap())
                .zip(pre_traces)
                .collect::<Vec<_>>();

            // Generate the permutation traces.
            let mut permutation_traces = Vec::with_capacity(chips.len());
            let mut chip_cumulative_sums = Vec::with_capacity(chips.len());
            tracing::debug_span!("generate permutation traces").in_scope(|| {
                chips
                    .par_iter()
                    .zip(traces.par_iter_mut())
                    .map(|(chip, (main_trace, pre_trace))| {
                        let (trace, local_sum) = chip.generate_permutation_trace(
                            *pre_trace,
                            main_trace,
                            &permutation_challenges,
                        );
                        let global_sum = if chip.commit_scope() == LookupScope::Local {
                            SepticDigest::<Val<SC>>::zero()
                        } else {
                            let main_trace_size = main_trace.height() * main_trace.width();
                            let last_row =
                                &main_trace.values[main_trace_size - 14..main_trace_size];
                            SepticDigest(SepticCurve {
                                x: SepticExtension::<Val<SC>>::from_basis_coefficients_fn(|i| {
                                    last_row[i]
                                }),
                                y: SepticExtension::<Val<SC>>::from_basis_coefficients_fn(|i| {
                                    last_row[i + 7]
                                }),
                            })
                        };
                        (trace, (global_sum, local_sum))
                    })
                    .unzip_into_vecs(&mut permutation_traces, &mut chip_cumulative_sums);
            });

            let global_cumulative_sum =
                chip_cumulative_sums.iter().map(|sums| sums.0).sum::<SepticDigest<Val<SC>>>();
            global_cumulative_sums.push(global_cumulative_sum);

            let local_cumulative_sum =
                chip_cumulative_sums.iter().map(|sums| sums.1).sum::<SC::Challenge>();

            if !local_cumulative_sum.is_zero() {
                tracing::warn!("Local cumulative sum is not zero");
                tracing::debug_span!("debug local lookups").in_scope(|| {
                    debug_lookups_with_all_chips::<SC, A>(
                        self,
                        pk,
                        std::slice::from_ref(shard),
                        LookupKind::all_kinds(),
                        LookupScope::Local,
                    )
                });
                panic!("Local cumulative sum is not zero");
            }

            // Compute some statistics.
            for i in 0..chips.len() {
                let trace_width = traces[i].0.width();
                let pre_width = traces[i].1.map_or(0, p3_matrix::Matrix::width);
                let permutation_width = permutation_traces[i].width()
                    * <SC::Challenge as BasedVectorSpace<SC::Val>>::DIMENSION;
                let total_width = trace_width + pre_width + permutation_width;
                tracing::debug!(
                    "{:<11} | Main Cols = {:<5} | Pre Cols = {:<5} | Perm Cols = {:<5} | Rows = {:<10} | Cells = {:<10}",
                    chips[i].name(),
                    trace_width,
                    pre_width,
                    permutation_width,
                    traces[i].0.height(),
                    total_width * traces[i].0.height(),
                );
            }

            tracing::info_span!("debug constraints").in_scope(|| {
                for i in 0..chips.len() {
                    let preprocessed_trace =
                        pk.chip_ordering.get(&chips[i].name()).map(|index| &pk.traces[*index]);
                    debug_constraints::<SC, A>(
                        chips[i],
                        preprocessed_trace,
                        &traces[i].0,
                        &permutation_traces[i],
                        &permutation_challenges,
                        &shard.public_values(),
                        &chip_cumulative_sums[i].1,
                        &chip_cumulative_sums[i].0,
                    );
                }
            });
        }

        tracing::info!("Constraints verified successfully");

        let global_cumulative_sum: SepticDigest<Val<SC>> =
            global_cumulative_sums.iter().copied().sum();

        // If the global cumulative sum is not zero, debug the lookups.
        if !global_cumulative_sum.is_zero() {
            tracing::warn!("Global cumulative sum is not zero");
            tracing::debug_span!("debug global lookups").in_scope(|| {
                debug_lookups_with_all_chips::<SC, A>(
                    self,
                    pk,
                    &records,
                    LookupKind::all_kinds(),
                    LookupScope::Global,
                )
            });
            tracing::warn!(
                "Global cumulative sum: {:?}, should be: {:?}",
                global_cumulative_sum,
                SepticDigest::<Val<SC>>::zero(),
            );
            panic!("Global cumulative sum is not zero");
        }
    }
}

impl<SC: StarkGenericConfig, A: MachineAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>>>>
    StarkMachine<SC, A>
{
    /// The setup preprocessing phase.
    ///
    /// Given a program, this function generates the proving and verifying keys. The keys correspond
    /// to the program code and other preprocessed columns such as lookup tables.
    #[instrument("setup machine", level = "debug", skip_all)]
    #[allow(clippy::map_unwrap_or)]
    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn setup(&self, program: &A::Program) -> (StarkProvingKey<SC>, StarkVerifyingKey<SC>) {
        let parent_span = tracing::debug_span!("generate preprocessed traces");
        let (named_preprocessed_traces, num_constraints): (Vec<_>, Vec<_>) =
            parent_span.in_scope(|| {
                self.chips()
                    .par_iter()
                    .map(|chip| {
                        let chip_name = chip.name();
                        let begin = Instant::now();
                        let prep_trace = chip.generate_preprocessed_trace(program);
                        tracing::debug!(
                            parent: &parent_span,
                            "generated preprocessed trace for chip {} in {:?}",
                            chip_name,
                            begin.elapsed()
                        );
                        // Assert that the chip width data is correct.
                        let expected_width = prep_trace.as_ref().map(|t| t.width()).unwrap_or(0);
                        assert_eq!(
                            expected_width,
                            chip.preprocessed_width(),
                            "Incorrect number of preprocessed columns for chip {chip_name}"
                        );

                        // Count the number of constraints.
                        let num_main_constraints = get_symbolic_constraints(
                            &chip.air,
                            AirLayout {
                                preprocessed_width: chip.preprocessed_width(),
                                main_width: chip.width(),
                                num_public_values: PROOF_MAX_NUM_PVS,
                                ..Default::default()
                            },
                        )
                        .len();

                        let num_permutation_constraints = count_permutation_constraints(
                            &chip.sends,
                            &chip.receives,
                            chip.logup_batch_size(),
                            chip.air.commit_scope(),
                        );

                        (
                            prep_trace.map(move |t| (chip.name(), t)),
                            (chip_name, num_main_constraints + num_permutation_constraints),
                        )
                    })
                    .unzip()
            });

        let mut named_preprocessed_traces =
            named_preprocessed_traces.into_iter().flatten().collect::<Vec<_>>();

        // Order the preprocessed chips BY NAME.
        //
        // The order is the order the round is committed in, and a verifier has
        // to know which committed column belongs to which chip.  Under a
        // height-first order that mapping can only come from the key
        // (`chip_information`); under NAME order any verifier reproduces it
        // from the machine's chip set alone, without key-carried chip
        // metadata.
        //
        // Nothing downstream wants the heights descending: the jagged packer
        // walks the list in the given order and accumulates offsets
        // (`compute_jagged_metadata_from_dims`), and `chip_ordering` indexes
        // `traces` by the same list either way.
        named_preprocessed_traces.sort_by(|a, b| a.0.cmp(&b.0));

        let pcs = self.config.pcs();
        // Only the serialisable domain description is kept -- it goes into the
        // verifying key's `chip_information`.
        let chip_information: Vec<_> = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| {
                let domain = pcs.natural_domain_for_degree(trace.height());
                let ser_domain = SerializableDomain::new(
                    domain.first_point(),
                    domain.size().trailing_zeros() as usize,
                );
                (name.to_owned(), ser_domain, (trace.width(), trace.height()))
            })
            .collect();

        // Commit to the preprocessed traces.  One path, always -- no height
        // threshold, no opt-in flag, no fallback.
        let named: Vec<(String, RowMajorMatrix<Val<SC>>)> = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| (name.to_string(), trace.clone()))
            .collect();
        // Setup produces the commitment AND the data needed to open it.
        // Keep both — the root goes
        // to the vk, the precompute is seeded into the proving key below, so the
        // opening round never has to re-derive the committed order.
        let prep_precomputed = SC::prep_precompute(&named, self.core_rev());
        let commit = prep_precomputed.commit_root();

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces =
            named_preprocessed_traces.into_iter().map(|(_, trace)| trace).collect::<Vec<_>>();

        let pc_start = program.pc_start();
        let initial_global_cumulative_sum = program.initial_global_cumulative_sum();

        (
            StarkProvingKey {
                commit: commit.clone(),
                pc_start,
                initial_global_cumulative_sum,
                traces,
                preprocessed_mles: std::sync::OnceLock::new(),
                // Seeded from the very precompute whose root became `commit`
                // above, so the opening round reuses the committed data
                // directly instead of rebuilding it from `traces`.
                preprocessed_data: {
                    let cell = std::sync::OnceLock::new();
                    let _ = cell.set(std::sync::Arc::new(prep_precomputed));
                    cell
                },
                prep_rev: self.core_rev(),
                chip_ordering: chip_ordering.clone(),
                constraints_map,
            },
            StarkVerifyingKey {
                commit,
                pc_start,
                initial_global_cumulative_sum,
                chip_information,
                chip_ordering,
            },
        )
    }

    /// The setup preprocessing phase. Same as `setup` but initial global cumulative sum is
    /// precomputed.
    pub fn setup_core(
        &self,
        program: &A::Program,
        initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    ) -> (StarkProvingKey<SC>, StarkVerifyingKey<SC>) {
        let parent_span = tracing::debug_span!("generate preprocessed traces");
        let (named_preprocessed_traces, num_constraints): (Vec<_>, Vec<_>) =
            parent_span.in_scope(|| {
                self.chips()
                    .par_iter()
                    .map(|chip| {
                        let chip_name = chip.name();
                        let begin = Instant::now();
                        let prep_trace = chip.generate_preprocessed_trace(program);
                        tracing::debug!(
                            parent: &parent_span,
                            "generated preprocessed trace for chip {} in {:?}",
                            chip_name,
                            begin.elapsed()
                        );
                        // Assert that the chip width data is correct.
                        let expected_width =
                            prep_trace.as_ref().map_or(0, p3_matrix::Matrix::width);
                        assert_eq!(
                            expected_width,
                            chip.preprocessed_width(),
                            "Incorrect number of preprocessed columns for chip {chip_name}"
                        );

                        // Count the number of constraints.
                        let num_main_constraints = get_symbolic_constraints(
                            &chip.air,
                            AirLayout {
                                preprocessed_width: chip.preprocessed_width(),
                                main_width: chip.width(),
                                num_public_values: PROOF_MAX_NUM_PVS,
                                ..Default::default()
                            },
                        )
                        .len();

                        let num_permutation_constraints = count_permutation_constraints(
                            &chip.sends,
                            &chip.receives,
                            chip.logup_batch_size(),
                            chip.air.commit_scope(),
                        );

                        (
                            prep_trace.map(move |t| (chip.name(), t)),
                            (chip_name, num_main_constraints + num_permutation_constraints),
                        )
                    })
                    .unzip()
            });

        let mut named_preprocessed_traces =
            named_preprocessed_traces.into_iter().flatten().collect::<Vec<_>>();

        // Order the preprocessed chips BY NAME.
        //
        // The order is the order the round is committed in, and a verifier has
        // to know which committed column belongs to which chip.  Under a
        // height-first order that mapping can only come from the key
        // (`chip_information`); under NAME order any verifier reproduces it
        // from the machine's chip set alone, without key-carried chip
        // metadata.
        //
        // Nothing downstream wants the heights descending: the jagged packer
        // walks the list in the given order and accumulates offsets
        // (`compute_jagged_metadata_from_dims`), and `chip_ordering` indexes
        // `traces` by the same list either way.
        named_preprocessed_traces.sort_by(|a, b| a.0.cmp(&b.0));

        let pcs = self.config.pcs();
        // Only the serialisable domain description is kept -- it goes into the
        // verifying key's `chip_information`.
        let chip_information: Vec<_> = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| {
                let domain = pcs.natural_domain_for_degree(trace.height());
                let ser_domain = SerializableDomain::new(
                    domain.first_point(),
                    domain.size().trailing_zeros() as usize,
                );
                (name.to_owned(), ser_domain, (trace.width(), trace.height()))
            })
            .collect();

        // Commit to the preprocessed traces.  One path, always -- no height
        // threshold, no opt-in flag, no fallback.
        let named: Vec<(String, RowMajorMatrix<Val<SC>>)> = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| (name.to_string(), trace.clone()))
            .collect();
        let commit = SC::prep_commit(&named, self.core_rev());

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces =
            named_preprocessed_traces.into_iter().map(|(_, trace)| trace).collect::<Vec<_>>();

        let pc_start = program.pc_start();

        (
            StarkProvingKey {
                commit: commit.clone(),
                pc_start,
                initial_global_cumulative_sum,
                traces,
                preprocessed_mles: std::sync::OnceLock::new(),
                preprocessed_data: std::sync::OnceLock::new(),
                prep_rev: self.core_rev(),
                chip_ordering: chip_ordering.clone(),
                constraints_map,
            },
            StarkVerifyingKey {
                commit,
                pc_start,
                initial_global_cumulative_sum,
                chip_information,
                chip_ordering,
            },
        )
    }

    /// The preprocessed round's chip set: every chip with preprocessed columns,
    /// BY NAME -- exactly what `setup` commits, and in the order it commits it.
    ///
    /// `setup` asserts that a chip's generated preprocessed trace is `Some` iff
    /// `preprocessed_width() > 0` (the `map_or(0, width) == preprocessed_width()`
    /// check), so the set is a property of the MACHINE and needs neither the
    /// program nor the verifying key to reconstruct.
    pub fn preprocessed_chip_dims(&self) -> Vec<(String, usize)> {
        let mut dims: Vec<(String, usize)> = self
            .chips()
            .iter()
            .filter_map(|c| {
                let w = <A as MachineAir<Val<SC>>>::preprocessed_width(&c.air);
                (w > 0).then(|| (<A as MachineAir<Val<SC>>>::name(&c.air), w))
            })
            .collect();
        dims.sort_by(|a, b| a.0.cmp(&b.0));
        dims
    }

    /// Generates the dependencies of the given records.
    #[allow(clippy::needless_for_each)]
    pub fn generate_dependencies(
        &self,
        records: &mut [A::Record],
        _opts: &<A::Record as MachineRecord>::Config,
        chips_filter: Option<&[String]>,
    ) -> Result<(), A::Error> {
        let chips = self
            .chips
            .iter()
            .filter(|chip| {
                if let Some(chips_filter) = chips_filter {
                    chips_filter.contains(&chip.name())
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        // `ZIREN_DEPS_CENSUS=1`: per-chip cost of this pass.  It is the largest
        // item in a core worker's shard prepare and ~21 chips reach it through
        // the DEFAULT impl, which runs a full host `generate_trace` and throws
        // the matrix away -- this says which of them that actually costs.
        let census = std::env::var("ZIREN_DEPS_CENSUS").is_ok_and(|v| v != "0");
        let mut census_rows: Vec<(String, u128)> = Vec::new();
        for record in records.iter_mut() {
            for chip in chips.iter() {
                // A chip the shard does not INCLUDE gets no trace and no
                // lookups in the proof -- `shard_chips` filters on exactly this
                // predicate -- so running its dependencies can only produce
                // events nothing will match.  In practice it produced none and
                // simply allocated: the default `generate_dependencies` is
                // `generate_trace`, which builds a PADDED trace matrix and
                // throws it away.  MEASURED on a reth shard at 8 GPU, where the
                // real tracegen runs on the DEVICE and this pass does not:
                // `Bls12381FpOpAssign` 50.8 ms, `MemoryGlobalFinalize` 15.9 ms,
                // `MemoryGlobalInit` 10.8 ms -- 79 ms of an 82 ms pass, for
                // chips with no events at all, against 125-227 us for the chips
                // actually doing work (`LoadWord`, `AddSub`, `StoreWord`).
                //
                // Checking it HERE rather than hoisting the filter is
                // load-bearing: `included` is evaluated against the record as
                // it stands, and `GlobalChip` (58th) is only included once the
                // syscall and memory chips ahead of it have appended their
                // `global_lookup_events` in this very loop.
                if !chip.included(record) {
                    continue;
                }
                let span = tracing::debug_span!("chip dependencies", chip = chip.name());
                let _enter = span.enter();
                let t_chip = std::time::Instant::now();

                let mut output = A::Record::default();
                if let Err(e) = chip.generate_dependencies(record, &mut output) {
                    tracing::error!(
                        "Error generating dependencies for chip {}: {:?}",
                        chip.name(),
                        e
                    );
                    return Err(e);
                }
                record.append(&mut output);
                if census {
                    census_rows.push((chip.name(), t_chip.elapsed().as_micros()));
                }
            }
            if census {
                census_rows.sort_by(|a, b| b.1.cmp(&a.1));
                let total: u128 = census_rows.iter().map(|r| r.1).sum();
                let top = census_rows
                    .iter()
                    .take(12)
                    .map(|(n, us)| format!("{n}={}us", us))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(">>> DEPS_CENSUS total_us={total} {top}");
                census_rows.clear();
            }
        }
        Ok(())
    }

    /// Verify that a proof is complete and valid given a verifying key and a claimed digest.
    #[instrument("verify", level = "info", skip_all)]
    #[allow(clippy::match_bool)]
    pub fn verify(
        &self,
        vk: &StarkVerifyingKey<SC>,
        proof: &MachineProof<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<(), MachineVerificationError<SC>>
    where
        SC::Challenger: Clone + Sync,
        SC: Sync,
        Com<SC>: Sync,
        ShardProof<SC>: Sync,
        A: Sync
            + for<'a> Air<VerifierConstraintFolder<'a, SC>>
            + for<'b> Air<
                crate::shard_level::basefold_constraint_folder::BasefoldConstraintFolder<
                    'b,
                    Val<SC>,
                    <SC as StarkGenericConfig>::Challenge,
                    <SC as StarkGenericConfig>::Challenge,
                >,
            >,
        // Threaded to the shard verifier's static OUTER generic
        // BaseFold verify (former `OUTER_JAGGED_VERIFY_HOOK`). Verify-only, no
        // VK / committed-byte impact; both rings satisfy it.
        SC: crate::BasefoldRing,
        SC::Challenger: 'static
            + p3_challenger::FieldChallenger<crate::jagged_pcs::JaggedVal>
            + p3_challenger::GrindingChallenger<Witness = crate::jagged_pcs::JaggedVal>
            + p3_challenger::CanObserve<
                <<SC as crate::BasefoldRing>::BfMmcs as p3_commit::Mmcs<
                    crate::jagged_pcs::JaggedVal,
                >>::Commitment,
            >,
    {
        // Observe the preprocessed commitment.
        vk.observe_into(challenger);

        // Verify the shard proofs.
        if proof.shard_proofs.is_empty() {
            return Err(MachineVerificationError::EmptyProof);
        }

        // Snapshot the (now observed) base challenger as an immutable, shareable value.
        // Each shard clones this snapshot independently, exactly as the serial loop did,
        // so the per-shard challenger state is bit-identical to the serial version.
        let base_challenger = &*challenger;

        // Verify each shard proof in parallel. Each iteration is fully independent: it clones
        // the base challenger (read-only), observes its own shard public values, and verifies
        // its own shard. The closure returns only the shard index on failure (a `Send` value),
        // so we do not require the rich `MachineVerificationError<SC>` to be `Send`. To preserve
        // the serial verdict exactly, we collect the indices of all failing shards and, if any,
        // re-verify the lowest-index failure serially to reconstruct the original typed error —
        // identical to what the serial `for` loop returned (it returned the first failure).
        // The preprocessed opening round's chip set, derived from the MACHINE
        // once for the whole proof.
        let prep_chip_dims = self.preprocessed_chip_dims();
        let prep_chip_dims = &prep_chip_dims;
        let failed_shard = tracing::debug_span!("verify shard proofs").in_scope(|| {
            let verify_one = |i: usize, shard_proof: &ShardProof<SC>| {
                tracing::debug_span!("verifying shard", shard = i).in_scope(|| {
                    let chips = self
                        .shard_chips_named(&shard_proof.basefold().chip_heights)
                        .collect::<Vec<_>>();
                    let mut shard_challenger = base_challenger.clone();
                    shard_challenger
                        .observe_slice(&shard_proof.public_values[0..self.num_pv_elts()]);
                    Verifier::verify_shard(
                        &self.config,
                        vk,
                        &chips,
                        prep_chip_dims,
                        &mut shard_challenger,
                        shard_proof,
                        self.core_rev,
                    )
                    .map_err(MachineVerificationError::InvalidShardProof)
                })
            };

            // Bound how many shard verifies run concurrently, preserving the
            // EXACT verdict: shards are independent, and we still collect ALL
            // failures and re-verify the lowest-index one serially for the
            // identical typed error.
            //
            // The cap is a conservative default, not a memory guard:
            // per-shard verify materializes no padded-dense-sized transient
            // (the weight MLE comes from the branching-program closed form
            // `full_jagged_evaluation`, ~38 ms, size-independent).  Verify is
            // not a throughput target; raise
            // `ZIREN_VERIFY_SHARD_CONCURRENCY` if it ever becomes one.
            let cap = std::env::var("ZIREN_VERIFY_SHARD_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&k| k > 0)
                .unwrap_or(8);

            let n = proof.shard_proofs.len();
            let mut failures: Vec<usize> = Vec::new();
            let mut start = 0usize;
            while start < n {
                let end = (start + cap).min(n);
                let mut chunk_failures: Vec<usize> = proof.shard_proofs[start..end]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(j, shard_proof)| {
                        verify_one(start + j, shard_proof).err().map(|_| start + j)
                    })
                    .collect();
                failures.append(&mut chunk_failures);
                start = end;
            }
            failures.into_iter().min().map(|i| verify_one(i, &proof.shard_proofs[i]))
        });

        if let Some(result) = failed_shard {
            // The lowest-index shard that failed in parallel; re-run serially for its typed error.
            result?;
        }

        // Verify the cumulative sum is 0.
        tracing::debug_span!("verify global cumulative sum is 0").in_scope(|| {
            let sum = proof
                .shard_proofs
                .iter()
                .map(ShardProof::global_cumulative_sum)
                .chain(once(vk.initial_global_cumulative_sum))
                .sum::<SepticDigest<Val<SC>>>();

            if !sum.is_zero() {
                tracing::error!("global cumulative sum: {:?}", sum);
                return Err(MachineVerificationError::NonZeroCumulativeSum(LookupScope::Global, 0));
            }

            Ok(())
        })
    }
}

/// Errors that can occur during machine verification.
pub enum MachineVerificationError<SC: StarkGenericConfig> {
    /// An error occurred during the verification of a shard proof.
    InvalidShardProof(VerificationError<SC>),
    /// An error occurred during the verification of a global proof.
    InvalidGlobalProof(VerificationError<SC>),
    /// The cumulative sum is non-zero.
    NonZeroCumulativeSum(LookupScope, usize),
    /// The public values digest is invalid.
    InvalidPublicValuesDigest,
    /// The debug lookups failed.
    DebugLookupsFailed,
    /// The proof is empty.
    EmptyProof,
    /// The public values are invalid.
    InvalidPublicValues(&'static str),
    /// The number of shards is too large.
    TooManyShards,
    /// The chip occurrence is invalid.
    InvalidChipOccurrence(String),
    /// The CPU is missing in the first shard.
    MissingCpuInFirstShard,
    /// The CPU log degree is too large.
    CpuLogDegreeTooLarge(usize),
    /// The verification key is not allowed.
    InvalidVerificationKey,
}

impl<SC: StarkGenericConfig> Debug for MachineVerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MachineVerificationError::InvalidShardProof(e) => {
                write!(f, "Invalid shard proof: {:?}", e)
            }
            MachineVerificationError::InvalidGlobalProof(e) => {
                write!(f, "Invalid global proof: {:?}", e)
            }
            MachineVerificationError::NonZeroCumulativeSum(scope, shard) => {
                write!(f, "Non-zero cumulative sum.  Scope: {}, Shard: {}", scope, shard)
            }
            MachineVerificationError::InvalidPublicValuesDigest => {
                write!(f, "Invalid public values digest")
            }
            MachineVerificationError::EmptyProof => {
                write!(f, "Empty proof")
            }
            MachineVerificationError::DebugLookupsFailed => {
                write!(f, "Debug lookups failed")
            }
            MachineVerificationError::InvalidPublicValues(s) => {
                write!(f, "Invalid public values: {}", s)
            }
            MachineVerificationError::TooManyShards => {
                write!(f, "Too many shards")
            }
            MachineVerificationError::InvalidChipOccurrence(s) => {
                write!(f, "Invalid chip occurrence: {}", s)
            }
            MachineVerificationError::MissingCpuInFirstShard => {
                write!(f, "Missing CPU in first shard")
            }
            MachineVerificationError::CpuLogDegreeTooLarge(log_degree) => {
                write!(f, "CPU log degree too large: {}", log_degree)
            }
            MachineVerificationError::InvalidVerificationKey => {
                write!(f, "Invalid verification key")
            }
        }
    }
}

impl<SC: StarkGenericConfig> std::fmt::Display for MachineVerificationError<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl<SC: StarkGenericConfig> std::error::Error for MachineVerificationError<SC> {}
