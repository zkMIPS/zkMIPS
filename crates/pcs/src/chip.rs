use std::hash::Hash;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{get_max_constraint_degree, AirLayout, SymbolicAirBuilder};
use p3_util::log2_ceil_usize;

use crate::{
    air::{LookupScope, MachineAir, MultiTableAirBuilder, ZKMAirBuilder},
    local_permutation_trace_width,
    lookup::{Lookup, LookupBuilder, LookupKind},
    scoped_lookups, PicusInfo,
};

use super::{eval_permutation_constraints, generate_permutation_trace, PROOF_MAX_NUM_PVS};

/// An Air that encodes lookups based on lookups.
pub struct Chip<F: Field, A> {
    /// The underlying AIR of the chip for constraint evaluation.
    pub air: A,
    /// The lookups that the chip sends.
    pub sends: Vec<Lookup<F>>,
    /// The lookups that the chip receives.
    pub receives: Vec<Lookup<F>>,
    /// The relative log degree of the quotient polynomial, i.e. `log2(max_constraint_degree - 1)`.
    pub log_quotient_degree: usize,
}

impl<F: Field, A> Chip<F, A> {
    /// The send lookups of the chip.
    pub fn sends(&self) -> &[Lookup<F>] {
        &self.sends
    }

    /// The receive lookups of the chip.
    pub fn receives(&self) -> &[Lookup<F>] {
        &self.receives
    }

    /// The relative log degree of the quotient polynomial, i.e. `log2(max_constraint_degree - 1)`.
    pub const fn log_quotient_degree(&self) -> usize {
        self.log_quotient_degree
    }

    /// Consumes the chip and returns the underlying air.
    pub fn into_inner(self) -> A {
        self.air
    }
}

impl<F: PrimeField32, A: MachineAir<F>> Chip<F, A> {
    /// Returns whether the given chip is included in the execution record of the shard.
    pub fn included(&self, shard: &A::Record) -> bool {
        self.air.included(shard)
    }
}

impl<F, A> Chip<F, A>
where
    F: Field,
    A: BaseAir<F>,
{
    /// Records the lookups and constraint degree from the air and creates a new chip.
    pub fn new(air: A) -> Self
    where
        A: MachineAir<F> + Air<LookupBuilder<F>> + Air<SymbolicAirBuilder<F>>,
    {
        let mut builder = LookupBuilder::new(air.preprocessed_width(), air.width());
        air.eval(&mut builder);
        let (sends, receives) = builder.lookups();

        let nb_byte_sends = sends.iter().filter(|s| s.kind == LookupKind::Byte).count();
        let nb_byte_receives = receives.iter().filter(|r| r.kind == LookupKind::Byte).count();
        tracing::debug!(
            "chip {} has {} byte lookups",
            air.name(),
            nb_byte_sends + nb_byte_receives
        );

        let mut max_constraint_degree = get_max_constraint_degree(
            &air,
            AirLayout {
                preprocessed_width: air.preprocessed_width(),
                main_width: air.width(),
                num_public_values: PROOF_MAX_NUM_PVS,
                ..Default::default()
            },
        );

        if !sends.is_empty() || !receives.is_empty() {
            max_constraint_degree = max_constraint_degree.max(3);
        }
        let log_quotient_degree = log2_ceil_usize(max_constraint_degree - 1);

        Self { air, sends, receives, log_quotient_degree }
    }

    /// Returns the number of lookups in the chip.
    #[inline]
    pub fn num_lookups(&self) -> usize {
        self.sends.len() + self.receives.len()
    }

    /// Returns the number of sent byte lookups in the chip.
    #[inline]
    pub fn num_sent_byte_lookups(&self) -> usize {
        self.sends.iter().filter(|i| i.kind == LookupKind::Byte).count()
    }

    /// Generates a permutation trace for the given matrix.
    pub fn generate_permutation_trace<EF: ExtensionField<F>>(
        &self,
        preprocessed: Option<&RowMajorMatrix<F>>,
        main: &RowMajorMatrix<F>,
        random_elements: &[EF],
    ) -> (RowMajorMatrix<EF>, EF)
    where
        F: PrimeField,
        A: MachineAir<F>,
    {
        let batch_size = self.logup_batch_size();
        generate_permutation_trace::<F, EF>(
            &self.sends,
            &self.receives,
            preprocessed,
            main,
            random_elements,
            batch_size,
        )
    }

    /// Returns the width of the permutation trace.
    #[inline]
    pub fn permutation_width(&self) -> usize {
        let (scoped_sends, scoped_receives) = scoped_lookups(self.sends(), self.receives());
        let empty = Vec::new();
        let local_sends = scoped_sends.get(&LookupScope::Local).unwrap_or(&empty);
        let local_receives = scoped_receives.get(&LookupScope::Local).unwrap_or(&empty);

        local_permutation_trace_width(
            local_sends.len() + local_receives.len(),
            self.logup_batch_size(),
        )
    }

    /// Returns the cost of a row in the chip.
    ///
    /// This is the per-row cell count of the trace the prover actually
    /// *materializes and commits*: preprocessed + main. It is the unit of the
    /// per-shard trace-AREA budget (`ELEMENT_THRESHOLD`), which the executor
    /// accumulates as `Σ_chip event_counts[chip] × cost(chip)` and compares
    /// against the size the jagged dense commitment is sized for.
    ///
    /// It deliberately does NOT include the permutation or quotient widths.
    /// Charging `4·perm + 4·quot` on top inflated the
    /// charge against a dense that only ever holds preprocessed + main: the
    /// measured inflation is ~1.75× for the tendermint and goat chip mixes, so
    /// shards were split ~1.75× earlier than the commitment shape required.
    ///
    /// `ELEMENT_THRESHOLD` is calibrated against this definition and must be
    /// re-derived whenever it changes — see `zkm_pcs::opts::ELEMENT_THRESHOLD`.
    #[inline]
    pub fn cost(&self) -> u64
    where
        A: MachineAir<F>,
    {
        let preprocessed_cols = self.preprocessed_width();
        let main_cols = self.width();
        (preprocessed_cols + main_cols) as u64
    }

    /// Returns the width of the quotient polynomial.
    #[inline]
    pub const fn quotient_width(&self) -> usize {
        1 << self.log_quotient_degree
    }

    /// Returns the log2 of the batch size.
    #[inline]
    pub const fn logup_batch_size(&self) -> usize {
        1 << self.log_quotient_degree
    }
}

impl<F, A> BaseAir<F> for Chip<F, A>
where
    F: Field,
    A: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        panic!("Chip should not use the `BaseAir` method, but the `MachineAir` method.")
    }
}

impl<F, A> MachineAir<F> for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    type Record = A::Record;

    type Program = A::Program;

    type Error = A::Error;

    fn name(&self) -> String {
        self.air.name()
    }

    fn preprocessed_width(&self) -> usize {
        <A as MachineAir<F>>::preprocessed_width(&self.air)
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        <A as MachineAir<F>>::preprocessed_num_rows(&self.air, program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &A::Program) -> Option<RowMajorMatrix<F>> {
        <A as MachineAir<F>>::generate_preprocessed_trace(&self.air, program)
    }

    fn num_rows(&self, input: &A::Record) -> Option<usize> {
        <A as MachineAir<F>>::num_rows(&self.air, input)
    }

    fn generate_trace(
        &self,
        input: &A::Record,
        output: &mut A::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        self.air.generate_trace(input, output)
    }

    fn generate_dependencies(
        &self,
        input: &A::Record,
        output: &mut A::Record,
    ) -> Result<(), Self::Error> {
        self.air.generate_dependencies(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        self.air.included(shard)
    }

    fn commit_scope(&self) -> crate::air::LookupScope {
        self.air.commit_scope()
    }

    fn picus_info(&self) -> PicusInfo {
        self.air.picus_info()
    }

    fn selectors_partition_real_rows(&self) -> bool {
        self.air.selectors_partition_real_rows()
    }

    fn picus_selector_specialization_allowed(&self, selector_name: &str) -> bool {
        self.air.picus_selector_specialization_allowed(selector_name)
    }
}

// Implement AIR directly on Chip, evaluating both execution and permutation constraints.
impl<'a, F, A, AB> Air<AB> for Chip<F, A>
where
    F: Field,
    A: Air<AB> + MachineAir<F>,
    AB: ZKMAirBuilder<F = F> + MultiTableAirBuilder<'a> + 'a,
{
    fn eval(&self, builder: &mut AB) {
        // Evaluate the execution trace constraints.
        self.air.eval(builder);
        // Evaluate permutation constraints.
        let batch_size = self.logup_batch_size();
        eval_permutation_constraints(
            &self.sends,
            &self.receives,
            batch_size,
            self.air.commit_scope(),
            builder,
        );
    }
}

impl<F, A> PartialEq for Chip<F, A>
where
    F: Field,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.air == other.air
    }
}

impl<F: Field, A: Eq> Eq for Chip<F, A> where F: Field + Eq {}

impl<F, A> Hash for Chip<F, A>
where
    F: Field,
    A: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.air.hash(state);
    }
}
