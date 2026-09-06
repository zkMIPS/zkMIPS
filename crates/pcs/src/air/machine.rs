use std::error::Error;

use p3_air::BaseAir;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::{septic_digest::SepticDigest, MachineRecord, PicusInfo};

pub use zkm_derive::MachineAir;

use super::LookupScope;

/// An AIR that is part of a multi table AIR arithmetization.
pub trait MachineAir<F: Field>: BaseAir<F> + 'static + Send + Sync {
    /// The execution record containing events for producing the air trace.
    type Record: MachineRecord;

    /// The program that defines the control flow of the machine.
    type Program: MachineProgram<F>;

    /// The type used for error handling.
    type Error: Error + Send + Sync;

    /// A unique identifier for this AIR as part of a machine.
    fn name(&self) -> String;

    /// The number of rows in the trace
    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        None
    }

    /// Generate the trace for a given execution record.
    ///
    /// - `input` is the execution record containing the events to be written to the trace.
    /// - `output` is the execution record containing events that the `MachineAir` can add to the
    ///   record such as byte lookup requests.
    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error>;

    /// Generate the dependencies for a given execution record.
    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        self.generate_trace(input, output)?;
        Ok(())
    }

    /// Whether this execution record contains events for this air.
    fn included(&self, shard: &Self::Record) -> bool;

    /// The width of the preprocessed trace.
    fn preprocessed_width(&self) -> usize {
        0
    }

    /// The number of rows in the preprocessed trace
    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        None
    }

    /// Generate the preprocessed trace given a specific program.
    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Specifies whether it's trace should be part of either the global or local commit.
    fn commit_scope(&self) -> LookupScope {
        LookupScope::Local
    }

    /// Returns information about Picus annotations on AIR columns.
    ///
    /// This includes:
    /// - Input ranges: columns marked with `#[picus(input)]`
    /// - Selector indices: columns marked with `#[picus(selector)]`
    fn picus_info(&self) -> PicusInfo {
        PicusInfo::default()
    }

    /// Whether the chip's `#[picus(selector)]` columns partition its real rows.
    ///
    /// When `true`, the Picus `top` module asserts that the selector columns sum to exactly
    /// `is_real` (or to one when `is_real` is specialized), rather than merely being boolean and
    /// mutually exclusive.  Opt-in: an instruction chip whose selectors are its opcode flags
    /// declares it; table chips keep the weaker default.
    fn selectors_partition_real_rows(&self) -> bool {
        false
    }

    /// Whether Picus should generate a selector-specialized module for `selector_name`.
    ///
    /// Override only when a selector value is impossible by trace construction and would make
    /// the specialized module contradictory (and therefore vacuously deterministic).
    fn picus_selector_specialization_allowed(&self, _selector_name: &str) -> bool {
        true
    }
}

/// A program that defines the control flow of a machine through a program counter.
pub trait MachineProgram<F>: Send + Sync {
    /// Gets the starting program counter.
    fn pc_start(&self) -> F;

    /// Gets the initial global cumulative sum.
    fn initial_global_cumulative_sum(&self) -> SepticDigest<F>;
}
