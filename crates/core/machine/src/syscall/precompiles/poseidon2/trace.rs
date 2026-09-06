use crate::operations::poseidon2::trace::populate_perm_deg3;
use crate::operations::poseidon2::WIDTH;
use crate::syscall::precompiles::poseidon2::columns::{Poseidon2MemCols, NUM_COLS};
use crate::syscall::precompiles::poseidon2::Poseidon2PermuteChip;
use crate::utils::pad_rows_fixed;
use crate::CoreChipError;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use std::borrow::BorrowMut;
use zkm_core_executor::events::{ByteRecord, Poseidon2PermuteEvent, PrecompileEvent};
use zkm_core_executor::syscalls::SyscallCode;
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_pcs::MachineAir;
use zkm_pcs::PicusInfo;

impl<F: PrimeField32> MachineAir<F> for Poseidon2PermuteChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "Poseidon2Permute".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        Poseidon2MemCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = input.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE);

        let mut rows = events
            .par_iter()
            .map(|(_, event)| {
                let event = if let PrecompileEvent::Poseidon2Permute(event) = event {
                    event
                } else {
                    unreachable!();
                };

                let mut row = [F::ZERO; NUM_COLS];
                self.event_to_row(event, &mut row, &mut Vec::new());
                row
            })
            .collect::<Vec<_>>();

        let mut dummy_row = [F::ZERO; NUM_COLS];
        let dummy_cols: &mut Poseidon2MemCols<F> = dummy_row.as_mut_slice().borrow_mut();
        dummy_cols.poseidon2 = populate_perm_deg3([F::ZERO; WIDTH], None);
        pad_rows_fixed(
            &mut rows,
            || dummy_row,
            input.fixed_log2_rows::<F, _>(self),
            <Poseidon2PermuteChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let events = input.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::Poseidon2Permute(event) = event {
                        event
                    } else {
                        unreachable!();
                    };

                    let mut row = [F::ZERO; NUM_COLS];
                    self.event_to_row(event, &mut row, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE).is_empty()
        }
    }
}

impl Poseidon2PermuteChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &Poseidon2PermuteEvent,
        input_row: &mut [F],
        blu: &mut impl ByteRecord,
    ) {
        let cols: &mut Poseidon2MemCols<F> = input_row.borrow_mut();
        cols.clk = F::from_u32(event.clk);
        cols.shard = F::from_u32(event.shard);
        cols.state_addr = F::from_u32(event.state_addr);
        cols.is_real = F::ONE;

        let input = event.pre_state.map(F::from_u32);
        let output = event.post_state.map(F::from_u32);
        cols.poseidon2 = populate_perm_deg3(input, Some(output));

        // Populate memory columns.
        for i in 0..WIDTH {
            cols.state_mem[i].populate(event.state_records[i], blu);
            cols.pre_state_range_check_cols[i].populate(blu, event.pre_state[i]);
            cols.post_state_range_check_cols[i].populate(blu, event.post_state[i]);
        }
    }
}
