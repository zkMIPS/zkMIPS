use std::borrow::BorrowMut;
use zkm_pcs::PicusInfo;

use p3_field::PrimeField32;
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use zkm_core_executor::events::PrecompileEvent;
use zkm_core_executor::syscalls::SyscallCode;
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_pcs::MachineAir;

use crate::syscall::precompiles::keccak_sponge::columns::{
    KeccakSpongeCols, NUM_KECCAK_SPONGE_COLS,
};
use crate::syscall::precompiles::keccak_sponge::KeccakSpongeChip;
use crate::CoreChipError;

impl<F: PrimeField32> MachineAir<F> for KeccakSpongeChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "KeccakSponge".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        KeccakSpongeCols::<u8>::picus_info()
    }

    /// This chip has NO dependencies, so `generate_dependencies` must do
    /// nothing.
    ///
    /// The `MachineAir` default (`crates/pcs/src/air/machine.rs`) is
    /// `self.generate_trace(input, output)?` — it materialises the whole trace
    /// purely so a chip whose `generate_trace` records byte lookups into
    /// `output` gets them registered.  `generate_trace` below takes `_: &mut
    /// Self::Record` and never writes to it, and the AIR
    /// (`keccak_sponge/air.rs`) issues no byte lookups at all, so the default
    /// builds a full KeccakSponge trace — a `p3_keccak::generate_trace_rows`
    /// permutation per sponge block, 24 rounds each — and drops it on the
    /// floor.
    ///
    /// That is not free: `generate_dependencies` runs inside the trace-gen
    /// workers' turn-taking critical section (`record_gen_sync.wait_for_turn`
    /// .. `advance_turn` in `crate::utils::prove`), so it is SERIAL host time
    /// on the prover's critical path.  Measured on reth (281 shards): 104.0 ms
    /// wall / 65.8 ms thread-CPU per shard, the single largest entry in the
    /// whole `generate_dependencies` pass.  Keccak-heavy guests (reth) pay it;
    /// keccak-free ones (tendermint) never see it.
    ///
    /// The override should compute ONLY what the pass actually needs; this
    /// chip's required dependency work is empty, so the override is empty.
    ///
    /// Byte-neutral: the removed call's only effect was allocating and
    /// dropping a matrix.
    fn generate_dependencies(
        &self,
        _input: &Self::Record,
        _output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows: Vec<[F; NUM_KECCAK_SPONGE_COLS]> = Vec::new();

        for (_, event) in input.get_precompile_events(SyscallCode::KECCAK_SPONGE) {
            let event = if let PrecompileEvent::KeccakSponge(event) = event {
                event
            } else {
                unreachable!()
            };

            let block_num = event.num_blocks();
            for i in 0..block_num {
                // The per-round columns are computed by `p3_keccak` from this
                // block's *absorbed* state (the permutation input).
                let p3_keccak_trace = generate_trace_rows::<F>(vec![event.xored_state_list[i]], 0);
                for round in 0..NUM_ROUNDS {
                    let mut row = [F::ZERO; NUM_KECCAK_SPONGE_COLS];
                    let p3_keccak_row = p3_keccak_trace.row_slice(round).unwrap();
                    row[..NUM_KECCAK_COLS].copy_from_slice(&p3_keccak_row);

                    let cols: &mut KeccakSpongeCols<F> = row.as_mut_slice().borrow_mut();
                    cols.clk = F::from_u32(event.clk);
                    cols.block = F::from_u32(i as u32);
                    cols.index = F::from_u32(round as u32);
                    cols.is_real = F::ONE;
                    rows.push(row);
                }
            }
        }

        let num_real_rows = rows.len();

        // Padding rows are valid keccak rounds of the zero state (so the
        // unconditional round constraints hold) with `is_real = 0` (so they
        // contribute nothing to the bus).
        let dummy_keccak_rows = generate_trace_rows::<F>(vec![[0u64; 25]], 0);
        let mut dummy_chunk: Vec<[F; NUM_KECCAK_SPONGE_COLS]> = Vec::new();
        for round in 0..NUM_ROUNDS {
            let dummy_row = dummy_keccak_rows.row_slice(round).unwrap();
            let mut row = [F::ZERO; NUM_KECCAK_SPONGE_COLS];
            row[..NUM_KECCAK_COLS].copy_from_slice(&dummy_row);
            let cols: &mut KeccakSpongeCols<F> = row.as_mut_slice().borrow_mut();
            cols.index = F::from_u32(round as u32);
            dummy_chunk.push(row);
        }

        let num_padded_rows =
            if num_real_rows == 0 { 0 } else { num_real_rows.next_power_of_two() };
        for i in num_real_rows..num_padded_rows {
            rows.push(dummy_chunk[i % NUM_ROUNDS]);
        }

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_KECCAK_SPONGE_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_SPONGE).is_empty()
        }
    }
}
