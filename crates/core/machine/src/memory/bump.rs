use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use itertools::Itertools;
use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use zkm_core_executor::events::{ByteLookupEvent, ByteRecord, MemoryBumpEvent, MemoryReadRecord};
use zkm_core_executor::{ByteOpcode, ExecutionRecord, Program, NUM_REGISTERS};
use zkm_derive::AlignedBorrow;
use zkm_pcs::air::{LookupScope, MachineAir, ZKMAirBuilder};

use crate::{
    air::MemoryAirBuilder,
    memory::MemoryReadCols,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

pub(crate) const NUM_MEMORY_BUMP_COLS: usize = size_of::<MemoryBumpCols<u8>>();

/// The columns of the `MemoryBump` chip.
///
/// One row per (register, shard): a *shadow read* of the register at `(shard, 0)` that advances
/// the register's memory-argument timestamp out of whatever earlier shard it was left in.
#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryBumpCols<T> {
    /// The shadow read.  This is a *full* [`crate::memory::MemoryAccessCols`], i.e. it pays for
    /// the shard-vs-clk comparison — but exactly once per (register, shard) instead of once per
    /// register access.
    pub access: MemoryReadCols<T>,

    /// The shard the register is bumped into.
    pub shard: T,

    /// The register address.  Constrained to be `< NUM_REGISTERS`.
    pub addr: T,

    /// Whether this is a real row.
    pub is_real: T,
}

/// A chip that bumps the memory-argument timestamp of every register touched in a shard.
///
/// Registers carry their `(shard, clk)` across shard boundaries, so the first access to a register
/// in a shard would have to compare *shards* rather than clks.  Rather than pay three columns for
/// that on every register access of every cycle, this chip performs one shadow read per (register,
/// shard) at `(shard, 0)`.  Because `clk` restarts at 0 each shard and register accesses live at
/// the sub-cycle positions `1..=4`, `(shard, 0)` is strictly below every real register access, so
/// the shadow read is forced to be the first link of the shard's access chain for that register.
///
/// Consequently every register access proven anywhere in the machine has `prev_shard == shard`,
/// which is what lets [`crate::memory::RegisterAccessCols`] be 6 columns instead of 9.
///
/// Ziren's "high limb" of the timestamp is the shard number, which changes once per shard, so the
/// chip is bounded by `NUM_REGISTERS` rows per shard instead of being driven by a 24-bit epoch.
#[derive(Default)]
pub struct MemoryBumpChip {}

impl MemoryBumpChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for MemoryBumpChip {
    fn width(&self) -> usize {
        NUM_MEMORY_BUMP_COLS
    }
}

/// The byte lookups a single bump row emits, shared by `generate_dependencies` and
/// `generate_trace` so the two can never drift.
fn bump_row_blu_events(event: &MemoryBumpEvent, blu: &mut impl ByteRecord) {
    // The shard is range-checked to 16 bits (the memory-argument ordering argument assumes both
    // comparands are < 2^24).
    blu.add_u16_range_check(event.shard as u16);

    // The address must be a register: bumping a non-register address would let the prover splice
    // an extra link into that address's access chain.
    blu.add_byte_lookup_event(ByteLookupEvent {
        opcode: ByteOpcode::LTU,
        a1: 1,
        a2: 0,
        b: event.addr as u8,
        c: NUM_REGISTERS as u8,
    });
}

impl<F: PrimeField32> MachineAir<F> for MemoryBumpChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "MemoryBump".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        MemoryBumpCols::<u8>::picus_info()
    }

    fn generate_dependencies(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        let blu_batches = input
            .bump_memory_events
            .chunks(1)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                for event in events {
                    bump_row_blu_events(event, &mut blu);
                    // The shadow read's own timestamp comparison limbs.
                    let mut cols = MemoryBumpCols::<F>::default();
                    cols.access.populate(read_record(event), &mut blu);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.bump_memory_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        Some(next_multiple_of_32(
            nb_rows,
            size_log2,
            <MemoryBumpChip as MachineAir<F>>::name(self).as_str(),
        ))
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.bump_memory_events;
        let padded_nb_rows = <MemoryBumpChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MEMORY_BUMP_COLS);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        values[..events.len() * NUM_MEMORY_BUMP_COLS]
            .par_chunks_mut(chunk_size * NUM_MEMORY_BUMP_COLS)
            .enumerate()
            .for_each(|(i, rows)| {
                rows.chunks_mut(NUM_MEMORY_BUMP_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let event = &events[idx];
                    let cols: &mut MemoryBumpCols<F> = row.borrow_mut();
                    let mut blu = Vec::new();
                    cols.access.populate(read_record(event), &mut blu);
                    cols.shard = F::from_u32(event.shard);
                    cols.addr = F::from_u32(event.addr);
                    cols.is_real = F::ONE;
                });
            });

        Ok(RowMajorMatrix::new(values, NUM_MEMORY_BUMP_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.bump_memory_events.is_empty()
        }
    }

    fn commit_scope(&self) -> LookupScope {
        LookupScope::Local
    }
}

/// The shadow read a bump event stands for: the register's value, re-read at `(shard, 0)`.
#[inline]
fn read_record(event: &MemoryBumpEvent) -> MemoryReadRecord {
    MemoryReadRecord {
        value: event.value,
        shard: event.shard,
        timestamp: 0,
        prev_shard: event.prev_shard,
        prev_timestamp: event.prev_timestamp,
    }
}

impl<AB> Air<AB> for MemoryBumpChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &MemoryBumpCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // The shard must be within 16 bits: the memory ordering argument compares shards when the
        // shards differ, and assumes both comparands are < 2^24.
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            local.shard,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            local.is_real,
        );

        // The address must be a register, i.e. `addr < NUM_REGISTERS`.
        //
        // This is load-bearing.  A bump splices an extra link into an address's per-shard access
        // chain at clk 0.  For a *register* that is harmless: register accesses only ever occur at
        // sub-cycle positions `1..=4`, so the new link can only be the first one, and the chain
        // stays sorted by clk.  For a general memory address, which can be accessed at sub-cycle
        // position 0 (i.e. clk 0 on the shard's first instruction), a bump could be spliced in
        // mid-shard and reset the chain, letting a later write be read by an earlier access.
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::LTU as u8),
            AB::Expr::ONE,
            local.addr,
            AB::Expr::from_u32(NUM_REGISTERS as u32),
            local.is_real,
        );

        // The shadow read itself, at `(shard, 0)`.  This is an ordinary memory access, so it
        // carries the full shard-vs-clk comparison and proves `(shard, 0) > (prev_shard,
        // prev_clk)`.
        builder.eval_memory_access(
            local.shard,
            AB::Expr::ZERO,
            local.addr,
            &local.access,
            local.is_real,
        );
    }
}
