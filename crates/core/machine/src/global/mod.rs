use std::{
    borrow::Borrow,
    mem::transmute,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use rayon_scan::ScanParallelIterator;
use std::borrow::BorrowMut;
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, GlobalLookupEvent},
    ByteOpcode, ExecutionRecord, Program,
};
use zkm_pcs::{
    air::{AirLookup, LookupScope, MachineAir},
    septic_curve::{SepticCurve, SepticCurveComplete},
    septic_digest::SepticDigest,
    septic_extension::{SepticBlock, SepticExtension},
    LookupKind, ZKMAirBuilder,
};

use crate::{
    operations::{
        GlobalAccumulationOperation, GlobalDigestRow, GlobalLookupOperation, Y6_TOP_BOUND,
    },
    utils::{indices_arr, next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};
use zkm_derive::AlignedBorrow;

const NUM_GLOBAL_COLS: usize = size_of::<GlobalCols<u8>>();

/// Creates the column map for the CPU.
const fn make_col_map() -> GlobalCols<usize> {
    let indices_arr = indices_arr::<NUM_GLOBAL_COLS>();
    unsafe { transmute::<[usize; NUM_GLOBAL_COLS], GlobalCols<usize>>(indices_arr) }
}

const GLOBAL_COL_MAP: GlobalCols<usize> = make_col_map();

pub const GLOBAL_INITIAL_DIGEST_POS: usize = GLOBAL_COL_MAP.accumulation.initial_digest[0].0[0];

// Option 2: bumped 64 -> 65 — the `index` column added to `GlobalCols`
// (for the GlobalAccumulation bus chain) shifts `accumulation.initial_digest`
// one position right.  Kept in sync with the struct-derived
// `GLOBAL_INITIAL_DIGEST_POS` by the assert in `name()` below.
pub const GLOBAL_INITIAL_DIGEST_POS_COPY: usize = 30;

/// Whether the `GlobalChip` trace (and with it the three per-row byte-table
/// lookups that need `lift_x`) is generated on the DEVICE.  When set, the host
/// `generate_dependencies` emits only the `U16Range(message[0])` lookups and the
/// device trace generator publishes the other multiplicities through
/// `ExecutionRecord::global_byte_lookups` for the Byte chip to fold in.  A host
/// `lift_x` is ~7 us; a reth shard has ~560 K global events, so leaving this
/// off on a GPU prover would add ~4 CPU-seconds per shard to the dependency pass.
static GLOBAL_BYTE_LOOKUPS_ON_DEVICE: AtomicBool = AtomicBool::new(false);

pub fn set_global_byte_lookups_on_device(on: bool) {
    GLOBAL_BYTE_LOOKUPS_ON_DEVICE.store(on, Ordering::Release);
}

pub fn global_byte_lookups_on_device() -> bool {
    GLOBAL_BYTE_LOOKUPS_ON_DEVICE.load(Ordering::Acquire)
}

/// Compute the digest row of every event, in parallel.
pub fn compute_global_digest_rows<F: PrimeField32>(
    events: &[GlobalLookupEvent],
) -> Vec<GlobalDigestRow> {
    events
        .par_iter()
        .with_min_len(1 << 10)
        .map(|event| {
            GlobalLookupOperation::<F>::digest_row(
                SepticBlock(event.message),
                event.is_receive,
                event.kind,
            )
        })
        .collect()
}

#[repr(C)]
pub struct Ghost {
    pub v: [usize; GLOBAL_INITIAL_DIGEST_POS_COPY],
}

#[derive(Default)]
pub struct GlobalChip;

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct GlobalCols<T: Copy> {
    pub message: [T; 7],
    pub kind: T,
    pub lookup: GlobalLookupOperation<T>,
    pub is_receive: T,
    pub is_send: T,
    pub is_real: T,
    /// Option 2: running real-row index for the GlobalAccumulation bus
    /// chain.  Set to the row's position by trace-gen.  Placed before
    /// `accumulation` so the latter (and its trailing cumulative_sum)
    /// stays at the end of the row (perm-constraint requirement).
    pub index: T,
    pub accumulation: GlobalAccumulationOperation<T, 1>,
}

impl<F: PrimeField32> MachineAir<F> for GlobalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        assert_eq!(GLOBAL_INITIAL_DIGEST_POS_COPY, GLOBAL_INITIAL_DIGEST_POS);
        "Global".to_string()
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let events = &input.global_lookup_events;

        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        // Aggregate per chunk into a COUNTING map, exactly like every other
        // chip in this machine.  `U16Range(message[0])` collapses to a couple
        // of dozen distinct keys per shard (`message[0]` is a shard index).
        //
        // The three range-check lookups of the digest columns (`y6_lo16`,
        // `y6_mid8`+`offset`, `LTU(y6_top, 63)`) need the lifted point.  On the
        // host path the rows are computed here ONCE (parallel), published in
        // `global_digests` for `generate_trace`, and counted; on the device path
        // (`global_byte_lookups_on_device()`) the GPU trace generator derives
        // and publishes those multiplicities itself.
        let digests = if global_byte_lookups_on_device() {
            None
        } else {
            let rows = input
                .global_digests
                .get(events.len())
                .unwrap_or_else(|| Arc::new(compute_global_digest_rows::<F>(events)));
            input.global_digests.publish(events.len(), rows.clone());
            Some(rows)
        };

        let blu_batches = events
            .chunks(chunk_size)
            .enumerate()
            .par_bridge()
            .map(|(ci, events)| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                let base = ci * chunk_size;
                events.iter().enumerate().for_each(|(k, event)| {
                    blu.add_u16_range_check(event.message[0].try_into().unwrap());
                    if let Some(rows) = &digests {
                        let row = &rows[base + k];
                        blu.add_u16_range_check(row.y6_lo16);
                        blu.add_u8_range_check(row.y6_mid8, row.offset);
                        blu.add_byte_lookup_event(ByteLookupEvent {
                            opcode: ByteOpcode::LTU,
                            a1: 1,
                            a2: 0,
                            b: row.y6_top,
                            c: Y6_TOP_BOUND,
                        });
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.global_lookup_events;
        let nb_rows = events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(
            nb_rows,
            size_log2,
            <GlobalChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.global_lookup_events;

        let nb_rows = events.len();
        let padded_nb_rows = <GlobalChip as MachineAir<F>>::num_rows(self, input).unwrap();
        // One `lift_x` per event: reuse the rows the dependency pass published.
        let digests = input
            .global_digests
            .get(nb_rows)
            .unwrap_or_else(|| Arc::new(compute_global_digest_rows::<F>(events)));
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_GLOBAL_COLS);
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 0) + 1;

        let mut chunks = values[..nb_rows * NUM_GLOBAL_COLS]
            .chunks_mut(chunk_size * NUM_GLOBAL_COLS)
            .collect::<Vec<_>>();

        let point_chunks = chunks
            .par_iter_mut()
            .enumerate()
            .map(|(i, rows)| {
                let mut point_chunks = Vec::with_capacity(chunk_size * NUM_GLOBAL_COLS + 1);
                if i == 0 {
                    point_chunks.push(SepticCurveComplete::Affine(SepticDigest::<F>::zero().0));
                }
                rows.chunks_mut(NUM_GLOBAL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut GlobalCols<F> = row.borrow_mut();
                    let event: &GlobalLookupEvent = &events[idx];
                    cols.message = event.message.map(F::from_u32);
                    cols.kind = F::from_u8(event.kind);
                    cols.lookup.populate_from_row(&digests[idx]);
                    cols.is_real = F::ONE;
                    // Option 2: running index for the GlobalAccumulation
                    // bus chain (real rows are placed contiguously at
                    // rows 0..nb_rows, so `idx` is the chain position).
                    cols.index = F::from_u32(idx as u32);
                    if event.is_receive {
                        cols.is_receive = F::ONE;
                    } else {
                        cols.is_send = F::ONE;
                    }
                    point_chunks.push(SepticCurveComplete::Affine(SepticCurve {
                        x: SepticExtension(cols.lookup.x_coordinate.0),
                        y: SepticExtension(cols.lookup.y_coordinate.0),
                    }));
                });
                point_chunks
            })
            .collect::<Vec<_>>();

        let points = point_chunks.into_iter().flatten().collect::<Vec<_>>();
        let cumulative_sum = points
            .into_par_iter()
            .with_min_len(1 << 15)
            .scan(|a, b| *a + *b, SepticCurveComplete::Infinity)
            .collect::<Vec<SepticCurveComplete<F>>>();

        // Publish the digest this scan just produced instead of making
        // `public_values()` re-fold every event from scratch.  `points` is the
        // `SepticDigest::zero()` offset followed by one point per event, so the
        // last element of the inclusive scan IS
        // `compute_global_cumulative_sum(events)`.  With no events the scan is
        // empty and the digest is the bare offset — note this differs from
        // `final_digest` below, which deliberately uses `dummy()` for the AIR's
        // padding rows.
        input.global_cumulative_sum.publish(
            nb_rows,
            SepticDigest(SepticCurve::convert(
                cumulative_sum.last().map_or_else(
                    || SepticDigest::<F>::zero().0,
                    zkm_pcs::septic_curve::SepticCurveComplete::point,
                ),
                |x: F| x.as_canonical_u32(),
            )),
        );
        // Padding rows carry the shard digest in their trailing columns (see
        // `GlobalAccumulationOperation::populate_dummy`).
        let final_digest = match cumulative_sum.last() {
            Some(digest) => digest.point(),
            None => SepticCurve::<F>::dummy(),
        };

        let chunk_size = std::cmp::max(padded_nb_rows / num_cpus::get(), 0) + 1;
        values.chunks_mut(chunk_size * NUM_GLOBAL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_GLOBAL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut GlobalCols<F> = row.borrow_mut();
                    if idx < nb_rows {
                        cols.accumulation.populate_real(&cumulative_sum[idx..idx + 2]);
                    } else {
                        cols.lookup.populate_dummy();
                        cols.accumulation.populate_dummy(final_digest);
                    }
                });
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_GLOBAL_COLS))
    }

    fn included(&self, _: &Self::Record) -> bool {
        true
    }

    fn commit_scope(&self) -> LookupScope {
        LookupScope::Global
    }
}

impl<F> BaseAir<F> for GlobalChip {
    fn width(&self) -> usize {
        NUM_GLOBAL_COLS
    }
}

impl<AB> Air<AB> for GlobalChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &GlobalCols<AB::Var> = (*local).borrow();

        // Receive the arguments, which consists of 7 message columns, `is_send`, `is_receive`, and `kind`.
        // In MemoryGlobal, MemoryLocal, Syscall chips, `is_send`, `is_receive`, `kind` are sent with correct constant values.
        // For a global send lookup, `is_send = 1` and `is_receive = 0` are used.
        // For a global receive lookup, `is_send = 0` and `is_receive = 1` are used.
        // For a memory global lookup, `kind = LookupKind::Memory` is used.
        // For a syscall global lookup, `kind = LookupKind::Syscall` is used.
        // Therefore, `is_send`, `is_receive` are already known to be boolean, and `kind` is also known to be a `u8` value.
        // Note that `local.is_real` is constrained to be boolean in `eval_single_digest`.
        builder.receive(
            AirLookup::new(
                vec![
                    local.message[0].into(),
                    local.message[1].into(),
                    local.message[2].into(),
                    local.message[3].into(),
                    local.message[4].into(),
                    local.message[5].into(),
                    local.message[6].into(),
                    local.is_send.into(),
                    local.is_receive.into(),
                    local.kind.into(),
                ],
                local.is_real.into(),
                LookupKind::Global,
            ),
            LookupScope::Local,
        );

        // Evaluate the lookup.
        GlobalLookupOperation::<AB::F>::eval_single_digest(
            builder,
            local.message.map(Into::into),
            local.lookup,
            local.is_receive.into(),
            local.is_send.into(),
            local.is_real,
            local.kind,
        );

        // Evaluate the local (is_real-gated) curve accumulation.
        GlobalAccumulationOperation::<AB::F, 1>::eval_accumulation(
            builder,
            [local.lookup],
            [local.is_real],
            local.accumulation,
        );

        // Option 2 GlobalAccumulation bus: chain the running digest via a
        // multiset-balanced control interaction instead of the legacy
        // when_transition `final_digest == next.initial_digest`.  Each
        // real row RECEIVEs (index, initial_digest) and SENDs (index+1,
        // cumulative_sum); the public-values AIR (`eval_global_sum`)
        // closes the chain at both ends — initial `(0, ZERO_DIGEST)`
        // (matched to row 0's initial_digest) and final `(global_count,
        // global_cumulative_sum)` (matched to the last real row's
        // cumulative_sum).  Multiplicity is `is_real`, so the chain spans
        // exactly the real rows and `global_count` equals the real count.
        let mut recv_vals: Vec<AB::Expr> = Vec::with_capacity(15);
        recv_vals.push(local.index.into());
        for i in 0..7 {
            recv_vals.push(local.accumulation.initial_digest[0].0[i].into());
        }
        for i in 0..7 {
            recv_vals.push(local.accumulation.initial_digest[1].0[i].into());
        }
        builder.receive(
            AirLookup::new(recv_vals, local.is_real.into(), LookupKind::GlobalAccumulation),
            LookupScope::Local,
        );

        let mut send_vals: Vec<AB::Expr> = Vec::with_capacity(15);
        send_vals.push(local.index.into() + AB::Expr::ONE);
        for i in 0..7 {
            send_vals.push(local.accumulation.cumulative_sum[0][0].0[i].into());
        }
        for i in 0..7 {
            send_vals.push(local.accumulation.cumulative_sum[0][1].0[i].into());
        }
        builder.send(
            AirLookup::new(send_vals, local.is_real.into(), LookupKind::GlobalAccumulation),
            LookupScope::Local,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programs::tests::simple_program;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Executor};
    use zkm_pcs::{air::MachineAir, ZKMCoreOpts};

    #[test]
    fn test_global_generate_trace() {
        let program = simple_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip: GlobalChip = GlobalChip;

        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{mem_event:?}");
        }
    }
}

