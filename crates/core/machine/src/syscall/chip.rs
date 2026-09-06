use core::fmt;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::IntoParallelRefIterator;
use p3_maybe_rayon::prelude::ParallelIterator;

use zkm_core_executor::events::{ByteRecord, GlobalLookupEvent, PrecompileEvent};
use zkm_core_executor::{events::SyscallEvent, ByteOpcode, ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_pcs::air::AirLookup;
use zkm_pcs::air::{LookupScope, MachineAir, ZKMAirBuilder};
use zkm_pcs::LookupKind;

use crate::{utils::next_multiple_of_32, CoreChipError};

/// The number of main trace columns for `SyscallChip`.
pub const NUM_SYSCALL_COLS: usize = size_of::<SyscallCols<u8>>();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyscallShardKind {
    Core,
    Precompile,
}

/// A chip that stores the syscall invocations.
pub struct SyscallChip {
    shard_kind: SyscallShardKind,
}

impl SyscallChip {
    pub const fn new(shard_kind: SyscallShardKind) -> Self {
        Self { shard_kind }
    }

    pub const fn core() -> Self {
        Self::new(SyscallShardKind::Core)
    }

    pub const fn precompile() -> Self {
        Self::new(SyscallShardKind::Precompile)
    }

    pub fn shard_kind(&self) -> SyscallShardKind {
        self.shard_kind
    }

    /// Pack a u32 result into two half-word values (lo = bytes 0-1, hi = bytes 2-3).
    fn pack_result_halves(result: u32) -> (u32, u32) {
        let rb = result.to_le_bytes();
        (rb[0] as u32 + (rb[1] as u32) * 256, rb[2] as u32 + (rb[3] as u32) * 256)
    }
}

/// The column layout for the chip.
///
/// `arg1` and `arg2` are NOT stored as columns. They are derived inline as
/// `arg1_lo + arg1_hi * 65536` to avoid redundant columns while keeping the
/// reduced field element available for local `send_syscall`/`receive_syscall`.
///
/// **Soundness**: `arg1_lo/hi` and `arg2_lo/hi` are U16Range-checked inside
/// `send_syscall_result_packed` (see `crates/pcs/src/air/builder.rs`).
/// Any chip using this interaction gets range-checked half-words automatically.
#[derive(PicusAnnotations, AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct SyscallCols<T: Copy> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clk of the syscall.
    pub clk: T,

    /// The syscall_id of the syscall.
    pub syscall_id: T,

    /// Half-word packed arg1: low 16 bits (byte0 + byte1 * 256).
    /// Currently only SysLinuxChip uses these in its AIR constraints (via
    /// receive_syscall_result). Non-linux precompiles receive the reduced
    /// arg1/arg2 through receive_syscall and don't use the half-words.
    /// If a new precompile needs byte-level argument access, it should use
    /// receive_syscall_result_packed to get these half-words.
    pub arg1_lo: T,
    /// Half-word packed arg1: high 16 bits (byte2 + byte3 * 256).
    pub arg1_hi: T,

    /// Half-word packed arg2: low 16 bits.
    pub arg2_lo: T,
    /// Half-word packed arg2: high 16 bits.
    pub arg2_hi: T,

    /// Half-word packed result (lo = byte0 + byte1*256, hi = byte2 + byte3*256).
    pub result_lo: T,
    pub result_hi: T,

    /// Whether the syscall is a linux syscall.
    pub is_linux: T,

    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for SyscallChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        format!("Syscall{}", self.shard_kind).to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        SyscallCols::<u8>::picus_info()
    }

    fn generate_dependencies(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        let is_receive = self.shard_kind == SyscallShardKind::Precompile;

        let event_triples: Vec<(&SyscallEvent, u32, u32)> = match self.shard_kind {
            SyscallShardKind::Core => input
                .syscall_events
                .iter()
                .filter(|e| {
                    (e.a_record.prev_value.to_le_bytes()[2] == 1)
                        || (e.a_record.prev_value.to_le_bytes()[1] != 0)
                })
                .map(|event| {
                    let is_linux = event.a_record.prev_value.to_le_bytes()[1] != 0;
                    let (rlo, rhi) = if is_linux {
                        Self::pack_result_halves(event.a_record.value)
                    } else {
                        (0, 0)
                    };
                    (event, rlo, rhi)
                })
                .collect(),
            SyscallShardKind::Precompile => input
                .precompile_events
                .all_events()
                .map(|(event, precompile)| {
                    let (rlo, rhi) = match precompile {
                        PrecompileEvent::Linux(le) => Self::pack_result_halves(le.v0),
                        _ => (0, 0),
                    };
                    (event, rlo, rhi)
                })
                .collect(),
        };

        // Emit all global events and byte lookups in a single pass.
        for &(event, rlo, rhi) in &event_triples {
            let (a1_lo, a1_hi) = Self::pack_result_halves(event.arg1);
            let (a2_lo, a2_hi) = Self::pack_result_halves(event.arg2);

            // Cross-shard argument linkage using collision-resistant half-word packing.
            output.global_lookup_events.push(GlobalLookupEvent {
                message: [event.shard, event.clk, event.syscall_id, a1_lo, a1_hi, a2_lo, a2_hi],
                is_receive,
                kind: LookupKind::Syscall as u8,
            });

            // Cross-shard result linkage to ensure both shards agree on the return value.
            output.global_lookup_events.push(GlobalLookupEvent {
                message: [event.shard, event.clk, event.syscall_id, rlo, rhi, 0, 0],
                is_receive,
                kind: LookupKind::SyscallResult as u8,
            });

            // U16Range checks for half-word columns (gated by is_real in the AIR).
            output.add_u16_range_check(a1_lo as u16);
            output.add_u16_range_check(a1_hi as u16);
            output.add_u16_range_check(a2_lo as u16);
            output.add_u16_range_check(a2_hi as u16);
        }

        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = match self.shard_kind() {
            SyscallShardKind::Core => &input.syscall_events,
            SyscallShardKind::Precompile => &input
                .precompile_events
                .all_events()
                .map(|(event, _)| event.to_owned())
                .collect::<Vec<_>>(),
        };
        let nb_rows = events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(
            nb_rows,
            size_log2,
            <SyscallChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let row_fn = |syscall_event: &SyscallEvent, precompile_event: Option<&PrecompileEvent>| {
            let mut row = [F::ZERO; NUM_SYSCALL_COLS];
            let cols: &mut SyscallCols<F> = row.as_mut_slice().borrow_mut();

            cols.shard = F::from_u32(syscall_event.shard);
            cols.clk = F::from_u32(syscall_event.clk);
            cols.syscall_id = F::from_u32(syscall_event.syscall_id);
            let a1b = syscall_event.arg1.to_le_bytes();
            cols.arg1_lo = F::from_u32(a1b[0] as u32 + (a1b[1] as u32) * 256);
            cols.arg1_hi = F::from_u32(a1b[2] as u32 + (a1b[3] as u32) * 256);
            let a2b = syscall_event.arg2.to_le_bytes();
            cols.arg2_lo = F::from_u32(a2b[0] as u32 + (a2b[1] as u32) * 256);
            cols.arg2_hi = F::from_u32(a2b[2] as u32 + (a2b[3] as u32) * 256);

            // For Core shard, a_record has real prev_value with linux_sys byte.
            // For Precompile shard, a_record is default (prev_value=0), so detect
            // linux from the PrecompileEvent variant instead.
            let is_linux = match precompile_event {
                Some(PrecompileEvent::Linux(_)) => true,
                Some(_) => false,
                None => syscall_event.a_record.prev_value.to_le_bytes()[1] != 0,
            };
            cols.is_linux = F::from_bool(is_linux);
            if is_linux {
                let result = match precompile_event {
                    Some(PrecompileEvent::Linux(linux_event)) => linux_event.v0,
                    _ => syscall_event.a_record.value,
                };
                let rb = result.to_le_bytes();
                cols.result_lo = F::from_u32(rb[0] as u32 + (rb[1] as u32) * 256);
                cols.result_hi = F::from_u32(rb[2] as u32 + (rb[3] as u32) * 256);
            }
            cols.is_real = F::ONE;

            row
        };

        let mut rows = match self.shard_kind {
            SyscallShardKind::Core => input
                .syscall_events
                .par_iter()
                .filter(|event| {
                    (event.a_record.prev_value.to_le_bytes()[2] == 1)
                        || (event.a_record.prev_value.to_le_bytes()[1] != 0)
                })
                .map(|event| row_fn(event, None))
                .collect::<Vec<_>>(),
            // `all_events()` iterates the deterministic-ordered event map; collect
            // the rows in that source order.  A previous `.par_bridge()` here was
            // UNORDERED, so under RAYON_NUM_THREADS>1 the SyscallPrecompile trace
            // rows came out in a nondeterministic order -> nondeterministic proof
            // -> `zerocheck rlc_eval != point_and_eval` verify-fail.  Sequential is
            // byte-identical to the RAYON=1 golden (par_bridge was already
            // sequential there) at negligible cost for this small chip.
            SyscallShardKind::Precompile => input
                .precompile_events
                .all_events()
                .map(|(event, precompile)| row_fn(event, Some(precompile)))
                .collect::<Vec<_>>(),
        };

        // Pad the trace to a power of two depending on the proof shape in `input`.
        rows.resize(
            <SyscallChip as MachineAir<F>>::num_rows(self, input).unwrap(),
            [F::zero(); NUM_SYSCALL_COLS],
        );

        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_SYSCALL_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            match self.shard_kind {
                SyscallShardKind::Core => {
                    shard
                        .syscall_events
                        .iter()
                        .filter(|e| {
                            (e.a_record.prev_value.to_le_bytes()[2] == 1)
                                || (e.a_record.prev_value.to_le_bytes()[1] != 0)
                        })
                        .take(1)
                        .count()
                        > 0
                }
                SyscallShardKind::Precompile => {
                    !shard.precompile_events.is_empty()
                        && shard.cpu_events.is_empty()
                        && shard.global_memory_initialize_events.is_empty()
                        && shard.global_memory_finalize_events.is_empty()
                }
            }
        }
    }

    fn commit_scope(&self) -> LookupScope {
        LookupScope::Local
    }
}

impl<AB> Air<AB> for SyscallChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &SyscallCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_linux);
        // is_linux can only be 1 when is_real is 1.
        builder.when(AB::Expr::one() - local.is_real).assert_zero(local.is_linux);
        // result_lo/result_hi must be zero when is_linux is 0, so they
        // can be used directly (degree 1) in the global lookup.
        builder.when_not(local.is_linux).assert_zero(local.result_lo);
        builder.when_not(local.is_linux).assert_zero(local.result_hi);

        // Derive reduced arg1/arg2 inline from half-word columns.
        // These are NOT stored as columns — saves 2 columns per row.
        let arg1: AB::Expr = local.arg1_lo.into()
            + Into::<AB::Expr>::into(local.arg1_hi) * AB::Expr::from_u32(65536);
        let arg2: AB::Expr = local.arg2_lo.into()
            + Into::<AB::Expr>::into(local.arg2_hi) * AB::Expr::from_u32(65536);

        // U16Range checks for ALL syscalls (not just linux), gated by is_real.
        // This ensures the global lookup's half-word args are always canonical.
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            local.arg1_lo,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            local.arg1_hi,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            local.arg2_lo,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
        builder.send_byte(
            AB::Expr::from_u8(ByteOpcode::U16Range as u8),
            local.arg2_hi,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );

        match self.shard_kind {
            SyscallShardKind::Core => {
                builder.receive_syscall(
                    local.shard,
                    local.clk,
                    local.syscall_id,
                    arg1.clone(),
                    arg2.clone(),
                    local.is_real,
                    LookupScope::Local,
                );

                builder.receive_syscall_result_packed(
                    local.shard,
                    local.clk,
                    local.result_lo,
                    local.result_hi,
                    local.arg1_lo,
                    local.arg1_hi,
                    local.arg2_lo,
                    local.arg2_hi,
                    local.is_linux,
                    LookupScope::Local,
                );

                // Cross-shard argument linkage using half-word packed args to prevent
                // reduce() collisions across shards.
                builder.send(
                    AirLookup::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.arg1_lo.into(),
                            local.arg1_hi.into(),
                            local.arg2_lo.into(),
                            local.arg2_hi.into(),
                            local.is_real.into() * AB::Expr::one(),
                            local.is_real.into() * AB::Expr::zero(),
                            AB::Expr::from_u8(LookupKind::Syscall as u8),
                        ],
                        local.is_real.into(),
                        LookupKind::Global,
                    ),
                    LookupScope::Local,
                );

                // Cross-shard result linkage ensuring both Core and Precompile shards
                // agree on the syscall return value.
                builder.send(
                    AirLookup::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.result_lo.into(),
                            local.result_hi.into(),
                            AB::Expr::zero(),
                            AB::Expr::zero(),
                            local.is_real.into() * AB::Expr::one(),
                            local.is_real.into() * AB::Expr::zero(),
                            AB::Expr::from_u8(LookupKind::SyscallResult as u8),
                        ],
                        local.is_real.into(),
                        LookupKind::Global,
                    ),
                    LookupScope::Local,
                );
            }
            SyscallShardKind::Precompile => {
                builder.send_syscall(
                    local.shard,
                    local.clk,
                    local.syscall_id,
                    arg1.clone(),
                    arg2.clone(),
                    local.is_real,
                    LookupScope::Local,
                );

                builder.send_syscall_result_packed(
                    local.shard,
                    local.clk,
                    local.result_lo,
                    local.result_hi,
                    local.arg1_lo,
                    local.arg1_hi,
                    local.arg2_lo,
                    local.arg2_hi,
                    local.is_linux,
                    LookupScope::Local,
                );

                // Cross-shard argument linkage using half-word packed args to prevent
                // reduce() collisions across shards.
                builder.send(
                    AirLookup::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.arg1_lo.into(),
                            local.arg1_hi.into(),
                            local.arg2_lo.into(),
                            local.arg2_hi.into(),
                            local.is_real.into() * AB::Expr::zero(),
                            local.is_real.into() * AB::Expr::one(),
                            AB::Expr::from_u8(LookupKind::Syscall as u8),
                        ],
                        local.is_real.into(),
                        LookupKind::Global,
                    ),
                    LookupScope::Local,
                );

                // Cross-shard result linkage ensuring both shards agree on the return value.
                builder.send(
                    AirLookup::new(
                        vec![
                            local.shard.into(),
                            local.clk.into(),
                            local.syscall_id.into(),
                            local.result_lo.into(),
                            local.result_hi.into(),
                            AB::Expr::zero(),
                            AB::Expr::zero(),
                            local.is_real.into() * AB::Expr::zero(),
                            local.is_real.into() * AB::Expr::one(),
                            AB::Expr::from_u8(LookupKind::SyscallResult as u8),
                        ],
                        local.is_real.into(),
                        LookupKind::Global,
                    ),
                    LookupScope::Local,
                );
            }
        }
    }
}

impl<F> BaseAir<F> for SyscallChip {
    fn width(&self) -> usize {
        NUM_SYSCALL_COLS
    }
}

impl fmt::Display for SyscallShardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyscallShardKind::Core => write!(f, "Core"),
            SyscallShardKind::Precompile => write!(f, "Precompile"),
        }
    }
}
