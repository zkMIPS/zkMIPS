//! SHA-256 extend **control chip** — the two endpoints of the
//! [`LookupKind::PrecompileChain`] state-chaining bus for SHA-256 extend.
//!
//! The single-row BaseFold zerocheck folder cannot evaluate the worker's
//! legacy `cycle_16`/`cycle_48` row-selector flag machinery, so the per-row
//! loop-index sequencing is carried on a LogUp bus instead.  This control chip
//! emits exactly one row per `SHA_EXTEND` syscall: it receives the syscall,
//! **sends** the initial index `i = 16`, and **receives** the final index
//! `i = 64`.  Each `ShaExtendChip` worker row receives `i` and sends `i + 1`,
//! so the LogUp multiset only balances when the per-syscall chain telescopes
//! `16 → 64` across exactly 48 worker rows, pinning each row's `i`.
//!
//! Built on Ziren's generic `PrecompileChain` kind (isolated by the leading
//! `syscall_id`) and scalar `clk`/`w_ptr`.  Extend carries no `a..h` digest
//! state, so the bus tuple is just `(pid, shard, clk, w_ptr, i)`.

use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{events::PrecompileEvent, syscalls::SyscallCode, ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_pcs::{
    air::{AirLookup, LookupScope, MachineAir},
    LookupKind, ZKMAirBuilder,
};

use crate::{utils::pad_rows_fixed, CoreChipError};

pub const NUM_SHA_EXTEND_CONTROL_COLS: usize = size_of::<ShaExtendControlCols<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,
    pub is_real: T,
}

/// SHA-256 extend control chip.  One row per `SHA_EXTEND` syscall.
#[derive(Default)]
pub struct ShaExtendControlChip;

impl ShaExtendControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for ShaExtendControlChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaExtendControlChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShaExtendControl".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        ShaExtendControlCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows: Vec<[F; NUM_SHA_EXTEND_CONTROL_COLS]> = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_EXTEND) {
            let event =
                if let PrecompileEvent::ShaExtend(event) = event { event } else { unreachable!() };
            let mut row = [F::ZERO; NUM_SHA_EXTEND_CONTROL_COLS];
            let cols: &mut ShaExtendControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);
            cols.is_real = F::ONE;
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_SHA_EXTEND_CONTROL_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <ShaExtendControlChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_CONTROL_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty()
        }
    }
}

impl<AB> Air<AB> for ShaExtendControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &ShaExtendControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Receive the SHA_EXTEND syscall once per real invocation.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_u32(SyscallCode::SHA_EXTEND.syscall_id()),
            local.w_ptr,
            AB::Expr::ZERO,
            local.is_real,
            LookupScope::Local,
        );

        // Leading precompile-ID field isolates this chain from other
        // precompiles sharing `LookupKind::PrecompileChain`.
        let pid = AB::Expr::from_u32(SyscallCode::SHA_EXTEND.syscall_id());

        let tuple = |index: AB::Expr| -> Vec<AB::Expr> {
            vec![pid.clone(), local.shard.into(), local.clk.into(), local.w_ptr.into(), index]
        };

        // Send the initial index `i = 16`.
        builder.send(
            AirLookup::new(
                tuple(AB::Expr::from_u32(16)),
                local.is_real.into(),
                LookupKind::PrecompileChain,
            ),
            LookupScope::Local,
        );

        // Receive the final index `i = 64`.
        builder.receive(
            AirLookup::new(
                tuple(AB::Expr::from_u32(64)),
                local.is_real.into(),
                LookupKind::PrecompileChain,
            ),
            LookupScope::Local,
        );
    }
}
