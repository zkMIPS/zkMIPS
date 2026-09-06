//! SHA-256 compress **control chip** — the two endpoints of the
//! [`LookupKind::PrecompileChain`] state-chaining bus for SHA-256 compress.
//!
//! The single-row BaseFold zerocheck folder cannot evaluate the worker's
//! legacy `when_first_row`/`when_transition`/`next.*` state machinery, so
//! the per-row state transition is carried on a LogUp bus instead.  This
//! control chip emits exactly one row per `SHA_COMPRESS` syscall: it
//! receives the syscall, **sends** the initial `a..h` digest at `index = 0`,
//! and **receives** the final `a..h` delta at `index = 80`.  Each
//! `ShaCompressChip` worker row receives `state @ index` and sends
//! `state @ index + 1`, so the LogUp multiset only balances when the chain
//! telescopes `0 → 80`, pinning the per-row ordering by multiplicity.
//!
//! Built on Ziren's generic `PrecompileChain` kind (isolated by the leading
//! `syscall_id`), scalar `clk/w_ptr/h_ptr`, and `Word<T>` state representation.

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
    LookupKind, Word, ZKMAirBuilder,
};

use crate::{utils::pad_rows_fixed, CoreChipError};

pub const NUM_SHA_COMPRESS_CONTROL_COLS: usize = size_of::<ShaCompressControlCols<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaCompressControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,
    pub h_ptr: T,
    pub is_real: T,
    /// The input digest `a..h` (the 8 `H` words), sent on the chain at `index = 0`.
    pub initial_state: [Word<T>; 8],
    /// The final digest delta `written_H[i] - H[i]` (the accumulated `a..h`),
    /// received off the chain at `index = 80`.
    pub final_state: [Word<T>; 8],
}

/// SHA-256 compress control chip.  One row per `SHA_COMPRESS` syscall.
#[derive(Default)]
pub struct ShaCompressControlChip;

impl ShaCompressControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for ShaCompressControlChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaCompressControlChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShaCompressControl".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        ShaCompressControlCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows: Vec<[F; NUM_SHA_COMPRESS_CONTROL_COLS]> = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::SHA_COMPRESS) {
            let event = if let PrecompileEvent::ShaCompress(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::ZERO; NUM_SHA_COMPRESS_CONTROL_COLS];
            let cols: &mut ShaCompressControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.w_ptr = F::from_canonical_u32(event.w_ptr);
            cols.h_ptr = F::from_canonical_u32(event.h_ptr);
            cols.is_real = F::ONE;
            for i in 0..8 {
                let prev = event.h[i];
                let written = event.h_write_records[i].value;
                // Initial state = input H; final state = the accumulated a..h
                // (written H minus the input H), matching the worker's bus.
                cols.initial_state[i] = Word::from(prev);
                cols.final_state[i] = Word::from(written.wrapping_sub(prev));
            }
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_SHA_COMPRESS_CONTROL_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <ShaCompressControlChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_COMPRESS_CONTROL_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty()
        }
    }
}

impl<AB> Air<AB> for ShaCompressControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &ShaCompressControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Receive the SHA_COMPRESS syscall once per real invocation.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_u32(SyscallCode::SHA_COMPRESS.syscall_id()),
            local.w_ptr,
            local.h_ptr,
            local.is_real,
            LookupScope::Local,
        );

        // Leading precompile-ID field isolates this chain from other
        // precompiles sharing `LookupKind::PrecompileChain`.
        let pid = AB::Expr::from_u32(SyscallCode::SHA_COMPRESS.syscall_id());

        // Send the initial state `(pid, shard, clk, w_ptr, h_ptr, 0, a..h)`.
        let mut send_vals: Vec<AB::Expr> = Vec::new();
        send_vals.push(pid.clone());
        send_vals.push(local.shard.into());
        send_vals.push(local.clk.into());
        send_vals.push(local.w_ptr.into());
        send_vals.push(local.h_ptr.into());
        send_vals.push(AB::Expr::ZERO);
        for word in local.initial_state.iter() {
            for b in word.0.iter() {
                send_vals.push((*b).into());
            }
        }
        builder.send(
            AirLookup::new(send_vals, local.is_real.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );

        // Receive the final state `(pid, shard, clk, w_ptr, h_ptr, 80, a..h)`.
        let mut recv_vals: Vec<AB::Expr> = Vec::new();
        recv_vals.push(pid);
        recv_vals.push(local.shard.into());
        recv_vals.push(local.clk.into());
        recv_vals.push(local.w_ptr.into());
        recv_vals.push(local.h_ptr.into());
        recv_vals.push(AB::Expr::from_u32(80));
        for word in local.final_state.iter() {
            for b in word.0.iter() {
                recv_vals.push((*b).into());
            }
        }
        builder.receive(
            AirLookup::new(recv_vals, local.is_real.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );
    }
}
