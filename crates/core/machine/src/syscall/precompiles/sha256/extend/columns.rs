use std::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use zkm_derive::AlignedBorrow;

use crate::{
    memory::{MemoryReadCols, MemoryWriteCols},
    operations::{
        Add4Operation, FixedRotateRightOperation, FixedShiftRightOperation, XorOperation,
    },
};

pub const NUM_SHA_EXTEND_COLS: usize = size_of::<ShaExtendCols<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendCols<T> {
    /// Inputs.
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,

    /// The loop index `i` for this row (16 ≤ i < 64).  Its per-row sequencing is
    /// pinned by the `PrecompileChain` state bus (see `eval_state_bus`): the
    /// `ShaExtendControlChip` seeds `i = 16` and drains `i = 64`, and each worker
    /// row receives `i` and sends `i + 1`, so the multiset only balances when the
    /// per-syscall chain telescopes `16 → 64`.  This replaces the legacy
    /// `cycle_16`/`cycle_48` row-selector flag machinery the single-row BaseFold
    /// zerocheck folder cannot evaluate.
    pub i: T,

    /// Inputs to `s0`.
    pub w_i_minus_15: MemoryReadCols<T>,
    pub w_i_minus_15_rr_7: FixedRotateRightOperation<T>,
    pub w_i_minus_15_rr_18: FixedRotateRightOperation<T>,
    pub w_i_minus_15_rs_3: FixedShiftRightOperation<T>,
    pub s0_intermediate: XorOperation<T>,

    /// `s0 := (w[i-15] rightrotate  7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)`.
    pub s0: XorOperation<T>,

    /// Inputs to `s1`.
    pub w_i_minus_2: MemoryReadCols<T>,
    pub w_i_minus_2_rr_17: FixedRotateRightOperation<T>,
    pub w_i_minus_2_rr_19: FixedRotateRightOperation<T>,
    pub w_i_minus_2_rs_10: FixedShiftRightOperation<T>,
    pub s1_intermediate: XorOperation<T>,

    /// `s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)`.
    pub s1: XorOperation<T>,

    /// Inputs to `s2`.
    pub w_i_minus_16: MemoryReadCols<T>,
    pub w_i_minus_7: MemoryReadCols<T>,

    /// `w[i] := w[i-16] + s0 + w[i-7] + s1`.
    pub s2: Add4Operation<T>,

    /// Result.
    pub w_i: MemoryWriteCols<T>,

    /// Selector.
    pub is_real: T,
}
