use std::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use zkm_derive::AlignedBorrow;
use zkm_pcs::Word;

use crate::{
    memory::MemoryReadWriteCols,
    operations::{AddOperation, GtColsBytes, IsZeroOperation},
};

pub const NUM_SYS_LINUX_COLS: usize = size_of::<SysLinuxCols<u8>>();

/// Linux Syscall AIR columns.
///
/// All branch selectors are **derived** from `syscall_id` / `a0` / `a1` via `IsZeroOperation`.
/// Intermediate values (`page_offset`, `upper_address`, `is_offset_0`) are computed inline
/// from byte decompositions, not stored.
#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct SysLinuxCols<T> {
    // ── Common inputs (15 cols) ────────────────────────────────────────
    pub shard: T,
    pub clk: T,
    pub syscall_id: T,
    pub a0: Word<T>,
    pub a1: Word<T>,
    pub result: Word<T>,

    // ── Memory access (26 cols) ────────────────────────────────────────
    /// Shared memory access for brk (read BRK), write (read A2), and mmap (write HEAP).
    /// Read-only guard: `when(is_brk + is_write).assert_word_eq(value, prev_value)`.
    pub inorout: MemoryReadWriteCols<T>,
    /// A3 output register write.
    pub output: MemoryReadWriteCols<T>,

    // ── Canonical syscall decoder (17 cols) ────────────────────────────
    pub decode_mmap: IsZeroOperation<T>,
    pub decode_mmap2: IsZeroOperation<T>,
    pub decode_clone: IsZeroOperation<T>,
    pub decode_exit_group: IsZeroOperation<T>,
    pub decode_brk: IsZeroOperation<T>,
    pub decode_fnctl: IsZeroOperation<T>,
    pub decode_read: IsZeroOperation<T>,
    pub decode_write: IsZeroOperation<T>,
    /// Stored: decode_mmap.result + decode_mmap2.result (for degree).
    pub is_mmap: T,

    // ── Canonical a0 / a1 decoder (10 cols) ────────────────────────────
    pub decode_a0_0: IsZeroOperation<T>,
    pub decode_a0_1: IsZeroOperation<T>,
    pub decode_a0_2: IsZeroOperation<T>,
    pub decode_a1_1: IsZeroOperation<T>,
    pub decode_a1_3: IsZeroOperation<T>,

    // ── Composite flags (3 cols) ───────────────────────────────────────
    pub is_mmap_a0_0: T,
    pub is_fnctl_a1_1: T,
    pub is_fnctl_a1_3: T,

    // ── mmap columns (15 cols) ─────────────────────────────────────────
    // page_offset, upper_address, is_offset_0 are computed inline, not stored.
    /// 4-bit decomposition of the low nibble of a1[1].
    /// page_offset = a1[0] + a1_byte1_lo * 256 where a1_byte1_lo = sum(bits * 2^i).
    pub a1_byte1_lo_bits: [T; 4],
    /// 4-bit decomposition of the high nibble of a1[1].
    pub a1_byte1_hi_bits: [T; 4],
    /// IsZero on page_offset for bidirectional is_offset_0 derivation.
    pub is_page_offset_zero: IsZeroOperation<T>,
    /// mmap size as a Word for bytewise heap update.
    pub mmap_size: Word<T>,
    /// Carry bits for upper_address + 0x1000 → mmap_size (unaligned case).
    /// carry[0]: byte1 overflow ((a1_byte1_hi + 1) * 16 >= 256, i.e. a1_byte1_hi == 15).
    /// carry[1]: byte2 overflow (a1[2] + carry[0] >= 256, i.e. a1[2] == 255 && carry[0] == 1).
    pub mmap_size_carry: [T; 2],
    /// AddOperation for new_heap = old_heap + mmap_size.
    pub heap_add: AddOperation<T>,

    // ── brk columns (8 cols) ───────────────────────────────────────────
    pub is_a0_gt_brk: GtColsBytes<T>,

    // ── bookkeeping (1 col) ────────────────────────────────────────────
    pub is_real: T,
}
