use std::mem::size_of;
use zkm_derive::AlignedBorrow;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;
use zkm_pcs::{air::PV_DIGEST_NUM_WORDS, Word};

use crate::operations::{IsZeroOperation, KoalaBearWordRangeChecker};

pub const NUM_SYSCALL_INSTR_COLS: usize = size_of::<SyscallInstrColumns<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct SyscallInstrColumns<T> {
    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every Syscall row is an instruction).
    /// Register-form: SYSCALL's three operands are the fixed registers
    /// `$v0` / `$a0` / `$a1`, so the R-type frame carries bare indices.
    pub frame: crate::frame::RTypeFrameCols<T>,

    pub pc: T,
    pub next_pc: T,
    pub num_extra_cycles: T,

    /// The `next_pc` RECEIVED on the `State` bus — a COLUMN because interaction
    /// values must be linear.  Equals `next_pc` on a normal row but `pc + 4` on
    /// a halt row (the predecessor's lookahead), so the chain telescopes into
    /// the halt.  Every other chip passes `next_pc` directly.
    pub state_recv_next_pc: T,

    /// Whether the current instruction is a halt instruction.
    pub is_halt: T,

    /// Whether the current syscall is linux syscall.
    pub is_sys_linux: T,

    /// IsZero check on prev_a_value[1] for bidirectional is_sys_linux.
    pub is_prev_a1_zero: IsZeroOperation<T>,

    pub syscall_id: T,

    pub op_a_value: Word<T>,

    pub is_enter_unconstrained: IsZeroOperation<T>,
    pub is_hint_len: IsZeroOperation<T>,
    pub is_halt_check: IsZeroOperation<T>,
    pub is_exit_group_check: IsZeroOperation<T>,
    pub is_commit: IsZeroOperation<T>,
    pub is_commit_deferred_proofs: IsZeroOperation<T>,

    pub index_bitmap: [T; PV_DIGEST_NUM_WORDS],

    /// KoalaBear range check for op_b_value.
    /// Active when send_to_table=1 (bug 4) OR is_halt=1 (exit code check).
    pub op_b_range_check: KoalaBearWordRangeChecker<T>,

    /// KoalaBear range check for op_c_value.
    /// Active when send_to_table=1 (bug 4) OR is_commit_deferred_proofs=1 (digest check).
    pub op_c_range_check: KoalaBearWordRangeChecker<T>,

    /// Stored boolean: 1 when op_b needs range check (send_to_table || is_halt).
    pub op_b_check: T,

    /// Stored boolean: 1 when op_c needs range check (send_to_table || is_commit_deferred_proofs).
    pub op_c_check: T,

    pub is_real: T,
}
