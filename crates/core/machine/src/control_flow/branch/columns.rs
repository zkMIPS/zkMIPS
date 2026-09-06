use std::mem::size_of;
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{PicusInfo, Word};

use crate::operations::{AddOperation, KoalaBearWordRangeChecker};

pub const NUM_BRANCH_COLS: usize = size_of::<BranchColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BranchColumns<T> {
    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every Branch row is an instruction).
    /// I-type across all six opcodes: `op_b` is a register (the zero-compare
    /// decodes read register 0), `op_c` the branch offset immediate.
    pub frame: crate::frame::ITypeFrameCols<T>,

    /// The current program counter.
    pub pc: T,

    /// The next program counter.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The inlined target addition: `target = next_pc + op_c` (the branch
    /// delay-slot target, proven in-row instead of via an AddSub request row).
    pub target_add: AddOperation<T>,

    /// The next next program counter.
    pub next_next_pc: Word<T>,

    /// Range check for next next program counter.
    /// Use it instead of check on target pc since reduced next_next_pc is directly used
    /// and target_pc equals to next_next_pc when it really works(the branch is taken).
    pub next_next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// Branch Instructions Selectors.
    #[picus(selector)]
    pub is_beq: T,
    #[picus(selector)]
    pub is_bne: T,
    #[picus(selector)]
    pub is_bltz: T,
    #[picus(selector)]
    pub is_blez: T,
    #[picus(selector)]
    pub is_bgtz: T,
    #[picus(selector)]
    pub is_bgez: T,

    /// The branching column is equal to:
    ///
    /// > is_beq & a_eq_b ||
    /// > is_bne & !a_eq_b ||
    /// > is_bltz & msb_a ||
    /// > is_bgtz & a_gt_0 ||
    /// > is_blez & !a_gt_0 ||
    /// > is_bgez & !msb_a
    pub is_branching: T,

    /// A branch only ever needs EQUALITY of `op_a`/`op_b` (BEQ/BNE) and the
    /// SIGN of `op_a` (the zero-compares read register 0 as `op_b`, so
    /// `a_eq_b` doubles as `a == 0` there).  The general 17-column signed
    /// `LtOperation` this chip used to carry — plus its two AND and one LTU
    /// byte lookups per row — is replaced by the 7 columns below and a single
    /// MSB lookup on the zero-compare rows.
    ///
    /// Equality is two `IsZero`s over the 16-bit limb differences
    /// `(a0-b0) + 256*(a1-b1)` and `(a2-b2) + 256*(a3-b3)`: with byte-shaped
    /// words each difference is in `[-65535, 65535]`, so it vanishes in the
    /// field iff both byte differences do.
    pub eq_lo: T,
    pub eq_lo_inv: T,
    pub eq_hi: T,
    pub eq_hi_inv: T,
    /// `eq_lo * eq_hi`, materialized so every consumer stays at degree <= 3.
    pub a_eq_b: T,

    /// The sign bit of `op_a` (bit 31), bound by an MSB byte lookup on the
    /// zero-compare rows — the only rows that consult it.
    pub msb_a: T,
    /// `(1 - msb_a) * (1 - a_eq_b)` — `op_a > 0` signed, materialized.
    /// Meaningful only on the zero-compare rows, where `op_b` is register 0.
    pub a_gt_0: T,
}
