use core::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use p3_keccak_air::KeccakCols;
use zkm_derive::AlignedBorrow;

/// Worker column layout for the keccak permutation.  Each row is **one** of the
/// 24 keccak-f rounds.  The round-to-round state hand-off and the multi-block
/// sponge are carried on the `PrecompileChain` buses (see `air` + the
/// `control` chip), replacing the legacy multi-row `p3_keccak` SubAir window and
/// the sponge row-selector machinery the single-row BaseFold folder cannot
/// evaluate.
///
/// `keccak` MUST stay the first field (offset 0): trace generation copies the
/// `p3_keccak` per-round columns into `row[..NUM_KECCAK_COLS]`.
#[derive(PicusAnnotations, AlignedBorrow)]
#[repr(C)]
pub(crate) struct KeccakSpongeCols<T> {
    pub keccak: KeccakCols<T>,
    /// The syscall clock (identifies which syscall this permutation belongs to).
    pub clk: T,
    /// Which sponge block (0..num_blocks) this permutation is for.
    pub block: T,
    /// This row's round index `0..24`, constrained to `Σ i·step_flags[i]`.
    pub index: T,
    pub is_real: T,
}

pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
