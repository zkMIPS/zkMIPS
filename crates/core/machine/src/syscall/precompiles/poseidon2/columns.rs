use core::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use zkm_derive::AlignedBorrow;

use crate::memory::MemoryWriteCols;
use crate::operations::poseidon2::{Poseidon2Operation, WIDTH};
use crate::operations::KoalaBearWordRangeChecker;

/// Poseidon2MemCols is the column layout for the poseidon2 permutation.
#[derive(PicusAnnotations, Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub(crate) struct Poseidon2MemCols<T: Copy> {
    pub poseidon2: Poseidon2Operation<T>,

    pub shard: T,
    pub clk: T,
    pub state_addr: T,

    /// Memory columns for the state
    pub state_mem: [MemoryWriteCols<T>; WIDTH],

    /// Columns to KoalaBear range check the state
    pub pre_state_range_check_cols: [KoalaBearWordRangeChecker<T>; WIDTH],
    pub post_state_range_check_cols: [KoalaBearWordRangeChecker<T>; WIDTH],

    pub is_real: T,
}

pub(crate) const NUM_COLS: usize = size_of::<Poseidon2MemCols<u8>>();
