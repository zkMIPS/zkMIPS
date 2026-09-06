use crate::memory::MemoryReadCols;
use crate::operations::{IsEqualWordOperation, XorOperation};
use zkm_derive::AlignedBorrow;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;
use zkm_pcs::Word;

/// `BooleanCircuitGarbleCols` is the worker column layout for the Boolean Circuit
/// Garble precompile.  One row per gate; the per-row sequencing (`gate_id`,
/// `input_address`) and the per-syscall constants (`gates_num`, `delta`) are
/// pinned by the [`crate::syscall::precompiles::boolean_circuit_garble::control`]
/// chip through the `LookupKind::PrecompileChain` state bus, replacing the legacy
/// `when_first_row`/`next.*` row-selector machinery the single-row BaseFold
/// zerocheck folder cannot evaluate.
#[derive(PicusAnnotations, AlignedBorrow)]
#[repr(C)]
pub struct BooleanCircuitGarbleCols<T> {
    pub shard: T,
    pub clk: T,
    pub is_real: T,
    /// Address of this gate's 17-word info block.  Received on the bus; the
    /// control seeds gate 0 at `syscall_input + 20` and each worker row sends
    /// `input_address + GATE_INFO_BYTES * 4`.
    pub input_address: T,
    /// This gate's index in `0..gates_num`.  Received on the bus (`gate_id`),
    /// sent as `gate_id + 1`; the chain telescopes `0 → gates_num`.
    pub gate_id: T,
    /// Total gate count for this syscall (constant across the chain).
    pub gates_num: T,
    /// Gate kind one-hot: `[AND, OR]` (sums to `is_real`).
    pub gate_type: [T; 2],
    /// The per-syscall `delta` mask `[u8; 16]`, received on the bus (the control
    /// reads it from memory and binds it).
    pub delta: [Word<T>; 4],
    /// gate_type, h0, h1, label_b, expected_ciphertext.
    pub gates_input_mem: [MemoryReadCols<T>; 17],
    pub aux1: [XorOperation<T>; 4],                   // h1 ^ h0
    pub aux2: [XorOperation<T>; 4],                   // h1 ^ h0 ^ label_b
    pub aux3: [XorOperation<T>; 4],                   // h1 ^ h0 ^ label_b ^ delta
    pub is_equal_words: [IsEqualWordOperation<T>; 4], // computed ciphertext == expected_ciphertext
    /// Running per-gate check products (`checks[2]` = this gate verified).
    pub checks: [T; 3],
}

pub const NUM_BOOLEAN_CIRCUIT_GARBLE_COLS: usize = size_of::<BooleanCircuitGarbleCols<u8>>();
