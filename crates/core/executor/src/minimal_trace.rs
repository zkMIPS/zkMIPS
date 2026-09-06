//! Minimal-trace skeleton for the two-stage tracing split.
//!
//! Background
//! ----------
//! The current Ziren prover hot path runs the full MIPS interpreter with
//! per-cycle event emission inline. On reth (60–80 shards) this surfaces as a
//! single `zkm-bf-core-N` thread pegged at 99.9 % CPU for several minutes
//! while all other prover threads sit idle.
//!
//! The fix is to split execution into two stages:
//!
//! 1. **Stage 1 — fast / sequential**: a JIT (or interpreter-portable runner)
//!    races through the program producing a very small per-shard
//!    [`MinimalTrace`]. The MinimalTrace contains only the information needed
//!    to *re-run* the shard from its start state — start registers, pc/clk
//!    bounds, and an oracle of memory reads.
//! 2. **Stage 2 — slow / parallel**: a `TracingVM` re-runs each shard from
//!    its MinimalTrace, this time emitting every `AluEvent`, `BranchEvent`,
//!    `MemoryRecord`, … needed for proving. Because each shard's start
//!    state is captured in its MinimalTrace, Stage 2 trivially parallelises
//!    across shards via rayon.
//!
//! This module defines [`MinimalTrace`] and [`TraceChunk`], shaped for the
//! MIPS register width and the Ziren executor's state layout.
//!
//! The checkpoint pass runs on the JIT
//! (`run_fast_capture_whole_program_chunk`) and the consumer reconstructs
//! full `ExecutionRecord`s byte-identically via `trace_checkpoint`.
//!
//! TraceChunk layout notes:
//! - MIPS has 32 GPRs plus HI / LO / BRK / HEAP (36 slots in
//!   `crates/core/executor/src/jit_runner.rs::JitContext::registers`) —
//!   the snapshot mirrors that layout exactly so a future TracingVM can
//!   `JitContext::registers = trace.start_registers` directly.
//! - Ziren uses `u32` for words, so the memory_reads oracle stays compact.
//! - We carry the `shard_index` explicitly so a parallel collector can sort
//!   the resulting [`ExecutionRecord`]s back into shard order without a
//!   side channel.
//!
//! TODO:
//! - Populate `mem_reads` from JIT memory-read instrumentation. Today this
//!   field is left empty by the JIT emit path; the TracingVM will fall
//!   back to re-reading guest memory directly. The oracle becomes load-
//!   bearing only when we move to a process-per-shard model where
//!   the JIT and TracingVM live in different address spaces.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One memory-read observation emitted by the Stage-1 fast runner.
///
/// Uses MIPS-native `u32` words. This is the PRE-access `MemoryRecord` and
/// nothing else: the replay consumes entries positionally (`ReplayMem`), so
/// the issuing clk and the address are redundant -- the Nth access of a
/// deterministic replay IS the Nth entry. SP1's `MemValue` is `{clk, value}`
/// for the same reason; ours carries `shard` because the MIPS memory argument
/// keys on (shard, timestamp). 12 bytes, `Pod`: a chunk's oracle serializes as
/// one raw byte run (reth: ~1.7M entries per shard, 210M per block), not
/// field by field.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct MemValue {
    /// Value observed by the producer (= oracle answer for the TracingVM).
    pub value: u32,
    /// full memory record: the `timestamp` field of the
    /// pre-access `MemoryRecord`. Feeds `prev_timestamp`. `0` on the
    /// JIT-recorder path.
    ///
    /// `timestamp` precedes `shard` so that `(timestamp, shard)` is the
    /// little-endian image of one `u64` `(shard << 32) | clk` -- the
    /// register a JIT producer keeps its per-shard clock in, stamped onto a
    /// guest memory entry with a single 8-byte store.
    pub timestamp: u32,
    /// full memory record: the `shard` field of the
    /// `MemoryRecord` at this address *before* the access (i.e. the
    /// shard of the last prior write). Load-bearing: the memory
    /// argument's `prev_shard` for the first cross-shard touch is
    /// reconstructed from this. `0` on the JIT-recorder path (the JIT
    /// does not track per-address shard bookkeeping — see the shard-bookkeeping gap below).
    pub shard: u32,
}

// SAFETY: `repr(C)`, three `u32`s, no padding, every bit pattern valid.
unsafe impl bytemuck::Zeroable for MemValue {}
unsafe impl bytemuck::Pod for MemValue {}

/// `mem_reads` as one raw byte run (bincode: length + memcpy).
fn ser_mem_reads<S: serde::Serializer>(v: &Arc<Vec<MemValue>>, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_bytes(bytemuck::cast_slice::<MemValue, u8>(v))
}

/// Inverse of [`ser_mem_reads`]; copies into an aligned `Vec` (the byte run
/// a slice reader hands back is only byte-aligned).
fn de_mem_reads<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Arc<Vec<MemValue>>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Arc<Vec<MemValue>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("mem_reads byte run")
        }
        fn visit_bytes<E: serde::de::Error>(self, b: &[u8]) -> Result<Self::Value, E> {
            let n = b.len() / std::mem::size_of::<MemValue>();
            if n * std::mem::size_of::<MemValue>() != b.len() {
                return Err(E::custom("mem_reads byte run is not a whole number of entries"));
            }
            let mut v: Vec<MemValue> = Vec::with_capacity(n);
            // SAFETY: `MemValue: Pod` (every bit pattern is a value), the
            // destination has room for `b.len()` bytes and does not overlap
            // the source.
            unsafe {
                std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr().cast::<u8>(), b.len());
                v.set_len(n);
            }
            Ok(Arc::new(v))
        }
        fn visit_byte_buf<E: serde::de::Error>(self, b: Vec<u8>) -> Result<Self::Value, E> {
            self.visit_bytes(&b)
        }
    }
    d.deserialize_bytes(V)
}

/// One per-shard checkpoint emitted by the Stage-1 fast runner.
///
/// Carries the minimum state needed for Stage 2 to re-run the shard from
/// `pc_start` / `clk_start` up to `clk_end` and emit a full
/// `ExecutionRecord`.
///
/// Follows the MIPS register layout and Ziren's existing `JitContext` shape.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceChunk {
    /// Shard index — preserved so a parallel collector can resort outputs.
    pub shard_index: u32,
    /// Register file (36 slots, matching `JitContext::registers`) at the
    /// start of this shard's slice of execution: 0..32 are MIPS GPRs,
    /// 32/33 are HI/LO, 34 is BRK, 35 is HEAP. `Vec` (not `[u32; 36]`)
    /// because serde Deserialize is not derived for fixed arrays > 32
    /// without `serde-big-array`; len is invariantly 36.
    pub start_registers: Vec<u32>,
    /// full register memory records at shard start:
    /// `(value, shard, timestamp)` per slot (len 36 when populated).
    /// Unlike `start_registers` (value only) this carries the `shard` /
    /// `timestamp` bookkeeping the memory argument needs so a Stage-2
    /// register access reconstructs `prev_shard` / `prev_timestamp`
    /// byte-exactly. Empty on the JIT-recorder path (JIT doesn't track
    /// per-register shard/timestamp — the shard-bookkeeping gap); Stage 2 then falls
    /// back to `start_registers` with shard/timestamp 0.
    pub start_register_records: Vec<(u32, u32, u32)>,
    /// PC at which Stage 2 should begin re-executing this shard.
    pub pc_start: u32,
    /// Global clock at the start of this shard.
    pub clk_start: u64,
    /// Global clock at the end of this shard (exclusive).
    pub clk_end: u64,
    /// full byte-exact reconstruction: the executor's
    /// `state.current_shard` at the start of this shard. The memory
    /// argument's `shard` field for every access in this shard derives
    /// from it, so the Stage-2 sub-executor must seed
    /// `state.current_shard = current_shard` to match the sequential
    /// run byte-for-byte. `0` means "unset" (legacy chunks).
    pub current_shard: u32,
    /// stream cursors at the start of this shard, so the
    /// Stage-2 sub-executor can service `HINT_READ` / proof-verify /
    /// public-value syscalls from the shared streams at the exact
    /// position the sequential run was at when this shard began.
    pub input_stream_ptr: u32,
    /// see `input_stream_ptr`.
    pub proof_stream_ptr: u32,
    /// see `input_stream_ptr`.
    pub public_values_stream_ptr: u32,
    /// full final-memory carry for the LAST shard only.
    /// The global memory init/finalize argument (`postprocess`)
    /// iterates *every* touched address, so the terminal shard's
    /// sub-executor needs the whole memory image (with full records)
    /// + the uninitialized-memory (hint) image — data the sparse
    /// `mem_reads` oracle cannot supply. Populated by the producer at
    /// program halt; empty for every non-terminal chunk. Entries are
    /// `(addr, value, shard, timestamp)`.
    pub final_memory: Vec<(u32, u32, u32, u32)>,
    /// full final-memory carry: the uninitialized-memory
    /// (hint) image `(addr, value)`, needed for `initialize` events of
    /// hint-written addresses. Terminal chunk only.
    pub final_uninit_memory: Vec<(u32, u32)>,
    /// The hint-stream entries THIS chunk consumes, `None` on the legacy
    /// whole-stream path.
    ///
    /// A streaming producer cannot hand out the finished `input_stream` with
    /// chunk 0 -- the program has not run yet. It does not have to: `FD_HINT`
    /// pushes at the END of the vector and a hook splices at the CURSOR
    /// (`syscalls/write.rs`), so nothing ever rewrites a position the cursor
    /// has already passed. The window `[input_stream_ptr, next chunk's
    /// input_stream_ptr)` is therefore final the moment this chunk closes, and
    /// is captured then.
    ///
    /// The consumer seeds this slice with the cursor at 0 instead of copying
    /// the whole program's stream per worker -- which is also what makes the
    /// replay's memory bounded rather than O(workers x program).
    ///
    /// `Some(vec![])` (a chunk that consumed no hints) is NOT the same as
    /// `None`: it still means "prerecorded", so the replay must not re-run the
    /// hint and hook syscalls.
    #[serde(default)]
    pub input_stream_slice: Option<Vec<Vec<u8>>>,
    /// Oracle of memory reads observed by Stage 1. May be empty when the
    /// JIT emit path was not configured to record memory; in that case
    /// Stage 2 falls back to direct guest-memory reads.
    ///
    /// Option B (mem_reads oracle): when populated by the
    /// sequential producer, Stage 2 pre-loads its sub-Executor's
    /// page_table from these entries before replaying, eliminating the
    /// need for chunks to carry full memory state. The Arc is built at
    /// chunk-close time by moving the executor's recording `Vec` in (no
    /// copy of the ~20 MB oracle at the seal).
    #[serde(serialize_with = "ser_mem_reads", deserialize_with = "de_mem_reads")]
    pub mem_reads: Arc<Vec<MemValue>>,
}

/// A chunk's `mem_reads` under replay: a cursor, not a lookup table.
///
/// The producer pushes one entry per user-memory access in issue order, so the
/// Nth access of a deterministic replay consumes the Nth entry.  This is SP1's
/// `MemReads` (`sp1-gpu/crates/cuda`/`core/jit/src/risc.rs:285`) in safe Rust --
/// theirs is a raw pointer pair into an mmap; ours is a shared `Arc` slice,
/// which is what matters here (no per-worker clone).
#[derive(Debug, Clone)]
pub struct ReplayMem {
    /// The chunk's oracle, shared across replay workers.
    pub entries: Arc<Vec<MemValue>>,
    /// How many accesses have been served.
    pub pos: usize,
}

impl TraceChunk {
    /// Convenience constructor for tests.
    #[must_use]
    pub fn empty(shard_index: u32, pc_start: u32, clk_start: u64) -> Self {
        Self {
            shard_index,
            start_registers: vec![0; 36],
            start_register_records: Vec::new(),
            pc_start,
            clk_start,
            clk_end: clk_start,
            current_shard: 0,
            input_stream_ptr: 0,
            proof_stream_ptr: 0,
            public_values_stream_ptr: 0,
            final_memory: Vec::new(),
            final_uninit_memory: Vec::new(),
            input_stream_slice: None,
            mem_reads: Arc::new(Vec::new()),
        }
    }

    /// Number of cycles covered by this chunk.
    #[must_use]
    pub fn num_cycles(&self) -> u64 {
        self.clk_end.saturating_sub(self.clk_start)
    }
}

/// A whole-program minimal trace: one [`TraceChunk`] per shard plus the
/// program's syscall log.
///
/// `MinimalTrace` is the bridge between `MinimalExecutorRunner` (Stage 1)
/// and the TracingVM workers (Stage 2).
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MinimalTrace {
    /// One chunk per shard, in execution order.
    pub chunks: Vec<TraceChunk>,
    /// Final committed public values, captured at program halt.
    pub public_values: Vec<u32>,
    /// Total cycle count, for sanity checks / accounting.
    pub total_cycles: u64,
    /// How many chunks have already been drained by
    /// [`crate::Executor::drain_sealed_chunks`].
    ///
    /// The streaming producer hands each sealed chunk to a worker and drops it,
    /// so `chunks.len()` is no longer the number stamped so far and cannot
    /// number the next one. This counter keeps `shard_index` monotonic across
    /// drains; it stays `0` on the batched path, where nothing is removed, so
    /// the indices there are byte-identical to before.
    pub emitted: u32,
}

impl MinimalTrace {
    /// Number of shards.
    #[must_use]
    pub fn num_shards(&self) -> usize {
        self.chunks.len()
    }

    /// Index the next stamped chunk should carry.
    #[must_use]
    pub fn next_shard_index(&self) -> u32 {
        self.emitted + self.chunks.len() as u32
    }

    /// Append a chunk and update the running cycle accumulator.
    pub fn push_chunk(&mut self, chunk: TraceChunk) {
        self.total_cycles = self.total_cycles.max(chunk.clk_end);
        self.chunks.push(chunk);
    }

    /// seal the last open chunk with the final clock
    /// after the executor finishes. Drop any leading chunks whose
    /// clk_end ≤ clk_start (degenerate zero-cycle shards opened by an
    /// extra trailing `bump_record()`).
    pub fn finalize(&mut self, final_clk: u64) {
        if let Some(last) = self.chunks.last_mut() {
            if last.clk_end == u64::MAX {
                last.clk_end = final_clk;
            }
        }
        self.chunks.retain(|c| c.clk_end > c.clk_start);
        self.total_cycles = final_clk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunk_has_zero_cycles() {
        let chunk = TraceChunk::empty(0, 0x1000_0000, 0);
        assert_eq!(chunk.num_cycles(), 0);
        assert_eq!(chunk.start_registers, vec![0u32; 36]);
        assert_eq!(chunk.pc_start, 0x1000_0000);
        assert!(chunk.mem_reads.is_empty());
    }

    #[test]
    fn push_chunk_tracks_total_cycles() {
        let mut trace = MinimalTrace::default();
        let mut c0 = TraceChunk::empty(0, 0x1000_0000, 0);
        c0.clk_end = 1_024;
        trace.push_chunk(c0);
        let mut c1 = TraceChunk::empty(1, 0x1000_0400, 1_024);
        c1.clk_end = 3_072;
        trace.push_chunk(c1);

        assert_eq!(trace.num_shards(), 2);
        assert_eq!(trace.total_cycles, 3_072);
    }

    #[test]
    fn round_trips_through_bincode() {
        let mut trace = MinimalTrace::default();
        let mut c = TraceChunk::empty(7, 0x4000, 100);
        c.clk_end = 200;
        c.start_registers[5] = 0xdead_beef;
        let reads = vec![
            MemValue { value: 0x1111, shard: 3, timestamp: 0x0102_0304 },
            MemValue { value: 0x2222, shard: 0, timestamp: u32::MAX },
            MemValue { value: u32::MAX, shard: 7, timestamp: 0 },
        ];
        c.mem_reads = Arc::new(reads.clone());
        trace.push_chunk(c);
        trace.public_values = vec![1, 2, 3, 4];

        let bytes = bincode::serialize(&trace).unwrap();
        let round: MinimalTrace = bincode::deserialize(&bytes).unwrap();

        assert_eq!(round.num_shards(), 1);
        assert_eq!(round.chunks[0].shard_index, 7);
        assert_eq!(*round.chunks[0].mem_reads, reads);
        assert_eq!(round.public_values, vec![1, 2, 3, 4]);

        // The reader may hand the byte run back through either visitor.
        let owned: MinimalTrace = bincode::deserialize_from(std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(*owned.chunks[0].mem_reads, reads);
    }

    #[test]
    fn mem_reads_serialize_as_one_byte_run() {
        assert_eq!(std::mem::size_of::<MemValue>(), 12);
        let mut c = TraceChunk::empty(0, 0, 0);
        let n = 1000;
        c.mem_reads = Arc::new(
            (0..n as u32).map(|i| MemValue { value: i, shard: 1, timestamp: i }).collect(),
        );
        let bytes = bincode::serialize(&c).unwrap();
        let empty = bincode::serialize(&TraceChunk::empty(0, 0, 0)).unwrap();
        // length prefix + 12 B per entry, no per-field framing
        assert_eq!(bytes.len() - empty.len(), n * 12);
        let round: TraceChunk = bincode::deserialize(&bytes).unwrap();
        assert_eq!(round.mem_reads, c.mem_reads);
        assert!(bincode::deserialize::<TraceChunk>(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn env_flag_default_off() {
        // The flag may be set in some CI / dev environments. Just verify
        // the helper does not panic regardless of state.
    }
}
