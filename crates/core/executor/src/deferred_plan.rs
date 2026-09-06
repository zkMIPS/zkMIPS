//! Deferred-shard planning over event REFERENCES instead of events.
//!
//! [`ExecutionRecord::split`] decides where the deferred (precompile +
//! global-memory) events of a program are cut into shards. It needs the
//! events in hand. A controller that never holds the records — the shards
//! were replayed elsewhere, from trace chunks — still has to make the same
//! decision, and hand each cut to whoever owns the events. This module is
//! that decision, expressed over per-event WEIGHTS and artifact handles:
//! the greedy grouping `split` applies to a prefix of the event stream is a
//! prefix of the grouping it applies to the whole, so feeding it one chunk's
//! events at a time, in chunk order, cuts exactly where a single call over
//! the concatenation would.
//!
//! Two rules are shared with `split` through [`precompile_split_weight`] and
//! [`precompile_split_threshold`], so the record path and the plan path
//! cannot drift apart.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use zkm_pcs::SplitOpts;

use crate::{
    events::{MemoryInitializeFinalizeEvent, PrecompileEvent, SyscallEvent},
    syscalls::{precompiles::keccak::sponge::GENERAL_BLOCK_SIZE_U32S, SyscallCode},
    ExecutionRecord, Program,
};

/// The per-code threshold `split` cuts precompile shards at.
#[must_use]
pub fn precompile_split_threshold(code: SyscallCode, opts: &SplitOpts) -> usize {
    match code {
        SyscallCode::KECCAK_SPONGE => opts.keccak,
        SyscallCode::SHA_EXTEND => opts.sha_extend,
        SyscallCode::SHA_COMPRESS => opts.sha_compress,
        SyscallCode::BOOLEAN_CIRCUIT_GARBLE => opts.boolean_circuit_garble,
        _ => opts.deferred,
    }
}

/// The weight one precompile event contributes towards its code's threshold.
///
/// Keccak and garble shards are cut by WORK (blocks absorbed, gates), every
/// other code by count. `None` means the code is cut by count.
#[must_use]
pub fn precompile_split_weight(code: SyscallCode, event: &PrecompileEvent) -> Option<usize> {
    match code {
        SyscallCode::KECCAK_SPONGE => Some(match event {
            // input_len_u32s is a multiple of GENERAL_BLOCK_SIZE_U32S.
            PrecompileEvent::KeccakSponge(e) => e.input_len_u32s as usize / GENERAL_BLOCK_SIZE_U32S,
            _ => 0,
        }),
        SyscallCode::BOOLEAN_CIRCUIT_GARBLE => Some(match event {
            PrecompileEvent::BooleanCircuitGarble(e) => e.num_gates() + 1,
            _ => 0,
        }),
        _ => None,
    }
}

/// A contiguous run of events inside one artifact: `[from, to)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSlice {
    /// Whatever the owner of the events uses to find them again.
    pub artifact: String,
    pub from: u32,
    pub to: u32,
}

/// One deferred shard, as the controller wants it built.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredShardPlan {
    /// A precompile shard: one syscall code, its events in stream order.
    Precompile { code: SyscallCode, slices: Vec<EventSlice> },
    /// A global-memory shard: `opts.memory` init and finalize events each
    /// (addr-sorted), and the address bits `split` stamps on its public
    /// values.
    Memory {
        init: Vec<EventSlice>,
        finalize: Vec<EventSlice>,
        previous_init_addr_bits: [u32; 32],
        last_init_addr_bits: [u32; 32],
        previous_finalize_addr_bits: [u32; 32],
        last_finalize_addr_bits: [u32; 32],
    },
}

/// One addr-sorted global-memory event stream, described by its owner.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEventsDesc {
    pub artifact: String,
    pub count: u32,
    /// The address of the LAST event of every `opts.memory`-sized chunk of
    /// the stream, in order — the only addresses the shard public values
    /// need. See [`MemoryEventsDesc::describe`].
    pub boundary_addrs: Vec<u32>,
}

impl MemoryEventsDesc {
    /// Describe an addr-SORTED event stream the caller stored under `artifact`.
    #[must_use]
    pub fn describe(artifact: String, events: &[MemoryInitializeFinalizeEvent], memory: usize) -> Self {
        let boundary_addrs = events
            .chunks(memory.max(1))
            .map(|c| c.last().expect("chunks are non-empty").addr)
            .collect();
        Self { artifact, count: events.len() as u32, boundary_addrs }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingEvent {
    artifact: u32,
    index: u32,
    weight: u32,
}

/// Whether to log one `DEFERRED_PLAN` line per cut (same switch as the
/// executor's `SHARD_CLOSE` census: `ZIREN_SHARD_CLOSE_CENSUS=1`).
fn deferred_census_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(std::env::var("ZIREN_SHARD_CLOSE_CENSUS").ok().as_deref(), Some("1") | Some("true"))
    })
}

/// The controller's mirror of the `deferred` record `split` consumes.
#[derive(Debug)]
pub struct DeferredPlanner {
    opts: SplitOpts,
    artifacts: Vec<String>,
    // Ordered by code so the shard order is a function of the events, not of
    // a hash map's iteration order (which is what `split` has).
    pending: BTreeMap<SyscallCode, VecDeque<PendingEvent>>,
    memory: Option<(MemoryEventsDesc, MemoryEventsDesc)>,
}

impl DeferredPlanner {
    #[must_use]
    pub fn new(opts: SplitOpts) -> Self {
        Self { opts, artifacts: Vec::new(), pending: BTreeMap::new(), memory: None }
    }

    fn artifact_id(&mut self, artifact: &str) -> u32 {
        if let Some(i) = self.artifacts.iter().rposition(|a| a == artifact) {
            return i as u32;
        }
        self.artifacts.push(artifact.to_string());
        (self.artifacts.len() - 1) as u32
    }

    /// Append one chunk's events of `code`, stored under `artifact` in stream
    /// order; `weights` gives every event's [`precompile_split_weight`] (any
    /// value for a count-cut code).
    pub fn push_precompile(&mut self, code: SyscallCode, artifact: &str, weights: impl IntoIterator<Item = u32>) {
        let a = self.artifact_id(artifact);
        let q = self.pending.entry(code).or_default();
        for (i, w) in weights.into_iter().enumerate() {
            q.push_back(PendingEvent { artifact: a, index: i as u32, weight: w });
        }
    }

    /// The program's global-memory init/finalize streams (terminal chunk only).
    pub fn push_memory(&mut self, init: MemoryEventsDesc, finalize: MemoryEventsDesc) {
        self.memory = Some((init, finalize));
    }

    /// Mirror of `ExecutionRecord::split(last, None, opts)`: the shards that
    /// call would cut now, in order, leaving the remainder pending.
    pub fn split(&mut self, last: bool) -> Vec<DeferredShardPlan> {
        let mut shards = Vec::new();
        let opts = self.opts;
        for (&code, q) in self.pending.iter_mut() {
            let threshold = precompile_split_threshold(code, &opts);
            let by_weight = matches!(code, SyscallCode::KECCAK_SPONGE | SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
            loop {
                // Find the end of the next full shard.
                let mut n = 0usize;
                let mut acc = 0usize;
                let mut full = false;
                for e in q.iter() {
                    if by_weight {
                        if acc + e.weight as usize > threshold && n > 0 {
                            full = true;
                            break;
                        }
                        acc += e.weight as usize;
                        n += 1;
                    } else {
                        n += 1;
                        if n == threshold {
                            full = true;
                            break;
                        }
                    }
                }
                if !full {
                    // The remainder: a shard only on the last call.
                    if last && n > 0 {
                        shards.push(Self::cut(&self.artifacts, code, q, n));
                    }
                    break;
                }
                shards.push(Self::cut(&self.artifacts, code, q, n));
            }
        }
        self.pending.retain(|_, q| !q.is_empty());

        if last {
            if let Some((init, fin)) = self.memory.take() {
                let m = opts.memory.max(1);
                let n_init = init.count as usize;
                let n_fin = fin.count as usize;
                let n_chunks = n_init.div_ceil(m).max(n_fin.div_ceil(m));
                let bits = |addr: u32| core::array::from_fn(|i| (addr >> i) & 1);
                let mut init_bits = [0u32; 32];
                let mut fin_bits = [0u32; 32];
                for i in 0..n_chunks {
                    let slice = |desc: &MemoryEventsDesc, n: usize| -> Vec<EventSlice> {
                        let from = (i * m).min(n);
                        let to = ((i + 1) * m).min(n);
                        if from == to {
                            Vec::new()
                        } else {
                            vec![EventSlice { artifact: desc.artifact.clone(), from: from as u32, to: to as u32 }]
                        }
                    };
                    let init_slice = slice(&init, n_init);
                    let fin_slice = slice(&fin, n_fin);
                    let previous_init_addr_bits = init_bits;
                    if !init_slice.is_empty() {
                        init_bits = bits(init.boundary_addrs[i]);
                    }
                    let previous_finalize_addr_bits = fin_bits;
                    if !fin_slice.is_empty() {
                        fin_bits = bits(fin.boundary_addrs[i]);
                    }
                    shards.push(DeferredShardPlan::Memory {
                        init: init_slice,
                        finalize: fin_slice,
                        previous_init_addr_bits,
                        last_init_addr_bits: init_bits,
                        previous_finalize_addr_bits,
                        last_finalize_addr_bits: fin_bits,
                    });
                }
                if deferred_census_enabled() {
                    tracing::warn!(
                        "DEFERRED_PLAN memory shards={n_chunks} init_events={n_init} finalize_events={n_fin} chunk={m}"
                    );
                }
            }
        }
        if deferred_census_enabled() && !shards.is_empty() {
            // One line per syscall code cut in this call: how many precompile shards the
            // deferred-split thresholds produce (the executor's `SHARD_CLOSE` census covers
            // only execution shards, so together they account for every core shard).
            let mut per_code: BTreeMap<String, usize> = BTreeMap::new();
            for shard in &shards {
                if let DeferredShardPlan::Precompile { code, .. } = shard {
                    *per_code.entry(format!("{code:?}")).or_insert(0) += 1;
                }
            }
            for (code, n) in per_code {
                tracing::warn!("DEFERRED_PLAN precompile code={code} shards={n} last={last}");
            }
        }
        shards
    }

    fn cut(artifacts: &[String], code: SyscallCode, q: &mut VecDeque<PendingEvent>, n: usize) -> DeferredShardPlan {
        let mut slices: Vec<EventSlice> = Vec::new();
        for e in q.drain(..n) {
            match slices.last_mut() {
                Some(s) if s.artifact == artifacts[e.artifact as usize] && s.to == e.index => s.to += 1,
                _ => slices.push(EventSlice {
                    artifact: artifacts[e.artifact as usize].clone(),
                    from: e.index,
                    to: e.index + 1,
                }),
            }
        }
        DeferredShardPlan::Precompile { code, slices }
    }

    /// Whether anything is still pending (a non-terminal remainder).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.memory.is_none()
    }
}

impl ExecutionRecord {
    /// The record `split` builds for one precompile shard.
    #[must_use]
    pub fn precompile_shard(
        program: std::sync::Arc<Program>,
        code: SyscallCode,
        events: Vec<(SyscallEvent, PrecompileEvent)>,
    ) -> Self {
        let mut record = ExecutionRecord::new(program);
        record.precompile_events.insert(code, events);
        record
    }

    /// The record `split` builds for one global-memory shard (the public
    /// values' address bits are the caller's — they come from the plan).
    #[must_use]
    pub fn memory_shard(
        program: std::sync::Arc<Program>,
        init: Vec<MemoryInitializeFinalizeEvent>,
        finalize: Vec<MemoryInitializeFinalizeEvent>,
    ) -> Self {
        let mut record = ExecutionRecord::new(program);
        record.global_memory_initialize_events = init;
        record.global_memory_finalize_events = finalize;
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> SplitOpts {
        SplitOpts::new(8)
    }

    #[test]
    fn count_cut_matches_chunks_exact_across_pushes() {
        // deferred threshold 8: 5 + 7 + 4 = 16 events => two full shards,
        // the second spanning all three artifacts; nothing left.
        let mut p = DeferredPlanner::new(opts());
        p.push_precompile(SyscallCode::SECP256K1_ADD, "a", vec![1; 5]);
        assert!(p.split(false).is_empty());
        p.push_precompile(SyscallCode::SECP256K1_ADD, "b", vec![1; 7]);
        let s = p.split(false);
        assert_eq!(
            s,
            vec![DeferredShardPlan::Precompile {
                code: SyscallCode::SECP256K1_ADD,
                slices: vec![
                    EventSlice { artifact: "a".into(), from: 0, to: 5 },
                    EventSlice { artifact: "b".into(), from: 0, to: 3 }
                ]
            }]
        );
        p.push_precompile(SyscallCode::SECP256K1_ADD, "c", vec![1; 4]);
        let s = p.split(true);
        assert_eq!(
            s,
            vec![DeferredShardPlan::Precompile {
                code: SyscallCode::SECP256K1_ADD,
                slices: vec![
                    EventSlice { artifact: "b".into(), from: 3, to: 7 },
                    EventSlice { artifact: "c".into(), from: 0, to: 4 }
                ]
            }]
        );
        assert!(p.is_empty());
    }

    #[test]
    fn weight_cut_is_greedy_like_split() {
        // keccak threshold = 8*8/24 = 2 blocks. Weights 1,1,1,2,2 =>
        // [1,1] [1] [2] [2]: a shard closes when the next event would
        // overflow, and an oversized single event still forms a shard.
        let mut p = DeferredPlanner::new(opts());
        p.push_precompile(SyscallCode::KECCAK_SPONGE, "k", vec![1, 1, 1, 2, 2]);
        let s = p.split(false);
        let ranges: Vec<(u32, u32)> = s
            .iter()
            .map(|x| match x {
                DeferredShardPlan::Precompile { slices, .. } => (slices[0].from, slices[0].to),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(ranges, vec![(0, 2), (2, 3), (3, 4)]);
        // The last event (weight 2) is the remainder; ships on the last call.
        let s = p.split(true);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn memory_shards_carry_split_addr_bits() {
        // memory threshold = 64*8 = 512; 1000 init, 600 finalize => 2 shards
        // (zip_longest), bits from the boundary addresses.
        let mut p = DeferredPlanner::new(opts());
        let init = MemoryEventsDesc { artifact: "i".into(), count: 1000, boundary_addrs: vec![0x1ff, 0x3e7] };
        let fin = MemoryEventsDesc { artifact: "f".into(), count: 600, boundary_addrs: vec![0x7ff, 0xfff] };
        p.push_memory(init, fin);
        let s = p.split(true);
        assert_eq!(s.len(), 2);
        let bits = |addr: u32| -> [u32; 32] { core::array::from_fn(|i| (addr >> i) & 1) };
        match &s[0] {
            DeferredShardPlan::Memory { init, finalize, previous_init_addr_bits, last_init_addr_bits, previous_finalize_addr_bits, last_finalize_addr_bits } => {
                assert_eq!(init[0].to - init[0].from, 512);
                assert_eq!(finalize[0].to - finalize[0].from, 512);
                assert_eq!(*previous_init_addr_bits, [0; 32]);
                assert_eq!(*last_init_addr_bits, bits(0x1ff));
                assert_eq!(*previous_finalize_addr_bits, [0; 32]);
                assert_eq!(*last_finalize_addr_bits, bits(0x7ff));
            }
            _ => unreachable!(),
        }
        match &s[1] {
            DeferredShardPlan::Memory { init, finalize, previous_init_addr_bits, last_init_addr_bits, previous_finalize_addr_bits, last_finalize_addr_bits } => {
                assert_eq!((init[0].from, init[0].to), (512, 1000));
                assert_eq!((finalize[0].from, finalize[0].to), (512, 600));
                assert_eq!(*previous_init_addr_bits, bits(0x1ff));
                assert_eq!(*last_init_addr_bits, bits(0x3e7));
                assert_eq!(*previous_finalize_addr_bits, bits(0x7ff));
                assert_eq!(*last_finalize_addr_bits, bits(0xfff));
            }
            _ => unreachable!(),
        }
    }
}
