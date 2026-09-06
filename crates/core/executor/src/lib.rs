mod air;
mod context;
mod cost;
/// Deferred-shard planning over event references (the controller's mirror
/// of `ExecutionRecord::split`).
pub mod deferred_plan;
pub mod events;
mod executor;
/// Flat guest memory for the minimal-trace producer (one 16-byte
/// `{value, timestamp, shard}` entry per word, COW unconstrained blocks).
pub mod flat_mem;
pub mod hook;
mod instruction;
mod io;
/// Native minimal-trace producer: the parent runs the guest as one JIT
/// function and the interpreter only handles the instructions it does not
/// lower. See the module docs.
pub mod jit_producer;
/// JIT runner — bridges the executor's [`Instruction`] stream and
/// runtime state to the [`zkm_core_jit`] driver and `JitFunction`
/// execution.
///
/// On Linux x86_64 the module exposes `build_jit_function`,
/// `build_context`, and `run_jit` so a caller can transpile a program
/// once and re-execute it many times.  On other platforms only the
/// portable surface (`to_driver_instruction`,
/// `instructions_to_driver_stream`, and `jit_unavailable`) remains —
/// the JIT itself isn't available and callers should fall back to
/// [`Executor::run`].
pub mod jit_runner;
pub mod memory;
/// Minimal-trace skeleton for the MinimalTrace + TracingVM split.
/// Defines the per-shard checkpoint format the JIT
/// emits and the TracingVM consumes. See module docs.
pub mod minimal_trace;
mod opcode;
mod program;
#[cfg(test)]
pub mod programs;
mod record;
pub mod reduce;
mod register;
pub mod report;
mod state;
pub mod subproof;
pub mod syscalls;
/// TracingVM scaffold + parallel driver. Consumes
/// `MinimalTrace` chunks and produces full `ExecutionRecord`s — one
/// per shard, rayon-parallelised across cores. See module docs.
pub mod tracing_vm;
mod utils;

pub use air::*;
pub use context::*;
pub use cost::*;
pub use executor::*;
pub use hook::*;
pub use instruction::*;
pub use opcode::*;
pub use program::*;
pub use record::*;
pub use reduce::*;
pub use register::*;
pub use report::*;
pub use state::*;
pub use subproof::*;
pub use utils::*;

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum OptionValTag {
    Some = 0,
    None,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct OptionU32 {
    pub tag: OptionValTag,
    pub value: u32,
}

impl From<Option<u32>> for OptionU32 {
    fn from(val: Option<u32>) -> Self {
        match val {
            Some(value) => OptionU32 { tag: OptionValTag::Some, value },
            None => OptionU32 { tag: OptionValTag::None, value: 0 },
        }
    }
}
