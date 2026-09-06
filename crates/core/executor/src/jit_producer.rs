//! Native minimal-trace producer: the parent's `execute_minimal` runs the
//! guest as one program-wide JIT function instead of the interpreter.
//!
//! SP1's parent is its JIT (`sp1_jit`): native code executes the program,
//! keeps the per-shard budgets in registers and hands the executor exactly
//! the interpreter's side effects — the flat memory, the register records,
//! the clock, the oracle of pre-access values and the shard-split
//! accounting. This module is the executor half of that design; the native
//! half is [`zkm_core_jit::backends::x86::producer`].
//!
//! Division of labour:
//!
//! * **Native** (`build_producer`): every ALU / load / store / branch /
//!   jump / misc instruction whose lowering matches the interpreter bit for
//!   bit. Per instruction it stamps the registers the interpreter's
//!   `rr`/`rw` stamp, bumps the clock by 5, pushes the pre-access value of
//!   every memory word read or written to the oracle, charges the trace
//!   area / chip heights / touched addresses and fences the shard where
//!   [`Executor::inc_shard_if_need`] would.
//! * **Interpreter** (through [`producer_handler`]): syscalls, statically
//!   invalid encodings and operand forms the lowering does not model. The
//!   handler syncs the native state into the executor, runs
//!   [`Executor::execute_cycle`] for that one instruction and syncs back.
//! * **Host driver** ([`run`]): shard fences (`inc_shard_if_need` +
//!   `bump_record`, exactly the interpreter loop's bookkeeping), oracle
//!   growth, program end.
//!
//! The chunks the producer emits are byte-identical to the interpreter's
//! (`tests::producer_matches_interpreter_*`); the parent's `execute_minimal`
//! uses it whenever the executor is in the plain minimal-trace configuration
//! (see `platform::eligible`), with no switch — SP1 has none either.
//!
//! Known, deliberate divergences from the interpreter (all shared with the
//! block JIT, none reachable by a well-formed guest): a taken branch to pc 0
//! falls through instead of ending the program; per-opcode counts in the
//! `ExecutionReport` are folded into `ADD` (the total is exact); an
//! out-of-range load traps into the interpreter, which panics on the flat
//! memory bound like the interpreter alone would.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
use crate::{ExecutionError, Executor};

/// Batches the producer has taken, for tests and for the one-line
/// `PRODUCER` census a caller can log. Not load-bearing.
#[doc(hidden)]
pub static PRODUCER_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod platform {
    use std::{
        collections::HashMap,
        hash::{Hash, Hasher},
        ptr::NonNull,
        sync::{Arc, Mutex, OnceLock},
    };

    use zkm_core_jit::{
        backends::x86::{
            producer::{
                build_producer, producer_reject, runs_in_interpreter, HeightCharge, ProducerConfig,
                ProducerInstr, ProducerReject, MAX_CHARGES, MAX_STAMPS, POS_A, POS_B, POS_C,
                POS_HI,
            },
            JIT_EXIT_BAD_JUMP, JIT_EXIT_FALL_OFF, JIT_EXIT_HOST, JIT_EXIT_ORACLE_FULL,
            JIT_EXIT_SHARD_FENCE,
        },
        context::PRODUCER_HEIGHT_SLOTS,
        driver::DriverInstruction,
        JitContext, JitFunction,
    };

    use crate::{
        events::{MemoryAccessPosition, MemoryRecord, NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC},
        executor::{charge_instruction, CORE_SHARD_CLK_LIMIT},
        jit_runner::{program_fingerprint_of, to_driver_instruction},
        minimal_trace::MemValue,
        program::MAX_MEMORY,
        ExecutionError, Executor, ExecutorMode, Instruction, MipsAirId, Opcode, Program, Register,
        NUM_REGISTERS,
    };

    const _: () = {
        assert!(POS_C == MemoryAccessPosition::C as u8);
        assert!(POS_B == MemoryAccessPosition::B as u8);
        assert!(POS_A == MemoryAccessPosition::A as u8);
        assert!(POS_HI == MemoryAccessPosition::HI as u8);
        // The native touch stub charges one `MemoryLocal` row per 4 touched addresses.
        assert!(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC == 4);
    };

    /// Size of one oracle entry: the native push copies the leading
    /// `{value, timestamp, shard}` of a flat entry.
    const ORACLE_ENTRY: usize = std::mem::size_of::<MemValue>();

    /// Host state reachable from the native code through `ctx.user_data`.
    struct Bridge<'a> {
        exec: *mut Executor<'a>,
        /// `state.clk` at the last sync-out; the number of natively executed
        /// instructions since is `(clk_now - clk_base) / 5`.
        clk_base: u32,
        /// Start of the oracle `Vec`'s buffer at the last sync-out.
        oracle_base: *mut u8,
        /// `state.memory.registers` right after `ENTER_UNCONSTRAINED`. The
        /// interpreter rolls every register the block touched back through
        /// `memory_diff`; the native code keeps no diff, so the whole file is
        /// restored from here at `EXIT_UNCONSTRAINED` (except `V0`, which the
        /// exit syscall itself writes afterwards).
        snapshot: Option<Vec<Option<MemoryRecord>>>,
        error: Option<ExecutionError>,
        done: bool,
        fence: bool,
    }

    /// Bring the native state back into the executor. Idempotent: calling it
    /// again without native progress in between changes nothing.
    fn sync_in(exec: &mut Executor<'_>, ctx: &JitContext, br: &mut Bridge<'_>) {
        let regs = &mut exec.state.memory.registers.registers;
        for i in 0..NUM_REGISTERS {
            let value = if i == 0 { 0 } else { ctx.registers[i] };
            let stamp = ctx.reg_stamps[i];
            if stamp != 0 {
                regs[i] = Some(MemoryRecord {
                    shard: (stamp >> 32) as u32,
                    timestamp: stamp as u32,
                    value,
                });
            } else if let Some(r) = regs[i].as_mut() {
                r.value = value;
            }
        }
        let clk_now = (ctx.clk_shard as u32).wrapping_sub(3);
        let executed = u64::from(clk_now.wrapping_sub(br.clk_base) / 5);
        exec.state.clk = clk_now;
        exec.state.global_clk += executed;
        if !exec.unconstrained {
            // The interpreter counts every constrained instruction by opcode;
            // the native code counts them once. `total_instruction_count`
            // stays exact.
            exec.report.opcode_counts[Opcode::ADD] += executed;
            exec.split_acct.import_budgets(ctx.area_left, &ctx.height_left, ctx.touched);
        }
        let len = (ctx.oracle_tail as usize - br.oracle_base as usize) / ORACLE_ENTRY;
        debug_assert!(len <= exec.recording_chunk_mem_reads.capacity());
        // SAFETY: the native code wrote `len` complete entries starting at
        // the buffer the last sync-out handed it (`oracle_base`), staying
        // below `oracle_end`, which sits inside the capacity.
        unsafe { exec.recording_chunk_mem_reads.set_len(len) };
        br.clk_base = clk_now;
    }

    /// The post-instruction `clk` at which the interpreter's `cpu_exit ||
    /// clk_exit` (see `inc_shard_if_need`) first holds, plus 3 so it compares
    /// directly against the pinned `clk + 3`.
    fn clk_limit(exec: &Executor<'_>) -> u32 {
        let msc = exec.max_syscall_cycles;
        let cpu = exec.shard_size.saturating_sub(msc);
        let clk = CORE_SHARD_CLK_LIMIT.saturating_sub(msc + MemoryAccessPosition::HI as u32);
        cpu.min(clk).saturating_add(3)
    }

    /// Hand the executor state to the native code.
    fn sync_out(exec: &mut Executor<'_>, ctx: &mut JitContext, br: &mut Bridge<'_>) {
        ctx.pc = exec.state.pc;
        let regs = &exec.state.memory.registers.registers;
        for i in 0..NUM_REGISTERS {
            let (value, stamp) = match regs[i] {
                None => (0, 0),
                Some(r) if r.shard == 0 && r.timestamp == 0 => (r.value, 0),
                Some(r) => (r.value, (u64::from(r.shard) << 32) | u64::from(r.timestamp)),
            };
            ctx.registers[i] = value;
            ctx.reg_stamps[i] = stamp;
        }
        ctx.registers[0] = 0;
        let shard = exec.state.current_shard;
        let clk = exec.state.clk;
        ctx.shard = shard;
        ctx.clk_shard = (u64::from(shard) << 32) | u64::from(clk.wrapping_add(3));
        br.clk_base = clk;
        if exec.unconstrained {
            // Nothing inside an unconstrained block is charged or fenced.
            ctx.clk_limit = u32::MAX;
            ctx.area_left = i64::MAX / 2;
            ctx.height_left.fill(i64::MAX / 2);
        } else {
            ctx.clk_limit = clk_limit(exec);
            exec.split_acct.export_budgets(
                &mut ctx.area_left,
                &mut ctx.height_left,
                &mut ctx.touched,
            );
        }
        // The active view changes at ENTER/EXIT (copy-on-write inside a block).
        let flat = exec.flat_mem.as_deref().expect("producer runs on the flat memory");
        ctx.memory = NonNull::new(flat.as_ptr().cast::<u8>());
        let oracle = &mut exec.recording_chunk_mem_reads;
        if oracle.capacity() == oracle.len() {
            oracle.reserve(oracle.len().max(1 << 16));
        }
        let base = oracle.as_mut_ptr().cast::<u8>();
        br.oracle_base = base;
        // SAFETY: offsets inside (or one past) the allocation.
        ctx.oracle_tail = unsafe { base.add(oracle.len() * ORACLE_ENTRY) };
        ctx.oracle_end = unsafe { base.add((oracle.capacity() - 1) * ORACLE_ENTRY) };
        // NOT `exit_code`: the handler sets it to stop the native code and
        // then syncs out, so clearing it here would swallow the request and
        // let the guest run on past a fence or a halt. `run` clears it before
        // each entry instead.
    }

    fn is_control_flow(ins: &Instruction) -> bool {
        ins.is_branch_instruction() || ins.is_jump_instruction()
    }

    /// `pc` is the delay slot of a branch/jump.
    fn is_delay_slot(program: &Program, pc: u32) -> bool {
        let idx = pc.wrapping_sub(program.pc_base) / 4;
        idx >= 1 && program.instructions.get(idx as usize - 1).is_some_and(is_control_flow)
    }

    /// The interpreter's end-of-program test on the current pc (`exited` is
    /// only ever set by a syscall, which runs in the interpreter).
    fn program_done(exec: &Executor<'_>) -> bool {
        let pc = exec.state.pc;
        pc == 0
            || pc.wrapping_sub(exec.program.pc_base) >= (exec.program.instructions.len() * 4) as u32
    }

    /// The native trap site: run one instruction in the interpreter.
    extern "C" fn producer_handler(ctx: *mut JitContext) -> u64 {
        // SAFETY: `run` installs `user_data` and keeps the bridge and the
        // executor alive for the whole native call.
        let ctx = unsafe { &mut *ctx };
        let br = unsafe { &mut *ctx.user_data.cast::<Bridge<'_>>() };
        let exec = unsafe { &mut *br.exec };
        sync_in(exec, ctx, br);
        let pc = ctx.pc;
        let pending = std::mem::replace(&mut ctx.pending_jump_at_start, 0);
        exec.state.pc = pc;
        exec.state.next_pc = if pending != 0 && is_delay_slot(&exec.program, pc) {
            pending
        } else {
            pc.wrapping_add(4)
        };
        let was_unconstrained = exec.unconstrained;
        match exec.execute_cycle() {
            Err(e) => {
                br.error = Some(e);
                ctx.exit_code = JIT_EXIT_HOST;
                return 1;
            }
            Ok(true) => {
                br.done = true;
                ctx.exit_code = JIT_EXIT_HOST;
            }
            Ok(false) => {
                if !exec.unconstrained && exec.shard_fence_due() {
                    br.fence = true;
                    ctx.exit_code = JIT_EXIT_HOST;
                }
            }
        }
        if was_unconstrained && !exec.unconstrained {
            if let Some(snapshot) = br.snapshot.take() {
                let regs = &mut exec.state.memory.registers.registers;
                for (i, r) in snapshot.into_iter().enumerate() {
                    if i != Register::V0 as usize {
                        regs[i] = r;
                    }
                }
            }
        }
        sync_out(exec, ctx, br);
        if !was_unconstrained && exec.unconstrained {
            br.snapshot = Some(exec.state.memory.registers.registers.clone());
        }
        0
    }

    /// Whether the executor is in the configuration the producer models:
    /// the parent's minimal-trace run on the flat memory, with the
    /// production shard limits only.
    fn eligible(exec: &Executor<'_>) -> bool {
        exec.executor_mode == ExecutorMode::Simple
            && exec.minimal_trace_collector.is_some()
            && exec.flat_mem.is_some()
            && exec.replay_mem.is_none()
            && !exec.skip_replay_bookkeeping
            && !exec.force_interpreter
            && exec.max_cycles.is_none()
            && exec.maximal_shapes.is_none()
            && !exec.lde_size_check
            && exec.shard_batch_size > 0
            && exec.split_acct.cost(MipsAirId::Cpu) == 0
            && !exec.unconstrained
            && !exec.state.exited
            && exec.state.next_pc == exec.state.pc.wrapping_add(4)
    }

    enum Built {
        Fn(Arc<JitFunction>),
        Rejected(ProducerReject),
        Unavailable(String),
    }

    /// Process-wide producers keyed by program and chip widths (the widths
    /// decide the per-instruction area charges).
    static PRODUCER_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Built>>>> = OnceLock::new();

    fn cache_key(exec: &Executor<'_>) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        program_fingerprint_of(&exec.program).hash(&mut h);
        for i in 0..<MipsAirId as enum_map::Enum>::LENGTH {
            exec.split_acct.cost(<MipsAirId as enum_map::Enum>::from_usize(i)).hash(&mut h);
        }
        h.finish()
    }

    fn cached_producer(exec: &Executor<'_>) -> Option<Arc<JitFunction>> {
        let cache = PRODUCER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().expect("producer cache poisoned");
        let built = guard.entry(cache_key(exec)).or_insert_with(|| Arc::new(build(exec))).clone();
        drop(guard);
        match &*built {
            Built::Fn(f) => Some(f.clone()),
            Built::Rejected(reject) => {
                tracing::debug!("minimal-trace producer declined: {reject:?}");
                None
            }
            Built::Unavailable(why) => {
                tracing::debug!("minimal-trace producer unavailable: {why}");
                None
            }
        }
    }

    fn build(exec: &Executor<'_>) -> Built {
        assert!(
            <MipsAirId as enum_map::Enum>::LENGTH <= PRODUCER_HEIGHT_SLOTS,
            "JitContext::height_left has too few slots for MipsAirId"
        );
        let program = &exec.program;
        let instrs: Vec<DriverInstruction> =
            program.instructions.iter().map(to_driver_instruction).collect();
        if let Some(reject) = producer_reject(&instrs) {
            tracing::warn!("minimal-trace producer: program rejected ({reject:?}); interpreting");
            return Built::Rejected(reject);
        }
        let plans = build_plans(&exec.split_acct, &program.instructions, &instrs);
        let cfg = ProducerConfig {
            pc_base: program.pc_base,
            global_slot: slot(MipsAirId::Global),
            global_cost: exec.split_acct.cost(MipsAirId::Global) as i64,
            memory_local_slot: slot(MipsAirId::MemoryLocal),
            memory_local_cost: exec.split_acct.cost(MipsAirId::MemoryLocal) as i64,
            max_memory: MAX_MEMORY as u32,
            syscall_handler: producer_handler,
        };
        match build_producer(&instrs, &plans, &cfg) {
            Ok(f) => Built::Fn(Arc::new(f)),
            Err(e) => {
                tracing::warn!("minimal-trace producer: build failed ({e}); interpreting");
                Built::Unavailable(e.to_string())
            }
        }
    }

    fn slot(air: MipsAirId) -> u8 {
        u8::try_from(crate::ShardSplitAccumulator::slot(air)).expect("air slot fits a byte")
    }

    /// One plan per instruction: what the charging block and the register
    /// access pattern of `execute_operation` do for it.
    fn build_plans(
        acct: &crate::ShardSplitAccumulator,
        instrs: &[Instruction],
        driver: &[DriverInstruction],
    ) -> Vec<ProducerInstr> {
        let mut scratch = acct.clone();
        let mut plans = Vec::with_capacity(instrs.len());
        for (ins, d) in instrs.iter().zip(driver) {
            let mut plan = ProducerInstr::default();
            if !runs_in_interpreter(d) {
                scratch.reset();
                let base_counts = scratch.event_counts(0);
                let base_area = scratch.trace_area(0);
                charge_instruction(&mut scratch, ins);
                plan.area = (scratch.trace_area(0) - base_area) as i64;
                for (air, &count) in &scratch.event_counts(0) {
                    let delta = count - base_counts[air];
                    if air == MipsAirId::Cpu || delta == 0 {
                        continue;
                    }
                    let n = plan.n_heights as usize;
                    assert!(n < MAX_CHARGES, "{ins:?} charges more than {MAX_CHARGES} chips");
                    plan.heights[n] = HeightCharge {
                        slot: slot(air),
                        count: u8::try_from(delta).expect("per-instruction row charge"),
                    };
                    plan.n_heights += 1;
                }
                plan_stamps(ins, &mut plan);
            }
            plans.push(plan);
        }
        for i in 1..plans.len() {
            if is_control_flow(&instrs[i - 1]) {
                let pred = plans[i - 1];
                let cur = &mut plans[i];
                cur.delay_slot = true;
                cur.n_pred = pred.n_heights;
                for (dst, charge) in cur.pred_slots.iter_mut().zip(&pred.heights) {
                    *dst = charge.slot;
                }
            }
        }
        plans
    }

    /// The registers `execute_operation` stamps for a natively run
    /// instruction, in the interpreter's access order; a register accessed
    /// twice keeps its last (highest) position.
    fn plan_stamps(ins: &Instruction, plan: &mut ProducerInstr) {
        let mut push = |reg: u32, pos: u8| {
            let reg = u8::try_from(reg).expect("register operand");
            assert!(usize::from(reg) < NUM_REGISTERS, "{ins:?}: register {reg}");
            let n = plan.n_stamps as usize;
            for s in &mut plan.stamps[..n] {
                if s.0 == reg {
                    s.1 = s.1.max(pos);
                    return;
                }
            }
            assert!(n < MAX_STAMPS, "{ins:?} stamps more than {MAX_STAMPS} registers");
            plan.stamps[n] = (reg, pos);
            plan.n_stamps += 1;
        };
        let (a, b, c) = (u32::from(ins.op_a), ins.op_b, ins.op_c);
        let (lo, hi) = (Register::LO as u32, Register::HI as u32);
        match ins.opcode {
            _ if ins.is_alu_instruction() => {
                if !ins.imm_c {
                    push(c, POS_C);
                    push(b, POS_B);
                } else if !ins.imm_b {
                    push(b, POS_B);
                }
                if ins.opcode.is_use_lo_hi_alu() {
                    push(lo, POS_A);
                    push(hi, POS_HI);
                } else {
                    push(a, POS_A);
                }
            }
            Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                push(c, POS_C);
                push(b, POS_B);
                push(a, POS_A);
                push(hi, POS_HI);
            }
            Opcode::SEXT | Opcode::WSBH | Opcode::EXT | Opcode::INS | Opcode::TEQ => {
                push(b, POS_B);
                push(a, POS_A);
            }
            Opcode::MEQ | Opcode::MNE => {
                push(c, POS_C);
                push(b, POS_B);
                push(a, POS_A);
            }
            _ if ins.is_memory_load_instruction() || ins.is_memory_store_instruction() => {
                push(b, POS_B);
                push(a, POS_A);
            }
            _ if ins.is_branch_instruction() => {
                if !ins.imm_b {
                    push(b, POS_B);
                }
                push(a, POS_A);
            }
            Opcode::Jump => {
                push(b, POS_B);
                push(a, POS_A);
            }
            Opcode::Jumpi | Opcode::JumpDirect => push(a, POS_A),
            // Syscalls and invalid encodings run in the interpreter, so they
            // never reach here; the arms above cover every other opcode.
            Opcode::SYSCALL | Opcode::UNIMPL => {}
            _ => unreachable!("{ins:?}: opcode with no stamp plan"),
        }
    }

    /// Close the shard the way the interpreter loop does; `true` when the
    /// batch is complete.
    fn fence(exec: &mut Executor<'_>, num_shards_executed: &mut u32) -> bool {
        let closed = exec.inc_shard_if_need();
        assert!(
            closed,
            "producer fenced at pc {:#x} where the interpreter would not close the shard",
            exec.state.pc
        );
        *num_shards_executed += 1;
        exec.bump_record();
        *num_shards_executed >= exec.shard_batch_size
    }

    fn finish(exec: &Executor<'_>) -> Result<Option<bool>, ExecutionError> {
        if exec.unconstrained {
            Err(ExecutionError::EndInUnconstrained())
        } else {
            Ok(Some(true))
        }
    }

    /// Run the producer from the current state until the program ends
    /// (`Some(true)`) or the batch is complete (`Some(false)`), or `None`
    /// when this executor configuration is not the producer's — the caller
    /// then interprets.
    pub(crate) fn run(
        exec: &mut Executor<'_>,
        num_shards_executed: &mut u32,
    ) -> Result<Option<bool>, ExecutionError> {
        if !eligible(exec) {
            return Ok(None);
        }
        let Some(jit_fn) = cached_producer(exec) else {
            return Ok(None);
        };
        super::PRODUCER_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut ctx = Box::new(JitContext::default());
        ctx.jump_table = NonNull::new(jit_fn.jump_table.as_ptr().cast_mut());
        ctx.jump_table_len = u32::try_from(jit_fn.jump_table.len()).expect("program size");
        let mut br = Bridge {
            exec: std::ptr::from_mut(exec),
            clk_base: 0,
            oracle_base: std::ptr::null_mut(),
            snapshot: None,
            error: None,
            done: false,
            fence: false,
        };
        ctx.user_data = std::ptr::from_mut(&mut br).cast();
        sync_out(exec, &mut ctx, &mut br);
        loop {
            ctx.exit_code = 0;
            // SAFETY: `ctx` points at live memory, the jump table and the
            // oracle buffer (refreshed by every sync-out), and `user_data`
            // at the bridge over this executor for the whole call.
            unsafe { jit_fn.call(&mut *ctx) };
            if let Some(e) = br.error.take() {
                return Err(e);
            }
            let code = ctx.exit_code;
            sync_in(exec, &ctx, &mut br);
            match code {
                JIT_EXIT_SHARD_FENCE => {
                    exec.state.pc = ctx.pc;
                    exec.state.next_pc = ctx.pc.wrapping_add(4);
                    if program_done(exec) {
                        return finish(exec);
                    }
                    if fence(exec, num_shards_executed) {
                        return Ok(Some(false));
                    }
                }
                JIT_EXIT_HOST => {
                    if br.done {
                        return Ok(Some(true));
                    }
                    assert!(br.fence, "producer host exit without a reason at pc {:#x}", ctx.pc);
                    br.fence = false;
                    if fence(exec, num_shards_executed) {
                        return Ok(Some(false));
                    }
                }
                JIT_EXIT_ORACLE_FULL => {
                    exec.state.pc = ctx.pc;
                    exec.state.next_pc = ctx.pc.wrapping_add(4);
                    let len = exec.recording_chunk_mem_reads.len();
                    exec.recording_chunk_mem_reads.reserve(len.max(1 << 16));
                }
                JIT_EXIT_FALL_OFF => {
                    exec.state.pc = ctx.pc;
                    return finish(exec);
                }
                JIT_EXIT_BAD_JUMP => {
                    exec.state.pc = ctx.bad_jump_target;
                    return finish(exec);
                }
                other => panic!("producer: unexpected exit {other:#x} at pc {:#x}", ctx.pc),
            }
            sync_out(exec, &mut ctx, &mut br);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub(crate) use platform::run;

/// No native producer on this platform: the interpreter runs everything.
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
pub(crate) fn run(
    _exec: &mut Executor<'_>,
    _num_shards_executed: &mut u32,
) -> Result<Option<bool>, ExecutionError> {
    Ok(None)
}
