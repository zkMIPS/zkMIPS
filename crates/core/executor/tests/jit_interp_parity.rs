//! Parity test: identical ALU-only programs must produce identical
//! final register state when run through the interpreter and through
//! the JIT.
//!
//! This is the floor of the "JIT by default" guarantee — anything
//! less and switching `Executor::run_fast`'s default would silently
//! produce wrong results.  Any opcode the JIT supports must produce
//! the same final register file as the interpreter on the same input.
//!
//! Workload: a deterministic chain of ADD/SUB/AND/OR/XOR/SLL/SRL on
//! T0–T7 that exercises both register-register and register-immediate
//! paths.  No memory, no syscalls — those land in (c) and (d) of the
//! JIT-by-default plan and have their own parity tests.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use zkm_core_executor::jit_runner::{
    build_context, build_jit_function, first_unsupported_opcode, run_jit, BuildParams,
};
use zkm_core_executor::{ExecutionError, Executor, Instruction, Opcode, Program, Register};
use zkm_pcs::ZKMCoreOpts;

/// End-to-end parity check on the real fibonacci ELF.
///
/// Loads the prebuilt MIPS ELF, runs `Executor::run_fast` twice — once
/// with `ZIREN_DISABLE_JIT=1` forcing the interpreter and once with
/// the JIT-by-default path — and asserts the resulting
/// `state.public_values_stream` matches byte-for-byte.  This is the
/// real proof that the memory bridge + syscall trampoline correctly
/// thread COMMIT-style syscalls through.
///
/// Skipped when the ELF isn't built (e.g. fresh checkout that never
/// ran `cargo run -p fibonacci-host`).
#[test]
fn real_fibonacci_elf_jit_matches_interpreter() {
    real_elf_parity(
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/fibonacci",
        Some(5u32.to_le_bytes().to_vec()),
    );
}

/// fib(1000) — exercises ~40k cycles, so any opcode that fires only
/// in extended runs will surface here.
#[test]
fn real_fibonacci_n1000_elf_jit_matches_interpreter() {
    real_elf_parity(
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/fibonacci",
        Some(1000u32.to_le_bytes().to_vec()),
    );
}

#[test]
fn real_large_sum_elf_jit_matches_interpreter() {
    real_elf_parity(
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/large-sum",
        None,
    );
}

#[test]
fn real_json_elf_jit_matches_interpreter() {
    real_elf_parity(
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/json",
        None,
    );
}

#[test]
fn real_keccak_elf_jit_matches_interpreter() {
    real_elf_parity(
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/keccak",
        None,
    );
}

// shape-bin / external-fixture parity tests removed — use the
// regular `examples/<name>/host` runners (which thread inputs from
// each example's host directory and exercise the JIT-by-default
// `Executor::run_fast` path end-to-end).

/// Parity test for DIVU + MFLO + MFHI: critical for the format
/// machinery's digit conversion (which divides by 10 repeatedly).
/// Uses register operands (imm_b=false, imm_c=false) — driver
/// dispatches to `t.divu(rs, rt)` then we read both Lo and Hi.
/// Builds enough instructions to clear `JIT_MIN_INSTR_COUNT`.
#[test]
fn divu_mfhi_mflo_jit_matches_interpreter() {
    use zkm_core_executor::Opcode;
    let mut instrs = Vec::with_capacity(700);
    // T0 = 12345
    instrs.push(Instruction::new(Opcode::ADD, Register::T0 as u8, 0, 12345, false, true));
    // T1 = 10
    instrs.push(Instruction::new(Opcode::ADD, Register::T1 as u8, 0, 10, false, true));
    // For each iter: DIVU T0 / T1 → Lo (quotient), Hi (remainder)
    //                MFLO T2; MFHI T3
    //                T0 = T2  (continue dividing the quotient)
    // After ceil(log10(12345)) ≈ 5 iters, T0 == 0, T2 == 0, T3 == leftmost digit
    // Pad with ADD ops to clear JIT_MIN_INSTR_COUNT (500).
    for _ in 0..5 {
        instrs.push(Instruction::new(
            Opcode::DIVU,
            Register::LO as u8,
            Register::T0 as u32,
            Register::T1 as u32,
            false,
            false,
        ));
        // MFLO  T2 ← Lo  (encoded as ADD T2, $LO, $0)
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T2 as u8,
            Register::LO as u32,
            Register::ZERO as u32,
            false,
            false,
        ));
        // MFHI  T3 ← Hi  (encoded as ADD T3, $HI, $0)
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T3 as u8,
            Register::HI as u32,
            Register::ZERO as u32,
            false,
            false,
        ));
        // T0 = T2 (continue)
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T0 as u8,
            Register::T2 as u32,
            Register::ZERO as u32,
            false,
            false,
        ));
    }
    // Pad with no-ops (ADD T4, T4, 0) to exceed the JIT threshold.
    while instrs.len() < 600 {
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T4 as u8,
            Register::T4 as u32,
            0,
            false,
            true,
        ));
    }
    let program = Program::new(instrs, 0, 0);

    // Interpreter
    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = interp.run_fast();
    let interp_t0 = interp.register(Register::T0);
    let interp_t2 = interp.register(Register::T2);
    let interp_t3 = interp.register(Register::T3);
    std::env::remove_var("ZIREN_DISABLE_JIT");

    // JIT
    let mut jit = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = jit.run_fast();
    let jit_t0 = jit.register(Register::T0);
    let jit_t2 = jit.register(Register::T2);
    let jit_t3 = jit.register(Register::T3);

    eprintln!(
        "[divu] interp T0={interp_t0} T2={interp_t2} T3={interp_t3}; jit T0={jit_t0} T2={jit_t2} T3={jit_t3}"
    );
    assert_eq!(interp_t0, jit_t0, "DIVU loop final T0 mismatch");
    assert_eq!(interp_t2, jit_t2, "DIVU final quotient (MFLO) mismatch");
    assert_eq!(interp_t3, jit_t3, "DIVU final remainder (MFHI) mismatch");
}

/// Parity for MULTU + MFLO + MFHI.
/// `JumpDirect`'s `op_b` is a byte offset RELATIVE to `next_pc`
/// (`execute_jump_direct`: `target_pc = op_b.wrapping_add(next_pc)`), unlike
/// `Jumpi`'s, which is already absolute.  The JIT driver used to hand the raw
/// offset to a lowering that takes an ABSOLUTE target, so every `JumpDirect`
/// jumped to the offset itself.  On a real ELF based well above zero that
/// address is outside the program, the dispatch index wrapped, and the
/// indirect jump left the jump table — taking the host process down with a
/// SIGSEGV rather than failing as a guest error.
///
/// The jump below skips exactly one instruction.  Land in the wrong place and
/// `$t0` keeps the value that instruction writes.
#[test]
fn jump_direct_target_is_relative_to_next_pc() {
    use zkm_core_executor::Opcode;
    let mut instrs = Vec::with_capacity(700);
    // 0: t0 = 111
    instrs.push(Instruction::new(Opcode::ADD, Register::T0 as u8, 0, 111, false, true));
    // 1: JumpDirect, link=$zero, offset=+8 -> target = next_pc + 8 = instr 4.
    instrs.push(Instruction::new(Opcode::JumpDirect, 0, 8, 0, true, true));
    // 2: delay slot — RUNS.
    instrs.push(Instruction::new(Opcode::ADD, Register::T1 as u8, 0, 1, false, true));
    // 3: SKIPPED by the jump.  If the target is wrong this runs.
    instrs.push(Instruction::new(Opcode::ADD, Register::T0 as u8, 0, 999, false, true));
    // 4: landing pad.
    instrs.push(Instruction::new(Opcode::ADD, Register::T2 as u8, 0, 222, false, true));
    // Pad past JIT_MIN_INSTR_COUNT so the JIT actually engages.
    while instrs.len() < 600 {
        instrs.push(Instruction::new(Opcode::ADD, Register::T3 as u8, 0, 0, false, true));
    }
    let program = Program::new(instrs, 0, 0);

    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = interp.run_fast();
    let (i0, i1, i2) = (
        interp.register(Register::T0),
        interp.register(Register::T1),
        interp.register(Register::T2),
    );
    std::env::remove_var("ZIREN_DISABLE_JIT");

    let mut jit = Executor::new(program, ZKMCoreOpts::default());
    let _ = jit.run_fast();
    let (j0, j1, j2) =
        (jit.register(Register::T0), jit.register(Register::T1), jit.register(Register::T2));

    eprintln!("[jump_direct] interp t0={i0} t1={i1} t2={i2}; jit t0={j0} t1={j1} t2={j2}");
    assert_eq!(i0, 111, "interpreter: the jump must skip the t0=999 instruction");
    assert_eq!(i1, 1, "interpreter: the delay slot must run");
    assert_eq!((i0, i1, i2), (j0, j1, j2), "JumpDirect: JIT diverges from the interpreter");
}

#[test]
fn multu_mfhi_mflo_jit_matches_interpreter() {
    use zkm_core_executor::Opcode;
    let mut instrs = Vec::with_capacity(700);
    instrs.push(Instruction::new(Opcode::ADD, Register::T0 as u8, 0, 0xDEAD_BEEFu32, false, true));
    instrs.push(Instruction::new(Opcode::ADD, Register::T1 as u8, 0, 0x1234_5678u32, false, true));
    instrs.push(Instruction::new(
        Opcode::MULTU,
        Register::LO as u8,
        Register::T0 as u32,
        Register::T1 as u32,
        false,
        false,
    ));
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T2 as u8,
        Register::LO as u32,
        Register::ZERO as u32,
        false,
        false,
    ));
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T3 as u8,
        Register::HI as u32,
        Register::ZERO as u32,
        false,
        false,
    ));
    while instrs.len() < 600 {
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T4 as u8,
            Register::T4 as u32,
            0,
            false,
            true,
        ));
    }
    let program = Program::new(instrs, 0, 0);

    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = interp.run_fast();
    let interp_t2 = interp.register(Register::T2);
    let interp_t3 = interp.register(Register::T3);
    std::env::remove_var("ZIREN_DISABLE_JIT");

    let mut jit = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = jit.run_fast();
    let jit_t2 = jit.register(Register::T2);
    let jit_t3 = jit.register(Register::T3);

    eprintln!(
        "[multu] interp lo={interp_t2:#x} hi={interp_t3:#x}; jit lo={jit_t2:#x} hi={jit_t3:#x}"
    );
    assert_eq!(interp_t2, jit_t2, "MULTU lo mismatch");
    assert_eq!(interp_t3, jit_t3, "MULTU hi mismatch");
}

/// Shared driver for "real ELF" parity tests.  `input_bytes` is an
/// optional pre-allocated stdin chunk for SYSHINTREAD; pass `None`
/// when the guest doesn't read input.
fn real_elf_parity(elf_path: &str, input_bytes: Option<Vec<u8>>) {
    let bytes = match std::fs::read(elf_path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[skip] ELF not built at {elf_path}");
            return;
        }
    };
    let program = Program::from(&bytes[..]).expect("parse ELF");
    if let Some(op) = first_unsupported_opcode(&program) {
        eprintln!("[skip] ELF contains JIT-unsupported opcode {op:#x}: {elf_path}");
        return;
    }

    let run_once = |disable_jit: bool| -> (Result<(), ExecutionError>, Vec<u8>, bool) {
        if disable_jit {
            std::env::set_var("ZIREN_DISABLE_JIT", "1");
        } else {
            std::env::remove_var("ZIREN_DISABLE_JIT");
        }
        let mut rt = Executor::new(program.clone(), ZKMCoreOpts::default());
        if let Some(ref data) = input_bytes {
            rt.state.input_stream.push(data.clone());
        }
        let res = rt.run_fast();
        let pvs = rt.state.public_values_stream.clone();
        let exited = rt.state.exited;
        std::env::remove_var("ZIREN_DISABLE_JIT");
        (res, pvs, exited)
    };

    let (interp_res, interp_pvs, interp_exited) = run_once(true);
    let (jit_res, jit_pvs, jit_exited) = run_once(false);

    eprintln!(
        "[{elf_path}] interp: exited={interp_exited} pvs_len={} res={interp_res:?}, jit: exited={jit_exited} pvs_len={} res={jit_res:?}",
        interp_pvs.len(),
        jit_pvs.len(),
    );
    assert_eq!(interp_exited, jit_exited, "exited flag must match for {elf_path}");
    assert_eq!(
        interp_pvs, jit_pvs,
        "public_values_stream must match between interp and JIT on {elf_path}"
    );
}

/// Build a synthetic ALU-only program that touches T0–T7 with a
/// deterministic mix of operations.  Zero memory traffic, zero
/// syscalls — both runtimes should agree on the final register file.
fn build_alu_chain(num_ops: usize) -> Program {
    let mut instrs = Vec::with_capacity(num_ops + 8);
    // Seed: t0 = 1, t1 = 2, t2 = 3, ... t7 = 8 via ADDi from $zero.
    for k in 0u8..8 {
        instrs.push(Instruction::new(
            Opcode::ADD,
            (Register::T0 as u8) + k,
            Register::ZERO as u32,
            (k as u32) + 1,
            false,
            true,
        ));
    }
    // Body: rotate through a small repertoire of ALU ops.
    for i in 0..num_ops {
        let dst = (Register::T0 as u8) + (i % 8) as u8;
        let src_a = (Register::T0 as u32) + ((i + 1) % 8) as u32;
        let src_b_imm = ((i as u32) & 0x7f) + 1;
        let op = match i % 5 {
            0 => Opcode::ADD,
            1 => Opcode::SUB,
            2 => Opcode::XOR,
            3 => Opcode::AND,
            _ => Opcode::OR,
        };
        instrs.push(Instruction::new(op, dst, src_a, src_b_imm, false, true));
    }
    Program::new(instrs, 0, 0)
}

/// Snapshot the lower 32 registers (the JIT's `[u32; 36]` first 32
/// match the interpreter's `state.regs[0..32]`).
fn snapshot_interp_regs(rt: &mut Executor) -> [u32; 32] {
    let mut out = [0u32; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = rt.register(Register::from(i as u8));
    }
    out
}

/// Build a tight loop:
///
///   t0 = N
///   t1 = 0
/// loop:
///   t1 += 1
///   t0 -= 1
///   bne t0, $zero, loop
///   nop  (delay slot — `add t2, t2, 0`)
///
/// At end of loop: t0=0, t1=N, t2=0.  This exercises the JIT's
/// delay-slot dispatch (BNE arms `delayed_jump_target` at the loop
/// branch; the delay-slot ADD executes; then the per-instruction
/// epilogue indirect-jumps via `ctx.jump_table` back to the loop top).
fn build_bne_loop(iterations: u32) -> Program {
    let mut instrs = Vec::with_capacity(8);
    // pc=0: t0 = $zero + iterations  (ADDi, op_c immediate)
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T0 as u8,
        Register::ZERO as u32,
        iterations,
        false,
        true,
    ));
    // pc=4: t1 = $zero + 0
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T1 as u8,
        Register::ZERO as u32,
        0,
        false,
        true,
    ));
    // pc=8: loop_top:  t1 += 1
    let loop_top: u32 = 8;
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T1 as u8,
        Register::T1 as u32,
        1,
        false,
        true,
    ));
    // pc=12: t0 -= 1   (encoded as t0 = t0 + (-1))
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T0 as u8,
        Register::T0 as u32,
        (-1i32) as u32,
        false,
        true,
    ));
    // pc=16: bne t0, $zero, loop_top   (op_c = byte offset relative to
    //         next_pc per the executor's encoding; offset = target - next_pc)
    let bne_pc: u32 = 16;
    let bne_offset: u32 = loop_top.wrapping_sub(bne_pc.wrapping_add(4));
    instrs.push(Instruction::new(
        Opcode::BNE,
        Register::T0 as u8,
        Register::ZERO as u32,
        bne_offset,
        false,
        false,
    ));
    // pc=20: delay slot — must execute even when branch is taken.
    //         `add t2, t2, 0` (no-op-equivalent, but observably writes t2).
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T2 as u8,
        Register::T2 as u32,
        0,
        false,
        true,
    ));
    Program::new(instrs, 0, 0)
}

#[test]
fn bne_loop_jit_matches_interpreter() {
    const N: u32 = 17;
    let program = build_bne_loop(N);

    // ── Interpreter (forced via env) ────────────────────────
    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    match interp.run_fast() {
        Ok(()) | Err(ExecutionError::ExceptionOrTrap()) => {}
        Err(e) => panic!("interpreter run_fast failed: {e:?}"),
    }
    let interp_regs = snapshot_interp_regs(&mut interp);
    std::env::remove_var("ZIREN_DISABLE_JIT");

    // ── JIT ────────────────────────────────────────────────
    // Note: this currently reaches the JIT path via Executor::run_fast
    // only if the program clears the `JIT_MIN_INSTR_COUNT` gate.  This
    // 6-instr fixture does not, so we exercise the JIT path directly to
    // validate the codegen rather than gate it.

    use zkm_core_executor::jit_runner::{build_context, build_jit_function, run_jit, BuildParams};
    let params = BuildParams {
        program_size: program.instructions.len(),
        memory_size: 4096,
        max_trace_size: 4096,
        pc_start: 0,
        pc_base: 0,
        clk_bump: 4,
        mem_read_recorder: None,
    };
    let jit_fn = build_jit_function(&program, params, None).expect("build_jit_function");
    let mut memory = vec![0u8; 4096];
    let mut trace_buf = vec![0u8; 4096];
    let jump_table_ptr: *const *const u8 = jit_fn.jump_table.as_ptr();
    let mut ctx = build_context(
        0,
        memory.as_mut_ptr(),
        jump_table_ptr,
        jit_fn.jump_table.len(),
        trace_buf.as_mut_ptr(),
        [0u32; 36],
    );
    // Set `delayed_jump_target` to the post-loop sentinel before
    // entering: when the loop exits via the branch falling through,
    // the delay-slot epilogue jumps to ctx.jump_table[exit_idx].  For
    // straight-line termination we let it run off the end into the
    // shared exit label (auto-bound by finalize) — the assembled tail
    // spills regs and returns.  Set ctx.exit_code so the per-instr
    // gate after the delay slot fires.
    //
    // Concrete plan: post-loop, the instruction after the delay slot
    // (pc=24) is past the program.  The JIT's natural fall-through is
    // through the spill+epilogue tail so we don't need any sentinel.
    unsafe { run_jit(&jit_fn, &mut ctx) };

    // Compare regs.
    let mut mismatches = Vec::new();
    for (i, (interp, jit)) in interp_regs.iter().zip(ctx.registers[..32].iter()).enumerate() {
        if interp != jit {
            mismatches.push((i, *interp, *jit));
        }
    }
    assert!(
        mismatches.is_empty(),
        "register-file divergence (interp vs JIT) after BNE loop iter={N}: {mismatches:#?}",
    );
    // Also confirm the loop actually executed N iterations (not just
    // straight-lined through).
    assert_eq!(ctx.registers[Register::T1 as usize], N, "t1 should equal N after N loop iters");
    assert_eq!(ctx.registers[Register::T0 as usize], 0, "t0 should hit 0 to exit the loop");
}

/// LWL/LWR/SWL/SWR parity: a tiny program that exercises all four
/// unaligned ops at every alignment offset (i = 0..3).  Uses SW to
/// seed two adjacent words in stack space, then runs LWL/LWR at each
/// offset and asserts the JIT's destination register matches the
/// interpreter's.  Validates the inline dynasm shift+mask+merge
/// sequences against the executor's reference semantics.
#[test]
fn unaligned_lwl_lwr_jit_matches_interpreter() {
    // Pick a stack address that's safely inside MAX_MEMORY (~2 GB).
    // 0x7000_0000 works: it's well-aligned and below the real stack
    // top so the host buffer's MAP_NORESERVE pages get committed
    // lazily.
    const BASE: u32 = 0x7000_0000;
    // Two 4-byte words: 0xAABBCCDD at BASE, 0x11223344 at BASE+4.
    // Stored via SW so both interp and JIT see the same memory state.
    let mut instrs = Vec::with_capacity(16);
    // S0 = BASE   (load via two ADDs since immediates are 16-bit-ish)
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::S0 as u8,
        Register::ZERO as u32,
        BASE,
        false,
        true,
    ));
    // T0 = 0xAABBCCDD
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T0 as u8,
        Register::ZERO as u32,
        0xAABB_CCDDu32,
        false,
        true,
    ));
    // T1 = 0x11223344
    instrs.push(Instruction::new(
        Opcode::ADD,
        Register::T1 as u8,
        Register::ZERO as u32,
        0x1122_3344u32,
        false,
        true,
    ));
    // SW t0, 0(s0)
    instrs.push(Instruction::new(
        Opcode::SW,
        Register::T0 as u8,
        Register::S0 as u32,
        0,
        false,
        true,
    ));
    // SW t1, 4(s0)
    instrs.push(Instruction::new(
        Opcode::SW,
        Register::T1 as u8,
        Register::S0 as u32,
        4,
        false,
        true,
    ));
    // For each i in 0..4: LWL t2, i(s0) with t2 pre-seeded to 0xFFFFFFFF
    //   then save the result to a unique register.
    let dest_lwl = [Register::T2, Register::T3, Register::T4, Register::T5];
    let dest_lwr = [Register::T6, Register::T7, Register::S1, Register::S2];
    for (i, dst) in dest_lwl.iter().enumerate() {
        // Pre-seed dst = 0xF0F0F0F0 so the merge has observable bits.
        instrs.push(Instruction::new(
            Opcode::ADD,
            *dst as u8,
            Register::ZERO as u32,
            0xF0F0_F0F0u32,
            false,
            true,
        ));
        // LWL dst, i(s0)
        instrs.push(Instruction::new(
            Opcode::LWL,
            *dst as u8,
            Register::S0 as u32,
            i as u32,
            false,
            true,
        ));
    }
    for (i, dst) in dest_lwr.iter().enumerate() {
        instrs.push(Instruction::new(
            Opcode::ADD,
            *dst as u8,
            Register::ZERO as u32,
            0x0F0F_0F0Fu32,
            false,
            true,
        ));
        instrs.push(Instruction::new(
            Opcode::LWR,
            *dst as u8,
            Register::S0 as u32,
            i as u32,
            false,
            true,
        ));
    }
    let program = Program::new(instrs, 0, 0);

    // Interp run.
    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = interp.run_fast();
    let interp_regs = snapshot_interp_regs(&mut interp);
    std::env::remove_var("ZIREN_DISABLE_JIT");

    // JIT run.
    let mut jit = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = jit.run_fast();
    let jit_regs = snapshot_interp_regs(&mut jit);

    let mut mismatches = Vec::new();
    for (i, (a, b)) in interp_regs.iter().zip(jit_regs.iter()).enumerate() {
        if a != b {
            mismatches.push((i, *a, *b));
        }
    }
    assert!(
        mismatches.is_empty(),
        "register-file divergence interp vs JIT after LWL/LWR sweep: {mismatches:#?}",
    );
}

/// HALT syscall (id 0x00) parity: build a tiny program that loads
/// V0 = 0 and issues SYSCALL.  Both the interpreter and the
/// JIT-with-syscall-bridge must terminate cleanly with `state.exited
/// == true`.  Validates that the JIT syscall trampoline correctly
/// recovers `&mut Executor` from `ctx.user_data`, dispatches the
/// HALT impl, and surfaces the halt back to the host so the JIT
/// loop exits via the per-instruction exit-code gate.
#[test]
fn halt_syscall_jit_matches_interpreter() {
    // pc=0: V0 = 0  (HALT id)
    // pc=4: A0 = 0  (exit code)
    // pc=8: SYSCALL
    // pc=12: ADD t0, t0, 1  (would-be no-op tail; must NOT execute
    //         after HALT; the gate fires at start of K+1)
    let instrs = vec![
        Instruction::new(Opcode::ADD, Register::V0 as u8, Register::ZERO as u32, 0, false, true),
        Instruction::new(Opcode::ADD, Register::A0 as u8, Register::ZERO as u32, 0, false, true),
        Instruction::new(
            Opcode::SYSCALL,
            Register::ZERO as u8,
            Register::ZERO as u32,
            0,
            false,
            true,
        ),
        // Sentinel that should NOT execute on either path.
        Instruction::new(
            Opcode::ADD,
            Register::T0 as u8,
            Register::ZERO as u32,
            0xdead_beef_u32,
            false,
            true,
        ),
    ];
    let program = Program::new(instrs, 0, 0);

    // Interp run.
    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut interp = Executor::new(program.clone(), ZKMCoreOpts::default());
    let _ = interp.run_fast();
    let interp_exited = interp.state.exited;
    let interp_t0 = interp.register(Register::T0);
    std::env::remove_var("ZIREN_DISABLE_JIT");

    // JIT path via the lower-level API (lifts the SYSCALL-skip gate
    // that try_run_fast_jit still applies until the memory bridge
    // lands).  This validates the syscall trampoline + HALT exit-code
    // gate end-to-end.
    use zkm_core_executor::jit_runner::{
        build_context, build_jit_function, jit_syscall_handler, run_jit, BuildParams,
        JitBridgeState, JitMemoryBridge,
    };
    let mut jit_executor = Executor::new(program.clone(), ZKMCoreOpts::default());
    jit_executor.executor_mode = zkm_core_executor::ExecutorMode::Simple;
    jit_executor.print_report = true;
    let params = BuildParams {
        program_size: program.instructions.len(),
        memory_size: 4096,
        max_trace_size: 4096,
        pc_start: 0,
        pc_base: 0,
        clk_bump: 4,
        mem_read_recorder: None,
    };
    let jit_fn = build_jit_function(
        &program,
        params,
        Some(jit_syscall_handler as zkm_core_executor::jit_runner::JitSyscallHandler),
    )
    .expect("build_jit_function");
    let mut mem_bridge = JitMemoryBridge::new().expect("mem bridge mmap");
    let memory_ptr = mem_bridge.as_ptr();
    let mut trace_buf = vec![0u8; 4096];
    let jump_table_ptr: *const *const u8 = jit_fn.jump_table.as_ptr();
    let mut ctx = build_context(
        0,
        memory_ptr,
        jump_table_ptr,
        jit_fn.jump_table.len(),
        trace_buf.as_mut_ptr(),
        [0u32; 36],
    );
    let executor_ptr: *mut Executor = &mut jit_executor;
    let bridge_ptr: *mut JitMemoryBridge = &mut mem_bridge;
    let mut bridge_state = JitBridgeState {
        executor: unsafe { &mut *executor_ptr },
        bridge: unsafe { &mut *bridge_ptr },
        unconstrained_reg_snapshot: None,
    };
    ctx.user_data = &mut bridge_state as *mut _ as *mut std::ffi::c_void;
    unsafe { run_jit(&jit_fn, &mut ctx) };
    ctx.user_data = std::ptr::null_mut();
    // Clear the high-bit halt sentinel (0x8000_0000 = "halted with
    // exit_code 0") for the comparison.
    let raw_exit = ctx.exit_code;
    let jit_exit = if raw_exit == 0x8000_0000 { 0 } else { raw_exit };
    let jit_t0 = ctx.registers[Register::T0 as usize];

    assert!(interp_exited, "interpreter should HALT cleanly");
    assert_ne!(raw_exit, 0, "JIT should signal exit (sentinel for HALT-with-zero)");
    assert_eq!(jit_exit, 0, "JIT exit code should normalise to 0 for clean HALT");
    assert_eq!(
        jit_t0, interp_t0,
        "T0 must match (sentinel ADD at pc=12 should NOT execute on either path)"
    );
    assert_eq!(jit_t0, 0, "T0 should still be 0 — the post-HALT ADD must not run");
}

#[test]
fn alu_chain_jit_matches_interpreter_for_register_file() {
    // Build a small enough chain to keep both paths fast in CI but
    // long enough to exercise register-allocator hot paths in the JIT.
    let program = build_alu_chain(2_000);

    // Pre-screen: ALU-only chain must be JIT-eligible.
    assert!(
        first_unsupported_opcode(&program).is_none(),
        "ALU-only program should not contain SYSCALL/LWL/LWR/SWL/SWR"
    );

    // ── Interpreter ──────────────────────────────────────────
    let mut rt = Executor::new(program.clone(), ZKMCoreOpts::default());
    match rt.run_fast() {
        Ok(()) | Err(ExecutionError::ExceptionOrTrap()) => {}
        Err(e) => panic!("interpreter run_fast failed: {e:?}"),
    }
    let interp_regs = snapshot_interp_regs(&mut rt);

    // ── JIT ───────────────────────────────────────────────────
    let params = BuildParams {
        program_size: program.instructions.len(),
        memory_size: 4096,
        max_trace_size: 4096,
        pc_start: 0,
        pc_base: 0,
        clk_bump: 4,
        mem_read_recorder: None,
    };
    let jit_fn = build_jit_function(&program, params, None).expect("build_jit_function");

    let mut memory = vec![0u8; 4096];
    let jump_table_ptr: *const *const u8 = std::ptr::null();
    let mut trace_buf = vec![0u8; 4096];
    let mut ctx = build_context(
        0,
        memory.as_mut_ptr(),
        jump_table_ptr,
        jit_fn.jump_table.len(),
        trace_buf.as_mut_ptr(),
        [0u32; 36],
    );
    unsafe { run_jit(&jit_fn, &mut ctx) };

    // Compare lower-32 register files.  HI/LO and reserved aren't
    // touched by this workload.
    let mut mismatches = Vec::new();
    for (i, (interp, jit)) in interp_regs.iter().zip(ctx.registers[..32].iter()).enumerate() {
        if interp != jit {
            mismatches.push((i, *interp, *jit));
        }
    }
    assert!(
        mismatches.is_empty(),
        "register-file divergence between interp and JIT after ALU chain: {mismatches:#?}",
    );
}
