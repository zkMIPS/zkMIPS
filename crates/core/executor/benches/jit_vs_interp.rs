//! Side-by-side timing harness for the MIPS interpreter
//! ([`Executor::run_very_fast`]) vs the JIT (raw `dynasm-rt`-emitted
//! native code via the [`zkm_core_jit`] transpiler).
//!
//! Workload: N synthetic `ADD` instructions of the form
//! `t0 = t0 + 1` repeated, executed back-to-back with no syscalls,
//! branches, or memory traffic.  This isolates the per-instruction
//! dispatch cost — the metric the JIT is designed to win on
//! (see `docs/jit_design.md`, section 1).
//!
//! Both paths execute the same logical workload; the harness reports
//! mean / min / max over `REPEATS` iterations and the JIT speedup.
//!
//! The JIT side runs the same end-to-end path the executor would
//! take if the JIT were the default: `jit_runner::build_jit_function`
//! lowers the [`Program`] via the MIPS driver, wraps it in the
//! SysV-ABI prologue/epilogue and the register seed/spill, and
//! `run_jit` invokes the resulting native code through a
//! [`JitContext`] built with `build_context`.
//!
//! Build with:
//!
//! ```sh
//! source ~/.zkm-toolchain/env
//! cargo bench --bench jit_vs_interp -p zkm-core-executor
//! ```

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use std::time::Instant;

use zkm_core_executor::{ExecutionError, Executor, Instruction, Opcode, Program, Register};
use zkm_pcs::ZKMCoreOpts;

const NUM_INSTRS: usize = 100_000;
const REPEATS: usize = 5;

fn build_program() -> Program {
    // `t0 = t0 + 1` repeated NUM_INSTRS times.
    let mut instrs = Vec::with_capacity(NUM_INSTRS);
    for _ in 0..NUM_INSTRS {
        instrs.push(Instruction::new(
            Opcode::ADD,
            Register::T0 as u8,
            Register::T0 as u32,
            1,
            false, // imm_b: false (op_b is a register)
            true,  // imm_c: true (op_c is the immediate 1)
        ));
    }
    Program::new(instrs, 0, 0)
}

/// Mode for the interpreter run.  Mirrors the three public
/// entry points on `Executor`: `run_very_fast` (Simple mode, no
/// report), `run_fast` (Simple mode + report), and `run` (Trace
/// mode + report — the path used by the prover).
#[derive(Copy, Clone)]
enum InterpMode {
    VeryFast,
    Fast,
    Trace,
}

fn run_interpreter(program: &Program, mode: InterpMode) -> Result<u64, ExecutionError> {
    // `Executor::run_fast` / `run_very_fast` now dispatch to the JIT by
    // default on supported programs.  For this bench the "interp" rows
    // are meant to measure the actual interpreter, so we force the
    // fallback via the runtime gate.
    let prev = std::env::var_os("ZIREN_DISABLE_JIT");
    std::env::set_var("ZIREN_DISABLE_JIT", "1");
    let mut runtime = Executor::new(program.clone(), ZKMCoreOpts::default());
    let start = Instant::now();
    // ExceptionOrTrap is expected once PC walks off the end — count
    // it as a normal termination for this microbench.
    let res = match mode {
        InterpMode::VeryFast => runtime.run_very_fast(),
        InterpMode::Fast => runtime.run_fast(),
        InterpMode::Trace => runtime.run(),
    };
    let elapsed = start.elapsed().as_nanos() as u64;
    match prev {
        Some(v) => std::env::set_var("ZIREN_DISABLE_JIT", v),
        None => std::env::remove_var("ZIREN_DISABLE_JIT"),
    }
    match res {
        Ok(()) | Err(ExecutionError::ExceptionOrTrap()) => {}
        Err(other) => return Err(other),
    }
    Ok(elapsed)
}

fn run_jit_alu_chain(program: &Program) -> u64 {
    use std::ptr;
    use zkm_core_executor::jit_runner::{build_context, build_jit_function, run_jit, BuildParams};

    // Use the same end-to-end path the executor would call if the
    // JIT were the default — `jit_runner::build_jit_function` (which
    // wraps the driver with prologue/epilogue/regs).
    let params = BuildParams {
        program_size: NUM_INSTRS,
        memory_size: 4096,
        max_trace_size: 4096,
        pc_start: 0,
        pc_base: 0,
        clk_bump: 4,
        mem_read_recorder: None,
    };
    let jit_fn = build_jit_function(program, params, None).expect("build_jit_function");

    let mut memory = vec![0u8; 4096];
    let jump_table_ptr: *const *const u8 = ptr::null();
    let mut trace_buf = vec![0u8; 4096];
    let mut ctx = build_context(
        0,
        memory.as_mut_ptr(),
        jump_table_ptr,
        jit_fn.jump_table.len(),
        trace_buf.as_mut_ptr(),
        [0u32; 36],
    );

    let start = Instant::now();
    unsafe { run_jit(&jit_fn, &mut ctx) };
    start.elapsed().as_nanos() as u64
}

fn report(name: &str, samples: &[u64]) {
    let n = samples.len() as f64;
    let total: u64 = samples.iter().sum();
    let mean = total as f64 / n;
    let min = *samples.iter().min().unwrap() as f64;
    let max = *samples.iter().max().unwrap() as f64;
    let per_instr = mean / NUM_INSTRS as f64;
    eprintln!(
        "{:<18} mean {:>9.2} ms  min {:>9.2} ms  max {:>9.2} ms  ({:>7.2} ns/instr)",
        name,
        mean / 1e6,
        min / 1e6,
        max / 1e6,
        per_instr,
    );
}

/// Run the real fibonacci ELF through `Executor::run_fast`.  Returns
/// the elapsed nanos.  `disable_jit=true` forces the interpreter
/// fallback; otherwise the JIT-by-default path runs.
fn run_real_elf_run_fast(elf_bytes: &[u8], disable_jit: bool, n: u32) -> u64 {
    if disable_jit {
        std::env::set_var("ZIREN_DISABLE_JIT", "1");
    } else {
        std::env::remove_var("ZIREN_DISABLE_JIT");
    }
    // Parse outside the timer — both paths pay it identically and
    // it's not the JIT's cost.
    let program = Program::from(elf_bytes).expect("parse fibonacci ELF");
    let mut rt = Executor::new(program, ZKMCoreOpts::default());
    rt.state.input_stream.push(n.to_le_bytes().to_vec());
    let start = Instant::now();
    let _ = rt.run_fast();
    let elapsed = start.elapsed().as_nanos() as u64;
    std::env::remove_var("ZIREN_DISABLE_JIT");
    elapsed
}

fn main() {
    let program = build_program();

    // Warmup — first run pays page-fault / icache costs we don't
    // want polluting the measurement.
    let _ = run_interpreter(&program, InterpMode::VeryFast).expect("warmup");
    let _ = run_jit_alu_chain(&program);

    let mut very_fast = Vec::with_capacity(REPEATS);
    let mut fast = Vec::with_capacity(REPEATS);
    let mut trace = Vec::with_capacity(REPEATS);
    let mut jit = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        very_fast.push(run_interpreter(&program, InterpMode::VeryFast).expect("very-fast"));
        fast.push(run_interpreter(&program, InterpMode::Fast).expect("fast"));
        trace.push(run_interpreter(&program, InterpMode::Trace).expect("trace"));
        jit.push(run_jit_alu_chain(&program));
    }

    eprintln!(
        "=== JIT vs interpreter ({} ADD instrs, {} repeats, after warmup) ===",
        NUM_INSTRS, REPEATS
    );
    report("interp very_fast", &very_fast);
    report("interp fast", &fast);
    report("interp trace", &trace);
    report("jit (call)", &jit);

    let mean = |s: &[u64]| s.iter().sum::<u64>() as f64 / REPEATS as f64;
    let jit_mean = mean(&jit);
    eprintln!("speedup vs very_fast: {:>7.2}x", mean(&very_fast) / jit_mean);
    eprintln!("speedup vs fast:      {:>7.2}x", mean(&fast) / jit_mean);
    eprintln!("speedup vs trace:     {:>7.2}x", mean(&trace) / jit_mean);

    // Real-world end-to-end: time `Executor::run_fast` on the
    // fibonacci ELF through both code paths.  This is what the
    // prover's `execute()` pre-pass actually measures.
    let elf_path =
        "/data/stephen/Ziren/examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/fibonacci";
    // Also try hello-world (no input) — exercises a different ELF.
    let hello_elf = "/data/stephen/Ziren/crates/test-artifacts/guests/target/elf-compilation/mipsel-zkm-zkvm-elf/release/hello-world";
    if let Ok(hello_bytes) = std::fs::read(hello_elf) {
        eprintln!();
        eprintln!("=== hello-world ELF (run_fast end-to-end, no input, {} repeats) ===", REPEATS);
        let _ = run_real_elf_run_fast(&hello_bytes, true, 0);
        let _ = run_real_elf_run_fast(&hello_bytes, false, 0);
        let mut interp_h = Vec::with_capacity(REPEATS);
        let mut jit_h = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            interp_h.push(run_real_elf_run_fast(&hello_bytes, true, 0));
            jit_h.push(run_real_elf_run_fast(&hello_bytes, false, 0));
        }
        eprintln!("interp run_fast    mean {:>9.3} ms", mean(&interp_h) / 1e6);
        eprintln!("JIT-by-default     mean {:>9.3} ms", mean(&jit_h) / 1e6);
        eprintln!("JIT vs interp on hello-world: {:>5.2}x", mean(&interp_h) / mean(&jit_h));
    }

    if let Ok(elf_bytes) = std::fs::read(elf_path) {
        // Sweep across input sizes so we can see where the
        // per-call transpile + mmap setup pays back.  fibonacci's
        // inner loop is short, so larger N means more dynamic
        // cycles amortising the same static program.
        for &n in &[20u32, 1000, 50_000] {
            eprintln!();
            eprintln!(
                "=== Real fibonacci ELF (run_fast end-to-end, n={n}, {} repeats) ===",
                REPEATS,
            );
            let _ = run_real_elf_run_fast(&elf_bytes, true, n);
            let _ = run_real_elf_run_fast(&elf_bytes, false, n);
            let mut interp_fib = Vec::with_capacity(REPEATS);
            let mut jit_fib = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                interp_fib.push(run_real_elf_run_fast(&elf_bytes, true, n));
                jit_fib.push(run_real_elf_run_fast(&elf_bytes, false, n));
            }
            let interp_mean = mean(&interp_fib);
            let jit_fib_mean = mean(&jit_fib);
            eprintln!(
                "interp run_fast    mean {:>9.3} ms  min {:>9.3} ms  max {:>9.3} ms",
                interp_mean / 1e6,
                *interp_fib.iter().min().unwrap() as f64 / 1e6,
                *interp_fib.iter().max().unwrap() as f64 / 1e6,
            );
            eprintln!(
                "JIT-by-default     mean {:>9.3} ms  min {:>9.3} ms  max {:>9.3} ms",
                jit_fib_mean / 1e6,
                *jit_fib.iter().min().unwrap() as f64 / 1e6,
                *jit_fib.iter().max().unwrap() as f64 / 1e6,
            );
            eprintln!("JIT vs interp on fib(n={n}): {:>5.2}x", interp_mean / jit_fib_mean);
        }
    } else {
        eprintln!();
        eprintln!("[skip real-fib bench] ELF not built at {elf_path}");
    }
}
