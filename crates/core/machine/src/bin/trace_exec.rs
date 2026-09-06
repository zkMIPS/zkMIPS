//! Run a dumped guest (`<block>-program.bin` + `<block>-stdin.bin`, from the
//! reth-processor host with `ZKM_DUMP=1`) through the interpreter.  Built with
//! `--config profile.release.debug-assertions=true` and run with
//! `TRACE_FILE=<path>`, the executor writes every constrained pc (u32 BE) to
//! the file — the input of `trace_hist`.
//!
//!   CARGO_TARGET_DIR=... cargo run --release --example trace_exec \
//!     --config profile.release.debug-assertions=true -- program.bin stdin.bin
use std::{env, fs, time::Instant};

use zkm_core_executor::{Executor, Program};
use zkm_core_machine::io::ZKMStdin;
use zkm_pcs::ZKMCoreOpts;

fn main() {
    let args: Vec<String> = env::args().collect();
    let elf = fs::read(&args[1]).expect("read program");
    let stdin: ZKMStdin =
        bincode::deserialize(&fs::read(&args[2]).expect("read stdin")).expect("decode stdin");
    let program = Program::from(&elf).expect("parse elf");
    let mut runtime = Executor::new(program, ZKMCoreOpts::default());
    runtime.write_vecs(&stdin.buffer);
    for (proof, vkey) in stdin.proofs.iter() {
        runtime.write_proof(proof.clone(), vkey.clone());
    }
    let t = Instant::now();
    runtime.run().expect("execute");
    eprintln!(
        "executed: {} cycles in {:.1} s (debug_assertions={})",
        runtime.state.global_clk,
        t.elapsed().as_secs_f64(),
        cfg!(debug_assertions)
    );
    let mut names: Vec<_> = runtime.report.cycle_tracker.iter().collect();
    names.sort_by(|a, b| b.1.cmp(a.1));
    for (name, cycles) in names {
        eprintln!("cycle-tracker {name}: {cycles}");
    }
}
