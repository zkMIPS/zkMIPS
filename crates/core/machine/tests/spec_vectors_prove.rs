//! Completeness of the circuits on spec-conformant executions.
//!
//! The executor-level vectors (`crates/core/executor/tests/spec_vectors/vectors.json`, one program
//! per (instruction, case), all of which match the Unicorn oracle) are proved end to end with the
//! CPU prover and verified.  A failure here means a valid MIPS execution the executor accepts is
//! rejected by the constraint system (or the prover cannot shape it) — an incompleteness.
//!
//! `SPEC_PROVE_LIMIT` caps the number of cases per instruction (default 1, so the run covers every
//! instruction once); `SPEC_PROVE_ONLY=ADD,SUB` restricts the mnemonics.

use std::collections::BTreeMap;

use serde_json::Value;
use zkm_core_executor::{Executor, Instruction, Program};
use zkm_core_machine::io::ZKMStdin;
use zkm_core_machine::utils::{run_test_core, setup_logger};
use zkm_pcs::{CpuProver, ZKMCoreOpts};

const VECTORS: &str = include_str!("../../executor/tests/spec_vectors/vectors.json");

fn parse_hex(s: &str) -> u32 {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex")
}

#[test]
fn spec_vectors_prove_and_verify() {
    setup_logger();
    let limit: usize =
        std::env::var("SPEC_PROVE_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let only: Option<Vec<String>> = std::env::var("SPEC_PROVE_ONLY")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let doc: Value = serde_json::from_str(VECTORS).expect("vectors.json");
    let code = doc["code"].as_u64().unwrap() as u32;
    let mut per_mnemonic: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut proved = 0usize;
    for v in doc["vectors"].as_array().unwrap() {
        let mnemonic = v["mnemonic"].as_str().unwrap().to_string();
        if let Some(only) = &only {
            if !only.contains(&mnemonic) {
                continue;
            }
        }
        let n = per_mnemonic.entry(mnemonic.clone()).or_insert(0);
        if *n >= limit {
            continue;
        }
        *n += 1;
        if v["expect_trap"].as_bool().unwrap() {
            continue; // trapping executions are rejected by the executor, nothing to prove
        }
        let name = v["name"].as_str().unwrap();
        let words: Vec<u32> =
            v["words"].as_array().unwrap().iter().map(|w| parse_hex(w.as_str().unwrap())).collect();
        let instructions: Vec<Instruction> =
            words.iter().map(|w| Instruction::decode_from(*w).unwrap()).collect();
        let mut program = Program::new(instructions, code, code);
        program.image = v["image"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, val)| (parse_hex(k), val.as_u64().unwrap() as u32))
            .collect();
        let started = std::time::Instant::now();
        let runtime = Executor::new(program, ZKMCoreOpts::default());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_test_core::<CpuProver<_, _>>(runtime, ZKMStdin::new(), None)
        }));
        match outcome {
            Ok(Ok(_)) => {
                proved += 1;
                eprintln!("proved {name} in {:.1}s", started.elapsed().as_secs_f64());
            }
            Ok(Err(e)) => failures.push(format!("{name}: verification failed: {e:?}")),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failures.push(format!("{name}: prover panicked: {msg}"));
            }
        }
    }
    eprintln!("spec vectors proved: {proved}, failed: {}", failures.len());
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} spec vector programs did not prove", failures.len());
}

/// Cannon's open_mips_tests, prepared exactly as in the executor conformance test (big-endian
/// words, the 0xbfXX region relocated to 0x7eXX, the harness return turned into a jump past the
/// program), proved and verified end to end.  Skipped when `CANNON_MIPS_TESTS` is not present.
#[test]
fn cannon_programs_prove_and_verify() {
    setup_logger();
    let dir = std::env::var("CANNON_MIPS_TESTS").unwrap_or_else(|_| {
        "/data/stephen/cannon-mips/mipsevm/open_mips_tests/test/bin".to_string()
    });
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("CANNON_MIPS_TESTS not found at {dir}; skipping");
        return;
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "bin").unwrap_or(false))
        .collect();
    files.sort();
    let mut failures = Vec::new();
    let mut proved = 0usize;
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        // Cannon-ABI programs and the big-endian-only byte/half/partial-word programs do not
        // complete successfully on a little-endian machine; they are not valid executions to prove.
        if name.starts_with("oracle")
            || matches!(
                name.as_str(),
                "brk"
                    | "exit_group"
                    | "lb"
                    | "lbu"
                    | "lh"
                    | "lhu"
                    | "lwl"
                    | "lwr"
                    | "sb"
                    | "sh"
                    | "swl"
                    | "swr"
            )
        {
            continue;
        }
        let bytes = std::fs::read(path).unwrap();
        let mut words: Vec<u32> =
            bytes.chunks(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect();
        for w in words.iter_mut() {
            if *w >> 26 == 0x0f && (*w & 0xff00) == 0xbf00 {
                *w = (*w & 0xffff_0000) | 0x7e00 | (*w & 0xff);
            }
        }
        let end = (4 * words.len()) as u32;
        let Some(pos) = (1..words.len())
            .find(|i| words[*i] == 0x03e0_0008 && words[i - 1] == 0xae11_0004)
            .or_else(|| words.iter().rposition(|w| *w == 0x03e0_0008))
        else {
            continue;
        };
        words[pos] = 0x0800_0000 | ((end >> 2) & 0x03ff_ffff);
        let instructions: Vec<Instruction> =
            words.iter().map(|w| Instruction::decode_from(*w).unwrap()).collect();
        let program = Program::new(instructions, 0, 0);
        let started = std::time::Instant::now();
        let runtime = Executor::new(program, ZKMCoreOpts::default());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_test_core::<CpuProver<_, _>>(runtime, ZKMStdin::new(), None)
        }));
        match outcome {
            Ok(Ok(_)) => {
                proved += 1;
                eprintln!("proved cannon {name} in {:.1}s", started.elapsed().as_secs_f64());
            }
            Ok(Err(e)) => failures.push(format!("{name}: verification failed: {e:?}")),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failures.push(format!("{name}: prover panicked: {msg}"));
            }
        }
    }
    eprintln!("cannon programs proved: {proved}, failed: {}", failures.len());
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} cannon programs did not prove", failures.len());
}
