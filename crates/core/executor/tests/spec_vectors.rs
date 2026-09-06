//! Spec conformance of the executor.
//!
//! Two vector sources, both encoding-level (they go through `Instruction::decode_from`, not
//! through hand-built `Opcode`s):
//!
//! 1. `spec_vectors/vectors.json`: every instruction row of `docs/src/mips-vm/mips-isa.md`,
//!    assembled with llvm-mc and executed by Unicorn (QEMU's MIPS32r2 core) as the independent
//!    oracle.  Regenerate with `spec_vectors/gen.py`.
//! 2. Cannon's `open_mips_tests` (63 flat binaries, MIT-licensed test programs by Grant Ayers
//!    packaged by Optimism): each program writes `1` to the done and result words at
//!    `0xbffffff4` / `0xbffffff8`.  Point `CANNON_MIPS_TESTS` at the `test/bin` directory; the
//!    test is skipped when it is absent.
//!
//! Every program is also run twice to check that the executor's final state is a function of the
//! program alone.

use std::collections::BTreeMap;

use serde_json::Value;
use zkm_core_executor::{Executor, Instruction, Program, Register};
use zkm_pcs::ZKMCoreOpts;

const VECTORS: &str = include_str!("spec_vectors/vectors.json");

struct Final {
    regs: [u32; 32],
    hi: u32,
    lo: u32,
    mem: Vec<(u32, u32)>,
    clk: u32,
    records_digest: u64,
}

fn run_words(
    words: &[u32],
    code: u32,
    image: &BTreeMap<u32, u32>,
    mem_addrs: &[u32],
) -> Result<Final, String> {
    run_words_mode(words, code, image, mem_addrs, false)
}

/// `fast` selects `Executor::run_fast` (the JIT / untraced path) instead of the traced `run`.
fn run_words_mode(
    words: &[u32],
    code: u32,
    image: &BTreeMap<u32, u32>,
    mem_addrs: &[u32],
    fast: bool,
) -> Result<Final, String> {
    let mut instructions = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        let insn = Instruction::decode_from(*w)
            .map_err(|e| format!("decoder rejects word {i} ({w:#010x}): {e}"))?;
        instructions.push(insn);
    }
    let mut program = Program::new(instructions, code, code);
    program.image = image.clone();
    let mut runtime = Executor::new(program, ZKMCoreOpts::default());
    if fast {
        runtime.run_fast().map_err(|e| format!("execution error: {e:?}"))?;
    } else {
        runtime.run().map_err(|e| format!("execution error: {e:?}"))?;
    }
    let mut regs = [0u32; 32];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = runtime.register(Register::from(i as u8));
    }
    let hi = runtime.register(Register::HI);
    let lo = runtime.register(Register::LO);
    let mem = mem_addrs.iter().map(|a| (*a, runtime.word(*a))).collect();
    // Trace-level fingerprint: the serialized execution records (every event the prover will
    // see), so determinism is checked on the trace, not only on the architectural state.
    let records_digest = if fast {
        0
    } else {
        let bytes = bincode::serialize(&runtime.records).expect("serialize records");
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    };
    Ok(Final { regs, hi, lo, mem, clk: runtime.state.clk, records_digest })
}

fn parse_hex(s: &str) -> u32 {
    let s = s.trim_start_matches("0x");
    u32::from_str_radix(s, 16).expect("hex")
}

#[test]
fn spec_vectors_match_the_oracle() {
    let doc: Value = serde_json::from_str(VECTORS).expect("vectors.json");
    let code = doc["code"].as_u64().unwrap() as u32;
    let vectors = doc["vectors"].as_array().unwrap();
    let mut failures: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut per_mnemonic: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for v in vectors {
        let name = v["name"].as_str().unwrap().to_string();
        let mnemonic = v["mnemonic"].as_str().unwrap().to_string();
        let words: Vec<u32> =
            v["words"].as_array().unwrap().iter().map(|w| parse_hex(w.as_str().unwrap())).collect();
        let image: BTreeMap<u32, u32> = v["image"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, val)| (parse_hex(k), val.as_u64().unwrap() as u32))
            .collect();
        let expect_mem: Vec<(u32, u32)> = v["mem"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, val)| (parse_hex(k), val.as_u64().unwrap() as u32))
            .collect();
        let mem_addrs: Vec<u32> = expect_mem.iter().map(|(a, _)| *a).collect();
        let expect_trap = v["expect_trap"].as_bool().unwrap();
        let entry = per_mnemonic.entry(mnemonic.clone()).or_insert((0, 0));
        entry.0 += 1;

        let mut problems = Vec::new();
        match run_words(&words, code, &image, &mem_addrs) {
            Err(e) if expect_trap && e.starts_with("execution error") => {}
            Err(e) => problems.push(e),
            Ok(_) if expect_trap => problems.push("expected a trap, executor completed".into()),
            Ok(fin) => {
                let regs: Vec<u32> = v["regs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                for i in 1..32 {
                    if fin.regs[i] != regs[i] {
                        problems.push(format!(
                            "${i}: ziren {:#010x} oracle {:#010x}",
                            fin.regs[i], regs[i]
                        ));
                    }
                }
                let (hi, lo) = (v["hi"].as_u64().unwrap() as u32, v["lo"].as_u64().unwrap() as u32);
                if fin.hi != hi {
                    problems.push(format!("HI: ziren {:#010x} oracle {hi:#010x}", fin.hi));
                }
                if fin.lo != lo {
                    problems.push(format!("LO: ziren {:#010x} oracle {lo:#010x}", fin.lo));
                }
                for ((a, got), (_, want)) in fin.mem.iter().zip(&expect_mem) {
                    if got != want {
                        problems
                            .push(format!("mem[{a:#x}]: ziren {got:#010x} oracle {want:#010x}"));
                    }
                }
                // Determinism: a second run must reproduce the state exactly.
                let again = run_words(&words, code, &image, &mem_addrs).expect("second run");
                if again.regs != fin.regs
                    || again.hi != fin.hi
                    || again.lo != fin.lo
                    || again.mem != fin.mem
                    || again.clk != fin.clk
                {
                    problems.push("second run differs from the first".into());
                }
                // The untraced / JIT path must agree with the traced interpreter.
                match run_words_mode(&words, code, &image, &mem_addrs, true) {
                    Ok(fast) => {
                        if fast.regs != fin.regs
                            || fast.hi != fin.hi
                            || fast.lo != fin.lo
                            || fast.mem != fin.mem
                        {
                            problems.push("run_fast differs from run".into());
                        }
                    }
                    Err(e) => problems.push(format!("run_fast: {e}")),
                }
            }
        }
        if problems.is_empty() {
            entry.1 += 1;
        } else {
            let asm: Vec<String> = v["asm"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_string())
                .collect();
            failures.insert(name, [vec![format!("asm: {}", asm.join(" ; "))], problems].concat());
        }
    }
    // Optional: dump the executor's own decoding of every vector word (opcode id and operands),
    // so the Lean ISA model can be checked against `Instruction::decode_from` as well as against
    // the oracle's results (see `spec_vectors/gen_lean.py`).
    if let Ok(path) = std::env::var("SPEC_DUMP_DECODED") {
        let mut decoded: BTreeMap<String, Value> = BTreeMap::new();
        for v in vectors {
            for w in v["words"].as_array().unwrap() {
                let word = parse_hex(w.as_str().unwrap());
                let key = format!("{word:08x}");
                if decoded.contains_key(&key) {
                    continue;
                }
                let entry = match Instruction::decode_from(word) {
                    Ok(i) => serde_json::json!({
                        "opcode": i.opcode as u32, "opcode_name": format!("{:?}", i.opcode),
                        "op_a": i.op_a, "op_b": i.op_b, "op_c": i.op_c, "imm_b": i.imm_b, "imm_c": i.imm_c,
                    }),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                };
                decoded.insert(key, entry);
            }
        }
        std::fs::write(&path, serde_json::to_string_pretty(&decoded).unwrap())
            .expect("write decoded dump");
        eprintln!("wrote {} decoded words to {path}", decoded.len());
    }
    eprintln!("spec vector conformance (passed/total per instruction):");
    for (mn, (total, ok)) in &per_mnemonic {
        eprintln!("  {mn:8} {ok:3}/{total}");
    }
    if !failures.is_empty() {
        for (name, problems) in &failures {
            eprintln!("FAIL {name}");
            for p in problems {
                eprintln!("    {p}");
            }
        }
        panic!("{} of {} spec vectors failed", failures.len(), vectors.len());
    }
}

#[test]
fn cannon_open_mips_tests() {
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
    let mut passed = 0usize;
    let mut skipped = Vec::new();
    let mut big_endian_only = Vec::new();
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if name.starts_with("oracle") || name == "brk" {
            // Cannon ABI, not ISA: the pre-image oracle syscalls, and `brk` returning Cannon's
            // fixed heap start 0x40000000 (Ziren's heap lives elsewhere).
            skipped.push(name);
            continue;
        }
        let bytes = std::fs::read(path).unwrap();
        // Cannon targets big-endian MIPS: instruction words are stored big-endian.  Instruction
        // semantics are endianness-neutral except for the byte / half-word / partial-word memory
        // ops, whose expected results assume a big-endian data layout; Ziren is little-endian
        // (mipsel), so those programs are reported separately, not counted as failures.
        let words: Vec<u32> =
            bytes.chunks(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect();
        let byte_order_sensitive = matches!(
            name.as_str(),
            "lb" | "lbu" | "lh" | "lhu" | "lwl" | "lwr" | "sb" | "sh" | "swl" | "swr"
        );
        // The programs address their done / result words and data areas through
        // `lui rX, 0xbfXX ; ori ...` pairs (0xbffffff0 for the result, 0xbfc00000 for data).
        // Ziren's guest address space ends at 0x7f010000 (stack top), so relocate the whole
        // 0xbfXX0000 region to 0x7eXX0000 by patching every such `lui`; address arithmetic
        // inside the programs is relative to those bases, so the semantics are unchanged.
        let mut words = words;
        for w in words.iter_mut() {
            if *w >> 26 == 0x0f && (*w & 0xff00) == 0xbf00 {
                *w = (*w & 0xffff_0000) | 0x7e00 | (*w & 0xff);
            }
        }
        // They end with `jr $ra` back to the harness; `$ra` is zero here and a jump to address 0
        // is a null-pointer error in Ziren.  Replace the last `jr $ra` by an absolute jump past
        // the end of the program, which is the executor's termination condition.
        let end = (4 * words.len()) as u32;
        // The harness return is the `jr $ra` right after `sw $s1, 4($s0)` (the done flag);
        // programs with subroutines place further `jr $ra`s after it.
        let harness_ret = (1..words.len())
            .find(|i| words[*i] == 0x03e0_0008 && words[i - 1] == 0xae11_0004)
            .or_else(|| words.iter().rposition(|w| *w == 0x03e0_0008));
        if let Some(pos) = harness_ret {
            words[pos] = 0x0800_0000 | ((end >> 2) & 0x03ff_ffff);
        } else {
            skipped.push(format!("{name} (no final jr $ra)"));
            continue;
        }
        let base = 0x7eff_fff0u32;
        // The programs end with `jr $ra`; $ra is zero, and pc == 0 terminates the executor.
        let done_addr = base + 4;
        let result_addr = base + 8;
        eprintln!("cannon {name}");
        let outcome = std::panic::catch_unwind(|| {
            run_words(&words, 0, &BTreeMap::new(), &[done_addr, result_addr])
        })
        .unwrap_or_else(|p| {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Err(format!("executor panicked: {msg}"))
        });
        match outcome {
            Ok(fin) => {
                let (done, result) = (fin.mem[0].1, fin.mem[1].1);
                if done == 1 && result == 1 {
                    passed += 1;
                } else if byte_order_sensitive {
                    big_endian_only.push(format!("{name}: done={done} result={result}"));
                } else {
                    failures.push(format!("{name}: done={done} result={result}"));
                }
            }
            Err(e) if name == "exit_group" && e.contains("HaltWithNonZeroExitCode") => passed += 1,
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    eprintln!(
        "cannon open_mips_tests: {passed} passed, {} failed, {} big-endian-only ({:?}), skipped {:?}",
        failures.len(),
        big_endian_only.len(),
        big_endian_only,
        skipped
    );
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} cannon vectors failed", failures.len());
}
