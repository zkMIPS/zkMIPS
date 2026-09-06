//! Generic experiment harness for the MIPS core machine.
//!
//! One reusable entry point for the questions that keep coming up when a proof
//! misbehaves, so an investigation does not need its own throwaway binary:
//!
//!   * `execute` — run a program and report shards, cycles and syscall usage.
//!   * `buses`   — per-`LookupKind` send/receive balance, the diagnostic that
//!                 localizes a `LogUp-GKR: public-values balance failed`.
//!   * `prove`   — full prove + verify.
//!
//! Reproducibility: every run prints a header naming the program, the commit,
//! and the environment variables that are known to change proof bytes, so a
//! result can be reproduced exactly.
//!
//! ```text
//! cargo run --release -p zkm-core-machine --example playground -- buses fibonacci
//! cargo run --release -p zkm-core-machine --example playground -- prove   /path/to/guest.elf
//! ```
//!
//! Reading a `buses` report: a bus is healthy when it prints BALANCED.  The
//! public-values boundary buses (`State`, `GlobalAccumulation`,
//! `MemoryGlobalInitControl`, `MemoryGlobalFinalizeControl`) are the exception —
//! they are closed by the public-values AIR, which this tool does not evaluate,
//! so each should show exactly TWO unmatched keys (the initial endpoint at `-1`
//! and the final endpoint at `+1`).  More than two means the chip chain failed
//! to telescope, and the extra pair names the row where it broke.

use p3_koala_bear::KoalaBear;
use zkm_core_executor::{Executor, Program};
use zkm_core_machine::{
    io::ZKMStdin,
    mips::MipsAir,
    utils::{run_test, run_test_io, setup_logger},
};
use zkm_pcs::air::MachineAir;
use zkm_pcs::MachineRecord;
use zkm_pcs::{
    debug_lookups_with_all_chips, koala_bear_poseidon2::KoalaBearPoseidon2, CpuProver, LookupKind,
    LookupScope, StarkMachine, ZKMCoreOpts,
};

/// Every bus the core machine carries.  Sweeping the whole list matters: the
/// public-values boundary buses are 9..=13, and an investigation that only
/// looks at the familiar 1..=8 will see nothing but architectural noise.
const ALL_KINDS: &[LookupKind] = &[
    LookupKind::Memory,
    LookupKind::Program,
    LookupKind::Instruction,
    LookupKind::Byte,
    LookupKind::Range,
    LookupKind::Syscall,
    LookupKind::Global,
    LookupKind::SyscallResult,
    LookupKind::State,
    LookupKind::GlobalAccumulation,
    LookupKind::MemoryGlobalInitControl,
    LookupKind::MemoryGlobalFinalizeControl,
    LookupKind::PrecompileChain,
];

/// Environment variables that change what gets proven or how it is shaped.
/// Printed on every run so a reported result carries the config that produced
/// it.  (`SHAPE_CHECK_FREQUENCY` is retired: shard limits are exact on every
/// cycle, so no frequency knob influences the proof digest.)
const REPRO_ENV: &[&str] = &[
    "SHARD_SIZE",
    "SHARD_BATCH_SIZE",
    "TRACE_GEN_WORKERS",
    "RAYON_NUM_THREADS",
    "ZKM_SKIP_PROGRAM_BUILD",
];

/// Accepts a built-in fixture name or a path to any guest ELF, so the same
/// harness works for a one-line repro and for a real workload.
fn resolve(name: &str) -> Program {
    let builtin: Option<&[u8]> = match name {
        "fibonacci" => Some(test_artifacts::FIBONACCI_ELF),
        "hello-world" => Some(test_artifacts::HELLO_WORLD_ELF),
        "sha3-chain" => Some(test_artifacts::SHA3_CHAIN_ELF),
        "keccak-sponge" => Some(test_artifacts::KECCAK_SPONGE_ELF),
        "unconstrained" => Some(test_artifacts::UNCONSTRAINED_ELF),
        _ => None,
    };
    if let Some(elf) = builtin {
        return Program::from(elf).expect("built-in fixture must parse");
    }
    let elf = std::fs::read(name)
        .unwrap_or_else(|e| panic!("not a known fixture and not a readable ELF: {name}: {e}"));
    Program::from(&elf).unwrap_or_else(|e| panic!("failed to parse ELF {name}: {e:?}"))
}

fn banner(cmd: &str, program: &str) {
    eprintln!("=== playground: {cmd} {program} ===");
    for k in REPRO_ENV {
        eprintln!("    {k}={}", std::env::var(k).unwrap_or_else(|_| "<unset>".into()));
    }
}

fn main() {
    setup_logger();
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "buses".into());
    let name = args.next().unwrap_or_else(|| "fibonacci".into());
    banner(&cmd, &name);
    // `all` (optional subset list) and `dump` (output dir) don't take a
    // fixture/ELF — resolve lazily.
    // `all` / `dump` don't take a fixture, and `lookups static` reports the
    // compile-time census only — resolve lazily so none of them needs an ELF.
    let program = if cmd == "all" || cmd == "dump" || (cmd == "lookups" && name == "static") {
        None
    } else {
        Some(resolve(&name))
    };
    let program = move || program.expect("mode needs a fixture");

    match cmd.as_str() {
        "execute" => {
            let mut rt = Executor::new(program(), ZKMCoreOpts::default());
            rt.run().expect("execution failed");
            eprintln!("shards       = {}", rt.records.len());
            eprintln!("global_clk   = {}", rt.state.global_clk);
            eprintln!("exited       = {}", rt.state.exited);
        }
        "buses" => {
            let program = program();
            let program_clone = program.clone();
            let mut rt = Executor::new(program, ZKMCoreOpts::default());
            rt.run().expect("execution failed");
            let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
                MipsAir::machine(KoalaBearPoseidon2::new());
            let (pkey, _) = machine.setup(&program_clone);
            let opts = ZKMCoreOpts::default();
            machine.generate_dependencies(&mut rt.records, &opts, None).expect("dependencies");
            let shards = rt.records;

            for kind in ALL_KINDS {
                for (i, shard) in shards.iter().enumerate() {
                    eprintln!("--- shard {i} LOCAL {kind:?}");
                    debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
                        &machine,
                        &pkey,
                        std::slice::from_ref(shard),
                        vec![*kind],
                        LookupScope::Local,
                    );
                }
                eprintln!("--- GLOBAL {kind:?}");
                debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
                    &machine,
                    &pkey,
                    &shards,
                    vec![*kind],
                    LookupScope::Global,
                );
            }
        }
        "prove" => {
            run_test::<CpuProver<_, _>>(program()).expect("prove + verify failed");
            eprintln!("prove + verify OK");
        }
        "widths" => {
            // Per-chip main-trace WIDTH census.  "Where does the trace area go"
            // is the recurring question behind every density comparison, and
            // answering it otherwise means re-deriving column counts by
            // hand from the `*Cols` struct definitions.  Width is static, so this
            // needs no execution; multiply by the per-chip row count for area.
            let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
                MipsAir::machine(KoalaBearPoseidon2::new());
            let mut rows: Vec<(String, usize, usize)> = machine
                .chips()
                .iter()
                .map(|c| {
                    (
                        MachineAir::<KoalaBear>::name(c),
                        p3_air::BaseAir::<KoalaBear>::width(c).max(1),
                        MachineAir::<KoalaBear>::preprocessed_width(c),
                    )
                })
                .collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1));
            let total: usize = rows.iter().map(|r| r.1).sum();
            eprintln!("{:<28} {:>7} {:>7}  {:>6}", "chip", "main_w", "prep_w", "%main");
            for (name, w, pw) in &rows {
                eprintln!(
                    "{:<28} {:>7} {:>7}  {:>5.1}%",
                    name,
                    w,
                    pw,
                    100.0 * *w as f64 / total as f64
                );
            }
            eprintln!("{:<28} {:>7}", "TOTAL main width", total);
        }
        "lookups" => {
            // Per-chip INTERACTION census, and the GKR cell budget it implies.
            //
            // The LogUp-GKR first layer is a dense `2^num_row_variables ×
            // 2^num_interaction_variables` grid, and every later layer halves
            // it, so the total folded cell count is ~`2^(nrv + niv + 1)` per
            // quadrant.  `niv` is NOT log2 of the interaction count: each chip
            // pads its own interaction count up to a power of two, the padded
            // counts are summed across the shard's chips, and that sum is
            // rounded up to a power of two again.  Both roundings are paid in
            // full — the padded columns are materialised with identity
            // fractions.  So the census that matters is per-chip
            // `next_power_of_two(sends + receives)`, and the only way to move
            // GKR cost is to push the shard-wide sum below a power of two.
            //
            // With a program argument the census is weighted by the REAL
            // per-shard chip sets and heights from an execution, so chips can
            // be ranked by cells rather than by interaction count.
            let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
                MipsAir::machine(KoalaBearPoseidon2::new());
            let kinds = ALL_KINDS;
            eprintln!(
                "{:<28} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6}  {}",
                "chip", "main_w", "prep_w", "send", "recv", "tot", "padded", "by kind"
            );
            let mut static_rows: Vec<(String, usize, usize, usize, usize, usize, String)> = vec![];
            for c in machine.chips() {
                let name = MachineAir::<KoalaBear>::name(c);
                let sends = c.sends().len();
                let recvs = c.receives().len();
                let tot = sends + recvs;
                let padded = tot.max(1).next_power_of_two();
                let mut by_kind: Vec<String> = vec![];
                for k in kinds {
                    let s = c.sends().iter().filter(|i| i.kind == *k).count();
                    let r = c.receives().iter().filter(|i| i.kind == *k).count();
                    if s + r > 0 {
                        by_kind.push(format!("{k:?}:{s}s/{r}r"));
                    }
                }
                static_rows.push((
                    name,
                    p3_air::BaseAir::<KoalaBear>::width(c).max(1),
                    MachineAir::<KoalaBear>::preprocessed_width(c),
                    sends,
                    recvs,
                    padded,
                    by_kind.join(" "),
                ));
            }
            static_rows.sort_by(|a, b| b.5.cmp(&a.5).then(a.0.cmp(&b.0)));
            for (name, w, pw, s, r, padded, kinds_s) in &static_rows {
                eprintln!(
                    "{:<28} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6}  {}",
                    name,
                    w,
                    pw,
                    s,
                    r,
                    s + r,
                    padded,
                    kinds_s
                );
            }

            // Weighted pass: execute and report, per shard, the padded
            // interaction axis the GKR circuit actually pays for.  Skipped
            // unless a real program is named, since the static table above
            // already answers "how wide is each chip".
            if name == "static" {
                eprintln!("\n(no program given: static census only)");
                return;
            }
            // One shard per `execute_record` batch: a real workload's full
            // record set does not fit in host memory (`rt.run()` retains
            // every shard), and the census only ever needs one at a time.
            let mut opts = ZKMCoreOpts::default();
            opts.shard_batch_size = 1;
            let mut rt = Executor::new(program(), opts);
            // A third argument names a bincode-serialised `ZKMStdin`, so the
            // census can run the same input a perf gate does.
            if let Some(p) = std::env::args().nth(3) {
                let bytes = std::fs::read(&p).expect("stdin fixture must be readable");
                let stdin: ZKMStdin =
                    bincode::deserialize(&bytes).expect("stdin fixture must deserialise");
                rt.write_vecs(&stdin.buffer);
                for (proof, vk) in stdin.proofs.iter() {
                    rt.write_proof(proof.clone(), vk.clone());
                }
            }
            // Per-chip accumulators over every shard: real cells (height ×
            // raw interactions) versus paid cells (height × padded
            // interactions).
            let mut real_cells: std::collections::BTreeMap<String, u128> = Default::default();
            let mut paid_cells: std::collections::BTreeMap<String, u128> = Default::default();
            let mut chip_rows: std::collections::BTreeMap<String, u128> = Default::default();
            let mut chip_appear: std::collections::BTreeMap<String, usize> = Default::default();
            let mut grid_cells: u128 = 0;
            let mut niv_hist: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut nrv_hist: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut num_shards = 0usize;
            let mut shard_padded: Vec<usize> = vec![];
            let mut shard_raw: Vec<usize> = vec![];
            let mut niv_raw_hist: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut dense_hist: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut dense_cols: Vec<(usize, u128)> = vec![];
            let mut grid_cells_raw: u128 = 0;
            // The chip SET a shard proves is not the set of chips with events:
            // deferred (precompile / global-memory) events move to their own
            // shards, and the shape config then canonicalises each record to a
            // whole cluster.  Reproduce that pipeline or the census counts
            // precompile columns on every core shard.
            let shape_config = zkm_core_machine::shape::CoreShapeConfig::<KoalaBear>::default();
            let mut deferred =
                zkm_core_executor::ExecutionRecord::new(std::sync::Arc::new(resolve(&name)));
            let split_opts = ZKMCoreOpts::default().split_opts;
            loop {
                let (mut records, done) = rt.execute_record(true).expect("execution failed");
                for rec in records.iter_mut() {
                    deferred.append(&mut rec.defer());
                }
                let mut deferred_shards = deferred.split(done, None, split_opts);
                records.append(&mut deferred_shards);
                for rec in records.iter_mut() {
                    if shape_config.fix_shape(rec).is_ok() {
                        zkm_core_machine::shape::canonicalize_shape_to_cluster(rec);
                    }
                }
                for rec in &records {
                    num_shards += 1;
                    let mut total_padded = 0usize;
                    let mut total_raw = 0usize;
                    let mut total_values: u128 = 0;
                    let mut max_h = 0usize;
                    let mut per_shard: Vec<(String, usize, usize, usize)> = vec![];
                    // Real per-chip row counts: what the GKR slab actually
                    // materialises (the padded tail is analytic, never stored),
                    // so cells are `height × num_interactions`.
                    let heights: std::collections::HashMap<String, usize> =
                        MipsAir::<KoalaBear>::core_heights(rec)
                            .into_iter()
                            .map(|(id, h)| (id.as_str().to_string(), h))
                            .collect();
                    for c in machine.shard_chips(rec) {
                        let name = MachineAir::<KoalaBear>::name(c);
                        let h = heights.get(&name).copied().unwrap_or(0).max(
                            rec.shape
                                .as_ref()
                                .and_then(|s| {
                                    std::str::FromStr::from_str(&name)
                                        .ok()
                                        .and_then(|id| s.height(&id))
                                })
                                .unwrap_or(0),
                        );
                        let h = if h > 0 {
                            h
                        } else {
                            match name.as_str() {
                                "Program" => rec.program.instructions.len(),
                                "Byte" => 1 << 16,
                                _ => 0,
                            }
                        };
                        let tot = c.sends().len() + c.receives().len();
                        let padded = tot.max(1).next_power_of_two();
                        total_padded += padded;
                        total_raw += tot;
                        max_h = max_h.max(h);
                        // Committed cells: the jagged commit pads each chip to a
                        // power-of-two height, and the size CLASS is
                        // `ceil(log2(Σ width × padded height))`.
                        total_values += (p3_air::BaseAir::<KoalaBear>::width(c).max(1) as u128)
                            * (h.max(1).next_power_of_two() as u128);
                        per_shard.push((name, h, tot, padded));
                    }
                    let nrv = max_h.max(1).next_power_of_two().trailing_zeros().max(2) as usize;
                    let niv = total_padded.max(1).next_power_of_two().trailing_zeros() as usize;
                    let niv_raw = total_raw.max(1).next_power_of_two().trailing_zeros() as usize;
                    *niv_raw_hist.entry(niv_raw).or_default() += 1;
                    shard_raw.push(total_raw);
                    let log_dense = 128 - (total_values.max(1) - 1).leading_zeros() as usize;
                    *dense_hist.entry(log_dense).or_default() += 1;
                    // Aggregate committed "columns": total cells divided by the
                    // padded row axis, i.e. how many columns wide the shard looks
                    // once every chip is stacked at the tallest chip's height.
                    let agg_cols = total_values / (1u128 << nrv);
                    dense_cols.push((log_dense, agg_cols));
                    grid_cells_raw += 4u128 << (nrv + niv_raw);
                    *niv_hist.entry(niv).or_default() += 1;
                    *nrv_hist.entry(nrv).or_default() += 1;
                    shard_padded.push(total_padded);
                    // Four quadrant MLEs (n0/n1/d0/d1) over a grid that halves
                    // every layer: 4 · 2^(nrv-1) · 2^niv · 2 folded cells.
                    grid_cells += 4u128 << (nrv + niv);
                    for (name, h, tot, padded) in per_shard {
                        *real_cells.entry(name.clone()).or_default() += h as u128 * tot as u128;
                        *paid_cells.entry(name.clone()).or_default() +=
                            h.max(1) as u128 * padded as u128;
                        *chip_rows.entry(name.clone()).or_default() += h as u128;
                        *chip_appear.entry(name).or_default() += 1;
                    }
                }
                if done {
                    break;
                }
            }
            shard_padded.sort_unstable();
            eprintln!("\nshards = {num_shards}");
            eprintln!(
                "shard padded-interaction sum: min={} median={} max={}",
                shard_padded.first().copied().unwrap_or(0),
                shard_padded.get(shard_padded.len() / 2).copied().unwrap_or(0),
                shard_padded.last().copied().unwrap_or(0)
            );
            eprintln!("num_row_variables histogram      = {nrv_hist:?}");
            eprintln!("num_interaction_variables hist   = {niv_hist:?}");
            eprintln!("GKR grid cells (all shards)      = {grid_cells}");
            shard_raw.sort_unstable();
            eprintln!(
                "shard RAW-interaction sum:    min={} median={} max={}",
                shard_raw.first().copied().unwrap_or(0),
                shard_raw.get(shard_raw.len() / 2).copied().unwrap_or(0),
                shard_raw.last().copied().unwrap_or(0)
            );
            eprintln!("num_interaction_variables (RAW)  = {niv_raw_hist:?}");
            eprintln!("GKR grid cells (RAW axis)        = {grid_cells_raw}");
            eprintln!("committed log_dense histogram    = {dense_hist:?}");
            let mut by_class: std::collections::BTreeMap<usize, Vec<u128>> = Default::default();
            for (c, cols) in &dense_cols {
                by_class.entry(*c).or_default().push(*cols);
            }
            for (c, mut v) in by_class {
                v.sort_unstable();
                eprintln!(
                    "  log_dense={c}: n={} agg_cols min={} median={} max={}",
                    v.len(),
                    v[0],
                    v[v.len() / 2],
                    v[v.len() - 1]
                );
            }
            let mut ranked: Vec<(String, u128, u128, u128, usize)> = paid_cells
                .iter()
                .map(|(n, p)| {
                    (
                        n.clone(),
                        *p,
                        *real_cells.get(n).unwrap_or(&0),
                        *chip_rows.get(n).unwrap_or(&0),
                        *chip_appear.get(n).unwrap_or(&0),
                    )
                })
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            let paid_total: u128 = ranked.iter().map(|r| r.1).sum();
            eprintln!(
                "\n{:<28} {:>16} {:>16} {:>7} {:>14} {:>6}",
                "chip", "paid_cells", "real_cells", "shards", "rows(sum)", "%paid"
            );
            for (name, paid, real, rows, appear) in &ranked {
                eprintln!(
                    "{:<28} {:>16} {:>16} {:>7} {:>14} {:>5.1}%",
                    name,
                    paid,
                    real,
                    appear,
                    rows,
                    100.0 * *paid as f64 / paid_total as f64
                );
            }
            eprintln!("{:<28} {:>16}", "TOTAL paid (chip cols)", paid_total);
        }
        "rows" => {
            // Per-chip ROW census.  Every row is a REAL instruction now: the
            // Instruction bus and its synthetic dependency rows are gone
            // (DivRem/CloClz/Misc prove their sub-operations in-row).
            let mut rt = Executor::new(program(), ZKMCoreOpts::default());
            rt.run().expect("execution failed");
            let mut cpu = 0usize;
            let mut tot = 0usize;
            eprintln!("{:<22} {:>10}", "alu chip", "rows");
            for rec in &rt.records {
                cpu += rec.cpu_events.len();
            }
            let mut report = |name: &str, rows: usize| {
                tot += rows;
                eprintln!("{name:<22} {rows:>10}");
            };
            macro_rules! census {
                ($field:ident, $label:literal) => {{
                    let mut r = 0usize;
                    for rec in &rt.records {
                        r += rec.$field.len();
                    }
                    report($label, r);
                }};
            }
            census!(add_sub_events, "AddSub");
            census!(bitwise_events, "Bitwise");
            census!(shift_left_events, "ShiftLeft");
            census!(shift_right_events, "ShiftRight");
            census!(lt_events, "Lt");
            census!(cloclz_events, "CloClz");
            census!(mul_events, "Mul");
            census!(divrem_events, "DivRem");
            eprintln!("{:<22} {:>10}", "ALU TOTAL", tot);
            eprintln!("cpu_events (one per executed instruction) = {cpu}");
            eprintln!("shards = {}", rt.records.len());
        }
        "dump" => {
            // Write <out>/<name>/{program.bin,stdin.bin} for every test
            // artifact, in the format `find_maximal_shapes --list` consumes —
            // the shape-artifact regeneration sweep runs the SAME corpus the
            // all-mode proves.
            let out = std::env::args().nth(2).expect("dump needs an output dir");
            let artifacts: &[(&str, &[u8])] = &[
                ("sha2-rust", test_artifacts::SHA2_RUST_ELF),
                ("fibonacci", test_artifacts::FIBONACCI_ELF),
                ("hello-world", test_artifacts::HELLO_WORLD_ELF),
                ("poseidon2-permute", test_artifacts::POSEIDON2_PERMUTE_ELF),
                ("sha2", test_artifacts::SHA2_ELF),
                ("sha-extend", test_artifacts::SHA_EXTEND_ELF),
                ("sha-compress", test_artifacts::SHA_COMPRESS_ELF),
                ("keccak-sponge", test_artifacts::KECCAK_SPONGE_ELF),
                ("ed25519", test_artifacts::ED25519_ELF),
                ("cycle-tracker", test_artifacts::CYCLE_TRACKER_ELF),
                ("ed-add", test_artifacts::ED_ADD_ELF),
                ("ed-decompress", test_artifacts::ED_DECOMPRESS_ELF),
                ("secp256k1-add", test_artifacts::SECP256K1_ADD_ELF),
                ("secp256k1-decompress", test_artifacts::SECP256K1_DECOMPRESS_ELF),
                ("secp256k1-double", test_artifacts::SECP256K1_DOUBLE_ELF),
                ("secp256r1-add", test_artifacts::SECP256R1_ADD_ELF),
                ("secp256r1-decompress", test_artifacts::SECP256R1_DECOMPRESS_ELF),
                ("secp256r1-double", test_artifacts::SECP256R1_DOUBLE_ELF),
                ("bn254-add", test_artifacts::BN254_ADD_ELF),
                ("bn254-double", test_artifacts::BN254_DOUBLE_ELF),
                ("bn254-mul", test_artifacts::BN254_MUL_ELF),
                ("secp256k1-mul", test_artifacts::SECP256K1_MUL_ELF),
                ("bls12381-add", test_artifacts::BLS12381_ADD_ELF),
                ("bls12381-double", test_artifacts::BLS12381_DOUBLE_ELF),
                ("bls12381-mul", test_artifacts::BLS12381_MUL_ELF),
                ("uint256-mul", test_artifacts::UINT256_MUL_ELF),
                ("bls12381-decompress", test_artifacts::BLS12381_DECOMPRESS_ELF),
                ("bls12381-fp", test_artifacts::BLS12381_FP_ELF),
                ("bls12381-fp2-mul", test_artifacts::BLS12381_FP2_MUL_ELF),
                ("bls12381-fp2-addsub", test_artifacts::BLS12381_FP2_ADDSUB_ELF),
                ("bn254-fp", test_artifacts::BN254_FP_ELF),
                ("bn254-fp2-addsub", test_artifacts::BN254_FP2_ADDSUB_ELF),
                ("bn254-fp2-mul", test_artifacts::BN254_FP2_MUL_ELF),
                ("u256xu2048-mul", test_artifacts::U256XU2048_MUL_ELF),
                ("unconstrained", test_artifacts::UNCONSTRAINED_ELF),
                // sha3-chain exercises KeccakSponge at a chained density no
                // other artifact reaches (test_sha3_chain_prove_simple was
                // shape-stale before it joined the corpus).  No stdin: the
                // guest hardcodes its input.
                ("sha3-chain", test_artifacts::SHA3_CHAIN_ELF),
            ];
            fn hexb2(s: &str) -> Vec<u8> {
                (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                    .collect()
            }
            for (name, elf) in artifacts {
                let dir = std::path::Path::new(&out).join(name);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join("program.bin"), elf).unwrap();
                let stdin = match *name {
                    "sha2-rust" => {
                        let input = b"hello world".to_vec();
                        let expected = hexb2(
                            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
                        );
                        let mut s = ZKMStdin::new();
                        s.write(&expected);
                        s.write(&input);
                        s
                    }
                    "secp256k1-decompress" => ZKMStdin::from(
                        hexb2("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
                            .as_slice(),
                    ),
                    "secp256r1-decompress" => ZKMStdin::from(
                        hexb2("036b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296")
                            .as_slice(),
                    ),
                    "bls12381-decompress" => ZKMStdin::from(
                        hexb2("97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb")
                            .as_slice(),
                    ),
                    _ => ZKMStdin::new(),
                };
                std::fs::write(dir.join("stdin.bin"), bincode::serialize(&stdin).unwrap()).unwrap();
                eprintln!("dumped {name}");
            }
        }
        "all" => {
            // Prove + verify EVERY test-artifact guest.  This is the gate an
            // architecture-level change must pass before anything downstream
            // (goldens, vk generation): the four ad-hoc fixtures cover the
            // integer core, but only the full set exercises every precompile
            // chip, every syscall path, and the panic/unconstrained edges.
            let artifacts: &[(&str, &[u8])] = &[
                ("sha2-rust", test_artifacts::SHA2_RUST_ELF),
                ("fibonacci", test_artifacts::FIBONACCI_ELF),
                ("hello-world", test_artifacts::HELLO_WORLD_ELF),
                ("poseidon2-permute", test_artifacts::POSEIDON2_PERMUTE_ELF),
                ("sha2", test_artifacts::SHA2_ELF),
                ("sha-extend", test_artifacts::SHA_EXTEND_ELF),
                ("sha-compress", test_artifacts::SHA_COMPRESS_ELF),
                ("keccak-sponge", test_artifacts::KECCAK_SPONGE_ELF),
                ("ed25519", test_artifacts::ED25519_ELF),
                ("cycle-tracker", test_artifacts::CYCLE_TRACKER_ELF),
                ("ed-add", test_artifacts::ED_ADD_ELF),
                ("ed-decompress", test_artifacts::ED_DECOMPRESS_ELF),
                ("secp256k1-add", test_artifacts::SECP256K1_ADD_ELF),
                ("secp256k1-decompress", test_artifacts::SECP256K1_DECOMPRESS_ELF),
                ("secp256k1-double", test_artifacts::SECP256K1_DOUBLE_ELF),
                ("secp256r1-add", test_artifacts::SECP256R1_ADD_ELF),
                ("secp256r1-decompress", test_artifacts::SECP256R1_DECOMPRESS_ELF),
                ("secp256r1-double", test_artifacts::SECP256R1_DOUBLE_ELF),
                ("bn254-add", test_artifacts::BN254_ADD_ELF),
                ("bn254-double", test_artifacts::BN254_DOUBLE_ELF),
                ("bn254-mul", test_artifacts::BN254_MUL_ELF),
                ("secp256k1-mul", test_artifacts::SECP256K1_MUL_ELF),
                ("bls12381-add", test_artifacts::BLS12381_ADD_ELF),
                ("bls12381-double", test_artifacts::BLS12381_DOUBLE_ELF),
                ("bls12381-mul", test_artifacts::BLS12381_MUL_ELF),
                ("uint256-mul", test_artifacts::UINT256_MUL_ELF),
                ("bls12381-decompress", test_artifacts::BLS12381_DECOMPRESS_ELF),
                ("bls12381-fp", test_artifacts::BLS12381_FP_ELF),
                ("bls12381-fp2-mul", test_artifacts::BLS12381_FP2_MUL_ELF),
                ("bls12381-fp2-addsub", test_artifacts::BLS12381_FP2_ADDSUB_ELF),
                ("bn254-fp", test_artifacts::BN254_FP_ELF),
                ("bn254-fp2-addsub", test_artifacts::BN254_FP2_ADDSUB_ELF),
                ("bn254-fp2-mul", test_artifacts::BN254_FP2_MUL_ELF),
                ("u256xu2048-mul", test_artifacts::U256XU2048_MUL_ELF),
                ("unconstrained", test_artifacts::UNCONSTRAINED_ELF),
                ("sha3-chain", test_artifacts::SHA3_CHAIN_ELF),
            ];
            // Fixtures that READ STDIN get their canonical inputs; an empty
            // stream hits the executor's "insufficient input data" error.
            fn hexb(s: &str) -> Vec<u8> {
                (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                    .collect()
            }
            fn artifact_stdin(name: &str) -> ZKMStdin {
                match name {
                    "sha2-rust" => {
                        // The guest reads (expected_hash, input) via io::read.
                        let input = b"hello world".to_vec();
                        let expected =
                            hexb("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
                        let mut stdin = ZKMStdin::new();
                        stdin.write(&expected);
                        stdin.write(&input);
                        stdin
                    }
                    // The curve generators, SEC1-compressed.
                    "secp256k1-decompress" => ZKMStdin::from(
                        hexb("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
                            .as_slice(),
                    ),
                    "secp256r1-decompress" => ZKMStdin::from(
                        hexb("036b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296")
                            .as_slice(),
                    ),
                    "bls12381-decompress" => ZKMStdin::from(
                        hexb("97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb")
                            .as_slice(),
                    ),
                    _ => ZKMStdin::new(),
                }
            }

            // Optional comma-separated subset: `playground all a,b,c`.
            let filter: Option<Vec<String>> =
                std::env::args().nth(2).map(|f| f.split(',').map(str::to_string).collect());

            let mut failed: Vec<&str> = vec![];
            for (name, elf) in artifacts {
                if let Some(f) = &filter {
                    if !f.iter().any(|x| x == name) {
                        continue;
                    }
                }
                eprint!("{name:<24} ");
                let program = Program::from(elf).expect("artifact must parse");
                let stdin = artifact_stdin(name);
                match std::panic::catch_unwind(|| {
                    run_test_io::<CpuProver<_, _>>(program, stdin).map(|_| ())
                }) {
                    Ok(Ok(_)) => eprintln!("PASS"),
                    Ok(Err(e)) => {
                        eprintln!("FAIL: {e:?}");
                        failed.push(name);
                    }
                    Err(_) => {
                        eprintln!("PANIC");
                        failed.push(name);
                    }
                }
            }
            if failed.is_empty() {
                eprintln!("ALL {} ARTIFACTS PASS", artifacts.len());
            } else {
                panic!("{} artifacts FAILED: {failed:?}", failed.len());
            }
        }
        other => {
            panic!(
                "unknown command {other:?}; expected execute | buses | prove | widths | rows | all"
            )
        }
    }
}
