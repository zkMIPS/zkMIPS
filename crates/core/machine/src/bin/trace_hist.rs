//! Histogram a `TRACE_FILE` pc trace (u32 BE per cycle): prints `pc count`
//! lines, most frequent first, for `addr2line -i` attribution.
//!
//!   cargo run --release --example trace_hist -- trace.bin > pcs.txt
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufReader, Read, Write},
};

fn main() {
    let path = env::args().nth(1).expect("trace file");
    let mut reader = BufReader::with_capacity(1 << 24, File::open(&path).expect("open trace"));
    let mut counts: HashMap<u32, u64> = HashMap::with_capacity(1 << 20);
    let mut buf = vec![0u8; 1 << 24];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        for chunk in buf[..n - n % 4].chunks_exact(4) {
            let pc = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            *counts.entry(pc).or_insert(0) += 1;
            total += 1;
        }
    }
    let mut v: Vec<(u32, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let out = std::io::stdout();
    let mut out = out.lock();
    writeln!(out, "# total_cycles {total} unique_pcs {}", v.len()).unwrap();
    for (pc, c) in v {
        writeln!(out, "{pc:08x} {c}").unwrap();
    }
}
