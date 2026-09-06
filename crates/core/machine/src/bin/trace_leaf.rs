//! Attribute the cycles a pc trace spends inside leaf functions (memcpy,
//! memcmp, allocator, keccak …) to the CALL SITE that entered them: the pc
//! executed just before the first instruction of the leaf.  Inline-chain
//! symbolization cannot see callers of out-of-line leaves; this can.
//!
//!   trace_leaf trace.bin leaves.txt   (lines: `start size name`, decimal)
//! prints `callsite_pc cycles name`, most expensive first.
use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::{BufReader, Read, Write},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let leaves: Vec<(u32, u32, String)> = fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.to_string()))
        })
        .collect();
    let mut reader = BufReader::with_capacity(1 << 24, File::open(&args[1]).unwrap());
    let mut buf = vec![0u8; 1 << 24];
    let mut prev: u32 = 0;
    // (leaf index, call site) while inside a leaf
    let mut inside: Option<(usize, u32)> = None;
    let mut counts: HashMap<(usize, u32), u64> = HashMap::new();
    let mut leaf_total = vec![0u64; leaves.len()];
    loop {
        let n = reader.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for chunk in buf[..n - n % 4].chunks_exact(4) {
            let pc = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if let Some((li, site)) = inside {
                let (s, sz, _) = &leaves[li];
                if pc >= *s && pc < s + sz {
                    *counts.entry((li, site)).or_insert(0) += 1;
                    leaf_total[li] += 1;
                } else {
                    inside = None;
                }
            }
            if inside.is_none() {
                if let Some(li) = leaves.iter().position(|(s, _, _)| *s == pc) {
                    inside = Some((li, prev));
                    *counts.entry((li, prev)).or_insert(0) += 1;
                    leaf_total[li] += 1;
                }
            }
            prev = pc;
        }
    }
    let out = std::io::stdout();
    let mut out = out.lock();
    for (li, (_, _, name)) in leaves.iter().enumerate() {
        writeln!(out, "# leaf {name} total_cycles {}", leaf_total[li]).unwrap();
    }
    let mut v: Vec<((usize, u32), u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for ((li, site), c) in v {
        writeln!(out, "{site:08x} {c} {}", leaves[li].2).unwrap();
    }
}
