//! List the enumerated recursion shapes (index, variant, child band summary)
//! so a targeted `build_compress_vks --start/--end/--indices` run can cover
//! just the compose / deferred / shrink classes.
use std::collections::BTreeSet;
use zkm_core_machine::shape::CoreShapeConfig;
use zkm_prover::{components::DefaultProverComponents, shapes::ZKMProofShape, ZKMProver, REDUCE_BATCH_SIZE};

fn main() {
    let prover = ZKMProver::<DefaultProverComponents>::new();
    let core = CoreShapeConfig::default();
    let rec = prover.compress_shape_config.as_ref().expect("recursion shape config");
    let all: BTreeSet<ZKMProofShape> =
        ZKMProofShape::generate(&core, rec, REDUCE_BATCH_SIZE).collect();
    for (i, s) in all.iter().enumerate() {
        let (variant, children) = match s {
            ZKMProofShape::Recursion(v) => ("Recursion", v.as_slice()),
            ZKMProofShape::Compress(v) => ("Compress", v.as_slice()),
            ZKMProofShape::Deferred(v) => ("Deferred", v.as_slice()),
            ZKMProofShape::Shrink(v) => ("Shrink", std::slice::from_ref(v)),
        };
        let summary: Vec<String> = children
            .iter()
            .map(|c| {
                c.inner
                    .iter()
                    .find(|(k, _)| k == "MemoryVar")
                    .map(|(_, h)| h.to_string())
                    .unwrap_or_else(|| format!("n{}", c.inner.len()))
            })
            .collect();
        println!("{i} {variant} arity={} memvar=[{}]", children.len(), summary.join(","));
    }
}
