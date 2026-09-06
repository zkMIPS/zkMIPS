//! HEIGHT-FORGERY DE-RISK HARNESS.
//!
//! The VK-identity change drops the per-prep-domain height loop
//! from `vk.hash` (crates/prover/src/types.rs:92-99) so that `VK =
//! f(chip-SET)` instead of `f(chip-SET, per-chip-heights)`.  That change
//! is IRREVERSIBLE (it re-keys the recursion vk_map).  Its safety rests
//! on a single claim:
//!
//!   > per-chip heights are REDUNDANTLY bound on the verify path — even
//!   > with the hash height loop removed, a forged height is rejected by
//!   > the substrate (witnessed degree bits → `full_geq` padding mask,
//!   > the LogUp-GKR reconstruction identity, the zerocheck identity,
//!   > and the jagged reduction).
//!
//! This module PROVES that claim EMPIRICALLY, BEFORE the hash change, by
//! FORGING a height in a real FIX-off shard proof and confirming the
//! honest verify path REJECTS it.  The hash is left UNTOUCHED here
//! (`vk.hash` still loops the heights) — these tests show the rejection
//! does NOT come from the hash loop (the host `verify_shard` path does
//! not even recompute the VK hash; it re-derives the transcript from the
//! observed `chip_heights` and consumes the `degree` bits directly).
//!
//! WHERE THE HEIGHT LIVES IN A FIX-off HOST SHARD PROOF
//! (`BasefoldShardProof<F, EF>`):
//!   * `chip_heights[name]`            — observed into the transcript
//!     prologue (shard_level/verifier.rs:199-204).
//!   * `opened_values.chips[c].quotient[0]`— the per-chip `degree` =
//!     BIG-ENDIAN (MSB at index 0) boolean decomposition of the REAL
//!     height (shard_level/prover.rs:948-969).  Consumed as the
//!     `full_geq` threshold in BOTH the LogUp-GKR reconstruction
//!     (verifier.rs:1714-1727) and the zerocheck (verifier.rs:1072-1084).
//!   * `opened_values.chips[c].log_degree` — the per-chip log height.
//!
//! A genuine height-LIE flips a `degree` bit: OVER-claim sets a higher
//! bit (claims a taller chip than the trace), UNDER-claim clears the
//! real top bit (claims a shorter chip).  Either perturbs the `full_geq`
//! padding mask → the reconstructed numerator/denominator diverge from
//! the GKR round walk → `LogupGkr(...)` reject.
//!
//! WHICH CHECK CATCHES IT is reported via the `BasefoldVerifyError`
//! variant carried inside `MachineVerificationError::InvalidShardProof`:
//!   * `LogupGkr(_)`  — reconstruction / full_geq mask (PRIMARY host
//!     substrate height bind).
//!   * `Zerocheck(_)` — zerocheck identity (also full_geq over degree).
//!   * `JaggedPcs(_)` — jagged reduction.
//!
//! Run (CPU, release, target on /data):
//!   CUDA_VISIBLE_DEVICES="" CARGO_TARGET_DIR=/data/ziren-divfix-target \
//!     cargo test --release -p zkm-core-machine \
//!       --features ... stage0_forgery -- --nocapture --test-threads=1
//!
//! NOTE: these are `#[ignore]`d by default (they prove real FIX-off
//! proofs, which is multi-second) — run them explicitly with
//! `-- --ignored`.

#![cfg(test)]
#![allow(clippy::type_complexity)]

use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear;

use zkm_core_executor::{Executor, Program, ZKMContext};
use zkm_pcs::{
    koala_bear_poseidon2::KoalaBearPoseidon2, CpuProver, MachineProof, MachineProver,
    MachineVerificationError, ShardProof, StarkGenericConfig, StarkMachine, ZKMCoreOpts,
};

use crate::io::ZKMStdin;
use crate::mips::MipsAir;
use crate::programs::tests::{
    fibonacci_program, sha3_chain_program, simple_memory_program, ssz_withdrawals_program,
};
use crate::utils::{prove_with_context, setup_logger};

type SC = KoalaBearPoseidon2;
type Val = KoalaBear;
type Challenge = p3_field::extension::BinomialExtensionField<KoalaBear, 4>;

/// Prove `program` on `stdin` with FIX_CORE_SHAPES OFF (raw heights —
/// the `shape_config = None` path).  Returns the `MachineProof` and a
/// freshly-built machine+vk so the caller can verify (and re-verify
/// mutated copies) honestly.
fn prove_fixoff(
    program: Program,
    stdin: ZKMStdin,
) -> (MachineProof<SC>, StarkMachine<SC, MipsAir<Val>>, zkm_pcs::StarkVerifyingKey<SC>) {
    let config = KoalaBearPoseidon2::new();
    let machine = MipsAir::machine(config);
    let prover = CpuProver::<_, _>::new(machine);

    let (pk, _) = prover.setup(&program);

    // FIX-off: shape_config = None ⇒ records stay at RAW heights.
    // `prove_with_context` runs the program internally from `stdin`.
    let (proof, _output, _cycles) = prove_with_context::<SC, CpuProver<_, _>>(
        &prover,
        &pk,
        program.clone(),
        &stdin,
        ZKMCoreOpts::default(),
        ZKMContext::default(),
        None,
    )
    .expect("FIX-off prove_with_context");

    // Independent machine + vk for verification.
    let config = KoalaBearPoseidon2::new();
    let machine = MipsAir::machine(config);
    let (_pk, vk) = machine.setup(&program);
    (proof, machine, vk)
}

/// Verify a (possibly mutated) `MachineProof` with a fresh challenger.
fn verify(
    machine: &StarkMachine<SC, MipsAir<Val>>,
    vk: &zkm_pcs::StarkVerifyingKey<SC>,
    proof: &MachineProof<SC>,
) -> Result<(), MachineVerificationError<SC>> {
    let mut challenger = machine.config().challenger();
    machine.verify(vk, proof, &mut challenger)?;
    // `StarkMachine::verify` is per-shard, so on its own it is not the whole
    // verification and this harness's "accepted" would overstate what was
    // checked.
    //
    // Note which adversary this covers.  The prologue observes EVERY public
    // value, `global_cumulative_sum` included, so mutating a finished proof's
    // digest desyncs Fiat-Shamir and the per-shard verify already rejects it —
    // no mutation this harness can construct reaches the line below.  What the
    // cross-shard sum catches is a dishonest PROVER: individually valid,
    // FS-consistent shards whose global sends and receives do not cancel.
    // Exercising that needs a perturbed record fed through proving, which this
    // harness does not model; the call is here so a future case of that shape
    // cannot pass by verifying only half the argument.
    crate::utils::global_sum::verify_global_cumulative_sum(vk, &proof.shard_proofs)
}

/// Map a verify result to a short tag naming WHICH substrate check
/// rejected (so the report can attribute the rejection).
fn reject_tag(res: &Result<(), MachineVerificationError<SC>>) -> String {
    match res {
        Ok(()) => "ACCEPTED".to_string(),
        Err(MachineVerificationError::InvalidShardProof(inner)) => {
            // The Display of BasefoldVerifyError carries the variant
            // prefix (LogupGkr/Zerocheck/JaggedPcs/...).
            format!("REJECTED::InvalidShardProof[{inner}]")
        }
        Err(other) => format!("REJECTED::{other:?}"),
    }
}

/// Find the first shard whose `basefold_shard_proof` carries an
/// `opened_values` with at least one chip that has a non-trivial
/// `degree` (`quotient[0]`), and return its index + the chip index +
/// the chip name.  Prefers a height-VARIED chip (real height > 2) so the
/// forgery is a meaningful "taller/shorter" lie.
///
/// CRITICAL: `opened_values.chips` is NAME-SORTED at proof time
/// (shard_level/prover.rs:927-931), so the `ci`-th opening corresponds to
/// the `ci`-th NAME in the name-sorted chip set — NOT the `chip_ordering`
/// HashMap order.  We recover the name from the sorted `chip_ordering`
/// keys so the transcript (`chip_heights[name]`) and the degree
/// (`opened_values.chips[ci]`) refer to the SAME chip.
fn pick_forge_target(proof: &MachineProof<SC>) -> (usize, usize, String, usize) {
    let mut fallback: Option<(usize, usize, String, usize)> = None;
    for (si, sp) in proof.shard_proofs.iter().enumerate() {
        let Some(bf) = sp.basefold_shard_proof.as_ref() else { continue };
        // Name-sorted chip names (the order `opened_values.chips` uses) —
        // `chip_heights` is a name-sorted BTreeMap over the same set.
        let sorted_names: Vec<String> = bf.chip_heights.keys().cloned().collect();
        for (ci, opening) in bf.opened_values.chips.iter().enumerate() {
            if !opening.quotient.first().map(|q| !q.is_empty()).unwrap_or(false) {
                continue;
            }
            let name = sorted_names.get(ci).cloned().unwrap_or_else(|| format!("chip{ci}"));
            let log_h = opening.log_degree;
            // Prefer a height-varied chip (log_h >= 2 so over/under both move).
            if log_h >= 2 {
                return (si, ci, name, log_h);
            }
            fallback.get_or_insert((si, ci, name, log_h));
        }
    }
    fallback.expect("no shard with a non-empty degree (quotient[0]) found")
}

/// Decode the big-endian degree bits (`quotient[0]`) into the claimed
/// height (the index of the single set bit), or `None` if not a clean
/// power-of-two indicator.
fn degree_bits_to_height(degree: &[Challenge]) -> Option<usize> {
    // BIG-ENDIAN, MSB at index 0. height = 2^(set-bit position from LSB).
    let bit_len = degree.len();
    let mut set: Option<usize> = None;
    for (i, &b) in degree.iter().enumerate() {
        if b == Challenge::ONE {
            let shift = bit_len - 1 - i;
            if set.is_some() {
                return None; // more than one bit set — not a clean indicator
            }
            set = Some(shift);
        } else if b != Challenge::ZERO {
            return None;
        }
    }
    set.map(|s| 1usize << s)
}

// ───────────────────────────── CONTROL ────────────────────────────────

/// CONTROL: an honest FIX-off proof VERIFIES.  Proves the harness's
/// verify path is REAL (not trivially rejecting everything), so a later
/// rejection in the forgery tests is meaningful.
#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_control_fixoff_honest_verifies() {
    setup_logger();
    let (proof, machine, vk) = prove_fixoff(fibonacci_program(), ZKMStdin::new());
    let res = verify(&machine, &vk, &proof);
    eprintln!("[STAGE0][CONTROL][fibonacci FIX-off] honest verify = {}", reject_tag(&res));
    assert!(res.is_ok(), "honest FIX-off proof must verify (control)");
}

/// CONTROL: an honest FIX-ON proof VERIFIES (sanity that the verify
/// path agrees on the standard configuration too).
#[test]
#[ignore = "proves a real FIX-on core proof (multi-second); run with --ignored"]
fn stage0_control_fixon_honest_verifies() {
    setup_logger();
    // FIX-on path: use the standard run_test_core helper (shape_config Some).
    let mut program = fibonacci_program();
    let shape_config = crate::shape::CoreShapeConfig::<Val>::default();
    shape_config.fix_preprocessed_shape(&mut program).unwrap();
    let mut runtime = Executor::new(program.clone(), ZKMCoreOpts::default());
    runtime.run().unwrap();
    let res = crate::utils::run_test_core::<CpuProver<_, _>>(
        runtime,
        ZKMStdin::new(),
        Some(&shape_config),
    );
    eprintln!(
        "[STAGE0][CONTROL][fibonacci FIX-on] honest verify = {}",
        if res.is_ok() { "ACCEPTED" } else { "REJECTED" }
    );
    assert!(res.is_ok(), "honest FIX-on proof must verify (control)");
}

// ───────────────────────────── FORGERY ────────────────────────────────

/// Core forgery driver.  Proves a FIX-off proof, confirms it verifies
/// honestly, then for the chosen target chip applies the supplied
/// `mutate` to the proof copy and re-verifies.  Asserts the outcome
/// matches `expect_reject`.  Returns the reject tag so the caller can
/// log it.
///
/// The degree-masked last-layer reconstruction (the height anchor) always
/// runs: `22616c7a` made it unconditional and removed the
/// `ZIREN_LOGUP_RECONSTRUCTION` escape hatch.
///
/// `expect_reject`: the de-risk assertion — `true` means the forgery
/// MUST be rejected (heights redundantly bound on this path); `false`
/// documents a path where the forgery SURVIVES (records, not a soundness
/// claim).
fn run_forgery(
    label: &str,
    program: Program,
    stdin: ZKMStdin,
    expect_reject: bool,
    mutate: impl Fn(&mut ShardProof<SC>, usize, &str),
) -> String {
    let (proof, machine, vk) = prove_fixoff(program, stdin);

    // Honest control first (per-call) — guards against a proof that is
    // already invalid for unrelated reasons, AND confirms the verify
    // path accepts honest proofs.
    let honest = verify(&machine, &vk, &proof);
    assert!(
        honest.is_ok(),
        "[{label}] honest FIX-off proof must verify before forging (got {})",
        reject_tag(&honest)
    );

    let (si, ci, name, log_h) = pick_forge_target(&proof);
    eprintln!(
        "[STAGE0][FORGE][{label}] target shard={si} chip_idx={ci} chip='{name}' \
         log_height={log_h}"
    );

    let mut forged = proof.clone();
    mutate(&mut forged.shard_proofs[si], ci, &name);

    let res = verify(&machine, &vk, &forged);
    let tag = reject_tag(&res);
    eprintln!("[STAGE0][FORGE][{label}] forged verify = {tag}");

    if expect_reject {
        assert!(
            res.is_err(),
            "[{label}] FORGED HEIGHT SURVIVED — heights are NOT \
             redundantly bound (BLOCKER)"
        );
    } else {
        assert!(res.is_ok(), "[{label}] expected forgery to SURVIVE but it was rejected: {tag}");
    }
    tag
}

/// OVER-CLAIM the degree bits: set a HIGHER bit (claim a taller chip
/// than the real trace).  Mutates `quotient[0]`, `log_degree`, and the
/// transcript `chip_heights` consistently so the lie is a coherent
/// "this chip is 2x taller" claim.
fn overclaim(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let opening = &mut bf.opened_values.chips[ci];
    let degree = &mut opening.quotient[0];
    let bit_len = degree.len();
    let real_h = degree_bits_to_height(degree);
    // Find the current top set bit; set the bit ABOVE it (one taller
    // power of two). Clear the old top bit so it stays a clean indicator
    // (claims 2x the height).
    let mut top: Option<usize> = None;
    for (i, b) in degree.iter().enumerate() {
        if *b == Challenge::ONE {
            top = Some(bit_len - 1 - i); // shift-from-LSB
        }
    }
    let top_shift = top.unwrap_or(0);
    let new_shift = (top_shift + 1).min(bit_len - 1);
    // big-endian index of `new_shift`
    let new_idx = bit_len - 1 - new_shift;
    let old_idx = bit_len - 1 - top_shift;
    degree[old_idx] = Challenge::ZERO;
    degree[new_idx] = Challenge::ONE;
    opening.log_degree = new_shift;
    bf.chip_heights.insert(name.to_string(), 1usize << new_shift);
    eprintln!(
        "[STAGE0][FORGE] OVER-claim chip='{name}': real_height={:?} -> claimed 2^{new_shift}",
        real_h
    );
}

/// UNDER-CLAIM the degree bits: clear the real top set bit (claim a
/// SHORTER chip than the real trace).
fn underclaim(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let opening = &mut bf.opened_values.chips[ci];
    let degree = &mut opening.quotient[0];
    let bit_len = degree.len();
    let real_h = degree_bits_to_height(degree);
    // Find the top set bit and clear it; set the bit below it (half the
    // height) so it remains a clean single-bit indicator.
    let mut top: Option<usize> = None;
    for (i, b) in degree.iter().enumerate() {
        if *b == Challenge::ONE {
            top = Some(bit_len - 1 - i);
        }
    }
    let top_shift = top.unwrap_or(1).max(1);
    let new_shift = top_shift - 1;
    let old_idx = bit_len - 1 - top_shift;
    let new_idx = bit_len - 1 - new_shift;
    degree[old_idx] = Challenge::ZERO;
    degree[new_idx] = Challenge::ONE;
    opening.log_degree = new_shift;
    bf.chip_heights.insert(name.to_string(), 1usize << new_shift);
    eprintln!(
        "[STAGE0][FORGE] UNDER-claim chip='{name}': real_height={:?} -> claimed 2^{new_shift}",
        real_h
    );
}

/// Forge ONLY the `degree` bits (`quotient[0]`) — leave the transcript
/// `chip_heights` and `log_degree` HONEST.  This is the SHARPEST
/// de-risk probe: it isolates the NON-TRANSCRIPT height-binding
/// substrate (the `full_geq` padding mask over the degree bits → the
/// LogUp-GKR reconstruction identity), the exact substrate the
/// VK-identity change relies on once heights leave both `vk.hash` AND the
/// transcript framing.  If THIS rejects, the degree-bit binding is
/// load-bearing on its own.  Over-claim: set a higher bit (taller chip).
fn forge_degree_only_overclaim(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let opening = &mut bf.opened_values.chips[ci];
    let degree = &mut opening.quotient[0];
    let bit_len = degree.len();
    let real_h = degree_bits_to_height(degree);
    let mut top: Option<usize> = None;
    for (i, b) in degree.iter().enumerate() {
        if *b == Challenge::ONE {
            top = Some(bit_len - 1 - i);
        }
    }
    let top_shift = top.unwrap_or(0);
    let new_shift = (top_shift + 1).min(bit_len - 1);
    let new_idx = bit_len - 1 - new_shift;
    let old_idx = bit_len - 1 - top_shift;
    degree[old_idx] = Challenge::ZERO;
    degree[new_idx] = Challenge::ONE;
    // NOTE: log_degree and chip_heights LEFT HONEST on purpose.
    eprintln!(
        "[STAGE0][FORGE] DEGREE-only OVER-claim chip='{name}': real_height={:?} -> degree bits claim 2^{new_shift} \
         (transcript log_height LEFT HONEST)",
        real_h
    );
}

/// Degree-only UNDER-claim (clear the real top degree bit; transcript
/// left honest).
fn forge_degree_only_underclaim(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let opening = &mut bf.opened_values.chips[ci];
    let degree = &mut opening.quotient[0];
    let bit_len = degree.len();
    let real_h = degree_bits_to_height(degree);
    let mut top: Option<usize> = None;
    for (i, b) in degree.iter().enumerate() {
        if *b == Challenge::ONE {
            top = Some(bit_len - 1 - i);
        }
    }
    let top_shift = top.unwrap_or(1).max(1);
    let new_shift = top_shift - 1;
    let old_idx = bit_len - 1 - top_shift;
    let new_idx = bit_len - 1 - new_shift;
    degree[old_idx] = Challenge::ZERO;
    degree[new_idx] = Challenge::ONE;
    // NOTE: log_degree and chip_heights LEFT HONEST on purpose.
    eprintln!(
        "[STAGE0][FORGE] DEGREE-only UNDER-claim chip='{name}': real_height={:?} -> degree bits claim 2^{new_shift} \
         (transcript log_height LEFT HONEST)",
        real_h
    );
}

/// Forge ONLY the transcript `chip_heights` (leave the degree bits
/// honest) — isolates whether the transcript observe alone binds height.
fn forge_transcript_only(sp: &mut ShardProof<SC>, _ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let cur = bf.chip_heights.get(name).copied().unwrap_or(0);
    let lie = cur.wrapping_add(1);
    bf.chip_heights.insert(name.to_string(), lie);
    eprintln!("[STAGE0][FORGE] TRANSCRIPT-only chip='{name}': height {cur} -> {lie}");
}

// ─────────────────────────────────────────────────────────────────────
// fibonacci (height-varied MIPS)
//
// Two height-binding mechanisms exist on the host shard-verify path:
//   (A) TRANSCRIPT — `chip_heights` is observed into the Fiat-Shamir
//       prologue.  Any height lie that touches it re-derives different
//       challenges → grinding / consistency rejects.  Binds REGARDLESS
//       of the `ZIREN_LOGUP_RECONSTRUCTION` gate.
//   (B) DEGREE-MASKED RECONSTRUCTION — the `degree` bits (`quotient[0]`)
//       feed `full_geq`, the per-chip padding mask; the last-layer
//       reconstruction re-derives num/den from the trace openings masked
//       by `full_geq(degree, ·)` and asserts equality with the GKR round
//       walk (verifier.rs:1861-1880).  This is the SUBSTRATE the
//       VK-identity change relies on once heights leave both the hash and
//       the transcript framing.  It always runs.
// ─────────────────────────────────────────────────────────────────────

// (A) TRANSCRIPT-COUPLED forgeries — reject via the transcript bind even
//     with the reconstruction gate OFF (the production default).

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_overclaim_fibonacci() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/overclaim",
        fibonacci_program(),
        ZKMStdin::new(),
        true, // MUST reject
        overclaim,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci OVER-claim (transcript+degree) => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_underclaim_fibonacci() {
    setup_logger();
    let tag =
        run_forgery("fibonacci/underclaim", fibonacci_program(), ZKMStdin::new(), true, underclaim);
    eprintln!("[STAGE0][VERDICT] fibonacci UNDER-claim (transcript+degree) => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_transcript_only_fibonacci() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/transcript-only",
        fibonacci_program(),
        ZKMStdin::new(),
        true,
        forge_transcript_only,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci TRANSCRIPT-only => {tag}");
}

// (B) DEGREE-ONLY forgeries WITH the reconstruction gate ON — the pure
//     substrate test (transcript LEFT HONEST): proves the degree-bit
//     `full_geq` reconstruction independently binds height.

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_degree_only_overclaim_fibonacci_recon_on() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/degree-only-overclaim",
        fibonacci_program(),
        ZKMStdin::new(),
        true, // MUST reject
        forge_degree_only_overclaim,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci DEGREE-only OVER-claim => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_degree_only_underclaim_fibonacci_recon_on() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/degree-only-underclaim",
        fibonacci_program(),
        ZKMStdin::new(),
        true,
        forge_degree_only_underclaim,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci DEGREE-only UNDER-claim => {tag}");
}

// (B') DEGREE-ONLY forgery with the reconstruction EXPLICITLY DISABLED
//      with the reconstruction disabled — DOCUMENTS that
//      a degree-only lie SURVIVES only when the degree anchor is turned OFF
//      on purpose.  The reconstruction is ON
//      by default, so this case requires the explicit `=0` override (which
//      `run_forgery(recon=false)` sets).  It pins the cause (the gate)
//      and proves the escape hatch still disables the check.

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_degree_only_overclaim_fibonacci_survives_when_recon_off() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/degree-only-overclaim-EXPLICITLY-OFF",
        fibonacci_program(),
        ZKMStdin::new(),
        false, // SURVIVES (degree anchor disabled on purpose) — documented
        forge_degree_only_overclaim,
    );
    eprintln!(
        "[STAGE0][CAVEAT] fibonacci DEGREE-only OVER-claim SURVIVES with \
         ZIREN_LOGUP_RECONSTRUCTION=0 (escape hatch disables the host degree \
         anchor) => {tag}"
    );
}

// (B'') the SAME degree-only forgery, but on
//       the TRUE PRODUCTION DEFAULT (env var UNSET) — the reconstruction
//       runs by default, so the lie MUST be REJECTED.  This is the headline
//       soundness assertion: with heights out of `vk.hash`,
//       the degree-masked reconstruction is the host check that
//       still binds height.  Does NOT use `run_forgery` (which forces the
//       var) — it removes the var to exercise the real default.
#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage1_degree_only_overclaim_fibonacci_rejected_on_default() {
    setup_logger();
    // TRUE production default: the var is UNSET.
    let (proof, machine, vk) = prove_fixoff(fibonacci_program(), ZKMStdin::new());
    let honest = verify(&machine, &vk, &proof);
    assert!(
        honest.is_ok(),
        "[stage1-default] honest FIX-off proof must verify on the un-gated \
         default (got {})",
        reject_tag(&honest)
    );
    let (si, ci, name, log_h) = pick_forge_target(&proof);
    eprintln!(
        "[STAGE1][FORGE][degree-only-DEFAULT] target shard={si} chip_idx={ci} \
         chip='{name}' log_height={log_h}"
    );
    let mut forged = proof.clone();
    forge_degree_only_overclaim(&mut forged.shard_proofs[si], ci, &name);
    let res = verify(&machine, &vk, &forged);
    let tag = reject_tag(&res);
    eprintln!("[STAGE1][FORGE][degree-only-DEFAULT] forged verify = {tag}");
    assert!(
        res.is_err(),
        "[stage1-default] DEGREE-only forgery SURVIVED on the un-gated \
         production default — the un-gate did NOT take effect (BLOCKER): {tag}"
    );
    eprintln!(
        "[STAGE1][VERDICT] fibonacci DEGREE-only OVER-claim REJECTED on the \
         un-gated production default => {tag}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// keccak / sha3 (precompile-uniform shards)
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "proves a real FIX-off keccak core proof (multi-second); run with --ignored"]
fn stage0_forge_overclaim_keccak() {
    setup_logger();
    let tag =
        run_forgery("keccak/overclaim", sha3_chain_program(), ZKMStdin::new(), true, overclaim);
    eprintln!("[STAGE0][VERDICT] keccak OVER-claim (transcript+degree) => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off keccak core proof (multi-second); run with --ignored"]
fn stage0_forge_underclaim_keccak() {
    setup_logger();
    let tag =
        run_forgery("keccak/underclaim", sha3_chain_program(), ZKMStdin::new(), true, underclaim);
    eprintln!("[STAGE0][VERDICT] keccak UNDER-claim (transcript+degree) => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off keccak core proof (multi-second); run with --ignored"]
fn stage0_forge_degree_only_overclaim_keccak_recon_on() {
    setup_logger();
    let tag = run_forgery(
        "keccak/degree-only-overclaim",
        sha3_chain_program(),
        ZKMStdin::new(),
        true,
        forge_degree_only_overclaim,
    );
    eprintln!("[STAGE0][VERDICT] keccak DEGREE-only OVER-claim => {tag}");
}

// ─────────────────────────────────────────────────────────────────────
// Jagged HASH-BIND forgeries.
//
// The hash-bind ties the per-chip (row_count, column_count) geometry to the
// FS-observed commitment via
//   main_commitment = compress([raw_root, hash(once(len) ++ rc ++ cc)]).
// The host shard verifier re-check (shard_level/verifier.rs, the jagged
// HASH-BIND re-check)
// recomputes this from the bundle's RAW root + packing and asserts it equals
// `main_commitment`.  Tampering with ANY count (row or column) in the bundle's
// packing — WITHOUT being able to forge the corresponding raw root — must make
// the recompute diverge and REJECT (`JaggedPcs(... IncorrectTableSizes)`).
// This is the count↔commitment tie that the legacy raw-root-only digest
// lacked entirely (a prover could witness any geometry).
// ─────────────────────────────────────────────────────────────────────

/// COLUMN-count tamper: bump a chip's `column_count` in the bundle's
/// packing (leaving the committed raw root untouched).  The host re-check
/// recomputes hash(counts) over the tampered count → != observed
/// main_commitment → reject.
fn forge_count_tamper_column(sp: &mut ShardProof<SC>, _ci: usize, _name: &str) {
    use zkm_pcs::shard_level::shard_proof::EvaluationProof;
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    match &mut bf.evaluation_proof {
        EvaluationProof::Bundle(bundle) => {
            // Find a chip with a nonzero column count and bump it by 1.
            let idx = bundle
                .packing
                .column_counts
                .iter()
                .position(|&c| c > 0)
                .expect("at least one chip with columns");
            let old = bundle.packing.column_counts[idx];
            bundle.packing.column_counts[idx] = old + 1;
            eprintln!(
                "[STAGE0][FORGE] COUNT-tamper column: packing.column_counts[{idx}] {old} -> {}",
                old + 1
            );
        }
        _other => panic!("expected EvaluationProof::Bundle for count-tamper"),
    }
}

/// ROW-count tamper: shift a packing OFFSET so one chip's derived
/// row_count (= offsets[i+1]-offsets[i]) changes.  Same effect: the hash over
/// the (now different) row_counts diverges from the observed main_commitment.
fn forge_count_tamper_row(sp: &mut ShardProof<SC>, _ci: usize, _name: &str) {
    use zkm_pcs::shard_level::shard_proof::EvaluationProof;
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    match &mut bf.evaluation_proof {
        EvaluationProof::Bundle(bundle) => {
            // Bump an interior offset (not the first, not the sentinel) so the
            // chip straddling it gets a different height.  Pick offset[1].
            let n = bundle.packing.offsets.len();
            assert!(n >= 3, "need >=3 offsets to tamper an interior boundary");
            let idx = 1.min(n - 2);
            let old = bundle.packing.offsets[idx];
            bundle.packing.offsets[idx] = old + 1;
            eprintln!(
                "[STAGE0][FORGE] COUNT-tamper row: packing.offsets[{idx}] {old} -> {} \
                 (changes a derived row_count)",
                old + 1
            );
        }
        _other => panic!("expected EvaluationProof::Bundle for row-count tamper"),
    }
}

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_count_tamper_column_fibonacci() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/count-tamper-column",
        fibonacci_program(),
        ZKMStdin::new(),
        true, // MUST reject (hash-bind)
        forge_count_tamper_column,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci COLUMN-count tamper (hash-bind) => {tag}");
}

#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage0_forge_count_tamper_row_fibonacci() {
    setup_logger();
    let tag = run_forgery(
        "fibonacci/count-tamper-row",
        fibonacci_program(),
        ZKMStdin::new(),
        true,
        forge_count_tamper_row,
    );
    eprintln!("[STAGE0][VERDICT] fibonacci ROW-count tamper (hash-bind) => {tag}");
}

// ═════════════════════════════════════════════════════════════════════
// STAGE 1 — ZERO-DEGREE MODEL SOUNDNESS.
//
// The missing-chip trace is a GENUINE HEIGHT-0 (0-row, full-width, zero)
// commit: a canonical-cluster chip a raw FIX-off shard lacks is committed
// with `row_count = 0`, so its `degree` bits (`quotient[0]`) are ALL ZERO
// (=> `full_geq == 1` => identity fraction (0,1) => excluded from the
// LogUp-GKR sum).  These two forgeries PROVE that model is sound — that the
// live degree-mask substrate cannot be fooled into (a) treating a genuinely
// missing (height-0) chip as active, nor (b) treating a real present active
// chip as missing.  Both forge ONLY the `degree` bits (`quotient[0]`) and
// LEAVE the transcript (`log_degree`, `chip_heights`) HONEST, so the
// rejection can only come from the degree-masked substrate (the
// `LogupGkr`/`Zerocheck` reconstruction), NOT the Fiat-Shamir transcript
// bind.  Run under the un-gated production default (reconstruction ON).
// ═════════════════════════════════════════════════════════════════════

/// Find the first shard+chip committed at genuine HEIGHT 0 — a missing
/// canonical-cluster chip: its `degree` bits (`quotient[0]`) are a NON-EMPTY
/// vector of ALL ZEROS (height 0 => every big-endian bit clear; a present
/// height-1 chip instead has the LSB set, and taller chips have a higher bit
/// set).  Returns `(shard_idx, chip_idx, chip_name)`.  Reports every height-0
/// chip it sees (the direct evidence that the band-cap injects missing chips
/// at h=0/degree=0).  Panics if NONE exist (which would itself mean the
/// injection is not firing — a validation failure).
fn pick_height0_missing_target(proof: &MachineProof<SC>) -> (usize, usize, String) {
    let mut found: Vec<(usize, String)> = Vec::new();
    let mut first: Option<(usize, usize, String)> = None;
    for (si, sp) in proof.shard_proofs.iter().enumerate() {
        let Some(bf) = sp.basefold_shard_proof.as_ref() else { continue };
        // Name-sorted chip names (the order `opened_values.chips` uses) —
        // `chip_heights` is a name-sorted BTreeMap over the same set.
        let sorted_names: Vec<String> = bf.chip_heights.keys().cloned().collect();
        for (ci, opening) in bf.opened_values.chips.iter().enumerate() {
            let Some(degree) = opening.quotient.first() else { continue };
            if degree.is_empty() {
                continue;
            }
            // ALL-ZERO degree bits <=> real height 0 (a missing chip).
            if degree.iter().all(|b| *b == Challenge::ZERO) {
                let name = sorted_names.get(ci).cloned().unwrap_or_else(|| format!("chip{ci}"));
                found.push((si, name.clone()));
                first.get_or_insert((si, ci, name));
            }
        }
    }
    eprintln!(
        "[STAGE1] HEIGHT-0/degree-0 missing chips in FIX-off proof: {} -> {:?}",
        found.len(),
        found,
    );
    first.expect(
        "no HEIGHT-0 missing chip in FIX-off proof — band-cap injection is NOT firing \
         (the whole zero-degree model is untested)",
    )
}

/// (a) A genuinely-MISSING chip committed at HEIGHT 0 (degree bits all zero)
/// forged to CLAIM ACTIVITY: set a single degree bit non-zero (claim height
/// 2^1).  Transcript LEFT HONEST.
fn forge_height0_claim_active(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let degree = &mut bf.opened_values.chips[ci].quotient[0];
    let bit_len = degree.len();
    // Ensure a clean all-zero start (it already is — height 0), then set the
    // bit for 2^1: big-endian index `bit_len - 1 - 1`.
    for b in degree.iter_mut() {
        *b = Challenge::ZERO;
    }
    let new_shift = 1usize.min(bit_len.saturating_sub(1));
    let new_idx = bit_len - 1 - new_shift;
    degree[new_idx] = Challenge::ONE;
    eprintln!(
        "[STAGE1][FORGE] HEIGHT-0 chip='{name}': degree 0 (missing) -> claims 2^{new_shift} \
         ACTIVE (transcript log_height LEFT HONEST)"
    );
}

/// (b) A real PRESENT active chip forged to CLAIM IT IS MISSING: zero ALL its
/// degree bits (claim height 0).  Transcript LEFT HONEST.
fn forge_present_claim_missing(sp: &mut ShardProof<SC>, ci: usize, name: &str) {
    let bf = sp.basefold_shard_proof.as_mut().expect("basefold proof");
    let degree = &mut bf.opened_values.chips[ci].quotient[0];
    let real_h = degree_bits_to_height(degree);
    for b in degree.iter_mut() {
        *b = Challenge::ZERO;
    }
    eprintln!(
        "[STAGE1][FORGE] PRESENT chip='{name}': real_height={real_h:?} -> degree ALL-ZERO \
         (claims missing/height-0, transcript log_height LEFT HONEST)"
    );
}

/// ★ FORGERY GATE (a): a genuinely-missing HEIGHT-0 chip forged to claim
/// activity MUST be rejected by the degree-masked substrate.  Uses the
/// un-gated production default (reconstruction ON).
#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage1_forge_height0_missing_claims_active_rejected() {
    setup_logger();
    // TRUE production default: the reconstruction runs unless explicitly "0".
    let (proof, machine, vk) = prove_fixoff(fibonacci_program(), ZKMStdin::new());
    let honest = verify(&machine, &vk, &proof);
    assert!(
        honest.is_ok(),
        "[stage1-a] honest FIX-off proof must verify before forging (got {})",
        reject_tag(&honest)
    );
    let (si, ci, name) = pick_height0_missing_target(&proof);
    eprintln!(
        "[STAGE1][FORGE][height0-claims-active] target shard={si} chip_idx={ci} chip='{name}'"
    );
    let mut forged = proof.clone();
    forge_height0_claim_active(&mut forged.shard_proofs[si], ci, &name);
    let res = verify(&machine, &vk, &forged);
    let tag = reject_tag(&res);
    eprintln!("[STAGE1][FORGE][height0-claims-active] forged verify = {tag}");
    assert!(
        res.is_err(),
        "[stage1-a] HEIGHT-0 missing chip forged to ACTIVE SURVIVED — the zero-degree \
         model is UNSOUND (SOUNDNESS HOLE / BLOCKER): {tag}"
    );
    eprintln!("[STAGE1][VERDICT] (a) HEIGHT-0 missing chip claiming activity REJECTED => {tag}");
}

/// ★ FORGERY GATE (b): a real present active chip forged to claim it is
/// missing (degree=0) MUST be rejected.  Uses `run_forgery` (which picks a
/// present, height-varied chip) with the reconstruction ON.
#[test]
#[ignore = "proves a real FIX-off core proof (multi-second); run with --ignored"]
fn stage1_forge_present_active_claims_missing_rejected() {
    setup_logger();
    let tag = run_forgery(
        "present-claims-missing",
        fibonacci_program(),
        ZKMStdin::new(),
        true, // MUST reject
        forge_present_claim_missing,
    );
    eprintln!(
        "[STAGE1][VERDICT] (b) PRESENT active chip claiming missing (degree=0) REJECTED => {tag}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// PREPROCESSED BINDING PROBE.
//
// Ziren commits preprocessed traces INTO EACH SHARD alongside the main
// traces, and the setup-time `vk.commit` is only OBSERVED into the
// Fiat-Shamir transcript (`machine.rs`'s `vk.observe_into`) -- it is never
// compared, and `preprocessed_commit()` has no callers.  The alternative
// design commits preprocessed ONCE at setup, keeps `preprocessed_data` in
// the proving key, and has the shard verifier check openings against
// `vec![vk.preprocessed_commit, *main_commitment]` -- two rounds, so the
// preprocessed columns are bound to the VK by the opening argument.
//
// Question: does Ziren's transcript observation alone bind them?  Prove
// one program and verify the proof against a DIFFERENT program's vk.  If
// that is accepted, a prover can prove anything and claim it is some other
// program.  If it is rejected, something already binds -- and the tag says
// which check caught it.
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "proves two real FIX-off core proofs (multi-second); run with --ignored"]
fn preprocessed_binding_cross_vk_probe() {
    use crate::programs::tests::{fibonacci_program, simple_program};
    setup_logger();

    // Honest proof of program B (`simple`).
    let (proof_b, machine, vk_b) = prove_fixoff(simple_program(), ZKMStdin::new());
    let honest = verify(&machine, &vk_b, &proof_b);
    assert!(honest.is_ok(), "anti-confound: program B must verify under its OWN vk");

    // Program A's vk (`fibonacci`) -- a different program, different
    // preprocessed traces, therefore a different `vk.commit`.
    let config = KoalaBearPoseidon2::new();
    let machine_a = MipsAir::machine(config);
    let (_pk_a, vk_a) = machine_a.setup(&fibonacci_program());
    assert_ne!(
        format!("{:?}", vk_a.commit),
        format!("{:?}", vk_b.commit),
        "the two programs must have distinct preprocessed commitments for this probe to mean anything"
    );

    let cross = verify(&machine, &vk_a, &proof_b);
    let tag = reject_tag(&cross);
    eprintln!("[PREP-BIND] program B's proof verified against program A's vk => {tag}");
    assert!(
        cross.is_err(),
        "SOUNDNESS: a proof of one program MUST NOT verify against another \
         program's verifying key; if it does, the preprocessed traces are \
         unbound and `vk.commit` is decorative"
    );
}

/// GATE (memory chips): honest FIX-off proofs of memory-heavy programs verify.
#[test]
#[ignore = "proves three real FIX-off core proofs; run with --ignored"]
fn stage0_control_fixoff_memory_programs_verify() {
    setup_logger();
    for (name, program) in [
        ("simple_memory", simple_memory_program()),
        ("ssz_withdrawals", ssz_withdrawals_program()),
        ("sha3_chain", sha3_chain_program()),
    ] {
        let (proof, machine, vk) = prove_fixoff(program, ZKMStdin::new());
        let res = verify(&machine, &vk, &proof);
        eprintln!(
            "[STAGE0][MEMGATE][{name} FIX-off] shards={} honest verify = {}",
            proof.shard_proofs.len(),
            reject_tag(&res)
        );
        assert!(res.is_ok(), "honest FIX-off proof of {name} must verify");
    }
}
