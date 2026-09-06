//! Top-level row-reduction shard LogUp-GKR prover.
//!
//! Pipeline: sample challenges → build GKR circuit → evaluate
//! unified output at the first eval point → walk layers bottom-up
//! (per-round sumcheck, observe openings, extend eval_point, update
//! numerator/denominator via the line formula) → compute per-chip
//! trace MLE evaluations at the terminal point → assemble proof.

use alloc::vec::Vec;
use std::collections::BTreeMap;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField};

use super::build::build_gkr_circuit;
use super::round::prove_gkr_round;
use crate::air::MachineAir;
use crate::logup_gkr::{GkrGrind, GKR_GRINDING_BITS};
use crate::multilinear::PaddedMle;
use crate::shard_level::logup_gkr_prover::evaluate_trace_columns_at_point;
use crate::shard_level::types::{ChipEvaluation, LogUpEvaluations, LogUpGkrOutput, LogupGkrProof};
use crate::zerocheck_prover::eq_mle_table;
use crate::Chip;

/// `preprocessed_traces[i]` may have width 0.
#[allow(clippy::too_many_arguments)]
pub fn prove_shard_logup_gkr_rows<F, EF, A, Challenger>(
    chips: &[&Chip<F, A>],
    preprocessed_traces: &[crate::multilinear::PaddedMle<F>],
    max_log_row_count: usize,
    challenger: &mut Challenger,
    // The shared per-chip analytic main-trace MLE (chip-index order),
    // built once in the shard dispatch over the `max_log_row_count` cube and
    // threaded read-only — the SOLE host main-trace source for this stage.
    // A host chip's `PaddedMle` carries a real inner (`guts == the raw
    // trace`, byte-for-byte); a device-resident / unexercised chip is a
    // `dummy` (inner `None`, width 0) whose real cells come from the
    // per-shard device provider.  The FULL-POINT opening
    // consumes the inner via `PaddedMle::eval_at` (== the on-the-fly
    // `evaluate_trace_columns_at_point`, unit-tested).
    shared_trace_mles: &[PaddedMle<F>],
) -> LogupGkrProof<F, EF>
where
    F: PrimeField + 'static,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    A: MachineAir<F>,
    Challenger: FieldChallenger<F> + 'static,
{
    // Proof-of-work grinding. MUST run BEFORE sampling alpha/beta to
    // match the in-circuit verifier's `check_witness`, which is the FIRST
    // challenger op in `verify_logup_gkr` (recursion logup_gkr.rs:347). p3's
    // `grind` finds the witness AND observes it into the challenger, so the
    // post-grind state — alpha/beta and the whole GKR transcript — is
    // identical between prover and verifier. Config-aware (see `GkrGrind`):
    // real grind for the Inner challenger, `F::ZERO` no-op for the
    // Outer/wrap challenger (never recursion-verified).
    let witness: F = challenger.gkr_grind(GKR_GRINDING_BITS);

    // Sample the LogUp challenges [alpha, beta].  `beta_seed_dim` = log2(max_arity
    // rounded up).  `betas.len()` = 1 + max_arity (slot 0 is for
    // argument_index, slots 1..=arity for per-column values).
    let alpha: EF = challenger.sample_algebra_element::<EF>();
    let max_arity = chips
        .iter()
        .flat_map(|chip| chip.sends().iter().chain(chip.receives().iter()))
        .map(|interaction| interaction.values.len() + 1)
        .max()
        .unwrap_or(1);
    let beta_seed_dim = max_arity.next_power_of_two().trailing_zeros() as usize;
    let beta_seed: Vec<EF> =
        (0..beta_seed_dim).map(|_| challenger.sample_algebra_element::<EF>()).collect();
    // Expand beta_seed to the partial-lagrange table over {0,1}^beta_seed_dim.
    let betas = if beta_seed.is_empty() { vec![EF::ONE] } else { eq_mle_table::<EF>(&beta_seed) };

    // GKR padding (VERIFY_VK enumerability): the GKR
    // round count is FIXED to `max_log_row_count - 1` regardless of
    // the actual (heterogeneous) chip heights — `build_gkr_circuit`
    // emits `num_row_variables - 1` round proofs (see `build.rs:94-117`
    // layer ladder), so `num_row_variables = max_log_row_count` yields
    // exactly `max_log_row_count - 1` rounds — the count the verifier
    // hard-asserts (`round_proofs.len() + 1 == max_log_row_count`); the
    // first-layer MLEs are lazily zero-padded to `max_log_row_count`.
    //
    // Device residency: chips whose host trace was emptied (device
    // resident) resolve their REAL height from the per-shard provider
    // inside the ceiling check below, so a device-resident tall chip
    // (e.g. np>0 Program at 2^19) is still bounds-checked.  Note the
    // FIXED `num_row_variables` already subsumes the original concern
    // (a data-dependent count shrinking below device-trace heights) —
    // the count can no longer shrink at all.
    debug_assert!(
        {
            let max_height = chips
                .iter()
                .zip(shared_trace_mles.iter())
                .map(|(_chip, pm)| pm.metadata_height().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let actual_log_height =
                max_height.max(1).next_power_of_two().trailing_zeros().max(2) as usize;
            actual_log_height <= max_log_row_count
        },
        "max trace log height (provider-resolved) exceeds the shard ceiling \
         max_log_row_count {max_log_row_count} — GKR padding would truncate"
    );
    let num_row_variables = max_log_row_count;

    let n_chips = chips.len();

    // Build GKR circuit + extract output MLEs.
    let _t_first = std::time::Instant::now();
    let _first_span = tracing::info_span!("logup_gkr_first_layer").entered();
    let (output, mut circuit) = build_gkr_circuit::<F, EF, A>(
        chips,
        preprocessed_traces,
        shared_trace_mles,
        alpha,
        &betas,
        num_row_variables,
    );
    let num_interaction_variables =
        output.numerator.len().trailing_zeros().saturating_sub(1) as usize;
    drop(_first_span);
    let _dt_first_us = _t_first.elapsed().as_micros() as u64;
    tracing::info!(
        elapsed_ms = _dt_first_us / 1000,
        chips = n_chips,
        sub_phase = "first_layer",
        "logup_gkr sub-phase done"
    );

    // Phase-4: the former scope-populate block (which drained device-resident
    // layer payloads into the task scope via the `logup_scope_populate` bundle
    // fn and stashed `generate_first_layer` onto the circuit's
    // `DeviceInputData`) was removed.  Both fns were always `None` from every
    // producer (the V3 layer cache was excised), so the block never ran; the
    // scope's `circuit` stays `None` exactly as before → byte-identical.

    // Observe circuit_output before sampling eval_point — without
    // this the prover's transcript skips an observation step the
    // verifier performs and round 0's claimed_sum check fails.
    for &n in output.numerator.iter() {
        for basis in n.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }
    for &d in output.denominator.iter() {
        for basis in d.as_basis_coefficients_slice() {
            challenger.observe(*basis);
        }
    }

    // Sample the first eval_point (dim = num_interaction_variables + 1).
    let mut eval_point: Vec<EF> = (0..(num_interaction_variables + 1))
        .map(|_| challenger.sample_algebra_element::<EF>())
        .collect();

    // LSB-first MLE evaluation to match the verifier
    // (`evaluate_mle_host`); `eq_mle_table` is MSB-first and would
    // diverge.
    fn evaluate_mle<EF: Field + Copy>(mle_evals: &[EF], point: &[EF]) -> EF {
        let mut weights: Vec<EF> = vec![EF::ONE];
        for &r in point {
            let old_len = weights.len();
            let mut next = vec![EF::ZERO; old_len * 2];
            for j in 0..old_len {
                let prod = weights[j] * r;
                next[j] = weights[j] - prod;
                next[j + old_len] = prod;
            }
            weights = next;
        }
        mle_evals.iter().zip(weights.iter()).fold(EF::ZERO, |acc, (v, w)| acc + *v * *w)
    }
    let mut numerator_eval: EF = evaluate_mle::<EF>(&output.numerator, &eval_point);
    let mut denominator_eval: EF = evaluate_mle::<EF>(&output.denominator, &eval_point);

    // Walk layers bottom-up.  `circuit.layers` is stored
    // top-down (first = largest num_row_vars); `pop_bottom` pops the
    // smallest first, which is the extraction source — skip it and
    // start from the next one up (num_row_variables == 1 terminal).
    //
    // Invariant check: after extract_outputs consumed layers[N-2] (the
    // terminal), the remaining layers we want to prove against are
    // layers[0..N-2] in bottom-up order.  Reverse the stack, skip the
    // layers[N-1] entry (which has num_row_variables == 0 and was
    // never extracted from), and iterate.
    let mut round_proofs = Vec::with_capacity(circuit.layers.len());
    circuit.layers.reverse();

    let _t_layers = std::time::Instant::now();
    let _layers_span = tracing::info_span!("logup_gkr_layer_transitions").entered();
    // `circuit.layers` is `Vec<LayerState>`. Skip the
    // num_row_variables == 0 terminal (only there to enable clean
    // termination of the build loop), then dispatch on the variant.
    //
    // D3c: this shared HOST driver is CpuProver-only — `build_gkr_circuit`
    // never constructs `LayerState::Device`, so the per-circuit device-drain
    // (formerly `gkr_device_hooks.layer_drain`) is not needed here.  The GPU
    // prover's device-native driver owns the device drain (`gpu_layer_drain_circuit_hook`).
    for state in circuit.layers.iter().filter(|l| l.num_row_variables() >= 1) {
        let lambda: EF = challenger.sample_algebra_element::<EF>();

        let round_proof = prove_gkr_round::<F, EF, _>(
            state,
            &eval_point,
            numerator_eval,
            denominator_eval,
            lambda,
            challenger,
        );

        // Observe order MUST match verifier: n0, n1, d0, d1.
        // Mismatched order desyncs the transcript at line_challenge.
        observe_ext::<F, EF, _>(challenger, round_proof.numerator_0);
        observe_ext::<F, EF, _>(challenger, round_proof.numerator_1);
        observe_ext::<F, EF, _>(challenger, round_proof.denominator_0);
        observe_ext::<F, EF, _>(challenger, round_proof.denominator_1);

        // Take the reduced point from the sumcheck as the base for the
        // next layer's eval_point; extend by the line challenge.  The layer
        // transition pairs ADJACENT rows, i.e. peels the row LSB (variable
        // `num_interaction_variables` of the LSB-first flat index), so the
        // line challenge is INSERTED there and the reduced point's row
        // coordinates shift up by one.  Must match `shard_level/verifier.rs`
        // and the recursion circuit (`logup_gkr.rs`).
        let mut next_eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();
        let line_challenge: EF = challenger.sample_algebra_element::<EF>();
        next_eval_point.insert(num_interaction_variables, line_challenge);

        // Line-formula: at the sumcheck's reduced point + line_challenge,
        //   n_eval = n_0 + line · (n_1 - n_0) = (1 - line) · n_0 + line · n_1
        //   d_eval = d_0 + line · (d_1 - d_0) = (1 - line) · d_0 + line · d_1
        numerator_eval = round_proof.numerator_0
            + (round_proof.numerator_1 - round_proof.numerator_0) * line_challenge;
        denominator_eval = round_proof.denominator_0
            + (round_proof.denominator_1 - round_proof.denominator_0) * line_challenge;

        eval_point = next_eval_point;
        round_proofs.push(round_proof);
    }
    let n_layers = round_proofs.len();

    drop(_layers_span);
    let _dt_layers_us = _t_layers.elapsed().as_micros() as u64;
    tracing::info!(
        elapsed_ms = _dt_layers_us / 1000,
        chips = n_chips,
        layers = n_layers,
        sub_phase = "layer_transitions",
        "logup_gkr sub-phase done"
    );

    // Per-chip trace evaluations. The eval_point has dim
    // `num_row_variables + num_interaction_variables + 1`; each
    // chip's evaluation point is the trailing `log(chip_height)`
    // coords.
    let _t_extract = std::time::Instant::now();
    let _extract_span = tracing::info_span!("logup_gkr_output_extract").entered();
    use p3_maybe_rayon::prelude::*;

    let chip_openings: BTreeMap<String, ChipEvaluation<EF>> = chips
        .par_iter()
        .zip(shared_trace_mles.par_iter())
        .zip(preprocessed_traces.par_iter())
        .map(|((chip, pm), prep_trace)| {
            // Device-only chip — its real height is baked into the dummy MLE
            // and read via `metadata_height()`; a host chip reads the
            // shared MLE's real row count.  Falls back to 1 (legacy
            // unexercised-chip) when absent.
            let main_height = pm.metadata_height().unwrap_or(1);
            let log_main_height = main_height.max(1).next_power_of_two().trailing_zeros() as usize;
            // The SHARED opening point: the trailing `max_log_row_count`
            // coords (= the full trace_point), LSB-first.  Every chip opens
            // here — the per-chip trailing-`log_h` point is gone with the
            // legacy claim that needed it.
            let full_eval_point: &[EF] = if eval_point.len() >= max_log_row_count {
                &eval_point[eval_point.len() - max_log_row_count..]
            } else {
                &eval_point[..]
            };
            // Verifier hard-checks `opening.main.local.len() ==
            // chip.width()`, so an unexercised chip must still emit a
            // zero vector of its declared width.
            let chip_main_width = <_ as p3_air::BaseAir<F>>::width(&chip.air);
            let prep_ref = prep_trace.real_trace_ref();

            // FULL-POINT openings at `full_eval_point` (the full
            // trace_point), for the LogUp last-layer reconstruction.  The GKR
            // leaf is LSB-first natural-row, so the full-point opening of the
            // zero-padded trace = Σ_{row<height} eq(row, trace_point)·trace[row]
            // (rows ≥ height implicitly zero) — exactly what the reconstruction
            // needs.  Host path: `evaluate_trace_columns_at_point` over the full
            // coords.
            // ALWAYS emit the full-point opening for every chip so
            // the shard-uniform rev(zeta) convention is unconditional on core.
            // Host chips consume the shared analytic trace-MLE (`PaddedMle::
            // eval_at`, transcript-neutral, reproduces
            // `evaluate_trace_columns_at_point` bit-for-bit).  On the host CPU
            // path device-only
            // chips serve a zero vector of declared width (unexercised/height-0);
            // width-0 chips emit an empty opening.
            let main_evals_full: Option<Vec<EF>> = if pm.inner().is_some() {
                Some(pm.eval_at::<EF>(full_eval_point))
            } else if chip_main_width > 0 {
                Some(vec![EF::ZERO; chip_main_width])
            } else {
                Some(Vec::new())
            };
            let prep_evals_full: Option<Vec<EF>> = if let Some(pt) = prep_ref {
                Some(evaluate_trace_columns_at_point::<F, EF>(pt.values, pt.width, full_eval_point))
            } else {
                None
            };

            (
                chip.name().to_string(),
                ChipEvaluation {
                    log_degree: u8::try_from(log_main_height).unwrap_or(0),
                    main_trace_evaluations_full: main_evals_full,
                    preprocessed_trace_evaluations_full: prep_evals_full,
                },
            )
        })
        .collect();
    drop(_extract_span);
    let _dt_extract_us = _t_extract.elapsed().as_micros() as u64;
    tracing::info!(
        elapsed_ms = _dt_extract_us / 1000,
        chips = n_chips,
        sub_phase = "output_extract",
        "logup_gkr sub-phase done"
    );

    // Verifier invariant `zerocheck_point.dim == gkr_point.dim ==
    // pcs_max_log_row_count`. Left-pad with ZERO when this shard is
    // shorter — padding binds the LSB row variables (never above
    // chip heights), trailing coords drive chip trace MLE evals.
    let mut trace_dim_point = if eval_point.len() >= num_row_variables {
        eval_point[eval_point.len() - num_row_variables..].to_vec()
    } else {
        eval_point.clone()
    };
    while trace_dim_point.len() < max_log_row_count {
        trace_dim_point.insert(0, EF::ZERO);
    }

    let proof = LogupGkrProof {
        circuit_output: LogUpGkrOutput {
            numerator: output.numerator,
            denominator: output.denominator,
        },
        round_proofs,
        logup_evaluations: LogUpEvaluations { point: trace_dim_point, chip_openings },
        witness,
    };

    // Observe slot 1 — the GKR trace openings (trace@ζ), observed HERE,
    // inside the GKR phase, before the shard driver samples the zerocheck's
    // α / γ / λ.  Keeping it inside this function (rather than in the shard
    // driver) makes it structurally impossible for a driver to sample a
    // zerocheck challenge against an unbound opening vector.  See
    // `shard_level::prover::observe_logup_gkr_openings`.
    crate::shard_level::prover::observe_logup_gkr_openings::<F, EF, Challenger>(
        challenger,
        chips.len(),
        &proof.logup_evaluations,
    );

    proof
}

#[inline]
fn observe_ext<F, EF, Challenger>(challenger: &mut Challenger, v: EF)
where
    F: Field,
    EF: BasedVectorSpace<F>,
    Challenger: CanObserve<F>,
{
    for c in v.as_basis_coefficients_slice() {
        challenger.observe(*c);
    }
}
