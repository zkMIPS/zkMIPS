//! First-layer generator for the row-only GKR backend.
//!
//! Per chip: walk interactions to build `(numerator, denominator)`
//! per row, pack into a row-major `(height × num_interactions)`
//! table, pad rows to the shared `num_row_variables` (zero-fill for
//! numerator, one-fill for denominator to preserve the
//! sum-of-fractions identity), then split by row PARITY (quadrant 0 =
//! even rows, quadrant 1 = odd rows) — the row LSB is the bit each layer
//! transition peels off.
//!
//! Row-major flat storage: `cells[row * num_interactions + col]`.
//! Viewed as an MLE, the row LSB is variable `num_interaction_variables`;
//! quadrant `b` holds the rows whose LSB is `b`, i.e. `row = 2k + b`.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;

use super::layer::{LogUpGkrCpuLayer, RowMajorTable};
use crate::air::MachineAir;
use crate::lookup::Lookup;
use crate::multilinear::PaddedMle;
use crate::Chip;

/// `denominator = α + Σ β_k · v_k` with `v_0 = argument_index` and
/// `v_k = lookup.values[k-1].apply(prep_row, main_row)`. Numerator
/// is the signed multiplicity (`+mult` send / `-mult` receive).
pub fn generate_interaction_vals<F: Field, EF: ExtensionField<F>>(
    interaction: &Lookup<F>,
    preprocessed_row: &[F],
    main_row: &[F],
    is_send: bool,
    alpha: EF,
    betas: &[EF],
) -> (F, EF) {
    let mut denominator = alpha;
    let mut betas_iter = betas.iter();

    let beta_0 = *betas_iter.next().expect("at least one beta required (argument_index slot)");
    denominator += beta_0 * EF::from_usize(interaction.argument_index());

    for (column, beta) in interaction.values.iter().zip(&mut betas_iter) {
        let v: F = column.apply::<F, F>(preprocessed_row, main_row);
        denominator += *beta * v;
    }

    let mut mult: F = interaction.multiplicity.apply::<F, F>(preprocessed_row, main_row);
    if !is_send {
        mult = -mult;
    }

    (mult, denominator)
}

/// Build a chip's per-row interaction tables.
///
/// Returns `(numer, denom)` row-major matrices of shape
/// `height × num_interactions`.  `height` is derived from
/// `main_values.len() / main_width` (the chip's main-trace rows sourced
/// from the shared trace-MLE inner).  When `preprocessed_trace` is
/// `None`, the per-row preprocessed slice is treated as empty.
pub fn build_chip_interaction_tables<
    F: PrimeField + Send + Sync,
    EF: ExtensionField<F> + Send + Sync,
>(
    interactions: &[(&Lookup<F>, bool)],
    main_values: &[F],
    main_width: usize,
    preprocessed_trace: Option<crate::basefold::TraceRef<'_, F>>,
    alpha: EF,
    betas: &[EF],
) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>) {
    let height = if main_width == 0 { 0 } else { main_values.len() / main_width };
    let num_interactions = interactions.len();

    // FLAKE FIX: see round.rs::flatten_layer note about KoalaBear
    // serde rejecting out-of-range u32s leaked from set_len uninit.
    let total = height * num_interactions;
    let mut numer_evals: Vec<F> = vec![F::ZERO; total];
    let mut denom_evals: Vec<EF> = vec![EF::ZERO; total];

    // Performance optimization: parallelize per-row interaction
    // computation (`par_chunks_exact_mut(num_interactions)`).
    // For chips with hundreds of thousands of rows (e.g. Program
    // at 524K), per-row parallelism is the right granularity — chip-level
    // alone leaves a single core doing the work for the largest chip.
    if height > 0 && num_interactions > 0 {
        use p3_maybe_rayon::prelude::*;
        let main_w = main_width;
        let prep_w = preprocessed_trace.map(|pt| pt.width).unwrap_or(0);
        let prep_values: Option<&[F]> = preprocessed_trace.map(|pt| pt.values);
        numer_evals
            .par_chunks_exact_mut(num_interactions)
            .zip(denom_evals.par_chunks_exact_mut(num_interactions))
            .enumerate()
            .for_each(|(row_idx, (numer_row, denom_row))| {
                let main_row = &main_values[row_idx * main_w..(row_idx + 1) * main_w];
                let prep_row: &[F] = match prep_values {
                    Some(pv) if prep_w > 0 => &pv[row_idx * prep_w..(row_idx + 1) * prep_w],
                    _ => &[],
                };
                for (col_idx, (interaction, is_send)) in interactions.iter().enumerate() {
                    let (numer, denom) = generate_interaction_vals::<F, EF>(
                        interaction,
                        prep_row,
                        main_row,
                        *is_send,
                        alpha,
                        betas,
                    );
                    numer_row[col_idx] = numer;
                    denom_row[col_idx] = denom;
                }
            });
    }

    (
        RowMajorMatrix::new(numer_evals, num_interactions),
        RowMajorMatrix::new(denom_evals, num_interactions),
    )
}

/// PaddedMle-aware PARITY split.  Split a row-major `(real_rows ×
/// num_cols)` buffer into its even rows (quadrant 0) and odd rows
/// (quadrant 1), returning the **real-only** prefix of each:
///   * quadrant 0 = rows `0, 2, 4, …` → `ceil(real_rows / 2)` rows,
///   * quadrant 1 = rows `1, 3, 5, …` → `floor(real_rows / 2)` rows.
///
/// The layer transition pairs ADJACENT rows `(2k, 2k+1)` (the row LSB is the
/// bit each GKR layer peels off, and the verifier binds the line challenge
/// there), so every chip halves at every layer regardless of how short it is
/// relative to the logical layer height.  The previous MSB split paired rows
/// `(k, k + 2^(R-1))`, which left every chip shorter than half the layer as a
/// pass-through copy in each layer above its own height.
///
/// Virtual rows beyond the real prefix are NOT materialized; consumers
/// (`ChipLayerState`) resolve them via the per-quadrant pad constant.
fn split_real_parity<F: Clone>(
    values: &[F],
    num_cols: usize,
    real_rows: usize,
) -> (Vec<F>, Vec<F>) {
    if num_cols == 0 || real_rows == 0 {
        return (Vec::new(), Vec::new());
    }
    debug_assert!(real_rows * num_cols <= values.len());
    let even_rows = real_rows.div_ceil(2);
    let odd_rows = real_rows / 2;
    let mut even: Vec<F> = Vec::with_capacity(even_rows * num_cols);
    let mut odd: Vec<F> = Vec::with_capacity(odd_rows * num_cols);
    for (r, row) in values.chunks_exact(num_cols).take(real_rows).enumerate() {
        if r & 1 == 0 {
            even.extend_from_slice(row);
        } else {
            odd.extend_from_slice(row);
        }
    }
    (even, odd)
}

/// Generate the GKR circuit's first layer from raw chip data.
///
/// Port of
/// `LogupGkrCpuTraceGenerator::generate_first_layer`.
///
/// Inputs:
/// - `chips`: per-chip (sends + receives) lookup specs (in BTreeSet
///   iteration order on the host side).
/// - `preprocessed_traces`: per-chip raw preprocessed traces.
///   `preprocessed_traces[i]` may be empty (`width == 0`).
/// - `shared_trace_mles`: the shared per-chip analytic main-trace MLE
///   (chip-index order).  A host chip's `PaddedMle` carries a real inner
///   (`guts == the raw trace`); a device-resident / unexercised chip is a
///   `dummy` (inner `None`, width 0) whose real cells come from the
///   per-shard device provider via the GPU hook.
/// - `alpha`, `betas`: post-commit challenges.  `betas` length must be
///   `1 + max_interaction_arity` (slot 0 is for `argument_index`,
///   slots 1..=arity are for the per-column values).
/// - `num_row_variables`: `log₂` of the per-shard padded row count
///   (max chip height, rounded up).  Must satisfy `>= 1`.
///
/// Output: a [`LogUpGkrCpuLayer<F, EF>`] with one
/// `(numerator_0, numerator_1, denominator_0, denominator_1)` table
/// per chip, each of shape `2^(num_row_variables - 1) × num_interactions`.
/// `num_row_variables` on the layer is set to `original - 1`
/// (the row MSB has been fixed).  `num_interaction_variables` is
/// `log₂` of the sum of the per-chip raw interaction counts, rounded up
/// to a power of two.
// D3c (Option-C divergence): the device interaction-eval seam (the `dev`
// `ShardDeviceOps` param + the `device_traces` provider it consumed) was
// REMOVED from this HOST generator — it is now the CpuProver-only host build.
// The GPU prover routes through its own device-native `generate_first_layer_native`
// copy (in `zkm-gpu-basefold`) whose interaction-eval arm calls the ziren-gpu
// kernel `prove_shard_interaction_eval_gpu` DIRECTLY (what
// `CudaShardDeviceOps::interaction_eval` forwarded to VERBATIM).
pub fn generate_first_layer<F, EF, A>(
    chips: &[&Chip<F, A>],
    preprocessed_traces: &[crate::multilinear::PaddedMle<F>],
    shared_trace_mles: &[PaddedMle<F>],
    alpha: EF,
    betas: &[EF],
    num_row_variables: usize,
) -> LogUpGkrCpuLayer<F, EF>
where
    F: PrimeField,
    EF: ExtensionField<F>,
    A: MachineAir<F>,
{
    assert!(num_row_variables >= 1, "num_row_variables must be >= 1");
    assert_eq!(chips.len(), shared_trace_mles.len(), "chip count vs main trace count");
    assert_eq!(chips.len(), preprocessed_traces.len(), "chip count vs preprocessed trace count");

    let mut numerator_0: Vec<RowMajorTable<F>> = Vec::with_capacity(chips.len());
    let mut denominator_0: Vec<RowMajorTable<EF>> = Vec::with_capacity(chips.len());
    let mut numerator_1: Vec<RowMajorTable<F>> = Vec::with_capacity(chips.len());
    let mut denominator_1: Vec<RowMajorTable<EF>> = Vec::with_capacity(chips.len());
    // The global interaction axis must be wide enough to hold every chip's
    // block, and the blocks are laid out RAW-CONTIGUOUSLY: `flatten_layer`,
    // `extract_outputs` and the verifier's reconstruction all advance the
    // running offset by a chip's `num_interactions`, never by a rounded-up
    // width, and all padding lands in one run at the global trailing end.
    // So the axis only has to cover the sum of the RAW counts; rounding each
    // chip up to a power of two first buys no alignment anything relies on,
    // it only inflates `2^num_interaction_variables` — and every cell of that
    // axis is materialised on every GKR layer.
    let mut total_chip_interactions: usize = 0;

    for ((chip, pm), prep_trace) in
        chips.iter().zip(shared_trace_mles.iter()).zip(preprocessed_traces.iter())
    {
        // Host main-trace cells come from the shared MLE inner
        // (`guts == the raw trace`, byte-for-byte); a device-resident /
        // unexercised chip is a `dummy` → empty cells, width 0 (its real
        // cells are served by the GPU hook / provider below).
        let (mt_values, mt_width): (&[F], usize) = match pm.inner().as_ref() {
            Some(mle) => (mle.guts().as_slice(), pm.num_polynomials()),
            None => (&[], 0),
        };
        let interactions: Vec<(&Lookup<F>, bool)> = chip
            .sends()
            .iter()
            .map(|s| (s, true))
            .chain(chip.receives().iter().map(|r| (r, false)))
            .collect();
        let num_interactions = interactions.len();

        let (numer_mat, denom_mat) = build_chip_interaction_tables::<F, EF>(
            &interactions,
            mt_values,
            mt_width,
            prep_trace.real_trace_ref(),
            alpha,
            betas,
        );

        // PaddedMle row optimisation:  do NOT materialise
        // the row padding here.  Compute the per-chip real row count,
        // then split the real prefix into the upper/lower halves
        // without expanding to `2^num_row_variables`.  Virtual rows
        // beyond `num_real_rows` are resolved at access time inside
        // `ChipLayerState` using each quadrant's identity-fraction
        // pad value (n* → 0, d* → 1).
        let chip_height: usize =
            if num_interactions == 0 { 0 } else { numer_mat.values.len() / num_interactions };
        debug_assert!(chip_height <= 1usize << num_row_variables);
        // Parity split: quadrant 0 = even rows, quadrant 1 = odd rows.
        let real_upper = chip_height.div_ceil(2);
        let real_lower = chip_height / 2;
        let (n_upper, n_lower) =
            split_real_parity(&numer_mat.values, num_interactions, chip_height);
        let (d_upper, d_lower) =
            split_real_parity(&denom_mat.values, num_interactions, chip_height);

        // Encode each half as a `RowMajorTable` with raw per-chip
        // `num_interactions` storage (no per-chip column padding —
        // the `PaddedMle` pattern; padding is virtual via
        // `num_interaction_variables` metadata).  Layer-wide global
        // `num_interaction_variables` is computed below from
        // `total_chip_interactions` (sum of per-chip raw counts).
        total_chip_interactions += num_interactions;
        let make_table = |cells: Vec<F>, real_rows: usize| -> RowMajorTable<F> {
            RowMajorTable::from_padded_cells(
                cells,
                num_row_variables - 1,
                num_interactions,
                real_rows,
            )
        };
        let make_table_ef = |cells: Vec<EF>, real_rows: usize| -> RowMajorTable<EF> {
            RowMajorTable::from_padded_cells(
                cells,
                num_row_variables - 1,
                num_interactions,
                real_rows,
            )
        };

        numerator_0.push(make_table(n_upper, real_upper));
        numerator_1.push(make_table(n_lower, real_lower));
        denominator_0.push(make_table_ef(d_upper, real_upper));
        denominator_1.push(make_table_ef(d_lower, real_lower));
    }

    let num_interaction_variables =
        total_chip_interactions.max(1).next_power_of_two().trailing_zeros() as usize;

    LogUpGkrCpuLayer {
        numerator_0,
        denominator_0,
        numerator_1,
        denominator_1,
        num_row_variables: num_row_variables - 1,
        num_interaction_variables,
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    use super::*;
    use crate::Challenge;

    type SC = crate::koala_bear_poseidon2::KoalaBearPoseidon2;
    type EF = Challenge<SC>;

    #[test]
    fn split_real_parity_deinterleaves_rows() {
        // 3 real rows × 2 cols: quadrant 0 gets rows 0 and 2, quadrant 1
        // gets row 1.
        let values: Vec<u32> = vec![10, 11, 20, 21, 30, 31];
        let (even, odd) = split_real_parity(&values, 2, 3);
        assert_eq!(even, vec![10, 11, 30, 31]);
        assert_eq!(odd, vec![20, 21]);
        // A single real row has no odd partner.
        let values: Vec<u32> = vec![1, 2];
        let (even, odd) = split_real_parity(&values, 2, 1);
        assert_eq!(even, vec![1, 2]);
        assert!(odd.is_empty());
        // Nothing real → nothing materialised.
        let (even, odd) = split_real_parity::<u32>(&[], 2, 0);
        assert!(even.is_empty() && odd.is_empty());
    }

    #[test]
    fn generate_interaction_vals_signs_multiplicity_for_receives() {
        use p3_air::{PairCol, VirtualPairCol};

        let interaction = Lookup {
            values: vec![],
            multiplicity: VirtualPairCol::new(
                vec![(PairCol::Main(0), KoalaBear::ONE)],
                KoalaBear::ZERO,
            ),
            kind: crate::lookup::LookupKind::Byte,
            scope: crate::air::LookupScope::Local,
        };
        let main_row = vec![KoalaBear::from_u32(7)];

        // Single-element betas vec: only the argument_index slot is active.
        let alpha = EF::from_u32(11);
        let beta_0 = EF::from_u32(13);
        let betas = vec![beta_0];

        let (n_send, _) = generate_interaction_vals::<KoalaBear, EF>(
            &interaction,
            &[],
            &main_row,
            true,
            alpha,
            &betas,
        );
        let (n_recv, _) = generate_interaction_vals::<KoalaBear, EF>(
            &interaction,
            &[],
            &main_row,
            false,
            alpha,
            &betas,
        );
        assert_eq!(n_send, KoalaBear::from_u32(7));
        assert_eq!(n_recv, -KoalaBear::from_u32(7));
    }

    #[test]
    fn generate_interaction_vals_denominator_includes_argument_index() {
        use p3_air::{PairCol, VirtualPairCol};

        // Two interactions: kind=Byte (argument_index=4) and kind=Range (=5).
        let interaction_byte = Lookup {
            values: vec![],
            multiplicity: VirtualPairCol::new(
                vec![(PairCol::Main(0), KoalaBear::ONE)],
                KoalaBear::ZERO,
            ),
            kind: crate::lookup::LookupKind::Byte,
            scope: crate::air::LookupScope::Local,
        };
        let interaction_range = Lookup {
            values: vec![],
            multiplicity: VirtualPairCol::new(
                vec![(PairCol::Main(0), KoalaBear::ONE)],
                KoalaBear::ZERO,
            ),
            kind: crate::lookup::LookupKind::Range,
            scope: crate::air::LookupScope::Local,
        };
        let main_row = vec![KoalaBear::ONE];
        let alpha = EF::ZERO;
        let beta_0 = EF::from_u32(2);
        let betas = vec![beta_0];

        let (_, d_byte) = generate_interaction_vals::<KoalaBear, EF>(
            &interaction_byte,
            &[],
            &main_row,
            true,
            alpha,
            &betas,
        );
        let (_, d_range) = generate_interaction_vals::<KoalaBear, EF>(
            &interaction_range,
            &[],
            &main_row,
            true,
            alpha,
            &betas,
        );
        // d = alpha + beta_0 * argument_index = 0 + 2 * argi
        assert_eq!(d_byte, EF::from_u32(2 * 4));
        assert_eq!(d_range, EF::from_u32(2 * 5));
    }
}
