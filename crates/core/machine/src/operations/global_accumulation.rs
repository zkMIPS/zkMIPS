use crate::operations::GlobalLookupOperation;
use p3_air::AirBuilder;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField32;
use zkm_derive::AlignedBorrow;
use zkm_pcs::air::SepticExtensionAirBuilder;
use zkm_pcs::septic_curve::SepticCurveComplete;
use zkm_pcs::ZKMAirBuilder;
use zkm_pcs::{
    septic_curve::SepticCurve,
    septic_extension::{SepticBlock, SepticExtension},
};

/// A set of columns needed to compute the global lookup elliptic curve digest.
/// It is critical that this struct is at the end of the main trace, as the permutation constraints will be dependent on this fact.
/// It is also critical the cumulative sum is at the end of this struct, for the same reason.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalAccumulationOperation<T, const N: usize> {
    pub initial_digest: [SepticBlock<T>; 2],
    pub cumulative_sum: [[SepticBlock<T>; 2]; N],
}

impl<T: Default, const N: usize> Default for GlobalAccumulationOperation<T, N> {
    fn default() -> Self {
        Self {
            initial_digest: core::array::from_fn(|_| SepticBlock::<T>::default()),
            cumulative_sum: core::array::from_fn(|_| {
                [SepticBlock::<T>::default(), SepticBlock::<T>::default()]
            }),
        }
    }
}

impl<F: PrimeField32, const N: usize> GlobalAccumulationOperation<F, N> {
    pub fn populate(
        &mut self,
        initial_digest: &mut SepticCurve<F>,
        global_lookup_cols: [GlobalLookupOperation<F>; N],
        is_real: [F; N],
    ) {
        self.initial_digest[0] = SepticBlock::from(initial_digest.x.0);
        self.initial_digest[1] = SepticBlock::from(initial_digest.y.0);

        for i in 0..N {
            let point_cur = SepticCurve {
                x: SepticExtension(global_lookup_cols[i].x_coordinate.0),
                y: SepticExtension(global_lookup_cols[i].y_coordinate.0),
            };
            assert!(is_real[i] == F::ONE || is_real[i] == F::ZERO);
            // Within a real row a padding slot (N > 1) keeps the running sum; whole
            // padding rows are laid out by `populate_dummy`.
            let sum_point = if is_real[i] == F::ONE {
                point_cur.add_incomplete(*initial_digest)
            } else {
                *initial_digest
            };
            self.cumulative_sum[i][0] = SepticBlock::from(sum_point.x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(sum_point.y.0);
            *initial_digest = sum_point;
        }
    }

    /// Lay out a padding row as a GENUINE addition `final_digest = (final_digest - dummy) +
    /// dummy`: `initial_digest = final_digest - dummy`, `cumulative_sum = final_digest`, with the
    /// lookup point set to `dummy` by `GlobalLookupOperation::populate_dummy`.  This keeps the
    /// unconditional `sum_checker_x == 0` true without a witness column, and keeps the shard's
    /// digest in the LAST row's trailing 14 columns, which is where the prover reads the chip's
    /// global cumulative sum from and what `permutation.rs` pins with `when_last_row`.  Padding
    /// rows are outside the `GlobalAccumulation` bus chain (multiplicity `is_real == 0`).
    pub fn populate_dummy(&mut self, final_digest: SepticCurve<F>) {
        let dummy = SepticCurve::<F>::dummy();
        let initial = final_digest.add_incomplete(dummy.neg());
        self.initial_digest[0] = SepticBlock::from(initial.x.0);
        self.initial_digest[1] = SepticBlock::from(initial.y.0);
        for i in 0..N {
            self.cumulative_sum[i][0] = SepticBlock::from(final_digest.x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(final_digest.y.0);
        }
    }

    pub fn populate_real(&mut self, sums: &[SepticCurveComplete<F>]) {
        let len = sums.len();
        debug_assert!(len >= 2);
        let sums = sums.iter().map(|complete_point| complete_point.point()).collect::<Vec<_>>();
        self.initial_digest[0] = SepticBlock::from(sums[0].x.0);
        self.initial_digest[1] = SepticBlock::from(sums[0].y.0);
        for i in 0..N {
            let s = &sums[(i + 1).min(len - 1)];
            self.cumulative_sum[i][0] = SepticBlock::from(s.x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(s.y.0);
        }
    }
}

impl<F: Field, const N: usize> GlobalAccumulationOperation<F, N> {
    pub fn eval_accumulation<AB: ZKMAirBuilder>(
        builder: &mut AB,
        global_lookup_cols: [GlobalLookupOperation<AB::Var>; N],
        local_is_real: [AB::Var; N],
        local_accumulation: GlobalAccumulationOperation<AB::Var, N>,
    ) {
        // First, constrain the control flow regarding `is_real`.
        // Constrain that all `is_real` values are boolean.
        for i in 0..N {
            builder.assert_bool(local_is_real[i]);
        }

        // Constrain that `is_real = 0` implies the next `is_real` values are all zero
        // (within-row, for N > 1).
        for i in 0..N - 1 {
            // `is_real[i] == 0` implies `is_real[i + 1] == 0`.
            builder.when_not(local_is_real[i]).assert_zero(local_is_real[i + 1]);
        }

        // Option 2: the cross-row `is_real` monotonicity is dropped — the
        // GlobalAccumulation bus does not require a contiguous real-row
        // prefix (the index chain + multiset balance handle it).

        // Next, constrain the accumulation.
        let initial_digest = SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[0][i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[1][i].into()
            }),
        };

        let assert_on_curve = |builder: &mut AB, point: SepticCurve<AB::Expr>| {
            builder.assert_septic_ext_eq(
                point.y.square(),
                SepticCurve::<AB::Expr>::curve_formula(point.x),
            );
        };

        let ith_cumulative_sum = |idx: usize| SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[idx][0].0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[idx][1].0[i].into()
            }),
        };

        let ith_point_to_add = |idx: usize| SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_lookup_cols[idx].x_coordinate.0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_lookup_cols[idx].y_coordinate.0[i].into()
            }),
        };

        // Option 2: the first-row `initial_digest == ZERO` anchor is
        // dropped — the GlobalAccumulation bus's initial endpoint
        // `(0, ZERO_DIGEST)`, emitted by the public-values AIR
        // (`eval_global_sum`) and received by row 0, enforces it.

        // Defense-in-depth: every witnessed running digest must stay on-curve even if the
        // incomplete Weierstrass addition edge case is triggered.
        assert_on_curve(builder, initial_digest.clone());

        // Constrain that when `is_real = 1`, addition is being carried out, and when `is_real = 0`, the sum remains the same.
        for i in 0..N {
            let current_sum =
                if i == 0 { initial_digest.clone() } else { ith_cumulative_sum(i - 1) };
            let point_to_add = ith_point_to_add(i);
            let next_sum = ith_cumulative_sum(i);
            assert_on_curve(builder, next_sum.clone());
            // `sum_checker_x` is degree 3 and is asserted UNCONDITIONALLY (SP1-hypercube
            // shape): padding rows are laid out as the genuine addition
            // `(final - dummy) + dummy == final` (`populate_dummy`), so no witnessed copy
            // is needed.  `sum_checker_y` is degree 2 and gated by
            // `is_real` (degree 3).  Together, on a real row, `next_sum == current_sum +
            // point_to_add` (incomplete addition, as before).
            let sum_checker_x = SepticCurve::<AB::Expr>::sum_checker_x(
                current_sum.clone(),
                point_to_add.clone(),
                next_sum.clone(),
            );
            let sum_checker_y = SepticCurve::<AB::Expr>::sum_checker_y(
                current_sum,
                point_to_add,
                next_sum,
            );
            builder.assert_septic_ext_eq(
                sum_checker_x,
                SepticExtension::<AB::Expr>::from_base_fn(|_| AB::Expr::ZERO),
            );
            builder.when(local_is_real[i]).assert_septic_ext_eq(
                sum_checker_y,
                SepticExtension::<AB::Expr>::from_base_fn(|_| AB::Expr::ZERO),
            );
        }

        // Option 2: the cross-row `final_digest == next.initial_digest`
        // chain is dropped — the GlobalAccumulation bus (emitted in
        // GlobalChip::eval as receive(index, initial_digest) +
        // send(index+1, cumulative_sum[N-1])) chains consecutive rows via
        // the multiset balance, and the public-values AIR closes the chain
        // at both ends (initial (0, ZERO), final (global_count,
        // global_cumulative_sum)).
    }
}
