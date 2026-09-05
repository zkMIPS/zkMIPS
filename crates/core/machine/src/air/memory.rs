use std::iter::once;

use p3_air::AirBuilder;
use p3_field::{Field, PrimeCharacteristicRing};
use zkm_core_executor::ByteOpcode;
use zkm_pcs::{
    air::{AirLookup, BaseAirBuilder, ByteAirBuilder, LookupScope},
    LookupKind,
};

use crate::{
    air::WordAirBuilder,
    memory::{MemoryAccessCols, MemoryCols, RegisterCols},
};

pub trait MemoryAirBuilder: BaseAirBuilder {
    /// Constrain a memory read or write.
    ///
    /// This method verifies that a memory access timestamp (shard, clk) is greater than the
    /// previous access's timestamp.  It will also add to the memory argument.
    fn eval_memory_access<E: Into<Self::Expr> + Clone>(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        memory_access: &impl MemoryCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let shard: Self::Expr = shard.into();
        let clk: Self::Expr = clk.into();
        let mem_access = memory_access.access();

        self.assert_bool(do_check.clone());

        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(mem_access, do_check.clone(), shard.clone(), clk.clone());

        // Defense-in-depth: memory words entering the subsystem must remain byte-shaped even
        // if an upstream chip forgot to range check them.
        //
        // A read-only access aliases `value` and `prev_value` onto the same columns;
        // checking them twice was two identical byte lookups per row (2 of the 29
        // interactions of every `LoadWord` row), so the aliased form checks once.
        if !memory_access.value_aliases_prev() {
            self.slice_range_check_u8(&memory_access.prev_value().0, do_check.clone());
        }
        self.slice_range_check_u8(&memory_access.value().0, do_check.clone());

        // Add to the memory argument.
        let addr = addr.into();
        let prev_shard = mem_access.prev_shard.clone().into();
        let prev_clk = mem_access.prev_clk.clone().into();
        let prev_values = once(prev_shard)
            .chain(once(prev_clk))
            .chain(once(addr.clone()))
            .chain(memory_access.prev_value().clone().map(Into::into))
            .collect();
        let current_values = once(shard)
            .chain(once(clk))
            .chain(once(addr.clone()))
            .chain(memory_access.value().clone().map(Into::into))
            .collect();

        // The previous values get sent with multiplicity = 1, for "read".
        self.send(
            AirLookup::new(prev_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );

        // The current values get "received", i.e. multiplicity = -1
        self.receive(
            AirLookup::new(current_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );
    }

    /// Constrain a register read or write.
    ///
    /// Registers share the memory argument and the memory address space with real memory, but the
    /// `MemoryBump` chip guarantees that the previous access to a register is always in the
    /// *current* shard: it inserts a shadow read at `(shard, 0)` on the register's first touch in
    /// the shard, and `(shard, 0)` is strictly below every real register access (those sit at the
    /// sub-cycle positions `1..=4` and `clk` restarts at 0 each shard).
    ///
    /// So `prev_shard` is not witnessed — it *is* `shard` — the `compare_clk` branch collapses to
    /// the clk comparison, and only the low limb and the top bit of the timestamp difference are
    /// witnessed.  See
    /// [`crate::memory::RegisterAccessCols`].
    fn eval_register_access<E: Into<Self::Expr> + Clone>(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        register_access: &impl RegisterCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let shard: Self::Expr = shard.into();
        let clk: Self::Expr = clk.into();
        let access = register_access.access();

        self.assert_bool(do_check.clone());

        let prev_clk: Self::Expr = access.prev_clk.clone().into();

        // Verify that the current access time is greater than the previous's.  Because
        // `prev_shard == shard`, this is always a clk comparison:
        //
        //   assert `0 <= clk - prev_clk - 1 < 2^25`
        //
        // decomposed as `diff_16bit_limb + diff_high * 2^16`.  Only the 16-bit
        // limb is a column; the 9-bit high limb is recovered as the linear
        // expression below and checked against the parametric range table.
        let diff_minus_one = clk.clone() - prev_clk.clone() - Self::Expr::ONE;
        let diff_16bit_limb: Self::Expr = access.diff_16bit_limb.clone().into();

        let diff_high_limb = (diff_minus_one - diff_16bit_limb.clone())
            * Self::F::from_u32(1 << 16).inverse();

        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::U16Range as u8),
            diff_16bit_limb,
            Self::Expr::ZERO,
            Self::Expr::ZERO,
            do_check.clone(),
        );
        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::Range as u8),
            diff_high_limb,
            Self::Expr::from_u8(9),
            Self::Expr::ZERO,
            do_check.clone(),
        );

        // Add to the memory argument, with `prev_shard` substituted by `shard`.
        let addr = addr.into();
        let prev_values = once(shard.clone())
            .chain(once(prev_clk))
            .chain(once(addr.clone()))
            .chain(register_access.prev_value().clone().map(Into::into))
            .collect();
        let current_values = once(shard)
            .chain(once(clk))
            .chain(once(addr))
            .chain(register_access.value().clone().map(Into::into))
            .collect();

        // The previous values get sent with multiplicity = 1, for "read".
        self.send(
            AirLookup::new(prev_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );

        // The current values get "received", i.e. multiplicity = -1
        self.receive(
            AirLookup::new(current_values, do_check, LookupKind::Memory),
            LookupScope::Local,
        );
    }

    /// Constraints a memory read or write to a slice of `MemoryAccessCols`.
    fn eval_memory_access_slice<E: Into<Self::Expr> + Copy>(
        &mut self,
        shard: impl Into<Self::Expr> + Copy,
        clk: impl Into<Self::Expr> + Clone,
        initial_addr: impl Into<Self::Expr> + Clone,
        memory_access_slice: &[impl MemoryCols<E>],
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) {
        for (i, access_slice) in memory_access_slice.iter().enumerate() {
            self.eval_memory_access(
                shard,
                clk.clone(),
                initial_addr.clone().into() + Self::Expr::from_usize(i * 4),
                access_slice,
                verify_memory_access,
            );
        }
    }

    /// Verifies the memory access timestamp.
    ///
    /// This method verifies that the current memory access happened after the previous one's.
    /// Specifically it will ensure that if the current and previous access are in the same shard,
    /// then the current's clk val is greater than the previous's.  If they are not in the same
    /// shard, then it will ensure that the current's shard val is greater than the previous's.
    fn eval_memory_access_timestamp(
        &mut self,
        mem_access: &MemoryAccessCols<impl Into<Self::Expr> + Clone>,
        do_check: impl Into<Self::Expr>,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let compare_clk: Self::Expr = mem_access.compare_clk.clone().into();
        let shard: Self::Expr = shard.clone().into();
        let prev_shard: Self::Expr = mem_access.prev_shard.clone().into();

        // First verify that compare_clk's value is correct.
        self.when(do_check.clone()).assert_bool(compare_clk.clone());
        self.when(do_check.clone()).when(compare_clk.clone()).assert_eq(shard.clone(), prev_shard);

        // Get the comparison timestamp values for the current and previous memory access.
        let prev_comp_value = self.if_else(
            mem_access.compare_clk.clone(),
            mem_access.prev_clk.clone(),
            mem_access.prev_shard.clone(),
        );

        let current_comp_val = self.if_else(compare_clk.clone(), clk.into(), shard.clone());

        // Assert `current_comp_val > prev_comp_val`, by asserting
        // `0 <= current_comp_val - prev_comp_val - 1 < 2^25`.
        //
        // Why that is equivalent.  Both comparands are separately bounded to `[0, 2^25)` — the
        // clk branch by [`Self::eval_range_check_25bits`] on the caller's `clk`, the shard branch
        // by the `U16Range` check on the shard index.  Write `d = a - b - 1` over the integers,
        // so `d` lies in `[-2^25, 2^25)`, and let `L` in `[0, 2^25)` be the value the limbs below
        // reconstruct; the constraint proves `d = L (mod p)`.
        //  * if `d >= 0`, then `d` and `L` are both in `[0, p)`, so `d = L >= 0`, i.e. `a > b`;
        //  * if `d < 0`, then `L = d + p >= p - 2^25`, which contradicts `L < 2^25` exactly when
        //    `p >= 2^26`.
        // KoalaBear has `p = 2^31 - 2^24 + 1`, so the argument holds for any width `<= 29 bits`
        // (`2^30 <= p < 2^31`); at 25 bits it carries a factor of ~32 of margin.  The width and
        // the executor's per-shard `clk` fence (`CORE_SHARD_CLK_LIMIT`) are the same number and
        // must be changed together.
        let diff_minus_one = current_comp_val - prev_comp_value - Self::Expr::ONE;

        // Verify that mem_access.ts_diff = diff_16bit_limb + diff_8bit_limb * 2^16
        // + diff_24bit_limb * 2^24.
        self.eval_range_check_25bits(
            diff_minus_one,
            mem_access.diff_16bit_limb.clone(),
            mem_access.diff_8bit_limb.clone(),
            mem_access.diff_24bit_limb.clone(),
            do_check,
        );
    }

    /// Verifies the inputted value is within 25 bits.
    ///
    /// `value = limb_16 + limb_8 * 2^16 + limb_24 * 2^24`, with `limb_16` checked to 16 bits and
    /// `limb_8` to 8 bits by the byte table and `limb_24` constrained boolean.
    ///
    /// `limb_24` is a column, NOT the residual of the equality.  `value` here is a timestamp
    /// difference whose comparands come out of an `if_else` on `compare_clk`, so it is already
    /// degree 2; a residual would make the guarded boolean assertion degree 5 and double the
    /// chip's quotient degree.  Witnessed, the equality stays exactly the degree it had at 24
    /// bits and the boolean assertion is degree 2 and needs no guard (the column is zero on
    /// padding rows).
    ///
    /// This is the bound the memory-argument ordering proof
    /// ([`Self::eval_memory_access_timestamp`]) assumes of both of its comparands, and it is the
    /// same width that proof range-checks their difference to.  The two must move together:
    /// widening the timestamp without widening the difference makes the argument INCOMPLETE (a
    /// legal gap stops fitting the limbs); widening the difference without bounding the
    /// timestamps makes it UNSOUND.
    fn eval_range_check_25bits(
        &mut self,
        value: impl Into<Self::Expr>,
        limb_16: impl Into<Self::Expr> + Clone,
        limb_8: impl Into<Self::Expr> + Clone,
        limb_24: impl Into<Self::Expr> + Clone,
        do_check: impl Into<Self::Expr> + Clone,
    ) {
        self.assert_bool(limb_24.clone().into());

        // Verify that value = limb_16 + limb_8 * 2^16 + limb_24 * 2^24.
        self.when(do_check.clone()).assert_eq(
            value,
            limb_16.clone().into()
                + limb_8.clone().into() * Self::Expr::from_u32(1 << 16)
                + limb_24.into() * Self::Expr::from_u32(1 << 24),
        );

        self.send_timestamp_limb_checks(limb_16, limb_8, do_check);
    }

    /// The range checks that bound a 25-bit timestamp split as
    /// `limb_16 + limb_high * 2^16` with `limb_high < 2^9`: a U16Range on the
    /// low limb and a parametric `Range(_, 9)` on the high one — no witnessed
    /// top bit at all.  For callers that build the timestamp FROM the limbs
    /// (the instruction frames) and so get the reconstruction identity free.
    fn send_timestamp_range_checks(
        &mut self,
        limb_16: impl Into<Self::Expr>,
        limb_high: impl Into<Self::Expr>,
        do_check: impl Into<Self::Expr> + Clone,
    ) {
        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::U16Range as u8),
            limb_16,
            Self::Expr::ZERO,
            Self::Expr::ZERO,
            do_check.clone(),
        );
        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::Range as u8),
            limb_high,
            Self::Expr::from_u8(9),
            Self::Expr::ZERO,
            do_check,
        );
    }

    /// The two byte-table range checks that bound a timestamp's low limbs, for callers that build
    /// the timestamp FROM those limbs (the instruction frame) and so get the reconstruction
    /// identity for free.  Such a caller still owes `assert_bool` on its top-bit column.
    fn send_timestamp_limb_checks(
        &mut self,
        limb_16: impl Into<Self::Expr>,
        limb_8: impl Into<Self::Expr>,
        do_check: impl Into<Self::Expr> + Clone,
    ) {
        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::U16Range as u8),
            limb_16,
            Self::Expr::ZERO,
            Self::Expr::ZERO,
            do_check.clone(),
        );

        self.send_byte(
            Self::Expr::from_u8(ByteOpcode::U8Range as u8),
            Self::Expr::ZERO,
            Self::Expr::ZERO,
            limb_8,
            do_check,
        )
    }
}
