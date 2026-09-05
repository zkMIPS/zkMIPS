//! The narrow (sub-word) stores: `SB` and `SH`.
//!
//! These need the offset flags to place the stored bytes inside the word, but
//! none of the load-side value or sign columns.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{
    events::{ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{air::MachineAir, PicusInfo, Word};

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    utils::next_multiple_of_32,
    CoreChipError,
};

use super::common::{
    eval_memory_common, eval_offset_flags, generate_memory_trace, populate_offset_flags,
    receive_memory_instruction, MemoryInstrCommonCols,
};

pub const NUM_STORE_NARROW_COLS: usize = size_of::<StoreNarrowColumns<u8>>();

/// A chip for the narrow store instructions.
#[derive(Default)]
pub struct StoreNarrowChip;

/// The column layout for `SB` and `SH`.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StoreNarrowColumns<T> {
    /// The columns shared by all memory instructions.
    pub common: MemoryInstrCommonCols<T>,

    /// Memory consistency columns.  Narrow stores read-modify-write.
    pub memory_access: crate::memory::MemoryReadWriteCols<T>,

    /// Whether this is a store byte instruction.
    #[picus(selector)]
    pub is_sb: T,
    /// Whether this is a store half instruction.
    #[picus(selector)]
    pub is_sh: T,

    /// Whether the least significant two bits of the address are one.
    pub ls_bits_is_one: T,
    /// Whether the least significant two bits of the address are two.
    pub ls_bits_is_two: T,
    /// Whether the least significant two bits of the address are three.
    pub ls_bits_is_three: T,
}

impl<F> BaseAir<F> for StoreNarrowChip {
    fn width(&self) -> usize {
        NUM_STORE_NARROW_COLS
    }
}

impl<AB> Air<AB> for StoreNarrowChip
where
    AB: ZKMCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &StoreNarrowColumns<AB::Var> = (*local).borrow();
        let common = &local.common;

        // SAFETY: both selectors are boolean and so is their sum.
        let is_real = local.is_sb + local.is_sh;
        builder.assert_bool(local.is_sb);
        builder.assert_bool(local.is_sh);
        builder.assert_bool(is_real.clone());

        eval_memory_common(builder, common, &local.memory_access, is_real.clone());

        let offset_is_zero = eval_offset_flags(
            builder,
            common.addr_ls_two_bits,
            local.ls_bits_is_one,
            local.ls_bits_is_two,
            local.ls_bits_is_three,
        );

        let one = AB::Expr::ONE;
        // The store data is the frame's committed `op_a` read; register 0
        // reads as zero by the frame's own pin.
        let a_val = common.a_val();
        let mem_val = *local.memory_access.value();
        let prev_mem_val = *local.memory_access.prev_value();

        // `SB`: the stored byte replaces the byte at the offset, the rest is unchanged.
        let sb_expected_stored_value = Word([
            a_val[0] * offset_is_zero.clone()
                + (one.clone() - offset_is_zero.clone()) * prev_mem_val[0],
            a_val[0] * local.ls_bits_is_one
                + (one.clone() - local.ls_bits_is_one) * prev_mem_val[1],
            a_val[0] * local.ls_bits_is_two
                + (one.clone() - local.ls_bits_is_two) * prev_mem_val[2],
            a_val[0] * local.ls_bits_is_three
                + (one.clone() - local.ls_bits_is_three) * prev_mem_val[3],
        ]);
        builder
            .when(local.is_sb)
            .assert_word_eq(mem_val.map(|x| x.into()), sb_expected_stored_value);

        // `SH` requires the offset to be zero or two.
        builder.when(local.is_sh).assert_zero(local.ls_bits_is_one + local.ls_bits_is_three);

        let a_is_lower_half = offset_is_zero;
        let a_is_upper_half = local.ls_bits_is_two;
        let sh_expected_stored_value = Word([
            a_val[0] * a_is_lower_half.clone()
                + (one.clone() - a_is_lower_half.clone()) * prev_mem_val[0],
            a_val[1] * a_is_lower_half.clone() + (one.clone() - a_is_lower_half) * prev_mem_val[1],
            a_val[0] * a_is_upper_half + (one.clone() - a_is_upper_half) * prev_mem_val[2],
            a_val[1] * a_is_upper_half + (one - a_is_upper_half) * prev_mem_val[3],
        ]);
        builder
            .when(local.is_sh)
            .assert_word_eq(mem_val.map(|x| x.into()), sh_expected_stored_value);

        let opcode = local.is_sb * Opcode::SB.as_field::<AB::F>()
            + local.is_sh * Opcode::SH.as_field::<AB::F>();

        // SAFETY: these stores keep `op_a` immutable.
        receive_memory_instruction(builder, common, opcode, is_real.clone(), is_real);
    }
}

impl StoreNarrowChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreNarrowColumns<F>,
        blu: &mut impl ByteRecord,
        program: &zkm_core_executor::Program,
    ) {
        let addr_ls_two_bits = cols.common.populate(event, blu, program);
        cols.memory_access.populate_trusted(event.mem_access, blu);
        populate_offset_flags(
            addr_ls_two_bits,
            &mut cols.ls_bits_is_one,
            &mut cols.ls_bits_is_two,
            &mut cols.ls_bits_is_three,
        );
        cols.is_sb = F::from_bool(matches!(event.opcode, Opcode::SB));
        cols.is_sh = F::from_bool(matches!(event.opcode, Opcode::SH));
        debug_assert!(matches!(event.opcode, Opcode::SB | Opcode::SH));
    }
}

impl<F: PrimeField32> MachineAir<F> for StoreNarrowChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "StoreNarrow".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        StoreNarrowColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(next_multiple_of_32(
            input.memory_store_narrow_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <Self as MachineAir<F>>::name(self).as_str(),
        ))
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let padded_nb_rows = <Self as MachineAir<F>>::num_rows(self, input).unwrap();
        let (trace, blu_events) = generate_memory_trace(
            &input.memory_store_narrow_events,
            padded_nb_rows,
            NUM_STORE_NARROW_COLS,
            |event, row, blu: &mut zkm_core_executor::events::ByteLookupMap| {
                let cols: &mut StoreNarrowColumns<F> = row.borrow_mut();
                self.event_to_row(event, cols, blu, &input.program);
            },
            // A padding row needs no neutralising: the typed frame's register-access
            // multiplicities are `is_real`, which is zero here already.
            |_row| {},
        );
        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());
        Ok(trace)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_store_narrow_events.is_empty()
        }
    }
}
