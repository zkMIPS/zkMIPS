//! The word-aligned stores: `SW` and `SC`.
//!
//! Like the word loads, the offset must be zero, so no offset flags are
//! witnessed and no masking is needed.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeField32;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{
    events::{ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{air::MachineAir, PicusInfo};

use crate::{
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    utils::next_multiple_of_32,
    CoreChipError,
};

use super::common::{
    assert_word_aligned, eval_memory_common, generate_memory_trace, receive_memory_instruction,
    MemoryInstrCommonCols,
};

pub const NUM_STORE_WORD_COLS: usize = size_of::<StoreWordColumns<u8>>();

/// A chip for the word-aligned store instructions.
#[derive(Default)]
pub struct StoreWordChip;

/// The column layout for `SW` and `SC`.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StoreWordColumns<T> {
    /// The columns shared by all memory instructions.
    pub common: MemoryInstrCommonCols<T>,

    /// Memory consistency columns.  Stores WRITE, so the previous value is a
    /// separate word from the written one.
    pub memory_access: crate::memory::MemoryReadWriteCols<T>,

    /// Whether this is a store word instruction.
    #[picus(selector)]
    pub is_sw: T,
    /// Whether this is a store conditional instruction.
    #[picus(selector)]
    pub is_sc: T,
}

impl<F> BaseAir<F> for StoreWordChip {
    fn width(&self) -> usize {
        NUM_STORE_WORD_COLS
    }
}

impl<AB> Air<AB> for StoreWordChip
where
    AB: ZKMCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &StoreWordColumns<AB::Var> = (*local).borrow();
        let common = &local.common;

        // SAFETY: both selectors are boolean and so is their sum.
        let is_real = local.is_sw + local.is_sc;
        builder.assert_bool(local.is_sw);
        builder.assert_bool(local.is_sc);
        builder.assert_bool(is_real.clone());

        eval_memory_common(builder, common, &local.memory_access, is_real.clone());
        assert_word_aligned(builder, common, is_real.clone());

        // The store data is the frame's committed `op_a` read directly; the
        // frame pins it to ZERO for register 0, which is exactly what storing
        // register 0 must store.
        let a_val = common.a_val();
        let mem_val = *local.memory_access.value();

        // `SW` writes `op_a` unmasked.
        builder.when(local.is_sw).assert_word_eq(mem_val, a_val);

        // `SC` writes the *previous* `op_a` and sets `op_a = 1`.  The success
        // flag write is discarded for register 0 (the frame pins the commit to
        // zero there), so the flag shape is only asserted off register 0.
        builder.when(local.is_sc).assert_word_eq(mem_val, common.prev_a_val());
        let sc_flag_gate = local.is_sc * (AB::Expr::ONE - common.frame.op_a_0);
        builder.when(sc_flag_gate.clone()).assert_one(a_val[0]);
        builder.when(sc_flag_gate.clone()).assert_zero(a_val[1]);
        builder.when(sc_flag_gate.clone()).assert_zero(a_val[2]);
        builder.when(sc_flag_gate).assert_zero(a_val[3]);

        let opcode = local.is_sw * Opcode::SW.as_field::<AB::F>()
            + local.is_sc * Opcode::SC.as_field::<AB::F>();

        // SAFETY: `SW` keeps `op_a` immutable; `SC` writes it.
        receive_memory_instruction(builder, common, opcode, local.is_sw.into(), is_real);
    }
}

impl StoreWordChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreWordColumns<F>,
        blu: &mut impl ByteRecord,
        program: &zkm_core_executor::Program,
    ) {
        cols.common.populate(event, blu, program);
        cols.memory_access.populate_trusted(event.mem_access, blu);
        cols.is_sw = F::from_bool(matches!(event.opcode, Opcode::SW));
        cols.is_sc = F::from_bool(matches!(event.opcode, Opcode::SC));
        debug_assert!(matches!(event.opcode, Opcode::SW | Opcode::SC));
    }
}

impl<F: PrimeField32> MachineAir<F> for StoreWordChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "StoreWord".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        StoreWordColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(next_multiple_of_32(
            input.memory_store_word_events.len(),
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
            &input.memory_store_word_events,
            padded_nb_rows,
            NUM_STORE_WORD_COLS,
            |event, row, blu: &mut zkm_core_executor::events::ByteLookupMap| {
                let cols: &mut StoreWordColumns<F> = row.borrow_mut();
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
            !shard.memory_store_word_events.is_empty()
        }
    }
}
