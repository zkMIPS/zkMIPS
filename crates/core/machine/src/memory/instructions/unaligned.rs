//! The MIPS unaligned memory instructions: `LWL`, `LWR`, `SWL`, `SWR`.
//!
//! These four share a shape — full offset flags plus the previous `op_a` value —
//! and are rare enough in compiled MIPS that giving each its own chip would cost
//! more in per-chip overhead than it saves in columns.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, BaseAir, WindowAccess};
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

pub const NUM_MEMORY_UNALIGNED_COLS: usize = size_of::<MemoryUnalignedColumns<u8>>();

/// A chip for the unaligned load/store instructions.
#[derive(Default)]
pub struct MemoryUnalignedChip;

/// The column layout for `LWL`, `LWR`, `SWL`, `SWR`.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryUnalignedColumns<T> {
    /// The columns shared by all memory instructions.
    pub common: MemoryInstrCommonCols<T>,

    /// Memory consistency columns.  `SWL`/`SWR` read-modify-write, so the
    /// read-write form is needed (the loads simply leave value == prev).
    pub memory_access: crate::memory::MemoryReadWriteCols<T>,

    /// Whether this is a load word left instruction.
    #[picus(selector)]
    pub is_lwl: T,
    /// Whether this is a load word right instruction.
    #[picus(selector)]
    pub is_lwr: T,
    /// Whether this is a store word left instruction.
    #[picus(selector)]
    pub is_swl: T,
    /// Whether this is a store word right instruction.
    #[picus(selector)]
    pub is_swr: T,

    /// Whether the least significant two bits of the address are one.
    pub ls_bits_is_one: T,
    /// Whether the least significant two bits of the address are two.
    pub ls_bits_is_two: T,
    /// Whether the least significant two bits of the address are three.
    pub ls_bits_is_three: T,

    /// `is_lwl * (1 - op_a_0)` — gates the LWL register bind so a register-0
    /// destination (whose commit the frame pins to zero) is exempt without
    /// raising the bind's degree past the chip's existing maximum.
    pub lwl_gate: T,
    /// `is_lwr * (1 - op_a_0)` — the LWR twin of `lwl_gate`.
    pub lwr_gate: T,
}

impl<F> BaseAir<F> for MemoryUnalignedChip {
    fn width(&self) -> usize {
        NUM_MEMORY_UNALIGNED_COLS
    }
}

impl<AB> Air<AB> for MemoryUnalignedChip
where
    AB: ZKMCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &MemoryUnalignedColumns<AB::Var> = (*local).borrow();
        let common = &local.common;

        // SAFETY: all selectors are boolean and so is their sum.
        let is_real = local.is_lwl + local.is_lwr + local.is_swl + local.is_swr;
        builder.assert_bool(local.is_lwl);
        builder.assert_bool(local.is_lwr);
        builder.assert_bool(local.is_swl);
        builder.assert_bool(local.is_swr);
        builder.assert_bool(is_real.clone());

        eval_memory_common(builder, common, &local.memory_access, is_real.clone());

        let offset_is_zero = eval_offset_flags(
            builder,
            common.addr_ls_two_bits,
            local.ls_bits_is_one,
            local.ls_bits_is_two,
            local.ls_bits_is_three,
        );

        // The loads must not change the memory value.
        builder
            .when(local.is_lwl + local.is_lwr)
            .assert_word_eq(*local.memory_access.value(), *local.memory_access.prev_value());

        // The load-side register binds run against the frame's committed
        // `op_a` access; the gates carry the `(1 - op_a_0)` factor as witness
        // columns so the binds keep their existing degree.
        builder.assert_eq(local.lwl_gate, local.is_lwl * (AB::Expr::ONE - common.frame.op_a_0));
        builder.assert_eq(local.lwr_gate, local.is_lwr * (AB::Expr::ONE - common.frame.op_a_0));

        let one = AB::Expr::ONE;
        let a_val = common.a_val();
        let prev_a_val = common.prev_a_val();
        let mem_val = *local.memory_access.value();
        let prev_mem_val = *local.memory_access.prev_value();

        // `LWR`: merge the bytes at and above the offset into the low bytes of `op_a`.
        let lwr_expected_load_value = Word([
            mem_val[0] * offset_is_zero.clone()
                + mem_val[1] * local.ls_bits_is_one
                + mem_val[2] * local.ls_bits_is_two
                + mem_val[3] * local.ls_bits_is_three,
            mem_val[1] * offset_is_zero.clone()
                + mem_val[2] * local.ls_bits_is_one
                + mem_val[3] * local.ls_bits_is_two
                + prev_a_val[1] * local.ls_bits_is_three,
            mem_val[2] * offset_is_zero.clone()
                + mem_val[3] * local.ls_bits_is_one
                + prev_a_val[2] * (one.clone() - local.ls_bits_is_one - offset_is_zero.clone()),
            mem_val[3] * offset_is_zero.clone()
                + prev_a_val[3] * (one.clone() - offset_is_zero.clone()),
        ]);
        builder.when(local.lwr_gate).assert_word_eq(a_val, lwr_expected_load_value);

        // `LWL`: merge the bytes at and below the offset into the high bytes of `op_a`.
        let lwl_expected_load_value = Word([
            mem_val[0] * local.ls_bits_is_three
                + prev_a_val[0] * (one.clone() - local.ls_bits_is_three),
            mem_val[1] * local.ls_bits_is_three
                + mem_val[0] * local.ls_bits_is_two
                + prev_a_val[1] * local.ls_bits_is_one
                + prev_a_val[1] * offset_is_zero.clone(),
            mem_val[2] * local.ls_bits_is_three
                + mem_val[1] * local.ls_bits_is_two
                + mem_val[0] * local.ls_bits_is_one
                + prev_a_val[2] * offset_is_zero.clone(),
            mem_val[3] * local.ls_bits_is_three
                + mem_val[2] * local.ls_bits_is_two
                + mem_val[1] * local.ls_bits_is_one
                + mem_val[0] * offset_is_zero.clone(),
        ]);
        builder.when(local.lwl_gate).assert_word_eq(a_val, lwl_expected_load_value);

        // `SWL`: store the high bytes of `op_a` at and below the offset.
        let swl_expected_stored_value = Word([
            a_val[3] * offset_is_zero.clone()
                + a_val[2] * local.ls_bits_is_one
                + a_val[1] * local.ls_bits_is_two
                + a_val[0] * local.ls_bits_is_three,
            prev_mem_val[1] * offset_is_zero.clone()
                + a_val[3] * local.ls_bits_is_one
                + a_val[2] * local.ls_bits_is_two
                + a_val[1] * local.ls_bits_is_three,
            prev_mem_val[2] * (offset_is_zero.clone() + local.ls_bits_is_one)
                + a_val[3] * local.ls_bits_is_two
                + a_val[2] * local.ls_bits_is_three,
            prev_mem_val[3] * (one.clone() - local.ls_bits_is_three)
                + a_val[3] * local.ls_bits_is_three,
        ]);
        builder
            .when(local.is_swl)
            .assert_word_eq(mem_val.map(|x| x.into()), swl_expected_stored_value);

        // `SWR`: store the low bytes of `op_a` at and above the offset.
        let swr_expected_stored_value = Word([
            a_val[0] * offset_is_zero.clone()
                + prev_mem_val[0] * (one.clone() - offset_is_zero.clone()),
            a_val[1] * offset_is_zero.clone()
                + a_val[0] * local.ls_bits_is_one
                + prev_mem_val[1] * (local.ls_bits_is_two + local.ls_bits_is_three),
            a_val[2] * offset_is_zero.clone()
                + a_val[1] * local.ls_bits_is_one
                + a_val[0] * local.ls_bits_is_two
                + prev_mem_val[2] * local.ls_bits_is_three,
            a_val[3] * offset_is_zero.clone()
                + a_val[2] * local.ls_bits_is_one
                + a_val[1] * local.ls_bits_is_two
                + a_val[0] * local.ls_bits_is_three,
        ]);
        builder
            .when(local.is_swr)
            .assert_word_eq(mem_val.map(|x| x.into()), swr_expected_stored_value);

        let opcode = local.is_lwl * Opcode::LWL.as_field::<AB::F>()
            + local.is_lwr * Opcode::LWR.as_field::<AB::F>()
            + local.is_swl * Opcode::SWL.as_field::<AB::F>()
            + local.is_swr * Opcode::SWR.as_field::<AB::F>();

        // SAFETY: the stores keep `op_a` immutable; the loads write it.
        receive_memory_instruction(builder, common, opcode, local.is_swl + local.is_swr, is_real);
    }
}

impl MemoryUnalignedChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut MemoryUnalignedColumns<F>,
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
        cols.is_lwl = F::from_bool(matches!(event.opcode, Opcode::LWL));
        cols.is_lwr = F::from_bool(matches!(event.opcode, Opcode::LWR));
        cols.is_swl = F::from_bool(matches!(event.opcode, Opcode::SWL));
        cols.is_swr = F::from_bool(matches!(event.opcode, Opcode::SWR));
        // `op_a_0` was just populated by the frame from the fetched instruction.
        let op_a_not_zero = F::ONE - cols.common.frame.op_a_0;
        cols.lwl_gate = cols.is_lwl * op_a_not_zero;
        cols.lwr_gate = cols.is_lwr * op_a_not_zero;
        debug_assert!(matches!(
            event.opcode,
            Opcode::LWL | Opcode::LWR | Opcode::SWL | Opcode::SWR
        ));
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryUnalignedChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "MemoryUnaligned".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        MemoryUnalignedColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(next_multiple_of_32(
            input.memory_unaligned_events.len(),
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
            &input.memory_unaligned_events,
            padded_nb_rows,
            NUM_MEMORY_UNALIGNED_COLS,
            |event, row, blu: &mut zkm_core_executor::events::ByteLookupMap| {
                let cols: &mut MemoryUnalignedColumns<F> = row.borrow_mut();
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
            !shard.memory_unaligned_events.is_empty()
        }
    }
}
