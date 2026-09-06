//! Boolean-circuit-garble **control chip** — the two endpoints of the
//! [`LookupKind::PrecompileChain`] state-chaining bus for the garble precompile.
//!
//! The single-row BaseFold zerocheck folder cannot evaluate the worker's legacy
//! `when_first_row` header logic or its `next.*` gate chaining, so the per-row
//! gate sequencing is carried on a LogUp bus instead.  This control chip emits
//! one row per `BOOLEAN_CIRCUIT_GARBLE` syscall: it receives the syscall, reads
//! the 5-word header (`num_gates` + the 16-byte `delta`), writes the output
//! word, **sends** the initial chain state `@ gate_id = 0` (with `input_address
//! = input_addr + 20`, past the header), and **receives** the final state
//! `@ gate_id = gates_num`.  Each `BooleanCircuitGarbleChip` worker row receives
//! `(gate_id, input_address)` and sends `(gate_id + 1, input_address + 68)`, so
//! the multiset only balances when the per-syscall chain telescopes
//! `0 → gates_num`, pinning each gate's index/address and the constancy of
//! `gates_num`/`delta`.

use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{events::PrecompileEvent, syscalls::SyscallCode, ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_pcs::{
    air::{AirLookup, LookupScope, MachineAir},
    LookupKind, Word, ZKMAirBuilder,
};

use crate::air::{MemoryAirBuilder, WordAirBuilder};
use crate::memory::{MemoryCols, MemoryReadCols, MemoryWriteCols};
use crate::syscall::precompiles::boolean_circuit_garble::GATE_INFO_BYTES;
use crate::{utils::pad_rows_fixed, CoreChipError};

pub const NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS: usize =
    size_of::<BooleanCircuitGarbleControlCols<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BooleanCircuitGarbleControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub is_real: T,
    pub input_address: T,
    pub output_address: T,
    pub gates_num: T,
    pub delta: [Word<T>; 4],
    /// The `num_gates` word read at `input_address`.
    pub num_gates_mem: MemoryReadCols<T>,
    /// The four `delta` words read at `input_address + 4 + 4*i`.
    pub delta_mem: [MemoryReadCols<T>; 4],
    /// The output word written at `output_address`.
    pub result_mem: MemoryWriteCols<T>,
}

/// Boolean-circuit-garble control chip.  One row per syscall.
#[derive(Default)]
pub struct BooleanCircuitGarbleControlChip;

impl BooleanCircuitGarbleControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for BooleanCircuitGarbleControlChip {
    fn width(&self) -> usize {
        NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for BooleanCircuitGarbleControlChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "BooleanCircuitGarbleControl".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        BooleanCircuitGarbleControlCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows: Vec<[F; NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS]> = Vec::new();
        let mut blu = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE) {
            let event = if let PrecompileEvent::BooleanCircuitGarble(event) = event {
                event
            } else {
                unreachable!()
            };
            let mut row = [F::ZERO; NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS];
            let cols: &mut BooleanCircuitGarbleControlCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.is_real = F::ONE;
            cols.input_address = F::from_canonical_u32(event.input_addr);
            cols.output_address = F::from_canonical_u32(event.output_addr);
            cols.gates_num = F::from_canonical_u32(event.num_gates);
            for i in 0..4 {
                let delta_i_bytes = event.delta[i].to_le_bytes();
                cols.delta[i]
                    .0
                    .iter_mut()
                    .enumerate()
                    .for_each(|(id, x)| *x = F::from_u8(delta_i_bytes[id]));
            }
            cols.num_gates_mem.populate(event.num_gates_read_record, &mut blu);
            for i in 0..4 {
                cols.delta_mem[i].populate(event.delta_read_records[i], &mut blu);
            }
            cols.result_mem.populate(event.output_write_record, &mut blu);
            rows.push(row);
        }

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <BooleanCircuitGarbleControlChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_BOOLEAN_CIRCUIT_GARBLE_CONTROL_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE).is_empty()
        }
    }
}

impl<AB> Air<AB> for BooleanCircuitGarbleControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &BooleanCircuitGarbleControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Receive the syscall once per real invocation.
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_u32(SyscallCode::BOOLEAN_CIRCUIT_GARBLE.syscall_id()),
            local.input_address,
            local.output_address,
            local.is_real,
            LookupScope::Local,
        );

        // Read the header: `num_gates` at `input_address`, `delta` at
        // `input_address + 4 + 4*i`.
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.input_address,
            &local.num_gates_mem,
            local.is_real,
        );
        for i in 0..4 {
            builder.eval_memory_access(
                local.shard,
                local.clk,
                local.input_address + AB::Expr::from_u32(4 + (i as u32) * 4),
                &local.delta_mem[i],
                local.is_real,
            );
        }

        // Bind `gates_num` to the `num_gates` word read from memory.
        let bytes_shift = AB::F::from_u32(256);
        let bs2 = bytes_shift.clone() * bytes_shift.clone();
        let bs3 = bs2.clone() * bytes_shift.clone();
        let mem_num_gates = local.num_gates_mem.access.value.0[0]
            + local.num_gates_mem.access.value.0[1] * bytes_shift
            + local.num_gates_mem.access.value.0[2] * bs2
            + local.num_gates_mem.access.value.0[3] * bs3;
        builder.when(local.is_real).assert_eq(local.gates_num.into(), mem_num_gates);

        // Bind the `delta` columns to the four `delta` words read from memory.
        for i in 0..4 {
            builder.when(local.is_real).assert_word_eq(local.delta[i], *local.delta_mem[i].value());
        }

        // Write the output word.
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.output_address,
            &local.result_mem,
            local.is_real,
        );

        // Leading precompile-ID field isolates this chain on `PrecompileChain`.
        let pid = AB::Expr::from_u32(SyscallCode::BOOLEAN_CIRCUIT_GARBLE.syscall_id());
        let tuple = |gate_id: AB::Expr, input_address: AB::Expr| -> Vec<AB::Expr> {
            let mut vals = vec![
                pid.clone(),
                local.shard.into(),
                local.clk.into(),
                gate_id,
                local.gates_num.into(),
                input_address,
            ];
            for word in local.delta.iter() {
                for b in word.0.iter() {
                    vals.push((*b).into());
                }
            }
            vals
        };

        // Send the initial chain state `@ gate_id = 0`, gates starting at
        // `input_address + 20` (past the 5-word header).
        let gates_start = local.input_address.into() + AB::Expr::from_u32(20);
        builder.send(
            AirLookup::new(
                tuple(AB::Expr::ZERO, gates_start.clone()),
                local.is_real.into(),
                LookupKind::PrecompileChain,
            ),
            LookupScope::Local,
        );

        // Receive the final chain state `@ gate_id = gates_num`, at
        // `input_address + 20 + GATE_INFO_BYTES*4 * gates_num`.
        let gates_end =
            gates_start + local.gates_num.into() * AB::Expr::from_u32((GATE_INFO_BYTES * 4) as u32);
        builder.receive(
            AirLookup::new(
                tuple(local.gates_num.into(), gates_end),
                local.is_real.into(),
                LookupKind::PrecompileChain,
            ),
            LookupScope::Local,
        );
    }
}
