use std::borrow::BorrowMut;
use zkm_pcs::PicusInfo;

use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use zkm_core_executor::{
    events::{ByteRecord, SyscallEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use zkm_pcs::air::MachineAir;

use crate::{
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

use super::{
    columns::{SyscallInstrColumns, NUM_SYSCALL_INSTR_COLS},
    SyscallInstrsChip,
};

impl<F: PrimeField32> MachineAir<F> for SyscallInstrsChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "SyscallInstrs".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        SyscallInstrColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.syscall_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <SyscallInstrsChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max((input.syscall_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <SyscallInstrsChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SYSCALL_INSTR_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_SYSCALL_INSTR_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                rows.chunks_mut(NUM_SYSCALL_INSTR_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SyscallInstrColumns<F> = row.borrow_mut();

                    if idx < input.syscall_events.len() {
                        let event = &input.syscall_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    } else {
                        // A padding row's frame needs no neutralising: the
                        // typed R-type frame's register-access multiplicities
                        // are `is_real`.
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_SYSCALL_INSTR_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.syscall_events.is_empty()
        }
    }
}

impl SyscallInstrsChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &SyscallEvent,
        cols: &mut SyscallInstrColumns<F>,
        _blu: &mut impl ByteRecord,
        program: &zkm_core_executor::Program,
    ) {
        // Every Syscall row is a real instruction owning its frame.
        cols.frame.populate_from_syscall(event, program, _blu);

        cols.is_real = F::ONE;
        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);
        cols.state_recv_next_pc = F::from_u32(event.recv_next_pc);

        cols.op_a_value = event.a_record.value.into();
        cols.syscall_id = F::from_u32(event.syscall_id);
        let syscall_id = F::from_u32(event.a_record.prev_value & 0xffff);
        let num_cycles = cols.frame.op_a_access.prev_value[3];

        cols.num_extra_cycles = num_cycles;
        cols.is_halt = F::from_bool(
            syscall_id == F::from_u32(SyscallCode::HALT.syscall_id())
                || syscall_id == F::from_u32(SyscallCode::SYS_EXT_GROUP.syscall_id()),
        );

        cols.is_sys_linux = F::from_bool(event.a_record.prev_value & 0x0ff00 != 0);

        let prev_a_bytes = event.a_record.prev_value.to_le_bytes();
        let is_halt_val = cols.is_halt == F::ONE;

        // Populate is_prev_a1_zero for bidirectional is_sys_linux constraint.
        cols.is_prev_a1_zero.populate_from_field_element(F::from_u8(prev_a_bytes[1]));

        // Populate `is_enter_unconstrained`.
        cols.is_enter_unconstrained.populate_from_field_element(
            syscall_id - F::from_u32(SyscallCode::ENTER_UNCONSTRAINED.syscall_id()),
        );

        // Populate `is_hint_len`.
        cols.is_hint_len.populate_from_field_element(
            syscall_id - F::from_u32(SyscallCode::SYSHINTLEN.syscall_id()),
        );

        // Populate `is_halt`.
        cols.is_halt_check
            .populate_from_field_element(syscall_id - F::from_u32(SyscallCode::HALT.syscall_id()));

        // Populate `is_exit_group`.
        cols.is_exit_group_check.populate_from_field_element(
            syscall_id - F::from_u32(SyscallCode::SYS_EXT_GROUP.syscall_id()),
        );

        // Populate `is_commit`.
        cols.is_commit.populate_from_field_element(
            syscall_id - F::from_u32(SyscallCode::COMMIT.syscall_id()),
        );

        // Populate `is_commit_deferred_proofs`.
        cols.is_commit_deferred_proofs.populate_from_field_element(
            syscall_id - F::from_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id()),
        );

        // If the syscall is `COMMIT` or `COMMIT_DEFERRED_PROOFS`, set the index bitmap and
        // digest word.
        if syscall_id == F::from_u32(SyscallCode::COMMIT.syscall_id())
            || syscall_id == F::from_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id())
        {
            let digest_idx = cols.frame.op_b_val().to_u32() as usize;
            cols.index_bitmap[digest_idx] = F::ONE;
        }

        // Populate unified KoalaBear range check flags and columns.
        let is_commit_deferred =
            syscall_id == F::from_u32(SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id());
        // Activate the KoalaBear range check only when the value travels as a single
        // reduced field element: the precompile bridge (`prev_a_bytes[2] == 1`),
        // `is_halt` (exit code), and `is_commit_deferred_proofs` (digest).  Linux
        // syscall args travel via U16-range-checked half-word columns in `SyscallChip`,
        // so the check is unnecessary there and would reject legal u32 args such as
        // AT_FDCWD = 0xFFFFFF9C.  Must match `air.rs`'s activation exactly.
        let send_to_precompile = prev_a_bytes[2] == 1;
        let op_b_needs_check = send_to_precompile || is_halt_val;
        let op_c_needs_check = send_to_precompile || is_commit_deferred;

        if op_b_needs_check {
            cols.op_b_check = F::ONE;
            cols.op_b_range_check.populate(_blu, event.arg1);
        }
        if op_c_needs_check {
            cols.op_c_check = F::ONE;
            cols.op_c_range_check.populate(_blu, event.arg2);
        }
    }
}
