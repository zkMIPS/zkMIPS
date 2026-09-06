use std::borrow::BorrowMut;
use zkm_pcs::PicusInfo;

use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use zkm_core_executor::{
    events::{ByteRecord, LinuxEvent, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use zkm_pcs::{air::MachineAir, Word};

use super::{
    columns::{SysLinuxCols, NUM_SYS_LINUX_COLS},
    SysLinuxChip,
};
use crate::{utils::pad_rows_fixed, CoreChipError};

impl<F: PrimeField32> MachineAir<F> for SysLinuxChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "SysLinux".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        SysLinuxCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = input.get_precompile_events(SyscallCode::SYS_LINUX);

        let mut rows = events
            .par_iter()
            .map(|(_, event)| {
                let event = if let PrecompileEvent::Linux(event) = event {
                    event
                } else {
                    unreachable!();
                };

                let mut row = [F::ZERO; NUM_SYS_LINUX_COLS];
                let cols: &mut SysLinuxCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu);
                row
            })
            .collect::<Vec<_>>();

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_SYS_LINUX_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <SysLinuxChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_SYS_LINUX_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let events = input.get_precompile_events(SyscallCode::SYS_LINUX);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::Linux(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    let mut row = [F::ZERO; NUM_SYS_LINUX_COLS];
                    let cols: &mut SysLinuxCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SYS_LINUX).is_empty()
        }
    }
}

impl SysLinuxChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &LinuxEvent,
        cols: &mut SysLinuxCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.a0 = event.a0.into();
        cols.a1 = event.a1.into();
        cols.shard = F::from_u32(event.shard);
        cols.clk = F::from_u32(event.clk);
        cols.syscall_id = F::from_u32(event.syscall_code);
        cols.is_real = F::ONE;
        cols.result = event.v0.into();
        cols.output.populate_write(event.write_records[0], blu);

        // ── Canonical syscall decoder ──────────────────────────────────
        let sid = F::from_u32(event.syscall_code);
        cols.decode_mmap
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_MMAP as u32));
        cols.decode_mmap2
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_MMAP2 as u32));
        cols.decode_clone
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_CLONE as u32));
        cols.decode_exit_group
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_EXT_GROUP as u32));
        cols.decode_brk.populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_BRK as u32));
        cols.decode_fnctl
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_FCNTL as u32));
        cols.decode_read
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_READ as u32));
        cols.decode_write
            .populate_from_field_element(sid - F::from_u32(SyscallCode::SYS_WRITE as u32));

        let is_mmap = event.syscall_code == SyscallCode::SYS_MMAP as u32
            || event.syscall_code == SyscallCode::SYS_MMAP2 as u32;
        cols.is_mmap = F::from_bool(is_mmap);

        // ── Canonical a0 / a1 decoder ──────────────────────────────────
        let a0_val = F::from_u32(event.a0);
        cols.decode_a0_0.populate_from_field_element(a0_val);
        cols.decode_a0_1.populate_from_field_element(a0_val - F::ONE);
        cols.decode_a0_2.populate_from_field_element(a0_val - F::TWO);

        let a1_val = F::from_u32(event.a1);
        cols.decode_a1_1.populate_from_field_element(a1_val - F::ONE);
        cols.decode_a1_3.populate_from_field_element(a1_val - F::from_u32(3));

        // ── Composite flags ────────────────────────────────────────────
        cols.is_mmap_a0_0 = F::from_bool(is_mmap && event.a0 == 0);
        cols.is_fnctl_a1_1 =
            F::from_bool(event.syscall_code == SyscallCode::SYS_FCNTL as u32 && event.a1 == 1);
        cols.is_fnctl_a1_3 =
            F::from_bool(event.syscall_code == SyscallCode::SYS_FCNTL as u32 && event.a1 == 3);

        // ── Branch-specific trace ──────────────────────────────────────
        match event.syscall_code {
            4045 => {
                // brk: read BRK register.
                assert!(event.write_records.len() == 1 && event.read_records.len() == 1);
                cols.is_a0_gt_brk.populate(event.a0, event.read_records[0].value, blu);
                cols.inorout.populate_read(event.read_records[0], blu);
            }
            4210 | 4090 => {
                // mmap / mmap2
                // byte-range-check a0 and a1 so decompositions are canonical.
                blu.add_u8_range_checks(&event.a0.to_le_bytes());
                blu.add_u8_range_checks(&event.a1.to_le_bytes());

                let a1_bytes = event.a1.to_le_bytes();
                let lo_nibble = a1_bytes[1] & 0x0F;
                let hi_nibble = (a1_bytes[1] >> 4) & 0x0F;
                for bit in 0..4 {
                    cols.a1_byte1_lo_bits[bit] = F::from_u32((lo_nibble as u32 >> bit) & 1);
                }
                for bit in 0..4 {
                    cols.a1_byte1_hi_bits[bit] = F::from_u32((hi_nibble as u32 >> bit) & 1);
                }

                let page_off = event.a1 & 0xFFF;
                let upper = (event.a1 >> 12) << 12;
                cols.is_page_offset_zero.populate_from_field_element(F::from_u32(page_off));

                if event.a0 == 0 {
                    assert!(event.write_records.len() == 2);
                    cols.inorout.populate_write(event.write_records[1], blu);
                    let size = if page_off == 0 { upper } else { upper + 0x1000 };
                    cols.mmap_size = Word::from(size);
                    blu.add_u8_range_checks(&size.to_le_bytes());

                    // Populate carry bits for byte-level mmap_size constraint.
                    if page_off != 0 {
                        let hi_nibble = (a1_bytes[1] >> 4) & 0x0F;
                        // carry[0]: (hi_nibble + 1) * 16 >= 256, i.e. hi_nibble == 15
                        if hi_nibble == 15 {
                            cols.mmap_size_carry[0] = F::ONE;
                            // carry[1]: a1[2] + 1 >= 256, i.e. a1[2] == 255
                            if a1_bytes[2] == 255 {
                                cols.mmap_size_carry[1] = F::ONE;
                            }
                        }
                    }

                    let old_heap = event.write_records[1].prev_value;
                    cols.heap_add.populate(blu, old_heap, size);
                }
            }
            4004 => {
                // write: read A2 register.
                assert!(event.read_records.len() == 1);
                cols.inorout.populate_read(event.read_records[0], blu);
            }
            4120 | 4246 | 4055 | 4003 => {
                // clone, exit_group, fnctl, read: no extra memory access needed.
            }
            _ => {
                // nop: unrecognized linux syscall.
            }
        }
    }
}
