use std::borrow::BorrowMut;

use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use zkm_core_executor::ByteOpcode;
use zkm_core_executor::{
    events::{BranchEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use zkm_pcs::{air::MachineAir, PicusInfo, Word};

use crate::{
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

impl<F: PrimeField32> MachineAir<F> for BranchChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Branch".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        BranchColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.branch_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <BranchChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max((input.branch_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <BranchChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_BRANCH_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_BRANCH_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                rows.chunks_mut(NUM_BRANCH_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut BranchColumns<F> = row.borrow_mut();

                    if idx < input.branch_events.len() {
                        let event = &input.branch_events[idx];
                        self.event_to_row(
                            event,
                            cols,
                            &mut blu,
                            &input.program,
                            input.public_values.execution_shard,
                        );
                    } else {
                        // A padding row's frame needs no neutralising: the
                        // typed I-type frame's register-access multiplicities
                        // are `is_real`.
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_BRANCH_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.branch_events.is_empty()
        }
    }
}

impl BranchChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &BranchEvent,
        cols: &mut BranchColumns<F>,
        blu: &mut zkm_core_executor::events::ByteLookupMap,
        program: &zkm_core_executor::Program,
        shard: u32,
    ) {
        // Every Branch row is a real instruction owning its frame.
        cols.frame.populate_from_branch(event, program, shard, blu);

        cols.pc = F::from_u32(event.pc);
        cols.is_beq = F::from_bool(matches!(event.opcode, Opcode::BEQ));
        cols.is_bne = F::from_bool(matches!(event.opcode, Opcode::BNE));
        cols.is_bltz = F::from_bool(matches!(event.opcode, Opcode::BLTZ));
        cols.is_bgtz = F::from_bool(matches!(event.opcode, Opcode::BGTZ));
        cols.is_blez = F::from_bool(matches!(event.opcode, Opcode::BLEZ));
        cols.is_bgez = F::from_bool(matches!(event.opcode, Opcode::BGEZ));

        let a_eq_b = event.a == event.b;

        let a_lt_b = (event.a as i32) < (event.b as i32);
        let a_gt_b = (event.a as i32) > (event.b as i32);

        // Equality gadget: IsZero of the two 16-bit limb differences.
        let ab = event.a.to_le_bytes();
        let bb = event.b.to_le_bytes();
        let limb_diff = |lo: usize| {
            F::from_u8(ab[lo]) - F::from_u8(bb[lo])
                + (F::from_u8(ab[lo + 1]) - F::from_u8(bb[lo + 1])) * F::from_u32(1 << 8)
        };
        let d_lo = limb_diff(0);
        let d_hi = limb_diff(2);
        if d_lo == F::ZERO {
            cols.eq_lo = F::ONE;
        } else {
            cols.eq_lo_inv = d_lo.inverse();
        }
        if d_hi == F::ZERO {
            cols.eq_hi = F::ONE;
        } else {
            cols.eq_hi_inv = d_hi.inverse();
        }
        cols.a_eq_b = cols.eq_lo * cols.eq_hi;
        debug_assert_eq!(cols.a_eq_b == F::ONE, a_eq_b);

        // Sign bit + a>0, bound (and looked up) only on the zero-compare rows.
        if matches!(event.opcode, Opcode::BLTZ | Opcode::BLEZ | Opcode::BGTZ | Opcode::BGEZ) {
            let msb = (event.a >> 31) & 1;
            cols.msb_a = F::from_u32(msb);
            cols.a_gt_0 = F::from_bool(msb == 0 && !a_eq_b);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: msb as u16,
                a2: 0,
                b: ab[3],
                c: 0,
            });
        }

        let branching = match event.opcode {
            Opcode::BEQ => a_eq_b,
            Opcode::BNE => !a_eq_b,
            Opcode::BLTZ => a_lt_b,
            Opcode::BLEZ => a_lt_b || a_eq_b,
            Opcode::BGTZ => a_gt_b,
            Opcode::BGEZ => a_eq_b || a_gt_b,
            _ => panic!("Invalid opcode: {}", event.opcode),
        };

        cols.next_pc = Word::from(event.next_pc);
        cols.next_next_pc = Word::from(event.next_next_pc);
        cols.next_pc_range_checker.populate(blu, event.next_pc);
        cols.next_next_pc_range_checker.populate(blu, event.next_next_pc);
        cols.is_branching = F::from_bool(branching);
        // The (when taken) target addition, with its byte events.
        if branching {
            cols.target_add.populate(blu, event.next_pc, event.c);
        } else {
            blu.add_u8_range_checks(&event.next_pc.to_le_bytes());
            blu.add_u8_range_checks(&event.next_next_pc.to_le_bytes());
        }
    }
}
