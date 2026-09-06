use crate::memory::RegisterCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use zkm_core_executor::{
    events::{ByteRecord, MovCondEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{
    air::{BaseAirBuilder, MachineAir, ZKMAirBuilder},
    PicusInfo,
};

use crate::{air::WordAirBuilder, CoreChipError};

use crate::utils::{next_multiple_of_32, zeroed_f_vec};

/// The number of main trace columns for `MovCondChip`.
pub const NUM_MOV_COND_COLS: usize = size_of::<MovCondCols<u8>>();

/// A chip that implements condition mov for the opcode MNE，MEQ.
#[derive(Default)]
pub struct MovCondChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct MovCondCols<T> {
    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// `op_c == 0`, via IsZeros over its two 16-bit limbs (`c0 + 256*c1`
    /// and `c2 + 256*c3`): with byte-shaped words each limb is in
    /// `[0, 65535]`, so it vanishes in the field iff both bytes do.  The
    /// old per-byte `IsZeroWordOperation` spent 11 columns on the same fact.
    pub c_eq_lo: T,
    pub c_eq_lo_inv: T,
    pub c_eq_hi: T,
    pub c_eq_hi_inv: T,
    /// `c_eq_lo * c_eq_hi`, materialized.
    pub c_eq_0: T,

    /// `is_meq * c_eq_0 + is_mne * (1 - c_eq_0)` — whether the conditional
    /// move fires, materialized so the register binds stay at degree <= 3.
    pub sel_moved: T,

    /// Flag indicating whether the opcode is `MNE`.
    #[picus(selector)]
    pub is_mne: T,

    /// Flag indicating whether the opcode is `MEQ`.
    #[picus(selector)]
    pub is_meq: T,

    /// Flag indicating whether the opcode is `WSBH`.
    #[picus(selector)]
    pub is_wsbh: T,

    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every MovCond row is an instruction).
    pub frame: crate::frame::InstructionFrameCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for MovCondChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "MovCond".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        MovCondCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.movcond_events.len(),
            input.fixed_log2_rows::<F, _>(self),
            <MovCondChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.movcond_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <MovCondChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MOV_COND_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_MOV_COND_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
                rows.chunks_mut(NUM_MOV_COND_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MovCondCols<F> = row.borrow_mut();

                    if idx < input.movcond_events.len() {
                        let event = &input.movcond_events[idx];
                        self.event_to_row(
                            event,
                            cols,
                            &mut blu,
                            &input.program,
                            input.public_values.execution_shard,
                        );
                    } else {
                        // Padding rows carry no instruction: neutralise the
                        // frame or its register-access multiplicities break the
                        // Memory bus.
                        cols.frame.populate_dependency();
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_MOV_COND_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.movcond_events.is_empty()
        }
    }
}

impl MovCondChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MovCondEvent,
        cols: &mut MovCondCols<F>,
        _blu: &mut impl ByteRecord,
        program: &zkm_core_executor::Program,
        shard: u32,
    ) {
        // Every MovCond row is a real instruction owning its frame.
        cols.frame.populate_from_movcond(event, program, shard, _blu);

        cols.pc = F::from_u32(event.pc);
        cols.next_pc = F::from_u32(event.next_pc);

        cols.is_meq = F::from_bool(matches!(event.opcode, Opcode::MEQ));
        cols.is_mne = F::from_bool(matches!(event.opcode, Opcode::MNE));
        cols.is_wsbh = F::from_bool(matches!(event.opcode, Opcode::WSBH));

        if !matches!(event.opcode, Opcode::WSBH) {
            let cb = event.c.to_le_bytes();
            let c_lo = F::from_u32(cb[0] as u32 + ((cb[1] as u32) << 8));
            let c_hi = F::from_u32(cb[2] as u32 + ((cb[3] as u32) << 8));
            if c_lo == F::ZERO {
                cols.c_eq_lo = F::ONE;
            } else {
                cols.c_eq_lo_inv = c_lo.inverse();
            }
            if c_hi == F::ZERO {
                cols.c_eq_hi = F::ONE;
            } else {
                cols.c_eq_hi_inv = c_hi.inverse();
            }
            cols.c_eq_0 = cols.c_eq_lo * cols.c_eq_hi;
            let fired = match event.opcode {
                Opcode::MEQ => event.c == 0,
                Opcode::MNE => event.c != 0,
                _ => unreachable!(),
            };
            cols.sel_moved = F::from_bool(fired);
        }
    }
}

impl<F> BaseAir<F> for MovCondChip {
    fn width(&self) -> usize {
        NUM_MOV_COND_COLS
    }
}

impl<AB> Air<AB> for MovCondChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &MovCondCols<AB::Var> = (*local).borrow();
        let is_real = local.is_mne + local.is_meq + local.is_wsbh;

        // A real instruction carries its own program fetch, register access and
        // `(clk, pc)` chaining.  MNE/MEQ/WSBH are sequential and never halt.
        crate::frame::eval_instruction_frame(
            builder,
            &local.frame,
            local.is_meq * Opcode::MEQ.as_field::<AB::F>()
                + local.is_mne * Opcode::MNE.as_field::<AB::F>()
                + local.is_wsbh * Opcode::WSBH.as_field::<AB::F>(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc + AB::Expr::from_u32(4),
            local.next_pc.into(),
            AB::Expr::ZERO,
            is_real.clone(),
        );
        // `op_c == 0` via the two 16-bit limb IsZeros, live on the
        // conditional-move rows only (WSBH ignores `op_c`).
        let mov = local.is_mne + local.is_meq;
        let cv = local.frame.op_c_val();
        let two_pow_8 = AB::Expr::from_u32(1 << 8);
        let c_lo = cv[0] + cv[1] * two_pow_8.clone();
        let c_hi = cv[2] + cv[3] * two_pow_8;
        builder.when(mov.clone()).assert_zero(local.c_eq_lo * c_lo.clone());
        builder
            .when(mov.clone())
            .assert_eq(local.c_eq_lo, AB::Expr::ONE - c_lo * local.c_eq_lo_inv);
        builder.when(mov.clone()).assert_zero(local.c_eq_hi * c_hi.clone());
        builder
            .when(mov.clone())
            .assert_eq(local.c_eq_hi, AB::Expr::ONE - c_hi * local.c_eq_hi_inv);
        builder.when(mov.clone()).assert_eq(local.c_eq_0, local.c_eq_lo * local.c_eq_hi);

        // Whether the conditional move fires.  Unguarded: every term carries
        // a selector, so the padding row's zeros satisfy it.
        builder.assert_eq(
            local.sel_moved,
            local.is_meq * local.c_eq_0 + local.is_mne * (AB::Expr::ONE - local.c_eq_0),
        );

        // Conditional-move semantics, straight against the frame's committed
        // register access.  A fired move copies `op_b` (unless the write is a
        // discarded register-0 write, which the frame pins to zero); a failed
        // one carries the previous value through unchanged — register 0's
        // previous value IS zero, so that case needs no gate.
        {
            let av = *local.frame.op_a_access.value();
            builder
                .when(local.sel_moved)
                .when_not(local.frame.instruction.op_a_0)
                .assert_word_eq(av, local.frame.op_b_val());
            builder
                .when(mov - local.sel_moved)
                .assert_word_eq(av, local.frame.op_a_access.prev_value);
        }

        self.eval_wsbh(builder, local);
        builder.assert_bool(local.is_mne);
        builder.assert_bool(local.is_meq);
        builder.assert_bool(local.is_wsbh);
        builder.assert_bool(is_real);
    }
}

impl MovCondChip {
    pub(crate) fn eval_wsbh<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MovCondCols<AB::Var>,
    ) {
        // The swapped bytes, bound to the committed register write through a
        // `(1 - op_a_0)` factor (a register-0 destination is discarded and
        // pinned to zero by the frame) — same degree as the old plain bind.
        let av = *local.frame.op_a_access.value();
        let bv = local.frame.op_b_val();
        let not_a0 = AB::Expr::ONE - local.frame.instruction.op_a_0;
        builder.when(local.is_wsbh).assert_eq(av[0], not_a0.clone() * bv[1]);
        builder.when(local.is_wsbh).assert_eq(av[1], not_a0.clone() * bv[0]);
        builder.when(local.is_wsbh).assert_eq(av[2], not_a0.clone() * bv[3]);
        builder.when(local.is_wsbh).assert_eq(av[3], not_a0 * bv[2]);
    }
}

#[cfg(test)]
mod tests {

    use crate::{utils, utils::run_test};

    use zkm_core_executor::{Instruction, Opcode, Program};

    use zkm_pcs::CpuProver;

    #[test]
    fn test_mov_cond_prove() {
        utils::setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xf, false, true),
            Instruction::new(Opcode::ADD, 28, 0, 0x8F8F, false, true),
            Instruction::new(Opcode::MEQ, 30, 29, 0, false, false),
            Instruction::new(Opcode::MEQ, 30, 29, 28, false, false),
            Instruction::new(Opcode::MEQ, 0, 29, 0, false, false),
            Instruction::new(Opcode::MEQ, 0, 29, 29, false, false),
            Instruction::new(Opcode::MNE, 30, 29, 28, false, false),
            Instruction::new(Opcode::MNE, 0, 29, 0, false, false),
            Instruction::new(Opcode::WSBH, 32, 29, 0, false, true),
            Instruction::new(Opcode::WSBH, 32, 31, 0, false, true),
            Instruction::new(Opcode::WSBH, 0, 29, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test::<CpuProver<_, _>>(program).unwrap();
    }
}
