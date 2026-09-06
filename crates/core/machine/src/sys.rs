use p3_koala_bear::KoalaBear;
use zkm_core_executor::events::{
    AluEvent, BranchEvent, CompAluEvent, JumpEvent, MemInstrEvent, MemoryBumpEvent,
    MemoryInitializeFinalizeEvent, MemoryLocalEvent, MiscEvent, MovCondEvent, SyscallEvent,
};
use zkm_core_executor::InstructionFfi;

use crate::alu::{BitwiseCols, BitwiseImmCols, CloClzCols, DivRemCols};
use crate::{
    alu::{
        AddSubCols, AddSubImmCols, LtCols, LtImmCols, MulCols, ShiftLeftCols, ShiftLeftImmCols,
        ShiftRightCols, ShiftRightImmCols,
    },
    control_flow::{BranchColumns, JumpColumns},
    memory::{
        LoadNarrowColumns, LoadWordColumns, MemoryBumpCols, MemoryInitCols, MemoryUnalignedColumns,
        SingleMemoryLocal, StoreNarrowColumns, StoreWordColumns,
    },
    misc::columns::MiscInstrColumns,
    misc::mov_cond::MovCondCols,
    syscall::chip::SyscallCols,
    syscall::instructions::columns::SyscallInstrColumns,
};

#[link(name = "zkm-core-machine-sys", kind = "static")]
extern "C-unwind" {
    pub fn add_sub_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut AddSubCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn add_sub_imm_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut AddSubImmCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn memory_local_event_to_row_koalabear(
        event: &MemoryLocalEvent,
        cols: &mut SingleMemoryLocal<KoalaBear>,
    );
    pub fn memory_bump_event_to_row_koalabear(
        event: &MemoryBumpEvent,
        cols: &mut MemoryBumpCols<KoalaBear>,
    );
    pub fn memory_global_event_to_row_koalabear(
        event: &MemoryInitializeFinalizeEvent,
        is_receive: bool,
        cols: &mut MemoryInitCols<KoalaBear>,
    );
    pub fn syscall_core_event_to_row_koalabear(
        event: &SyscallEvent,
        cols: &mut SyscallCols<KoalaBear>,
    );
    pub fn syscall_precompile_event_to_row_koalabear(
        event: &SyscallEvent,
        cols: &mut SyscallCols<KoalaBear>,
    );
    pub fn lt_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut LtCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn lt_imm_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut LtImmCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn bitwise_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut BitwiseCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn bitwise_imm_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut BitwiseImmCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn clo_clz_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut CloClzCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn branch_event_to_row_koalabear(
        event: &BranchEvent,
        cols: &mut BranchColumns<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn jump_event_to_row_koalabear(
        event: &JumpEvent,
        cols: &mut JumpColumns<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn misc_instrs_event_to_row_koalabear(
        event: &MiscEvent,
        cols: &mut MiscInstrColumns<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn mov_cond_event_to_row_koalabear(
        event: &MovCondEvent,
        cols: &mut MovCondCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn shift_left_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut ShiftLeftCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn shift_left_imm_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut ShiftLeftImmCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn shift_right_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut ShiftRightCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn shift_right_imm_event_to_row_koalabear(
        event: &AluEvent,
        cols: &mut ShiftRightImmCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn div_rem_event_to_row_koalabear(
        event: &CompAluEvent,
        cols: &mut DivRemCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn mul_event_to_row_koalabear(
        event: &CompAluEvent,
        cols: &mut MulCols<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );
    pub fn memory_load_narrow_event_to_row_koalabear(
        event: &MemInstrEvent,
        cols: &mut LoadNarrowColumns<KoalaBear>,
        instruction: InstructionFfi,
    );
    pub fn memory_load_word_event_to_row_koalabear(
        event: &MemInstrEvent,
        cols: &mut LoadWordColumns<KoalaBear>,
        instruction: InstructionFfi,
    );
    pub fn memory_store_narrow_event_to_row_koalabear(
        event: &MemInstrEvent,
        cols: &mut StoreNarrowColumns<KoalaBear>,
        instruction: InstructionFfi,
    );
    pub fn memory_store_word_event_to_row_koalabear(
        event: &MemInstrEvent,
        cols: &mut StoreWordColumns<KoalaBear>,
        instruction: InstructionFfi,
    );
    pub fn memory_unaligned_event_to_row_koalabear(
        event: &MemInstrEvent,
        cols: &mut MemoryUnalignedColumns<KoalaBear>,
        instruction: InstructionFfi,
    );
    pub fn syscall_instrs_event_to_row_koalabear(
        event: &SyscallEvent,
        cols: &mut SyscallInstrColumns<KoalaBear>,
        instruction: InstructionFfi,
        shard: u32,
    );

    // Septic-extension self-checks inside the C++ library. Only this module's
    // own test suite calls them, so they stay private to `sys` instead of
    // sitting in the crate's public API.
    fn test_mul();
    fn test_inv();
    fn test_sqrt();
    fn test_curve_formula();
}

#[cfg(test)]
mod tests {
    use crate::sys::{test_curve_formula, test_inv, test_mul, test_sqrt};

    #[test]
    fn test_septic() {
        unsafe { test_mul() };
        unsafe { test_inv() };
        unsafe { test_sqrt() };
        unsafe { test_curve_formula() };
    }
}

#[cfg(test)]
mod parity_tests {
    //! FFI ⇔ Rust trace parity on REAL executed records.
    //!
    //! The per-chip fixtures (`alu::add_sub`, `alu::mul`) only exercise
    //! dependency-shaped events; this suite executes real programs so every
    //! instruction chip's C++ `event_to_row` is checked against the Rust one
    //! on rows that carry a live instruction frame — the exact bytes the GPU
    //! tracegen kernels must reproduce.

    use std::borrow::BorrowMut;

    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use zkm_core_executor::{
        ExecutionRecord, Executor, Instruction, InstructionFfi, Opcode, Program,
    };
    use zkm_pcs::{air::MachineAir, ZKMCoreOpts};

    use crate::alu::{
        AddSubChip, AddSubCols, AddSubImmChip, AddSubImmCols, BitwiseChip, BitwiseCols,
        BitwiseImmChip, BitwiseImmCols, CloClzChip, CloClzCols, DivRemChip, DivRemCols, LtChip,
        LtCols, LtImmChip, LtImmCols, MulChip, MulCols, ShiftLeft, ShiftLeftCols, ShiftLeftImm,
        ShiftLeftImmCols, ShiftRightChip, ShiftRightCols, ShiftRightImmChip, ShiftRightImmCols,
        NUM_ADD_SUB_COLS, NUM_ADD_SUB_IMM_COLS, NUM_BITWISE_COLS, NUM_BITWISE_IMM_COLS,
        NUM_CLOCLZ_COLS, NUM_DIVREM_COLS, NUM_LT_COLS, NUM_LT_IMM_COLS, NUM_MUL_COLS,
        NUM_SHIFT_LEFT_COLS, NUM_SHIFT_LEFT_IMM_COLS, NUM_SHIFT_RIGHT_COLS,
        NUM_SHIFT_RIGHT_IMM_COLS,
    };
    use crate::control_flow::{
        BranchChip, BranchColumns, JumpChip, JumpColumns, NUM_BRANCH_COLS, NUM_JUMP_COLS,
    };
    use crate::memory::{
        LoadNarrowChip, LoadNarrowColumns, LoadWordChip, LoadWordColumns, MemoryUnalignedChip,
        MemoryUnalignedColumns, StoreNarrowChip, StoreNarrowColumns, StoreWordChip,
        StoreWordColumns, NUM_LOAD_NARROW_COLS, NUM_LOAD_WORD_COLS, NUM_MEMORY_UNALIGNED_COLS,
        NUM_STORE_NARROW_COLS, NUM_STORE_WORD_COLS,
    };
    use crate::misc::mov_cond::{MovCondChip, MovCondCols, NUM_MOV_COND_COLS};
    use crate::misc::others::columns::{MiscInstrColumns, NUM_MISC_INSTR_COLS};
    use crate::misc::MiscInstrsChip;
    use crate::programs::tests::fibonacci_program;
    use crate::syscall::instructions::{
        columns::{SyscallInstrColumns, NUM_SYSCALL_INSTR_COLS},
        SyscallInstrsChip,
    };

    type F = KoalaBear;

    /// The instruction passed for a dependency row — never read by the C++
    /// side (`is_instruction == 0` skips the frame), any well-formed value do.
    fn dummy_instruction() -> InstructionFfi {
        Instruction::new(Opcode::ADD, 0, 0, 0, true, true).into()
    }

    /// Build the FFI-side trace: one C++ `event_to_row` call per event row,
    /// the chip's pad shape beyond.
    #[allow(clippy::too_many_arguments)]
    fn build_ffi_trace<E>(
        events: &[E],
        num_cols: usize,
        height: usize,
        program: &Program,
        shard: u32,
        is_instruction: impl Fn(&E) -> bool,
        pc_of: impl Fn(&E) -> u32,
        fill: impl Fn(&E, &mut [F], InstructionFfi, u32),
        pad: impl Fn(&mut [F]),
    ) -> RowMajorMatrix<F> {
        let mut values = vec![F::default(); height * num_cols];
        for (i, row) in values.chunks_mut(num_cols).enumerate() {
            if i < events.len() {
                let event = &events[i];
                let instruction = if is_instruction(event) {
                    program.fetch(pc_of(event)).into()
                } else {
                    dummy_instruction()
                };
                fill(event, row, instruction, shard);
            } else {
                pad(row);
            }
        }
        RowMajorMatrix::new(values, num_cols)
    }

    /// Panic with the first few differing cells, chip-relative.
    fn assert_traces_eq(rust: &RowMajorMatrix<F>, ffi: &RowMajorMatrix<F>, what: &str) {
        if rust == ffi {
            return;
        }
        let w = rust.width();
        let mut diffs = vec![];
        for (i, (r, f)) in rust.values.iter().zip(ffi.values.iter()).enumerate() {
            if r != f {
                diffs.push((i / w, i % w, *r, *f));
                if diffs.len() >= 12 {
                    break;
                }
            }
        }
        panic!(
            "{what}: FFI trace diverges from Rust; first diffs (row, col, rust, ffi): {diffs:?}"
        );
    }

    /// Run every instruction chip of `record` through both sides and compare.
    fn check_record(record: &ExecutionRecord, label: &str) {
        let program = &record.program;
        let shard = record.public_values.execution_shard;

        macro_rules! check {
            ($chip:expr, $events:expr, $ColsTy:ty, $num_cols:expr, $ffi:ident, $pad:expr) => {{
                let chip = $chip;
                let rust: RowMajorMatrix<F> =
                    chip.generate_trace(record, &mut ExecutionRecord::default()).unwrap();
                let ffi = build_ffi_trace(
                    $events,
                    $num_cols,
                    rust.height(),
                    program,
                    shard,
                    |e| e.is_instruction != 0,
                    |e| e.pc,
                    |e, row, instr, sh| {
                        let cols: &mut $ColsTy = row.borrow_mut();
                        unsafe {
                            crate::sys::$ffi(e, cols, instr, sh);
                        }
                    },
                    $pad,
                );
                assert_traces_eq(
                    &rust,
                    &ffi,
                    &format!("{label}: {}", MachineAir::<F>::name(&chip)),
                );
            }};
        }

        // The universal pad: zero row + neutralised frame.
        macro_rules! dep_pad {
            ($ColsTy:ty) => {
                |row: &mut [F]| {
                    let cols: &mut $ColsTy = row.borrow_mut();
                    cols.frame.populate_dependency();
                }
            };
        }

        check!(
            AddSubChip::default(),
            &record.add_sub_events,
            AddSubCols<F>,
            NUM_ADD_SUB_COLS,
            add_sub_event_to_row_koalabear,
            // The typed R-type frame needs no neutralising: a padding row is
            // simply zero.
            |_row: &mut [F]| {}
        );
        check!(
            AddSubImmChip::default(),
            &record.add_sub_imm_events,
            AddSubImmCols<F>,
            NUM_ADD_SUB_IMM_COLS,
            add_sub_imm_event_to_row_koalabear,
            // Typed I-type frame — zero padding, as above.
            |_row: &mut [F]| {}
        );
        check!(
            BitwiseChip::default(),
            &record.bitwise_events,
            BitwiseCols<F>,
            NUM_BITWISE_COLS,
            bitwise_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            BitwiseImmChip::default(),
            &record.bitwise_imm_events,
            BitwiseImmCols<F>,
            NUM_BITWISE_IMM_COLS,
            bitwise_imm_event_to_row_koalabear,
            // Typed I-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            LtChip::default(),
            &record.lt_events,
            LtCols<F>,
            NUM_LT_COLS,
            lt_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            LtImmChip::default(),
            &record.lt_imm_events,
            LtImmCols<F>,
            NUM_LT_IMM_COLS,
            lt_imm_event_to_row_koalabear,
            // Typed I-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            CloClzChip::default(),
            &record.cloclz_events,
            CloClzCols<F>,
            NUM_CLOCLZ_COLS,
            clo_clz_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            ShiftLeft::default(),
            &record.shift_left_events,
            ShiftLeftCols<F>,
            NUM_SHIFT_LEFT_COLS,
            shift_left_event_to_row_koalabear,
            |row: &mut [F]| {
                let cols: &mut ShiftLeftCols<F> = row.borrow_mut();
                // Mirrors shift_left's padded_row_template.
                use p3_field::PrimeCharacteristicRing;
                cols.shift_by_n_bits[0] = F::ONE;
                cols.shift_by_n_bytes[0] = F::ONE;
                cols.bit_shift_multiplier = F::ONE;
            }
        );
        check!(
            ShiftLeftImm::default(),
            &record.shift_left_imm_events,
            ShiftLeftImmCols<F>,
            NUM_SHIFT_LEFT_IMM_COLS,
            shift_left_imm_event_to_row_koalabear,
            |row: &mut [F]| {
                // Mirrors the chip's padded_row_template; the typed frame
                // itself needs no neutralising.
                let cols: &mut ShiftLeftImmCols<F> = row.borrow_mut();
                cols.shift_by_n_bits[0] = F::ONE;
                cols.shift_by_n_bytes[0] = F::ONE;
                cols.bit_shift_multiplier = F::ONE;
            }
        );
        check!(
            ShiftRightChip::default(),
            &record.shift_right_events,
            ShiftRightCols<F>,
            NUM_SHIFT_RIGHT_COLS,
            shift_right_event_to_row_koalabear,
            |row: &mut [F]| {
                let cols: &mut ShiftRightCols<F> = row.borrow_mut();
                // Mirrors shift_right's padding branch.
                use p3_field::PrimeCharacteristicRing;
                cols.shift_by_n_bits[0] = F::ONE;
                cols.shift_by_n_bytes[0] = F::ONE;
            }
        );
        check!(
            ShiftRightImmChip::default(),
            &record.shift_right_imm_events,
            ShiftRightImmCols<F>,
            NUM_SHIFT_RIGHT_IMM_COLS,
            shift_right_imm_event_to_row_koalabear,
            |row: &mut [F]| {
                // Mirrors the chip's padding branch; the typed frame itself
                // needs no neutralising.
                let cols: &mut ShiftRightImmCols<F> = row.borrow_mut();
                cols.shift_by_n_bits[0] = F::ONE;
                cols.shift_by_n_bytes[0] = F::ONE;
            }
        );
        check!(
            MulChip::default(),
            &record.mul_events,
            MulCols<F>,
            NUM_MUL_COLS,
            mul_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            DivRemChip::default(),
            &record.divrem_events,
            DivRemCols<F>,
            NUM_DIVREM_COLS,
            div_rem_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            BranchChip::default(),
            &record.branch_events,
            BranchColumns<F>,
            NUM_BRANCH_COLS,
            branch_event_to_row_koalabear,
            // Typed I-type frame — zero padding.
            |_row: &mut [F]| {}
        );
        check!(
            JumpChip::default(),
            &record.jump_events,
            JumpColumns<F>,
            NUM_JUMP_COLS,
            jump_event_to_row_koalabear,
            dep_pad!(JumpColumns<F>)
        );
        check!(
            MovCondChip::default(),
            &record.movcond_events,
            MovCondCols<F>,
            NUM_MOV_COND_COLS,
            mov_cond_event_to_row_koalabear,
            dep_pad!(MovCondCols<F>)
        );
        check!(
            MiscInstrsChip::default(),
            &record.misc_events,
            MiscInstrColumns<F>,
            NUM_MISC_INSTR_COLS,
            misc_instrs_event_to_row_koalabear,
            dep_pad!(MiscInstrColumns<F>)
        );
        check!(
            SyscallInstrsChip::default(),
            &record.syscall_events,
            SyscallInstrColumns<F>,
            NUM_SYSCALL_INSTR_COLS,
            syscall_instrs_event_to_row_koalabear,
            // Typed R-type frame — zero padding.
            |_row: &mut [F]| {}
        );

        // The five memory chips: the FFI takes no shard (the event carries it).
        macro_rules! check_mem {
            ($chip:expr, $events:expr, $ColsTy:ty, $num_cols:expr, $ffi:ident) => {{
                let chip = $chip;
                let rust: RowMajorMatrix<F> =
                    chip.generate_trace(record, &mut ExecutionRecord::default()).unwrap();
                let ffi = build_ffi_trace(
                    $events,
                    $num_cols,
                    rust.height(),
                    program,
                    shard,
                    |e| e.is_instruction != 0,
                    |e| e.pc,
                    |e, row, instr, _sh| {
                        let cols: &mut $ColsTy = row.borrow_mut();
                        unsafe {
                            crate::sys::$ffi(e, cols, instr);
                        }
                    },
                    // The memory chips carry a typed I-type frame, whose
                    // register-access multiplicities are `is_real`: a padding
                    // row is simply zero and needs no neutralising.
                    |_row: &mut [F]| {},
                );
                assert_traces_eq(
                    &rust,
                    &ffi,
                    &format!("{label}: {}", MachineAir::<F>::name(&chip)),
                );
            }};
        }

        check_mem!(
            LoadNarrowChip::default(),
            &record.memory_load_narrow_events,
            LoadNarrowColumns<F>,
            NUM_LOAD_NARROW_COLS,
            memory_load_narrow_event_to_row_koalabear
        );
        check_mem!(
            LoadWordChip::default(),
            &record.memory_load_word_events,
            LoadWordColumns<F>,
            NUM_LOAD_WORD_COLS,
            memory_load_word_event_to_row_koalabear
        );
        check_mem!(
            StoreNarrowChip::default(),
            &record.memory_store_narrow_events,
            StoreNarrowColumns<F>,
            NUM_STORE_NARROW_COLS,
            memory_store_narrow_event_to_row_koalabear
        );
        check_mem!(
            StoreWordChip::default(),
            &record.memory_store_word_events,
            StoreWordColumns<F>,
            NUM_STORE_WORD_COLS,
            memory_store_word_event_to_row_koalabear
        );
        check_mem!(
            MemoryUnalignedChip::default(),
            &record.memory_unaligned_events,
            MemoryUnalignedColumns<F>,
            NUM_MEMORY_UNALIGNED_COLS,
            memory_unaligned_event_to_row_koalabear
        );
    }

    fn run_and_check(program: Program, label: &str) {
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        for (i, record) in runtime.records.iter().enumerate() {
            // Coverage note: a chip with zero events still validates its
            // padding shape, but not the live instruction frame.
            eprintln!(
                "{label}[shard {i}] events: add_sub={} add_sub_imm={} bitwise={} bitwise_imm={} lt={} lt_imm={} cloclz={} sll={} sll_imm={} sr={} sr_imm={} \
                 mul={} divrem={} branch={} jump={} movcond={} misc={} syscall={} \
                 mem(ln={} lw={} sn={} sw={} un={})",
                record.add_sub_events.len(),
                record.add_sub_imm_events.len(),
                record.bitwise_events.len(),
                record.bitwise_imm_events.len(),
                record.lt_events.len(),
                record.lt_imm_events.len(),
                record.cloclz_events.len(),
                record.shift_left_events.len(),
                record.shift_left_imm_events.len(),
                record.shift_right_events.len(),
                record.shift_right_imm_events.len(),
                record.mul_events.len(),
                record.divrem_events.len(),
                record.branch_events.len(),
                record.jump_events.len(),
                record.movcond_events.len(),
                record.misc_events.len(),
                record.syscall_events.len(),
                record.memory_load_narrow_events.len(),
                record.memory_load_word_events.len(),
                record.memory_store_narrow_events.len(),
                record.memory_store_word_events.len(),
                record.memory_unaligned_events.len(),
            );
            check_record(record, &format!("{label}[shard {i}]"));
        }
    }

    #[test]
    fn test_all_instruction_chips_ffi_eq_rust_fibonacci() {
        run_and_check(fibonacci_program(), "fibonacci");
    }

    #[test]
    fn test_all_instruction_chips_ffi_eq_rust_u256_mul() {
        // The long-multiplication guest exercises the Misc chip
        // (MADDU/MSUBU) that fibonacci and keccak never touch.
        run_and_check(Program::from(test_artifacts::U256XU2048_MUL_ELF).unwrap(), "u256x2048-mul");
    }

    #[test]
    fn test_all_instruction_chips_ffi_eq_rust_keccak() {
        run_and_check(Program::from(test_artifacts::KECCAK_SPONGE_ELF).unwrap(), "keccak-sponge");
    }
}
