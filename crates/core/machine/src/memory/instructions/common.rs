//! Columns and constraints shared by every memory-instruction chip.
//!
//! The memory instructions used to live in a single 79-column, 14-selector
//! union chip (`MemoryInstrs`).  Every row of that chip paid for the columns of
//! every other memory opcode: a `LW` row carried the store-masking flags, the
//! sign-extension gadget and the unaligned-load scratch it never used.
//!
//! The union is now split into per-width chips (see the sibling modules), each
//! of which embeds this shared block plus only the columns its own opcodes
//! need.  The block also carries the *inlined* effective-address addition:
//! `addr = op_b + op_c` is proven here with an [`AddOperation`] (value + 3
//! carries) instead of being delegated to the `AddSub` chip over the ALU bus,
//! which removes one 19-cell `AddSub` dependency row per memory instruction.

use crate::memory::RegisterCols;
use std::mem::size_of;

use p3_air::AirBuilder;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::{PicusInfo, Word};

use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent, MemoryAccessPosition},
    ByteOpcode, NUM_REGISTERS,
};
use zkm_primitives::consts::WORD_SIZE;

use crate::air::WordAirBuilder;
use crate::{
    air::ZKMCoreAirBuilder,

    operations::{AddOperation, IsZeroOperation, KoalaBearWordRangeChecker},
    utils::zeroed_f_vec,
};

/// The number of columns shared by every memory-instruction chip.
pub const NUM_MEMORY_INSTR_COMMON_COLS: usize = size_of::<MemoryInstrCommonCols<u8>>();

/// The columns every memory instruction needs, regardless of width or direction.
#[derive(AlignedBorrow, PicusAnnotations, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryInstrCommonCols<T> {
    /// Program fetch, register access and `(clk, pc)` chaining; live on every
    /// real row (every memory-instruction row is an instruction).
    pub frame: crate::frame::ITypeFrameCols<T>,

    /// The current/next program counter of the instruction.
    pub pc: T,
    pub next_pc: T,

    /// The effective address `op_b + op_c`, computed INLINE.
    ///
    /// `addr_add.value` is the (unaligned) address word; the three carry bits
    /// prove the byte-wise addition.  This replaces the `send_alu(ADD, ..)`
    /// the union chip used to emit, and with it the `AddSub` row it required.
    pub addr_add: AddOperation<T>,

    /// The address's least significant two bits, i.e. `addr_word[0] & 0b11`.
    ///
    /// The aligned address is the expression `addr_word.reduce() -
    /// addr_ls_two_bits`; it is no longer a witnessed column.
    pub addr_ls_two_bits: T,

    /// Gadget to verify that the address word is within the Koala-Bear field.
    pub addr_word_range_checker: KoalaBearWordRangeChecker<T>,

    /// This is used to check if the most significant three bytes of the memory address are all
    /// zero.
    pub most_sig_bytes_zero: IsZeroOperation<T>,
}

impl<T: Copy> MemoryInstrCommonCols<T> {
    /// The (unaligned) effective address word.
    #[inline]
    pub fn addr_word(&self) -> Word<T> {
        self.addr_add.value
    }

    /// The shard this instruction executed in.
    #[inline]
    pub fn shard(&self) -> T {
        self.frame.shard
    }

    /// The value of the second operand — the frame's `op_b` register read.
    #[inline]
    pub fn op_b_value(&self) -> Word<T> {
        self.frame.op_b_val()
    }

    /// The value of the third operand — the address offset.
    #[inline]
    pub fn op_c_value(&self) -> Word<T> {
        self.frame.op_c_val()
    }

    /// The previous value of `op_a`, as committed by its register access.
    #[inline]
    pub fn prev_a_val(&self) -> Word<T> {
        self.frame.op_a_access.prev_value
    }

    /// The committed value of `op_a` — the frame's register access, directly.
    ///
    /// There is no separate `op_a_value` column any more.  The frame pins this
    /// word to ZERO when `op_a` is register 0 (`eval_i_type_frame`), so a
    /// chip binding a computed value to it must either gate on `op_a_0` or
    /// absorb the `(1 - op_a_0)` factor into the bound expression; a STORE's
    /// read needs neither (storing register 0 stores 0).
    #[inline]
    pub fn a_val(&self) -> Word<T> {
        *self.frame.op_a_access.value()
    }
}

/// Constrains everything that is common to all memory instructions:
///
/// 1. `addr_word = op_b + op_c` (inlined — no `AddSub` row).
/// 2. `addr_word` is a canonical Koala-Bear word whose bytes are range checked.
/// 3. `addr_word >= NUM_REGISTERS`, so memory instructions cannot alias registers.
/// 4. `addr_ls_two_bits = addr_word[0] & 0b11`.
/// 5. The memory access at the aligned address.
///
/// Returns the aligned-address expression (`addr_word.reduce() - addr_ls_two_bits`).
pub fn eval_memory_common<AB: ZKMCoreAirBuilder>(
    builder: &mut AB,
    cols: &MemoryInstrCommonCols<AB::Var>,
    memory_access: &impl crate::memory::MemoryCols<AB::Var>,
    is_real: AB::Expr,
) -> AB::Expr {
    // Verify `addr_word = op_b + op_c` in-place.  The OPERANDS need no
    // re-check — `op_b` is a register-file read (every write into the file is
    // range checked, so the multiset argument carries byte shape to every
    // read) and `op_c` is the program-table immediate (committed in the vk) —
    // so only the fresh address word is range checked.
    AddOperation::<AB::F>::eval_check_value_only(
        builder,
        cols.op_b_value(),
        cols.op_c_value(),
        cols.addr_add,
        is_real.clone(),
    );
    let addr_word = cols.addr_add.value;

    // Range check the addr_word to be a valid koalabear word.
    KoalaBearWordRangeChecker::<AB::F>::range_check(
        builder,
        addr_word,
        cols.addr_word_range_checker,
        is_real.clone(),
    );

    // We check that `addr_word >= NUM_REGISTERS`, or `addr_word > NUM_REGISTERS - 1` to avoid
    // registers.  Check that if the most significant bytes are zero, then the least significant
    // byte is at least NUM_REGISTERS.
    builder.send_byte(
        ByteOpcode::LTU.as_field::<AB::F>(),
        AB::Expr::ONE,
        AB::Expr::from_u8(NUM_REGISTERS as u8 - 1),
        addr_word[0],
        cols.most_sig_bytes_zero.result,
    );

    // SAFETY: Check that the above interaction is only sent if the row is real.
    builder.when(cols.most_sig_bytes_zero.result).assert_one(is_real.clone());

    // Check the most_sig_byte_zero flag.  The three most significant bytes are byte range
    // checked by `AddOperation::eval`, so the only way their sum is zero is if all are zero.
    IsZeroOperation::<AB::F>::eval(
        builder,
        addr_word[1] + addr_word[2] + addr_word[3],
        cols.most_sig_bytes_zero,
        is_real.clone(),
    );

    // Check the correct value of addr_ls_two_bits.
    builder.send_byte(
        ByteOpcode::AND.as_field::<AB::F>(),
        cols.addr_ls_two_bits,
        addr_word[0],
        AB::Expr::from_u8(0b11),
        is_real.clone(),
    );

    // The aligned address is now an expression rather than a witnessed column: the
    // union chip witnessed `addr_aligned` and asserted `addr_aligned +
    // addr_ls_two_bits == addr_word.reduce()`, which is exactly this definition.
    let addr_aligned = addr_word.reduce::<AB>() - cols.addr_ls_two_bits;

    // Trusted: the word moves between memory and a register (see
    // `MemoryAirBuilder::eval_memory_access_trusted`).
    builder.eval_memory_access_trusted(
        cols.shard(),
        crate::frame::clk_from_i_type_frame::<AB>(&cols.frame)
            + AB::Expr::from_u32(MemoryAccessPosition::Memory as u32),
        addr_aligned.clone(),
        memory_access,
        is_real,
    );

    addr_aligned
}

/// The shared instruction plumbing for the memory chips (the Instruction-bus
/// receive is gone: every row is a real instruction serving itself via the
/// frame).
///
/// Every memory chip supplies the same constants: `next_next_pc = next_pc + 4`,
/// `num_extra_cycles = 0`, `is_rw_a = 1`, `is_check_memory = 1`, `is_halt = 0`,
/// `is_sequential = 1`.
pub fn receive_memory_instruction<AB: ZKMCoreAirBuilder>(
    builder: &mut AB,
    cols: &MemoryInstrCommonCols<AB::Var>,
    opcode: AB::Expr,
    op_a_immutable: AB::Expr,
    is_real: AB::Expr,
) {

    // A real instruction carries its own program fetch, register access and
    // `(clk, pc)` chaining.  Memory instructions are sequential, never halt.
    // The plain stores read op_a immutably (the per-chip `op_a_immutable`
    // expr, NOT including SC).
    crate::frame::eval_i_type_frame(
        builder,
        &cols.frame,
        opcode,
        cols.pc.into(),
        cols.next_pc.into(),
        cols.next_pc + AB::Expr::from_u32(4),
        cols.next_pc.into(),
        AB::Expr::ZERO,
        is_real.clone(),
    );
    // The plain stores read op_a immutably: the register write carries the
    // previous value through unchanged.
    builder
        .when(op_a_immutable.clone() * is_real.clone())
        .assert_word_eq(*cols.frame.op_a_access.value(), cols.frame.op_a_access.prev_value);
    // No `op_a_value` binding remains: the chips compute directly on the
    // frame's committed register access (`a_val()`), and the frame itself
    // pins that word to zero on register-0 rows.
    let _ = is_real;
}

/// Constrains that the address is word aligned, for the opcodes that require it.
///
/// The `LW`/`LL`/`SW`/`SC` chips do not witness the three offset flags at all;
/// they simply pin `addr_ls_two_bits` to zero.
pub fn assert_word_aligned<AB: ZKMCoreAirBuilder>(
    builder: &mut AB,
    cols: &MemoryInstrCommonCols<AB::Var>,
    is_real: AB::Expr,
) {
    builder.when(is_real).assert_zero(cols.addr_ls_two_bits);
}

impl<F: PrimeField32> MemoryInstrCommonCols<F> {
    /// Populates the shared columns from a memory-instruction event.
    ///
    /// Returns the two least significant bits of the effective address, which the
    /// per-chip populate functions use to derive their offset flags and values.
    pub fn populate(
        &mut self,
        event: &MemInstrEvent,
        blu: &mut impl ByteRecord,
        program: &zkm_core_executor::Program,
    ) -> u8 {
        // Every memory-instruction row is a real instruction owning its frame.
        self.frame.populate_from_mem(event, program, blu);

        debug_assert!(self.frame.shard != F::ZERO);
        self.pc = F::from_u32(event.pc);
        self.next_pc = F::from_u32(event.next_pc);
        // The memory access is populated per chip (loads carry read-only
        // consistency columns; stores carry read-write ones).

        // Inline effective-address addition (emits the u8 range checks for the
        // resulting address word only — the operands are pre-checked).
        let memory_addr = self.addr_add.populate_check_value_only(blu, event.b, event.c);
        debug_assert_eq!(memory_addr, event.b.wrapping_add(event.c));
        self.addr_word_range_checker.populate(blu, memory_addr);

        let addr_ls_two_bits = (memory_addr % WORD_SIZE as u32) as u8;
        self.addr_ls_two_bits = F::from_u8(addr_ls_two_bits);

        // Add byte lookup event to verify correct calculation of addr_ls_two_bits.
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: addr_ls_two_bits as u16,
            a2: 0,
            b: memory_addr.to_le_bytes()[0],
            c: 0b11,
        });

        let addr_word: Word<F> = memory_addr.into();
        self.most_sig_bytes_zero
            .populate_from_field_element(addr_word[1] + addr_word[2] + addr_word[3]);

        if self.most_sig_bytes_zero.result == F::ONE {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: NUM_REGISTERS as u8 - 1,
                c: memory_addr.to_le_bytes()[0],
            });
        }

        addr_ls_two_bits
    }
}

/// Shared trace-generation driver for the memory chips.
///
/// Identical in structure to the driver the union chip used: one parallel pass
/// over the (already opcode-partitioned) event slice, each worker accumulating
/// its own byte-lookup map.
pub(crate) fn generate_memory_trace<F: PrimeField32>(
    events: &[MemInstrEvent],
    padded_nb_rows: usize,
    num_cols: usize,
    event_to_row: impl Fn(&MemInstrEvent, &mut [F], &mut zkm_core_executor::events::ByteLookupMap) + Sync + Send,
    pad_row: impl Fn(&mut [F]) + Sync + Send,
) -> (RowMajorMatrix<F>, Vec<zkm_core_executor::events::ByteLookupMap>) {
    let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);
    let mut values = zeroed_f_vec(padded_nb_rows * num_cols);

    let blu_events = values
        .chunks_mut(chunk_size * num_cols)
        .enumerate()
        .par_bridge()
        .map(|(i, rows)| {
            let mut blu: zkm_core_executor::events::ByteLookupMap = Default::default();
            rows.chunks_mut(num_cols).enumerate().for_each(|(j, row)| {
                let idx = i * chunk_size + j;
                if idx < events.len() {
                    event_to_row(&events[idx], row, &mut blu);
                } else {
                    // Padding rows carry no instruction: neutralise the frame
                    // or its register-access multiplicities break the Memory
                    // bus.
                    pad_row(row);
                }
            });
            blu
        })
        .collect::<Vec<_>>();

    (RowMajorMatrix::new(values, num_cols), blu_events)
}

/// Constrains the three sub-word offset flags against `addr_ls_two_bits`.
///
/// Returns the `offset_is_zero` expression (`1 - one - two - three`).  Only the
/// chips whose opcodes can address a sub-word offset witness these flags; the
/// word-aligned chips use [`assert_word_aligned`] instead.
pub fn eval_offset_flags<AB: ZKMCoreAirBuilder>(
    builder: &mut AB,
    addr_ls_two_bits: AB::Var,
    ls_bits_is_one: AB::Var,
    ls_bits_is_two: AB::Var,
    ls_bits_is_three: AB::Var,
) -> AB::Expr {
    let offset_is_zero = AB::Expr::ONE - ls_bits_is_one - ls_bits_is_two - ls_bits_is_three;

    builder.assert_bool(ls_bits_is_one);
    builder.assert_bool(ls_bits_is_two);
    builder.assert_bool(ls_bits_is_three);
    builder.assert_bool(offset_is_zero.clone());

    // SAFETY: due to these constraints at most one of the four flags can be non-zero;
    // as their sum is 1, exactly one flag is on with value 1.
    builder.when(offset_is_zero.clone()).assert_zero(addr_ls_two_bits);
    builder.when(ls_bits_is_one).assert_one(addr_ls_two_bits);
    builder.when(ls_bits_is_two).assert_eq(addr_ls_two_bits, AB::Expr::TWO);
    builder.when(ls_bits_is_three).assert_eq(addr_ls_two_bits, AB::Expr::from_u8(3));

    offset_is_zero
}

/// Populates the three offset flags from the low two address bits.
#[inline]
pub fn populate_offset_flags<F: PrimeField32>(
    addr_ls_two_bits: u8,
    ls_bits_is_one: &mut F,
    ls_bits_is_two: &mut F,
    ls_bits_is_three: &mut F,
) {
    *ls_bits_is_one = F::from_bool(addr_ls_two_bits == 1);
    *ls_bits_is_two = F::from_bool(addr_ls_two_bits == 2);
    *ls_bits_is_three = F::from_bool(addr_ls_two_bits == 3);
}
