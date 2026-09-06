//! The per-instruction "frame" that every instruction-bearing chip needs once
//! the `Cpu` dispatch hub is gone.
//!
//! Today `CpuChip` is a hub: it fetches the instruction (`send_program`), reads
//! and writes the registers (`eval_registers`), chains `(clk, pc)` on the
//! `State` bus, and then hands a fully decoded instruction to the opcode chip
//! over the `Instruction` bus.  The opcode chips are pure receivers.  That costs
//! a second full row per executed instruction: `Cpu` is 59 columns wide and has
//! one row for EVERY instruction, on top of the opcode row that also exists.
//!
//! The hub is unnecessary — each instruction chip can carry its own frame, with
//! `Program` / `InstructionFetch` / `InstructionDecode` alongside.  This module
//! is the shared piece that makes that possible: a chip embeds
//! [`InstructionFrameCols`] and calls [`eval_instruction_frame`], after which it
//! no longer needs `receive_instruction` and `Cpu` no longer needs a row for it.
//!
//! Migration note: the columns here duplicate `CpuCols` deliberately.  While
//! both exist the area is WORSE (both rows are present), so the win only lands
//! when the last chip migrates and `Cpu` is dropped from `MipsAir`.

use p3_air::AirBuilder;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use zkm_core_executor::events::SyscallEvent;
use zkm_core_executor::events::{
    AluEvent, BranchEvent, ByteLookupEvent, ByteRecord, CompAluEvent, JumpEvent, MemInstrEvent,
    MemoryAccessPosition, MemoryRecordEnum, MiscEvent, MovCondEvent, OptionMemoryRecordEnumTag,
};
use zkm_core_executor::{ByteOpcode, Program};
use zkm_derive::AlignedBorrow;
use zkm_pcs::Word;

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder, ZKMCoreAirBuilder},
    instruction::InstructionCols,
    memory::{RegisterCols, RegisterReadCols, RegisterReadWriteCols},
};

/// Everything an instruction-bearing chip needs to stand on its own.
///
/// `pc` / `next_pc` are NOT here: most opcode chips already carry them (e.g.
/// `AddSubCols`), so they stay owned by the chip and are passed in.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct InstructionFrameCols<T> {
    /// The shard this instruction executed in.
    pub shard: T,
    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The middle 8 bit limb of clk.
    pub clk_high_limb: T,
    /// The most significant bit of clk, i.e. bit 24.
    ///
    /// A per-shard `clk` runs to `2^25` (`CORE_SHARD_CLK_LIMIT`): the memory-argument ordering
    /// proof needs every timestamp it compares bounded by the same width it range-checks
    /// differences to, and this bit is where that bound comes from on an instruction row.

    /// The decoded instruction, bound to `pc` through the `Program` bus.
    pub instruction: InstructionCols<T>,

    /// Register accesses for the three operands.
    pub op_a_access: RegisterReadWriteCols<T>,
    pub op_b_access: RegisterReadCols<T>,
    pub op_c_access: RegisterReadCols<T>,
}

impl<T: Copy> InstructionFrameCols<T> {
    /// The value of the second operand.
    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }

    /// The value of the third operand.
    pub fn op_c_val(&self) -> Word<T> {
        *self.op_c_access.value()
    }
}

/// The frame's `clk`, reassembled from its three limbs.
///
/// Chips that keep a private `clk` column for a second memory access (Mul/DivRem's HI write, the
/// memory chips' data access, the syscall table send) tie it to the frame with this expression —
/// there is exactly one definition of what `clk` means so the two cannot drift.
pub fn clk_from_frame<AB: AirBuilder>(frame: &InstructionFrameCols<AB::Var>) -> AB::Expr {
    AB::Expr::from_u32(1u32 << 16) * frame.clk_high_limb + frame.clk_16bit_limb
}

/// Evaluate the frame: program fetch, register access, and `(clk, pc)` chaining.
///
/// This is the union of what `CpuChip::eval` does today minus the
/// `send_instruction` dispatch, which disappears entirely once every chip owns
/// its frame.  `is_real` must already be constrained boolean by the caller.
#[allow(clippy::too_many_arguments)]
pub fn eval_instruction_frame<AB>(
    builder: &mut AB,
    frame: &InstructionFrameCols<AB::Var>,
    // The chip's OWN opcode, as an expression over its selector flags.
    opcode: AB::Expr,
    // Exprs, not Vars: the control-flow and memory chips carry `next_pc` /
    // `next_next_pc` as `Word` columns and pass `word.reduce::<AB>()`.
    pc: AB::Expr,
    next_pc: AB::Expr,
    next_next_pc: AB::Expr,
    // What the `State` receive carries.  Equal to `next_pc` everywhere except
    // the syscall chip's halt row, which receives its predecessor's `pc + 4`
    // lookahead while its own `next_pc` is the exit signal 0.
    recv_next_pc: AB::Expr,
    // Extra cycles this instruction adds to `clk` — ZERO except syscalls.
    num_extra_cycles: AB::Expr,
    is_real: AB::Expr,
) where
    AB: ZKMCoreAirBuilder,
{
    let clk = clk_from_frame::<AB>(frame);

    // ★ On a NON-instruction row every frame column is zero, which would leave
    // the op_b / op_c register-access multiplicities below (`ONE - imm_b`)
    // equal to ONE — the chip would RECEIVE register accesses nobody sent and
    // the LogUp multiset would break with "public-values balance failed".
    //
    // Force the immediate flags high there instead of multiplying the
    // multiplicities by `is_real`: that keeps them degree 1, where
    // `is_real * (ONE - imm_b)` would be degree 2 and risks "degree multiple is
    // too high".  This is exactly the trick `CpuChip::eval` already uses for its
    // padding rows.
    let not_real = AB::Expr::ONE - is_real.clone();
    builder.when(not_real.clone()).assert_zero(AB::Expr::ONE - frame.instruction.imm_b);
    builder.when(not_real).assert_zero(AB::Expr::ONE - frame.instruction.imm_c);

    // The instruction at `pc` must be the one the program committed to.
    builder.send_program(pc.clone(), frame.instruction, is_real.clone());

    // ...and the chip's SELECTORS must be that instruction.  The opcode column
    // is bound to the program table by the send above, but nothing ties the
    // selectors to it: without this a row could satisfy every constraint while
    // computing a different operation than the program holds at `pc`, and a
    // chip could claim a `pc` belonging to another chip entirely.  The caller
    // passes the opcode as an expression over its own selectors, so the two
    // cannot disagree.
    builder.when(is_real.clone()).assert_eq(frame.instruction.opcode, opcode);

    // Shard fits in 16 bits; clk decomposes into a 16-bit and an 8-bit limb.
    // Mirrors `CpuChip::eval_shard_clk` — the trace side must add the matching
    // U16Range/U8Range byte events for every instruction row.
    builder.send_byte(
        AB::Expr::from_u8(ByteOpcode::U16Range as u8),
        frame.shard,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        is_real.clone(),
    );
    // `clk` is BUILT from these limbs above, so the reconstruction identity is free and only
    // the limb bounds have to be paid: 16 + 8 bits from the byte table, and the top bit
    // constrained boolean.  The boolean assertion is unguarded — the column is zero on every
    // padding / dependency row — which keeps it degree 2.
    builder.send_timestamp_range_checks(frame.clk_16bit_limb, frame.clk_high_limb, is_real.clone());

    // Immediates bypass the register read.
    builder.when(frame.instruction.imm_b).assert_word_eq(frame.op_b_val(), frame.instruction.op_b);
    builder.when(frame.instruction.imm_c).assert_word_eq(frame.op_c_val(), frame.instruction.op_c);

    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::B as u32),
        frame.instruction.op_b[0],
        &frame.op_b_access,
        AB::Expr::ONE - frame.instruction.imm_b,
    );
    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::C as u32),
        frame.instruction.op_c[0],
        &frame.op_c_access,
        AB::Expr::ONE - frame.instruction.imm_c,
    );

    // Writes to register 0 are discarded.
    builder.when(frame.instruction.op_a_0).assert_word_zero(*frame.op_a_access.value());
    // The immutable-read rule (`op_a_access.value == prev_value`) lives in the
    // chips that read op_a immutably (Branch, TEQ, the plain stores) — NOT
    // here with a ZERO guard: a zero-guarded constraint still consumes an RLC
    // slot on the verifier while the device bytecode optimizer may elide it.

    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::A as u32),
        frame.instruction.op_a,
        &frame.op_a_access,
        is_real.clone(),
    );

    // Always range check the word written to `op_a` — mirrors
    // `CpuChip::eval_registers` (JUMP instructions may witness an invalid word).
    builder.slice_range_check_u8(&frame.op_a_access.access.value.0, is_real.clone());

    // `(clk, pc)` chaining.  The LogUp multiset balance forces row i+1's
    // `(pc, next_pc)` to equal row i's `(next_pc, next_next_pc)`; the boundary
    // endpoints are emitted by the public-values AIR.
    builder.receive_state(frame.shard, clk.clone(), pc, recv_next_pc, is_real.clone());
    builder.send_state(
        frame.shard,
        clk + AB::Expr::from_u32(5) + num_extra_cycles,
        next_pc,
        next_next_pc,
        is_real,
    );
}

impl<F: PrimeField32> InstructionFrameCols<F> {
    /// Populate the frame for a REAL instruction from an ALU event, emitting
    /// the byte events its constraints request (shard/clk range checks, the
    /// op_a word range check, and the register-consistency lookups).
    ///
    /// The instruction is fetched from the program by pc, exactly as
    /// `cpu/trace.rs` does — it is not carried in the event.
    pub fn populate_from_alu(
        &mut self,
        event: &AluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// The shared population body — see [`Self::populate_from_alu`] for the
    /// contract.  Split out because `AluEvent` and `CompAluEvent` carry the
    /// same frame fields but are distinct FFI types.
    #[allow(clippy::too_many_arguments)]
    fn populate_raw(
        &mut self,
        clk: u32,
        pc: u32,
        recv_next_pc: u32,
        a: u32,
        b: u32,
        c: u32,
        a_record: zkm_core_executor::events::OptionMemoryRecordEnum,
        b_record: zkm_core_executor::events::OptionMemoryReadRecord,
        c_record: zkm_core_executor::events::OptionMemoryReadRecord,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.shard = F::from_u32(shard);
        let clk_16 = (clk & 0xffff) as u16;
        let clk_high = ((clk >> 16) & 0x1ff) as u16;
        self.clk_16bit_limb = F::from_u16(clk_16);
        self.clk_high_limb = F::from_u16(clk_high);
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            shard as u16,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::Range, clk_high, 0, 9, 0));

        self.instruction.populate(&program.fetch(pc));
        let _ = recv_next_pc;

        *self.op_a_access.value_mut() = a.into();
        *self.op_b_access.value_mut() = b.into();
        *self.op_c_access.value_mut() = c.into();
        if !matches!(a_record.tag, OptionMemoryRecordEnumTag::None) {
            self.op_a_access.populate_register(a_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = b_record.tag {
            self.op_b_access.populate_register(b_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = c_record.tag {
            self.op_c_access.populate_register(c_record, blu);
        }

        // The op_a word range check reads back the COLUMN value, exactly as
        // `cpu/trace.rs` does — NOT `a`.  `populate` above overwrites
        // `access.value` with the RECORD's value, and the two differ on a
        // no-link jump (op_a = r0: record value 0, while `a` carries the link).
        // Supplying `a`'s bytes there leaves the AIR's request for (0, 0)
        // unmatched and breaks the Byte bus by exactly 2 events per such row.
        let a_bytes = self
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });
    }

    /// `CompAluEvent` variant of [`Self::populate_from_alu`] — the frame
    /// fields carry the same names, so the body is delegated through the
    /// shared raw form.
    pub fn populate_from_comp_alu(
        &mut self,
        event: &CompAluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `BranchEvent` variant of [`Self::populate_from_alu`].  The caller must
    /// additionally set `op_a_immutable = ONE`: a branch READS `op_a`, and both
    /// the frame rule and Cpu's legacy bus tuple carry that flag high.
    pub fn populate_from_branch(
        &mut self,
        event: &BranchEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `JumpEvent` variant of [`Self::populate_from_alu`].  Unlike a branch, a
    /// jump WRITES `op_a` (the link register), so `op_a_immutable` stays 0.  On
    /// a NO-link jump `op_a` is register 0: the record value is 0 while
    /// `event.a` still carries the would-be link — which is why the op_a range
    /// check inside `populate_raw` reads back the COLUMN, not the event.
    pub fn populate_from_jump(
        &mut self,
        event: &JumpEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `MovCondEvent` variant of [`Self::populate_from_alu`].  MNE/MEQ are
    /// `is_rw_a` instructions: op_a keeps its previous value when the condition
    /// fails, so the frame carries `hi_or_prev_a = prev_a` and the caller sets
    /// `is_rw_a` for them (WSBH is a plain write).
    pub fn populate_from_movcond(
        &mut self,
        event: &MovCondEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `MiscEvent` variant of [`Self::populate_from_alu`].
    /// MADDU/MSUBU/MADD/MSUB/INS read-and-write op_a (`is_rw_a`); TEQ reads
    /// op_a immutably.  The chip ties both flags in its AIR.
    pub fn populate_from_misc(
        &mut self,
        event: &MiscEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `SyscallEvent` variant of [`Self::populate_from_alu`].  A syscall
    /// reads-and-writes op_a (the id comes in, the result goes out), so
    /// `is_rw_a = 1` and `hi_or_prev_a` is the op_a PREVIOUS value — read back
    /// from the populated access, since the event carries the write record.
    /// `num_extra_cycles` is byte 3 of the incoming syscall code, exactly as
    /// `cpu/trace.rs` derived it.
    pub fn populate_from_syscall(
        &mut self,
        event: &SyscallEvent,
        program: &Program,
        blu: &mut impl ByteRecord,
    ) {
        let a_record: zkm_core_executor::events::OptionMemoryRecordEnum = if event.a_record_is_real
        {
            Some(MemoryRecordEnum::Write(event.a_record)).into()
        } else {
            None.into()
        };
        self.populate_raw(
            event.clk,
            event.pc,
            event.recv_next_pc,
            event.a_record.value,
            event.arg1,
            event.arg2,
            a_record,
            event.b_record,
            event.c_record,
            program,
            event.shard,
            blu,
        );
    }

    /// Neutralise the frame on a row that carries no instruction — dependency
    /// rows AND padding rows.  The not-real rule forces the immediate flags
    /// high so the op_b / op_c register-access multiplicities (`ONE - imm_b`)
    /// vanish; forgetting this on either row kind breaks the Memory bus.
    pub fn populate_dependency(&mut self) {
        self.instruction.imm_b = F::ONE;
        self.instruction.imm_c = F::ONE;
    }
}

/// The frame for a chip whose every instruction is I-type: `op_b` is a
/// register and `op_c` is an immediate.
///
/// [`InstructionFrameCols`] is 42 columns because it can serve any operand
/// shape: it carries a register access for `op_c` and a full four-limb word for
/// `op_b`, plus the two `imm_*` flags that select between them.  A chip that
/// only ever executes I-type instructions knows all of that statically, so it
/// pays for none of it:
///
/// * `op_c_access` (7 columns) is gone — `op_c` is never a register read, so
///   its access multiplicity was identically zero.
/// * `imm_b` / `imm_c` (2 columns) are gone — they are the constants 0 and 1.
/// * `op_b` narrows from a `Word` to a single column (3 saved): it is a
///   register index.  Nothing here has to *assert* that the upper three limbs
///   are zero — the `Program` bus binds all four limbs against the preprocessed
///   program table, so an index that did not fit a byte would fail to match.
///
/// 42 → 30 columns on every row of the chip.  Which chips qualify is a property
/// of the opcodes they receive, not of the AIR: a chip that sees more than one
/// operand shape (`AddSub` executes both `ADD` and `ADDI`) must keep the
/// universal frame or split.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct ITypeFrameCols<T> {
    /// The shard this instruction executed in.
    pub shard: T,
    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The middle 8 bit limb of clk.
    pub clk_high_limb: T,
    /// The most significant bit of clk, i.e. bit 24.  See

    /// The opcode for this cycle.
    pub opcode: T,
    /// The first operand — a register index.
    pub op_a: T,
    /// The second operand — a register INDEX, not a word.
    pub op_b: T,
    /// The third operand — the immediate itself.
    pub op_c: Word<T>,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,

    /// Register accesses for the two register operands.
    pub op_a_access: RegisterReadWriteCols<T>,
    pub op_b_access: RegisterReadCols<T>,
}

impl<T: Copy> ITypeFrameCols<T> {
    /// The value of the second operand — the register read.
    #[inline]
    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }

    /// The value of the third operand.  An immediate IS its value: there is no
    /// register access to consult, which is exactly the column this frame saves.
    #[inline]
    pub fn op_c_val(&self) -> Word<T> {
        self.op_c
    }
}

/// The frame's `clk`, reassembled from its three limbs — see
/// [`clk_from_frame`], which this must agree with exactly.
pub fn clk_from_i_type_frame<AB: AirBuilder>(frame: &ITypeFrameCols<AB::Var>) -> AB::Expr {
    AB::Expr::from_u32(1u32 << 16) * frame.clk_high_limb + frame.clk_16bit_limb
}

/// Rebuild the universal `Program`-bus tuple from the narrow columns.
///
/// The preprocessed program table is shared by every chip, so the tuple must
/// keep its 13 slots — but only 7 of them have to be *columns*.  The rest are
/// the constants the I-type shape implies.
fn i_type_instruction<AB: AirBuilder>(
    frame: &ITypeFrameCols<AB::Var>,
) -> InstructionCols<AB::Expr> {
    InstructionCols {
        opcode: frame.opcode.into(),
        op_a: frame.op_a.into(),
        op_b: Word([frame.op_b.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO]),
        op_c: frame.op_c.map(Into::into),
        op_a_0: frame.op_a_0.into(),
        imm_b: AB::Expr::ZERO,
        imm_c: AB::Expr::ONE,
    }
}

/// Evaluate an I-type frame.  Constrains exactly what
/// [`eval_instruction_frame`] does for an I-type row — the two functions must
/// be read together, and any rule added to one belongs in the other.
///
/// Note what is *absent*: the universal frame forces `imm_b` / `imm_c` high on
/// a non-instruction row so the `ONE - imm_b` register-access multiplicities
/// vanish there.  Here the multiplicities are `is_real` directly (the same
/// degree, since `imm_b` was a column), so a padding row needs no neutralising
/// at all and the chips' `populate_dependency` calls disappear with it.
#[allow(clippy::too_many_arguments)]
pub fn eval_i_type_frame<AB>(
    builder: &mut AB,
    frame: &ITypeFrameCols<AB::Var>,
    // The chip's OWN opcode -- see [`eval_instruction_frame`].
    opcode: AB::Expr,
    pc: AB::Expr,
    next_pc: AB::Expr,
    next_next_pc: AB::Expr,
    recv_next_pc: AB::Expr,
    num_extra_cycles: AB::Expr,
    is_real: AB::Expr,
) where
    AB: ZKMCoreAirBuilder,
{
    let clk = clk_from_i_type_frame::<AB>(frame);

    // The instruction at `pc` must be the one the program committed to.
    builder.send_program(pc.clone(), i_type_instruction::<AB>(frame), is_real.clone());
    builder.when(is_real.clone()).assert_eq(frame.opcode, opcode);

    // Shard fits in 16 bits; clk decomposes into a 16-bit and an 8-bit limb.
    builder.send_byte(
        AB::Expr::from_u8(ByteOpcode::U16Range as u8),
        frame.shard,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        is_real.clone(),
    );
    builder.send_timestamp_range_checks(frame.clk_16bit_limb, frame.clk_high_limb, is_real.clone());

    // `op_b` is read from the register file; `op_c` is the immediate and needs
    // no access at all.
    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::B as u32),
        frame.op_b,
        &frame.op_b_access,
        is_real.clone(),
    );

    // Writes to register 0 are discarded.
    builder.when(frame.op_a_0).assert_word_zero(*frame.op_a_access.value());

    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::A as u32),
        frame.op_a,
        &frame.op_a_access,
        is_real.clone(),
    );

    builder.slice_range_check_u8(&frame.op_a_access.access.value.0, is_real.clone());

    // `(clk, pc)` chaining.
    builder.receive_state(frame.shard, clk.clone(), pc, recv_next_pc, is_real.clone());
    builder.send_state(
        frame.shard,
        clk + AB::Expr::from_u32(5) + num_extra_cycles,
        next_pc,
        next_next_pc,
        is_real,
    );
}

impl<F: PrimeField32> ITypeFrameCols<F> {
    /// Populate the frame for a memory instruction.  Mirrors
    /// [`InstructionFrameCols::populate_raw`] with the I-type columns dropped —
    /// notably there is no `op_c` register access to populate, which also means
    /// no byte events for one.
    pub fn populate_from_mem(
        &mut self,
        event: &MemInstrEvent,
        program: &Program,
        blu: &mut impl ByteRecord,
    ) {
        let shard = event.shard;
        self.shard = F::from_u32(shard);
        let clk_16 = (event.clk & 0xffff) as u16;
        let clk_high = ((event.clk >> 16) & 0x1ff) as u16;
        self.clk_16bit_limb = F::from_u16(clk_16);
        self.clk_high_limb = F::from_u16(clk_high);
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            shard as u16,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::Range, clk_high, 0, 9, 0));

        let instruction = program.fetch(event.pc);
        // The shape this frame is specialised for.  A chip that ever violates
        // this would silently commit an instruction the `Program` bus cannot
        // match, so it is worth asserting where the assumption is made.
        debug_assert!(
            !instruction.imm_b && instruction.imm_c,
            "an I-type frame received a non-I-type instruction: {:?}",
            instruction.opcode
        );
        debug_assert!(instruction.op_b < 256, "op_b is not a register index");
        // Dropping `op_c_access` also drops the byte events its `populate`
        // emitted.  That is only sound because an immediate `op_c` never
        // produces a register read to record — the AIR gave the access
        // multiplicity `ONE - imm_c = 0`, so any event here was already
        // unmatched on the byte bus.  `MemInstrEvent` no longer carries a
        // `c_record` at all; `Executor::emit_mem_instr_event` asserts it.
        self.opcode = instruction.opcode.as_field::<F>();
        self.op_a = F::from_u32(instruction.op_a as u32);
        self.op_b = F::from_u32(instruction.op_b);
        self.op_c = instruction.op_c.into();
        self.op_a_0 = F::from_bool(instruction.op_a == 0);

        *self.op_a_access.value_mut() = event.a.into();
        *self.op_b_access.value_mut() = event.b.into();
        if !matches!(event.a_record.tag, OptionMemoryRecordEnumTag::None) {
            self.op_a_access.populate_register(event.a_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = event.b_record.tag {
            self.op_b_access.populate_register(event.b_record, blu);
        }

        // Read the op_a range check back off the COLUMN, not the event — see
        // [`InstructionFrameCols::populate_raw`] for the no-link-jump case that
        // makes the two differ.
        let a_bytes = self
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });
    }

    /// Populate the frame for an immediate-form ALU instruction.  Mirrors
    /// [`Self::populate_from_mem`], but an `AluEvent` carries no shard of its
    /// own, so the shard arrives as a parameter exactly as it does in
    /// [`RTypeFrameCols::populate_from_alu`].
    pub fn populate_from_alu(
        &mut self,
        event: &AluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        debug_assert!(
            matches!(event.c_record.tag, OptionMemoryRecordEnumTag::None),
            "an I-type frame received a register read for op_c"
        );
        self.populate_i_raw(
            event.clk,
            event.pc,
            event.a,
            event.b,
            event.a_record,
            event.b_record,
            program,
            shard,
            blu,
        );
    }

    /// `BranchEvent` variant — every branch is I-type once the zero-compare
    /// decodes read register 0 for their comparand.
    pub fn populate_from_branch(
        &mut self,
        event: &BranchEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        debug_assert!(
            matches!(event.c_record.tag, OptionMemoryRecordEnumTag::None),
            "an I-type frame received a register read for op_c"
        );
        self.populate_i_raw(
            event.clk,
            event.pc,
            event.a,
            event.b,
            event.a_record,
            event.b_record,
            program,
            shard,
            blu,
        );
    }

    /// The shared population body for the AluEvent-shaped events — see
    /// [`Self::populate_from_mem`] for the contract.
    #[allow(clippy::too_many_arguments)]
    fn populate_i_raw(
        &mut self,
        clk: u32,
        pc: u32,
        a: u32,
        b: u32,
        a_record: zkm_core_executor::events::OptionMemoryRecordEnum,
        b_record: zkm_core_executor::events::OptionMemoryReadRecord,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.shard = F::from_u32(shard);
        let clk_16 = (clk & 0xffff) as u16;
        let clk_high = ((clk >> 16) & 0x1ff) as u16;
        self.clk_16bit_limb = F::from_u16(clk_16);
        self.clk_high_limb = F::from_u16(clk_high);
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            shard as u16,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::Range, clk_high, 0, 9, 0));

        let instruction = program.fetch(pc);
        // The shape this frame is specialised for — see
        // [`Self::populate_from_mem`] for why this is asserted here.
        debug_assert!(
            !instruction.imm_b && instruction.imm_c,
            "an I-type frame received a non-I-type instruction: {:?}",
            instruction.opcode
        );
        debug_assert!(instruction.op_b < 256, "op_b is not a register index");
        self.opcode = instruction.opcode.as_field::<F>();
        self.op_a = F::from_u32(instruction.op_a as u32);
        self.op_b = F::from_u32(instruction.op_b);
        self.op_c = instruction.op_c.into();
        self.op_a_0 = F::from_bool(instruction.op_a == 0);

        *self.op_a_access.value_mut() = a.into();
        *self.op_b_access.value_mut() = b.into();
        if !matches!(a_record.tag, OptionMemoryRecordEnumTag::None) {
            self.op_a_access.populate_register(a_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = b_record.tag {
            self.op_b_access.populate_register(b_record, blu);
        }

        // Column-read-back for the op_a range check, as in
        // [`Self::populate_from_mem`].
        let a_bytes = self
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });
    }
}

/// The frame for a chip whose every instruction is R-type: `op_b` and `op_c`
/// are both registers, and there are no immediates at all.  Relative to
/// [`InstructionFrameCols`] this drops the two `Word` operand carriers to bare
/// register indices and both `imm_*` flags — 8 columns per row.
///
/// The register-form half of a split chip (`AddSub` splits into a register
/// half on this frame and an immediate half on [`ITypeFrameCols`]) uses this;
/// see the shape note above `ITypeFrameCols` for why a chip that mixes operand
/// shapes cannot.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct RTypeFrameCols<T> {
    /// The shard this instruction executed in.
    pub shard: T,
    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The middle 8 bit limb of clk.
    pub clk_high_limb: T,
    /// The most significant bit of clk, i.e. bit 24.  See

    /// The opcode for this cycle.
    pub opcode: T,
    /// The first operand — a register index.
    pub op_a: T,
    /// The second operand — a register index.
    pub op_b: T,
    /// The third operand — a register index.
    pub op_c: T,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,

    /// Register accesses for the three register operands.
    pub op_a_access: RegisterReadWriteCols<T>,
    pub op_b_access: RegisterReadCols<T>,
    pub op_c_access: RegisterReadCols<T>,
}

impl<T: Copy> RTypeFrameCols<T> {
    /// The value of the second operand.
    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }

    /// The value of the third operand.
    pub fn op_c_val(&self) -> Word<T> {
        *self.op_c_access.value()
    }
}

/// The frame's `clk` — see [`clk_from_frame`].
pub fn clk_from_r_type_frame<AB: AirBuilder>(frame: &RTypeFrameCols<AB::Var>) -> AB::Expr {
    AB::Expr::from_u32(1u32 << 16) * frame.clk_high_limb + frame.clk_16bit_limb
}

/// Rebuild the universal `Program`-bus tuple from the narrow columns — the
/// R-type constants are `imm_b = imm_c = 0` and both operand words carry a
/// bare register index in their low limb.
fn r_type_instruction<AB: AirBuilder>(
    frame: &RTypeFrameCols<AB::Var>,
) -> InstructionCols<AB::Expr> {
    InstructionCols {
        opcode: frame.opcode.into(),
        op_a: frame.op_a.into(),
        op_b: Word([frame.op_b.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO]),
        op_c: Word([frame.op_c.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO]),
        op_a_0: frame.op_a_0.into(),
        imm_b: AB::Expr::ZERO,
        imm_c: AB::Expr::ZERO,
    }
}

/// Evaluate an R-type frame.  Constrains exactly what
/// [`eval_instruction_frame`] does for a register-form row — the two must be
/// read together, and any rule added to one belongs in the other.  As with
/// [`eval_i_type_frame`], the register-access multiplicities are `is_real`
/// directly, so a padding row needs no neutralising.
#[allow(clippy::too_many_arguments)]
pub fn eval_r_type_frame<AB>(
    builder: &mut AB,
    frame: &RTypeFrameCols<AB::Var>,
    // The chip's OWN opcode -- see [`eval_instruction_frame`].
    opcode: AB::Expr,
    pc: AB::Expr,
    next_pc: AB::Expr,
    next_next_pc: AB::Expr,
    recv_next_pc: AB::Expr,
    num_extra_cycles: AB::Expr,
    is_real: AB::Expr,
) where
    AB: ZKMCoreAirBuilder,
{
    let clk = clk_from_r_type_frame::<AB>(frame);

    // The instruction at `pc` must be the one the program committed to.
    builder.send_program(pc.clone(), r_type_instruction::<AB>(frame), is_real.clone());
    builder.when(is_real.clone()).assert_eq(frame.opcode, opcode);

    // Shard fits in 16 bits; clk decomposes into a 16-bit and an 8-bit limb.
    builder.send_byte(
        AB::Expr::from_u8(ByteOpcode::U16Range as u8),
        frame.shard,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        is_real.clone(),
    );
    builder.send_timestamp_range_checks(frame.clk_16bit_limb, frame.clk_high_limb, is_real.clone());

    // Both source operands are read from the register file.
    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::B as u32),
        frame.op_b,
        &frame.op_b_access,
        is_real.clone(),
    );
    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::C as u32),
        frame.op_c,
        &frame.op_c_access,
        is_real.clone(),
    );

    // Writes to register 0 are discarded.
    builder.when(frame.op_a_0).assert_word_zero(*frame.op_a_access.value());

    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::A as u32),
        frame.op_a,
        &frame.op_a_access,
        is_real.clone(),
    );
    builder.slice_range_check_u8(&frame.op_a_access.access.value.0, is_real.clone());

    // `(clk, pc)` chaining.
    builder.receive_state(frame.shard, clk.clone(), pc, recv_next_pc, is_real.clone());
    builder.send_state(
        frame.shard,
        clk + AB::Expr::from_u32(5) + num_extra_cycles,
        next_pc,
        next_next_pc,
        is_real,
    );
}

impl<F: PrimeField32> RTypeFrameCols<F> {
    /// Populate the frame for a register-form ALU instruction.  Mirrors
    /// [`InstructionFrameCols::populate_raw`] with the R-type columns dropped.
    pub fn populate_from_alu(
        &mut self,
        event: &AluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_r_raw(
            event.clk,
            event.pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `CompAluEvent` variant of [`Self::populate_from_alu`] — the two event
    /// types carry identically named frame fields, exactly as with the
    /// universal frame's pair.
    pub fn populate_from_comp_alu(
        &mut self,
        event: &CompAluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.populate_r_raw(
            event.clk,
            event.pc,
            event.a,
            event.b,
            event.c,
            event.a_record,
            event.b_record,
            event.c_record,
            program,
            shard,
            blu,
        );
    }

    /// `SyscallEvent` variant — SYSCALL is register-form (`$v0`, `$a0`, `$a1`
    /// are all register indices); the id comes in through `op_a`, the result
    /// goes out.  Mirrors [`InstructionFrameCols::populate_from_syscall`].
    pub fn populate_from_syscall(
        &mut self,
        event: &SyscallEvent,
        program: &Program,
        blu: &mut impl ByteRecord,
    ) {
        let a_record: zkm_core_executor::events::OptionMemoryRecordEnum = if event.a_record_is_real
        {
            Some(MemoryRecordEnum::Write(event.a_record)).into()
        } else {
            None.into()
        };
        self.populate_r_raw(
            event.clk,
            event.pc,
            event.a_record.value,
            event.arg1,
            event.arg2,
            a_record,
            event.b_record,
            event.c_record,
            program,
            event.shard,
            blu,
        );
    }

    /// The shared population body — see [`Self::populate_from_alu`] for the
    /// contract.
    #[allow(clippy::too_many_arguments)]
    fn populate_r_raw(
        &mut self,
        clk: u32,
        pc: u32,
        a: u32,
        b: u32,
        c: u32,
        a_record: zkm_core_executor::events::OptionMemoryRecordEnum,
        b_record: zkm_core_executor::events::OptionMemoryReadRecord,
        c_record: zkm_core_executor::events::OptionMemoryReadRecord,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.shard = F::from_u32(shard);
        let clk_16 = (clk & 0xffff) as u16;
        let clk_high = ((clk >> 16) & 0x1ff) as u16;
        self.clk_16bit_limb = F::from_u16(clk_16);
        self.clk_high_limb = F::from_u16(clk_high);
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            shard as u16,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::Range, clk_high, 0, 9, 0));

        let instruction = program.fetch(pc);
        // The shape this frame is specialised for — see
        // [`ITypeFrameCols::populate_from_mem`] for why this is asserted here.
        debug_assert!(
            !instruction.imm_b && !instruction.imm_c,
            "an R-type frame received a non-R-type instruction: {:?}",
            instruction.opcode
        );
        debug_assert!(instruction.op_b < 256, "op_b is not a register index");
        debug_assert!(instruction.op_c < 256, "op_c is not a register index");
        self.opcode = instruction.opcode.as_field::<F>();
        self.op_a = F::from_u32(instruction.op_a as u32);
        self.op_b = F::from_u32(instruction.op_b);
        self.op_c = F::from_u32(instruction.op_c);
        self.op_a_0 = F::from_bool(instruction.op_a == 0);

        *self.op_a_access.value_mut() = a.into();
        *self.op_b_access.value_mut() = b.into();
        *self.op_c_access.value_mut() = c.into();
        if !matches!(a_record.tag, OptionMemoryRecordEnumTag::None) {
            self.op_a_access.populate_register(a_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = b_record.tag {
            self.op_b_access.populate_register(b_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = c_record.tag {
            self.op_c_access.populate_register(c_record, blu);
        }

        // Read the op_a range check back off the COLUMN, not the event — see
        // [`InstructionFrameCols::populate_raw`] for the no-link-jump case
        // that makes the two differ.
        let a_bytes = self
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });
    }
}

/// The frame for a chip whose every instruction is a SHAMT form: `op_b` a
/// register and `op_c` a 5-bit shift amount.  Relative to [`ITypeFrameCols`]
/// the immediate needs only ONE column — the `Program` bus binds it to the
/// decoded `op_c` and pins the upper limbs of the bus tuple to zero, so the
/// scalar is exact with no extra range check.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct ShamtFrameCols<T> {
    /// The shard this instruction executed in.
    pub shard: T,
    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The middle 8 bit limb of clk.
    pub clk_high_limb: T,
    /// The most significant bit of clk, i.e. bit 24.  See

    /// The opcode for this cycle.
    pub opcode: T,
    /// The first operand — a register index.
    pub op_a: T,
    /// The second operand — a register INDEX, not a word.
    pub op_b: T,
    /// The third operand — the shift amount itself, as a bare scalar.
    pub op_c: T,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,

    /// Register accesses for the two register operands.
    pub op_a_access: RegisterReadWriteCols<T>,
    pub op_b_access: RegisterReadCols<T>,
}

impl<T: Copy> ShamtFrameCols<T> {
    /// The value of the second operand — the register read.
    #[inline]
    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }
}

/// The frame's `clk` — see [`clk_from_frame`].
pub fn clk_from_shamt_frame<AB: AirBuilder>(frame: &ShamtFrameCols<AB::Var>) -> AB::Expr {
    AB::Expr::from_u32(1u32 << 16) * frame.clk_high_limb + frame.clk_16bit_limb
}

/// Rebuild the universal `Program`-bus tuple from the narrow columns — the
/// shamt constants are `imm_b = 0`, `imm_c = 1`, and the immediate's upper
/// limbs zero (a shamt is 5 bits, so the decoded `op_c` lives entirely in the
/// low limb; the bus equality is what makes the scalar exact).
fn shamt_instruction<AB: AirBuilder>(frame: &ShamtFrameCols<AB::Var>) -> InstructionCols<AB::Expr> {
    InstructionCols {
        opcode: frame.opcode.into(),
        op_a: frame.op_a.into(),
        op_b: Word([frame.op_b.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO]),
        op_c: Word([frame.op_c.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO]),
        op_a_0: frame.op_a_0.into(),
        imm_b: AB::Expr::ZERO,
        imm_c: AB::Expr::ONE,
    }
}

/// Evaluate a shamt frame.  Constrains exactly what [`eval_i_type_frame`]
/// does — the two must be read together, and any rule added to one belongs in
/// the other.  Register-access multiplicities are `is_real` directly, so a
/// padding row needs no neutralising.
#[allow(clippy::too_many_arguments)]
pub fn eval_shamt_frame<AB>(
    builder: &mut AB,
    frame: &ShamtFrameCols<AB::Var>,
    // The chip's OWN opcode -- see [`eval_instruction_frame`].
    opcode: AB::Expr,
    pc: AB::Expr,
    next_pc: AB::Expr,
    next_next_pc: AB::Expr,
    recv_next_pc: AB::Expr,
    num_extra_cycles: AB::Expr,
    is_real: AB::Expr,
) where
    AB: ZKMCoreAirBuilder,
{
    let clk = clk_from_shamt_frame::<AB>(frame);

    // The instruction at `pc` must be the one the program committed to.
    builder.send_program(pc.clone(), shamt_instruction::<AB>(frame), is_real.clone());
    builder.when(is_real.clone()).assert_eq(frame.opcode, opcode);

    // Shard fits in 16 bits; clk decomposes into a 16-bit and an 8-bit limb.
    builder.send_byte(
        AB::Expr::from_u8(ByteOpcode::U16Range as u8),
        frame.shard,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        is_real.clone(),
    );
    builder.send_timestamp_range_checks(frame.clk_16bit_limb, frame.clk_high_limb, is_real.clone());

    // `op_b` is read from the register file; the shamt needs no access.
    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::B as u32),
        frame.op_b,
        &frame.op_b_access,
        is_real.clone(),
    );

    // Writes to register 0 are discarded.
    builder.when(frame.op_a_0).assert_word_zero(*frame.op_a_access.value());

    builder.eval_register_access(
        frame.shard,
        clk.clone() + AB::F::from_u32(MemoryAccessPosition::A as u32),
        frame.op_a,
        &frame.op_a_access,
        is_real.clone(),
    );

    builder.slice_range_check_u8(&frame.op_a_access.access.value.0, is_real.clone());

    // `(clk, pc)` chaining.
    builder.receive_state(frame.shard, clk.clone(), pc, recv_next_pc, is_real.clone());
    builder.send_state(
        frame.shard,
        clk + AB::Expr::from_u32(5) + num_extra_cycles,
        next_pc,
        next_next_pc,
        is_real,
    );
}

impl<F: PrimeField32> ShamtFrameCols<F> {
    /// Populate the frame for a shamt-form shift.  Mirrors
    /// [`ITypeFrameCols::populate_from_alu`] with the immediate collapsed to
    /// its low limb.
    pub fn populate_from_alu(
        &mut self,
        event: &AluEvent,
        program: &Program,
        shard: u32,
        blu: &mut impl ByteRecord,
    ) {
        self.shard = F::from_u32(shard);
        let clk_16 = (event.clk & 0xffff) as u16;
        let clk_high = ((event.clk >> 16) & 0x1ff) as u16;
        self.clk_16bit_limb = F::from_u16(clk_16);
        self.clk_high_limb = F::from_u16(clk_high);
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            shard as u16,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::Range, clk_high, 0, 9, 0));

        let instruction = program.fetch(event.pc);
        // The shape this frame is specialised for — see
        // [`ITypeFrameCols::populate_from_mem`] for why this is asserted here.
        debug_assert!(
            !instruction.imm_b && instruction.imm_c,
            "a shamt frame received a non-immediate instruction: {:?}",
            instruction.opcode
        );
        debug_assert!(instruction.op_b < 256, "op_b is not a register index");
        debug_assert!(instruction.op_c < 32, "op_c is not a 5-bit shift amount");
        debug_assert!(
            matches!(event.c_record.tag, OptionMemoryRecordEnumTag::None),
            "a shamt frame received a register read for op_c"
        );
        self.opcode = instruction.opcode.as_field::<F>();
        self.op_a = F::from_u32(instruction.op_a as u32);
        self.op_b = F::from_u32(instruction.op_b);
        self.op_c = F::from_u32(instruction.op_c);
        self.op_a_0 = F::from_bool(instruction.op_a == 0);

        *self.op_a_access.value_mut() = event.a.into();
        *self.op_b_access.value_mut() = event.b.into();
        if !matches!(event.a_record.tag, OptionMemoryRecordEnumTag::None) {
            self.op_a_access.populate_register(event.a_record, blu);
        }
        if let OptionMemoryRecordEnumTag::Read = event.b_record.tag {
            self.op_b_access.populate_register(event.b_record, blu);
        }

        // Column-read-back for the op_a range check, as in
        // [`ITypeFrameCols::populate_from_mem`].
        let a_bytes = self
            .op_a_access
            .access
            .value
            .0
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[0] as u8,
            c: a_bytes[1] as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: a_bytes[2] as u8,
            c: a_bytes[3] as u8,
        });
    }
}
