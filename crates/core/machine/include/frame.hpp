#pragma once

#include "instruction.hpp"
#include "memory.hpp"
#include "prelude.hpp"
#include "utils.hpp"

// Mirrors `InstructionFrameCols`'s populate helpers
// (crates/core/machine/src/frame/mod.rs) MINUS the byte-lookup emission: GPU
// tracegen only fills columns, the byte lookups come from the host
// `generate_dependencies` pass.  FFI parity with the Rust side is required or
// the GPU trace diverges from the host and the frame constraints fail.
namespace zkm_core_machine_sys::frame {

// The shared population body — `InstructionFrameCols::populate_raw`.
// `instruction` must be the program's instruction at the event's pc; the
// caller only invokes this on a REAL instruction row (`is_instruction != 0`).
template<class F>
__ZKM_HOSTDEV__ void populate_raw(
    InstructionFrameCols<F>& frame,
    const uint32_t clk,
    const uint32_t recv_next_pc,
    const uint32_t a,
    const uint32_t b,
    const uint32_t c,
    const OptionMemoryRecordEnum& a_record,
    const OptionMemoryReadRecord& b_record,
    const OptionMemoryReadRecord& c_record,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    frame.shard = F::from_canonical_u32(shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((clk >> 16) & 0x3ff));

    cpu::populate_instruction<F>(frame.instruction, instruction);
    (void)recv_next_pc;

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, a);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, b);
    write_word_from_u32_v2<F>(frame.op_c_access.access.value, c);

    // `populate_register_read_write` overwrites `access.value` with the
    // RECORD's value — the two differ on a no-link jump (op_a = r0), and the
    // AIR range-checks the COLUMN, so record-wins is the correct order.
    memory::populate_register_read_write<F>(frame.op_a_access, a_record);
    if (b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, b_record);
    }
    if (c_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_c_access, c_record);
    }
}

// `InstructionFrameCols::populate_from_alu` (also `populate_from_comp_alu` —
// the two event types carry identically named frame fields).
template<class E, class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_alu(
    InstructionFrameCols<F>& frame,
    const E& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a, event.b, event.c,
        event.a_record, event.b_record, event.c_record, instruction, shard);
}

// `InstructionFrameCols::populate_from_branch` — a branch READS op_a.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_branch(
    InstructionFrameCols<F>& frame,
    const BranchEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a, event.b, event.c,
        event.a_record, event.b_record, event.c_record, instruction, shard);
}

// `InstructionFrameCols::populate_from_jump` — a jump WRITES op_a (the link
// register), so `op_a_immutable` stays 0.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_jump(
    InstructionFrameCols<F>& frame,
    const JumpEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a, event.b, event.c,
        event.a_record, event.b_record, event.c_record, instruction, shard);
}

// `InstructionFrameCols::populate_from_movcond` — MNE/MEQ are `is_rw_a`
// (op_a keeps its previous value when the condition fails); WSBH is a plain
// write.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_movcond(
    InstructionFrameCols<F>& frame,
    const MovCondEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a, event.b, event.c,
        event.a_record, event.b_record, event.c_record, instruction, shard);
    if (event.opcode != Opcode::WSBH) {
    }
}

// `InstructionFrameCols::populate_from_misc` — MADDU/MSUBU/MADD/MSUB/INS
// read-and-write op_a; TEQ reads it immutably.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_misc(
    InstructionFrameCols<F>& frame,
    const MiscEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a, event.b, event.c,
        event.a_record, event.b_record, event.c_record, instruction, shard);
}

// `ITypeFrameCols::populate_from_mem` — every memory instruction is I-type
// (`op_b` a register, `op_c` an immediate) and reads-and-writes op_a; the plain
// stores (NOT SC) read it immutably.  This is `populate_raw` with the columns
// the shape makes constant left out: no `op_c` register access, no immediate
// flags, and `op_b` written as a bare register index rather than a word.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_mem(
    ITypeFrameCols<F>& frame,
    const MemInstrEvent& event,
    const InstructionFfi& instruction
) {
    frame.shard = F::from_canonical_u32(event.shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(event.clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((event.clk >> 16) & 0x3ff));

    frame.opcode = F::from_canonical_u32((uint32_t)instruction.opcode);
    frame.op_a = F::from_canonical_u32((uint32_t)instruction.op_a);
    frame.op_b = F::from_canonical_u32(instruction.op_b);
    write_word_from_u32_v2<F>(frame.op_c, instruction.op_c);
    frame.op_a_0 = F::from_bool(instruction.op_a == 0);  // 0 = Register::X0

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, event.a);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, event.b);

    // Record-wins ordering, as in `populate_raw`.
    memory::populate_register_read_write<F>(frame.op_a_access, event.a_record);
    if (event.b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, event.b_record);
    }
}

// `RTypeFrameCols::populate_from_alu` (also `populate_from_comp_alu` — the
// two event types carry identically named frame fields) — a register-form ALU
// instruction reads op_b and op_c and reads-and-writes op_a; all three
// operands are bare register indices, so there are no immediate flags and no
// operand words.
template<class E, class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_alu_r(
    RTypeFrameCols<F>& frame,
    const E& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    frame.shard = F::from_canonical_u32(shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(event.clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((event.clk >> 16) & 0x3ff));

    frame.opcode = F::from_canonical_u32((uint32_t)instruction.opcode);
    frame.op_a = F::from_canonical_u32((uint32_t)instruction.op_a);
    frame.op_b = F::from_canonical_u32(instruction.op_b);
    frame.op_c = F::from_canonical_u32(instruction.op_c);
    frame.op_a_0 = F::from_bool(instruction.op_a == 0);  // 0 = Register::X0

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, event.a);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, event.b);
    write_word_from_u32_v2<F>(frame.op_c_access.access.value, event.c);

    // Record-wins ordering, as in `populate_raw`.
    memory::populate_register_read_write<F>(frame.op_a_access, event.a_record);
    if (event.b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, event.b_record);
    }
    if (event.c_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_c_access, event.c_record);
    }
}

// `ITypeFrameCols::populate_from_alu` (also `populate_from_branch` — the
// AluEvent-shaped events carry identically named frame fields) — an
// immediate-form instruction on the I-type frame.  Identical to
// `populate_from_mem` except the event type: these events carry no shard of
// their own, so it arrives as a parameter.
template<class E, class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_alu_imm(
    ITypeFrameCols<F>& frame,
    const E& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    frame.shard = F::from_canonical_u32(shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(event.clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((event.clk >> 16) & 0x3ff));

    frame.opcode = F::from_canonical_u32((uint32_t)instruction.opcode);
    frame.op_a = F::from_canonical_u32((uint32_t)instruction.op_a);
    frame.op_b = F::from_canonical_u32(instruction.op_b);
    write_word_from_u32_v2<F>(frame.op_c, instruction.op_c);
    frame.op_a_0 = F::from_bool(instruction.op_a == 0);  // 0 = Register::X0

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, event.a);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, event.b);

    // Record-wins ordering, as in `populate_raw`.
    memory::populate_register_read_write<F>(frame.op_a_access, event.a_record);
    if (event.b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, event.b_record);
    }
}

// `ShamtFrameCols::populate_from_alu` — a shamt-form shift on the shamt
// frame: `op_b` a register, the immediate collapsed to its low limb.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_alu_shamt(
    ShamtFrameCols<F>& frame,
    const AluEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    frame.shard = F::from_canonical_u32(shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(event.clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((event.clk >> 16) & 0x3ff));

    frame.opcode = F::from_canonical_u32((uint32_t)instruction.opcode);
    frame.op_a = F::from_canonical_u32((uint32_t)instruction.op_a);
    frame.op_b = F::from_canonical_u32(instruction.op_b);
    frame.op_c = F::from_canonical_u32(instruction.op_c);
    frame.op_a_0 = F::from_bool(instruction.op_a == 0);  // 0 = Register::X0

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, event.a);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, event.b);

    // Record-wins ordering, as in `populate_raw`.
    memory::populate_register_read_write<F>(frame.op_a_access, event.a_record);
    if (event.b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, event.b_record);
    }
}

// `InstructionFrameCols::populate_from_syscall` — the id comes in through
// op_a, the result goes out; `hi_or_prev_a` and `num_extra_cycles` read back
// the POPULATED column, exactly as the Rust side does.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_syscall(
    InstructionFrameCols<F>& frame,
    const SyscallEvent& event,
    const InstructionFfi& instruction
) {
    OptionMemoryRecordEnum a_record = {};
    if (event.a_record_is_real) {
        a_record.tag = OptionMemoryRecordEnumTag::Write;
        a_record.value = event.a_record.value;
        a_record.timestamp = event.a_record.timestamp;
        a_record.prev_timestamp = event.a_record.prev_timestamp;
        a_record.prev_value = event.a_record.prev_value;
    } else {
        a_record.tag = OptionMemoryRecordEnumTag::None;
    }
    populate_raw<F>(
        frame, event.clk, event.recv_next_pc, event.a_record.value, event.arg1,
        event.arg2, a_record, event.b_record, event.c_record, instruction,
        event.shard);
}

// `RTypeFrameCols::populate_from_syscall` — SYSCALL is register-form; the id
// comes in through op_a, the result goes out.  Mirrors the universal variant.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_from_syscall_r(
    RTypeFrameCols<F>& frame,
    const SyscallEvent& event,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    frame.shard = F::from_canonical_u32(shard);
    frame.clk_16bit_limb = F::from_canonical_u16((uint16_t)(event.clk & 0xffff));
    frame.clk_high_limb = F::from_canonical_u16((uint16_t)((event.clk >> 16) & 0x3ff));

    frame.opcode = F::from_canonical_u32((uint32_t)instruction.opcode);
    frame.op_a = F::from_canonical_u32((uint32_t)instruction.op_a);
    frame.op_b = F::from_canonical_u32(instruction.op_b);
    frame.op_c = F::from_canonical_u32(instruction.op_c);
    frame.op_a_0 = F::from_bool(instruction.op_a == 0);  // 0 = Register::X0

    write_word_from_u32_v2<F>(frame.op_a_access.access.value, event.a_record.value);
    write_word_from_u32_v2<F>(frame.op_b_access.access.value, event.arg1);
    write_word_from_u32_v2<F>(frame.op_c_access.access.value, event.arg2);

    // Record-wins ordering, as in `populate_raw`.
    OptionMemoryRecordEnum a_record = {};
    if (event.a_record_is_real) {
        a_record.tag = OptionMemoryRecordEnumTag::Write;
        a_record.value = event.a_record.value;
        a_record.timestamp = event.a_record.timestamp;
        a_record.prev_timestamp = event.a_record.prev_timestamp;
        a_record.prev_value = event.a_record.prev_value;
    } else {
        a_record.tag = OptionMemoryRecordEnumTag::None;
    }
    memory::populate_register_read_write<F>(frame.op_a_access, a_record);
    if (event.b_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_b_access, event.b_record);
    }
    if (event.c_record.tag == OptionMemoryRecordEnumTag::Read) {
        memory::populate_register_read<F>(frame.op_c_access, event.c_record);
    }
}

// `InstructionFrameCols::populate_dependency` — neutralise the frame on a row
// that carries no instruction (dependency AND padding rows): the not-real rule
// forces the immediate flags high so the op_b / op_c register-access
// multiplicities (`ONE - imm_b`) vanish.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_dependency(InstructionFrameCols<F>& frame) {
    frame.instruction.imm_b = F::one();
    frame.instruction.imm_c = F::one();
}

}  // namespace zkm_core_machine_sys::frame
