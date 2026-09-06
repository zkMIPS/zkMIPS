#pragma once

#include "frame.hpp"
#include "prelude.hpp"
#include "utils.hpp"
#include "kb31_septic_extension_t.hpp"

namespace zkm_core_machine_sys::shift_left_imm {
    template<class F>
    __ZKM_HOSTDEV__ void event_to_row(
    const AluEvent& event,
    ShiftLeftImmCols<F>& cols,
    const InstructionFfi& instruction,
    const uint32_t shard
) {
    // Every row is a real instruction owning its frame.
    frame::populate_from_alu_shamt<F>(cols.frame, event, instruction, shard);

        auto a = u32_to_le_bytes(event.a);
        auto b = u32_to_le_bytes(event.b);

        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::one();
        // 5 shamt bits (mirror of `sll_imm/mod.rs`: `SHAMT_BITS`).
        for (uint32_t i = 0; i < 5; i += 1) {
            cols.c_least_sig_byte[i] = F::from_canonical_u32((event.c >> i) & 1);
        }

        // Variables for bit shifting (the multiplier is pinned by the AIR as
        // (1 + c0)(1 + 3 c1)(1 + 15 c2); no selector array).
        uint32_t num_bits_to_shift = event.c % BYTE_SIZE;

        uint32_t bit_shift_multiplier = 1u << num_bits_to_shift;
        cols.bit_shift_multiplier = F::from_canonical_u32(bit_shift_multiplier);

        uint32_t carry = 0u;
        uint32_t base = 1u << BYTE_SIZE;
        uint8_t bit_shift_result[WORD_SIZE] = {0u};
        uint8_t bit_shift_result_carry[WORD_SIZE] = {0u};
        for (uint32_t i = 0; i < WORD_SIZE; i++) {
            uint32_t v = b[i] * bit_shift_multiplier + carry;
            carry = v / base;
            bit_shift_result[i] = v % base;
            bit_shift_result_carry[i] = carry;
        }
        write_word_from_le_bytes_v2<F>(cols.bit_shift_result, bit_shift_result);
        write_word_from_le_bytes_v2<F>(cols.bit_shift_result_carry, bit_shift_result_carry);

        // Variables for byte shifting.
        uint32_t num_bytes_to_shift = (event.c & 0b11111u) / BYTE_SIZE;
        for (uint32_t i = 0; i < WORD_SIZE; i++) {
            cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
        }

        // Sanity check.
        for (uint32_t i = num_bytes_to_shift; i < WORD_SIZE; i++) {
            assert(
                cols.bit_shift_result[i - num_bytes_to_shift] == F::from_canonical_u8(a[i])
            );
        }
    }
}  // namespace zkm::shift_left_imm
