#pragma once

#include <cstdlib>

#include "prelude.hpp"
#include "utils.hpp"

namespace zkm_core_machine_sys::memory {

template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_access(
    MemoryAccessCols<F>& self,
    const MemoryRecord& current_record,
    const MemoryRecord& prev_record
) {
    write_word_from_u32_v2<F>(self.value, current_record.value);

    self.prev_shard = F::from_canonical_u32(prev_record.shard);
    self.prev_clk = F::from_canonical_u32(prev_record.timestamp);

    // Fill columns used for verifying current memory access time value is greater than
    // previous's.
    const bool use_clk_comparison = prev_record.shard == current_record.shard;
    self.compare_clk = F::from_bool(use_clk_comparison);
    const uint32_t prev_time_value = use_clk_comparison ? prev_record.timestamp : prev_record.shard;
    const uint32_t current_time_value =
        use_clk_comparison ? current_record.timestamp : current_record.shard;

    const uint32_t diff_minus_one = current_time_value - prev_time_value - 1;
    const uint16_t diff_16bit_limb = (uint16_t)(diff_minus_one & 0xffff);
    self.diff_16bit_limb = F::from_canonical_u16(diff_16bit_limb).val;
    // High limb: bits 16..26 (TIMESTAMP_HIGH_LIMB_BITS = 10, CORE_SHARD_CLK_LIMIT = 2^26).
    self.diff_high_limb = F::from_canonical_u16((uint16_t)((diff_minus_one >> 16) & 0x3ff));
}

// ---------------------------------------------------------------------------
// Register accesses.
//
// The `MemoryBump` chip guarantees `prev_shard == shard` for every register
// access (it inserts a shadow read at `(shard, 0)` on the register's first
// touch in the shard).  So a register access witnesses neither `prev_shard`
// nor `compare_clk` nor the high limb of the timestamp difference -- the
// high limb is a linear expression in the AIR.  9 columns -> 6.
// ---------------------------------------------------------------------------

template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_register_access(
    RegisterAccessCols<F>& self,
    const uint32_t timestamp,
    const uint32_t prev_timestamp,
    const uint32_t value
) {
    write_word_from_u32_v2<F>(self.value, value);
    self.prev_clk = F::from_canonical_u32(prev_timestamp);

    const uint32_t diff_minus_one = timestamp - prev_timestamp - 1;
    self.diff_16bit_limb = F::from_canonical_u16((uint16_t)(diff_minus_one & 0xffff)).val;
}

// Takes the NARROW register record the instruction events carry: a register
// access is witnessed by its value and its two timestamps and nothing else, so
// there is nothing here the wide `MemoryReadRecord` would have added.  The
// caller has already checked the tag.
template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void
populate_register_read(RegisterReadCols<F>& self, const OptionMemoryReadRecord& record) {
    populate_register_access<F>(self.access, record.timestamp, record.prev_timestamp, record.value);
}

template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_register_read_write(
    RegisterReadWriteCols<F>& self,
    const OptionMemoryRecordEnum& record
) {
    if (record.tag == OptionMemoryRecordEnumTag::None) {
        return;
    }
    // No read/write branch left: the two arms differed only in what went into
    // `prev_value`, and the conversion to `OptionMemoryRecordEnum` resolved it
    // (a read leaves the previous value equal to its own value).
    write_word_from_u32_v2<F>(self.prev_value, record.prev_value);
    populate_register_access<F>(
        self.access,
        record.timestamp,
        record.prev_timestamp,
        record.value
    );
}

template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void
populate_read(MemoryReadCols<F>& self, const MemoryReadRecord& record) {
    const MemoryRecord current_record = {
        .shard = record.shard,
        .timestamp = record.timestamp,
        .value = record.value,
    };
    const MemoryRecord prev_record = {
        .shard = record.prev_shard,
        .timestamp = record.prev_timestamp,
        .value = record.value,
    };
    populate_access<F>(self.access, current_record, prev_record);
}


template<class F>
__ZKM_HOSTDEV__ __ZKM_INLINE__ void populate_read_write_v2(
    MemoryReadWriteCols<F>& self,
    const MemoryRecordEnum& record
) {
    MemoryRecord current_record;
    MemoryRecord prev_record;
    switch (record.tag) {
        case MemoryRecordEnum::Tag::Read:
            current_record = {
                .shard = record.read._0.shard,
                .timestamp = record.read._0.timestamp,
                .value = record.read._0.value,
            };
            prev_record = {
                .shard = record.read._0.prev_shard,
                .timestamp = record.read._0.prev_timestamp,
                .value = record.read._0.value,
            };
            break;
        case MemoryRecordEnum::Tag::Write:
            current_record = {
                .shard = record.write._0.shard,
                .timestamp = record.write._0.timestamp,
                .value = record.write._0.value,
            };
            prev_record = {
                .shard = record.write._0.prev_shard,
                .timestamp = record.write._0.prev_timestamp,
                .value = record.write._0.prev_value,
            };
            break;
        default:
            // Unreachable. `None` case guarded above.
            assert(false);
            break;
    }
    write_word_from_u32_v2<F>(self.prev_value, prev_record.value);
    populate_access<F>(self.access, current_record, prev_record);
}
}  // namespace zkm_core_machine_sys::memory
