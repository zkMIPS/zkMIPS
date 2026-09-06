use serde::{Deserialize, Serialize};

use super::memory::MemoryRecordEnum;

/// CPU Event.
///
/// This object encapsulates the information needed to prove a CPU operation. This includes its
/// shard, opcode, operands, and other relevant information.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct CpuEvent {
    /// The clock cycle.
    pub clk: u32,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The next after the next program counter.
    pub next_next_pc: u32,
    /// The exit code.
    pub exit_code: u32,
}

/// A REGISTER read as the frame consumes it: the tag plus the three values
/// `RegisterAccessCols::populate_access` actually witnesses.
///
/// A register access never crosses a shard boundary — `populate_access` records
/// that `prev_shard` "is not witnessed, because it is guaranteed to equal
/// `shard`" — so the `shard` / `prev_shard` pair of the wrapped
/// `MemoryReadRecord` was 8 B per operand per cycle that reached no column.
/// The equality they existed to assert is now checked once, here, at the
/// conversion, where both are still in hand.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct OptionMemoryReadRecord {
    pub tag: OptionMemoryRecordEnumTag,
    /// The value read.
    pub value: u32,
    /// The access timestamp.
    pub timestamp: u32,
    /// The timestamp of the previous access to this register.
    pub prev_timestamp: u32,
}

impl OptionMemoryReadRecord {
    /// The absent access; also what a `Write` collapses to, since nothing
    /// consumes the write arm of a read-only operand.
    #[must_use]
    pub const fn none(tag: OptionMemoryRecordEnumTag) -> Self {
        Self { tag, value: 0, timestamp: 0, prev_timestamp: 0 }
    }
}

impl From<Option<MemoryRecordEnum>> for OptionMemoryReadRecord {
    fn from(record: Option<MemoryRecordEnum>) -> Self {
        match record {
            Some(MemoryRecordEnum::Read(read)) => {
                debug_assert_eq!(
                    read.shard, read.prev_shard,
                    "register read at addr-time {} has prev_shard {} != shard {}: the \
                     MemoryBump shadow read is missing",
                    read.timestamp, read.prev_shard, read.shard
                );
                OptionMemoryReadRecord {
                    tag: OptionMemoryRecordEnumTag::Read,
                    value: read.value,
                    timestamp: read.timestamp,
                    prev_timestamp: read.prev_timestamp,
                }
            }
            Some(MemoryRecordEnum::Write(_)) => Self::none(OptionMemoryRecordEnumTag::Write),
            None => Self::none(OptionMemoryRecordEnumTag::None),
        }
    }
}

/// A REGISTER read-and-write as the frame consumes it.  The read and write
/// arms differ in exactly one witnessed column — `prev_value`, which a read
/// leaves equal to its own `value` — so they collapse into one record and the
/// consumer no longer branches on the tag except to skip an absent access.
///
/// See [`OptionMemoryReadRecord`] for why `shard` / `prev_shard` are gone.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct OptionMemoryRecordEnum {
    pub tag: OptionMemoryRecordEnumTag,
    /// The value after the access.
    pub value: u32,
    /// The access timestamp.
    pub timestamp: u32,
    /// The timestamp of the previous access to this register.
    pub prev_timestamp: u32,
    /// The value BEFORE the access: a write's `prev_value`, a read's own
    /// `value` — which is what `RegisterReadWriteCols::populate` wrote into the
    /// `prev_value` column for a read.
    pub prev_value: u32,
}

impl OptionMemoryRecordEnum {
    /// The absent access.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            tag: OptionMemoryRecordEnumTag::None,
            value: 0,
            timestamp: 0,
            prev_timestamp: 0,
            prev_value: 0,
        }
    }
}

impl From<Option<MemoryRecordEnum>> for OptionMemoryRecordEnum {
    fn from(record: Option<MemoryRecordEnum>) -> Self {
        match record {
            Some(MemoryRecordEnum::Read(read)) => {
                debug_assert_eq!(
                    read.shard, read.prev_shard,
                    "register read at addr-time {} has prev_shard {} != shard {}: the \
                     MemoryBump shadow read is missing",
                    read.timestamp, read.prev_shard, read.shard
                );
                OptionMemoryRecordEnum {
                    tag: OptionMemoryRecordEnumTag::Read,
                    value: read.value,
                    timestamp: read.timestamp,
                    prev_timestamp: read.prev_timestamp,
                    prev_value: read.value,
                }
            }
            Some(MemoryRecordEnum::Write(write)) => {
                debug_assert_eq!(
                    write.shard, write.prev_shard,
                    "register write at addr-time {} has prev_shard {} != shard {}: the \
                     MemoryBump shadow read is missing",
                    write.timestamp, write.prev_shard, write.shard
                );
                OptionMemoryRecordEnum {
                    tag: OptionMemoryRecordEnumTag::Write,
                    value: write.value,
                    timestamp: write.timestamp,
                    prev_timestamp: write.prev_timestamp,
                    prev_value: write.prev_value,
                }
            }
            None => Self::none(),
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[repr(u8)]
pub enum OptionMemoryRecordEnumTag {
    Read = 0,
    Write,
    None,
}
