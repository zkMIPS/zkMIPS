use p3_field::PrimeField32;
use zkm_core_executor::events::{
    ByteRecord, MemoryReadRecord, MemoryRecord, MemoryRecordEnum, MemoryWriteRecord,
};

use super::{
    MemoryAccessCols, MemoryReadCols, MemoryReadWriteCols, MemoryWriteCols, RegisterAccessCols,
    RegisterReadCols, RegisterReadWriteCols,
};
use crate::air::{TIMESTAMP_HIGH_LIMB_BITS, TIMESTAMP_HIGH_LIMB_MASK};

/// Which memory words a `populate_access` records byte range checks for; mirrors the
/// sends of `eval_memory_access` (`Both`/`ValueOnly`) and `eval_memory_access_trusted`
/// (`None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteChecks {
    Both,
    ValueOnly,
    None,
}

impl<F: PrimeField32> RegisterReadCols<F> {
    /// Populate from the NARROW register record the instruction events carry.
    ///
    /// A register access is witnessed by its value and its two timestamps and
    /// nothing else, so this is the whole of `populate` for a register — see
    /// [`OptionMemoryReadRecord`], which carries exactly those.  The caller has
    /// already established the access is real.
    pub fn populate_register(
        &mut self,
        record: zkm_core_executor::events::OptionMemoryReadRecord,
        output: &mut impl ByteRecord,
    ) {
        self.access.populate_register_access(
            record.timestamp,
            record.prev_timestamp,
            record.value,
            output,
        );
    }

    pub fn populate(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        self.access.populate_access(
            record.timestamp,
            record.prev_timestamp,
            record.value,
            record.shard,
            record.prev_shard,
            output,
        );
    }
}

impl<F: PrimeField32> RegisterReadWriteCols<F> {
    /// Narrow twin of [`Self::populate`] — see
    /// [`RegisterReadCols::populate_register`].
    ///
    /// There is no read/write branch left: the two arms differed only in what
    /// went into `prev_value`, and the conversion to
    /// [`OptionMemoryRecordEnum`] already resolved that (a read's previous
    /// value is its own value).  The caller has established the access is real.
    pub fn populate_register(
        &mut self,
        record: zkm_core_executor::events::OptionMemoryRecordEnum,
        output: &mut impl ByteRecord,
    ) {
        self.prev_value = record.prev_value.into();
        self.access.populate_register_access(
            record.timestamp,
            record.prev_timestamp,
            record.value,
            output,
        );
    }

    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        match record {
            MemoryRecordEnum::Read(r) => {
                self.prev_value = r.value.into();
                self.access.populate_access(
                    r.timestamp,
                    r.prev_timestamp,
                    r.value,
                    r.shard,
                    r.prev_shard,
                    output,
                );
            }
            MemoryRecordEnum::Write(w) => {
                self.prev_value = w.prev_value.into();
                self.access.populate_access(
                    w.timestamp,
                    w.prev_timestamp,
                    w.value,
                    w.shard,
                    w.prev_shard,
                    output,
                );
            }
        }
    }
}

impl<F: PrimeField32> RegisterAccessCols<F> {
    /// Populate a register access from the columns it actually witnesses.
    ///
    /// [`Self::populate_access`] additionally takes `shard` / `prev_shard` to
    /// assert they are equal; the narrow record has neither, because that
    /// assertion moved to the conversion that drops them.
    pub(crate) fn populate_register_access(
        &mut self,
        timestamp: u32,
        prev_timestamp: u32,
        value: u32,
        output: &mut impl ByteRecord,
    ) {
        self.value = value.into();
        self.prev_clk = F::from_u32(prev_timestamp);

        let diff_minus_one = timestamp.wrapping_sub(prev_timestamp).wrapping_sub(1);
        let diff_16bit_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_16bit_limb = F::from_u16(diff_16bit_limb);

        output.add_u16_range_check(diff_16bit_limb);
        output.add_bit_range_check(
            ((diff_minus_one >> 16) & TIMESTAMP_HIGH_LIMB_MASK) as u16,
            TIMESTAMP_HIGH_LIMB_BITS,
        );
    }

    /// Populate a register access.
    ///
    /// `prev_shard` is only taken to assert the `MemoryBump` invariant in debug builds — it is not
    /// witnessed, because it is guaranteed to equal `shard`.
    pub(crate) fn populate_access(
        &mut self,
        timestamp: u32,
        prev_timestamp: u32,
        value: u32,
        shard: u32,
        prev_shard: u32,
        output: &mut impl ByteRecord,
    ) {
        debug_assert_eq!(
            shard, prev_shard,
            "register access at addr-time {timestamp} has prev_shard {prev_shard} != shard \
             {shard}: the MemoryBump shadow read is missing"
        );
        self.value = value.into();
        self.prev_clk = F::from_u32(prev_timestamp);

        let diff_minus_one = timestamp.wrapping_sub(prev_timestamp).wrapping_sub(1);
        let diff_16bit_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_16bit_limb = F::from_u16(diff_16bit_limb);

        // Add a byte table lookup with the 16Range op.
        output.add_u16_range_check(diff_16bit_limb);

        // The 9-bit high limb is a recovered linear expression in the AIR, not
        // a column; it is checked against the parametric range table.
        output.add_bit_range_check(
            ((diff_minus_one >> 16) & TIMESTAMP_HIGH_LIMB_MASK) as u16,
            TIMESTAMP_HIGH_LIMB_BITS,
        );
    }
}

impl<F: PrimeField32> MemoryWriteCols<F> {
    pub fn populate(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.prev_value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, ByteChecks::Both, output);
    }
}

impl<F: PrimeField32> MemoryReadCols<F> {
    pub fn populate(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        self.populate_with(record, ByteChecks::ValueOnly, output);
    }

    /// Populate for a chip that evaluates the access with `eval_memory_access_trusted`
    /// (no byte range checks): the memory-instruction chips, whose words come from and go to
    /// registers.
    pub fn populate_trusted(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        self.populate_with(record, ByteChecks::None, output);
    }

    fn populate_with(
        &mut self,
        record: MemoryReadRecord,
        byte_checks: ByteChecks,
        output: &mut impl ByteRecord,
    ) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.access.populate_access(current_record, prev_record, byte_checks, output);
    }
}

impl<F: PrimeField32> MemoryReadWriteCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        match record {
            MemoryRecordEnum::Read(read_record) => self.populate_read(read_record, output),
            MemoryRecordEnum::Write(write_record) => self.populate_write(write_record, output),
        }
    }

    /// Populate for a chip that evaluates the access with `eval_memory_access_trusted`
    /// (no byte range checks); see `MemoryReadCols::populate_trusted`.
    pub fn populate_trusted(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        // `output` still receives the timestamp range checks; only the value byte checks go.
        match record {
            MemoryRecordEnum::Read(r) => self.populate_read_with(r, ByteChecks::None, output),
            MemoryRecordEnum::Write(w) => self.populate_write_with(w, ByteChecks::None, output),
        }
    }

    pub fn populate_write(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        self.populate_write_with(record, ByteChecks::Both, output);
    }

    fn populate_write_with(
        &mut self,
        record: MemoryWriteRecord,
        byte_checks: ByteChecks,
        output: &mut impl ByteRecord,
    ) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.prev_value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, byte_checks, output);
    }

    pub fn populate_read(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        self.populate_read_with(record, ByteChecks::Both, output);
    }

    fn populate_read_with(
        &mut self,
        record: MemoryReadRecord,
        byte_checks: ByteChecks,
        output: &mut impl ByteRecord,
    ) {
        let current_record =
            MemoryRecord { value: record.value, shard: record.shard, timestamp: record.timestamp };
        let prev_record = MemoryRecord {
            value: record.value,
            shard: record.prev_shard,
            timestamp: record.prev_timestamp,
        };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, byte_checks, output);
    }
}

impl<F: PrimeField32> MemoryAccessCols<F> {
    pub(crate) fn populate_access(
        &mut self,
        current_record: MemoryRecord,
        prev_record: MemoryRecord,
        byte_checks: ByteChecks,
        output: &mut impl ByteRecord,
    ) {
        self.value = current_record.value.into();

        // Match the byte range checks emitted by `eval_memory_access`: one per memory word
        // (`Both`), one for a read-only access whose `prev_value` IS `value` (`ValueOnly`,
        // `MemoryCols::value_aliases_prev`), none for the memory-instruction chips
        // (`None`, `eval_memory_access_trusted`).
        match byte_checks {
            ByteChecks::Both => {
                output.add_u8_range_checks(&prev_record.value.to_le_bytes());
                output.add_u8_range_checks(&current_record.value.to_le_bytes());
            }
            ByteChecks::ValueOnly => {
                output.add_u8_range_checks(&current_record.value.to_le_bytes())
            }
            ByteChecks::None => {}
        }

        self.prev_shard = F::from_u32(prev_record.shard);
        self.prev_clk = F::from_u32(prev_record.timestamp);

        // Fill columns used for verifying current memory access time value is greater than
        // previous's.
        let use_clk_comparison = prev_record.shard == current_record.shard;
        self.compare_clk = F::from_bool(use_clk_comparison);
        let prev_time_value =
            if use_clk_comparison { prev_record.timestamp } else { prev_record.shard };
        let current_time_value =
            if use_clk_comparison { current_record.timestamp } else { current_record.shard };

        let diff_minus_one = (current_time_value - prev_time_value).wrapping_sub(1);
        let diff_16bit_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_16bit_limb = F::from_u16(diff_16bit_limb);
        let diff_high_limb = ((diff_minus_one >> 16) & TIMESTAMP_HIGH_LIMB_MASK) as u16;
        self.diff_high_limb = F::from_u16(diff_high_limb);

        // Add a byte table lookup with the 16Range op.
        output.add_u16_range_check(diff_16bit_limb);

        // Bound the high limb against the parametric range table.
        output.add_bit_range_check(diff_high_limb, TIMESTAMP_HIGH_LIMB_BITS);
    }
}
