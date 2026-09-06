use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_koala_bear::KoalaBear;

use crate::{events::NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC, MipsAirId, Opcode, NUM_REGISTERS};

const BYTE_NUM_ROWS: u64 = 1 << 16;
const MAX_PROGRAM_SIZE: u64 = 1 << 22;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_mips_lde_size(
    num_events_per_air: EnumMap<MipsAirId, u64>,
    costs_per_air: &HashMap<MipsAirId, u64>,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&MipsAirId::Byte];

    // Compute the program chip contribution.
    cells += MAX_PROGRAM_SIZE * costs_per_air[&MipsAirId::Program];

    // There is no Cpu chip any more (every instruction chip carries its own
    // frame); `MipsAirId::Cpu` survives only as the VIRTUAL cycles axis for
    // shard splitting and shape banding, and contributes no area.

    // Compute the addsub chip contribution.
    cells += (num_events_per_air[MipsAirId::AddSub]).next_power_of_two()
        * costs_per_air[&MipsAirId::AddSub];

    // Compute the immediate-form addsub chip contribution.
    cells += (num_events_per_air[MipsAirId::AddSubImm]).next_power_of_two()
        * costs_per_air[&MipsAirId::AddSubImm];

    // Compute the immediate-form bitwise chip contribution.
    cells += (num_events_per_air[MipsAirId::BitwiseImm]).next_power_of_two()
        * costs_per_air[&MipsAirId::BitwiseImm];

    // Compute the immediate-form shift and compare chip contributions.
    for air in [MipsAirId::ShiftLeftImm, MipsAirId::ShiftRightImm, MipsAirId::LtImm] {
        cells += (num_events_per_air[air]).next_power_of_two() * costs_per_air[&air];
    }

    // Compute the mul chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Mul]).next_power_of_two() * costs_per_air[&MipsAirId::Mul];

    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[MipsAirId::Bitwise]).next_power_of_two()
        * costs_per_air[&MipsAirId::Bitwise];

    // Compute the shift left chip contribution.
    cells += (num_events_per_air[MipsAirId::ShiftLeft]).next_power_of_two()
        * costs_per_air[&MipsAirId::ShiftLeft];

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[MipsAirId::ShiftRight]).next_power_of_two()
        * costs_per_air[&MipsAirId::ShiftRight];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[MipsAirId::DivRem]).next_power_of_two()
        * costs_per_air[&MipsAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Lt]).next_power_of_two() * costs_per_air[&MipsAirId::Lt];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryLocal]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryLocal];

    // Compute the memory bump chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryBump]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryBump];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[MipsAirId::Branch]).next_power_of_two()
        * costs_per_air[&MipsAirId::Branch];

    // Compute the jump chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Jump]).next_power_of_two() * costs_per_air[&MipsAirId::Jump];

    // Compute the SyscallInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallInstrs];

    // Compute the memory-instruction chip contributions.
    for air in [
        MipsAirId::LoadNarrow,
        MipsAirId::LoadWord,
        MipsAirId::StoreNarrow,
        MipsAirId::StoreWord,
        MipsAirId::MemoryUnaligned,
    ] {
        cells += (num_events_per_air[air]).next_power_of_two() * costs_per_air[&air];
    }

    // Compute the MiscInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::MiscInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::MiscInstrs];

    // Compute the cloclz chip contribution.
    cells += (num_events_per_air[MipsAirId::CloClz]).next_power_of_two()
        * costs_per_air[&MipsAirId::CloClz];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallCore]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallCore];

    // Compute the global chip contribution.
    cells += (num_events_per_air[MipsAirId::Global]).next_power_of_two()
        * costs_per_air[&MipsAirId::Global];

    cells * ((core::mem::size_of::<KoalaBear>() << 1) as u64)
}

/// Estimate
/// Maps the opcode counts to the number of events in each air.
#[must_use]
pub fn estimate_mips_event_counts(
    cpu_cycles: u64,
    touched_addresses: u64,
    syscalls_sent: u64,
    opcode_counts: EnumMap<Opcode, u64>,
) -> EnumMap<MipsAirId, u64> {
    let mut events_counts: EnumMap<MipsAirId, u64> = EnumMap::default();
    // Compute the number of events in the cpu chip.
    events_counts[MipsAirId::Cpu] = cpu_cycles;

    // Compute the number of events in the add sub chip.  Opcode counts cannot
    // see the operand FORM, so the immediate-form rows (which really land on
    // the narrower `AddSubImm`) are billed to the wider register-form air —
    // a conservative over-estimate.  The live accumulator routes them exactly
    // (see the executor's bookkeeping block); this pure function survives only
    // for its equivalence test and the offline shape tooling.
    events_counts[MipsAirId::AddSub] = opcode_counts[Opcode::ADD] + opcode_counts[Opcode::SUB];

    // Compute the number of events in the mul chip.
    events_counts[MipsAirId::Mul] =
        opcode_counts[Opcode::MUL] + opcode_counts[Opcode::MULT] + opcode_counts[Opcode::MULTU];

    // Compute the number of events in the bitwise chip.
    events_counts[MipsAirId::Bitwise] = opcode_counts[Opcode::XOR]
        + opcode_counts[Opcode::OR]
        + opcode_counts[Opcode::AND]
        + opcode_counts[Opcode::NOR];

    // Compute the number of events in the shift left chip.
    events_counts[MipsAirId::ShiftLeft] = opcode_counts[Opcode::SLL];

    // Compute the number of events in the shift right chip.
    events_counts[MipsAirId::ShiftRight] =
        opcode_counts[Opcode::SRL] + opcode_counts[Opcode::SRA] + opcode_counts[Opcode::ROR];

    // Compute the number of events in the divrem chip.
    events_counts[MipsAirId::DivRem] = opcode_counts[Opcode::DIV] + opcode_counts[Opcode::DIVU];

    // Compute the number of events in the lt chip.
    events_counts[MipsAirId::Lt] = opcode_counts[Opcode::SLT] + opcode_counts[Opcode::SLTU];

    // Compute the number of events in the memory local chip.
    events_counts[MipsAirId::MemoryLocal] =
        touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);

    // Compute the number of events in the memory bump chip: at most one shadow read per
    // register per shard.
    events_counts[MipsAirId::MemoryBump] = NUM_REGISTERS as u64;

    // Compute the number of events in the branch chip.
    events_counts[MipsAirId::Branch] = opcode_counts[Opcode::BEQ]
        + opcode_counts[Opcode::BNE]
        + opcode_counts[Opcode::BGTZ]
        + opcode_counts[Opcode::BGEZ]
        + opcode_counts[Opcode::BLTZ]
        + opcode_counts[Opcode::BLEZ];

    // Compute the number of events in the jump chip.
    events_counts[MipsAirId::Jump] = opcode_counts[Opcode::Jump]
        + opcode_counts[Opcode::Jumpi]
        + opcode_counts[Opcode::JumpDirect];

    // Compute the number of events in the memory-instruction chips.
    events_counts[MipsAirId::LoadNarrow] = opcode_counts[Opcode::LB]
        + opcode_counts[Opcode::LBU]
        + opcode_counts[Opcode::LH]
        + opcode_counts[Opcode::LHU];
    events_counts[MipsAirId::LoadWord] = opcode_counts[Opcode::LW] + opcode_counts[Opcode::LL];
    events_counts[MipsAirId::StoreNarrow] = opcode_counts[Opcode::SB] + opcode_counts[Opcode::SH];
    events_counts[MipsAirId::StoreWord] = opcode_counts[Opcode::SW] + opcode_counts[Opcode::SC];
    events_counts[MipsAirId::MemoryUnaligned] = opcode_counts[Opcode::LWL]
        + opcode_counts[Opcode::LWR]
        + opcode_counts[Opcode::SWL]
        + opcode_counts[Opcode::SWR];

    // Compute the number of events in the MiscInstrs chip.
    events_counts[MipsAirId::MiscInstrs] = opcode_counts[Opcode::INS]
        + opcode_counts[Opcode::EXT]
        + opcode_counts[Opcode::SEXT]
        + opcode_counts[Opcode::MADDU]
        + opcode_counts[Opcode::MSUBU]
        + opcode_counts[Opcode::MADD]
        + opcode_counts[Opcode::MSUB]
        + opcode_counts[Opcode::TEQ];

    events_counts[MipsAirId::MovCond] =
        opcode_counts[Opcode::WSBH] + opcode_counts[Opcode::MNE] + opcode_counts[Opcode::MEQ];

    // Compute the number of events in the auipc chip.
    events_counts[MipsAirId::CloClz] = opcode_counts[Opcode::CLO] + opcode_counts[Opcode::CLZ];

    // Compute the number of events in the syscall core chip.
    events_counts[MipsAirId::SyscallCore] = syscalls_sent;

    // Compute the number of events in the global chip.
    events_counts[MipsAirId::Global] = 2 * touched_addresses + syscalls_sent;

    // No dependency rows: every instruction chip proves its own
    // sub-operations in-row (the Instruction bus is gone).

    events_counts
}

/// Maps an opcode to the core AIR that charges it a main-trace row, mirroring the grouping
/// performed by [`estimate_mips_event_counts`].
///
/// This is the lookup that turns the periodic O(chips) re-estimate into an O(1)
/// per-instruction accumulate.
///
/// Returns `None` for the opcodes that [`estimate_mips_event_counts`] does not attribute to
/// any chip: `SYSCALL` (the `SyscallInstrs` chip is never populated by the estimator),
/// `MOD` / `MODU` (the estimator's `DivRem` count is `DIV + DIVU` only) and `UNIMPL`.
#[must_use]
pub const fn mips_air_id_from_opcode(opcode: Opcode) -> Option<MipsAirId> {
    Some(match opcode {
        Opcode::ADD | Opcode::SUB => MipsAirId::AddSub,
        Opcode::MUL | Opcode::MULT | Opcode::MULTU => MipsAirId::Mul,
        Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR => MipsAirId::Bitwise,
        Opcode::SLL => MipsAirId::ShiftLeft,
        Opcode::SRL | Opcode::SRA | Opcode::ROR => MipsAirId::ShiftRight,
        Opcode::DIV | Opcode::DIVU => MipsAirId::DivRem,
        Opcode::SLT | Opcode::SLTU => MipsAirId::Lt,
        Opcode::BEQ | Opcode::BNE | Opcode::BGTZ | Opcode::BGEZ | Opcode::BLTZ | Opcode::BLEZ => {
            MipsAirId::Branch
        }
        Opcode::Jump | Opcode::Jumpi | Opcode::JumpDirect => MipsAirId::Jump,
        Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU => MipsAirId::LoadNarrow,
        Opcode::LW | Opcode::LL => MipsAirId::LoadWord,
        Opcode::SB | Opcode::SH => MipsAirId::StoreNarrow,
        Opcode::SW | Opcode::SC => MipsAirId::StoreWord,
        Opcode::LWL | Opcode::LWR | Opcode::SWL | Opcode::SWR => MipsAirId::MemoryUnaligned,
        Opcode::INS
        | Opcode::EXT
        | Opcode::SEXT
        | Opcode::MADDU
        | Opcode::MSUBU
        | Opcode::MADD
        | Opcode::MSUB
        | Opcode::TEQ => MipsAirId::MiscInstrs,
        Opcode::WSBH | Opcode::MNE | Opcode::MEQ => MipsAirId::MovCond,
        Opcode::CLO | Opcode::CLZ => MipsAirId::CloClz,
        Opcode::SYSCALL | Opcode::MOD | Opcode::MODU | Opcode::UNIMPL => return None,
    })
}

/// The immediate-form sibling of [`mips_air_id_from_opcode`]: the air an
/// `imm_c` instruction's row lands on, for the opcodes whose chip is split by
/// operand form.  `None` means the opcode has no split — bill via
/// [`mips_air_id_from_opcode`] as before.
#[must_use]
pub const fn mips_imm_air_from_opcode(opcode: Opcode) -> Option<MipsAirId> {
    Some(match opcode {
        Opcode::ADD | Opcode::SUB => MipsAirId::AddSubImm,
        Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR => MipsAirId::BitwiseImm,
        Opcode::SLL => MipsAirId::ShiftLeftImm,
        Opcode::SRL | Opcode::SRA | Opcode::ROR => MipsAirId::ShiftRightImm,
        Opcode::SLT | Opcode::SLTU => MipsAirId::LtImm,
        _ => return None,
    })
}

/// Exact, incrementally-maintained per-shard trace area and tallest-chip height.
///
/// Each event bumps the owning chip's height and adds that chip's width to a running area, so
/// the shard-split test at [`Self::check_shard_limit`] is a pair of comparisons against live
/// state rather than a periodic O(chips) re-estimate.
///
/// It replaces the `SHAPE_CHECK_FREQUENCY`-gated block that called
/// [`estimate_mips_event_counts`] + [`pad_mips_event_counts`], and is *exactly* equal to that
/// estimator's output on every cycle — not an approximation of it. `estimate_mips_event_counts`
/// was already a pure function of `(clk / 5, local_mem, syscalls_sent, opcode counts)`, all of
/// which the executor already maintains incrementally, so the periodic recompute was
/// rebuilding a value it could have carried. Because the state is exact at *every* cycle
/// rather than only at multiples of a check frequency, the worst-case `pad_mips_event_counts`
/// inflation that covered the blind window between two checks is no longer needed.
///
/// The `Cpu` chip is deliberately *not* accumulated here: its row count is `clk / 5` by
/// definition and `clk` is already exact executor state, so it is folded in as a single
/// multiply-add inside [`Self::check_shard_limit`]. That keeps the accumulator's write set to
/// the few sites that already maintain `LocalCounts`, instead of also having to hook every
/// `clk` bump (including the variable-width precompile bumps, which are the easiest to miss).
#[derive(Debug, Clone)]
pub struct ShardSplitAccumulator {
    /// Running `Σ_chip height[chip] × width[chip]` over every chip except `Cpu`.
    trace_area: u64,
    /// Running `max_chip height[chip]` over every chip except `Cpu`.
    max_height: u64,
    /// Per-chip row counts.
    heights: EnumMap<MipsAirId, u64>,
    /// Main-trace width per chip, as an array rather than the `HashMap` the periodic path
    /// hashed into once per chip per check.
    costs: EnumMap<MipsAirId, u64>,
    /// Distinct addresses touched in this shard, i.e. `LocalCounts::local_mem`. Kept here so
    /// the `MemoryLocal` `div_ceil` and the `Global` row count stay O(1).
    touched_addresses: u64,
    /// The trace-area budget for one shard (`ELEMENT_THRESHOLD`).
    element_threshold: u64,
    /// The per-chip row cap for one shard.
    height_threshold: u64,
}

impl ShardSplitAccumulator {
    /// Create an accumulator over the given per-chip main-trace widths and split thresholds.
    #[must_use]
    pub fn new(
        costs: &HashMap<MipsAirId, u64>,
        element_threshold: u64,
        height_threshold: u64,
    ) -> Self {
        let mut cost_map: EnumMap<MipsAirId, u64> = EnumMap::default();
        for (air, &cost) in costs {
            cost_map[*air] = cost;
        }
        let mut this = Self {
            trace_area: 0,
            max_height: 0,
            heights: EnumMap::default(),
            costs: cost_map,
            touched_addresses: 0,
            element_threshold,
            height_threshold,
        };
        this.reset();
        this
    }

    /// Clear all per-shard state. Called at every shard boundary.
    pub fn reset(&mut self) {
        self.heights = EnumMap::default();
        self.trace_area = 0;
        self.max_height = 0;
        self.touched_addresses = 0;
        // The memory-bump chip charges one shadow read per register per shard unconditionally,
        // so it is seeded rather than accumulated (cf. `estimate_mips_event_counts`, which sets
        // `MemoryBump` to `NUM_REGISTERS` regardless of the event counts).
        self.bump(MipsAirId::MemoryBump, NUM_REGISTERS as u64);
    }

    /// Add `count` rows to `air`, keeping `trace_area` and `max_height` in step.
    #[inline]
    fn bump(&mut self, air: MipsAirId, count: u64) {
        let height = &mut self.heights[air];
        *height += count;
        let height = *height;
        if height > self.max_height {
            self.max_height = height;
        }
        self.trace_area += count * self.costs[air];
    }

    /// Charge `count` rows for `opcode`, including the chips it induces rows on indirectly.
    ///
    /// Mirrors the `DivRem` fan-out that [`estimate_mips_event_counts`] applies after the fact
    /// (`Mul += DivRem`, `Lt += DivRem`). The other cross-chip dependencies — the extra
    /// `AddSub` / `Lt` / shift rows an instruction induces — are already expressed as explicit
    /// `Opcode` increments by the executor's bookkeeping block, so they arrive here as ordinary
    /// calls and need no special handling.
    #[inline]
    pub fn add_opcode(&mut self, opcode: Opcode, count: u64) {
        let Some(air) = mips_air_id_from_opcode(opcode) else {
            return;
        };
        self.bump(air, count);
        // No dependency rows: every instruction chip proves its own
        // sub-operations in-row (the Instruction bus is gone).
    }

    /// Charge `count` rows directly to `air`, for the rows the opcode->air map
    /// cannot place: ADD/SUB split into two chips by operand form, which only
    /// the executor's decoded instruction can see.
    #[inline]
    pub fn add_air(&mut self, air: MipsAirId, count: u64) {
        self.bump(air, count);
    }

    /// Record one newly-touched address, charging the `MemoryLocal` and `Global` chips.
    #[inline]
    pub fn add_touched_address(&mut self) {
        // `MemoryLocal` packs `NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC` addresses per row, so a
        // new address opens a row only when the previous ones exactly filled the last one —
        // the incremental form of the estimator's `div_ceil`.
        if self.touched_addresses.is_multiple_of(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64) {
            self.bump(MipsAirId::MemoryLocal, 1);
        }
        self.touched_addresses += 1;
        self.bump(MipsAirId::Global, 2);
    }

    /// Whether this shard has reached its trace-area budget / its per-chip height cap.
    ///
    /// `cpu_cycles` is the executed-cycle count (`clk / 5`); see the type-level note on why
    /// it is passed in rather than accumulated.
    /// With the Cpu chip gone its width is absent from `costs` (EnumMap defaults it
    /// to 0), so the `cpu_cycles` term contributes NO area.  It no longer participates
    /// in the HEIGHT cap either: the cap's job is keeping every REAL chip under the
    /// recursion row cube, and the Cpu pseudo-term only bounded clk — which the
    /// executor's `clk_exit` fence (`CORE_SHARD_CLK_LIMIT`, the width the frame's
    /// 26-bit clk decomposition range-checks) bounds independently and terminally.
    /// Measured Aug24 (block 21M): the pseudo-term closed 43 of 48 shards while the
    /// tallest REAL chip sat at 25% of the cap.  Removing it moves shards to the
    /// AREA fence — see `ELEMENT_THRESHOLD` for the budget that keeps the biggest
    /// shard's LogUp-GKR slab on a 32 GiB card.  Measured on reth (Aug26): 164
    /// shards -> 132, combined 158.7 -> 142.0 s.
    #[inline]
    #[must_use]
    pub fn check_shard_limit(&self, cpu_cycles: u64) -> (bool, bool) {
        let area = self.trace_area + cpu_cycles * self.costs[MipsAirId::Cpu];
        (area >= self.element_threshold, self.max_height >= self.height_threshold)
    }

    /// The live trace area, including the `Cpu` contribution. For diagnostics only.
    #[must_use]
    pub fn trace_area(&self, cpu_cycles: u64) -> u64 {
        self.trace_area + cpu_cycles * self.costs[MipsAirId::Cpu]
    }

    /// The live tallest-chip height, including `Cpu`. For diagnostics only.
    #[must_use]
    pub fn max_height(&self, cpu_cycles: u64) -> u64 {
        if cpu_cycles > self.max_height {
            cpu_cycles
        } else {
            self.max_height
        }
    }

    /// The live per-chip row counts, including `Cpu`. For diagnostics and for cross-checking
    /// against [`estimate_mips_event_counts`].
    #[must_use]
    pub fn event_counts(&self, cpu_cycles: u64) -> EnumMap<MipsAirId, u64> {
        let mut counts = self.heights;
        counts[MipsAirId::Cpu] = cpu_cycles;
        counts
    }

    /// The producer's slot for `air`: its index in the `EnumMap` order (declaration order,
    /// NOT the discriminant), which is what `heights` is laid out in.
    #[must_use]
    pub fn slot(air: MipsAirId) -> usize {
        <MipsAirId as enum_map::Enum>::into_usize(air)
    }

    /// Main-trace width of `air`.
    #[must_use]
    pub fn cost(&self, air: MipsAirId) -> u64 {
        self.costs[air]
    }

    /// Hand the live budgets to the JIT producer as remaining amounts: `area_left =
    /// element_threshold - trace_area`, `height_left[slot(air)] = height_threshold -
    /// heights[air]`, plus the touched-address count. The producer charges by subtracting and
    /// fences at `<= 0`, which is exactly `check_shard_limit`'s `>=`.
    ///
    /// # Panics
    ///
    /// Panics if `height_left` has fewer than `MipsAirId::LENGTH` slots.
    pub fn export_budgets(&self, area_left: &mut i64, height_left: &mut [i64], touched: &mut u64) {
        *area_left = self.element_threshold as i64 - self.trace_area as i64;
        for (air, &height) in &self.heights {
            height_left[Self::slot(air)] = self.height_threshold as i64 - height as i64;
        }
        *touched = self.touched_addresses;
    }

    /// Take the budgets back from the producer (the inverse of [`Self::export_budgets`]).
    ///
    /// # Panics
    ///
    /// Panics if `height_left` has fewer than `MipsAirId::LENGTH` slots.
    pub fn import_budgets(&mut self, area_left: i64, height_left: &[i64], touched: u64) {
        self.trace_area = (self.element_threshold as i64 - area_left) as u64;
        self.max_height = 0;
        for (air, height) in &mut self.heights {
            *height = (self.height_threshold as i64 - height_left[Self::slot(air)]) as u64;
            if *height > self.max_height {
                self.max_height = *height;
            }
        }
        self.touched_addresses = touched;
    }
}

/// Pads the event counts to account for the worst case jump in events across N cycles.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_mips_event_counts(
    mut event_counts: EnumMap<MipsAirId, u64>,
    num_cycles: u64,
) -> EnumMap<MipsAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        MipsAirId::Cpu => *v += num_cycles,
        MipsAirId::AddSub => *v += 5 * num_cycles,
        // Only real immediate-form ADD/SUB instructions land here (the
        // synthetic charges all bill the register-form air), so the growth is
        // bounded by one row per cycle.
        MipsAirId::AddSubImm => *v += num_cycles,
        MipsAirId::BitwiseImm => *v += num_cycles,
        MipsAirId::ShiftLeftImm => *v += num_cycles,
        MipsAirId::ShiftRightImm => *v += num_cycles,
        MipsAirId::LtImm => *v += num_cycles,
        MipsAirId::Mul => *v += 4 * num_cycles,
        MipsAirId::Bitwise => *v += 3 * num_cycles,
        MipsAirId::ShiftLeft => *v += num_cycles,
        MipsAirId::ShiftRight => *v += num_cycles,
        MipsAirId::DivRem => *v += 4 * num_cycles,
        MipsAirId::Lt => *v += 2 * num_cycles,
        MipsAirId::MemoryLocal => *v += 64 * num_cycles,
        // Bounded by the register count, not by the cycle count.
        MipsAirId::MemoryBump => *v += NUM_REGISTERS as u64,
        MipsAirId::Branch => *v += 8 * num_cycles,
        MipsAirId::Jump => *v += 2 * num_cycles,
        MipsAirId::SyscallInstrs => *v += num_cycles,
        MipsAirId::LoadNarrow => *v += 8 * num_cycles,
        MipsAirId::LoadWord => *v += 8 * num_cycles,
        MipsAirId::StoreNarrow => *v += 8 * num_cycles,
        MipsAirId::StoreWord => *v += 8 * num_cycles,
        MipsAirId::MemoryUnaligned => *v += 8 * num_cycles,
        MipsAirId::MiscInstrs => *v += 8 * num_cycles, // TODO: Check this value.
        MipsAirId::CloClz => *v += 3 * num_cycles,     // TODO: Check this value.
        MipsAirId::SyscallCore => *v += 2 * num_cycles,
        MipsAirId::Global => *v += 64 * num_cycles,
        _ => (),
    });
    event_counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mips_costs;
    use enum_map::Enum;

    /// Every opcode the MIPS executor can retire, so the equivalence test below covers the
    /// whole `Opcode` -> `MipsAirId` map rather than a hand-picked subset.
    fn all_opcodes() -> Vec<Opcode> {
        (0..Opcode::LENGTH).map(Opcode::from_usize).collect()
    }

    fn costs() -> HashMap<MipsAirId, u64> {
        mips_costs().into_iter().map(|(k, v)| (k, v as u64)).collect()
    }

    /// The incremental accumulator must agree with [`estimate_mips_event_counts`] EXACTLY, for
    /// every chip, at every point in the stream — not just at the end and not approximately.
    ///
    /// This is the property the whole `SHAPE_CHECK_FREQUENCY` removal rests on: the periodic
    /// re-estimate could be dropped precisely because the accumulator reproduces it, so if this
    /// ever fails, shard boundaries have silently moved.
    #[test]
    fn accumulator_matches_estimator_over_a_mixed_opcode_stream() {
        let costs = costs();
        let opcodes = all_opcodes();
        let mut acc = ShardSplitAccumulator::new(&costs, u64::MAX, u64::MAX);
        let mut reference: EnumMap<Opcode, u64> = EnumMap::default();
        let mut touched: u64 = 0;

        // A deterministic, badly-behaved stream: opcode choice and address-touch decisions are
        // driven by a cheap LCG so the counts land on every `div_ceil` boundary of the
        // `MemoryLocal` packing rather than only on multiples of it.
        let mut rng: u64 = 0x243f_6a88_85a3_08d3;
        for step in 0..20_000u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let opcode = opcodes[(rng >> 33) as usize % opcodes.len()];
            let count = 1 + (rng >> 17) % 3;

            acc.add_opcode(opcode, count);
            reference[opcode] += count;

            if rng % 3 == 0 {
                acc.add_touched_address();
                touched += 1;
            }

            // `Cpu` is derived rather than accumulated, so sweep it too.
            let cpu_cycles = step * 5 / 5;
            let expected = estimate_mips_event_counts(cpu_cycles, touched, 0, reference);
            assert_eq!(
                acc.event_counts(cpu_cycles),
                expected,
                "per-chip counts diverged at step {step} (opcode {opcode:?})"
            );

            let expected_area: u64 =
                expected.iter().map(|(air, &n)| n * costs.get(&air).copied().unwrap_or(0)).sum();
            assert_eq!(acc.trace_area(cpu_cycles), expected_area, "area diverged at step {step}");

            let expected_max = expected.iter().map(|(_, &h)| h).max().unwrap_or(0);
            assert_eq!(acc.max_height(cpu_cycles), expected_max, "height diverged at step {step}");
        }
    }

    /// A reset must restore exactly the state a fresh accumulator starts in, including the
    /// unconditional per-shard `MemoryBump` seed. A reset that dropped the seed would understate
    /// every shard's area by `NUM_REGISTERS * width(MemoryBump)`.
    #[test]
    fn reset_restores_a_fresh_shard() {
        let costs = costs();
        let mut acc = ShardSplitAccumulator::new(&costs, u64::MAX, u64::MAX);
        let fresh = acc.event_counts(0);

        for opcode in all_opcodes() {
            acc.add_opcode(opcode, 7);
        }
        for _ in 0..37 {
            acc.add_touched_address();
        }
        assert_ne!(acc.event_counts(0), fresh);

        acc.reset();
        assert_eq!(acc.event_counts(0), fresh);
        assert_eq!(acc.event_counts(0), estimate_mips_event_counts(0, 0, 0, EnumMap::default()));
    }

    /// The split test must fire exactly at the threshold, on whichever limit is reached first.
    #[test]
    fn check_shard_limit_fires_on_either_budget() {
        let costs = costs();
        let add_sub_width = costs[&MipsAirId::AddSub];
        // A fresh shard is not empty: it already carries the unconditional `MemoryBump` seed.
        let seed_area = ShardSplitAccumulator::new(&costs, u64::MAX, u64::MAX).trace_area(0);

        // Area budget sized so the second AddSub row is exactly what crosses it
        // (the Cpu chip is gone: cycles contribute no area, only height).
        let mut acc = ShardSplitAccumulator::new(&costs, seed_area + add_sub_width * 2, u64::MAX);
        acc.add_opcode(Opcode::ADD, 1);
        assert_eq!(acc.check_shard_limit(1), (false, false));
        acc.add_opcode(Opcode::ADD, 1);
        assert_eq!(acc.check_shard_limit(2), (true, false));

        // Height budget, independent of area — and of CYCLES: the Cpu pseudo-term
        // is gone (clk is bounded by the executor's terminal `clk_exit` fence),
        // so only a REAL chip's rows can trip the height half.
        let mut acc = ShardSplitAccumulator::new(&costs, u64::MAX, 100);
        assert_eq!(acc.check_shard_limit(1_000_000), (false, false));
        for _ in 0..100 {
            acc.add_opcode(Opcode::ADD, 1);
        }
        assert_eq!(acc.check_shard_limit(1_000_000), (false, true));
    }
}
