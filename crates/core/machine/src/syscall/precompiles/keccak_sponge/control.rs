//! Keccak-sponge **control** chip — the sponge layer + the two `PrecompileChain`
//! bus endpoints, one row per sponge block.
//!
//! The [`super::KeccakSpongeChip`] worker only evaluates the bare keccak-f
//! permutation (one round/row, round-chained on bus P).  Everything the legacy
//! multi-row chip did with `next.*`/`when_*` selectors — the multi-block sponge
//! (memory reads of each input block, the rate absorb XOR, the block-to-block
//! state hand-off, the output squeeze + write, the syscall receive) — lives here
//! instead, with the block-to-block state carried on bus B.
//!
//! Per block `b` (state laid out as 50 `u32` words / 100 16-bit limbs):
//!   * receive the syscall once (`is_first_block`);
//!   * read this block's `rate` input words + (first block) the input length;
//!   * absorb: `xored[w] = original_state[w] XOR input[w]` over the rate;
//!   * **bus P**: SEND the absorbed state `@ (b, round 0)` (seed) and RECEIVE the
//!     permuted state `@ (b, round 24)` (drain);
//!   * **bus B**: RECEIVE `original_state @ b` (= prev block's permuted state;
//!     forced to 0 on the first block) and SEND the permuted state `@ b+1`;
//!   * final block: write the first [`KECCAK_GENERAL_OUTPUT_U32S`] permuted words.
//!
//! All sponge state is held in byte (`Word`) form for the absorb/memory; the
//! byte→limb conversion (`limb = b0 + b1·256`, lane = `5·y + x`) happens only
//! when building the bus tuples, matching the worker's `p3_keccak` limb layout.

use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;
use zkm_derive::PicusAnnotations;
use zkm_pcs::PicusInfo;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_keccak_air::U64_LIMBS;
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_pcs::{
    air::{AirLookup, LookupScope, MachineAir},
    LookupKind, Word, ZKMAirBuilder,
};

use crate::air::{MemoryAirBuilder, WordAirBuilder};
use crate::memory::{MemoryCols, MemoryReadCols, MemoryWriteCols};
use crate::operations::XorOperation;
use crate::syscall::precompiles::keccak_sponge::air::{KECCAK_BUS_BLOCK, KECCAK_BUS_ROUND};
use crate::syscall::precompiles::keccak_sponge::utils::keccakf_u32s;
use crate::syscall::precompiles::keccak_sponge::{
    KECCAK_GENERAL_OUTPUT_U32S, KECCAK_GENERAL_RATE_U32S, KECCAK_STATE_U32S,
};
use crate::{utils::pad_rows_fixed, CoreChipError};

pub const NUM_KECCAK_SPONGE_CONTROL_COLS: usize = size_of::<KeccakSpongeControlCols<u8>>();

#[derive(PicusAnnotations, AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct KeccakSpongeControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub is_real: T,
    pub block: T,
    pub is_first_block: T,
    pub is_final_block: T,
    /// `is_real ∧ ¬is_first_block` (bus B receive selector).
    pub do_block_recv: T,
    /// `is_real ∧ ¬is_final_block` (bus B send selector).
    pub do_block_send: T,
    pub input_address: T,
    pub output_address: T,
    /// `S_{b-1}`: the pre-absorb state (= 0 on the first block).
    pub original_state: [Word<T>; KECCAK_STATE_U32S],
    /// `S_b`: the post-permutation state (received on bus P @ round 24).
    pub recv_state: [Word<T>; KECCAK_STATE_U32S],
    /// `original_state[rate] XOR input` (the absorb).
    pub xored_general_rate: [XorOperation<T>; KECCAK_GENERAL_RATE_U32S],
    pub block_mem: [MemoryReadCols<T>; KECCAK_GENERAL_RATE_U32S],
    pub input_length_mem: MemoryReadCols<T>,
    pub output_mem: [MemoryWriteCols<T>; KECCAK_GENERAL_OUTPUT_U32S],
}

/// Keccak-sponge control chip.  One row per sponge block.
#[derive(Default)]
pub struct KeccakSpongeControlChip;

impl KeccakSpongeControlChip {
    pub const fn new() -> Self {
        Self {}
    }

    fn event_to_rows<F: PrimeField32>(
        &self,
        event: &zkm_core_executor::events::KeccakSpongeEvent,
        rows: &mut Option<Vec<[F; NUM_KECCAK_SPONGE_CONTROL_COLS]>>,
        blu: &mut impl ByteRecord,
    ) {
        let block_num = event.num_blocks();
        // `state_u32s` tracks the running sponge state in word form: it is the
        // pre-absorb state at the start of each block and the post-permute state
        // after `keccakf_u32s`.
        let mut state_u32s = [0u32; KECCAK_STATE_U32S];

        for b in 0..block_num {
            let mut row = [F::ZERO; NUM_KECCAK_SPONGE_CONTROL_COLS];
            let cols: &mut KeccakSpongeControlCols<F> = row.as_mut_slice().borrow_mut();

            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.is_real = F::ONE;
            cols.block = F::from_canonical_u32(b as u32);
            cols.is_first_block = F::from_bool(b == 0);
            cols.is_final_block = F::from_bool(b == block_num - 1);
            cols.do_block_recv = F::from_bool(b != 0);
            cols.do_block_send = F::from_bool(b != block_num - 1);
            cols.output_address = F::from_canonical_u32(event.output_addr);
            cols.input_address = F::from_canonical_u32(
                event.input_addr + b as u32 * KECCAK_GENERAL_RATE_U32S as u32 * 4,
            );

            // original_state = S_{b-1} (word form).
            for j in 0..KECCAK_STATE_U32S {
                cols.original_state[j] = Word::from(state_u32s[j]);
            }

            // Read this block's `rate` input words + absorb.
            for j in 0..KECCAK_GENERAL_RATE_U32S {
                cols.block_mem[j]
                    .populate(event.input_read_records[b * KECCAK_GENERAL_RATE_U32S + j], blu);
                let xored = cols.xored_general_rate[j].populate(
                    blu,
                    state_u32s[j],
                    event.input[b * KECCAK_GENERAL_RATE_U32S + j],
                );
                state_u32s[j] = xored;
            }

            // First block: read the input length.
            if b == 0 {
                cols.input_length_mem.populate(event.input_length_record, blu);
            }

            // Permute: state_u32s becomes S_b.
            keccakf_u32s(&mut state_u32s);
            for j in 0..KECCAK_STATE_U32S {
                cols.recv_state[j] = Word::from(state_u32s[j]);
            }

            // Final block: write the squeezed output.
            if b == block_num - 1 {
                for j in 0..KECCAK_GENERAL_OUTPUT_U32S {
                    cols.output_mem[j].populate(event.output_write_records[j], blu);
                }
            }

            if let Some(rows) = rows.as_mut() {
                rows.push(row);
            }
        }
    }
}

impl<F> BaseAir<F> for KeccakSpongeControlChip {
    fn width(&self) -> usize {
        NUM_KECCAK_SPONGE_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for KeccakSpongeControlChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "KeccakSpongeControl".to_string()
    }

    fn picus_info(&self) -> PicusInfo {
        KeccakSpongeControlCols::<u8>::picus_info()
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let mut blu: Vec<ByteLookupEvent> = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::KECCAK_SPONGE) {
            let event = if let PrecompileEvent::KeccakSponge(event) = event {
                event
            } else {
                unreachable!()
            };
            self.event_to_rows::<F>(event, &mut None, &mut blu);
        }
        output.add_byte_lookup_events(blu);
        Ok(())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows: Vec<[F; NUM_KECCAK_SPONGE_CONTROL_COLS]> = Vec::new();
        let mut wrapped = Some(rows);
        let mut blu: Vec<ByteLookupEvent> = Vec::new();
        for (_, event) in input.get_precompile_events(SyscallCode::KECCAK_SPONGE) {
            let event = if let PrecompileEvent::KeccakSponge(event) = event {
                event
            } else {
                unreachable!()
            };
            self.event_to_rows::<F>(event, &mut wrapped, &mut blu);
        }
        rows = wrapped.unwrap();

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_KECCAK_SPONGE_CONTROL_COLS],
            input.fixed_log2_rows::<F, _>(self),
            <KeccakSpongeControlChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_KECCAK_SPONGE_CONTROL_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_SPONGE).is_empty()
        }
    }
}

impl<AB> Air<AB> for KeccakSpongeControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let local: &KeccakSpongeControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_first_block);
        builder.assert_bool(local.is_final_block);
        // Bus-B selectors.
        builder
            .assert_eq(local.do_block_recv, local.is_real * (AB::Expr::ONE - local.is_first_block));
        builder
            .assert_eq(local.do_block_send, local.is_real * (AB::Expr::ONE - local.is_final_block));

        // Receive the syscall once (first block).
        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_u32(SyscallCode::KECCAK_SPONGE.syscall_id()),
            local.input_address,
            local.output_address,
            local.is_first_block,
            LookupScope::Local,
        );

        // Memory: input-length read (first block), rate-word reads (every block),
        // output writes (final block).
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.output_address + AB::Expr::from_u32(64),
            &local.input_length_mem,
            local.is_first_block,
        );
        for w in 0..KECCAK_GENERAL_RATE_U32S {
            builder.eval_memory_access(
                local.shard,
                local.clk,
                local.input_address + AB::Expr::from_u32(w as u32 * 4),
                &local.block_mem[w],
                local.is_real,
            );
        }
        for j in 0..KECCAK_GENERAL_OUTPUT_U32S {
            builder.eval_memory_access(
                local.shard,
                local.clk + AB::Expr::ONE,
                local.output_address + AB::Expr::from_u32(j as u32 * 4),
                &local.output_mem[j],
                local.is_final_block,
            );
        }

        // The first block's pre-absorb state is zero.
        for w in 0..KECCAK_STATE_U32S {
            for k in 0..4 {
                builder.when(local.is_first_block).assert_zero(local.original_state[w].0[k]);
            }
        }

        // Absorb: xored[w] = original_state[w] XOR input[w] over the rate.
        for w in 0..KECCAK_GENERAL_RATE_U32S {
            XorOperation::<AB::F>::eval(
                builder,
                local.original_state[w],
                local.block_mem[w].access.value,
                local.xored_general_rate[w],
                local.is_real,
            );
        }

        // Final block: the squeezed output words equal the permuted state.
        for j in 0..KECCAK_GENERAL_OUTPUT_U32S {
            builder
                .when(local.is_final_block)
                .assert_word_eq(*local.output_mem[j].value(), local.recv_state[j]);
        }

        self.eval_state_buses(builder, local);
    }
}

impl KeccakSpongeControlChip {
    /// Convert a 50-word state (given by a word→bytes accessor) into the
    /// 100-element 16-bit-limb bus payload, in `p3_keccak` `(y, x, limb)` order
    /// (lane = `5·y + x`; each `u64` lane = words `2·lane` / `2·lane+1`).
    fn state_limbs<AB: ZKMAirBuilder>(
        word_bytes: impl Fn(usize) -> [AB::Expr; 4],
    ) -> Vec<AB::Expr> {
        let shift = AB::Expr::from_u32(256);
        let mut limbs = Vec::with_capacity(5 * 5 * U64_LIMBS);
        for y in 0..5 {
            for x in 0..5 {
                let lane = 5 * y + x;
                let lo = word_bytes(2 * lane);
                let hi = word_bytes(2 * lane + 1);
                limbs.push(lo[0].clone() + lo[1].clone() * shift.clone());
                limbs.push(lo[2].clone() + lo[3].clone() * shift.clone());
                limbs.push(hi[0].clone() + hi[1].clone() * shift.clone());
                limbs.push(hi[2].clone() + hi[3].clone() * shift.clone());
            }
        }
        limbs
    }

    fn eval_state_buses<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakSpongeControlCols<AB::Var>,
    ) {
        let pid = AB::Expr::from_u32(SyscallCode::KECCAK_SPONGE.syscall_id());

        let orig = |w: usize| local.original_state[w].0.map(Into::into);
        let recv = |w: usize| local.recv_state[w].0.map(Into::into);
        // The absorbed state S_b': rate words = xored, capacity words = original.
        let absorbed = |w: usize| -> [AB::Expr; 4] {
            if w < KECCAK_GENERAL_RATE_U32S {
                local.xored_general_rate[w].value.0.map(Into::into)
            } else {
                local.original_state[w].0.map(Into::into)
            }
        };

        // --- Bus P (round chain): seed @ round 0, drain @ round 24. ---
        let p_header = |round: u32| -> Vec<AB::Expr> {
            vec![
                pid.clone(),
                AB::Expr::from_u32(KECCAK_BUS_ROUND),
                local.clk.into(),
                local.block.into(),
                AB::Expr::from_u32(round),
            ]
        };
        let mut p_send = p_header(0);
        p_send.extend(Self::state_limbs::<AB>(absorbed));
        builder.send(
            AirLookup::new(p_send, local.is_real.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );
        let mut p_recv = p_header(24);
        p_recv.extend(Self::state_limbs::<AB>(recv));
        builder.receive(
            AirLookup::new(p_recv, local.is_real.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );

        // --- Bus B (block chain): receive original @ block, send permuted @ block+1. ---
        let b_header = |block: AB::Expr| -> Vec<AB::Expr> {
            vec![pid.clone(), AB::Expr::from_u32(KECCAK_BUS_BLOCK), local.clk.into(), block]
        };
        let mut b_recv = b_header(local.block.into());
        b_recv.extend(Self::state_limbs::<AB>(orig));
        builder.receive(
            AirLookup::new(b_recv, local.do_block_recv.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );
        let mut b_send = b_header(local.block.into() + AB::Expr::ONE);
        b_send.extend(Self::state_limbs::<AB>(recv));
        builder.send(
            AirLookup::new(b_send, local.do_block_send.into(), LookupKind::PrecompileChain),
            LookupScope::Local,
        );
    }
}
