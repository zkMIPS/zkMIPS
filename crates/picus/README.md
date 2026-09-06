# Determinism extraction (Picus + Lean 4)

`zkm-picus` turns every chip of the MIPS machine into a set of *determinism modules* and writes
them as

- `*.picus` programs for the Picus solver, and
- Lean 4 files whose theorems state the same obligations, to be checked (and proved) in Lean.

A module is deterministic when its outputs are a function of its inputs.  For an AIR that means:
given everything a row receives from other tables, the values it produces for other tables are
forced by its constraints.

## Pipeline

1. `MipsAir::<KoalaBear>::chips()` supplies the chips; `--chip NAME` (repeatable), `--all`, or
   `--list`.
2. [`PicusBuilder`](src/picus_builder.rs) evaluates the chip's AIR once.  Its `Expr` type is
   Plonky3's `SymbolicExpression<KoalaBear>`, so the builder is an ordinary `AirBuilder` +
   `MessageBuilder` that records `assert_zero` polynomials and `send` / `receive` lookups.
3. [`lower`](src/lower.rs) turns the recorded symbolic trees into Picus expressions over a fixed
   variable numbering (main column `i` is variable `i`), memoizing shared sub-trees and binding
   sub-trees larger than `--reify-threshold` to fresh variables.
4. The interface is derived from the lookups (see the table in `picus_builder.rs`): program
   fetches, memory reads, the received CPU / precompile state and syscall arguments are inputs;
   memory writes, the sent state, syscall calls and global sends are outputs.  Byte-table lookups
   become range / bit constraints or calls to abstract helper modules (`byte_and`, …).
5. One module is emitted per `#[picus(selector)]` column, specialized with that selector at one,
   the others at zero and `is_real` at one (`Chip__is_xxx`); a chip without selectors gets one
   module.  A `top` module proves the selector shape (boolean, mutually exclusive, or a partition
   of the real rows when the chip declares `selectors_partition_real_rows`).
6. `--format picus|lean|both` writes `<picus-out-dir>/<Chip>.picus` and
   `<lean-out-dir>/ZirenDet/Chips/<Chip>.lean` (plus `ZirenDet.lean` and the prelude
   `ZirenDet/Basic.lean`).

Every chip is a single-row AIR today (cross-row sequencing lives on lookup buses: `State`,
`GlobalAccumulation`, `PrecompileChain`, …), so one extraction phase suffices; the old
`FirstRow` / `Transition` / `LastRow` phases and the Instruction-bus opcode routing are gone.

## Annotations

Annotations are metadata on the column struct; they never change the AIR.

```rust
use zkm_derive::{AlignedBorrow, PicusAnnotations};
use zkm_pcs::PicusInfo;

#[derive(AlignedBorrow, PicusAnnotations, Default, Clone, Copy)]
#[repr(C)]
pub struct AddSubCols<T> {
    #[picus(input)]
    pub pc: T,
    pub next_pc: T,
    #[picus(selector)]
    pub is_add: T,
    #[picus(selector)]
    pub is_sub: T,
    pub frame: RTypeFrameCols<T>,
}
```

and on the chip:

```rust
fn picus_info(&self) -> PicusInfo {
    AddSubCols::<u8>::picus_info()
}
```

- `#[picus(selector)]` — an opcode / row-type flag; one specialized module per selector.
- `#[picus(input)]` / `#[picus(output)]` — add a column to the interface on top of what the
  lookups imply.  Most chips need none: the interface is inferred.
- A field named `is_real` is detected automatically and specialized to one.
- `#[picus(transition_input)]` / `#[picus(transition_output)]` are accepted for compatibility
  with the multi-row history; with single-row chips they have no effect.
- `#[derive(PicusProjection)]` describes a semantic slice of a larger witness layout for
  operation summaries (see `PicusProjectionInfo`); no chip uses it after the sub-AIRs were
  inlined.

Chip-level hooks on `MachineAir`: `selectors_partition_real_rows()` (stronger `top` contract) and
`picus_selector_specialization_allowed(name)` (skip impossible selector values).

## Usage

```bash
cargo run -p zkm-picus -- --list
cargo run -p zkm-picus -- --chip AddSub --picus-out-dir picus_out --lean-out-dir lean/ZirenDet
cargo run -p zkm-picus -- --all --format lean --lean-out-dir lean/ZirenDet
```

Options: `--assume-selectors-deterministic`, `--shrcarry-summary abstract|precise`,
`--column-output-mode interactions-only|all-non-inputs-are-outputs`, `--reify-threshold N`
(0 disables), `--keep-padding` (do not specialize `is_real = 1`).

## Lean

`lean/ZirenDet` is a Lake project pinned to Mathlib `v4.33.1`.  For every module `M` the
generated file contains a witness structure `M.W`, `M.constraints`, `M.inputs`, `M.outputs`,
`M.assumed`, the relation `M.rel`, and

```lean
theorem M.deterministic (h_Aux : ∀ i o o', Aux.rel i o → Aux.rel i o' → o = o') …
    (w w' : W) (hw : constraints w) (hw' : constraints w')
    (hin : inputs w = inputs w') (hassume : assumed w = assumed w') :
    outputs w = outputs w'
```

Abstract helper modules (byte-table operations) have an `opaque rel`; their determinism is a
hypothesis of the callers, never an axiom.  The closing tactic `picus_det` tries the cheap
closers and otherwise leaves `sorry`, so files always elaborate and the open obligations are
the `declaration uses 'sorry'` warnings.

Build on the GPU box (never on the dev host):

```bash
rsync -a --exclude .lake lean/ZirenDet/ ant-5090-2:/mnt_zkm/stephen/lean/ZirenDet/
ssh ant-5090-2 'export ELAN_HOME=/mnt_zkm/stephen/.elan PATH=/mnt_zkm/stephen/.elan/bin:$PATH \
  XDG_CACHE_HOME=/mnt_zkm/stephen/.cache; cd /mnt_zkm/stephen/lean/ZirenDet && lake build'
```

## Adding a chip

1. Make sure `MipsAir::chips()` includes it and `name()` is stable.
2. Derive `PicusAnnotations` on its column struct, mark selectors, add `picus_info()`.
3. Run the extractor on it and read the module interface in the `.picus` file (inputs first,
   then outputs); if a value the chip is responsible for is missing, annotate it as
   `#[picus(output)]`.
4. If the AIR uses a new lookup kind, add its direction to `Emitter::handle_lookup`.
