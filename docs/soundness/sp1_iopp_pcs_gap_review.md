# Ziren vs SP1 Hypercube: IOPP and PCS gap review

Date: 2026-09-06. Ziren at `feat/upgrade-plonky3` 49977616; SP1 at `main` 7ea83d0 (v6.3.1,
`/data/stephen/sp1-latest`, includes the `slop` workspace). Every claim cites a file that was
read for this review; measured numbers come from the Sep 5-6 sessions recorded in the memory
files named in section 5 and are marked *(measured)*. Items that could not be verified are
listed in section 4 rather than guessed.

Both stacks are the same protocol family. Ziren's shard prover is a port of Hypercube's
transcript (LogUp-GKR → zerocheck → jagged PCS, `crates/pcs/src/shard_level/prover.rs:1-2`,
`crates/pcs/src/shard_level/verifier.rs:1-3`) with one structural swap: the inner dense PCS is
WHIR instead of BaseFold (`crates/pcs/src/config.rs:190-198`, `crates/pcs/src/whir/mod.rs:1-8`).
The gap is therefore mostly parameters, machine size and engineering, not protocol design.

## 1. Side-by-side

| item | Ziren (49977616) | SP1 Hypercube (7ea83d0) |
|---|---|---|
| field / extension | KoalaBear, degree-4 extension (`docs/soundness/ziren.soundcalc.toml` `field`) | KoalaBear, degree-4 (`crates/primitives/src/lib.rs:28,31`) |
| hash / digest | Poseidon2 width 16, 8+13 rounds, 8-felt digest (`crates/pcs/src/kb31_poseidon2.rs:10,68-69`) | Poseidon2 over KoalaBear (`SP1GlobalContext`; permutation constants not re-read) |
| commitment | jagged over stacked stripes, `2^21` rows × 32 polys per stripe, two commit rounds (preprocessed, main) (`crates/pcs/src/jagged_pcs.rs:78,84`, `crates/pcs/src/whir/stacked.rs:1-19`) | jagged over stacked, `CORE_LOG_STACKING_HEIGHT 21`, 2 commitments (`crates/prover/src/components.rs:16`, `crates/hypercube/src/verifier/shard.rs:38`) |
| inner IOPP (core, compress, shrink) | **WHIR**, folds `[3,6,6]` from 21 variables, rate 1/4 → 1/32 → 1/256 (`crates/pcs/src/whir/jagged.rs:141-157`) | **BaseFold/FRI**, fold arity 2 per round (`fold_even_odd`, `slop/crates/basefold-prover/src/fri.rs:11,118`), one Merkle commitment per variable (`fri.rs:108`, `slop/crates/basefold/src/verifier.rs:318-340`), rate 1/4 (`crates/primitives/src/fri_params.rs:5-6`) |
| WHIR availability | production path (`WHIR_INNER_PCS = true`, `kb31_poseidon2.rs:345`) | implemented (`slop/crates/whir`, 2,332 lines; `hypercube/src/verifier/shard.rs:778 from_config`; `recursion/circuit/src/basefold/whir.rs`, 951 lines) but **not wired**: every stage builds `from_basefold_parameters` (`components.rs:62,81,96,113`) and the recursion proof type is `RecursiveBasefoldProof` (`recursion/circuit/src/shard.rs:44`) |
| queries / PoW, inner | `[71, 51, 49]` per round, query PoW 16 per round, 2 OOD samples, final PoW 16 (`whir/jagged.rs:159-170`) | 124 queries, PoW 16: `ceil((100-16)/-log2(0.625))` (`fri_params.rs:9-14,44-57`) |
| queries / PoW, wrap | BaseFold `(log_blowup 3, 94, 22)` (`crates/pcs/src/basefold/config.rs:203-223`) | shrink and wrap `(3, 94, 22)`: `ceil((100-22)/-log2(0.5625))` (`fri_params.rs:17-42`) |
| stated security target | **64 bits, unique-decoding regime** (soundcalc report `docs/soundness/ziren.soundcalc-report.md`, "Final bits of security 64"); Johnson bound 65 *(measured, Sep 5)* | **100 bits, unique decoding** (`fri_params.rs:44 SP1_TARGET_BITS_OF_SECURITY = 100`, comment at line 54); no in-tree soundness report (section 4) |
| LogUp-GKR grind | `GKR_GRINDING_BITS = 0` (`crates/pcs/src/logup_gkr.rs:24`) | `GKR_GRINDING_BITS = 12` (`hypercube/src/verifier/shard.rs:41`) |
| GKR layout | row-only backend: halve rows to one row, then interactions; layer transitions pair adjacent rows (LSB) (`crates/pcs/src/shard_level/row_gkr/mod.rs:1-7`, `row_gkr/transition.rs:10-14`) | row layers then an interaction layer when `num_row_variables == 1` (`hypercube/src/logup_gkr/logup_poly.rs` `PolynomialLayer`), `initial_number_of_variables == num_interaction_variables + 1` (`logup_gkr/prover.rs:159`) |
| zerocheck | per-chip lazy `ZeroCheckPoly`, λ-RLC of chips, eq-weighted, evaluation points 0, 2, 4 (`crates/pcs/src/shard_level/zerocheck_poly.rs:1-14,826`) | same shape: λ-RLC, α for constraints, points 0, 2, 4, degree ≤ 3 with a TODO for flexibility (`hypercube/src/verifier/shard.rs:304-366`, `prover/zerocheck/sum_as_poly.rs:184-186,289`) |
| constraint degree | core/compress 3, **wrap 9** (soundcalc toml `air_max_degree`, `WRAP_DEGREE` per the toml comment) | compress/shrink/wrap 3 (`components.rs:22-24`) |
| jagged evaluation | HR18 branching program via a `2(log_m+1)`-variable sumcheck (`crates/pcs/src/jagged_branching_program.rs:11-20`, `jagged_eval_sumcheck.rs:1-28`) | same HR18 branching program and sumcheck (`slop/crates/jagged/src/poly.rs:6-22`, `jagged/src/jagged_eval/sumcheck_eval.rs:46,184`) |
| core shard shape | rows ≤ `2^22`, stacking `2^21`, area cap `ELEMENT_THRESHOLD 460,000,000` (`crates/pcs/src/opts.rs:13,173`) | `CORE_MAX_LOG_ROW_COUNT 22`, stacking 21 (`components.rs:16-17`), `HEIGHT_THRESHOLD 2^22` (`crates/core/executor/src/opts.rs:14`) |
| recursion shard shape | compress dense length `2^21`, band height `2^20` (soundcalc toml `compress`), reduce arity 4 (`crates/prover/src/lib.rs:193`) | compress stacking 20 / rows 21 (`crates/verifier/src/compressed/config.rs:1-2`), shrink 18 / 19 (`components.rs:37-38`), `RECURSION_LOG_TRACE_AREA 27`, `SHRINK_LOG_TRACE_AREA 25` (`components.rs:33-35`); reduce arity constant not found (section 4) |
| machine width (core) | 64 chips, 36,489 main columns, 23,555 constraints, 22,637 interactions, widest lookup tuple 105 *(soundcalc census, Sep 5)* | 3,741 columns, 1,911 interactions *(same census tool, Sep 5; not re-run here)* |
| in-circuit verifier | 25,604 lines in `crates/recursion/circuit/src` (`whir_circuit.rs` 974, `basefold_verifier.rs` 1,801, `logup_gkr.rs` 1,108, `zerocheck.rs` 1,021) | 9,809 lines in `crates/recursion/circuit/src` (`basefold/mod.rs` 660, `basefold/whir.rs` 951, `jagged/` 892, `zerocheck.rs` 436, `logup_gkr.rs` 377) |
| recursion leaf area | first leaves `2^26` to `2.5·2^25` cells at ET 460 M *(measured Aug 27)* | bound `2^27` cells (`RECURSION_LOG_TRACE_AREA 27`, `components.rs:33`) |
| compress proof size | **617,618 B** on the UDR-64 schedule; 79% of the earlier 858 KB proof was round-0 WHIR openings *(measured Sep 5)* | not found in tree (section 4) |
| soundness tooling | `docs/soundness/ziren.soundcalc.toml` + report + `soundcalc_census` bin | only the query formula in `fri_params.rs`; audits under `audits/` (8 files) |

## 2. Where Ziren is behind, and where it is ahead

### Behind

**B1. Security target: 64 provable bits versus SP1's 100.** SP1 derives every query count from
`SP1_TARGET_BITS_OF_SECURITY = 100` under the unique-decoding bound (`fri_params.rs:44-57`).
Ziren chose 64 bits on Sep 5 (`whir/jagged.rs:134-140`; soundcalc report "64 bits"). The
mechanism is the WHIR query schedule `[71,51,49]` + PoW 16; the earlier UDR-100 schedule
`[124,93,85]` cost +16-17% wall at every card count and +64% proof bytes *(measured Sep 5)*.
Under the Johnson bound both schedules cap at 65 bits because the fold terms bind (soundcalc
report), so the only way to a provable 100 is more queries (or more PoW) in every round.
Size of the gap: 36 bits of provable soundness; closing it at today's structure is ~+16% wall.

**B2. LogUp-GKR grind 0 versus 12, on a 12x larger interaction set.** Ziren removed the GKR
grind (`logup_gkr.rs:24`, −5.8% wall *(measured Sep 5)*); SP1 grinds 12 bits
(`shard.rs:41`). Ziren's LogUp-GKR term is 84 bits on core with M = 22,637 interactions
(soundcalc report), which does not bind at the 64-bit target but would be the first term to
bind on the way to 100 (M is 12x SP1's 1,911, i.e. about 3.6 bits worse before grinding).

**B3. Machine width: 36,489 columns and 22,637 interactions versus 3,741 and 1,911.** This is
the largest engineering gap and it feeds every IOPP cost: commitment bytes, GKR terms
(7.76 G terms per block, 1 interaction ≈ 2.1 column-equivalents *(measured Sep 5)*), and the
recursion leaf's round-0 opening work (leaf size ∝ stripes × 2^ff0 with stripes = area / 2^21,
`whir/jagged.rs:125-133`). The precompile chips carry most of it: `Bls12381DoubleAssign`
2,430 columns, `U256XU2048Mul` 2,773, `KeccakSponge` 2,637, `Secp256k1DoubleAssign` 1,614
(extractor census of the same commit, `crates/picus --list`). SP1's core stays under 4 K
columns because precompile shards are separate and narrower.

**B4. Compress stacking height.** Ziren stacks compress at `2^21` like core (soundcalc toml
`compress.dense_length`); SP1 drops to 20 for compress and 18 for shrink
(`compressed/config.rs:1`, `components.rs:37`). A taller stack means fewer, wider stripes:
round-0 leaves in the compress proof already open 40 and 56 stripes per query *(measured
Sep 5)*. Whether 20 or 22 is better for Ziren is a measurement, not a given: fewer stripes
shrink each leaf, more stripes reduce padding waste.

**B5. Wrap constraint degree 9 versus 3.** Ziren's outer machine is degree 9 (soundcalc toml
`wrap.air_max_degree`, `WRAP_DEGREE`), SP1's is 3 (`components.rs:24`). Degree does not enter
the BaseFold query count, but it multiplies the zerocheck round polynomial size and the gnark
circuit's constraint evaluation. Not measured here.

**B6. Deployment binding.** Production runs `VERIFY_VK=false`, under which the compose
program's child-VK membership check is a no-op *(recorded Sep 5,
`reference_soundcalc_soundness_sep5`)*, and the regenerated map holds 169 keys with
open-ended leaf coverage *(Sep 6)*. Whatever the IOPP bits are, the recursion chain does not
bind child verification keys in that mode. SP1's equivalent switch was not located (section
4), so this is stated as a Ziren caveat, not a comparison.

### Ahead

**A1. WHIR is the production inner PCS.** Three committed rounds and one final polynomial
(`whir/jagged.rs:141-170`) instead of 21 arity-2 BaseFold rounds (`fri.rs:108-118`,
`verifier.rs:318-340`). Per query the verifier authenticates 3 Merkle paths plus the stir
folds instead of 21 paths; this is why the leaf circuit came in at `2^26` cells against
SP1's `2^27` bound and why in-circuit WHIR verification is 387 K instructions against
421-431 K for the BaseFold opening it replaced *(measured Aug 27)*. SP1 has the same code
(`slop/crates/whir`, `recursion/circuit/src/basefold/whir.rs`) but has not switched.

**A2. Proof bytes.** A BaseFold opening at rate 1/4 with 124 queries authenticates one sibling
pair and a Merkle path per round for 21 rounds; a WHIR opening authenticates three leaves.
Ziren's compress proof is 617,618 B *(measured)*; SP1's is not in the tree, so no number is
claimed, but the round structure alone puts Ziren's per-query cost far lower at equal query
counts.

**A3. GKR memory traffic.** Ziren's layer transitions pair adjacent rows and every chip halves
every layer (`row_gkr/transition.rs:10-14`); the GPU port of that layout cut GKR kernel time
20% and the block wall 21-28% *(measured Sep 6)*. SP1's GPU GKR kernels exist
(`sp1-gpu/crates/sys/lib/logup_gkr/{first_layer,round}.cu`) but their pairing was not
reviewed (section 4).

**A4. Soundness accounting is written down.** Ziren carries a machine census, a soundcalc
model of jagged-over-WHIR with its rate escalation, and the report; SP1's tree exposes only
the UDR query formula. (Ziren's own docstrings carried a capacity-style "100 bits" claim until
Sep 5; the report is the corrected source.)

**A5. Parity items.** Zerocheck transcript (λ-RLC, points 0/2/4, degree 3), HR18 jagged
evaluation, stacking height 21 for core, two commit rounds, OOD-sampled WHIR rounds: same
design on both sides (table rows above).

## 3. Recommendations, in priority order

1. **Decide the security policy explicitly and pin it in code the way SP1 does.** Add a
   `TARGET_BITS_OF_SECURITY` constant next to `core_whir_config` (`crates/pcs/src/whir/jagged.rs:124`)
   and derive the per-round query counts from it with the UDR formula
   `q_r = ceil((target − pow_r) / −log2((1+ρ_r)/2))`, as `fri_params.rs:47-57` does. Today the
   numbers are literals (`whir/jagged.rs:159`). If 64 stays the policy, say so in
   `docs/soundness/`; if 100 is required, the cheapest provable route from the Sep 5 census is
   PoW 20-28 per round plus stacking 22 and `ROUND0_FF` 2, which the size model puts at
   ~615-680 KiB and roughly the UDR-100 wall (+16%) *(estimates, `reference_compress_proof_size_anatomy_sep5`)*.
   Soundness caveat: only UDR numbers are provable for jagged; the Johnson figure (65) is
   capped by the fold terms and cannot be raised by queries.

2. **Land PoW 20 with the next VK campaign.** `perf/whir-pow20` (f508d608) keeps 64 UDR bits
   with queries 65/47/45 and cuts the compress proof 7% at neutral wall *(measured Sep 6,
   second agent)*. It is the first step of recommendation 1 in either policy.

3. **Restore a LogUp-GKR grind before any move toward 100 bits.** With M = 22,637 the GKR
   term is 84 bits at grind 0; 16 bits of grind cost 5.8% wall on the host path *(measured)*.
   SP1's 12 bits (`shard.rs:41`) is the reference. Alternatively cut M: the LoadWord read-side
   duplicate byte checks and the 21 byte lookups per row are the documented targets
   (`reference_gkr_terms_census_sep5`).

4. **Attack machine width, starting with the precompile chips.** Column-count parity with SP1
   is not reachable (different ISA and precompile set), but every 1% of area is ~1% of wall on
   one card *(Sep 5-6 measurements)*. Candidates with file pointers: `Global` (100 columns,
   24.5% of cells, `crates/core/machine/src/global/mod.rs`), the field-op precompiles
   (`crates/core/machine/src/syscall/precompiles/{weierstrass,fptower,edwards}`), and the
   memory instruction chips' byte lookups (`crates/core/machine/src/memory/instructions`).

5. **Measure compress stacking at 20 and 22.** One-line change (`DEFAULT_LOG_STACKING_HEIGHT`,
   `crates/pcs/src/jagged_pcs.rs:78`, and the recursion band shapes); the leaf-size model
   says 22 shrinks round-0 leaves, SP1's choice of 20 for compress (`compressed/config.rs:1`)
   says padding waste matters more for their shapes. Every recursion VK moves either way.

6. **Bring wrap to degree 3 if the gnark circuit cost matters.** SP1's `WRAP_DEGREE 3`
   (`components.rs:24`) against Ziren's 9. Not measured; scope it with the outer circuit
   owners.

7. **Do not spend effort on the leaf circuit or the jagged-eval sub-protocol.** Both are at
   parity or better (A1, A5); the remaining prover-time gap is in the shard prover phases
   (GKR bytes, zerocheck occupancy, commit hashing), which are internal to Ziren's GPU code,
   not protocol differences.

## 4. Not verified

- SP1 proof sizes (compress, shrink, wrap): no constant, test or document in the tree states
  them; the A2 comparison is structural only.
- SP1's Poseidon2 round constants / width for `SP1GlobalContext`: not re-read.
- SP1's reduce (compose) arity: no `REDUCE_BATCH_SIZE`-style constant found in
  `crates/prover/src/*.rs`; the recursion tree shape was not traced.
- SP1's interleave batch size for stacking: not found in `crates/prover/src` or
  `crates/hypercube/src` (Ziren uses 32, `jagged_pcs.rs:84`).
- SP1's VK-membership verification switch (the counterpart of Ziren's `VERIFY_VK`): not
  located under `crates/prover/src`.
- SP1 GPU GKR layer pairing (MSB versus LSB) in `sp1-gpu/crates/sys/lib/logup_gkr`: files
  exist, kernels not reviewed.
- Whether SP1's shipped provers or hosted network run a WHIR configuration: the open-source
  `main` at 7ea83d0 wires BaseFold; `big_beautiful_whir_config` (`slop/crates/whir/src/config.rs:78-116`,
  rate 1/2 start, folds 4, queries 84/21/12/9, PoW 16) is the schedule Ziren's docstring
  copied and soundcalc scored at 27 UDR bits, so if SP1 used it their provable level would be
  the same 27, not 100; this could not be confirmed either way.
- The SP1 column/interaction census (3,741 / 1,911) is from the Sep 5 run of Ziren's census
  tool against an earlier SP1 checkout, not re-run against 7ea83d0.
- `docs/paper` is not present at 49977616 (it lives on another branch); the paper's IOPP/PCS
  section was not cross-checked.

## 5. Sources

Ziren files: `crates/pcs/src/{config.rs, kb31_poseidon2.rs, jagged_pcs.rs, opts.rs, logup_gkr.rs, whir/{mod,config,jagged,stacked,interleaved}.rs, basefold/{mod,config}.rs, shard_level/{prover,verifier,zerocheck_poly,types,shard_proof}.rs, shard_level/row_gkr/{mod,transition,layer,top_level}.rs, jagged_eval_sumcheck.rs, jagged_branching_program.rs}`, `crates/prover/src/lib.rs`, `crates/recursion/circuit/src/*`, `docs/soundness/{ziren.soundcalc.toml, ziren.soundcalc-report.md}`.

SP1 files: `crates/primitives/src/{lib,fri_params}.rs`, `crates/prover/src/components.rs`, `crates/verifier/src/compressed/config.rs`, `crates/core/executor/src/opts.rs`, `crates/hypercube/src/{verifier/shard.rs, verifier/proof.rs, logup_gkr/*, prover/zerocheck/*}`, `crates/recursion/circuit/src/*`, `slop/crates/{basefold, basefold-prover, whir, jagged, stacked, sumcheck, multilinear, primitives}/src/*`, `audits/`.

Measured numbers: memory files `reference_soundcalc_soundness_sep5`, `reference_compress_proof_size_anatomy_sep5`, `project_udr64_schedule_sep5`, `project_udr100_landed_sep5`, `reference_gkr_byte_census_sep6`, `reference_gkr_terms_census_sep5`, `reference_leaf_size_vs_sp1_aug27`, `project_gkr_lsb_pairing_sep6`, `project_clk26_shard_fence_sep6`.
