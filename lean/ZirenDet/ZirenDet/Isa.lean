/-!
# Executable MIPS32r2 semantics (the Ziren guest ISA)

A Lean model of the instruction set in `docs/src/mips-vm/mips-isa.md`, written to be *executed*:
`decode` reads the same 32-bit words the Ziren executor decodes, `step` implements each row of the
spec table (delay slots included), and `run` executes a program placed at `code` until the pc
leaves it.  `ZirenDet/IsaVectors.lean` (generated) checks this model against the oracle vectors
that the Rust executor is also checked against, so the three agree on every vector: oracle
(QEMU), executor (Rust), and this model.  The determinism theorems in `ZirenDet/Chips` will be
stated against this model.

Conventions: little-endian byte memory (mipsel), `$zero` reads as zero and ignores writes,
`HI`/`LO` are separate registers, traps (`TEQ`) stop execution with `trapped = true`.
-/

namespace ZirenDet.Isa

abbrev W := BitVec 32

/-- Sparse little-endian byte memory: an association list with a zero default. -/
structure Mem where
  bytes : List (W × BitVec 8)

namespace Mem
def empty : Mem := ⟨[]⟩

def readByte (m : Mem) (a : W) : BitVec 8 :=
  match m.bytes.find? (fun p => p.1 == a) with
  | some p => p.2
  | none => 0

def writeByte (m : Mem) (a : W) (b : BitVec 8) : Mem :=
  ⟨(a, b) :: m.bytes.filter (fun p => p.1 != a)⟩

/-- Little-endian word at `a` (no alignment requirement; callers align). -/
def readWord (m : Mem) (a : W) : W :=
  (m.readByte (a + 3)) ++ (m.readByte (a + 2)) ++ (m.readByte (a + 1)) ++ (m.readByte a)

def writeWord (m : Mem) (a : W) (v : W) : Mem :=
  (((m.writeByte a (v.extractLsb' 0 8)).writeByte (a + 1) (v.extractLsb' 8 8)).writeByte
      (a + 2) (v.extractLsb' 16 8)).writeByte (a + 3) (v.extractLsb' 24 8)

def readHalf (m : Mem) (a : W) : BitVec 16 :=
  (m.readByte (a + 1)) ++ (m.readByte a)

def writeHalf (m : Mem) (a : W) (v : BitVec 16) : Mem :=
  (m.writeByte a (v.extractLsb' 0 8)).writeByte (a + 1) (v.extractLsb' 8 8)
end Mem

/-- Machine state.  `pc` is the instruction being executed, `nextPc` the one after it (so a
branch in `pc` retargets the instruction after the delay slot). -/
structure State where
  pc : W
  nextPc : W
  gpr : Fin 32 → W
  hi : W
  lo : W
  mem : Mem
  trapped : Bool

def State.reg (s : State) (r : Fin 32) : W := if r = 0 then 0 else s.gpr r

def State.setReg (s : State) (r : Fin 32) (v : W) : State :=
  if r = 0 then s else { s with gpr := fun i => if i = r then v else s.gpr i }

/-- Instructions, one constructor per spec row (aliases such as `ADDU`/`ADD` share semantics but
keep their own constructor so the decoder is a transcription of the table). -/
inductive Insn where
  | add (rd rs rt : Fin 32) | addu (rd rs rt : Fin 32) | sub (rd rs rt : Fin 32) | subu (rd rs rt : Fin 32)
  | and (rd rs rt : Fin 32) | or (rd rs rt : Fin 32) | xor (rd rs rt : Fin 32) | nor (rd rs rt : Fin 32)
  | slt (rd rs rt : Fin 32) | sltu (rd rs rt : Fin 32) | mul (rd rs rt : Fin 32)
  | addi (rt rs : Fin 32) (imm : BitVec 16) | addiu (rt rs : Fin 32) (imm : BitVec 16)
  | andi (rt rs : Fin 32) (imm : BitVec 16) | ori (rt rs : Fin 32) (imm : BitVec 16)
  | xori (rt rs : Fin 32) (imm : BitVec 16) | slti (rt rs : Fin 32) (imm : BitVec 16)
  | sltiu (rt rs : Fin 32) (imm : BitVec 16) | lui (rt : Fin 32) (imm : BitVec 16)
  | sll (rd rt : Fin 32) (sa : BitVec 5) | srl (rd rt : Fin 32) (sa : BitVec 5)
  | sra (rd rt : Fin 32) (sa : BitVec 5) | rotr (rd rt : Fin 32) (sa : BitVec 5)
  | sllv (rd rt rs : Fin 32) | srlv (rd rt rs : Fin 32) | srav (rd rt rs : Fin 32) | rotrv (rd rt rs : Fin 32)
  | beq (rs rt : Fin 32) (off : BitVec 16) | bne (rs rt : Fin 32) (off : BitVec 16)
  | bgez (rs : Fin 32) (off : BitVec 16) | bgtz (rs : Fin 32) (off : BitVec 16)
  | blez (rs : Fin 32) (off : BitVec 16) | bltz (rs : Fin 32) (off : BitVec 16)
  | bal (off : BitVec 16)
  | j (target : BitVec 26) | jal (target : BitVec 26) | jr (rs : Fin 32) | jalr (rd rs : Fin 32)
  | lb (rt base : Fin 32) (off : BitVec 16) | lbu (rt base : Fin 32) (off : BitVec 16)
  | lh (rt base : Fin 32) (off : BitVec 16) | lhu (rt base : Fin 32) (off : BitVec 16)
  | lw (rt base : Fin 32) (off : BitVec 16) | lwl (rt base : Fin 32) (off : BitVec 16)
  | lwr (rt base : Fin 32) (off : BitVec 16) | ll (rt base : Fin 32) (off : BitVec 16)
  | sb (rt base : Fin 32) (off : BitVec 16) | sh (rt base : Fin 32) (off : BitVec 16)
  | sw (rt base : Fin 32) (off : BitVec 16) | swl (rt base : Fin 32) (off : BitVec 16)
  | swr (rt base : Fin 32) (off : BitVec 16) | sc (rt base : Fin 32) (off : BitVec 16)
  | mfhi (rd : Fin 32) | mflo (rd : Fin 32) | mthi (rs : Fin 32) | mtlo (rs : Fin 32)
  | mult (rs rt : Fin 32) | multu (rs rt : Fin 32) | div (rs rt : Fin 32) | divu (rs rt : Fin 32)
  | madd (rs rt : Fin 32) | maddu (rs rt : Fin 32) | msub (rs rt : Fin 32) | msubu (rs rt : Fin 32)
  | clo (rd rs : Fin 32) | clz (rd rs : Fin 32) | seb (rd rt : Fin 32) | seh (rd rt : Fin 32)
  | wsbh (rd rt : Fin 32)
  | ext (rt rs : Fin 32) (msbd lsb : BitVec 5) | ins (rt rs : Fin 32) (msb lsb : BitVec 5)
  | movn (rd rs rt : Fin 32) | movz (rd rs rt : Fin 32) | teq (rs rt : Fin 32)
  | nop
  deriving Repr, DecidableEq

/-! ## Decoder — a transcription of the encoding columns of the spec table. -/

def field (w : W) (lo len : Nat) : Nat := (w.extractLsb' lo len).toNat

def reg (w : W) (lo : Nat) : Fin 32 := ⟨field w lo 5, by
  have := (w.extractLsb' lo 5).isLt; simpa [field] using this⟩

def decode (w : W) : Option Insn :=
  let op := field w 26 6
  let rs := reg w 21
  let rt := reg w 16
  let rd := reg w 11
  let sa : BitVec 5 := w.extractLsb' 6 5
  let fn := field w 0 6
  let imm : BitVec 16 := w.extractLsb' 0 16
  let target : BitVec 26 := w.extractLsb' 0 26
  match op with
  | 0 =>  -- SPECIAL
    match fn with
    | 0x20 => some (.add rd rs rt) | 0x21 => some (.addu rd rs rt)
    | 0x22 => some (.sub rd rs rt) | 0x23 => some (.subu rd rs rt)
    | 0x24 => some (.and rd rs rt) | 0x25 => some (.or rd rs rt)
    | 0x26 => some (.xor rd rs rt) | 0x27 => some (.nor rd rs rt)
    | 0x2a => some (.slt rd rs rt) | 0x2b => some (.sltu rd rs rt)
    | 0x00 => some (.sll rd rt sa)
    | 0x02 => some (if field w 21 5 = 1 then .rotr rd rt sa else .srl rd rt sa)
    | 0x03 => some (.sra rd rt sa)
    | 0x04 => some (.sllv rd rt rs)
    | 0x06 => some (if sa.toNat = 1 then .rotrv rd rt rs else .srlv rd rt rs)
    | 0x07 => some (.srav rd rt rs)
    | 0x08 => some (.jr rs) | 0x09 => some (.jalr rd rs)
    | 0x0a => some (.movz rd rs rt) | 0x0b => some (.movn rd rs rt)
    | 0x0f => some .nop  -- SYNC
    | 0x10 => some (.mfhi rd) | 0x11 => some (.mthi rs)
    | 0x12 => some (.mflo rd) | 0x13 => some (.mtlo rs)
    | 0x18 => some (.mult rs rt) | 0x19 => some (.multu rs rt)
    | 0x1a => some (.div rs rt) | 0x1b => some (.divu rs rt)
    | 0x34 => some (.teq rs rt)
    | _ => none
  | 1 =>  -- REGIMM
    match field w 16 5 with
    | 0x00 => some (.bltz rs imm) | 0x01 => some (.bgez rs imm)
    | 0x11 => some (.bal imm)
    | 0x1f => some .nop  -- SYNCI
    | _ => none
  | 2 => some (.j target) | 3 => some (.jal target)
  | 4 => some (.beq rs rt imm) | 5 => some (.bne rs rt imm)
  | 6 => some (.blez rs imm) | 7 => some (.bgtz rs imm)
  | 8 => some (.addi rt rs imm) | 9 => some (.addiu rt rs imm)
  | 0xa => some (.slti rt rs imm) | 0xb => some (.sltiu rt rs imm)
  | 0xc => some (.andi rt rs imm) | 0xd => some (.ori rt rs imm)
  | 0xe => some (.xori rt rs imm) | 0xf => some (.lui rt imm)
  | 0x1c =>  -- SPECIAL2
    match fn with
    | 0x00 => some (.madd rs rt) | 0x01 => some (.maddu rs rt)
    | 0x02 => some (.mul rd rs rt)
    | 0x04 => some (.msub rs rt) | 0x05 => some (.msubu rs rt)
    | 0x20 => some (.clz rd rs) | 0x21 => some (.clo rd rs)
    | _ => none
  | 0x1f =>  -- SPECIAL3
    match fn with
    | 0x00 => some (.ext rt rs (w.extractLsb' 11 5) sa)
    | 0x04 => some (.ins rt rs (w.extractLsb' 11 5) sa)
    | 0x20 =>
      match field w 6 5 with
      | 0x02 => some (.wsbh rd rt) | 0x10 => some (.seb rd rt) | 0x18 => some (.seh rd rt)
      | _ => none
    | _ => none
  | 0x20 => some (.lb rt rs imm) | 0x21 => some (.lh rt rs imm)
  | 0x22 => some (.lwl rt rs imm) | 0x23 => some (.lw rt rs imm)
  | 0x24 => some (.lbu rt rs imm) | 0x25 => some (.lhu rt rs imm)
  | 0x26 => some (.lwr rt rs imm)
  | 0x28 => some (.sb rt rs imm) | 0x29 => some (.sh rt rs imm)
  | 0x2a => some (.swl rt rs imm) | 0x2b => some (.sw rt rs imm)
  | 0x2e => some (.swr rt rs imm)
  | 0x30 => some (.ll rt rs imm) | 0x33 => some .nop  -- PREF
  | 0x38 => some (.sc rt rs imm)
  | _ => none

/-! ## Semantics -/

def sext16 (i : BitVec 16) : W := i.signExtend 32
def zext16 (i : BitVec 16) : W := i.zeroExtend 32
def boff (i : BitVec 16) : W := (i.signExtend 32) <<< 2

def clz32 (x : W) : W := BitVec.ofNat 32 (32 - (Nat.log2 x.toNat + 1) |> fun n => if x.toNat = 0 then 32 else n)
def clo32 (x : W) : W := clz32 (~~~x)

def mul64s (a b : W) : BitVec 64 := (a.signExtend 64) * (b.signExtend 64)
def mul64u (a b : W) : BitVec 64 := (a.zeroExtend 64) * (b.zeroExtend 64)
def hilo (s : State) : BitVec 64 := s.hi ++ s.lo
def setHiLo (s : State) (v : BitVec 64) : State :=
  { s with hi := v.extractLsb' 32 32, lo := v.extractLsb' 0 32 }

/-- Little-endian `LWL`: the bytes from the aligned word up to `a` land in the high part. -/
def lwl (mem : Mem) (a : W) (old : W) : W :=
  let b := (a.extractLsb' 0 2).toNat          -- byte offset 0..3
  let word := mem.readWord (a &&& ~~~(3 : W))
  let sh := 8 * (3 - b)
  ((word <<< sh) ||| (old &&& (((1 : W) <<< sh) - 1)))

/-- Little-endian `LWR`: the bytes from `a` to the end of the word land in the low part. -/
def lwr (mem : Mem) (a : W) (old : W) : W :=
  let b := (a.extractLsb' 0 2).toNat
  let word := mem.readWord (a &&& ~~~(3 : W))
  let sh := 8 * b
  if b = 0 then word else ((word >>> sh) ||| (old &&& ~~~(((1 : W) <<< (32 - sh)) - 1)))

/-- Little-endian `SWL`: the high `b+1` bytes of `v` go to the low `b+1` bytes of the aligned
word (addresses `base..a`); the rest of the word is kept. -/
def swl (mem : Mem) (a : W) (v : W) : Mem :=
  let b := (a.extractLsb' 0 2).toNat
  let base := a &&& ~~~(3 : W)
  let word := mem.readWord base
  let keep : W := if b = 3 then 0 else ~~~(((1 : W) <<< (8 * (b + 1))) - 1)
  mem.writeWord base ((word &&& keep) ||| (v >>> (8 * (3 - b))))

def swr (mem : Mem) (a : W) (v : W) : Mem :=
  let b := (a.extractLsb' 0 2).toNat
  let base := a &&& ~~~(3 : W)
  let word := mem.readWord base
  let sh := 8 * b
  let keep : W := if b = 0 then 0 else ((1 : W) <<< sh) - 1
  mem.writeWord base ((word &&& keep) ||| (v <<< sh))

/-- One instruction.  Returns the successor state; `nextPc` is advanced unless a branch or jump
retargets it. -/
def exec (s : State) (i : Insn) : State :=
  let r := s.reg
  let pc := s.pc
  let seq : State := { s with pc := s.nextPc, nextPc := s.nextPc + 4 }
  let branch (cond : Bool) (off : BitVec 16) : State :=
    if cond then { s with pc := s.nextPc, nextPc := pc + 4 + boff off } else seq
  let jumpTo (t : W) : State := { s with pc := s.nextPc, nextPc := t }
  let jtarget (t : BitVec 26) : W := ((pc + 4) &&& 0xf0000000) ||| ((t.zeroExtend 32) <<< 2)
  let addr (base : Fin 32) (off : BitVec 16) : W := r base + sext16 off
  let set (st : State) (rd : Fin 32) (v : W) : State := st.setReg rd v
  match i with
  | .add rd rs rt | .addu rd rs rt => set seq rd (r rs + r rt)
  | .sub rd rs rt | .subu rd rs rt => set seq rd (r rs - r rt)
  | .and rd rs rt => set seq rd (r rs &&& r rt)
  | .or rd rs rt => set seq rd (r rs ||| r rt)
  | .xor rd rs rt => set seq rd (r rs ^^^ r rt)
  | .nor rd rs rt => set seq rd (~~~(r rs ||| r rt))
  | .slt rd rs rt => set seq rd (if (r rs).slt (r rt) then 1 else 0)
  | .sltu rd rs rt => set seq rd (if (r rs).ult (r rt) then 1 else 0)
  | .mul rd rs rt => set seq rd (r rs * r rt)
  | .addi rt rs imm | .addiu rt rs imm => set seq rt (r rs + sext16 imm)
  | .andi rt rs imm => set seq rt (r rs &&& zext16 imm)
  | .ori rt rs imm => set seq rt (r rs ||| zext16 imm)
  | .xori rt rs imm => set seq rt (r rs ^^^ zext16 imm)
  | .slti rt rs imm => set seq rt (if (r rs).slt (sext16 imm) then 1 else 0)
  | .sltiu rt rs imm => set seq rt (if (r rs).ult (sext16 imm) then 1 else 0)
  | .lui rt imm => set seq rt ((zext16 imm) <<< 16)
  | .sll rd rt sa => set seq rd (r rt <<< sa.toNat)
  | .srl rd rt sa => set seq rd (r rt >>> sa.toNat)
  | .sra rd rt sa => set seq rd ((r rt).sshiftRight sa.toNat)
  | .rotr rd rt sa => set seq rd ((r rt).rotateRight sa.toNat)
  | .sllv rd rt rs => set seq rd (r rt <<< ((r rs).extractLsb' 0 5).toNat)
  | .srlv rd rt rs => set seq rd (r rt >>> ((r rs).extractLsb' 0 5).toNat)
  | .srav rd rt rs => set seq rd ((r rt).sshiftRight ((r rs).extractLsb' 0 5).toNat)
  | .rotrv rd rt rs => set seq rd ((r rt).rotateRight ((r rs).extractLsb' 0 5).toNat)
  | .beq rs rt off => branch (r rs == r rt) off
  | .bne rs rt off => branch (r rs != r rt) off
  | .bgez rs off => branch (!(r rs).msb) off
  | .bgtz rs off => branch (!(r rs).msb && r rs != 0) off
  | .blez rs off => branch ((r rs).msb || r rs == 0) off
  | .bltz rs off => branch ((r rs).msb) off
  | .bal off => set (branch true off) 31 (pc + 8)
  | .j t => jumpTo (jtarget t)
  | .jal t => set (jumpTo (jtarget t)) 31 (pc + 8)
  | .jr rs => jumpTo (r rs)
  | .jalr rd rs => set (jumpTo (r rs)) rd (pc + 8)
  | .lb rt base off => set seq rt ((s.mem.readByte (addr base off)).signExtend 32)
  | .lbu rt base off => set seq rt ((s.mem.readByte (addr base off)).zeroExtend 32)
  | .lh rt base off => set seq rt ((s.mem.readHalf (addr base off)).signExtend 32)
  | .lhu rt base off => set seq rt ((s.mem.readHalf (addr base off)).zeroExtend 32)
  | .lw rt base off | .ll rt base off => set seq rt (s.mem.readWord (addr base off))
  | .lwl rt base off => set seq rt (lwl s.mem (addr base off) (r rt))
  | .lwr rt base off => set seq rt (lwr s.mem (addr base off) (r rt))
  | .sb rt base off => { seq with mem := s.mem.writeByte (addr base off) ((r rt).extractLsb' 0 8) }
  | .sh rt base off => { seq with mem := s.mem.writeHalf (addr base off) ((r rt).extractLsb' 0 16) }
  | .sw rt base off => { seq with mem := s.mem.writeWord (addr base off) (r rt) }
  | .sc rt base off => set { seq with mem := s.mem.writeWord (addr base off) (r rt) } rt 1
  | .swl rt base off => { seq with mem := swl s.mem (addr base off) (r rt) }
  | .swr rt base off => { seq with mem := swr s.mem (addr base off) (r rt) }
  | .mfhi rd => set seq rd s.hi
  | .mflo rd => set seq rd s.lo
  | .mthi rs => { seq with hi := r rs }
  | .mtlo rs => { seq with lo := r rs }
  | .mult rs rt => setHiLo seq (mul64s (r rs) (r rt))
  | .multu rs rt => setHiLo seq (mul64u (r rs) (r rt))
  | .div rs rt => { seq with lo := (r rs).sdiv (r rt), hi := (r rs).srem (r rt) }
  | .divu rs rt => { seq with lo := (r rs).udiv (r rt), hi := (r rs).umod (r rt) }
  | .madd rs rt => setHiLo seq (hilo s + mul64s (r rs) (r rt))
  | .maddu rs rt => setHiLo seq (hilo s + mul64u (r rs) (r rt))
  | .msub rs rt => setHiLo seq (hilo s - mul64s (r rs) (r rt))
  | .msubu rs rt => setHiLo seq (hilo s - mul64u (r rs) (r rt))
  | .clo rd rs => set seq rd (clo32 (r rs))
  | .clz rd rs => set seq rd (clz32 (r rs))
  | .seb rd rt => set seq rd (((r rt).extractLsb' 0 8).signExtend 32)
  | .seh rd rt => set seq rd (((r rt).extractLsb' 0 16).signExtend 32)
  | .wsbh rd rt =>
    let x := r rt
    set seq rd ((x.extractLsb' 16 8) ++ (x.extractLsb' 24 8) ++ (x.extractLsb' 0 8) ++ (x.extractLsb' 8 8))
  | .ext rt rs msbd lsb =>
    let size := msbd.toNat + 1
    let v := (r rs >>> lsb.toNat) &&& (((1 : W) <<< size) - 1)
    set seq rt v
  | .ins rt rs msb lsb =>
    let size := msb.toNat + 1 - lsb.toNat
    let mask : W := (((1 : W) <<< size) - 1) <<< lsb.toNat
    set seq rt ((r rt &&& ~~~mask) ||| ((r rs <<< lsb.toNat) &&& mask))
  | .movn rd rs rt => if r rt != 0 then set seq rd (r rs) else seq
  | .movz rd rs rt => if r rt == 0 then set seq rd (r rs) else seq
  | .teq rs rt => if r rs == r rt then { s with trapped := true } else seq
  | .nop => seq

/-- Program: words at `code`, executed until `pc` leaves `[code, code + 4·len)`. -/
structure Program where
  code : W
  words : List W

def Program.fetch (p : Program) (pc : W) : Option W :=
  let idx := ((pc - p.code) >>> 2).toNat
  if pc < p.code then none else p.words[idx]?

def step (p : Program) (s : State) : Option State := do
  let w ← p.fetch s.pc
  let i ← decode w
  pure (exec s i)

/-- Run with fuel; `none` if a word does not decode or fuel runs out. -/
def run (p : Program) : Nat → State → Option State
  | 0, _ => none
  | fuel + 1, s =>
    if s.trapped then some s
    else match p.fetch s.pc with
      | none => some s              -- pc left the program: done
      | some w =>
        match decode w with
        | none => none
        | some i => run p fuel (exec s i)

def initState (code : W) (image : List (W × W)) : State :=
  { pc := code, nextPc := code + 4, gpr := fun _ => 0, hi := 0, lo := 0,
    mem := image.foldl (fun m (a, v) => m.writeWord a v) Mem.empty, trapped := false }

/-- Executable check used by the generated vector file: run `words` at `code` from `image` and
compare the register file, HI/LO and the given memory words. -/
def check (code : W) (words : List W) (image : List (W × W)) (regs : List W) (hi lo : W)
    (mem : List (W × W)) (expectTrap : Bool) : Bool :=
  match run ⟨code, words⟩ 100000 (initState code image) with
  | none => false
  | some s =>
    if expectTrap then s.trapped
    else
      !s.trapped &&
      (List.range 32).all (fun i => regs[i]? == some (s.reg ⟨i % 32, Nat.mod_lt _ (by decide)⟩)) &&
      s.hi == hi && s.lo == lo &&
      mem.all (fun (a, v) => s.mem.readWord a == v)

end ZirenDet.Isa
