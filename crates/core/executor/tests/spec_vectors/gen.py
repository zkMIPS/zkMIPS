#!/usr/bin/env python3
"""Generate MIPS32r2 instruction test vectors from the ISA spec table with an independent oracle.

For every instruction row of `docs/src/mips-vm/mips-isa.md` this script assembles a small program
(register set-up via `li`, then the instruction under test, then landing pads) with `llvm-mc`,
executes it with Unicorn (QEMU's MIPS core) and records the final register file, HI/LO and a data
window.  `spec_vectors.rs` replays the same program words through the Ziren executor and compares.

Conventions shared with the Rust side:
  code at CODE (pc_start = pc_base = CODE), data window of DATA_WORDS words at DATA, program ends
  when pc reaches CODE + 4*len (both emulators run exactly the program), traps are expected to
  surface as an execution error in Ziren.

Requirements: llvm-mc (LLVM >= 15, MIPS target) on PATH or LLVM_MC, and the `unicorn` Python
package (PYTHONPATH may point at a --target install).  Output: vectors.json next to this file.
"""
import json
import os
import random
import struct
import subprocess
import sys

from unicorn import Uc, UcError, UC_ARCH_MIPS, UC_HOOK_CODE, UC_MODE_MIPS32, UC_MODE_LITTLE_ENDIAN
from unicorn import mips_const as M

LLVM_MC = os.environ.get("LLVM_MC", "/usr/lib/llvm-18/bin/llvm-mc")
CODE = 0x1000
DATA = 0x3000
DATA_WORDS = 16
SEED = int(os.environ.get("SPEC_VECTORS_SEED", "20260906"))
CASES = int(os.environ.get("SPEC_VECTORS_CASES", "10"))

REG_NAMES = ["zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5",
             "t6", "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1",
             "gp", "sp", "fp", "ra"]
UC_REG = [getattr(M, f"UC_MIPS_REG_{i}") for i in range(32)]
# Scratch registers the templates draw from (never $zero, $at, $k0/$k1, $sp, $ra).
POOL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

EDGE = [0, 1, 2, 0xFFFFFFFF, 0xFFFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001, 0x0000FFFF,
        0xFFFF0000, 0x00010000, 0x12345678, 0xDEADBEEF, 0x00000080, 0x00008000, 0x7FFF7FFF]


def assemble(lines):
    """Assemble `lines` (already in noreorder form) and return the list of 32-bit words."""
    src = ".set noreorder\n.set noat\n" + "\n".join(lines) + "\n"
    out = subprocess.run(
        [LLVM_MC, "-triple=mipsel-unknown-linux", "-mcpu=mips32r2", "-show-encoding"],
        input=src, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"llvm-mc failed on {lines!r}:\n{out.stderr}")
    words = []
    for line in out.stdout.splitlines():
        if "encoding:" not in line:
            continue
        enc = line.split("encoding:")[1].strip().strip("[]")
        bs = bytes(int(b, 16) for b in enc.split(","))
        for i in range(0, len(bs), 4):
            words.append(struct.unpack("<I", bs[i:i + 4])[0])
    return words


def li(reg, val):
    return [f"li ${REG_NAMES[reg]}, {val & 0xFFFFFFFF}"]


def rnd_word(rng):
    r = rng.random()
    if r < 0.35:
        return rng.choice(EDGE)
    if r < 0.5:
        return rng.randrange(0, 256)
    return rng.getrandbits(32)


def sext16(v):
    return v - 0x10000 if v & 0x8000 else v


def run_unicorn(words, image):
    mu = Uc(UC_ARCH_MIPS, UC_MODE_MIPS32 + UC_MODE_LITTLE_ENDIAN)
    mu.mem_map(0, 0x10000)
    mu.mem_write(CODE, b"".join(struct.pack("<I", w) for w in words))
    for addr, val in image.items():
        mu.mem_write(addr, struct.pack("<I", val))
    end = CODE + 4 * len(words)
    # Unicorn's PC register is not reliable after `emu_start` returns; count executed
    # instructions instead and treat exhausting the budget as "did not reach the end".
    executed = {"n": 0}

    def on_code(_uc, _addr, _size, _data):
        executed["n"] += 1

    mu.hook_add(UC_HOOK_CODE, on_code)
    trap = None
    try:
        mu.emu_start(CODE, end, timeout=2_000_000, count=10_000)
    except UcError as e:
        trap = str(e)
    pc = executed["n"]
    regs = [mu.reg_read(UC_REG[i]) & 0xFFFFFFFF for i in range(32)]
    hi = mu.reg_read(M.UC_MIPS_REG_HI) & 0xFFFFFFFF
    lo = mu.reg_read(M.UC_MIPS_REG_LO) & 0xFFFFFFFF
    mem = {}
    for i in range(DATA_WORDS):
        a = DATA + 4 * i
        mem[a] = struct.unpack("<I", mu.mem_read(a, 4))[0]
    return {"regs": regs, "hi": hi, "lo": lo, "mem": mem, "pc": pc, "trap": trap,
            "reached_end": trap is None and pc < 10_000}


class Case:
    def __init__(self, name, rng):
        self.name = name
        self.rng = rng
        self.pre = []          # set-up lines
        self.body = []         # instruction under test + pads
        self.image = {}
        self.expect_trap = False
        self.used = set()

    def reg(self, exclude=()):
        while True:
            r = self.rng.choice(POOL)
            if r not in self.used and r not in exclude:
                self.used.add(r)
                return r

    def set(self, reg, val):
        self.pre += li(reg, val)
        return reg

    def data_window(self):
        for i in range(DATA_WORDS):
            self.image[DATA + 4 * i] = rnd_word(self.rng)


R = REG_NAMES


def name(r):
    return "$" + R[r]


# --- templates ------------------------------------------------------------------------------
# Each template mutates the case: pushes set-up into c.pre and the instruction(s) into c.body.

def t_rrr(mn, no_overflow=False):
    def f(c):
        rng = c.rng
        a, b = rnd_word(rng), rnd_word(rng)
        if no_overflow:
            # MIPS ADD/SUB trap on signed overflow; the spec row gives the non-trapping result.
            while True:
                sa, sb = a - (1 << 32) if a >> 31 else a, b - (1 << 32) if b >> 31 else b
                res = sa + sb if mn in ("add",) else sa - sb
                if -(1 << 31) <= res < (1 << 31):
                    break
                a, b = rnd_word(rng), rnd_word(rng)
        rs, rt, rd = c.reg(), c.reg(), c.reg()
        c.set(rs, a)
        c.set(rt, b)
        c.body.append(f"{mn} {name(rd)}, {name(rs)}, {name(rt)}")
    return f


def t_rri(mn, signed=True, no_overflow=False):
    def f(c):
        rng = c.rng
        a = rnd_word(rng)
        imm = rng.choice([0, 1, 0xFFFF, 0x8000, 0x7FFF, rng.getrandbits(16)])
        if no_overflow:
            while True:
                sa = a - (1 << 32) if a >> 31 else a
                if -(1 << 31) <= sa + sext16(imm) < (1 << 31):
                    break
                a = rnd_word(rng)
        rs, rt = c.reg(), c.reg()
        c.set(rs, a)
        val = sext16(imm) if signed else imm
        c.body.append(f"{mn} {name(rt)}, {name(rs)}, {val}")
    return f


def t_shift_imm(mn):
    def f(c):
        rs, rd = c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        c.body.append(f"{mn} {name(rd)}, {name(rs)}, {c.rng.choice([0, 1, 7, 8, 15, 16, 31])}")
    return f


def t_shift_var(mn):
    def f(c):
        rt, rs, rd = c.reg(), c.reg(), c.reg()
        c.set(rt, rnd_word(c.rng))
        c.set(rs, c.rng.choice([0, 1, 31, 32, 33, 0xFFFFFFFF, c.rng.getrandbits(32)]))
        c.body.append(f"{mn} {name(rd)}, {name(rt)}, {name(rs)}")
    return f


def t_lui(c):
    rt = c.reg()
    c.body.append(f"lui {name(rt)}, {c.rng.choice([0, 1, 0x8000, 0xFFFF, c.rng.getrandbits(16)])}")


def t_branch(mn, two_regs):
    def f(c):
        rng = c.rng
        rs, rt = c.reg(), c.reg()
        a = rnd_word(rng)
        b = a if rng.random() < 0.4 else rnd_word(rng)
        c.set(rs, a)
        if two_regs:
            c.set(rt, b)
        m1, m2 = c.reg(), c.reg()
        cond = f"{name(rs)}, {name(rt)}, 12" if two_regs else f"{name(rs)}, 12"
        # taken -> skips the two `li` after the delay slot.
        c.body += [f"{mn} {cond}", "nop", f"li {name(m1)}, 1", f"li {name(m1)}, 2", f"li {name(m2)}, 3"]
    return f


def t_bal(c):
    m1, m2 = c.reg(), c.reg()
    c.body += ["bal 12", "nop", f"li {name(m1)}, 1", f"li {name(m1)}, 2", f"li {name(m2)}, 3"]


def t_jump(mn):
    def f(c):
        m1, m2 = c.reg(), c.reg()
        # Target is the last `li`; absolute address = CODE + 4 * (len(pre) + 4).
        c.body += [f"{mn} __TGT__", "nop", f"li {name(m1)}, 1", f"li {name(m1)}, 2", f"li {name(m2)}, 3"]
        c.jump_target_index = 4
    return f


def t_jr(link):
    def f(c):
        rs = c.reg()
        m1, m2 = c.reg(), c.reg()
        c.jump_reg = rs
        c.body += [(f"jalr {name(rs)}" if link else f"jr {name(rs)}"), "nop",
                   f"li {name(m1)}, 1", f"li {name(m1)}, 2", f"li {name(m2)}, 3"]
        c.jump_target_index = 4
    return f


def t_load(mn, align):
    def f(c):
        c.data_window()
        base, rt = c.reg(), c.reg()
        off = c.rng.randrange(0, 4 * (DATA_WORDS - 1))
        off -= off % align
        c.set(base, DATA)
        c.body.append(f"{mn} {name(rt)}, {off}({name(base)})")
    return f


def t_store(mn, align):
    def f(c):
        c.data_window()
        base, rt = c.reg(), c.reg()
        off = c.rng.randrange(0, 4 * (DATA_WORDS - 1))
        off -= off % align
        c.set(base, DATA)
        c.set(rt, rnd_word(c.rng))
        c.body.append(f"{mn} {name(rt)}, {off}({name(base)})")
    return f


def t_hilo_move(mn):
    def f(c):
        rs, rd = c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        if mn == "mfhi":
            c.body += [f"mthi {name(rs)}", f"mfhi {name(rd)}"]
        elif mn == "mflo":
            c.body += [f"mtlo {name(rs)}", f"mflo {name(rd)}"]
        elif mn == "mthi":
            c.body += [f"mthi {name(rs)}", f"mfhi {name(rd)}"]
        else:
            c.body += [f"mtlo {name(rs)}", f"mflo {name(rd)}"]
    return f


def t_mult(mn):
    def f(c):
        rs, rt = c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        c.set(rt, rnd_word(c.rng))
        c.body.append(f"{mn} {name(rs)}, {name(rt)}")
    return f


def t_div(mn):
    def f(c):
        rs, rt = c.reg(), c.reg()
        a, b = rnd_word(c.rng), rnd_word(c.rng)
        while b == 0:  # division by zero is UNPREDICTABLE in the spec; not a vector
            b = rnd_word(c.rng)
        c.set(rs, a)
        c.set(rt, b)
        c.body.append(f"{mn} $zero, {name(rs)}, {name(rt)}")
    return f


def t_madd(mn):
    def f(c):
        rs, rt, h, l = c.reg(), c.reg(), c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        c.set(rt, rnd_word(c.rng))
        c.set(h, rnd_word(c.rng))
        c.set(l, rnd_word(c.rng))
        c.body += [f"mthi {name(h)}", f"mtlo {name(l)}", f"{mn} {name(rs)}, {name(rt)}"]
    return f


def t_ext(c):
    rs, rt = c.reg(), c.reg()
    c.set(rs, rnd_word(c.rng))
    pos = c.rng.randrange(0, 32)
    size = c.rng.randrange(1, 33 - pos)
    c.body.append(f"ext {name(rt)}, {name(rs)}, {pos}, {size}")


def t_ins(c):
    rs, rt = c.reg(), c.reg()
    c.set(rs, rnd_word(c.rng))
    c.set(rt, rnd_word(c.rng))
    pos = c.rng.randrange(0, 32)
    size = c.rng.randrange(1, 33 - pos)
    c.body.append(f"ins {name(rt)}, {name(rs)}, {pos}, {size}")


def t_movcond(mn):
    def f(c):
        rs, rt, rd = c.reg(), c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        c.set(rt, c.rng.choice([0, 0, 1, rnd_word(c.rng)]))
        c.set(rd, rnd_word(c.rng))
        c.body.append(f"{mn} {name(rd)}, {name(rs)}, {name(rt)}")
    return f


def t_teq(c):
    rs, rt = c.reg(), c.reg()
    a = rnd_word(c.rng)
    trap = c.rng.random() < 0.3
    c.set(rs, a)
    c.set(rt, a if trap else (a ^ 1))
    c.expect_trap = trap
    c.body.append(f"teq {name(rs)}, {name(rt)}")


def t_nop(line):
    def f(c):
        rs = c.reg()
        c.set(rs, DATA)
        c.body.append(line.format(base=name(rs)))
    return f


def t_llsc(c):
    c.data_window()
    base, rt, rv = c.reg(), c.reg(), c.reg()
    off = c.rng.randrange(0, DATA_WORDS) * 4
    c.set(base, DATA)
    c.set(rv, rnd_word(c.rng))
    c.body += [f"ll {name(rt)}, {off}({name(base)})", f"sc {name(rv)}, {off}({name(base)})"]


def t_unary(mn):
    def f(c):
        rs, rd = c.reg(), c.reg()
        c.set(rs, rnd_word(c.rng))
        c.body.append(f"{mn} {name(rd)}, {name(rs)}")
    return f


TEMPLATES = {
    "ADD": t_rrr("add", no_overflow=True), "ADDU": t_rrr("addu"), "SUB": t_rrr("sub", no_overflow=True),
    "SUBU": t_rrr("subu"), "AND": t_rrr("and"), "OR": t_rrr("or"), "XOR": t_rrr("xor"),
    "NOR": t_rrr("nor"), "SLT": t_rrr("slt"), "SLTU": t_rrr("sltu"), "MUL": t_rrr("mul"),
    "ADDI": t_rri("addi", no_overflow=True), "ADDIU": t_rri("addiu"), "ANDI": t_rri("andi", signed=False),
    "ORI": t_rri("ori", signed=False), "XORI": t_rri("xori", signed=False),
    "SLTI": t_rri("slti"), "SLTIU": t_rri("sltiu"),
    "SLL": t_shift_imm("sll"), "SRL": t_shift_imm("srl"), "SRA": t_shift_imm("sra"),
    "ROTR": t_shift_imm("rotr"),
    "SLLV": t_shift_var("sllv"), "SRLV": t_shift_var("srlv"), "SRAV": t_shift_var("srav"),
    "ROTRV": t_shift_var("rotrv"),
    "LUI": t_lui,
    "BEQ": t_branch("beq", True), "BNE": t_branch("bne", True), "BGEZ": t_branch("bgez", False),
    "BGTZ": t_branch("bgtz", False), "BLEZ": t_branch("blez", False), "BLTZ": t_branch("bltz", False),
    "BAL": t_bal,
    "J": t_jump("j"), "JAL": t_jump("jal"), "JR": t_jr(False), "JALR": t_jr(True),
    "LB": t_load("lb", 1), "LBU": t_load("lbu", 1), "LH": t_load("lh", 2), "LHU": t_load("lhu", 2),
    "LW": t_load("lw", 4), "LWL": t_load("lwl", 1), "LWR": t_load("lwr", 1), "LL": t_llsc,
    "SB": t_store("sb", 1), "SH": t_store("sh", 2), "SW": t_store("sw", 4),
    "SWL": t_store("swl", 1), "SWR": t_store("swr", 1), "SC": t_llsc,
    "MFHI": t_hilo_move("mfhi"), "MFLO": t_hilo_move("mflo"), "MTHI": t_hilo_move("mthi"),
    "MTLO": t_hilo_move("mtlo"),
    "MULT": t_mult("mult"), "MULTU": t_mult("multu"), "DIV": t_div("div"), "DIVU": t_div("divu"),
    "MADD": t_madd("madd"), "MADDU": t_madd("maddu"), "MSUB": t_madd("msub"), "MSUBU": t_madd("msubu"),
    "CLO": t_unary("clo"), "CLZ": t_unary("clz"), "SEB": t_unary("seb"), "SEH": t_unary("seh"),
    "WSBH": t_unary("wsbh"), "EXT": t_ext, "INS": t_ins,
    "MOVN": t_movcond("movn"), "MOVZ": t_movcond("movz"), "TEQ": t_teq,
    "SYNC": t_nop("sync"), "SYNCI": t_nop("synci 0({base})"), "PREF": t_nop("pref 0, 0({base})"),
}
# SYSCALL is exercised by the Cannon Linux-syscall vectors and the guest programs, not here.


def spec_instructions(spec_path):
    names = []
    for line in open(spec_path):
        if not line.startswith("|"):
            continue
        cell = line.split("|")[1].strip()
        if cell and cell.upper() == cell and cell.isalpha() and cell not in ("SYSCALL",):
            names.append(cell)
    return names


def build_case(mn, idx, rng):
    c = Case(f"{mn}_{idx}", rng)
    TEMPLATES[mn](c)
    pre_words = assemble(c.pre) if c.pre else []
    body = list(c.body)
    if hasattr(c, "jump_target_index"):
        target = CODE + 4 * (len(pre_words) + c.jump_target_index)
        body = [b.replace("__TGT__", str(target)) for b in body]
        if hasattr(c, "jump_reg"):
            pre_words = assemble(c.pre + li(c.jump_reg, target))
    words = pre_words + assemble(body)
    result = run_unicorn(words, c.image)
    if c.expect_trap:
        ok = result["trap"] is not None
    else:
        ok = result["reached_end"]
    return {
        "name": c.name,
        "mnemonic": mn,
        "asm": c.pre + body,
        "words": [f"{w:08x}" for w in words],
        "image": {f"{a:#x}": v for a, v in sorted(c.image.items())},
        "expect_trap": c.expect_trap,
        "oracle_ok": ok,
        "oracle_note": result["trap"],
        "regs": result["regs"],
        "hi": result["hi"],
        "lo": result["lo"],
        "mem": {f"{a:#x}": v for a, v in sorted(result["mem"].items())},
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    spec = os.path.join(repo, "docs", "src", "mips-vm", "mips-isa.md")
    names = spec_instructions(spec)
    missing = [n for n in names if n not in TEMPLATES]
    rng = random.Random(SEED)
    vectors = []
    for mn in names:
        if mn not in TEMPLATES:
            continue
        for i in range(CASES):
            vectors.append(build_case(mn, i, rng))
    bad = [v["name"] for v in vectors if not v["oracle_ok"]]
    out = {
        "seed": SEED, "code": CODE, "data": DATA, "data_words": DATA_WORDS,
        "spec_instructions": names, "no_template": missing, "vectors": vectors,
    }
    with open(os.path.join(here, "vectors.json"), "w") as f:
        json.dump(out, f, indent=0)
    print(f"spec instructions: {len(names)}, templated: {len(names) - len(missing)}, "
          f"no template: {missing}, vectors: {len(vectors)}, oracle problems: {bad[:10]}")


if __name__ == "__main__":
    sys.exit(main())
