//! Opcodes for ZKM.

use enum_map::Enum;
use num_enum::TryFromPrimitive;
use p3_field::Field;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// An opcode (short for "operation code") specifies the operation to be performed by the processor.
#[allow(non_camel_case_types)]
#[derive(
    TryFromPrimitive,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    Enum,
)]
#[repr(u8)]
pub enum Opcode {
    // ALU
    ADD = 0,   // ADDSUB
    SUB = 1,   // ADDSUB
    MUL = 2,   // MUL
    MULT = 3,  // MUL
    MULTU = 4, // MUL
    DIV = 5,   // DIVREM
    DIVU = 6,  // DIVREM
    MOD = 7,   // DIVREM
    MODU = 8,  // DIVREM
    SLL = 9,   // SLL
    SRL = 10,  // SR
    SRA = 11,  // SR
    ROR = 12,  // SR
    SLT = 13,  // LT
    SLTU = 14, // LT
    AND = 15,  // BITWISE
    OR = 16,   // BITWISE
    XOR = 17,  // BITWISE
    NOR = 18,  // BITWISE
    CLZ = 19,  // CLO_CLZ
    CLO = 20,  // CLO_CLZ
    // Control FLow
    BEQ = 21,        // BRANCH
    BGEZ = 22,       // BRANCH
    BGTZ = 23,       // BRANCH
    BLEZ = 24,       // BRANCH
    BLTZ = 25,       // BRANCH
    BNE = 26,        // BRANCH
    Jump = 27,       // JUMP
    Jumpi = 28,      // JUMP
    JumpDirect = 29, // JUMP
    SYSCALL = 30,    // SYSCALL
    // Memory Op
    LB = 31,  // LOAD
    LBU = 32, // LOAD
    LH = 33,  // LOAD
    LHU = 34, // LOAD
    LW = 35,  // LOAD
    LWL = 36, // LOAD
    LWR = 37, // LOAD
    LL = 38,  // LOAD
    SB = 39,  // STORE
    SH = 40,  // STORE
    SW = 41,  // STORE
    SWL = 42, // STORE
    SWR = 43, // STORE
    SC = 44,  // STORE
    // Misc
    INS = 45,   // INS
    MADDU = 46, // MADDSUB
    MSUBU = 47, // MADDSUB
    MADD = 48,  // MADDSUB
    MSUB = 49,  // MADDSUB
    MEQ = 50,   // MOVCOND
    MNE = 51,   // MOVCOND
    WSBH = 52,  // WSBH
    EXT = 53,   // EXT
    TEQ = 54,   // TEQ
    SEXT = 55,  // SEXT

    // Syscall
    UNIMPL = 0xff,
}

impl Opcode {
    /// Get the mnemonic for the opcode.
    #[must_use]
    pub const fn mnemonic(&self) -> &str {
        match self {
            Opcode::ADD => "add",
            Opcode::SUB => "sub",
            Opcode::MULT => "mult",
            Opcode::MULTU => "multu",
            Opcode::MUL => "mul",
            Opcode::DIV => "div",
            Opcode::DIVU => "divu",
            Opcode::SLL => "sll",
            Opcode::SRL => "srl",
            Opcode::SRA => "sra",
            Opcode::ROR => "ror",
            Opcode::SLT => "slt",
            Opcode::SLTU => "sltu",
            Opcode::AND => "and",
            Opcode::OR => "or",
            Opcode::XOR => "xor",
            Opcode::NOR => "nor",
            Opcode::CLZ => "clz",
            Opcode::CLO => "clo",
            Opcode::BEQ => "beq",
            Opcode::BNE => "bne",
            Opcode::BGEZ => "bgez",
            Opcode::BLEZ => "blez",
            Opcode::BGTZ => "bgtz",
            Opcode::BLTZ => "bltz",
            Opcode::Jump => "jump",
            Opcode::Jumpi => "jumpi",
            Opcode::JumpDirect => "jump_direct",
            Opcode::LB => "lb",
            Opcode::LBU => "lbu",
            Opcode::LH => "lh",
            Opcode::LHU => "lhu",
            Opcode::LW => "lw",
            Opcode::LWL => "lwl",
            Opcode::LWR => "lwr",
            Opcode::LL => "ll",
            Opcode::SB => "sb",
            Opcode::SH => "sh",
            Opcode::SW => "sw",
            Opcode::SWL => "swl",
            Opcode::SWR => "swr",
            Opcode::SC => "sc",
            Opcode::SYSCALL => "syscall",
            Opcode::TEQ => "teq",
            Opcode::MEQ => "meq",
            Opcode::MNE => "mne",
            Opcode::SEXT => "seb",
            Opcode::WSBH => "wsbh",
            Opcode::EXT => "ext",
            Opcode::INS => "ins",
            Opcode::MADDU => "maddu",
            Opcode::MSUBU => "msubu",
            Opcode::MOD => "mod",
            Opcode::MODU => "modu",
            Opcode::MADD => "madd",
            Opcode::MSUB => "msub",
            Opcode::UNIMPL => "unimpl",
        }
    }

    /// Convert the opcode to a field element.
    #[must_use]
    pub fn as_field<F: Field>(self) -> F {
        F::from_u32(self as u32)
    }

    pub fn is_use_lo_hi_alu(&self) -> bool {
        matches!(
            self,
            Opcode::DIV
                | Opcode::DIVU
                | Opcode::MULT
                | Opcode::MULTU
                | Opcode::MADDU
                | Opcode::MSUBU
                | Opcode::MADD
                | Opcode::MSUB
        )
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

/// Byte Opcode.
///
/// This represents a basic operation that can be performed on a byte. Usually, these operations
/// are performed via lookup tables on that iterate over the domain of two 8-bit values. The
/// operations include both bitwise operations (AND, OR, XOR) as well as basic arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum ByteOpcode {
    /// Bitwise AND.
    AND = 0,
    /// Bitwise OR.
    OR = 1,
    /// Bitwise XOR.
    XOR = 2,
    /// Shift Left Logical.
    SLL = 3,
    /// Unsigned 8-bit Range Check.
    U8Range = 4,
    /// Shift Right with Carry.
    ShrCarry = 5,
    /// Unsigned Less Than.
    LTU = 6,
    /// Most Significant Bit.
    MSB = 7,
    /// Unsigned 16-bit Range Check.
    U16Range = 8,
    /// Bitwise NOR.
    NOR = 9,
    /// Parametric bit-width range check: `a1 < 2^b`, `b <= 16`.  Served by the
    /// dedicated `RangeChip` table, NOT the byte table.
    Range = 10,
}
