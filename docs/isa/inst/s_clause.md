---
mnemonic: S_CLAUSE
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "5"
pages: [243, 244, 245]
source: RDNA3 ISA PDF p.243-245
---

# S_CLAUSE (opcode 5)

Mark the beginning of a clause. The next instruction determines the clause type, which may be one of the following types. • Image Load (non-sample instructions ) • Image Sample • Image Store • Image Atomic • Buffer/Global/Scratch Load • Buffer/Global/Scratch Store • Buffer/Global/Scratch Atomic • Flat Load • Flat Store • Flat Atomic • LDS (loads, stores, atomics may be in same clause) • Scalar Memory • Vector ALU Once the clause type is determined, any instruction encountered within the clause that is not of the same type (and not an internal instruction described below) is illegal and may lead to undefined behaviour. Attempting to issue S_CLAUSE while inside a clause is also illegal. Instructions that are processed internally do not interrupt the clause. The following instructions are internal: • S_NOP, • S_WAITCNT and its variants, unless they read an SGPR, • S_SLEEP, • S_DELAY_ALU. Halting or killing a wave breaks the clause. VALU exceptions and other traps that cause the shader to enter its trap handler breaks the clause. The single-step debug mode breaks the clause. The clause length must be between 2 and 63 instructions, inclusive. Clause breaks may be from 1 to 15, or may be disabled entirely. Clause length and breaks are encoded in the SIMM16 argument as follows:

### LENGTH = SIMM16[5:0]

This field is set to the logical number of instructions in the clause, minus 1 (e.g. if a clause has 4 instructions, program this field to 3). The minimum number of instructions required for a clause is 2 and the maximum number of instructions is 63, therefore this field must be programmed in the range [1, 62] inclusive.

### BREAK_SPAN = SIMM16[11:8]

This field is set to the number of instructions to issue before each clause break. If set to zero then there are no clause breaks. If set to nonzero value then the maximum number of instructions between clause breaks is 15. The following instruction types cannot appear in a clause: • SALU • Export • Branch • Message • LDSDIR • VINTERP • GDS To schedule an S_WAITCNT or S_DELAY_ALU instruction for the first instruction in the clause, the waitcnt/delay instruction must appear before the S_CLAUSE instruction so that S_CLAUSE can accurately determine the clause type. S_DELAY_ALU must not appear inside a clause. The features are orthogonal; ALU clauses should be structured to avoid any stalling.
