---
mnemonic: S_AND_NOT0_WREXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "53"
pages: [232]
source: RDNA3 ISA PDF p.232-232
---

# S_AND_NOT0_WREXEC_B64 (opcode 53)

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero. Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is intended to help accelerate waterfalling.
