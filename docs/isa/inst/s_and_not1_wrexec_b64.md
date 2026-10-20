---
mnemonic: S_AND_NOT1_WREXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "55"
pages: [233]
source: RDNA3 ISA PDF p.233-233
---

# S_AND_NOT1_WREXEC_B64 (opcode 55)

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero. Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is intended to help accelerate waterfalling.

### Notes

In particular, the following sequence of waterfall code is optimized by using a WREXEC instead of two separate scalar ops:
