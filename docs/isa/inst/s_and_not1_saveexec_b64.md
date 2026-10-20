---
mnemonic: S_AND_NOT1_SAVEEXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "49"
pages: [231]
source: RDNA3 ISA PDF p.231-231
---

# S_AND_NOT1_SAVEEXEC_B64 (opcode 49)

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination register. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
