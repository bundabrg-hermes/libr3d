---
mnemonic: S_OR_NOT1_SAVEEXEC_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "50"
pages: [231]
source: RDNA3 ISA PDF p.231-231
---

# S_OR_NOT1_SAVEEXEC_B32 (opcode 50)

Calculate bitwise OR on the scalar input and the negation of the EXEC mask, store the calculated result into the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination register. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
