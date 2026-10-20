---
mnemonic: S_OR_NOT0_SAVEEXEC_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "46"
pages: [230]
source: RDNA3 ISA PDF p.230-230
---

# S_OR_NOT0_SAVEEXEC_B32 (opcode 46)

Calculate bitwise OR on the EXEC mask and the negation of the scalar input, store the calculated result into the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination register. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
