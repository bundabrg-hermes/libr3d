---
mnemonic: S_AND_SAVEEXEC_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "32"
pages: [226]
source: RDNA3 ISA PDF p.226-226
---

# S_AND_SAVEEXEC_B32 (opcode 32)

Calculate bitwise AND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination register. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
