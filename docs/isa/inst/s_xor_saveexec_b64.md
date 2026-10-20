---
mnemonic: S_XOR_SAVEEXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "37"
pages: [228]
source: RDNA3 ISA PDF p.228-228
---

# S_XOR_SAVEEXEC_B64 (opcode 37)

Calculate bitwise XOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination register. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
