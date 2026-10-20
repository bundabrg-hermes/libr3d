---
mnemonic: S_NOR_SAVEEXEC_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "40"
pages: [228, 229]
source: RDNA3 ISA PDF p.228-229
---

# S_NOR_SAVEEXEC_B32 (opcode 40)

Bitwise NOR with EXEC mask. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
