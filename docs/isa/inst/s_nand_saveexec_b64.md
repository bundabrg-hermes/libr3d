---
mnemonic: S_NAND_SAVEEXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "39"
pages: [228]
source: RDNA3 ISA PDF p.228-228
---

# S_NAND_SAVEEXEC_B64 (opcode 39)

Bitwise NAND with EXEC mask. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
