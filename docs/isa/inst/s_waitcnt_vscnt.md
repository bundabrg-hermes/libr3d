---
mnemonic: S_WAITCNT_VSCNT
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "24"
pages: [214, 215]
source: RDNA3 ISA PDF p.214-215
---

# S_WAITCNT_VSCNT (opcode 24)

Wait for the counts of outstanding vector store events -- vector memory stores and atomics that DO NOT return data -- to be at or below the specified level. This counter is not used in 'all-in-order' mode. Waits for the following condition to hold before continuing: To wait on a literal constant only, write 'null' for the GPR argument. This opcode may only appear inside a clause if the SGPR operand is set to NULL. See also S_WAITCNT.
