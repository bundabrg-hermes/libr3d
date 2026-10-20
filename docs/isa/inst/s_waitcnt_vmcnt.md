---
mnemonic: S_WAITCNT_VMCNT
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "25"
pages: [215]
source: RDNA3 ISA PDF p.215-215
---

# S_WAITCNT_VMCNT (opcode 25)

Wait for the counts of outstanding vector memory events -- everything except for memory stores and atomics- without-return -- to be at or below the specified level. When in 'all-in-order' mode, wait for all vector memory events. Waits for the following condition to hold before continuing: To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT. This opcode may only appear inside a clause if the SGPR operand is set to NULL. See also S_WAITCNT.
