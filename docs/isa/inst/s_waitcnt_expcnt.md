---
mnemonic: S_WAITCNT_EXPCNT
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "26"
pages: [215]
source: RDNA3 ISA PDF p.215-215
---

# S_WAITCNT_EXPCNT (opcode 26)

Wait for the counts of outstanding export events to be at or below the specified level. Waits for the following condition to hold before continuing: To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT. This opcode may only appear inside a clause if the SGPR operand is set to NULL. See also S_WAITCNT.
