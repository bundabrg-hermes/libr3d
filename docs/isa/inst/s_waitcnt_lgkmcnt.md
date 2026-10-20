---
mnemonic: S_WAITCNT_LGKMCNT
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "27"
pages: [215, 216]
source: RDNA3 ISA PDF p.215-216
---

# S_WAITCNT_LGKMCNT (opcode 27)

Wait for the counts of outstanding DS (LG), scalar memory (K) and message (M) events to be at or below the specified level. Waits for the following condition to hold before continuing: To wait on a literal constant only, write 'null' for the GPR argument or use S_WAITCNT. This opcode may only appear inside a clause if the SGPR operand is set to NULL. See also S_WAITCNT.
