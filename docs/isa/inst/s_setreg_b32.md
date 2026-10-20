---
mnemonic: S_SETREG_B32
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "18"
pages: [213]
source: RDNA3 ISA PDF p.213-213
---

# S_SETREG_B32 (opcode 18)

Write some or all of the LSBs of source argument into a hardware register. The SIMM16 argument is encoded as follows:

### ID = SIMM16[5:0]

ID of hardware register to access.

### OFFSET = SIMM16[10:6]

LSB offset of register bits to access.

### SIZE = SIMM16[15:11]

Size of register bits to access, minus 1. Set this field to 31 to read/write all bits of the hardware register.
