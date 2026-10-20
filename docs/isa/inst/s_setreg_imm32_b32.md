---
mnemonic: S_SETREG_IMM32_B32
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "19"
pages: [213, 214]
source: RDNA3 ISA PDF p.213-214
---

# S_SETREG_IMM32_B32 (opcode 19)

Write some or all of the LSBs of a 32-bit literal constant into a hardware register; this instruction requires a 32- bit literal constant. The SIMM16 argument is encoded as follows:

### ID = SIMM16[5:0]

ID of hardware register to access.

### OFFSET = SIMM16[10:6]

LSB offset of register bits to access.

### SIZE = SIMM16[15:11]

Size of register bits to access, minus 1. Set this field to 31 to read/write all bits of the hardware register.
