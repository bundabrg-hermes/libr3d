---
mnemonic: S_WAITCNT
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "9"
pages: [247]
source: RDNA3 ISA PDF p.247-247
---

# S_WAITCNT (opcode 9)

Wait for the counts of outstanding lds, vector-memory and export/vmem-write-data to be at or below the specified levels. The SIMM16 argument is encoded as:

### EXP = SIMM16[2:0]

Export wait count. 0x7 means do not wait on EXPCNT.

### LGKM = SIMM16[9:4]

LGKM wait count. 0x3f means do not wait on LGKMCNT.

### VM = SIMM16[15:10]

VM wait count. 0x3f means do not wait on VMCNT. Waits for all of the following conditions to hold before continuing: VMCNT only counts vector memory loads, image sample instructions, and vector memory atomics that return data. Contrast with the VSCNT counter. See also S_WAITCNT_VSCNT.
