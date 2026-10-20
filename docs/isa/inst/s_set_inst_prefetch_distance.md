---
mnemonic: S_SET_INST_PREFETCH_DISTANCE
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "4"
pages: [243]
source: RDNA3 ISA PDF p.243-243
---

# S_SET_INST_PREFETCH_DISTANCE (opcode 4)

Change instruction prefetch mode. This controls how many cachelines ahead of the current PC the shader will try to prefetch. SIMM16[1:0] specifies the prefetch mode to switch to. Prefetch modes are:

### PREFETCH_SAFE (0x0)

Reserved, do not use.

### PREFETCH_1_LINE (0x1)

Prefetch 1 cache line ahead of PC; keep 2 lines behind PC.

### PREFETCH_2_LINES (0x2)

Prefetch 2 cache lines ahead of PC; keep 1 line behind PC.

### PREFETCH_3_LINES (0x3)

Prefetch 3 cache lines ahead of PC; keep 0 lines behind PC. SIMM16[15:2] must be set to zero.
