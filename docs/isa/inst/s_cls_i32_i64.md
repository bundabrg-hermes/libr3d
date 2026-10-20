---
mnemonic: S_CLS_I32_I64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "13"
pages: [220]
source: RDNA3 ISA PDF p.220-220
---

# S_CLS_I32_I64 (opcode 13)

Count the number of leading bits that are the same as the sign bit of a scalar input and store the result into a scalar register. Store -1 if all input bits are the same.
