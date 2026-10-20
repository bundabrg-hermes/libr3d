---
mnemonic: S_BITREPLICATE_B64_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "20"
pages: [222]
source: RDNA3 ISA PDF p.222-222
---

# S_BITREPLICATE_B64_B32 (opcode 20)

Substitute each bit of a 32 bit scalar input with two instances of itself and store the result into a 64 bit scalar register.

### Notes

This opcode can be used to convert a quad mask into a pixel mask; given quad mask in s0, the following sequence produces a pixel mask in s2: To perform the inverse operation see S_QUADMASK_B64.
