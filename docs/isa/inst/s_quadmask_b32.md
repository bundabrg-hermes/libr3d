---
mnemonic: S_QUADMASK_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "26"
pages: [224]
source: RDNA3 ISA PDF p.224-224
---

# S_QUADMASK_B32 (opcode 26)

Reduce a pixel mask from the scalar input into a quad mask, store the result in a scalar register and set SCC iff the result is nonzero.

### Notes

To perform the inverse operation see S_BITREPLICATE_B64_B32.
