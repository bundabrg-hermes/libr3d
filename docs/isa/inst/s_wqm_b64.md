---
mnemonic: S_WQM_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "29"
pages: [225]
source: RDNA3 ISA PDF p.225-225
---

# S_WQM_B64 (opcode 29)

Given an active pixel mask in a scalar input, calculate whole quad mode mask for that input, store the result into a scalar register and set SCC iff the result is nonzero. In whole quad mode, if any pixel in a quad is active then all pixels of the quad are marked active.
