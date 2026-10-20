---
mnemonic: S_MOVRELSD_2_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "68"
pages: [235]
source: RDNA3 ISA PDF p.235-235
---

# S_MOVRELSD_2_B32 (opcode 68)

Move data from a relatively-indexed scalar register into another relatively-indexed scalar register, using different offsets for each index.

### Notes

Example: The following instruction sequence performs the move s25 <= s17:
