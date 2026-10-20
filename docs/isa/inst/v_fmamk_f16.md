---
mnemonic: V_FMAMK_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "55"
pages: [270]
source: RDNA3 ISA PDF p.270-270
---

# V_FMAMK_F16 (opcode 55)

Multiply a FP16 value with a literal constant and add a second FP16 value using fused multiply-add.

### Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers. Supports round mode, exception flags, saturation.
