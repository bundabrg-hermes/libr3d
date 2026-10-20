---
mnemonic: V_FMAAK_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "56"
pages: [270, 271]
source: RDNA3 ISA PDF p.270-271
---

# V_FMAAK_F16 (opcode 56)

Multiply two FP16 values and add a literal constant using fused multiply-add.

### Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers. Supports round mode, exception flags, saturation.
