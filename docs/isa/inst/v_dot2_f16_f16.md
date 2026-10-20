---
mnemonic: V_DOT2_F16_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "614"
pages: [436, 437]
source: RDNA3 ISA PDF p.436-437
---

# V_DOT2_F16_F16 (opcode 614)

Dot product of packed FP16 values.

### Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are ignored.
