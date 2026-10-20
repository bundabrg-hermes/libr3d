---
mnemonic: V_DOT2_BF16_BF16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "615"
pages: [437]
source: RDNA3 ISA PDF p.437-437
---

# V_DOT2_BF16_BF16 (opcode 615)

Dot product of packed brain-float values.

### Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are ignored.
