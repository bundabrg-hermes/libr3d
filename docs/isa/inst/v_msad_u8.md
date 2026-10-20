---
mnemonic: V_MSAD_U8
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "569"
pages: [424]
source: RDNA3 ISA PDF p.424-424
---

# V_MSAD_U8 (opcode 569)

Masked sum of absolute differences with accumulation, overflow into upper bits is allowed. Components where the reference value in S1 is zero are not included in the sum.
