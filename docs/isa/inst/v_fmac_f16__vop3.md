---
mnemonic: V_FMAC_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "310"
pages: [412]
source: RDNA3 ISA PDF p.412-412
---

# V_FMAC_F16 (opcode 310)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply- add.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
