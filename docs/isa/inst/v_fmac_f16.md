---
mnemonic: V_FMAC_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "54"
pages: [270]
source: RDNA3 ISA PDF p.270-270
---

# V_FMAC_F16 (opcode 54)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply- add.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
