---
mnemonic: V_FRACT_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "95"
pages: [298]
source: RDNA3 ISA PDF p.298-298
---

# V_FRACT_F16 (opcode 95)

Compute the fractional portion of an FP16 float input and store the result in floating point format into a vector register.

### Notes

0.5ULP accuracy, denormals are accepted. This is intended to comply with the DX specification of fract where the function behaves like an extension of integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(- 1.2) = 0.8 in DX.
