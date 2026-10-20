---
mnemonic: V_FRACT_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "32"
pages: [282]
source: RDNA3 ISA PDF p.282-282
---

# V_FRACT_F32 (opcode 32)

Compute the fractional portion of a single-precision float input and store the result in floating point format into a vector register.

### Notes

0.5ULP accuracy, denormals are accepted. This is intended to comply with the DX specification of fract where the function behaves like an extension of integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(- 1.2) = 0.8 in DX. Obey round mode, result clamped to 0x3f7fffff.
