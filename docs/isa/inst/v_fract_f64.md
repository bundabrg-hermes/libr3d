---
mnemonic: V_FRACT_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "62"
pages: [290, 291]
source: RDNA3 ISA PDF p.290-291
---

# V_FRACT_F64 (opcode 62)

Compute the fractional portion of a double-precision float input and store the result in floating point format into a vector register.

### Notes

0.5ULP accuracy, denormals are accepted. This is intended to comply with the DX specification of fract where the function behaves like an extension of integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(- 1.2) = 0.8 in DX. Obey round mode, result clamped to 0x3fefffffffffffff.
