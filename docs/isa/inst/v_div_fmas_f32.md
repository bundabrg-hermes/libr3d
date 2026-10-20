---
mnemonic: V_DIV_FMAS_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "567"
pages: [424]
source: RDNA3 ISA PDF p.424-424
---

# V_DIV_FMAS_F32 (opcode 567)

Single precision FMA with fused scale. This opcode performs a standard Fused Multiply-Add operation and conditionally scales the resulting exponent if VCC is set.

### Notes

Input denormals are not flushed, but output flushing is allowed.
