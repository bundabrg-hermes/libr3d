---
mnemonic: V_CVT_OFF_F32_I4
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "398"
pages: [378]
source: RDNA3 ISA PDF p.378-378
---

# V_CVT_OFF_F32_I4 (opcode 398)

Convert from a signed 4-bit integer to a single-precision float using an offset table and store the result into a vector register. Used for interpolation in shader. Lookup table on S0[3:0]: S0 binary Result 1000 -0.5000f 1001 -0.4375f 1010 -0.3750f 1011 -0.3125f 1100 -0.2500f 1101 -0.1875f 1110 -0.1250f 1111 -0.0625f 0000 +0.0000f 0001 +0.0625f 0010 +0.1250f 0011 +0.1875f 0100 +0.2500f 0101 +0.3125f 0110 +0.3750f 0111 +0.4375f
