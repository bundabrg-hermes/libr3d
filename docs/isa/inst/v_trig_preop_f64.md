---
mnemonic: V_TRIG_PREOP_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "815"
pages: [449]
source: RDNA3 ISA PDF p.449-449
---

# V_TRIG_PREOP_F64 (opcode 815)

Look Up 2/PI (S0.f64) with segment select S1.u[4:0]. This operation returns an aligned, double precision segment of 2/PI needed to do range reduction on S0.f64 (double-precision value). Multiple segments can be specified through S1.u[4:0]. Rounding is round-to-zero. Large inputs (exp > 1968) are scaled to avoid loss of precision through denormalization.
