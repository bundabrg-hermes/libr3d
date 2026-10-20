---
mnemonic: V_CMP_CLASS_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "126"
pages: [479]
source: RDNA3 ISA PDF p.479-479
---

# V_CMP_CLASS_F32 (opcode 126)

IEEE numeric class function specified in S1.u, performed on S0.f. The function reports true if the floating point value is any of the numeric types selected in S1.u according to the following list: S1.u[0] -- value is a signaling NAN. S1.u[1] -- value is a quiet NAN. S1.u[2] -- value is negative infinity. S1.u[3] -- value is a negative normal value. S1.u[4] -- value is a negative denormal value. S1.u[5] -- value is negative zero. S1.u[6] -- value is positive zero. S1.u[7] -- value is a positive denormal value. S1.u[8] -- value is a positive normal value. S1.u[9] -- value is positive infinity.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
