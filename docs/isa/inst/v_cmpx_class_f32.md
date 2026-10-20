---
mnemonic: V_CMPX_CLASS_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "254"
pages: [357, 358]
source: RDNA3 ISA PDF p.357-358
---

# V_CMPX_CLASS_F32 (opcode 254)

IEEE numeric class function specified in S1.u, performed on S0.f. The function reports true if the floating point value is any of the numeric types selected in S1.u according to the following list: S1.u[0] -- value is a signaling NAN. S1.u[1] -- value is a quiet NAN. S1.u[2] -- value is negative infinity. S1.u[3] -- value is a negative normal value. S1.u[4] -- value is a negative denormal value. S1.u[5] -- value is negative zero. S1.u[6] -- value is positive zero. S1.u[7] -- value is a positive denormal value. S1.u[8] -- value is a positive normal value. S1.u[9] -- value is positive infinity.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
