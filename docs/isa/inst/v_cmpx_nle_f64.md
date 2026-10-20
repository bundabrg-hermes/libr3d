---
mnemonic: V_CMPX_NLE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "172"
pages: [344]
source: RDNA3 ISA PDF p.344-344
---

# V_CMPX_NLE_F64 (opcode 172)

Return 1 iff A not less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
