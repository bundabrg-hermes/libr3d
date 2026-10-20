---
mnemonic: V_CMPX_NE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "205"
pages: [351, 352]
source: RDNA3 ISA PDF p.351-352
---

# V_CMPX_NE_U32 (opcode 205)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
