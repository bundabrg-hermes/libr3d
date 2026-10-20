---
mnemonic: V_CMPX_LT_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "177"
pages: [345]
source: RDNA3 ISA PDF p.345-345
---

# V_CMPX_LT_I16 (opcode 177)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
