---
mnemonic: V_CMP_LE_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "51"
pages: [467]
source: RDNA3 ISA PDF p.467-467
---

# V_CMP_LE_I16 (opcode 51)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
