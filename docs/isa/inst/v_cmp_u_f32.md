---
mnemonic: V_CMP_U_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "24"
pages: [312]
source: RDNA3 ISA PDF p.312-312
---

# V_CMP_U_F32 (opcode 24)

Return 1 iff A not orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
