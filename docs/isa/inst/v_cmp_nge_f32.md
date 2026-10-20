---
mnemonic: V_CMP_NGE_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "25"
pages: [312]
source: RDNA3 ISA PDF p.312-312
---

# V_CMP_NGE_F32 (opcode 25)

Return 1 iff A not greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
