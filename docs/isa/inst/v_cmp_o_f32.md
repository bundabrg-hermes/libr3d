---
mnemonic: V_CMP_O_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "23"
pages: [311, 312]
source: RDNA3 ISA PDF p.311-312
---

# V_CMP_O_F32 (opcode 23)

Return 1 iff A orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
