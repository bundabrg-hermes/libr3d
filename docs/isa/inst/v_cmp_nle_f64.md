---
mnemonic: V_CMP_NLE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "44"
pages: [317]
source: RDNA3 ISA PDF p.317-317
---

# V_CMP_NLE_F64 (opcode 44)

Return 1 iff A not less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
