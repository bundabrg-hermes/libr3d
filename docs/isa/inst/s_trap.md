---
mnemonic: S_TRAP
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "16"
pages: [248]
source: RDNA3 ISA PDF p.248-248
---

# S_TRAP (opcode 16)

Enter the trap handler. This instruction may be generated internally as well in response to a host trap (HT = 1) or an exception. TrapID 0 is reserved for hardware use and should not be used in a shader-generated trap.
