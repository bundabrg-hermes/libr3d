---
mnemonic: S_ENDPGM
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "48"
pages: [252]
source: RDNA3 ISA PDF p.252-252
---

# S_ENDPGM (opcode 48)

End of program; terminate wavefront. The hardware implicitly executes S_WAITCNT 0 and S_WAITCNT_VSCNT 0 before executing this instruction. See S_ENDPGM_SAVED for the context-switch version of this instruction and S_ENDPGM_ORDERED_PS_DONE for the POPS critical region version of this instruction.
