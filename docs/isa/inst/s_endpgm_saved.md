---
mnemonic: S_ENDPGM_SAVED
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "49"
pages: [253]
source: RDNA3 ISA PDF p.253-253
---

# S_ENDPGM_SAVED (opcode 49)

End of program; signal that a wave has been saved by the context-switch trap handler and terminate wavefront. The hardware implicitly executes S_WAITCNT 0 and S_WAITCNT_VSCNT 0 before executing this instruction. See S_ENDPGM for additional variants.
