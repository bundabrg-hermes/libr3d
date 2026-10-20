---
mnemonic: S_ENDPGM_ORDERED_PS_DONE
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "50"
pages: [253]
source: RDNA3 ISA PDF p.253-253
---

# S_ENDPGM_ORDERED_PS_DONE (opcode 50)

End of program; signal that a wave has exited its POPS critical section and terminate wavefront. The hardware implicitly executes S_WAITCNT 0 and S_WAITCNT_VSCNT 0 before executing this instruction. This instruction is an optimization that combines S_SENDMSG(MSG_ORDERED_PS_DONE) and S_ENDPGM; there may be cases where the message needs to be sent separately, in which case the shader can be terminated with a normal S_ENDPGM instruction. See S_ENDPGM for additional variants.
