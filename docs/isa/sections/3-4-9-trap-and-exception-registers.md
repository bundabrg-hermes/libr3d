# 3.4.9. Trap and Exception registers

_(pages 34-35)_

Each type of exception can be enabled or disabled independently by setting, or clearing, bits in the TRAPSTS register’s EXCP_EN field. This section describes the registers that control and report shader exceptions. Trap temporary SGPRs (TTMP*) are privileged for writes - they can be written only when in the trap handler (STATUS.PRIV = 1). TTMPs cannot be read by the user shader (returns zero). When the shader is not privileged (STATUS.PRIV==0), writes to these are ignored. TMA and TBA are read-only; they can be accessed through S_SENDMSG_RTN. When a trap is taken (either user initiated, exception or host initiated), the shader hardware generates an S_TRAP instruction. This loads trap information into a pair of SGPRS: HT is set to one for host initiated traps, and zero for user traps (s_trap) or exceptions. TRAP_ID is zero for exceptions, or the user/host trapID for those traps.

### STATUS . TRAP_EN

This bit tells the shader whether or not a trap handler is present. When one is not present, traps are not taken no matter whether they’re floating point, user or host-initiated traps. When the trap handler is present, the wave uses an extra 16 SGPRs for trap processing. If trap_en == 0, all traps and exceptions are ignored, and s_trap is converted by hardware to NOP.

### MODE . EXCP_EN[8:0]

Exception enable mask. Defines which of the sources of exception cause the shader to jump to the trap handler when the exception occurs. 1 = enable traps; 0 = disable traps. MEMVIOL and Illegal-Instruction jump to the trap handler and cannot be masked off.

### TRAPSTS Register

TRAPSTS contains information about traps and exceptions, and may be written by user shader or trap handler.
