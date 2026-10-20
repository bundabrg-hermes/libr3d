# 13.4.1. GWS and Ordered Count Programming Rule

_(pages 143-143)_

"GWS" instructions (ordered count and GWS*) must be issued as a single instruction clause of the form: S_WAITCNT LGKMcnt==0 // this is only necessary if there might be any outstanding GDS instructions GWS_instruction S_WAITCNT LGKMcnt==0 <any instruction except: S_ENDPGM (pad with NOP if the next instruction is s_endpgm) Before issuing a GWS or Ordered Count instruction, the user must make sure that there are no outstanding GDS instructions. Failure to do this may cause a "NACK" to arrive out of order. Programming Rule: the source and destination VGPRs in a GWS or ordered count instruction must not be the same. When an ordered count operation is NACK’d, the destination VGPR may be written with data. If this VGPR is the same as the source VGPR, that prevents the instruction from being replayed later if it was interrupted due to a context switch.
