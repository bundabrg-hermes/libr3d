# 7.2.5. Instructions using SGPRs as Mask or Carry

_(pages 71-71)_

Every VALU instruction can use SGPRs as a constant, but the following can read or write SGPRs as masks or carry: "VCC" in the above table refers to VCC in a VOP2 or VOPC encoding, or any SGPR specified in the SRC2 or SDST field for VOP3 encoding, except for DIV_FMAS that implicitly reads VCC (no choice). V_CMPX is the only VALU instruction that writes EXEC.
