---
mnemonic: S_SENDMSG_RTN_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "77"
pages: [237]
source: RDNA3 ISA PDF p.237-237
---

# S_SENDMSG_RTN_B64 (opcode 77)

Send a message to upstream control hardware. SSRC[7:0] contains the message type encoded in the instruction directly (this instruction does not read an SGPR). The message is expected to return a response from the upstream control hardware and the result is written to SDST. Use s_waitcnt lgkmcnt(…) to wait for the response on the dependent instruction. S_SENDMSG_RTN* instructions return data in-order among themselves but out-of-order with other instructions that manipulate lgkmcnt (including S_SENDMSG and S_SENDMSGHALT). If the message returns a 32 bit value then this instruction fills the upper bits of SDST with zero. If SDST is VCC then VCCZ is undefined.
