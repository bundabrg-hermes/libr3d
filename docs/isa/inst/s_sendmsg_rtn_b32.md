---
mnemonic: S_SENDMSG_RTN_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "76"
pages: [237]
source: RDNA3 ISA PDF p.237-237
---

# S_SENDMSG_RTN_B32 (opcode 76)

Send a message to upstream control hardware. SSRC[7:0] contains the message type encoded in the instruction directly (this instruction does not read an SGPR). The message is expected to return a response from the upstream control hardware and the result is written to SDST. Use s_waitcnt lgkmcnt(…) to wait for the response on the dependent instruction. S_SENDMSG_RTN* instructions return data in-order among themselves but out-of-order with other instructions that manipulate lgkmcnt (including S_SENDMSG and S_SENDMSGHALT). If the message returns a 64 bit value then only the lower 32 bits are written to SDST. If SDST is VCC then VCCZ is undefined.
