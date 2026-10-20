---
mnemonic: S_WAKEUP
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "52"
pages: [253]
source: RDNA3 ISA PDF p.253-253
---

# S_WAKEUP (opcode 52)

Allow a wave to 'ping' all the other waves in its threadgroup to force them to wake up early from an S_SLEEP instruction. The ping is ignored if the waves are not sleeping. This allows for efficient polling on a memory location. The waves which are polling can sit in a long S_SLEEP between memory reads, but the wave which writes the value can tell them all to wake up early now that the data is available. This method is also safe from races because if any wave misses the ping, everything is expected to work fine (waves which missed it just complete their S_SLEEP). If the wave executing S_WAKEUP is in a threadgroup (in_wg set), then it wakes up all waves associated with the same threadgroup ID. Otherwise, S_WAKEUP is treated as an S_NOP.
