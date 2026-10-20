---
mnemonic: S_SLEEP
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "3"
pages: [243]
source: RDNA3 ISA PDF p.243-243
---

# S_SLEEP (opcode 3)

Cause a wave to sleep for up to ~8000 clocks. The wave sleeps for (64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0]) clocks. The exact amount of delay is approximate. Compare with S_NOP. When SIMM16[6:0] is zero then no sleep occurs.

### Notes

Examples:
