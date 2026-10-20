---
mnemonic: S_NOP
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "0"
pages: [242]
source: RDNA3 ISA PDF p.242-242
---

# S_NOP (opcode 0)

Do nothing. Delay issue of next instruction by a small, fixed amount. Insert 0..15 wait states based on SIMM16[3:0]. 0x0 means the next instruction can issue on the next clock, 0xf means the next instruction can issue 16 clocks later.

### Notes

Examples:
