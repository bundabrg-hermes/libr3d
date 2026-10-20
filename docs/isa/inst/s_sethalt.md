---
mnemonic: S_SETHALT
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "2"
pages: [242]
source: RDNA3 ISA PDF p.242-242
---

# S_SETHALT (opcode 2)

Set or clear the HALT or FATAL_HALT status bits. The particular status bit is chosen by halt type control as indicated in SIMM16[2]; 0 = HALT bit select; 1 = FATAL_HALT bit select. When halt type control is set to 0 (HALT bit select): Set HALT bit to value of SIMM16[0]; 1 = halt, 0 = clear HALT bit. The halt flag is ignored while PRIV == 1 (inside trap handlers) but the shader halts after the handler returns if HALT is still set at that time. When halt type control is set to 1 (FATAL HALT bit select): Set FATAL_HALT bit to value of SIMM16[0]; 1 = fatal_halt, 0 = clear FATAL_HALT bit. Setting the fatal_halt flag halts the shader in or outside of the trap handlers.
