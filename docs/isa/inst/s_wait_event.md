---
mnemonic: S_WAIT_EVENT
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "11"
pages: [247, 248]
source: RDNA3 ISA PDF p.247-248
---

# S_WAIT_EVENT (opcode 11)

Wait for an event to occur or a condition to be satisfied before continuing. The SIMM16 argument specifies which event(s) to wait on.

### DONT_WAIT_EXPORT_READY = SIMM16[0]

If this value is ZERO then sleep until the export_ready bit is 1. If the export_ready bit is already 1, no sleep occurs. Effect is the same as the export_ready check performed before issuing an export instruction. No wait occurs if this value is ONE. This wait can be broken or preempted by KILL, context-save, host trap, single-step or trap after instruction events. IB waits for the event to occur before processing internal exceptions which can delay entry to the trap handler for a significant amount of time.
