---
mnemonic: S_BARRIER
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "61"
pages: [254]
source: RDNA3 ISA PDF p.254-254
---

# S_BARRIER (opcode 61)

Synchronize waves within a threadgroup. If not all waves of the threadgroup have been created yet, waits for entire group before proceeding. If some waves in the threadgroup have already terminated, this waits on only the surviving waves. Barriers are legal inside trap handlers. Barrier instructions do not wait for any counters to go to zero before issuing. If the barrier is being used to protect an outstanding memory operation use the appropriate S_WAITCNT instruction before the barrier.
