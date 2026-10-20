---
mnemonic: V_MAX_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "57"
pages: [271, 272]
source: RDNA3 ISA PDF p.271-272
---

# V_MAX_F16 (opcode 57)

Select the maximum of two floating point inputs and store the result into a vector register.

### Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation. Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can be a denorm and this operation outputs as a denorm regardless of output denorm mode.
