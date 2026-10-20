---
mnemonic: V_MAX_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "810"
pages: [447, 448]
source: RDNA3 ISA PDF p.447-448
---

# V_MAX_F64 (opcode 810)

Select the maximum of two floating point inputs and store the result into a vector register.

### Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation. Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can be a denorm and this operation outputs as a denorm regardless of output denorm mode.
