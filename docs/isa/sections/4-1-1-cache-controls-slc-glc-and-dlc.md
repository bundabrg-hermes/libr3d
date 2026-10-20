# 4.1.1. Cache Controls: SLC, GLC and DLC

_(pages 44-46)_

Scalar and vector memory instructions contain bits that control cache behavior. The SLC, GLC and DLC instruction bits influence cache behavior for loads, stores, and atomics.

### GLC

controls the graphics first-level cache

### SLC

controls the graphics L2 cache

### DLC

controls the Memory-Attached Last-Level cache (MALL) if it is present (ignored otherwise) Typically loads use GLC=0 (except for load-acquire). GLC=1 forces a miss in the first level cache and reads data rom the L2 cache. If there was a line in the GPU L0 that matched, it is invalidated; L2 is reread.

### Shader LOAD ops (load, sample, gather, etc…)

• For S_BUFFER_LOAD instructions, LLC_NOALLOC comes from V#.LLC_noalloc. For S_LOAD, LLC_NOALLOC is zero. • SMEM operations have SLC set to zero.

### Shader STORE / ATOMIC ops (all are device scope)

"Temporal Hint" = expect data to have temporal reuse. "SRD" = Shader Resource Descriptor • ISA.GLC ⇒ this is a scope bit for load operations (including sample, gather, etc…) ◦0 : CU (work-group) scope ◦1 : DEVICE scope ◦All stores/atomic ops are device scope (GLC has non-perf related functionality) • ISA.SLC ⇒ Temporal Hint for graphic client caches ◦0 : Regular ◦1 : Stream (non-temporal) • ISA.DLC ⇒ Temporal Hint for Infinity Cache ◦0 : Regular ◦1 : Non-temporal GLC is used by atomics to indicate: • 0: return nothing • 1: return pre-operation value from memory to VGPR
