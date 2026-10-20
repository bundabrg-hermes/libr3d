# 9.4.1. Range Checking

_(pages 97-99)_

Buffer addresses are checked against the size of the memory buffer. Loads that are out of range return zero, and stores and atomics are dropped. Range checking is per-component for non-formatted loads and stores that are larger than one DWORD. Note that load/store_B64, B96 and B128 are considered "2-DWORD/3-DWORD/4- DWORD load/store", and each DWORD is bounds checked separately. The method of clamping is controlled by a 2-bit field in the buffer resource: OOB_SELECT (Out of Bounds select). Table 46. Buffer Out Of Bounds Selection Notes: 1. Loads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1). 2. Stores that are out-of-range do not store anything. 3. Load/store-format-* instruction and atomics are range-checked "all or nothing" - either entirely in or out. 4. Load/store-B{64,96,128} and range-check per component. For MTBUF, if any component of the thread is out of bounds, the whole thread is considered out of bounds and returns zero. For MUBUF, only the components that are out of bounds return zero.

### 9.4.1.1. Structured Buffer

The address calculation for swizzle_en==0 is: (unswizzled structured buffer) NumRecords for structured buffer is in units of stride.

### 9.4.1.2. Raw Buffer

NumRecords for raw buffer is in units of bytes. This is an exact range check, meaning it includes the payload and handles multi-DWORD and unaligned correctly. The stride field is ignored.

### 9.4.1.3. Scratch Buffer

The address calculation for swizzle_en = 0 is…(unswizzled scratch buffer) Swizzle of scratch buffer is also supported (and is typical). The MSBs of the TID (TID / 64) is folded into baseOffset. No range checking (using OOB mode 2).

### 9.4.1.4. Scalar Memory

Scalar memory does the following, that works with RAW buffers and unswizzled structured buffers: Address Out-of-Range if: offset >= ( (stride==0 ? 1 : stride) * num_records).

### Notes

1. Loads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1). Stores that are out of range do not write anything. 2. Load/store-format-* instruction and atomics are range-checked "all or nothing" - either entirely in or out. 3. Load/store-DWORD-x{2,3,4} perform range-check per component.
