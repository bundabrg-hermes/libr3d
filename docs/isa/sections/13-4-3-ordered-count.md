# 13.4.3. Ordered Count

_(pages 143-144)_

Ordered count generates a pointer in wave-creation order to an append buffer of unlimited size. Ordered Alloc generates a pointer to a ring buffer of finite size which is returned to the wave in "VDST". The ordered alloc counter can be issued up to 4 times from a shader. Ordered count and alloc use the same instruction - the difference is in how the GDS counters are initialized with their config registers. The GDS unit supports an instruction that operates on dedicated append/consume counters: • DS_ORDERED_COUNT Takes one value from the first valid lane and sends to GDS. For shaders that use this function, this instruction must be issued once and only once per wave. The GDS receives these in arbitrary order from different waves across the chip, but processes them in the order the waves were created. The GDS contains a large fifo to hold these pending requests.

### Instruction Fields

### ORDERED COUNT Targets

The OFFSET0[5:2] field of ordered-count instructions reference one of 16 registers in GDS. These are listed in the GDS section: GS NGG Streamout Instructions. See: GS NGG Streamout Instructions Only the ADD instruction may be used on targets that are 64 bits (offset[5:2] = 8 - 15). Exchange can only be used with offset[5:2] = 4 - 7.

### APPEND and CONSUME

Append and Consume count bits in EXEC and add or subtract the count from the GDS stored value. GDS now only operates on a single lane, but for Append & Consume the full EXEC mask is still considered.
