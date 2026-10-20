---
mnemonic: DS_ORDERED_COUNT
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "63"
pages: [526, 527]
source: RDNA3 ISA PDF p.526-527
---

# DS_ORDERED_COUNT (opcode 63)

GDS-only: Intercepted by GDS and processed by ordered append module. The ordered append module queues request until this request wave is the oldest in the queue at which time the oldest wave request is dispatched to the DS with an atomic opcode indicated by OFFSET1[5:4]. Unlike append/consume this operation is sent even if there are no valid lanes when it is issued. The GDS adds zero and advances the tracking walker that needs to match up with the dispatch counter. The following attributes are encoded in the instruction: • OFFSET0[7:2] contains the ordered_count_index (in dwords). • OFFSET1[0] contains the wave_release flag. • OFFSET1[1] contains the wave_done flag. • OFFSET1[5:4] contains the ord_idx_opcode: 2'b00 = DS_ADD_RTN_U32, 2'b01 = DS_STOREXCHG_RTN_B32, 2'b11 = DS_WRAP_RTN_B32. • VGPR_DST is the VGPR the result is written to. • VGPR_ADDR specifies the increment in the first valid lane. If no lanes are valid (EXEC = 0) then the increment is zero. • M0 normally carries {16'gds_base, 16'gds_size} for GDS usage. gds_base[15:2] is ordered_count_base[13:0] (in dwords) and gds_size is used to hold the logical_wave_id, the width is based on total number of waves in the chip. The wave type is determined automatically based on the ME_ID and QUEUE_ID of the wavefront.
