# 12.1.3. LDS Access Methods

_(pages 128-129)_

There are 3 forms of Local Data Share access:

### Direct Load

Loads a single DWORD from LDS and broadcasts the data to a VGPR across all lanes.

### Indexed load/store and Atomic ops

Load/store address comes from a VGPR and data to/from VGPR. LDS-ops require up to 3 inputs: 2data+1addr and immediate return VGPR.

### Parameter Interpolation Load

Reads pixel parameters from LDS per quad and loads them into one VGPR. Reads all 3 parameters per quad (P1, P1-P0 and P2-P0) and loads them into 3 lanes within the quad (the 4th lane receives zero). The following sections describe these methods.
