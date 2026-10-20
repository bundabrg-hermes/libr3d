# 8.4. Alignment and Bounds Checking

_(pages 87-87)_

### SDST

The value of SDST must be even for fetches of two DWORDs, or a multiple of four for larger fetches. If this rule is not followed, invalid data can result.

### SBASE

The value of SBASE must be even for S_BUFFER_LOAD (specifying the address of an SGPR which is a multiple of four). If SBASE is out-of-range, the value from SGPR0 is used.

### OFFSET

The value of OFFSET has no alignment restrictions.
