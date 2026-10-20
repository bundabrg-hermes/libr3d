# 3.4.4. NULL

_(pages 32-32)_

NULL is a scalar source and destination. Reading NULL returns zero, writing to NULL has no effect (write data is discarded). NULL may be used anywhere scalar sources can normally be used: • When NULL is used as the destination of an SALU instruction, the instruction executes: SDST is not written but SCC is updated (if the instruction normally updates SCC). • NULL may not be used as an S#, V# or T#.
