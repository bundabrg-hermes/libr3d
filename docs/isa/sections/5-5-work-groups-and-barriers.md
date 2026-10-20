# 5.5. Work-groups and Barriers

_(pages 51-52)_

Work-groups are collections of waves running on the same work-group processor that can synchronize and share data. Up to 1024 work-items (16 wave64’s or 32 wave32’s) can be combined into a work-group. When multiple waves are in a work-group, the S_BARRIER instruction can be used to force each wave to wait until all other waves reach the same instruction; then, all waves continue. Work-groups of a single wave treat all barrier instructions as S_NOP. If a wave executes an S_BARRIER before all of the waves of the work-group have been created, the wave waits until the work-group is complete. Any wave may terminate early using S_ENDPGM, and the barrier is considered satisfied when the remaining live waves reach their barrier instruction.
