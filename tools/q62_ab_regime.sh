#!/bin/bash
# Q62 w66 regime driver: 5 interleaved rounds F,B for one regime. Runs inside a
# container with host docker CLI+socket mounted; card work is sibling containers.
# Args: REGIME (df0|df3)
R=$1
NS=$([ "$R" = "df0" ] && echo 0 || echo 3)
OUT=/home/brendan/scratch/r3d/w66_q62/ab_${R}.txt
: > $OUT
for r in 1 2 3 4 5; do
  for d in fp8 bf16; do
    T=${R}${d}r${r}
    bash /home/brendan/scratch/r3d/w66_q62/leg.sh $T $d $NS 2 512 >> $OUT 2>&1 || echo "LEGFAIL $T" >> $OUT
  done
done
echo "REGIME_DONE $R" >> $OUT
