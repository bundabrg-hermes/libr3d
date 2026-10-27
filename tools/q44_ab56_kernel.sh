#!/usr/bin/env bash
# q44_ab.sh <round> — verify-band kernel A/B on v159, same binary, lever the only axis.
# A/B adjacency per leg within the round; rot=4 (cold, 4 planes > 96MB IC); Q35 shapes.
set -u
R=$1
SCR=/home/brendan/scratch/r3d/q44
LIB=/home/brendan/projects/libr3d/libr3d/build_v159/radiance_home/kernels
IMG=radiance-dev:v1.11.0
DEV=(--device /dev/kfd --device /dev/dri --group-add video --cap-add=SYS_ADMIN --security-opt seccomp=unconfined)
out=$SCR/ab56_r$R.out; : > $out
leg() { # leg <lmap> <M> <K> <N> <legname>
  local lm=$1 M=$2 K=$3 N=$4 n=$5
  docker run --rm --memory=2400m --memory-swap=2400m "${DEV[@]}" \
      -e HIP_VISIBLE_DEVICES=1 -e LIBR3D_SO=/blib/libr3d.so -e R3D_DEBUG_GEMM=1 \
      -e R3D_Q8X_LMAP=$lm -v "$SCR":/scratch -v "$LIB":/blib:ro "$IMG" \
      bash -lc "cd /scratch && ./m4prof q8x $M $K $N 24 4" \
      > $SCR/ab56_r${R}_${n}_${lm}.res 2> $SCR/ab56_r${R}_${n}_${lm}.err
  # marker BEFORE the RESULT lines so the adjudicator pairs them correctly
  echo "LEG $n lm=$lm rc=$? liveness=$(grep -c 'arm launched' $SCR/ab56_r${R}_${n}_${lm}.err)" >> $out
  cat $SCR/ab56_r${R}_${n}_${lm}.res >> $out
  grep -m2 "arm launched" $SCR/ab56_r${R}_${n}_${lm}.err >> $out
}
for s in "A 12288 4096" "B 24576 4096" "C 4096 12288"; do set -- $s
  for M in 5 6; do leg 0 $M $3 $2 ${1}_m$M; leg 1 $M $3 $2 ${1}_m$M; done
done
echo "ROUND $R DONE" >> $out
