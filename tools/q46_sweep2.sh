#!/usr/bin/env bash
# Q46 follow-on sweep: the SERVE decode geometry (h256 24Q/4KV fold rows=1 --
# battery attn_bf family, qwen3.5-9b), widths inside max-model-len 4096.
# Same protocol as q46_sweep.sh: exclusive card, one binary, alternating arms,
# 3 rounds, hard gfx1100-alone assertion.
set -uo pipefail
REPO=$HOME/projects/libr3d/libr3d
if [ -z "${Q46_IN_CONTAINER:-}" ]; then
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    -e ROCR_VISIBLE_DEVICES=1 -e Q46_IN_CONTAINER=1 \
    -v $REPO:/work -w /work --shm-size=4g --entrypoint /bin/bash \
    radiance-dev:v1.11.0 /work/tools/q46_sweep2.sh
  exit $?
fi
cd /work
mkdir -p q46logs; LOG=q46logs/q46_sweep2.log; : > $LOG
say() { echo "$@" | tee -a $LOG; }
dv=$(./q46bin/dev0); echo "$dv" | tee -a $LOG
case "$dv" in "1 device(s) [0:gfx1100]") ;; *) echo "FATAL: not gfx1100 alone" | tee -a $LOG; exit 3;; esac
export LIBR3D_SO=/work/build_v154/radiance_home/kernels/libr3d.so
export BENCH_HD=256
say "== serve-geometry sweep start $(date -u +%H:%M:%S) =="
for R in 1 2 3; do
 for CTX in 512 1024 2048 4096; do
  for ARM in "p0:0:" "p1cu48:1:" "p1cu96:1:96"; do
    NAME=${ARM%%:*}; rest=${ARM#*:}; POL=${rest%%:*}; CU=${rest#*:}
    f=q46logs/s2_${CTX}_${NAME}_r${R}
    declare -a EV=(R3D_SPLIT_POLICY=$POL R3D_DEBUG_DECODE=1)
    [ -n "$CU" ] && EV+=(R3D_SPLIT_CU=$CU)
    env "${EV[@]}" ./q46bin/bench_attn_decode time 24 4 1 $CTX 15 8 > $f.txt 2>&1 || say "RUNFAIL $f"
    arm=$(grep -m1 -oE "spl=[0-9]+ fold=[01]" $f.txt)
    res=$(grep RESULT $f.txt)
    [ -z "$res" ] && say "EMPTY LEG $f"
    say "r$R ctx=$CTX $NAME ${arm:-NO-SPLITK-LINE} $res"
  done
 done
done
say "== ALL DONE $(date -u +%H:%M:%S) =="
