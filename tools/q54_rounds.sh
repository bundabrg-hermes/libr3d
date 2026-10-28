#!/usr/bin/env bash
# q54_rounds.sh -- three crossed rounds, then the adjudication. Run detached:
#   setsid nohup bash q54_rounds.sh > /home/brendan/scratch/r3d/q54/rounds.log 2>&1 < /dev/null &
# Precondition: spec9g STOPPED (timing exclusivity on GPU1).
set -u
SCR=/home/brendan/scratch/r3d/q54
REPO=/home/brendan/projects/libr3d/libr3d
for R in 1 2 3; do
  REV=$(( R % 2 == 0 ? 1 : 0 ))
  Q54_R=$R Q54_REV=$REV bash "$SCR/q54_ab.sh"
  echo "q54 round $R (rev=$REV) finished rc=$?"
done
cat "$SCR"/ab_r1.out "$SCR"/ab_r2.out "$SCR"/ab_r3.out > "$SCR/ab_all.log"
python3 "$REPO/tools/ab_parse.py" "$SCR/ab_all.log" > "$SCR/ab_verdict.txt" 2>&1
echo "ADJUDICATED rc=$? -> $SCR/ab_verdict.txt"
