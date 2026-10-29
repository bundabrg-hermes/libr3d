#!/bin/bash
# Q62 w66 single A/B leg: launch serve (detached), wait, settle, measure 512-tok gen,
# parse engine tok/s, cleanup. Args: TAG DTYPE NSPEC SETTLE NTK
set -e
TAG=$1; DTYPE=$2; NSPEC=$3; SETTLE=$4; NTK=${5:-512}
NAME=w66_q62_$TAG
LOG=/home/brendan/scratch/r3d/w66_q62/leg_${NAME}.log
docker rm -f $NAME >/dev/null 2>&1 || true
rm -f /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log
docker run -d --name $NAME --memory=3g --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e ROCR_VISIBLE_DEVICES=1 \
  -v /home/brendan/projects/libr3d/libr3d/build_v164wip/radiance_home:/rhome \
  -v /srv/models/rad:/models -v /home/brendan/scratch/r3d/w66_q62:/scratch \
  c21457846d47 bash -c "/opt/radiance/bin/radiance --model /models/qwen3.8-27b-mxfp4.rad --radiance-home /rhome:/opt/radiance/share/radiance --mm-max-patches 0 --kv-cache-dtype $DTYPE --num-speculative-tokens $NSPEC --max-model-len 8192 --max-num-seqs 1 --placement all_vram --host 0.0.0.0 --port 9007 > /scratch/serve_${NAME}.log 2>&1" >/dev/null
# wait for listening
for i in $(seq 1 60); do
  if grep -q "server: listening" /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log 2>/dev/null; then break; fi
  if grep -q "startup stopped" /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log 2>/dev/null; then grep -E "^E " /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log | head -3; echo "LEGFAIL boot $NAME"; docker rm -f $NAME >/dev/null; exit 2; fi
  sleep 5
done
grep -q "server: listening" /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log 2>/dev/null || { tail -3 /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log 2>/dev/null; echo "LEGFAIL timeout $NAME"; docker rm -f $NAME >/dev/null; exit 2; }
SALT=$RANDOM$RANDOM
curl -s -o /dev/null -X POST http://localhost:9007/tokenize -H 'Content-Type: application/json' -d '{"prompt":"warm"}' || true
for i in $(seq 1 $SETTLE); do
  curl -s -o /dev/null --max-time 900 -X POST http://localhost:9007/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write 400 words on salt $SALT$i.\"}],\"max_tokens\":256,\"temperature\":0}" || true
done
NT=$(curl -s --max-time 900 -X POST http://localhost:9007/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write 500 words on salt ${SALT}meas.\"}],\"max_tokens\":$NTK,\"temperature\":0}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("usage",{}).get("completion_tokens",0))' || echo 0)
sleep 1
RATELINE=$(grep "tok/s" /home/brendan/scratch/r3d/w66_q62/serve_${NAME}.log | tail -1)
docker rm -f $NAME >/dev/null
echo "RESULT tag=$TAG dtype=$DTYPE nspec=$NSPEC ntok=$NT"
echo "RATELINE: $RATELINE"
