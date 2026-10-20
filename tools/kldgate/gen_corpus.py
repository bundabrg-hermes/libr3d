#!/usr/bin/env python3
"""Generate corpus.jsonl for kldgate. Prompts span the shapes real traffic
presents: long code (TS/C++/Python), prose, math, structured data, mixed.
Target >= 512 prefill tokens each; one short one to keep coverage of the
small-M edge."""
import json
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl")

P = []

P.append(("ts-component", """You are a code assistant. Continue the following TypeScript exactly where it stops.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface RoomMember {
  id: string;
  room: string;
  role: "owner" | "editor" | "viewer";
  lastSeen: Date;
}

interface Room {
  id: string;
  name: string;
  members: RoomMember[];
  createdAt: string;
  maxConcurrency: number;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function RoomsPanel({ onSelect }: { onSelect: (room: Room) => void }) {
  const [editing, setEditing] = useState<Room | null>(null);
  const qc = useQueryClient();
  const rooms = useQuery({ queryKey: ["rooms"], queryFn: () => api<Room[]>("/rooms") });
  const rename = useMutation({
    mutationFn: (room: Room) =>
      api<Room>(`/rooms/${room.id}`, { method: "PATCH", body: JSON.stringify({ name: room.name }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rooms"] }),
  });
  // The rail keeps focus in the list so keyboard users never lose their place."""))

P.append(("cpp-kernel", """You are a GPU kernel reviewer. Continue the implementation comment and code.

// r3d_gemv_bf16 -- decode (M==1) row kernel for the RDNA3 path.
// One wave32 per output row: the wave strides the K dimension in 16-byte
// (8 x bf16) vectors, converts to f32 at the load boundary, accumulates
// ascending-k, and butterflies the row sum across the 32 lanes. The contract
// is libref's: accumulation order ascending in k, no FMA contraction, and the
// result is rounded once, at the store.
__global__ void r3d_gemv_bf16(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w,
                              float* __restrict__ y,
                              int K, int N) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= N) return;
  const int lane = threadIdx.x;
  const __nv_bfloat16* wr = w + (size_t)row * K;
  float acc = 0.f;
  // eight-wide vector loads: K is always a multiple of 256 in this model family
  for (int k = lane * 8; k < K; k += 32 * 8) {
    uint4 xv = *reinterpret_cast<const uint4*>(x + k);
    uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
    const __nv_bfloat16* xe = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc += __bfloat162float(xe[i]) * __bfloat162float(we[i]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down("""))

P.append(("prose-essay", """Continue the essay in the same voice and register.

The trouble with any machine that remembers is that memory is not a warehouse; it is a
conversation. A librarian who files every book by the sound of its title will find, sooner or
later, that the reader who asks for "the book about the war and the bee" is asking a question the
catalog cannot parse, and the answer sitting three shelves away might as well be in another
country. We built our retrieval systems as filing cabinets because filing cabinets were what we
understood, and then we acted surprised when the machines remembered everything except the
things we meant. There is an older craft hiding inside all of this. The medieval memory arts
taught that a location is not a container but a relationship: you did not store an image in a
place, you made the place and the image the same object, so that walking the imagined street was
thinking itself. When our engines embed a sentence into a vector and call the coordinates a
"location," they are, without crediting anyone, doing what a Roman orator did with an imagined
peristyle. The difference is one of accountability. The orator knew the memory was a construction,
flawed and personal, and the ledger kept by the filing cabinet gives the machine's construction
the confidence of a land registry. What should worry us is not that machines forget. It is that
they misremember with the paperwork of certainty, and that we have trained ourselves to read
paperwork as truth. The repair is not better filing. It is teaching the system, as we teach an
apprentice, that every act of recall is an act of reconstruction, and that reconstruction owes a"""))

P.append(("math-derivation", """Continue the derivation line by line, same notation.

Let the residual stream at layer l be h_l in R^d, and let the RMSNorm be
  hhat_l = h_l * g_l / sqrt(mean(h_l^2) + eps).
The attention block writes h_{l+1} = h_l + W_O [head_1; ...; head_H] with
  head_j = softmax(Q_j K_j^T / sqrt(d_h) + M) V_j,
where M is the causal mask, M_ij = 0 for j <= i and -inf otherwise. Consider two
implementations A and B of the same function that differ only in the order of
the accumulation inside the block-scaled dot product:
  s_ij^A = sum_{k=1..d_h} q_ik k_jk          (ascending k, fp32)
  s_ij^B = sum_{t=0..T-1} sum_{k in tile_t} q_ik k_jk  (tile-parallel, fp32)
Both round each product to fp32 and add in finite precision, so the only
difference is association. For fp32 with d_h = 128, the worst-case summation
error bound is
  |E| <= gamma_{d_h} * sum_k |q_ik k_jk|,   gamma_n = n u / (1 - n u),
with u = 2^-24 the unit roundoff. Then a perturbation e_ij of the score changes
the row softmax by at most
  ||p' - p||_1 <= 2 * sum_j p_j |e_ij|  (to first order, by Pinsker-style bounding of the log-sum-exp),
so a per-score error of 2e-3 costs at most ~4e-3 of total variation on any row
whose mass concentrates where |e| is small. The KL view: KL(p||p') ≈ 1/2 sum_j p_j"""))

P.append(("python-daemon", """Continue the Python exactly.

import asyncio, json, logging, signal
from dataclasses import dataclass, field

log = logging.getLogger("router")

@dataclass
class Backend:
    url: str
    weight: int = 1
    inflight: int = 0
    broken_until: float = 0.0

    def healthy(self, now: float) -> bool:
        return self.broken_until <= now

class Pool:
    def __init__(self, backends):
        self.backends = backends
        self._cond = asyncio.Condition()

    async def acquire(self, now: float):
        async with self._cond:
            while True:
                ok = [b for b in self.backends if b.healthy(now)]
                if ok:
                    best = min(ok, key=lambda b: (b.inflight / max(b.weight, 1)))
                    best.inflight += 1
                    return best
                wake = min(b.broken_until for b in self.backends)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=max(0.05, wake - now))
                except asyncio.TimeoutError:
                    pass
                now = asyncio.get_running_loop().time()

    async def release(self, b: Backend, ok: bool, now: float):
        async with self._cond:
            b.inflight -= 1
            if not ok:
                b.broken_until = now + 30.0
                log.warning("backoff %s for 30s", b.url)
            self._cond.notify_all"""))

P.append(("json-config", """Complete the JSON object exactly, keeping the schema and closing braces.

{
  "deployment": "lab-v4",
  "engine": {
    "binary": "/opt/radiance/bin/radiance",
    "model": "/models/moe-flash-fp8-iq4r.rad",
    "tp": 2,
    "tp_wire": "wht6",
    "kv_cache_dtype": "fp8",
    "max_model_len": 200000,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 2048,
    "placement": {
      "mode": "expert_tiered",
      "host_pool_mib": 8192,
      "ngram": "disk",
      "gpu_headroom_mib": 96
    },
    "prefix_cache_mib": 4096,
    "checkpoint_slots": 32,
    "speculative": { "tokens": 3, "kind": "ngram" }
  },
  "gateway": {
    "base_url": "http://10.0.5.20:9002/v1",
    "max_retries": 2,
    "timeout_s": 600,
    "drop_params": true
  },
  "guards": [
    { "id": "concurrency", "rule": "sessions_130k <= 5", "why": "no graceful preemption; wedge risk" },
    { "id": "wire-floor", "rule": "single-seq decode exact, prefill lossy at wht6", "why" :"""))

P.append(("csv-table", """Summarize what the last row says, then continue the table with two more plausible rows.

model,size_b,card,quant,ctx,decode_tps,prefix_cache_hit_pct,notes
Qwen3.8-27B,27,R9700,Q4_K_XL,256000,26.4,71,single card fits all
Qwen3.8-27B,27,R9700x2,fp8,200000,86.1,88,tp2 dflash2 depth7
MiniCPM5-2B,2,7900XTX,bf16,16384,11.1,0,GEMV decode row kernel
MiniCPM5-2B,2,7900XTX,bf16,16384,0.35,0,M-parameter arena bug pre-fix
Qwen3.6-35B-A3B,35,R9700x2,Q4_K,131072,41.2,64,abandoned coexist with 27B
gemma-3-4b,4,7900XTX,bf16,8192,29.8,"""))

P.append(("mixed-long", """You are a careful technical editor. Rewrite and continue, keeping every number exact.

The incident report reads as follows. At 07:37 the host began swapping at 90% of RAM while
serving a 114 GB container on 31 GiB of memory; the placement plan had pinned an 8 GiB host
pool for experts, 4 GiB of prefix cache, and 3.6 GiB across 32 checkpoint snapshot slots of
112.73 MiB each, and measured RSS reached 26.5 GB before the first request arrived. The engine
reports zero preemptions throughout, which is not a virtue: the admission path never reclaims
from already-admitted sessions, so the 32-slot max_num_seqs limit admits more than the 585,488
token pool can hold at 130k contexts, and the failure mode observed on the previous day was a
420-second stall followed by a GPU reset with an unrecoverable queue state in dmesg. The safe
operating envelope, measured rather than inferred, is four to five concurrent 130k sessions.
Three things followed from this. First, the host pool is elastic: it lends VRAM to the KV cache
and takes it back, so shrinking the pool by 7 GiB costs prefix-cache depth, not correctness.
Second, the disk tier exists and is honest about its price -- a routed expert read a layer at a
time means a step paced by the drive, roughly 2.2 GB/s sustained, which for a 65 GiB resident
set is not a latency you can hide behind a scheduler. Third, the wire: at wht6 the all-reduce
above 128 KiB is quantized, so the exact function is"""))

P.append(("short-probe", "Write a haiku about a cat that sleeps on a warm graphics card.\n\nHaiku:"))

# deep-prefill composites: KL from accumulation drift grows with context depth,
# so these are the sharpest cases for the #113 failure class.
P.append(("code-long", P[0][1] + "\n\n// and the second file:\n" + P[1][1] + "\n\n# and the service around it:\n" + P[4][1]))
P.append(("prose-long", P[2][1] + "\n\n" + P[7][1]))

out = open(out_path, "w")
for pid, text in [(p[0], p[1]) for p in P]:
    out.write(json.dumps({"id": pid, "text": text}) + "\n")
out.close()
print("wrote", len(P), "prompts")
for p in P:
    print(" ", p[0], len(p[1]), "chars")
