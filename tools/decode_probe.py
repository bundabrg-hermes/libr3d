#!/usr/bin/env python3
"""Decode-rate probe for a radiance serve (split-KV A/B).

Pair-differencing: run the same calibrated prompt at n_predict 8 and 108; the
decode rate is 100 / (t108 - t8). The subtraction removes prefill, queueing and
response overheads without trusting any engine-internal counter, and the fresh
uuid prefix keeps the prefix cache from manufacturing a baseline (same lesson
ttft_probe.py carries). Token counts calibrate against the engine's /tokenize.

Usage:  decode_probe.py [--url URL] --ctx N [--reps 3] [--tag LABEL]
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid

FILLER = ("The lighthouse keeper logged the weather every evening, and the "
          "harbor master read the log before opening the gates. ")


def http_json(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def calibrate(url, target):
    units = max(1, target // 22)
    text = FILLER * units
    count = -1
    for _ in range(14):
        count = http_json(url + "/tokenize", {"content": text, "add_special": False})["count"]
        if count == target:
            return text
        units = max(1, units + (target - count) // 22)
        text = FILLER * units
    print("calibrate: landed on %d, not %d" % (count, target))
    return text


def run(url, prompt, n):
    t0 = time.perf_counter()
    http_json(url + "/completion", {
        "prompt": uuid.uuid4().hex[:12] + "\n" + prompt,
        "n_predict": n, "temperature": 0, "seed": 7, "stream": False,
    })
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9005")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prompt = calibrate(args.url, args.ctx)
    rates = []
    for _ in range(args.reps):
        t_short = run(args.url, prompt, 8)
        t_long = run(args.url, prompt, 108)
        rate = 100.0 / (t_long - t_short)
        rates.append(rate)
        print("  pair: t8=%.3f t108=%.3f -> %.2f tok/s" % (t_short, t_long, rate))
    print("decode_probe ctx=%d %s: median %.2f tok/s  (all %s)"
          % (args.ctx, args.tag, statistics.median(rates),
             ["%.1f" % r for r in rates]))


if __name__ == "__main__":
    main()
