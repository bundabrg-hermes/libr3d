#!/usr/bin/env python3
"""Prefill TTFT probe for a radiance serve (libr3d kernel A/B).

Measures time-to-first-token by completing with n_predict=1 at temperature 0:
the request latency is then prefill plus one decode step. Every request carries
a fresh uuid prefix so the prefix cache can never manufacture a phantom
baseline (a historical lesson: cached-prefix A/B read a regression that wasn't
there). Token counts are calibrated against the engine's own /tokenize, not an
estimate.

Usage:  ttft_probe.py [--url URL] --tokens N [--reps 5] [--tag LABEL]
Reports every probe plus the median, so a warm-up outlier is visible, not hidden.
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid

FILLER = ("The lighthouse keeper logged the weather every evening, and the "
          "harbor master read the log before opening the gates. ")


def http_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def calibrate(url, target):
    """Filler text whose /tokenize count lands on target (BOS/special offset
    is constant per text shape and cancels out of the delta we care about)."""
    units = max(1, target // 22)
    text = FILLER * units
    count = -1
    for _ in range(12):
        count = http_json(url + "/tokenize", {"content": text, "add_special": False})["count"]
        if count == target:
            return text
        units = max(1, units + (target - count) // 22)
        text = FILLER * units
    print("calibrate: landed on %d, not %d" % (count, target))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9005")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    body = calibrate(args.url, args.tokens)
    ts, evals = [], []
    for _ in range(args.reps):
        prompt = "%s\n%s" % (uuid.uuid4(), body)
        t0 = time.perf_counter()
        resp = http_json(args.url + "/completion",
                         {"prompt": prompt, "n_predict": 1, "temperature": 0,
                          "stream": False})
        ts.append(time.perf_counter() - t0)
        evals.append(resp.get("tokens_evaluated"))
    print("%s tokens=%d evals=%s ttft=%s median=%.3fs min=%.3fs" % (
        args.tag or "probe", args.tokens, evals,
        ["%.3f" % t for t in ts], statistics.median(ts), min(ts)))


if __name__ == "__main__":
    main()
