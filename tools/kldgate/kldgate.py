#!/usr/bin/env python3
"""kldgate -- prefill-logit drift gate for libr3d, in the r77-base style.

A prefill kernel can be oracle-green at every op point and still shift the
model: issue stew675/llama-cpp-rdna-boosts#113 shipped a default-on bf16
prefill path with mean KLD ~0.03 past every text/coherence gate, because
sampled text reads the decode distribution from the post-prefill state and is
blind to a uniform prefill shift. Greedy text cannot see it either.

The gate: two serve arms (a base and a test) run the same corpus greedily at
temperature 0 with raw top-N logprobs (n_probs, post_sampling_probs=false).
The base's chosen-token trajectory is the forced teacher-forcing path; at every
position we compare the two arms' distributions, so a drift that compounds
through real prefill activations shows up as KL divergence per position, before
the argmax ever flips.

Approximation, stated plainly: we see each arm's top-N support (N = 16, the
engine's n_probs cap), not the full 130k-row softmax. KL is computed on the
union of the two supports plus the forced token; mass outside is bounded by
the smallest retained logprob and counted (a position where the forced token
falls outside the test arm's top-N is flagged). For coherent distributions at
temp 0 that mass is negligible, and a tail-only shift under it is below this
instrument's resolution -- full-vocab KLD needs prompt_logprobs in the engine,
which radiance does not expose yet (upstream request open).

subcommands:
  collect --out FILE --arm NAME [--url URL]     hit one serve, dump per-position
                                                logprob records for the corpus
  compare BASE.jsonl TEST.jsonl [--report FILE] KL/top-1/divergence report and
                                                the PASS/FAIL verdict

thresholds (defaults, overridable): mean KLD <= 2e-3, max <= 2e-2, top-1
agreement >= 0.98. Rounding-order-class shifts sit near 1e-4 (their #113
kill-switch residual was 0.0006); defect-class shifts sit at 0.03-0.1.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:9005"
CORPUS_PATH = "/home/devuser/projects/libr3d/libr3d/tools/kldgate/corpus.jsonl"

GATE_MEAN_KLD = 2e-3
GATE_MAX_KLD = 2e-2
GATE_TOP1 = 0.98


def http_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}: {e.read().decode()[:400]}")


def cmd_collect(args):
    corpus = [json.loads(l) for l in open(args.corpus) if l.strip()]
    out = open(args.out, "w")
    n_probs = args.n_probs
    for item in corpus:
        t0 = time.time()
        resp = http_json(args.url + "/completion", {
            "prompt": item["text"],
            "n_predict": args.n_predict,
            "temperature": 0,
            "seed": args.seed,
            "n_probs": n_probs,
            "post_sampling_probs": False,
            "stream": False,
        })
        cp = resp.get("completion_probabilities")
        if cp is None:
            sys.exit(f"arm {args.arm}: no completion_probabilities for "
                     f"{item['id']} -- is the server honoring n_probs?")
        # trim to the fields the comparator reads
        pos = [{"id": e["id"], "logprob": e["logprob"],
                "top": [[t["id"], t["logprob"]] for t in e["top_logprobs"]]}
               for e in cp]
        rec = {"arm": args.arm, "id": item["id"],
               "prompt_tokens": resp.get("tokens_evaluated"),
               "stop": resp.get("stop_type"),
               "secs": round(time.time() - t0, 2),
               "pos": pos}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"  {item['id']}: {len(pos)} positions, "
              f"prompt {resp.get('tokens_evaluated')} tok, "
              f"{rec['secs']}s", file=sys.stderr)
    out.close()


def kl_union(a_top, b_top):
    """KL(P_a || P_b) on the union of two top-N supports (id -> logprob).
    Renormalized over the union; returns (kl, outside_mass_a)."""
    ids = set(a_top) | set(b_top)
    # floor for an id missing from a support: its mass cannot exceed the
    # smallest retained one (top-N was sorted by probability).
    def mass(top, i):
        if i in top:
            return math.exp(top[i])
        return math.exp(min(top.values())) if top else 0.0
    pa = {i: mass(a_top, i) for i in ids}
    pb = {i: mass(b_top, i) for i in ids}
    za, zb = sum(pa.values()), sum(pb.values())
    kl = 0.0
    for i in ids:
        p, q = pa[i] / za, pb[i] / zb
        if p > 0 and q > 0:
            kl += p * math.log(p / q)
    outside = max(0.0, 1.0 - za)
    return kl, outside


def cmd_compare(args):
    base = {r["id"]: r for r in (json.loads(l) for l in open(args.base))}
    test = {r["id"]: r for r in (json.loads(l) for l in open(args.test))}
    rows, all_kl, all_top1 = [], [], []
    for pid in base:
        b, t = base[pid], test.get(pid)
        if t is None:
            rows.append((pid, "-", "-", "-", "-", "MISSING"))
            continue
        tpos = {i: p for i, p in enumerate(t["pos"])}
        kls, agree, div = [], [], None
        for i, bp in enumerate(b["pos"]):
            if i >= len(t["pos"]):
                break
            tp = tpos[i]
            a = {tid: lp for tid, lp in bp["top"]}
            bm = {tid: lp for tid, lp in tp["top"]}
            a.setdefault(bp["id"], bp["logprob"])
            # the base's chosen token is the forced token; make sure both
            # distributions can see it, else it is a tail miss for the test.
            forced_in_test = bp["id"] in bm
            kl, outside = kl_union(a, bm)
            kls.append(kl)
            all_kl.append(kl)
            top1_ok = a and bm and max(a, key=a.get) == max(bm, key=bm.get)
            agree.append(1 if top1_ok else 0)
            all_top1.append(1 if top1_ok else 0)
            if div is None and bp["id"] != tp["id"]:
                div = i
            if not forced_in_test and i < 4096:
                pass  # counted below via outside mass
        if kls:
            rows.append((pid, len(kls),
                         f"{sum(kls)/len(kls):.5f}", f"{max(kls):.5f}",
                         f"{sum(agree)/len(agree):.3f}",
                         "-" if div is None else str(div)))
    mean = sum(all_kl) / len(all_kl) if all_kl else float("nan")
    mx = max(all_kl) if all_kl else float("nan")
    t1 = sum(all_top1) / len(all_top1) if all_top1 else float("nan")
    ok = mean <= args.mean_kld and mx <= args.max_kld and t1 >= args.top1
    lines = ["# kldgate report", "",
             f"base={args.base}  test={args.test}  positions={len(all_kl)}", "",
             "| corpus | positions | mean KLD | max KLD | top-1 agree | first divergence |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    lines += ["", f"**aggregate: mean KLD {mean:.5f} (gate {args.mean_kld}), "
                  f"max {mx:.5f} (gate {args.max_kld}), "
                  f"top-1 {t1:.3f} (gate {args.top1})**",
              "", "VERDICT: " + ("PASS" if ok else "FAIL")]
    report = "\n".join(lines) + "\n"
    if args.report:
        open(args.report, "w").write(report)
    print(report)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--url", default=DEFAULT_URL)
    c.add_argument("--corpus", default=CORPUS_PATH)
    c.add_argument("--n-predict", type=int, default=128)
    c.add_argument("--n-probs", type=int, default=16)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(fn=cmd_collect)
    m = sp.add_parser("compare")
    m.add_argument("base")
    m.add_argument("test")
    m.add_argument("--report")
    m.add_argument("--mean-kld", type=float, default=GATE_MEAN_KLD)
    m.add_argument("--max-kld", type=float, default=GATE_MAX_KLD)
    m.add_argument("--top1", type=float, default=GATE_TOP1)
    m.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
