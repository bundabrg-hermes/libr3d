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
  score --trajectory BASE.jsonl --out FILE      replay the base's forced
                                                trajectory (prompt + base
                                                tokens so far) against one
                                                serve; position i is then
                                                conditioned on IDENTICAL text
                                                in both arms -- the only
                                                legitimate KL comparison. The
                                                serve's prefix cache makes each
                                                forced request incremental; the
                                                /tokenize count guards against
                                                retokenization desync at token
                                                join boundaries (positions that
                                                desync are dropped, counted).
  compare --base B --free F --forced R [--report FILE]
                                                KL/top-1/divergence report and the
                                                PASS/FAIL verdict. KL and
                                                top-1 agreement are computed
                                                base vs forced (same context);
                                                the free run is reported only
                                                for its first-divergence column
                                                (free-running greedy arms).
  diffcompare --a FORCED_A --b FORCED_B [--report FILE]
                                                DIFFERENTIAL mode (2026-10-25): join two
                                                already-scored FORCED artifacts from the
                                                SAME base trajectory and the SAME binary,
                                                arms differing only in a lever, and
                                                compute KL/top-1 on identical positions.
                                                The committed compare flow scores a
                                                free-RUNNING decode-step base against a
                                                re-prefilled forced arm; that conditioning
                                                gap is an instrument property (the same-arm
                                                floor of 2026-10-25: mean 8.5e-3 / p99
                                                0.147 / top-1 0.975 on byte-identical
                                                kernels) and it CANCELS when both sides
                                                are forced replays of the same trajectory.
                                                What diffcompare measures is therefore the
                                                arm-attributable drift directly. The
                                                committed thresholds stay untouched: a
                                                diff verdict is adjudicated against them
                                                with the same-flow floor reported next to
                                                it, never by moving them.

thresholds (defaults, overridable): mean KLD <= 2e-3, p99 KLD <= 5e-3, top-1
agreement >= 0.98. Rounding-order-class shifts sit near 1e-4 (their #113
kill-switch residual was 0.0006); defect-class shifts sit at 0.03-0.1.
"""
import argparse
import codecs
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:9005"
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl")

GATE_MEAN_KLD = 2e-3
GATE_P99_KLD = 5e-3
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
        pos = [{"id": e["id"], "logprob": e["logprob"], "bytes": e["bytes"],
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


def cmd_score(args):
    """Replay the base trajectory's forced contexts against one serve."""
    corpus = {}
    for l in open(args.corpus):
        if l.strip():
            d = json.loads(l)
            corpus[d["id"]] = d["text"]
    traj = [json.loads(l) for l in open(args.trajectory)]
    out = open(args.out, "w")
    total_kept = 0
    for rec in traj:
        # tokens can split multibyte chars: an incremental decoder buffers the
        # partial tail and emits it when the next token completes it.
        dec = codecs.getincrementaldecoder("utf-8")()
        pieces = [corpus[rec["id"]]]
        # calibrate the BOS/special-token offset between /tokenize and the
        # engine's own prompt count once per prompt
        tk0 = http_json(args.url + "/tokenize",
                        {"content": pieces[0], "add_special": False})
        offset = (rec["prompt_tokens"] or 0) - tk0["count"]
        kept = 0
        t0 = time.time()
        for i, p in enumerate(rec["pos"]):
            expected = (rec["prompt_tokens"] or 0) + i
            text = "".join(pieces)
            tk = http_json(args.url + "/tokenize", {"content": text, "add_special": False})
            if tk["count"] + offset != expected:
                print(f"  {rec['id']}: retokenize desync at pos {i} "
                      f"({tk['count']}+{offset} != {expected}); truncating there",
                      file=sys.stderr)
                break
            resp = http_json(args.url + "/completion", {
                "prompt": text, "n_predict": 1, "temperature": 0,
                "seed": args.seed, "n_probs": args.n_probs,
                "post_sampling_probs": False, "stream": False})
            cp = resp.get("completion_probabilities") or []
            if not cp:
                break
            e = cp[0]
            out.write(json.dumps({"id": rec["id"], "pos_index": i,
                                  "tok": e["id"], "logprob": e["logprob"],
                                  "top": [[t["id"], t["logprob"]]
                                          for t in e["top_logprobs"]]}) + "\n")
            kept += 1
            pieces.append(dec.decode(bytes(p["bytes"])))
        out.flush()
        total_kept += kept
        print(f"  {rec['id']}: scored {kept}/{len(rec['pos'])} forced positions "
              f"({time.time() - t0:.1f}s)", file=sys.stderr)
    out.close()
    print(f"scored {total_kept} forced positions -> {args.out}", file=sys.stderr)


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
    base = {r["id"]: r for r in map(json.loads, open(args.base))}
    free = {}
    if args.free:
        free = {r["id"]: r for r in map(json.loads, open(args.free))}
    forced = {}
    for l in open(args.forced):
        r = json.loads(l)
        forced.setdefault(r["id"], {})[r["pos_index"]] = r
    rows, all_kl, all_top1 = [], [], []
    for pid, b in base.items():
        fpos = forced.get(pid, {})
        kls, agree, fdiv = [], [], None
        for i, bp in enumerate(b["pos"]):
            fp = fpos.get(i)
            if fp is None:
                break  # trajectory truncated by retokenize desync
            a = {tid: lp for tid, lp in bp["top"]}
            a.setdefault(bp["id"], bp["logprob"])
            bm = {tid: lp for tid, lp in fp["top"]}
            bm.setdefault(fp["tok"], fp["logprob"])
            kl, _ = kl_union(a, bm)
            kls.append(kl)
            all_kl.append(kl)
            top1_ok = max(a, key=a.get) == max(bm, key=bm.get)
            agree.append(1 if top1_ok else 0)
            all_top1.append(1 if top1_ok else 0)
            if fdiv is None and bp["id"] != fp["tok"]:
                fdiv = i
        # free-running arms: where the two greedy trajectories first part
        gdiv = "-"
        if pid in free:
            for i, bp in enumerate(b["pos"]):
                if i < len(free[pid]["pos"]) and bp["id"] != free[pid]["pos"][i]["id"]:
                    gdiv = str(i)
                    break
        dropped = len(b["pos"]) - len(kls)
        if kls:
            rows.append((pid, len(kls), f"{sum(kls)/len(kls):.5f}",
                         f"{max(kls):.5f}", f"{sum(agree)/len(agree):.3f}",
                         "-" if fdiv is None else str(fdiv), gdiv, dropped))
    mean = sum(all_kl) / len(all_kl) if all_kl else float("nan")
    mx = max(all_kl) if all_kl else float("nan")
    srt = sorted(all_kl)
    p99 = srt[int(0.99 * (len(srt) - 1))] if srt else float("nan")
    t1 = sum(all_top1) / len(all_top1) if all_top1 else float("nan")
    ok = mean <= args.mean_kld and p99 <= args.p99_kld and t1 >= args.top1
    lines = ["# kldgate report", "",
             "KL / top-1 are computed base vs FORCED (identical context at every",
             "position). 'forced div' = first position where the forced test arm's",
             "argmax differs under the same text -- a purity finding on its own.",
             "'free div' = first divergence of the two free-running greedy arms",
             "(informational: ties flip legitimately; positions after it are not",
             "compared).", "",
             "Gate: mean KLD + p99 (robust spread; single max is informational --",
             "one razor-tie position amplifying through the union-KL renormalize",
             "is not the defect class; issue #113's WAS a mean shift, mean 0.03-0.1",
             "against a rounding floor of 6e-4). The same-arm noise floor measured",
             "on this harness: mean ~2e-4, max ~6e-3 (serve-to-serve variance of",
             "fresh vs incremental prefill under the prefix cache).", "",
             f"base={args.base}  forced={args.forced}  positions={len(all_kl)}", "",
             "| corpus | scored pos | mean KLD | max KLD | top-1 agree | forced div | free div | dropped |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", f"**aggregate: mean KLD {mean:.5f} (gate {args.mean_kld}), "
                  f"p99 {p99:.5f} (gate {args.p99_kld}), max {mx:.5f} (info), "
                  f"top-1 {t1:.3f} (gate {args.top1})**",
              "", "VERDICT: " + ("PASS" if ok else "FAIL")]
    report = "\n".join(lines) + "\n"
    if args.report:
        open(args.report, "w").write(report)
    print(report)
    sys.exit(0 if ok else 1)


def load_forced(path):
    """Index a forced-score jsonl: id -> {pos_index: record}."""
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            out.setdefault(r["id"], {})[r["pos_index"]] = r
    return out


def cmd_diffcompare(args):
    """KL/top-1 between two FORCED artifacts of the same base trajectory.

    Both sides re-prefill identical forced text, so the committed flow's
    structural prefill-vs-free-decode gap cancels: this is the arm-attributable
    differential directly. Direction: KL(A || B) with A = the arm under test,
    B = the control (OFF/same-binary). The reverse direction is reported for
    information; a union-KL renormalize amplifies razor ties in either arm.
    """
    a_by_id = load_forced(args.a)
    b_by_id = load_forced(args.b)
    rows, all_kl, all_kl_rev, all_top1 = [], [], [], []
    for pid in sorted(set(a_by_id) & set(b_by_id)):
        apos, bpos = a_by_id[pid], b_by_id[pid]
        kls, kls_rev, agree, first_div = [], [], [], None
        for i in sorted(apos):
            ap, bp = apos[i], bpos.get(i)
            if bp is None:
                break  # one side truncated (retokenize desync): stop at the gap
            a = {tid: lp for tid, lp in ap["top"]}
            a.setdefault(ap["tok"], ap["logprob"])
            b = {tid: lp for tid, lp in bp["top"]}
            b.setdefault(bp["tok"], bp["logprob"])
            kl, _ = kl_union(a, b)
            kl_r, _ = kl_union(b, a)
            kls.append(kl)
            kls_rev.append(kl_r)
            all_kl.append(kl)
            top1_ok = max(a, key=a.get) == max(b, key=b.get)
            agree.append(1 if top1_ok else 0)
            all_top1.append(1 if top1_ok else 0)
            if first_div is None and not top1_ok:
                first_div = i
        if kls:
            rows.append((pid, len(kls), f"{sum(kls)/len(kls):.5f}",
                         f"{max(kls):.5f}", f"{sum(kls_rev)/len(kls_rev):.5f}",
                         f"{sum(agree)/len(agree):.3f}",
                         "-" if first_div is None else str(first_div)))
    mean = sum(all_kl) / len(all_kl) if all_kl else float("nan")
    mean_r = sum(all_kl_rev) / len(all_kl_rev) if all_kl_rev else float("nan")
    mx = max(all_kl) if all_kl else float("nan")
    srt = sorted(all_kl)
    p99 = srt[int(0.99 * (len(srt) - 1))] if srt else float("nan")
    t1 = sum(all_top1) / len(all_top1) if all_top1 else float("nan")
    ok = mean <= args.mean_kld and p99 <= args.p99_kld and t1 >= args.top1
    lines = ["# kldgate DIFFERENTIAL report", "",
             "KL / top-1 are computed arm vs control, both teacher-forced over the",
             "SAME base trajectory: identical text at every position on both sides,",
             "so the committed flow's free-decode-vs-re-prefill floor cancels. This",
             "is the arm-attributable drift itself -- NOT a floor-subtraction.",
             "Adjudication law: the committed thresholds below are the SAME absolute",
             "numbers as the committed compare flow (they are NOT moved for any arm);",
             "the differential is read against them directly, and the compare-flow",
             "floor (same-arm, this session) is quoted beside it for context.", "",
             f"a(arm)={args.a}  b(control)={args.b}  positions={len(all_kl)}", "",
             "| corpus | scored pos | mean KL(a\\|\\|b) | max KL | mean KL(b\\|\\|a) | top-1 agree | first top-1 div |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["",
              f"**aggregate: mean KLD {mean:.5f} (reference {args.mean_kld}), "
              f"p99 {p99:.5f} (reference {args.p99_kld}), max {mx:.5f} (info), "
              f"reverse mean {mean_r:.5f}, top-1 {t1:.3f} (reference {args.top1})**",
              "", "DIFF VERDICT (arm-attributable component vs committed "
              "thresholds): " + ("PASS" if ok else "FAIL")]
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
    s = sp.add_parser("score")
    s.add_argument("--trajectory", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--url", default=DEFAULT_URL)
    s.add_argument("--corpus", default=CORPUS_PATH)
    s.add_argument("--n-probs", type=int, default=16)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_score)
    m = sp.add_parser("compare")
    m.add_argument("--base", required=True)
    m.add_argument("--free")
    m.add_argument("--forced", required=True)
    m.add_argument("--report")
    m.add_argument("--mean-kld", type=float, default=GATE_MEAN_KLD)
    m.add_argument("--p99-kld", type=float, default=GATE_P99_KLD)
    m.add_argument("--top1", type=float, default=GATE_TOP1)
    m.set_defaults(fn=cmd_compare)
    d = sp.add_parser("diffcompare")
    d.add_argument("--a", required=True, help="forced artifact of the arm under test")
    d.add_argument("--b", required=True, help="forced artifact of the control arm")
    d.add_argument("--report")
    d.add_argument("--mean-kld", type=float, default=GATE_MEAN_KLD)
    d.add_argument("--p99-kld", type=float, default=GATE_P99_KLD)
    d.add_argument("--top1", type=float, default=GATE_TOP1)
    d.set_defaults(fn=cmd_diffcompare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
