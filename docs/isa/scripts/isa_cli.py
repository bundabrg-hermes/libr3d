#!/usr/bin/env python3
"""isa — CLI for the RDNA3 ISA corpus.

  isa get V_PERMLANE16_B32      full instruction entry (all encodings)
  isa get v_add_f32_e32         LLVM names work (_e32/_e64 stripped)
  isa search "broadcast one lane to the wave"
  isa find PERMLANE             mnemonic substring
  isa list --format VOP3P       mnemonics by encoding format
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isasearch import ISA, ROOT  # noqa: E402


def fmt_doc(d, show_snip=True):
    head = f"[{d['kind']}] {d['path']}"
    if d["mnemonic"]:
        head += f"  {d['mnemonic']} ({d['format']}, §{d['section']})"
    line = head + f"  <{d['src']} {d['score']:.3f}>"
    out = [line]
    if show_snip and d.get("snippet"):
        out.append("    " + d["snippet"].replace("\n", " ")[:300])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(prog="isa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("get"); p.add_argument("name")
    p = sub.add_parser("search"); p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=5); p.add_argument("--mode", default="auto",
                   choices=["auto", "lex", "vec"]); p.add_argument("--sections", action="store_true")
    p = sub.add_parser("find"); p.add_argument("substr")
    p = sub.add_parser("list"); p.add_argument("--format", dest="fmt")
    a = ap.parse_args()
    isa = ISA()

    if a.cmd == "get":
        docs = isa.full_text(a.name)
        if not docs:
            print(f"no entry for {a.name!r}")
            return 1
        for d in docs:
            print(f"### {d['path']}\n")
            print(d["text"])
        return 0

    if a.cmd == "search":
        q = " ".join(a.query)
        res = isa.search(q, k=a.k, mode=a.mode, kind="narrative" if a.sections else None)
        if not res:
            print("no results")
            return 1
        for d in res:
            print(fmt_doc(d))
        return 0

    if a.cmd == "find":
        rows = isa.conn.execute(
            "SELECT mnemonic, format, section, path FROM docs "
            "WHERE upper(mnemonic) LIKE ? ORDER BY mnemonic LIMIT 100",
            ("%" + a.substr.upper() + "%",),
        ).fetchall()
        for r in rows:
            print(f"{r['mnemonic']:<40} {r['format']:<8} {r['path']}")
        print(f"({len(rows)} matches)")
        return 0

    if a.cmd == "list":
        sql = "SELECT mnemonic, format FROM docs WHERE kind='inst'"
        args: tuple = ()
        if a.fmt:
            sql += " AND format = ?"
            args = (a.fmt.upper(),)
        rows = isa.conn.execute(sql + " ORDER BY mnemonic", args).fetchall()
        for r in rows:
            print(r["mnemonic"])
        print(f"({len(rows)} instructions)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
