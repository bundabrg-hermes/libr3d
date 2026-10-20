#!/usr/bin/env python3
"""Build isa.db: docs + FTS5 + chunk embeddings (sqlite-vec) from the md corpus."""
import json
import re
import sqlite3
import sys
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "isa.db"
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TARGET = 1400


def parse_fm(text):
    fm = {}
    body = text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"')
        body = text[m.end():]
    return fm, body


def chunk_body(body, overlap_words=40):
    paras = [p for p in body.split("\n\n") if p.strip()]
    chunks, cur = [], []
    n = 0
    for p in paras:
        cur.append(p)
        n += len(p)
        if n >= CHUNK_TARGET:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [body]


def main():
    from fastembed import TextEmbedding

    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(str(DB))
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE docs(
          id INTEGER PRIMARY KEY, kind TEXT, title TEXT, mnemonic TEXT,
          mnemonic_sp TEXT, aliases TEXT, format TEXT, section TEXT,
          opcode TEXT, pages TEXT, path TEXT, body TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(title, mnemonic_sp, body);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, seq INTEGER, text TEXT);
        """
    )
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[384])")
    conn.execute("INSERT INTO meta VALUES('model', ?)", (MODEL,))
    conn.execute(
        "INSERT INTO meta VALUES('source_sha', ?)",
        (__import__("hashlib").sha256((ROOT / "source" / "rdna3-isa.pdf").read_bytes()).hexdigest(),),
    )

    files = sorted((ROOT / "inst").glob("*.md")) + sorted((ROOT / "sections").glob("*.md"))
    docs, chunks = [], []
    for path in files:
        rel = str(path.relative_to(ROOT))
        text = path.read_text()
        fm, body = parse_fm(text)
        if path.parent.name == "inst":
            kind, title = "inst", fm.get("mnemonic", path.stem)
            mnemonic = fm.get("mnemonic", "")
            aliases = fm.get("aliases", "")
            fmt = fm.get("format", "")
            opcode = fm.get("opcode", "")
            pages = fm.get("pages", "")
        else:
            kind, title, mnemonic, aliases, fmt, opcode, pages = "section", "", "", "", "", "", ""
            m = re.search(r"^# (.+)$", body, re.M)
            title = m.group(1) if m else path.stem
        msp = (mnemonic or title).replace("_", " ").lower()
        if aliases:
            for a in re.findall(r"[A-Z0-9_]+", aliases):
                msp += "\n" + a.replace("_", " ").lower()
        docs.append(dict(kind=kind, title=title, mnemonic=mnemonic, mnemonic_sp=msp,
                         aliases=aliases, format=fmt, section=fm.get("section", ""),
                         opcode=opcode, pages=pages, path=rel, body=body))
        header = (f"RDNA3 ISA {fmt} instruction {mnemonic}." if kind == "inst"
                  else f"RDNA3 ISA reference, section: {title}.")
        for seq, ch in enumerate(chunk_body(body)):
            chunks.append((len(chunks), len(docs), seq, header + "\n" + ch))

    for d in docs:
        cur = conn.execute(
            "INSERT INTO docs(kind,title,mnemonic,mnemonic_sp,aliases,format,section,opcode,pages,path,body)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (d["kind"], d["title"], d["mnemonic"], d["mnemonic_sp"], d["aliases"],
             d["format"], d["section"], d["opcode"], d["pages"], d["path"], d["body"]))
        conn.execute("INSERT INTO fts(rowid,title,mnemonic_sp,body) VALUES(?,?,?,?)",
                     (cur.lastrowid, d["title"], d["mnemonic_sp"], d["body"]))
    conn.commit()

    emb = TextEmbedding(MODEL)
    texts = [c[3] for c in chunks]
    batch = 96
    for start in range(0, len(texts), batch):
        vecs = list(emb.embed(texts[start:start + batch]))
        with conn:
            for (cid, doc_id, seq, _), v in zip(chunks[start:start + batch], vecs):
                conn.execute("INSERT INTO chunks VALUES(?,?,?,?)", (cid, doc_id, seq, texts[cid]))
                conn.execute("INSERT INTO vec_chunks VALUES(?,?)", (cid, serialize_float32(list(map(float, v)))))
        print(f"embedded {min(start + batch, len(texts))}/{len(texts)}", flush=True)

    conn.execute("INSERT INTO meta VALUES('docs', ?)", (str(len(docs)),))
    conn.execute("INSERT INTO meta VALUES('chunks', ?)", (str(len(chunks)),))
    conn.commit()
    conn.execute("VACUUM")
    print(json.dumps({"docs": len(docs), "chunks": len(chunks), "db": str(DB)}))


if __name__ == "__main__":
    sys.exit(main())
