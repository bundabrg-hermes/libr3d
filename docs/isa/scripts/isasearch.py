"""Search library over the RDNA3 ISA corpus (isa.db: FTS5 + sqlite-vec)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import sqlite_vec

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "isa.db"
MODEL = "BAAI/bge-small-en-v1.5"
QINSTRUCT = (
    "Instruct: Given a question about the AMD RDNA3 shader instruction set "
    "(gfx11), retrieve passages that answer it.\nQuery: "
)
IDENT_RE = re.compile(r"\b(?:[SVD]|T|D)?[A-Z][A-Z0-9]*_[A-Z0-9_]+\b|\bEXP\b", re.I)
LLVM_SUFFIXES = ("_E32", "_E64", "_SDWA", "_SDPM")


def connect(db: Path = DB) -> sqlite3.Connection:
    # MCP servers dispatch tools on worker threads; sqlite is compiled serialized.
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class ISA:
    def __init__(self, db: Path = DB):
        self.conn = connect(db)
        self._emb = None

    # ---------- embedding ----------
    @property
    def embedder(self):
        if self._emb is None:
            from fastembed import TextEmbedding

            self._emb = TextEmbedding(MODEL)
        return self._emb

    def _embed_query(self, text: str):
        return list(self.embedder.embed([QINSTRUCT + text]))[0]

    # ---------- lexical ----------
    def lexical(self, q: str, k: int = 10):
        toks = re.findall(r"[A-Za-z0-9_]{2,}", q)
        if not toks:
            return []
        qstr = " OR ".join('"%s"' % t for t in toks)
        sql = (
            "SELECT rowid AS id, bm25(fts, 5.0, 20.0, 1.0) AS score, "
            "snippet(fts, 2, '[', ']', ' … ', 18) AS snip FROM fts "
            "WHERE fts MATCH ? ORDER BY score LIMIT ?"
        )
        out = []
        try:
            for r in self.conn.execute(sql, (qstr, k * 2)):
                out.append(dict(doc_id=r["id"], score=r["score"], snippet=r["snip"], src="lex"))
            for t in [x for x in toks if "_" in x]:
                phrase = t.lower().replace("_", " ")
                try:
                    rows = self.conn.execute(
                        "SELECT rowid AS id, bm25(fts, 5.0, 20.0, 1.0) AS score, "
                        "snippet(fts, 2, '[', ']', ' … ', 18) AS snip FROM fts "
                        "WHERE fts.mnemonic_sp MATCH ? ORDER BY score LIMIT 5",
                        (f'"{phrase}"',),
                    )
                    for r in rows:
                        out.insert(0, dict(doc_id=r["id"], score=r["score"] - 100.0,
                                           snippet=r["snip"], src="mnemonic"))
                except sqlite3.OperationalError:
                    pass
        except sqlite3.OperationalError:
            return []
        best: dict[int, dict] = {}
        for r in out:
            best.setdefault(r["doc_id"], r)
        return sorted(best.values(), key=lambda r: r["score"])[:k]

    # ---------- vector ----------
    def vector(self, q: str, k: int = 10, kind: str | None = None):
        from sqlite_vec import serialize_float32

        vec = self._embed_query(q)
        sql = (
            "SELECT c.doc_id AS id, v.distance AS dist, substr(c.text, 1, 400) AS snip "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance"
        )
        params: list = [serialize_float32(list(map(float, vec))), k * (4 if kind else 1)]
        best: dict[int, dict] = {}
        for r in self.conn.execute(sql, params):
            if kind:
                row = self.conn.execute("SELECT kind FROM docs WHERE id=?", (r["id"],)).fetchone()
                if not row or (kind == "narrative" and row["kind"] == "inst"):
                    continue
                if kind == "inst" and row["kind"] != "inst":
                    continue
            if r["id"] not in best:
                best[r["id"]] = dict(doc_id=r["id"], score=r["dist"], snippet=r["snip"], src="vec")
            if len(best) >= k:
                break
        return list(best.values())

    # ---------- exact ----------
    def exact(self, name: str):
        n = name.upper()
        cands = [n] + [n[: -len(suf)] for suf in LLVM_SUFFIXES if n.endswith(suf)]
        rows = []
        for c in cands:
            rows = self.conn.execute(
                "SELECT * FROM docs WHERE upper(mnemonic) = ? ORDER BY kind", (c,)
            ).fetchall()
            if rows:
                return rows
        return self.conn.execute(
            "SELECT * FROM docs WHERE upper(mnemonic) LIKE ? ORDER BY kind LIMIT 12",
            (cands[0] + "%",),
        ).fetchall()

    # ---------- hybrid ----------
    def search(self, q: str, k: int = 5, mode: str = "auto", kind: str | None = None):
        idents = IDENT_RE.findall(q)
        prose_words = len([w for w in re.split(r"\W+", q) if w])
        results: list[dict] = []
        seen: set[int] = set()

        for t in idents[:2]:
            for r in self.exact(t):
                d = self._doc_result(r, score=0.0, src="exact", snippet=None)
                if d["doc_id"] not in seen:
                    seen.add(d["doc_id"])
                    results.append(d)

        lex = self.lexical(q, k * 2) if mode in ("auto", "lex") else []
        vec = []
        want_vec = mode in ("auto", "vec") and not (
            mode == "auto" and idents and prose_words < 5
        )
        if want_vec:
            try:
                vec = self.vector(q, k * 2, kind=kind)
            except Exception:
                vec = []

        # RRF fuse
        fused: dict[int, dict] = {}
        for ranking, name in ((lex, "lex"), (vec, "vec")):
            for rank, r in enumerate(ranking):
                e = fused.setdefault(
                    r["doc_id"], dict(doc_id=r["doc_id"], rrf=0.0, src=set(), snippet=r["snippet"])
                )
                e["rrf"] += 1.0 / (60 + rank)
                e["src"].add(name)
        for doc_id, e in sorted(fused.items(), key=lambda kv: -kv[1]["rrf"]):
            if doc_id in seen or len(results) >= k:
                continue
            row = self.conn.execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()
            if kind == "narrative" and row["kind"] == "inst":
                continue
            if kind == "inst" and row["kind"] != "inst":
                continue
            d = self._doc_result(row, score=e["rrf"], src="+".join(sorted(e["src"])),
                                 snippet=e["snippet"])
            d["lex"] = "lex" in e["src"]
            d["vec"] = "vec" in e["src"]
            results.append(d)
        return results[:k]

    def _doc_result(self, row, score, src, snippet):
        text = None
        if src == "exact" or snippet is None:
            p = ROOT / row["path"]
            text = p.read_text()
        return dict(
            doc_id=row["id"], kind=row["kind"], mnemonic=row["mnemonic"],
            format=row["format"], section=row["section"], path=row["path"],
            score=score, src=src, snippet=snippet, text=text,
        )

    def full_text(self, name: str):
        rows = self.exact(name)
        out = []
        for row in rows:
            out.append(dict(path=row["path"], text=(ROOT / row["path"]).read_text()))
        return out
