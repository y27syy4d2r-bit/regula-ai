import json
import re
import sqlite3
from collections import defaultdict
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import DEFAULT, MODEL_NAME, QUERY_PREFIX, RERANK_MODEL, data_paths

STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
        "be", "must", "what", "which", "who", "how", "do", "does", "can", "should",
        "shall", "that", "this", "with", "by", "as", "at", "it", "under", "dora"}
ART_REF = re.compile(r"\bart(?:icle|\.)?\s*(\d{1,2})\b(?:\s*\(\s*(\d+)\s*\))?", re.I)
# DEF_Q = re.compile(
#     r"^\s*(?:what\s+(?:is|are)|what\s+does|define|definition\s+of|meaning\s+of)\s+"
#     r"(?:an?\s+|the\s+)?(.+?)(?:\s+mean)?\s*\??\s*$", re.I)

DEF_Q = re.compile(
    r"^\s*(?:what\s+does\s+(?:dora\s+)?(?:mean\s+by|define\s+as)"
    r"|how\s+does\s+(?:dora\s+)?define"
    r"|what\s+(?:is|are)|what\s+does|define|definition\s+of|meaning\s+of)\s+"
    r"(?:an?\s+|the\s+)?(.+?)(?:\s+mean)?\s*\??\s*$", re.I)
DEF_HEAD = re.compile(r"^\(\d+\)\s*[‘“'\"]\s*([^’”\"']+?)\s*[’”\"']")


# ---------- cached resources ----------
@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=None)
def _resources(tag):
    db, emb, ids = data_paths(tag)
    con = sqlite3.connect(db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con, np.load(emb), json.loads(ids.read_text())


# ---------- the individual signals ----------
def semantic(query, cfg, n=20):
    _, emb, ids = _resources(cfg.tag)
    q = _model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    sims = emb @ q
    best = {}                                   # chunk_id -> best window score
    for i in np.argsort(-sims)[:80]:
        cid = ids[i]
        if cid not in best:
            best[cid] = float(sims[i])
        if len(best) == n:
            break
    return best


def keyword(query, cfg, n=20):
    con = _resources(cfg.tag)[0]
    tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in STOP and len(t) > 1]
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    rows = con.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY bm25(chunks_fts, 0.0, 2.0, 1.0) LIMIT ?", (match, n)).fetchall()
    return [r[0] for r in rows]


def direct(query, cfg, cap=6):
    con = _resources(cfg.tag)[0]
    ids = []
    for m in ART_REF.finditer(query):
        art, para = int(m.group(1)), m.group(2)
        if para:
            rows = con.execute("SELECT chunk_id FROM chunks WHERE article=? AND paragraph=?",
                               (art, para)).fetchall()
        else:
            rows = con.execute("SELECT chunk_id FROM chunks WHERE article=? ORDER BY rowid",
                               (art,)).fetchall()
        ids += [r[0] for r in rows]
    return ids[:cap]


def _norm(s):
    return re.sub(r"[^a-z0-9 -]", "", s.lower().replace("’", "'")).strip()


def _stem(t):
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


@lru_cache(maxsize=None)
def _definitions(tag):
    con = _resources(tag)[0]
    out = []
    for cid, text in con.execute(
            "SELECT chunk_id, text FROM chunks WHERE article=3 AND paragraph IS NOT NULL"):
        m = DEF_HEAD.match(text)
        if m:
            out.append((cid, _norm(m.group(1))))
    return out


def definitions(query, cfg, cap=3):
    m = DEF_Q.match(query)
    if not m:
        return []
    term = _norm(m.group(1))
    defs = _definitions(cfg.tag)
    exact = [c for c, d in defs if d == term]
    if exact:
        return exact
    return [c for c, d in defs if len(term) >= 4 and term in d][:cap]


@lru_cache(maxsize=None)
def _titles(tag):
    con = _resources(tag)[0]
    first = {}
    for art, title, cid in con.execute(
            "SELECT article, article_title, chunk_id FROM chunks ORDER BY rowid"):
        if art != 3 and art not in first:
            toks = {_stem(t) for t in re.findall(r"\w+", title.lower()) if t not in STOP}
            first[art] = (cid, toks)
    return list(first.values())


def title_matches(query, cfg):
    q = {_stem(t) for t in re.findall(r"\w+", query.lower()) if t not in STOP}
    return [cid for cid, toks in _titles(cfg.tag) if len(toks) >= 2 and toks <= q]


# ---------- optional reranker ----------
def _rerank(query, cids, con):
    rows = [con.execute("SELECT citation, article_title, text FROM chunks WHERE chunk_id=?",
                        (c,)).fetchone() for c in cids]
    pairs = [(query, f"{r['citation']} - {r['article_title']}\n{r['text'][:2000]}") for r in rows]
    scores = _reranker().predict(pairs)
    return [cids[i] for i in np.argsort(-np.asarray(scores))]


# ---------- the fusion ----------
def search(query, k=6, cfg=DEFAULT):
    con = _resources(cfg.tag)[0]
    sem, kw = semantic(query, cfg), keyword(query, cfg)
    dr = direct(query, cfg) if cfg.use_direct else []
    dfn = definitions(query, cfg) if cfg.use_definition else []
    ttl = title_matches(query, cfg) if cfg.use_title else []
    explicit = set(dr + dfn)

    score = defaultdict(float)
    for cid in explicit:
        score[cid] += cfg.explicit_boost
    for cid in ttl:
        score[cid] += cfg.title_bonus
    for rank, cid in enumerate(sem):
        score[cid] += cfg.w_sem / (cfg.rrf_k + rank + 1)
    for rank, cid in enumerate(kw):
        score[cid] += cfg.w_kw / (cfg.rrf_k + rank + 1)

    ranked = sorted(score, key=score.get, reverse=True)
    pinned = [c for c in ranked if c in explicit]       # explicit hits stay on top
    rest = [c for c in ranked if c not in explicit]
    if cfg.use_reranker:
        rest = _rerank(query, rest[:cfg.rerank_top_n], con) + rest[cfg.rerank_top_n:]
    top = (pinned + rest)[:k]

    results = []
    for cid in top:
        r = dict(con.execute("SELECT * FROM chunks WHERE chunk_id=?", (cid,)).fetchone())
        flags = (("direct", cid in dr), ("definition", cid in dfn), ("title", cid in ttl),
                 ("semantic", cid in sem), ("keyword", cid in kw))
        r.update(score=round(score[cid], 4), cosine=sem.get(cid),
                 via=[n for n, hit in flags if hit])
        results.append(r)
    top_cosine = max(sem.values()) if sem else 0.0
    return results, top_cosine