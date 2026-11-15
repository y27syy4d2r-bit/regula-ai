import json
import sqlite3
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DEFAULT, MODEL_NAME, VERSIONS, data_paths


def windows(text, size=1200):
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:size])
            line = line[size:]
        if cur and len(cur) + len(line) + 1 > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def build(cfg=DEFAULT):
    db_path, emb_path, ids_path = data_paths(cfg.tag)
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT chunk_id, citation, article_title, text FROM chunks").fetchall()
    units, ids = [], []
    for cid, cit, title, text in rows:
        for w in windows(text):
            units.append(f"{cit} - {title}\n{w}")
            ids.append(cid)
    print(f"{db_path.name}: {len(rows)} chunks -> {len(units)} embedding units")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(units, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(emb_path, emb.astype("float32"))
    ids_path.write_text(json.dumps(ids))
    print(f"Saved {emb.shape} to {emb_path.name} and {ids_path.name}")


if __name__ == "__main__":
    build(VERSIONS[sys.argv[1]] if len(sys.argv) > 1 else DEFAULT)