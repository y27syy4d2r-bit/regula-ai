import sqlite3
import sys

from config import DEFAULT, VERSIONS, data_paths
from ingest import add_citations, fetch, parse, split_definitions, validate

COLUMNS = ["chunk_id", "regulation", "chapter", "chapter_title", "article",
           "article_title", "paragraph", "citation", "text", "source",
           "source_url", "version", "language"]


def build_db(cfg=DEFAULT):
    db_path, _, _ = data_paths(cfg.tag)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    records = parse(fetch())
    if cfg.split_definitions:
        records = split_definitions(records)
    records = add_citations(records)
    validate(records)

    db_path.unlink(missing_ok=True)          # rebuild from scratch each time
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, regulation TEXT, chapter TEXT,
            chapter_title TEXT, article INTEGER, article_title TEXT,
            paragraph TEXT, citation TEXT, text TEXT, source TEXT,
            source_url TEXT, version TEXT, language TEXT
        )""")
    con.executemany(
        f"INSERT INTO chunks VALUES ({','.join('?' * len(COLUMNS))})",
        [tuple(r[c] for c in COLUMNS) for r in records])

    con.execute("""CREATE VIRTUAL TABLE chunks_fts USING fts5(
        chunk_id UNINDEXED, article_title, text, tokenize='porter unicode61')""")
    con.execute("""INSERT INTO chunks_fts (chunk_id, article_title, text)
                   SELECT chunk_id, article_title, text FROM chunks""")
    con.commit()
    con.close()
    print(f"Wrote {len(records)} chunks to {db_path}")


if __name__ == "__main__":
    build_db(VERSIONS[sys.argv[1]] if len(sys.argv) > 1 else DEFAULT)