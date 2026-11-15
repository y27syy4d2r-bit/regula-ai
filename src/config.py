from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Cfg:
    name: str
    tag: str = ""
    split_definitions: bool = True
    w_sem: float = 1.0
    w_kw: float = 1.0
    rrf_k: int = 60
    explicit_boost: float = 1.0
    title_bonus: float = 0.03
    use_direct: bool = True
    use_definition: bool = True
    use_title: bool = True
    use_reranker: bool = False
    rerank_top_n: int = 20
    use_lead: bool = True          # lead-paragraph expansion in answer.py/prepare()


def data_paths(tag=""):     
    d = ROOT / "data"
    return d / f"dora{tag}.db", d / f"embeddings{tag}.npy", d / f"embedding_ids{tag}.json"


DEFAULT = Cfg("current")

# Cumulative versions: each one differs from the previous in exactly one setting.
A  = Cfg("A", use_direct=False, use_definition=False, use_title=False)   # semantic + keyword only
B0 = replace(A,  name="B0", use_direct=True)        # + explicit "Article N" lookup
B1 = replace(B0, name="B1", use_definition=True)    # + definition routing
B2 = replace(B1, name="B2", use_title=True)         # + title bonus
B3 = replace(B2, name="B3", w_sem=1.5)              # + semantic weight (tune on dev only)
C  = replace(B3, name="C",  use_reranker=True)      # + cross-encoder reranker

V0 = Cfg("V0", tag="_V0", split_definitions=False,
         use_direct=False, use_definition=False, use_title=False)

VERSIONS = {c.name: c for c in (DEFAULT, A, B0, B1, B2, B3, C, V0)}