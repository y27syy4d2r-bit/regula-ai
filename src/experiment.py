import csv
import json
import sys
from dataclasses import asdict
from statistics import mean

from answer import prepare
from config import ROOT, VERSIONS
from retrieve import search


def sp(s):
    return {x.strip() for x in (s or "").split("|") if x.strip()}


def load_gold(split):
    path = ROOT / "eval" / "gold.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [r for r in rows if split == "all" or r["split"] == split]


def evaluate(version, split):
    cfg = VERSIONS[version]
    per_q = []
    for r in load_gold(split):
        out = prepare(r["question"], k=8, cfg=cfg)          # what the app would do
        if r["type"] == "out_of_scope":
            per_q.append({"id": r["id"], "type": r["type"],
                          "rejected": out["status"] != "ok"})
            continue
        ga = {int(a) for a in sp(r["expected_articles"])}
        gc = sp(r.get("gold_citations"))
        res, _ = search(r["question"], k=10, cfg=cfg)        # raw ranking
        arts = [x["article"] for x in res]
        cits = [x["citation"] for x in res]
        ctx = {x["citation"] for x in out["results"]}        # after lead-paragraph expansion
        rank = next((i + 1 for i, a in enumerate(arts) if a in ga), None)
        per_q.append({
            "id": r["id"], "type": r["type"], "rank": rank,
            "art3": int(any(a in ga for a in arts[:3])),
            "art6": int(any(a in ga for a in arts[:6])),
            "complete6": int(ga <= set(arts[:6])),
            "mrr": 1 / rank if rank else 0.0,
            "para6": len(gc & set(cits[:6])) / len(gc) if gc else None,
            "ctx_para": len(gc & ctx) / len(gc) if gc else None,
            "false_gate": int(out["status"] != "ok"),
            "missing_para6": sorted(gc - set(cits[:6])),
            "top6": cits[:6],
        })

    ins = [q for q in per_q if "rank" in q]
    oos = [q for q in per_q if "rejected" in q]

    def avg(key):
        vals = [q[key] for q in ins if q[key] is not None]
        return round(mean(vals), 3) if vals else None

    summary = {
        "n_in_scope": len(ins),
        "art@3": avg("art3"), "art@6": avg("art6"), "complete@6": avg("complete6"),
        "mrr": avg("mrr"), "para@6": avg("para6"), "ctx_para": avg("ctx_para"),
        "false_gate": avg("false_gate"),
        "oos_rejected": f"{sum(q['rejected'] for q in oos)}/{len(oos)}",
    }
    return {"version": version, "split": split, "config": asdict(cfg),
            "summary": summary, "per_question": per_q}


def main(versions, split):
    out_dir = ROOT / "eval" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in versions:
        result = evaluate(v, split)
        (out_dir / f"retrieval_{v}_{split}.json").write_text(json.dumps(result, indent=1))
        print(v, split, result["summary"])


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else "dev")