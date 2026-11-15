import json
import sys
from config import ROOT


def load(v, split):
    p = ROOT / "eval" / "runs" / f"retrieval_{v}_{split}.json"
    return json.loads(p.read_text())


def main(versions, split):
    data = {v: load(v, split) for v in versions}
    keys = ["art@3", "art@6", "complete@6", "mrr", "para@6",
            "ctx_para", "false_gate", "oos_rejected"]
    print(f"\n{split} split\n{'metric':<14}" + "".join(f"{v:>9}" for v in versions))
    for k in keys:
        print(f"{k:<14}" + "".join(f"{str(data[v]['summary'].get(k)):>9}" for v in versions))

    for a, b in zip(versions, versions[1:]):
        qa = {q["id"]: q for q in data[a]["per_question"]}
        qb = {q["id"]: q for q in data[b]["per_question"]}
        print(f"\nChanges {a} -> {b}")
        changed = False
        for qid, x in qa.items():
            y = qb[qid]
            if "rank" not in x:
                continue
            if (x["rank"], x["para6"]) != (y["rank"], y["para6"]):
                changed = True
                print(f"  {qid}: rank {x['rank']} -> {y['rank']}, "
                      f"para@6 {x['para6']} -> {y['para6']}, "
                      f"still missing: {y['missing_para6']}")
        if not changed:
            print("  none")


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else "dev")