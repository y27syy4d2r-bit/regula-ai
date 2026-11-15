import re
from retrieve import search
from config import DEFAULT
from retrieve import search, _resources


COSINE_FLOOR = 0.55   # catches the absurd questions only;

OTHER_REGS = re.compile(
    r"\b(gdpr|general data protection|mica|crypto[- ]asset|nis ?2|ai act|psd2|"
    r"mifid|basel|sox|sarbanes|iso ?27001|nist|pci)\b", re.I)

DELEGATED = re.compile(
    r"(?i:\b(threshold|template|technical standard|criteria|materiality|delegated|"
    r"deadline|time limit|timeline|within \d+ hours?|how (?:long|quickly))s?\b)"
    r"|\b(?:RTS|ITS)\b")       

SCOPE_NOTE = ("Based on the original text of Regulation (EU) 2022/2554 only. It does not "
              "include delegated regulations, implementing regulations or technical "
              "standards (RTS/ITS), so it is not a complete picture of current DORA "
              "obligations. Not legal advice.")


def prepare(query, k=8,cfg=DEFAULT):
    notes, status = [], "ok"

    m = OTHER_REGS.search(query)

    if m:
        status = "other_regulation"
        notes.append(f"Your question mentions **{m.group(0)}**, which is not part of the "
                     "base DORA text this tool covers. The provisions below are the "
                     "closest DORA passages, not answers about it.")

    results, top_cosine = search(query, k=k,cfg=cfg)
    results = add_lead_paragraphs(results,cfg)
    explicit = any("direct" in r["via"] or "definition" in r["via"] for r in results)

    if status == "ok" and top_cosine < COSINE_FLOOR and not explicit:
        status = "low_confidence"
        notes.append("I couldn't find a passage in the base DORA text that clearly "
                     "addresses this question.")

    if status == "ok" and DELEGATED.search(query):
        notes.append("DORA delegates much of this detail (thresholds, templates, "
                     "criteria) to technical standards, which this version does not "
                     "cover. The base articles below set out the obligation, not the "
                     "detailed rules.")

    return {"status": status, "notes": notes, "results": results,
            "top_cosine": top_cosine}

def add_lead_paragraphs(results, cfg, n_articles=3, cap=10):
    con = _resources(cfg.tag)[0]
    seen = {r["chunk_id"] for r in results}
    out, arts = list(results), []
    for r in results:
        if r["article"] != 3 and r["article"] not in arts:
            arts.append(r["article"])
    for art in arts[:n_articles]:
        row = con.execute("SELECT * FROM chunks WHERE article=? AND paragraph='1'",
                          (art,)).fetchone()
        if row and row["chunk_id"] not in seen:
            d = dict(row)
            d.update(score=0, cosine=None, via=["lead"])
            out.append(d)
            seen.add(d["chunk_id"])
    return out[:cap]            