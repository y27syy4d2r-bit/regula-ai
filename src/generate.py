import os
import re
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from ingest import ROOT

load_dotenv(ROOT / ".env")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM = """You answer questions about the EU Digital Operational Resilience Act \
(Regulation (EU) 2022/2554), using ONLY the numbered provisions supplied by the user message.

Rules:
1. Use only the supplied provisions. Never use outside knowledge, other regulations, \
or technical standards you may know of.
2. End every sentence with the label(s) of the provision(s) that support it, like [S1] or [S2][S4]. \
Never cite a label that was not supplied.
3. Preserve the legal force of the wording: do not turn "shall" into "should" or "may" into "must".
4. Set `coverage` to "full" if the provisions contain everything needed to answer; "partial" \
if they contain some of what was asked but not all (for example the obligation is stated but \
the detailed rules or time limits are delegated to technical standards); "none" only if they \
contain nothing relevant. For "full" and "partial", `answer` must state what the provisions DO \
say, with citations. Never leave the answer empty or merely say the text does not answer the \
question. Describe the missing part in `gaps`.
5. If a provision says the detail will be set out in regulatory or implementing technical \
standards, say so and do not invent the detail.
6. If a provision refers to another article that was not supplied, say that it refers to \
that article rather than describing its content.
7. Write in plain English, concisely, with no legal advice and no recommendations.
8. Fill `sources_used` with the labels you cited.
9. Answer the question that was asked, in the first sentence. If the provisions cover only \
related requirements and not what was asked (for example, the question asks for a list but \
the provisions only describe frequency), do not present the related requirements as the answer: \
say so in the first sentence and describe the missing part in `gaps`.
10. If the question asks for types, kinds, requirements, steps, criteria or any list, \
enumerate the items exactly as the provisions list them, completely and in their original \
order. Never replace a list with a general description such as "appropriate tests". \
If the provision contains the list, the answer must contain it.
11. Combining facts from several provisions is expected. Do not refuse just because no single \
provision describes the relationship the question asks about: state what each provision says, \
and describe their relationship only as far as the wording supports it.
12. For yes/no, "does A satisfy B" or comparison questions, the first sentence must be the \
direct answer (Yes, No, or Not necessarily), followed by the reasons from the provisions, \
comparing what each provision requires."""


class SourceUse(BaseModel):
    label: str       # e.g. "S3"
    subject: str     # short phrase: what that provision is about

class Grounded(BaseModel):
    coverage: Literal["full", "partial", "none"]
    answer: str
    gaps: str
    sources_used: list[SourceUse]



def build_prompt(question, results):
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[S{i}] {r['citation']} - {r['article_title']}\n{r['text']}")
    return ("PROVISIONS:\n\n" + "\n\n".join(blocks) +
            f"\n\nQUESTION: {question}")


CITE = re.compile(r"\[S(\d+)\]")
ART = re.compile(r"\bArticles?\s+(\d{1,2})\b", re.I)


def _reading_order(r):
    p = r["paragraph"]
    return (r["article"], int(p) if p and p.isdigit() else 0)


def verify(parsed, results):
    """Return a list of problems; an empty list means the answer passed."""
    problems = []
    n = len(results)
    labels = {int(x) for x in CITE.findall(parsed.answer)}
    bad = [x for x in labels if not 1 <= x <= n]
    if bad:
        problems.append(f"cites unsupplied sources: {bad}")
    if parsed.coverage != "none":
        if not labels:
            problems.append("answer has no citations")
        for s in re.split(r"(?<=[.!?\]])\s+", parsed.answer.strip()):
            if s and not CITE.search(s):
                problems.append(f"uncited sentence: {s[:60]}")
    allowed = {r["article"] for r in results}
    for r in results:
        allowed |= {int(x) for x in ART.findall(r["text"])}
    stray = {int(x) for x in ART.findall(parsed.answer)} - allowed
    if stray:
        problems.append(f"mentions articles not in the supplied text: {sorted(stray)}")
    return problems

def render(answer, results):
    """Swap [S1] labels for readable citations like [Article 19(1)]."""
    def sub(m):
        i = int(m.group(1))
        return f"[{results[i - 1]['citation']}]" if 1 <= i <= len(results) else m.group(0)
    return CITE.sub(sub, answer)

def generate(question, results, retries=1):
    results = sorted(results, key=_reading_order)
    client = genai.Client()          # reads GOOGLE_API_KEY
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0,
        response_mime_type="application/json",
        response_schema=Grounded,
    )
    prompt = build_prompt(question, results)
    last = []
    for _ in range(retries + 1):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            parsed = Grounded.model_validate_json(resp.text)
        except Exception as e:                       # rate limit, bad JSON, network
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        last = verify(parsed, results)
        if not last:
            used = []
            for s in parsed.sources_used:
                m = re.search(r"\d+", s.label)
                if m and 1 <= int(m.group()) <= len(results):
                    used.append((results[int(m.group()) - 1]["citation"], s.subject))
            return {"ok": True,
                    "coverage": parsed.coverage,
                    "answerable": parsed.coverage != "none",
                    "gaps": parsed.gaps,
                    "answer": render(parsed.answer, results),
                    "used": used}
    return {"ok": False, "reason": "failed verification: " + "; ".join(last)}