import pathlib
import re
import requests
from bs4 import BeautifulSoup, Tag

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "dora.html"

URL = "https://eur-lex.europa.eu/eli/reg/2022/2554/oj/eng"
HEADERS = {"User-Agent": "RegulaAI research prototype"}

REGULATION = "Regulation (EU) 2022/2554"
ART_ID = re.compile(r"^art_\d+$")
PARA_ID = re.compile(r"^\d{3}\.\d{3}$")
CHAPTER_ID = re.compile(r"^cpt_[IVX]+$")


def fetch(path=RAW_PATH):
    p = pathlib.Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text, encoding="utf-8")
    return r.text


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def lines(node):
    """Flatten a block into text lines, keeping (a)/(b) table rows together."""
    out = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "p":
            t = clean(child.get_text(" ", strip=True))
            if t:
                out.append(t)
        elif child.name == "table":
            for tr in child.find_all("tr"):
                if tr.find_parent("table") is not child:   # skip nested tables
                    continue
                cells = tr.find_all("td", recursive=False)
                t = clean(" ".join(td.get_text(" ", strip=True) for td in cells))
                if t:
                    out.append(t)
        elif child.name == "div":
            out.extend(lines(child))
    return out


def chapter_info(art):
    cpt = art.find_parent("div", id=CHAPTER_ID)
    if not cpt:
        return "", ""
    num = cpt.find("p", class_="oj-ti-section-1")
    title = cpt.find("div", id=re.compile(r"^cpt_[IVX]+\.tit_1$"))
    return (clean(num.get_text()) if num else cpt["id"],
            clean(title.get_text(" ")) if title else "")


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for art in soup.find_all("div", id=ART_ID):
        number = int(art["id"].split("_")[1])
        title_el = art.find("p", class_="oj-sti-art")
        title = clean(title_el.get_text(" ")) if title_el else ""
        chapter, chapter_title = chapter_info(art)

        def add(para, text):
            records.append({
                "chunk_id": f"DORA-A{number}-P{para or 0}",
                "regulation": REGULATION,
                "chapter": chapter,
                "chapter_title": chapter_title,
                "article": number,
                "article_title": title,
                "paragraph": para,
                "text": text,
                "source": "EUR-Lex",
                "source_url": URL,
                "version": "Original Official Journal text",
                "language": "en",
            })

        loose = []   # content not inside a numbered-paragraph div
        for child in art.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "div" and PARA_ID.match(child.get("id", "")):
                ls = lines(child)
                if not ls:
                    continue
                m = re.match(r"^(\d+)\.\s+(.*)", ls[0])
                para = m.group(1) if m else None
                if m:
                    ls[0] = m.group(2)
                add(para, "\n".join(ls))
            elif child.name == "p" and "oj-normal" in (child.get("class") or []):
                loose.append(clean(child.get_text(" ")))
            elif child.name == "table":
                loose.extend(lines(Tag(name="div", builder=None)) if False else
                             [clean(" ".join(td.get_text(" ", strip=True)
                                             for td in tr.find_all("td", recursive=False)))
                              for tr in child.find_all("tr")
                              if tr.find_parent("table") is child])
        loose = [l for l in loose if l]
        if loose:
            add(None, "\n".join(loose))
    return records

def split_definitions(records):
    """Split Article 3 into one chunk per numbered definition."""
    out = []
    for r in records:
        if r["article"] == 3 and r["paragraph"] is None:
            intro, defs, cur = [], [], None
            for line in r["text"].split("\n"):
                m = re.match(r"^\((\d+)\)\s+", line)
                if m:
                    cur = {"n": int(m.group(1)), "lines": [line]}
                    defs.append(cur)
                elif cur:
                    cur["lines"].append(line)
                else:
                    intro.append(line)
            if intro:
                out.append({**r, "text": "\n".join(intro)})
            for d in defs:
                out.append({**r,
                            "chunk_id": f"DORA-A3-P{d['n']}",
                            "paragraph": str(d["n"]),
                            "text": "\n".join(d["lines"])})
        else:
            out.append(r)
    return out


def add_citations(records):
    for r in records:
        a, p = r["article"], r["paragraph"]
        if p is None:
            r["citation"] = f"Article {a}"
        elif a == 3:
            r["citation"] = f"Article 3, point ({p})"
        else:
            r["citation"] = f"Article {a}({p})"
    return records

def validate(records):
    arts = {r["article"] for r in records}
    assert arts == set(range(1, 65)), f"Missing articles: {set(range(1, 65)) - arts}"
    empty = [r["chunk_id"] for r in records if not r["text"].strip()]
    assert not empty, f"Empty chunks: {empty}"
    ids = [r["chunk_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids"
    print(f"OK: {len(records)} chunks across {len(arts)} articles")
    defs = sorted(int(r["paragraph"]) for r in records if r["article"] == 3 and r["paragraph"])
    assert defs == list(range(1, len(defs) + 1)), "Gap in Article 3 definitions"
    print(f"Article 3 definitions: {len(defs)}")


def main():
    html = fetch()
    records = parse(html)
    validate(records)

    # Spot checks
    for r in records:
        if r["article"] == 19 and r["paragraph"] in ("1", "4"):
            print(f"\n{r['chunk_id']} | {r['chapter']} | {r['article_title']}")
            print(r["text"][:400], "...")

    longest = sorted(records, key=lambda r: len(r["text"]), reverse=True)[:5]
    print("\nLongest chunks:")
    for r in longest:
        print(f"  {r['chunk_id']}: {len(r['text']):,} chars")
    records = add_citations(split_definitions(parse(html)))
    validate(records)

if __name__ == "__main__":
    main()