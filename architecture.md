# RegulaAI Architecture

## Overview

RegulaAI is a retrieval-augmented generation (RAG) prototype for
answering questions about the original English text of Regulation (EU)
2022/2554, the Digital Operational Resilience Act (DORA).

The application separates the work into two main stages:

1.  **Retrieval:** identify the DORA provisions most relevant to a
    question.
2.  **Answer generation:** ask Gemini to answer using only the retrieved
    provisions, then run deterministic citation and article-reference
    checks before displaying the answer.

The knowledge base is deliberately narrow. It uses the original Official
Journal text of DORA and does not include delegated or implementing
regulations, regulatory or implementing technical standards (RTS/ITS),
or other supplementary material. The application warns users about this
limitation.

## Pipeline at a glance

``` text
EUR-Lex DORA HTML
       |
       v
Fetch and cache source HTML
       |
       v
Parse legal structure
(articles, chapters, paragraphs, titles)
       |
       v
Split Article 3 definitions
       |
       v
Validate records and assign citations
       |
       +------------------------------+
       |                              |
       v                              v
SQLite chunks table             SQLite FTS5 index
       |                              |
       v                              |
Embedding windows                      |
       |                              |
       v                              |
BGE sentence embeddings               |
       |                              |
       +---------------+--------------+
                       |
User question          |
       |               |
       v               v
Semantic retrieval + keyword retrieval
       |
       +--> optional explicit Article lookup
       +--> optional definition routing
       +--> optional title matching
       |
       v
Reciprocal-rank fusion and explicit-hit pinning
       |
       v
Optional cross-encoder reranking
       |
       v
Top-k provisions
       |
       v
Lead-paragraph context expansion
       |
       v
Gemini structured answer (optional in UI)
       |
       v
Citation / article-reference verification
       |
       v
Rendered answer and source provisions
```

## 1. Source acquisition

The ingestion code retrieves the English DORA text from EUR-Lex:

-   Regulation: **Regulation (EU) 2022/2554**
-   Source URL: `https://eur-lex.europa.eu/eli/reg/2022/2554/oj/eng`
-   Source label: EUR-Lex
-   Version label stored in records: `Original Official Journal text`

The fetched HTML is cached at `data/raw/dora.html`. If that file already
exists, the fetch function reads the local copy rather than requesting
the source again. This makes builds repeatable against the cached input,
but it also means the source is not automatically refreshed by that
function.

## 2. Legal-structure parsing and chunking

BeautifulSoup parses the HTML. The parser extracts article number,
article title, chapter information, paragraph number, and provision
text. It handles paragraph blocks and table rows and stores provisions
as structured records.

Each record carries metadata including:

-   `chunk_id`
-   regulation name
-   chapter and chapter title
-   article number and title
-   paragraph number
-   human-readable citation
-   provision text
-   source URL and version
-   language

The ingestion code splits Article 3 into individual definition chunks.
This supports targeted retrieval of terms such as "vulnerability" or
"ICT asset". The chunk IDs use a consistent article/paragraph pattern,
for example `DORA-A3-P16`.

Citations are generated from the parsed structure. Ordinary provisions
use forms such as `Article 12(2)`, while Article 3 definitions use
`Article 3, point (16)`.

Validation checks that all 64 articles are represented, that chunks are
non-empty, that chunk IDs are unique, and that Article 3 definition
numbering has no gaps.

## 3. Storage and lexical index

The structured records are stored in SQLite in a `chunks` table. The
database also contains an FTS5 virtual table, `chunks_fts`, indexing the
article title and provision text.

The FTS5 index uses the `porter unicode61` tokenizer. Keyword retrieval
removes a small stop-word list, tokenizes the query, and submits the
remaining terms as an OR expression to FTS5. Results are ordered using
SQLite's BM25 ranking.

## 4. Semantic index

The embedding build process creates text windows from each provision.
Each embedding unit includes the citation and article title followed by
a window of provision text. The default window size is 1,200 characters.

The sentence-transformer model is `BAAI/bge-small-en-v1.5`. Embeddings
are normalized and stored as float32 NumPy arrays, alongside a JSON
mapping back to chunk IDs.

At query time, the query is prefixed with:

`Represent this sentence for searching relevant passages:`

The query embedding is normalized. Similarity is calculated by dot
product against the normalized passage embeddings, which is equivalent
to cosine similarity for these vectors. If several windows map to the
same legal chunk, the highest-scoring window score is retained for that
chunk.

## 5. Retrieval and ranking

The baseline retrieval combines semantic and keyword rankings using
reciprocal-rank fusion (RRF). Optional signals are added in the
cumulative configurations:

-   direct lookup when a question explicitly references an article or
    paragraph;
-   routing of definition-style questions to Article 3 definitions;
-   a bonus when the query contains all meaningful tokens from an
    article title;
-   a higher semantic weight;
-   optional cross-encoder reranking.

The exact configurations and formulas are documented in
[retrieval-ranking.md](retrieval-ranking.md).

Explicit direct and definition hits are not merely given a score bonus:
after score fusion they are pinned ahead of the remaining candidates.
This is an important detail when interpreting the ranking behaviour.

## 6. Context expansion

After retrieval, `prepare()` may append the first paragraph of up to
three distinct retrieved articles, excluding Article 3. This is intended
to provide introductory context around retrieved provisions. The
expanded context is capped at ten results.

The retrieval evaluation distinguishes the raw ranking from this
expanded context: `para@6` is calculated from the raw top six results,
while `ctx_para` is calculated from the result set after lead-paragraph
expansion.

## 7. Answer generation

The Streamlit interface allows the user to enter a DORA question, select
the number of provisions to retrieve, and toggle Gemini answer
generation.

When enabled, the application sends the retrieved provisions and
question to Gemini. The prompt instructs the model to:

-   use only the supplied provisions;
-   preserve the legal force of the text (for example, not changing
    "shall" into "should");
-   cite every answer sentence using supplied source labels;
-   distinguish full, partial, and absent coverage;
-   identify gaps rather than inventing details delegated to technical
    standards;
-   avoid describing the content of articles that were not supplied.

The response is parsed into a Pydantic schema with `coverage`, `answer`,
`gaps`, and `sources_used` fields. Generation uses a structured JSON
response schema and temperature 0.

## 8. Verification and display

Before an answer is accepted, the verifier checks that:

1.  every citation label refers to a supplied result;
2.  an answer with coverage other than `none` contains citations;
3.  every sentence contains a citation;
4.  article numbers mentioned in the answer are either among the
    retrieved articles or referenced in the supplied provision text.

Accepted labels such as `[S1]` are rendered as legal citations such as
`[Article 12(2)]`. If generation fails verification, the interface falls
back to showing the retrieved provisions rather than presenting the
answer as verified.

These checks are structural safeguards. They do **not** establish that a
cited provision semantically entails every claim in the sentence. The
system should therefore be described as having citation-format and
reference checks, not a complete legal fact-checking or entailment
verifier.

## 9. Scope and limitations

-   The knowledge base contains the original DORA text only.
-   The local source cache is not automatically refreshed by the fetch
    function.
-   The parser and chunking strategy can affect retrieval, especially
    for long or structurally complex provisions.
-   Direct article lookup has a six-chunk cap, which can truncate
    retrieval for a long article request.
-   The cross-encoder is optional and disabled in the default
    configuration.
-   The citation verifier checks labels, sentence citation presence, and
    article references, but not semantic entailment.
-   The system is a research prototype and not legal advice.

## Main implementation modules

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  `ingest.py`                         Fetch/cache EUR-Lex HTML, parse
                                      articles, split Article 3
                                      definitions, validate records

  `build_db.py`                       Build SQLite `chunks` and FTS5
                                      tables

  `embed.py`                          Create passage windows and semantic
                                      embeddings

  `retrieve.py`                       Semantic, keyword, direct,
                                      definition, title retrieval; fusion
                                      and reranking

  `config.py`                         Model names, paths, ranking
                                      parameters, and experiment
                                      configurations

  `answer.py`                         Scope checks, retrieval
                                      preparation, lead-paragraph
                                      expansion

  `generate.py`                       Gemini prompt, structured output,
                                      citation checks and rendering

  `app.py`                            Streamlit user interface

  `eval/`                             Gold questions, evaluation scripts
                                      and saved run outputs
  -----------------------------------------------------------------------
