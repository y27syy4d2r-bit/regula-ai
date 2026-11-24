# Retrieval and Ranking Methods

## Purpose

This document describes the six cumulative retrieval configurations
evaluated in RegulaAI: **A, B0, B1, B2, B3, and C**.

These are not six independent retrieval systems. They form an
incremental sequence: each configuration carries forward the
previous settings and changes one setting at a time. This makes it
possible to observe the measured effect of each added signal on the
development set, while keeping the underlying pipeline largely fixed.

## Shared retrieval foundation

Every configuration uses two retrieval channels:

1.  **Semantic retrieval** with `BAAI/bge-small-en-v1.5`.
2.  **Keyword retrieval** with SQLite FTS5/BM25.

### Semantic retrieval

The query is encoded with the BGE query prefix:

``` text
Represent this sentence for searching relevant passages:
```

Passage embeddings are normalized. Similarity is computed by dot
product, equivalent to cosine similarity for normalized vectors. The
implementation searches the best embedding windows and keeps the highest
window score for each legal chunk.

### Keyword retrieval

The query is tokenized, lowercased, and filtered using a stop-word list.
Remaining tokens are combined using an OR expression for FTS5. SQLite
BM25 determines the keyword result order.

## Ranking formula

Let:

- $r_s(d)$ be the one-based rank of chunk $d$ in the semantic results (0 if $d$ is not in that list);
- $r_k(d)$ be its one-based rank in the keyword results (0 if absent);
- $w_{\text{sem}}$ and $w_{\text{kw}}$ be the channel weights;
- $K$ be the RRF constant (`rrf_k`, set to 60);
- $I_{\text{explicit}}(d) = 1$ if $d$ was returned by direct article lookup or definition routing, otherwise 0;
- $I_{\text{title}}(d) = 1$ if $d$ is the first chunk of an article whose title words all appear in the question, otherwise 0;
- $b_{\text{explicit}}$ be `explicit_boost` (1.0);
- $b_{\text{title}}$ be `title_bonus` (0.03).

The fused score implemented by the code is:

$$
S(d) = b_{\text{explicit}}\, I_{\text{explicit}}(d) + b_{\text{title}}\, I_{\text{title}}(d) + [r_s(d) > 0]\,\frac{w_{\text{sem}}}{K + r_s(d)} + [r_k(d) > 0]\,\frac{w_{\text{kw}}}{K + r_k(d)}
$$

A channel contributes nothing when a chunk is absent from that channel's
returned list. In the implementation, the loop uses zero-based
enumeration and adds the contribution (w/(K + rank + 1)), which is the
one-based-rank form shown above.

**Important:** after sorting by this score, direct and definition
matches are separated into a `pinned` list and placed before all other
ranked candidates. Thus the final order is not determined by the
additive score alone. The pinned explicit matches retain their relative
score order, followed by the remaining candidates.

## Configuration sequence

| Version | Direct lookup | Definition routing | Title bonus | Semantic weight | Cross-encoder |
|:-------:|:-------------:|:------------------:|:-----------:|:---------------:|:-------------:|
| A       | Off           | Off                | Off         | 1.0             | Off           |
| B0      | On            | Off                | Off         | 1.0             | Off           |
| B1      | On            | On                 | Off         | 1.0             | Off           |
| B2      | On            | On                 | On          | 1.0             | Off           |
| B3      | On            | On                 | On          | 1.5             | Off           |
| C       | On            | On                 | On          | 1.5             | On            |

All configurations retain keyword weight 1.0 and RRF constant 60. The explicit boost is 1.0 and the title bonus is 0.03.

### A --- Semantic + keyword baseline

Configuration:

``` python
A = Cfg(
    "A",
    use_direct=False,
    use_definition=False,
    use_title=False
)
```

A combines semantic and keyword rankings using RRF. It does not use
direct article lookup, definition routing, title matching, or
cross-encoder reranking.

Formula:

$$ S_A(d) = `\frac{1}{60+r_s(d)}`{=tex} + `\frac{1}{60+r_k(d)}`{=tex} $$

This is the baseline against which the later additions are compared.

### B0 --- Add direct article lookup

Configuration change:

``` python
B0 = replace(A, name="B0", use_direct=True)
```

B0 adds explicit article/paragraph lookup. The query parser recognises
references such as `Article 12`, `Art. 12`, and `Article 12(2)`. It then
retrieves matching chunk IDs from SQLite.

Direct matches receive the explicit boost and are pinned ahead of
non-explicit candidates. The implementation caps direct lookup at six
chunks.

The score is the A score plus:

$$ 1.0 `\cdot `{=tex}I\_{direct}(d) $$

The final pinned ordering means direct hits are placed ahead of the rest
regardless of whether their RRF score alone would have put them there.

### B1 --- Add definition routing

Configuration change:

``` python
B1 = replace(B0, name="B1", use_definition=True)
```

B1 adds routing for definition-style questions to Article 3. The code
recognises common forms such as "What is ...?", "Define ...",
"definition of ...", and related patterns. It normalizes the requested
term and compares it with the labels of Article 3 definition chunks.
Exact matches are preferred; a limited substring fallback is also used.

Definition matches receive the same explicit boost as direct hits and
are included in the pinned set:

$$ S\_{B1}(d) = S_A(d) + 1.0 `\cdot `{=tex}I\_{direct
`\lor `{=tex}definition}(d) $$

with direct and definition hits pinned before the remaining candidates.

### B2 --- Add article-title matching

Configuration change:

``` python
B2 = replace(B1, name="B2", use_title=True)
```

B2 adds a title-matching signal. The code tokenizes and lightly
normalizes article-title words. An article is matched when its
meaningful title tokens are all present in the query; Article 3 is
excluded from this title index.

A title match receives a small additive bonus:

$$ S\_{B2}(d) = S\_{B1}(d) + 0.03 `\cdot `{=tex}I\_{title}(d) $$

Unlike direct and definition matches, title matches are not pinned. They
compete through the combined score.

### B3 --- Increase semantic channel weight

Configuration change:

``` python
B3 = replace(B2, name="B3", w_sem=1.5)
```

B3 increases the semantic RRF weight from 1.0 to 1.5. Keyword weight
remains 1.0.

$$ S\_{B3}(d) = 1.0 `\cdot `{=tex}I\_{explicit}(d) + 0.03
`\cdot `{=tex}I\_{title}(d) + `\frac{1.5}{60+r_s(d)}`{=tex} +
`\frac{1.0}{60+r_k(d)}`{=tex} $$

The code comment explicitly says to tune this value on the development
set only.

### C --- Add cross-encoder reranking

Configuration change:

``` python
C = replace(B3, name="C", use_reranker=True)
```

C retains B3's retrieval and score fusion, then applies the
`BAAI/bge-reranker-base` cross-encoder to the first 20 non-pinned
candidates.

For each candidate, the reranker receives a pair consisting of the query
and a text representation containing the citation, article title, and up
to the first 2,000 characters of the provision. Candidates are reordered
by the cross-encoder's predicted score.

Pinned direct and definition matches are excluded from this reranking
step and remain at the front. Candidates beyond the reranking window
retain their previous order after the reranked subset.

This stage is a reordering operation, not another additive term in the
RRF formula.

## Other retrieval and context details

-   Semantic and keyword retrieval each produce ranked candidate lists.
-   The implementation uses up to 20 candidates from each channel before
    fusion.
-   The final retrieval function returns the requested top-k results.
-   The application can then append lead paragraphs from up to three
    retrieved articles (excluding Article 3), subject to a ten-result
    context cap.
-   The optional reranker is disabled in the default configuration.

## Interpretation cautions

1.  **Cumulative ablation:** A→B0→B1→B2→B3→C is a sequential experiment.
    It estimates the incremental effect of each change in that sequence;
    it is not a full factorial experiment testing every combination.
2.  **Pinned matches:** direct and definition matches are explicitly
    promoted to the top. Their effect is stronger than a small score
    bonus.
3.  **Development-set tuning:** B3's semantic weight was changed with a
    development-set tuning comment. The test split should remain
    untouched until configuration choices are fixed.
4.  **Reranking is not guaranteed to help:** C can reduce retrieval
    quality. The cross-encoder should be retained only if held-out
    evaluation supports it.
5.  **Top-k cap:** direct article lookup is capped at six chunks. A
    request for an entire long article may therefore not retrieve every
    chunk.
