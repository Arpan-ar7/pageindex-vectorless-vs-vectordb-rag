# VectorDB vs Vectorless RAG: results

Three systems, the same 10 questions about "Attention Is All You Need", the same answering model (**Claude Sonnet**, on the Claude subscription). Only the retrieval differs.

Notebook: [`src/vectorless_vs_vectordb_rag/compare/compare.ipynb`](../src/vectorless_vs_vectordb_rag/compare/compare.ipynb)
Script: `uv run vectorless-vs-vectordb-rag compare` ([`compare/evaluate.py`](../src/vectorless_vs_vectordb_rag/compare/evaluate.py))
Raw data: [`results/compare/runs.json`](../results/compare/runs.json), [`per_question.csv`](../results/compare/per_question.csv), [`summary.csv`](../results/compare/summary.csv)

| System | How it finds the text |
|---|---|
| **Vector RAG** | one similarity search in Chroma, top 5 chunks of ~750 characters (`RecursiveCharacterTextSplitter`) |
| **Agentic vector RAG** | similarity search, then Claude grades the chunks and may retry with a better query |
| **PageIndex (local)** | Claude reads the document tree and opens the pages it chooses (whole pages) |

PageIndex **cloud** was not included: `PAGEINDEX_API_KEY` is commented out in `.env`. The local tree is the weaker one (see [pageindex_cloud_vs_local.md](pageindex_cloud_vs_local.md)), so the cloud tree would likely do at least as well.

---

## Results

![summary](../results/compare/summary.png)

| | Vector RAG | Agentic vector RAG | PageIndex (local) |
|---|---|---|---|
| Hit rate (read ≥ 1 correct page) | 100% | 100% | 100% |
| Recall of correct pages | 95% | 95% | 95% |
| Precision (correct / pages read) | 53% | 53% | **72%** |
| Answer score (1-5, judged) | 4.4 | 4.7 | **4.8** |
| Seconds per question | **7.0** | 11.5 | 7.7 |
| Characters read by the LLM | **3.5k** | 7.0k | 8.7k |
| LLM calls per question | **1** | 2 | 2.1 |

![per question](../results/compare/recall_heatmap.png)

---

## What we learned

### 1. Finding the right page was easy for all three
Short, well-structured paper: every system reached a correct page for every question. Vector search reached 95% recall (it missed page 7 for q10) and pulled in more unrelated pages (53% precision vs 72%), because 5 chunks span ~3 pages.

### 2. Answer quality depends on *what* the LLM reads
Vector RAG hands Claude **fragments**. The judge's most common complaint about vector answers:
*"it says the table / footnote is cut off, but the reference page contains it"* (q2, q7, q9, q10).
PageIndex opens **whole pages**, so tables and footnotes arrive intact.

> **q9: How does changing the number of attention heads affect quality?**
> Vector RAG: "single-head is 0.9 BLEU worse, too many heads also hurts; the context doesn't give the exact numbers" (score 4)
> PageIndex: quoted Table 3 row by row, h=1: 24.9, 4: 25.5, 8: 25.8, 16: 25.8, 32: 25.4 BLEU (score 5)

### 3. Agentic RAG doubled the cost for no real gain on this set
The grader said "relevant" on the first try for all 10 questions, so the retry loop never ran and it read exactly the same pages as plain vector RAG. Its higher score (4.7 vs 4.4) therefore comes from judge variation, not better retrieval. The grading question (*"does the context contain the answer?"*) passes partial context. Ways to make it useful:
- grade for *completeness*, not just relevance
- on retry, widen the search (bigger `TOP_K`) instead of only rewriting the query

### 4. Cost and speed
Vector RAG is the cheapest: 1 call and the least text. PageIndex reads ~2.4x more text (the tree plus full pages) but is only ~1.5 s slower: its 2 model turns share one Claude Code process, while agentic RAG starts Claude twice.

### 5. Chunking: word windows vs recursive splitter
Vector RAG first used a hand-written chunker (120-word windows, 30-word overlap). It now uses LangChain's
`RecursiveCharacterTextSplitter` (750 characters, 190 overlap, about the same size), which prefers to cut at
paragraph or line breaks. Only the splitting method changed.

| | Word chunker (old) | Recursive splitter (new) |
|---|---|---|
| Chunks | 71 (avg ~708 chars) | 73 (avg ~648 chars) |
| Recall (vector / agentic) | **100% / 100%** | 95% / 95% |
| Precision (vector / agentic) | **62% / 62%** | 53% / 53% |
| Answer score (vector / agentic) | 4.2 / 4.1 | **4.4 / 4.7** |

**Verdict: no clear winner.** Retrieval got slightly worse (q10 missed page 7), while judged scores rose. The two
vector systems retrieved the *same* pages yet scored differently, so the judge itself varies by about ±0.3:
treat small score changes as noise. Chunk size and top-k matter more than the splitting rule.
Old results are kept in `results/compare_word_chunks/`.

---

## When to use which

| Situation | Better choice |
|---|---|
| Many documents, quick factual lookups, tight budget | Vector RAG |
| Long structured documents (papers, reports, manuals), answers that need whole tables or sections | PageIndex / vectorless |
| You need an explanation of *why* a page was chosen | PageIndex (the tool calls show its reasoning path) |
| Bad first searches are common (vague questions, jargon mismatch) | Agentic RAG, with a stricter grader |

**Idea to test next:** *parent-page retrieval*. Search small chunks (precise matching), then send the **whole page** each chunk came from (complete context). It combines cheap vector search with the full-page context that made PageIndex's answers better.

---

## Caveats
- 10 questions on one 15-page paper: a small sample on an easy document for vector search.
- The judge is Claude, the same family as the answering model. Answers were shuffled and unlabeled to limit bias.
- PageIndex sometimes added correct facts from the model's own memory (FLOPs in q6); the judge flagged them as "not on the shown page".
- Latency includes ~2 s of Claude Code start-up per call, and 4 runs ran in parallel.
- Answers are cached in `runs.json`; delete it to run everything again from scratch.
