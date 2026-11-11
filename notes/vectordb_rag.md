# VectorDB RAG: how it was built

The classic RAG pipeline on the same PDF as the PageIndex notes ([`data/Attention.pdf`](../data/Attention.pdf), 15 pages), then wrapped in LangGraph.

Notebook: [`src/vectorless_vs_vectordb_rag/vectordb/vectordb.ipynb`](../src/vectorless_vs_vectordb_rag/vectordb/vectordb.ipynb)
Python modules: [`vectordb/ingest.py`](../src/vectorless_vs_vectordb_rag/vectordb/ingest.py) (indexing), [`vectordb/rag.py`](../src/vectorless_vs_vectordb_rag/vectordb/rag.py) (graphs)

---

## Pipeline

```
INDEXING (once)                                   QUERYING (every question)
PDF → pages → chunks → embed → Chroma DB          question → embed → top-5 chunks → Claude answers
```

| Step | Choice | Why |
|---|---|---|
| Read | `pypdf` 7, page by page | every chunk keeps its page number (citations + comparison with PageIndex) |
| Chunk | 120 words, 30 overlap, per page | split on words so no word is cut; overlap keeps boundary sentences whole |
| Embed | `nvidia/nemotron-3.5-embed-1b:free` via OpenRouter | free, 2048 numbers per vector, 32k-token input, vectors come normalized |
| Store | Chroma 1.5 `PersistentClient` → `results/chroma_db/` | on disk, stores text + `page` metadata, cosine distance |
| Answer | Claude Sonnet via `claude-agent-sdk` | runs on the Claude subscription, no API key |
| Orchestrate | LangGraph 1.2 | plain graph + agentic graph |

Numbers: **71 chunks**, embedded in **one** API request (100 texts per batch).

---

## Things that are easy to get wrong

| Trap | What we did |
|---|---|
| Chroma silently embeds queries with its own default model (MiniLM, 384 numbers), which clashes with our 2048-number vectors | attach our model to the collection: `embedding_function=OpenAIEmbeddingFunction(api_base=OpenRouter, ...)` |
| Old Chroma style `metadata={"hnsw:space": "cosine"}` | Chroma 1.x style `configuration={"hnsw": {"space": "cosine"}}` |
| Re-running `add` creates errors or duplicates | `upsert` (insert or update) |
| Passing vectors to `upsert` still re-embeds? | No: when `embeddings=` is given, Chroma uses them (verified: 0 extra API calls) |
| API returns vectors out of order | sort `response.data` by `index` |
| Chroma reports **distance**, not similarity | distance = 1 − cosine similarity (0.466 distance = 0.534 similarity) |
| The Claude SDK is async, Jupyter already runs an event loop | `ask_claude` runs `asyncio.run` in a worker thread |
| `ANTHROPIC_API_KEY` set in the environment bills the API | removed from `os.environ` before calling Claude Code |
| Your CLAUDE.md / memory leaking into the answering model | `setting_sources=[]`, `tools=[]` |
| Two threads opening the Chroma DB at the same time (web app) | open the collection once, behind a `threading.Lock` |

---

## The two LangGraph graphs

**Plain RAG**
```
START → retrieve → generate → END
```

**Agentic RAG**: Claude grades the chunks (structured JSON output: `relevant`, `better_query`) and can retry the search up to 2 times.
```
START → retrieve → grade ──relevant / out of retries──→ generate → END
           ↑          │
           └─ retry ──┘
```

LangGraph basics used:
- state = a `TypedDict`; each node returns only the keys it changes
- `add_conditional_edges("grade", route)`: the function's return value names the next node
- `graph.stream(..., stream_mode="updates")` shows each node's update as it runs
- `graph.get_graph().draw_mermaid()` prints the diagram (paste into mermaid.live)

Build and test `retrieve` / `generate` as plain functions first, then wire them into the graph: a bug is easier to find outside the graph.

---

## Sample results (from the notebook)

| Question | Pages retrieved | Result |
|---|---|---|
| What is multi-head attention? | 5, 5, 4 (top 3) | correct section 3.2.2 |
| How are positional encodings computed? | 6, 6, 3, 6, 5 | correct formulas, cites p. 6 |
| What optimizer and learning rate schedule? | 7, 8 | Adam (0.9, 0.98, 1e-9), warmup 4000 |

How it compares with PageIndex: [vectordb_vs_vectorless.md](vectordb_vs_vectorless.md)
