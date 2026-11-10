# PageIndex: Vectorless vs VectorDB RAG

Comparing two ways to do RAG (Retrieval-Augmented Generation) on the same documents:

- **Vectorless RAG**: the document becomes a tree (like a table of contents), and the LLM *reasons* over the tree to pick which pages to read. No embeddings, no vector database. Built with [PageIndex](https://github.com/VectifyAI/PageIndex).
- **VectorDB RAG**: the classic approach. Split into chunks, embed them, and retrieve the most similar chunks.

## Progress

- [x] Vectorless RAG with the PageIndex SDK, **cloud** mode
- [x] Vectorless RAG with the PageIndex SDK, **local** mode
- [ ] Vectorless RAG from scratch
- [ ] VectorDB RAG
- [ ] Comparison on the same questions (retrieval accuracy, answer quality, cost, speed)

Notes so far: [PageIndex cloud vs local](notes/pageindex_cloud_vs_local.md)

## Project structure

```
src/vectorless_vs_vectordb_rag/
├── vectorless/pageindex.ipynb   # PageIndex SDK: cloud + local
├── vectordb/                    # vector DB RAG (coming)
└── compare/                     # evaluation (coming)
data/                            # input PDFs
results/                         # generated trees and outputs
notes/                           # write-ups
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
PAGEINDEX_API_KEY=...        # only needed for cloud mode
```

Then open `src/vectorless_vs_vectordb_rag/vectorless/pageindex.ipynb` and select the `.venv` kernel.

Test document: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (`data/Attention.pdf`).
