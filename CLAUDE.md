# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is
A learning project comparing two RAG approaches on the same PDF (`data/Attention.pdf`, "Attention Is All You Need", 15 pages):
- **Vectorless RAG**: PageIndex builds a table-of-contents tree; the LLM reasons over it and opens pages (`src/vectorless_vs_vectordb_rag/vectorless/`).
- **VectorDB RAG**: chunks → Nemotron embeddings (OpenRouter) → Chroma → LangGraph graphs (plain + agentic) → Claude answers (`src/vectorless_vs_vectordb_rag/vectordb/`).
- **Comparison**: 10 questions with known gold pages, 3 systems, judged by Claude (`src/vectorless_vs_vectordb_rag/compare/`).
- **Website**: Streamlit app (`src/vectorless_vs_vectordb_rag/app.py`).

Results and write-ups: `notes/summary.md`, `notes/vectordb_vs_vectorless.md`.

## How the user wants to work
- **Teach, don't code by default.** Explain concepts and steps, show small snippets in chat. Only create or edit files (including notebook cells) when the user explicitly asks.
- **Always use the latest APIs** of the installed versions (check `uv.lock` / `.venv` before writing code). Never suggest deprecated patterns (e.g. `PyPDF2`, Chroma `metadata={"hnsw:space": ...}`, Streamlit `use_container_width`).
- Every file starts with a header saying what it is for; code is commented step by step.
- Keep explanations simple and practical.

## Commands
```bash
uv sync                                                   # install everything (Python 3.14)
uv run vectorless-vs-vectordb-rag index                   # build the Chroma DB -> results/chroma_db/
uv run vectorless-vs-vectordb-rag ask "question"          # ask all three systems
uv run vectorless-vs-vectordb-rag compare                 # full evaluation (cached in results/compare/runs.json)
uv run vectorless-vs-vectordb-rag app                     # website at http://localhost:8501
uv run jupyter nbconvert --to notebook --execute --inplace <notebook>   # run a notebook headless
```

## Setup facts
- `.env` holds `OPENROUTER_API_KEY` (embeddings) and optionally `PAGEINDEX_API_KEY` (cloud mode; currently commented out by the user, so cloud is not used). Never print or commit keys.
- **Claude answers run on the user's Claude subscription** via `claude-agent-sdk` (`llm.ask_claude`), not an API key. `config.py` removes `ANTHROPIC_API_KEY` from the environment so the subscription is used. PageIndex uses `claude-code/sonnet` the same way.
- **OpenRouter free tier: 50 requests/day, embeddings included.** Batch embeddings (all 71 chunks fit in one request) and reuse cached query vectors (`results/query_embeddings.json`). Resets at 00:00 UTC.

## Code map
| File | Role |
|---|---|
| `config.py` | all paths, model names, chunk size, top-k |
| `llm.py` | `ask_claude(prompt, system, schema)`: text or structured JSON |
| `vectordb/ingest.py` | PDF → chunks (120 words, 30 overlap, per page) → embeddings → Chroma |
| `vectordb/rag.py` | `retrieve`, `generate`, `rag_graph()`, `agentic_graph()`, `run_vector`, `run_agentic` |
| `vectorless/pageindex_rag.py` | `run_pageindex`: reads `.events` to record which pages were opened |
| `compare/evaluate.py` | `QUESTIONS` (with gold pages), `SYSTEMS`, run → judge → tables → chart |
| `app.py` | Streamlit: Ask / Comparison results / How it works |
| `*.ipynb` | learning notebooks; `vectordb.ipynb` and `compare.ipynb` mirror the `.py` modules |

## Gotchas
- Notebooks run from their own folder: use `PROJECT_ROOT = Path.cwd().parents[2]`.
- Attach the embedding function to the Chroma collection, or Chroma falls back to its 384-dim default model (ours is 2048-dim).
- Open the Chroma collection / PageIndex client once behind a lock: parallel threads opening Chroma at the same time crash.
- `claude-agent-sdk` is async: `ask_claude` runs it in a worker thread so it works inside Jupyter and Streamlit.
- Use `tools=[]` and `setting_sources=[]` for answering calls so this file and user settings do not leak into answers.
- `results/chroma_db/`, `results/pageindex_local/` and `results/query_embeddings.json` are generated and git-ignored.
- `vectordb/chatpdf.ipynb` is the user's own practice notebook: do not edit it.
