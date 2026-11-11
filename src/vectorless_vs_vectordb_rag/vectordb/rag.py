"""
FILE:     vectordb/rag.py
PURPOSE:  Answer questions with vector RAG (the QUERYING half). Finds the closest chunks in the
          Chroma DB and lets Claude answer from them, organised as two LangGraph graphs:

            plain:    START -> retrieve -> generate -> END
            agentic:  START -> retrieve -> grade -> (generate -> END | retry -> retrieve)

          The agentic graph lets Claude check the chunks first and search again with a better
          query if they do not contain the answer.

NEEDS:    the vector DB from vectordb/ingest.py (run `uv run vectorless-vs-vectordb-rag index` once)
USED BY:  compare/evaluate.py, app.py, the CLI `ask`
RUN:      uv run python -m vectorless_vs_vectordb_rag.vectordb.rag "How are positional encodings computed?"
"""
import json
import sys
import threading
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm import ask_claude
from .ingest import embed, get_collection

# Instructions for the answering model: stay grounded in the chunks and cite pages
ANSWER_SYSTEM = (
    "You answer questions about a research paper using ONLY the context given. "
    "Cite pages like [p. 4]. If the context does not contain the answer, say so. Be concise."
)

# The JSON shape Claude must return when grading the chunks (agentic graph)
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean", "description": "true if the context contains the answer"},
        "better_query": {"type": "string", "description": "a better search query if not relevant"},
    },
    "required": ["relevant", "better_query"],
}

# Query text -> vector, saved on disk: a question asked once never costs another API call,
# even after a restart. This matters because OpenRouter's free tier allows only 50 requests
# per day, and embedding requests count too.
_query_cache: dict[str, list[float]] = (
    json.loads(config.QUERY_CACHE_PATH.read_text(encoding="utf-8")) if config.QUERY_CACHE_PATH.exists() else {}
)
_collection = None                         # the Chroma collection, opened once
_lock = threading.Lock()                    # protects the one-time opening (see collection())


def collection():
    """Open the Chroma collection once and reuse it.
    The lock stops two threads (e.g. the website asking two systems in parallel) from opening
    the DB at the same moment, which Chroma does not allow."""
    global _collection
    with _lock:
        if _collection is None:
            _collection = get_collection()
    return _collection


# ---------- plain functions (tested on their own first, then used as graph nodes) ----------
def embed_queries(texts: list[str]) -> None:
    """Embed every query that is not cached yet, all in ONE API request."""
    todo = [t for t in dict.fromkeys(texts) if t not in _query_cache]   # dict.fromkeys = remove duplicates, keep order
    if todo:
        vectors = embed(todo)
        with _lock:                                    # several threads may add queries at once
            for text, vec in zip(todo, vectors):
                _query_cache[text] = vec.tolist()
            config.QUERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.QUERY_CACHE_PATH.write_text(json.dumps(_query_cache), encoding="utf-8")


def retrieve(text: str, k: int = config.TOP_K) -> list[dict]:
    """Return the k chunks whose vectors are closest to the question's vector."""
    embed_queries([text])
    res = collection().query(query_embeddings=[_query_cache[text]], n_results=k)
    # Chroma returns parallel lists (one list per query); [0] = our single query
    return [
        {"id": i, "page": m["page"], "text": d, "distance": round(dist, 3)}   # distance = 1 - cosine similarity
        for i, m, d, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]


def format_context(chunks: list[dict]) -> str:
    """Chunks as text for the prompt, each tagged with its page so Claude can cite it."""
    return "\n\n".join(f"[p. {c['page']}] {c['text']}" for c in chunks)


def generate(question: str, chunks: list[dict]) -> str:
    """Claude answers the question using only the retrieved chunks."""
    return ask_claude(f"Context:\n{format_context(chunks)}\n\nQuestion: {question}", system=ANSWER_SYSTEM)


# ---------- plain RAG graph: START -> retrieve -> generate -> END ----------
class RAGState(TypedDict, total=False):
    """The shared state passed between nodes. Each node returns only the keys it changes."""
    question: str
    chunks: list[dict]
    answer: str


@lru_cache(maxsize=1)               # build the graph once, reuse it for every question
def rag_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", lambda s: {"chunks": retrieve(s["question"])})
    g.add_node("generate", lambda s: {"answer": generate(s["question"], s["chunks"])})
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()              # compile() turns the description into a runnable graph


# ---------- agentic RAG graph: retrieve -> grade -> (generate | retry -> retrieve) ----------
class AgentState(TypedDict, total=False):
    question: str            # the user's question (never changes)
    search_query: str        # what goes to the vector DB (Claude may rewrite it)
    chunks: list[dict]       # the latest search results
    relevant: bool           # Claude's verdict on those chunks
    retries: int             # how many times we searched again
    grades: int              # number of grading calls (= LLM calls before generate)
    chars_read: int          # context characters Claude read while grading (cost measure)
    queries_tried: list[str] # every search query used, for display
    answer: str


def _retrieve(s: AgentState) -> dict:
    """Search with Claude's rewritten query if there is one, otherwise with the question."""
    q = s.get("search_query") or s["question"]
    return {"chunks": retrieve(q), "queries_tried": s.get("queries_tried", []) + [q]}


def _grade(s: AgentState) -> dict:
    """Claude checks whether the chunks answer the question and suggests a better query if not."""
    ctx = format_context(s["chunks"])
    verdict = ask_claude(
        f"Question: {s['question']}\n\nContext:\n{ctx}\n\nDoes this context contain the answer? "
        "If not, suggest a better search query using words likely to appear in the paper.",
        schema=GRADE_SCHEMA,                       # reply comes back as {"relevant": ..., "better_query": ...}
    )
    return {"relevant": verdict["relevant"], "search_query": verdict["better_query"],
            "grades": s.get("grades", 0) + 1, "chars_read": s.get("chars_read", 0) + len(ctx)}


def _route(s: AgentState) -> Literal["retry", "generate"]:
    """Decide the next node: answer if the chunks are good or we ran out of retries, else search again."""
    return "generate" if s["relevant"] or s.get("retries", 0) >= config.MAX_RETRIES else "retry"


@lru_cache(maxsize=1)
def agentic_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade", _grade)
    g.add_node("retry", lambda s: {"retries": s.get("retries", 0) + 1})   # count the attempt
    g.add_node("generate", lambda s: {"answer": generate(s["question"], s["chunks"])})
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route)      # _route's return value names the next node
    g.add_edge("retry", "retrieve")               # the loop: retry goes back to retrieve
    g.add_edge("generate", END)
    return g.compile()


# ---------- one function per system, same result shape for the comparison and the website ----------
def run_vector(question: str) -> dict:
    """Plain vector RAG: pages read, chunks, LLM calls, characters read, answer."""
    out = rag_graph().invoke({"question": question})        # invoke = run the graph to the end
    return {"pages": sorted({c["page"] for c in out["chunks"]}), "chunks": out["chunks"], "llm_calls": 1,
            "chars_read": len(format_context(out["chunks"])), "answer": out["answer"]}


def run_agentic(question: str) -> dict:
    """Agentic vector RAG: same fields, plus the search queries it tried."""
    out = agentic_graph().invoke({"question": question})
    return {"pages": sorted({c["page"] for c in out["chunks"]}), "chunks": out["chunks"],
            "llm_calls": out["grades"] + 1,                  # every grading call + the final answer
            "chars_read": out["chars_read"] + len(format_context(out["chunks"])),
            "answer": out["answer"], "queries_tried": out["queries_tried"]}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How are positional encodings computed?"
    r = run_vector(q)
    print("Pages:", r["pages"], "\n")
    print(r["answer"])
