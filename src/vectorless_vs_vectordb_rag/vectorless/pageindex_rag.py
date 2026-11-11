"""
FILE:     vectorless/pageindex_rag.py
PURPOSE:  Answer questions with vectorless RAG (PageIndex, local mode). Claude reads the document
          tree built in pageindex.ipynb, opens the pages it thinks are relevant, and answers.
          It also records WHICH pages Claude opened, so the comparison can score retrieval.

NEEDS:    the local tree in results/pageindex_local/ (built once in vectorless/pageindex.ipynb)
USED BY:  compare/evaluate.py, app.py, the CLI `ask`
RUN:      uv run python -m vectorless_vs_vectordb_rag.vectorless.pageindex_rag "How are positional encodings computed?"
"""
import sys
import threading

from pageindex import PageIndexClient

from .. import config

_client = None                  # the PageIndex client, created once
_lock = threading.Lock()        # protects the one-time creation when several threads ask at once


def client() -> PageIndexClient:
    """Create the PageIndex client once (thread-safe) and reuse it.
    "claude-code/sonnet" = Claude Sonnet through Claude Code on your subscription, the same answering
    model the vector side uses, so the comparison is fair."""
    global _client
    with _lock:
        if _client is None:
            _client = PageIndexClient(
                index={"model": "claude-code/sonnet",                      # local mode: tree stored on disk
                       "storage_path": str(config.PAGEINDEX_LOCAL_PATH)},
                chat="claude-code/sonnet",                                 # the model that answers
            )
    return _client


def parse_pages(spec) -> set[int]:
    """Turn PageIndex's page argument into numbers: '4-5' -> {4, 5};  '3,6' -> {3, 6}."""
    pages: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:                                   # a range like "4-5"
            lo, hi = part.split("-", 1)
            pages.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():                              # a single page like "6"
            pages.add(int(part))
    return pages


def run_pageindex(question: str) -> dict:
    """Ask PageIndex one question. Returns the answer plus what it read: pages, characters, tool calls.

    `.events` streams the run as typed dicts, e.g.
        {"type": "tool_call", "name": "get_page_content", "arguments": {"pages": "4-5"}}
        {"type": "tool_result", "output": "...page text..."}
        {"type": "answer", "delta": "...next piece of the answer..."}
    """
    pages, chars, tool_calls, answer, steps = set(), 0, 0, [], []
    for ev in client().chat(question, doc_id=config.PAGEINDEX_DOC_ID, stream=True).events:
        if ev["type"] == "tool_call":                     # Claude asked for the tree or for pages
            tool_calls += 1
            steps.append(f"{ev['name']} {ev['arguments']}")
            if ev["name"] == "get_page_content":
                pages |= parse_pages(ev["arguments"].get("pages", ""))
        elif ev["type"] == "tool_result":                 # what came back: everything the model had to read
            chars += len(str(ev["output"]))
        elif ev["type"] == "answer":                      # the answer arrives in small pieces
            answer.append(ev["delta"])
    return {"pages": sorted(pages), "chars_read": chars,
            "llm_calls": tool_calls + 1,                  # one model turn per tool call + the final answer
            "answer": "".join(answer), "steps": steps}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How are positional encodings computed?"
    r = run_pageindex(q)
    print("Steps:", *r["steps"], sep="\n  ")
    print("Pages:", r["pages"], "\n")
    print(r["answer"])
