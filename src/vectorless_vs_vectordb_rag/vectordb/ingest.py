"""
FILE:     vectordb/ingest.py
PURPOSE:  Build the vector database (the INDEXING half of vector RAG). Run it once; the DB is
          saved on disk at results/chroma_db/ and reused by every question afterwards.

          PDF -> pages -> chunks -> embeddings (Nemotron via OpenRouter) -> Chroma DB

USED BY:  vectordb/rag.py (embed + get_collection), compare/evaluate.py (load_pages), the CLI `index`
RUN:      uv run vectorless-vs-vectordb-rag index
     or:  uv run python -m vectorless_vs_vectordb_rag.vectordb.ingest
"""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from pypdf import PdfReader

from .. import config


# ---------- step 1: read the PDF ----------
def load_pages(pdf_path: Path = config.PDF_PATH) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...]. Pages count from 1, like PageIndex does."""
    with PdfReader(pdf_path) as reader:                    # "with" closes the file when done
        return [{"page": i, "text": p.extract_text()} for i, p in enumerate(reader.pages, start=1)]


# ---------- step 2: cut pages into chunks ----------
def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.OVERLAP) -> list[str]:
    """Overlapping windows of `size` words. Splitting on words never cuts a word in half."""
    words = text.split()            # also cleans the messy line breaks and spaces from the PDF
    step = size - overlap           # 120 - 30 = move forward 90 words each time
    # max(..., 1) stops a tiny last chunk that only repeats the overlap, and keeps short pages
    return [" ".join(words[s:s + size]) for s in range(0, max(len(words) - overlap, 1), step)]


def build_chunks(pages: list[dict]) -> list[dict]:
    """Chunk page by page, so a chunk never spans two pages and its page number is always right."""
    return [
        {"id": f"p{p['page']}-c{j}",    # unique id, e.g. "p3-c1" = page 3, chunk 1 (Chroma needs ids)
         "page": p["page"],              # kept for citations and for the comparison with PageIndex
         "text": t}
        for p in pages
        for j, t in enumerate(chunk_text(p["text"]))
    ]


# ---------- step 3: turn text into vectors ----------
@lru_cache(maxsize=1)               # create the client once and reuse it
def openrouter() -> OpenAI:
    """OpenRouter speaks the OpenAI API, so the official openai SDK works with a different base_url."""
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def embed(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Return one vector per text (shape: number of texts x 2048), in the same order as `texts`."""
    vectors = []
    for i in range(0, len(texts), batch_size):         # many texts per request (all 71 chunks fit in one)
        resp = openrouter().embeddings.create(model=config.EMBED_MODEL, input=texts[i:i + batch_size])
        # sort by index: guards against the API returning vectors out of order
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return np.array(vectors, dtype=np.float32)


# ---------- step 4: store in the vector DB ----------
def get_collection() -> chromadb.Collection:
    """Open the Chroma collection (or create it the first time), with our embedding model attached.

    Attaching our model matters: without it Chroma silently uses its own default model
    (384 numbers per vector), which would clash with Nemotron's 2048-number vectors.
    """
    embed_fn = OpenAIEmbeddingFunction(            # Chroma's OpenAI helper, pointed at OpenRouter
        api_key_env_var="OPENROUTER_API_KEY",
        model_name=config.EMBED_MODEL,
        api_base=config.OPENROUTER_BASE_URL,
    )
    db = chromadb.PersistentClient(path=config.CHROMA_PATH)   # Persistent = saved on disk, not just RAM
    return db.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        configuration={"hnsw": {"space": "cosine"}},          # similarity measure: cosine (Chroma 1.x style)
    )


def build_index() -> int:
    """Run steps 1-4 and return how many chunks are in the DB.
    Safe to re-run: upsert = insert or update, so chunks with the same id are replaced, not duplicated."""
    chunks = build_chunks(load_pages())
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),   # we pass our own vectors, so Chroma makes no extra API call
        documents=[c["text"] for c in chunks],           # the text, returned with search results
        metadatas=[{"page": c["page"]} for c in chunks], # the page number of each chunk
    )
    return collection.count()


if __name__ == "__main__":
    print("Chunks in DB:", build_index(), "->", config.CHROMA_PATH)
