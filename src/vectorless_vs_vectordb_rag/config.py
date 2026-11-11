"""
FILE:     config.py
PURPOSE:  One place for every setting of the project: file paths, model names and RAG knobs.
          Change a value here and every module (notebook-style code, CLI, website) picks it up.

USED BY:  llm.py, vectordb/ingest.py, vectordb/rag.py, vectorless/pageindex_rag.py,
          compare/evaluate.py, app.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- project root and API keys ----------
# This file is src/vectorless_vs_vectordb_rag/config.py, so parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load OPENROUTER_API_KEY (and PAGEINDEX_API_KEY if present) from the .env file
load_dotenv(PROJECT_ROOT / ".env")

# Claude Code bills an Anthropic API key if one is set; remove it so the Claude subscription is used
os.environ.pop("ANTHROPIC_API_KEY", None)

# ---------- files and folders ----------
PDF_PATH = PROJECT_ROOT / "data" / "Attention.pdf"                  # the document we ask questions about
CHROMA_PATH = PROJECT_ROOT / "results" / "chroma_db"                # the vector DB (generated)
PAGEINDEX_LOCAL_PATH = PROJECT_ROOT / "results" / "pageindex_local" # PageIndex local tree (generated)
COMPARE_DIR = PROJECT_ROOT / "results" / "compare"                  # evaluation results + charts
QUERY_CACHE_PATH = PROJECT_ROOT / "results" / "query_embeddings.json" # saved question vectors (saves API calls)

# ---------- models ----------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"   # OpenRouter speaks the OpenAI API at this address
EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"      # embeddings: free, 2048 numbers per vector
CHAT_MODEL = "sonnet"                                  # answers: Claude Sonnet via claude-agent-sdk

# ---------- vector RAG settings ----------
COLLECTION_NAME = "attention_paper"   # name of the Chroma collection (like a table name)
CHUNK_SIZE = 120                      # words per chunk
OVERLAP = 30                          # words shared by neighbouring chunks
TOP_K = 5                             # chunks retrieved per question
MAX_RETRIES = 2                       # agentic RAG: how many times Claude may rewrite the search query

# ---------- PageIndex ----------
PAGEINDEX_DOC_ID = "pi-674f81e1daac4d75a9e3c7aed9d2949f"   # local tree built in vectorless/pageindex.ipynb
