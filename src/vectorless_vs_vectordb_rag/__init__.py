"""
FILE:     __init__.py  (package entry point)
PURPOSE:  Marks this folder as the `vectorless_vs_vectordb_rag` Python package and holds the
          command line tool `vectorless-vs-vectordb-rag` (registered in pyproject.toml).

COMMANDS:
    uv run vectorless-vs-vectordb-rag index              # build the Chroma vector DB from the PDF
    uv run vectorless-vs-vectordb-rag ask "question"     # ask all three RAG systems one question
    uv run vectorless-vs-vectordb-rag compare            # run the full 10-question evaluation
    uv run vectorless-vs-vectordb-rag app                # start the Streamlit website
"""
import sys


def main() -> None:
    """Read the command from the terminal and run the matching part of the project."""
    # sys.argv = ["vectorless-vs-vectordb-rag", "<command>", "<extra words>..."]
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("help", [])

    # Imports happen inside each branch, so `help` starts instantly
    # (loading Chroma, LangGraph and PageIndex takes a few seconds)
    if cmd == "index":
        from .vectordb.ingest import build_index
        print("Chunks in DB:", build_index())

    elif cmd == "ask":
        from .compare.evaluate import SYSTEMS
        question = " ".join(args) or "How are positional encodings computed?"
        for name, run in SYSTEMS.items():              # Vector RAG, Agentic vector RAG, PageIndex
            r = run(question)
            print(f"\n=== {name}  pages={r['pages']}\n{r['answer']}")

    elif cmd == "compare":
        from .compare.evaluate import main as compare_main
        compare_main()

    elif cmd == "app":
        import subprocess
        from pathlib import Path
        print("Website: http://localhost:8501  (Ctrl+C to stop)")
        # Start Streamlit on app.py (it sits next to this file)
        # --server.headless true  -> skip Streamlit's first-run email prompt, which blocks some terminals
        # --browser.gatherUsageStats false -> do not send usage statistics
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"])

    else:
        print(__doc__)                                 # unknown command -> show the help text above
