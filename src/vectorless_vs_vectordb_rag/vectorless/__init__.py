"""
FOLDER:   vectorless/
PURPOSE:  The Vectorless RAG side of the project (PageIndex): no embeddings, no vector DB.
          The LLM reads a table-of-contents tree of the document and decides which pages to open.
          pageindex.ipynb  -> builds and explores the trees (cloud and local)
          pageindex_rag.py -> asks questions on the local tree, reporting which pages were opened
"""
