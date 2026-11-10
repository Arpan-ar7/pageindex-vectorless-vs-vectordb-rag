# PageIndex: Cloud vs Local

The same PDF ([`data/Attention.pdf`](../data/Attention.pdf), "Attention Is All You Need", 15 pages) was indexed twice with the PageIndex SDK: once in the **cloud** and once **locally**. This note covers what happened and how the two results differ.

Notebook: [`src/vectorless_vs_vectordb_rag/vectorless/pageindex.ipynb`](../src/vectorless_vs_vectordb_rag/vectorless/pageindex.ipynb)

---

## Files

| What | Cloud | Local |
|---|---|---|
| Tree | [`results/attention_tree.json`](../results/attention_tree.json) (a copy we saved; the original is on PageIndex's servers) | [`tree.json`](../results/pageindex_local/docs/pi-674f81e1daac4d75a9e3c7aed9d2949f/tree.json) |
| Page text | On PageIndex's servers | [`pages.json`](../results/pageindex_local/docs/pi-674f81e1daac4d75a9e3c7aed9d2949f/pages.json) |
| Document info | On PageIndex's servers | [`doc.json`](../results/pageindex_local/docs/pi-674f81e1daac4d75a9e3c7aed9d2949f/doc.json) |
| List of indexed docs | PageIndex dashboard | [`manifest.json`](../results/pageindex_local/manifest.json) |
| Folder | — | [`results/pageindex_local/`](../results/pageindex_local/) |
| `doc_id` | `pi-cmwaj1rxf002k0cnvz5fmrjsl` | `pi-674f81e1daac4d75a9e3c7aed9d2949f` |

---

## Setup difference (one line)

```python
# Cloud: PageIndex servers build the tree (uses PAGEINDEX_API_KEY)
PageIndexClient(index="cloud", chat="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")

# Local: our own LLM builds the tree on our machine (uses OPENROUTER_API_KEY only)
PageIndexClient(
    index={"model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
           "storage_path": "../../../results/pageindex_local"},
    chat="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
)
```

Questions are answered the same way in both: `client.chat()` uses our OpenRouter model either way.

---

## Overall comparison

| | Cloud | Local |
|---|---|---|
| Who builds the tree | PageIndex servers | Our machine + our LLM (Nemotron via OpenRouter) |
| Keys used | `PAGEINDEX_API_KEY` (index) + `OPENROUTER_API_KEY` (chat) | `OPENROUTER_API_KEY` only |
| How it reads the PDF | OCR + layout/image understanding | PDF text layer only (`flash` mode, the default) |
| Where the tree lives | PageIndex servers (fetch with `get_tree`) | On disk: `tree.json` |
| Cost | PageIndex credits per upload | OpenRouter calls (free tier for us) |
| Nodes in tree | 25 | 14 |
| Max depth | 4 levels | 2 levels |
| Section numbers | Kept (3.1, 3.2.2 …) | Lost |
| Quality | Matches the paper's real structure | Some sections misplaced (see below) |

---

## The two trees side by side

**Cloud** (correct structure):
```
[0000] Attention Is All You Need (p1-15)
   [0001] Attention Is All You Need (intro) (p1-1)
   [0002] Abstract (p1-2)
   [0003] 1 Introduction (p2-2)
   [0004] 2 Background (p2-2)
   [0005] 3 Model Architecture (p2-6)
      [0006] 3 Model Architecture (intro) (p2-3)
      [0007] 3.1 Encoder and Decoder Stacks (p3-3)
      [0008] 3.2 Attention (p3-5)
         [0009] 3.2 Attention (intro) (p3-4)
         [0010] 3.2.1 Scaled Dot-Product Attention (p4-4)
         [0011] 3.2.2 Multi-Head Attention (p4-5)
         [0012] 3.2.3 Applications of Attention in our Model (p5-5)
      [0013] 3.3 Position-wise Feed-Forward Networks (p5-5)
      [0014] 3.4 Embeddings and Softmax (p5-6)
      [0015] 3.5 Positional Encoding (p6-6)
   [0016] 4 Why Self-Attention (p6-7)
   [0017] 5 Training (p7-8)
   [0018] 6 Results (p8-10)
      [0019] 6.1 Machine Translation (p8-8)
      [0020] 6.2 Model Variations (p8-9)
      [0021] 6.3 English Constituency Parsing (p9-10)
   [0022] 7 Conclusion (p10-10)
   [0023] References (p10-13)
   [0024] Attention Visualizations (p13-15)
```

**Local** (flatter, with mistakes marked ⚠️):
```
[0000] Abstract (p1-2)
[0001] Introducing the Transformer Model Architecture (p2-2)
[0002] Model Architecture (p2-6)
   [0003] 1 Introduction (p2-2)          ⚠️ should be top level, not inside Model Architecture
   [0004] 2 Background (p2-3)            ⚠️ same
   [0005] Encoder and Decoder Stacks (p3-3)
   [0006] Attention (p3-5)               ⚠️ no children: Scaled Dot-Product / Multi-Head merged in
   [0007] Position-wise Feed-Forward Networks (p5-5)
   [0008] Embeddings and Softmax (p5-6)
   [0009] Positional Encoding (p6-6)
[0010] Why Self-Attention (p6-7)
[0011] Training (p7-8)
[0012] Results (p8-10)                   ⚠️ no sub-sections 6.1 / 6.2 / 6.3
[0013] Conclusion (p10-15)               ⚠️ also swallows References + Attention Visualizations
```

### Why the local tree is worse
- **Local `flash` mode only reads the PDF's text.** It has no layout analysis or OCR, so the LLM has to guess which lines are headings and how deep they go.
- **The cloud uses OCR and layout understanding.** It sees font sizes and numbering like `3.2.2`, so it recovers the real hierarchy.

### Why this matters for retrieval
The model picks sections **from the tree**, so the tree's quality limits retrieval quality:
- *"How does multi-head attention work?"* Cloud can go straight to `3.2.2 Multi-Head Attention (p4-5)`. Local can only pick the broader `Attention (p3-5)` and read more pages.
- *"Which papers are cited?"* Cloud has a `References` node. Local hides references inside `Conclusion`, so the model has to guess to look there.

**Lesson:** in vectorless RAG, **building a good tree is most of the work.**

---

## What happened along the way (problems and fixes)

| Problem | Cause | Fix |
|---|---|---|
| `.env` keys showed `MISSING` | `.env` was empty (not saved) | Saved the file (Ctrl+S) |
| `index= and ... (api_key) are two spellings` | Passed `api_key=` and `index="cloud"` together | Remove `api_key=`: `index="cloud"` reads `PAGEINDEX_API_KEY` from the environment itself |
| `FileNotFoundError: data/...pdf` | A notebook runs from its own folder, not the project root | Use `../../../data/...` |
| `SSLError ... UNEXPECTED_EOF` | Unstable connection to `api.pageindex.ai` during a large upload (6 MB PDF) | Switched to the smaller `Attention.pdf`; retried |
| PDF uploaded twice (`Attention_1.pdf`) | Re-ran the upload cell; the SDK does not check for duplicates | Hard-coded the `doc_id` and commented out the upload line |
| Local `SyntaxError` | Extra `(` in `submit_document(("...", wait=True)["doc_id"])` | Hard-coded `local_doc_id` (already indexed); fixed line kept as a comment |
| Notebook edits disappeared | VS Code saved its open (old) copy over the file | Use **Revert File** before running/saving |

---

## Possible improvement to try
Local indexing has a second mode: `mode="standard"` builds the full tree with the LLM instead of `flash`. It may recover the sub-sections better, but it makes more LLM calls:
```python
local_client.submit_document("../../../data/Attention.pdf", mode="standard", wait=True)
```

## Next
Part 3: build vectorless RAG **from scratch** (our own tree builder → tree search → answer), then compare it with a vector-DB RAG.
