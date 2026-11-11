"""
FILE:     compare/evaluate.py
PURPOSE:  Compare the three RAG systems fairly: run every system on the same 10 questions,
          score which pages each one read against the known correct ("gold") pages, let Claude
          judge the answers, and save the tables and the summary chart.

          Steps:  run_all() -> judge_all() -> build_tables() -> save_chart()

OUTPUT:   results/compare/runs.json         every answer + pages + timing (also the cache)
          results/compare/per_question.csv  one row per (system, question)
          results/compare/summary.csv       averages per system
          results/compare/summary.png       the chart
USED BY:  app.py (questions, systems, colours, tables), the CLI `ask` and `compare`
RUN:      uv run vectorless-vs-vectordb-rag compare
          Finished runs are cached in runs.json, so a re-run only does the missing work.
          Delete runs.json to start over.
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd

from .. import config
from ..llm import ask_claude
from ..vectordb.ingest import load_pages
from ..vectordb.rag import embed_queries, run_agentic, run_vector
from ..vectorless.pageindex_rag import run_pageindex

# Gold pages = the PDF pages that actually contain the answer (checked by searching the page text)
QUESTIONS = [
    {"id": "q1",  "question": "How does multi-head attention work?",                                                      "gold": [4, 5]},
    {"id": "q2",  "question": "What is the formula for scaled dot-product attention, and why are the dot products scaled?", "gold": [4]},
    {"id": "q3",  "question": "How are positional encodings computed?",                                                  "gold": [6]},
    {"id": "q4",  "question": "What optimizer and learning rate schedule were used for training?",                      "gold": [7]},
    {"id": "q5",  "question": "Which regularization techniques were used during training?",                             "gold": [7, 8]},
    {"id": "q6",  "question": "What hardware was used to train the models and how long did training take?",             "gold": [7]},
    {"id": "q7",  "question": "What BLEU scores did the big Transformer get on WMT 2014 English-German and English-French?", "gold": [1, 8]},
    {"id": "q8",  "question": "How well does the Transformer do on English constituency parsing?",                      "gold": [9, 10]},
    {"id": "q9",  "question": "How does changing the number of attention heads affect translation quality?",            "gold": [8, 9]},
    {"id": "q10", "question": "Why use self-attention instead of recurrent or convolutional layers?",                   "gold": [6, 7]},
]

# The three systems: name -> function(question) that returns {"pages", "answer", "llm_calls", "chars_read", ...}
SYSTEMS: dict[str, Callable[[str], dict]] = {
    "Vector RAG": run_vector,
    "Agentic vector RAG": run_agentic,
    "PageIndex (local)": run_pageindex,
}
# One fixed colour per system, used in every chart (blue, orange, aqua: a colour-blind-safe set)
COLORS = {"Vector RAG": "#2a78d6", "Agentic vector RAG": "#eb6834", "PageIndex (local)": "#1baf7a"}

RUNS_FILE = config.COMPARE_DIR / "runs.json"

# Judge instructions + the JSON shape of its reply: one score (1-5) and reason per answer
JUDGE_SYSTEM = ("You grade answers to a question about a research paper. Use the reference pages as ground truth. "
                "5 = fully correct and complete, 4 = correct with small gaps, 3 = partly correct, "
                "2 = mostly wrong or missing, 1 = wrong or no answer. Ignore style and length.")
JUDGE_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5},
                   "reason": {"type": "string"}},
    "required": ["label", "score", "reason"]}}}, "required": ["scores"]}


# ---------- cache file: {"<system>|<question id>": {...result...}} ----------
def load_runs() -> dict:
    """Read finished runs from disk (empty dict on the first run)."""
    return json.loads(RUNS_FILE.read_text(encoding="utf-8")) if RUNS_FILE.exists() else {}


def save_runs(runs: dict) -> None:
    """Write all runs to disk (called after every finished run, so a crash loses nothing)."""
    config.COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- step 1: run the systems ----------
def run_all(runs: dict, workers: int = 4) -> dict:
    """Run every (system, question) pair that is not cached yet, in parallel."""
    embed_queries([q["question"] for q in QUESTIONS])          # all question vectors in one request
    todo = [(s, q) for s in SYSTEMS for q in QUESTIONS if f"{s}|{q['id']}" not in runs]

    def job(item):
        s, q = item
        start = time.perf_counter()
        try:
            res = SYSTEMS[s](q["question"])
        except Exception as e:                                  # one failure must not stop the batch
            return f"{s}|{q['id']}", {"error": f"{type(e).__name__}: {e}"}
        res.pop("chunks", None)                                 # keep the cache file small
        res["latency_s"] = round(time.perf_counter() - start, 1)
        return f"{s}|{q['id']}", res

    # Run `workers` jobs at the same time: each one mostly waits for Claude, so threads speed it up
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, res in pool.map(job, todo):
            if "error" in res:
                print("FAILED", key, res["error"][:200])
                continue
            runs[key] = res
            save_runs(runs)
            print(f"done {key:28s} pages={res['pages']}  {res['latency_s']}s")
    return runs


# ---------- step 2: judge the answers ----------
def judge_all(runs: dict, workers: int = 4) -> dict:
    """LLM-as-judge: per question, the 3 answers shuffled and unlabeled, scored 1-5 against the gold pages."""
    page_text = {p["page"]: p["text"] for p in load_pages()}   # full text of every page = the ground truth

    def judge(q: dict) -> None:
        keys = [f"{s}|{q['id']}" for s in SYSTEMS]
        # skip if a system's run is missing, or if this question was already judged
        if not all(k in runs for k in keys) or all("score" in runs[k] for k in keys):
            return
        order = keys[:]
        random.Random(q["id"]).shuffle(order)                     # shuffle so the judge can't favour a position
        labels = {k: chr(65 + i) for i, k in enumerate(order)}    # A, B, C instead of system names (blind)
        reference = "\n\n".join(f"[page {p}]\n{page_text[p]}" for p in q["gold"])
        answers = "\n\n".join(f"### Answer {labels[k]}\n{runs[k]['answer']}" for k in order)
        verdict = ask_claude(f"Reference pages:\n{reference}\n\nQuestion: {q['question']}\n\n{answers}",
                             system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA)
        # map the judge's labels back to systems ("Answer B" or "B" -> "B")
        by_label = {s["label"].strip().upper()[-1]: s for s in verdict["scores"]}
        for k in keys:
            runs[k]["score"] = by_label[labels[k]]["score"]
            runs[k]["judge_reason"] = by_label[labels[k]]["reason"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge, QUESTIONS))
    save_runs(runs)
    return runs


# ---------- step 3: score retrieval and build tables ----------
def build_tables(runs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-question rows and per-system averages.

    hit       = read at least one correct page (1 or 0)
    recall    = share of the correct pages that were read
    precision = share of the pages read that were correct (low = lots of noise)
    """
    rows = []
    for q in QUESTIONS:
        gold = set(q["gold"])
        for s in SYSTEMS:
            r = runs.get(f"{s}|{q['id']}")
            if not r:
                continue
            got = set(r["pages"])                     # pages the system read; got & gold = the correct ones
            rows.append({
                "system": s, "id": q["id"], "question": q["question"], "gold": sorted(gold), "pages": r["pages"],
                "hit": int(bool(got & gold)), "recall": len(got & gold) / len(gold),
                "precision": len(got & gold) / len(got) if got else 0.0,
                "pages_read": len(got), "chars_read": r["chars_read"], "latency_s": r["latency_s"],
                "llm_calls": r["llm_calls"], "score": r.get("score"), "answer": r["answer"],
            })
    df = pd.DataFrame(rows)
    summary = (df.groupby("system", sort=False)
                 .agg(hit_rate=("hit", "mean"), recall=("recall", "mean"), precision=("precision", "mean"),
                      pages_read=("pages_read", "mean"), chars_read=("chars_read", "mean"),
                      latency_s=("latency_s", "mean"), llm_calls=("llm_calls", "mean"), score=("score", "mean"))
                 .reindex(list(SYSTEMS)).round(2))
    return df, summary


# ---------- step 4: chart ----------
def save_chart(summary: pd.DataFrame) -> None:
    """Small multiples: one metric per panel (no dual axes), colour fixed per system."""
    import matplotlib
    matplotlib.use("Agg")                          # no window, just a PNG file
    import matplotlib.pyplot as plt

    panels = [("hit_rate", "Hit rate", "{:.0%}", (0, 1.05)), ("recall", "Recall", "{:.0%}", (0, 1.05)),
              ("precision", "Precision", "{:.0%}", (0, 1.05)), ("score", "Answer score (1-5)", "{:.2f}", (0, 5.3)),
              ("pages_read", "Pages read", "{:.1f}", None), ("chars_read", "Chars read by LLM", "{:,.0f}", None),
              ("latency_s", "Seconds / question", "{:.1f}", None), ("llm_calls", "LLM calls", "{:.1f}", None)]
    # each panel: (column, title, number format, fixed y range or None = fit to the data)
    names = list(SYSTEMS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), facecolor="#fcfcfb")
    for ax, (col, title, fmt, ylim) in zip(axes.flat, panels):
        vals = summary[col]
        bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names], width=0.6,
                      edgecolor="#fcfcfb", linewidth=2)
        for b, v in zip(bars, vals):                     # write the value on top of each bar
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(range(len(names)), [n.replace(" (local)", "\n(local)").replace("Agentic ", "Agentic\n") for n in names])
        ax.set_ylim(*(ylim or (0, vals.max() * 1.18)))
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.COMPARE_DIR / "summary.png", dpi=150)


def main() -> None:
    """Full evaluation: run (missing runs only) -> judge -> tables -> chart -> print the summary."""
    runs = judge_all(run_all(load_runs()))
    df, summary = build_tables(runs)
    df.drop(columns="answer").to_csv(config.COMPARE_DIR / "per_question.csv", index=False)
    summary.to_csv(config.COMPARE_DIR / "summary.csv")
    save_chart(summary)
    print(summary.to_string())


if __name__ == "__main__":
    main()
