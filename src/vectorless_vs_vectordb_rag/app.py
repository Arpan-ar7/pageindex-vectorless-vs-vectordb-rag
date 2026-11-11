"""
FILE:     app.py
PURPOSE:  The Streamlit website. Three tabs:
            💬 Ask                   -> ask all three RAG systems the same question side by side,
                                        with the pages each one read (green = a correct page)
            📊 Comparison results    -> charts and tables from results/compare/runs.json
            🧠 How each system works -> short explanation of each system

NEEDS:    the vector DB (`index`), the PageIndex local tree, and Claude Code logged in
RUN:      uv run vectorless-vs-vectordb-rag app        then open http://localhost:8501
     or:  uv run streamlit run src/vectorless_vs_vectordb_rag/app.py

NOTE:     Streamlit re-runs this whole file from top to bottom on every click.
          That is why slow work sits in @st.cache_data functions: results are remembered.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import altair as alt
import pandas as pd
import streamlit as st

# Streamlit runs this file as a script, so use absolute imports (the package is installed by uv)
from vectorless_vs_vectordb_rag import config
from vectorless_vs_vectordb_rag.compare.evaluate import COLORS, QUESTIONS, SYSTEMS, build_tables, load_runs

# Browser tab title and icon; "wide" uses the full screen width (room for 3 answer columns)
st.set_page_config(page_title="Vectorless vs VectorDB RAG", page_icon="📄", layout="wide")


# ---------- cached work ----------
@st.cache_data(show_spinner=False, max_entries=200)
def ask(system: str, question: str) -> dict:
    """Run one system on one question. Cached: the same question never spends a second call."""
    start = time.perf_counter()
    res = SYSTEMS[system](question)
    res["latency_s"] = round(time.perf_counter() - start, 1)
    return res


@st.cache_data(show_spinner=False)
def comparison_tables(mtime: float):
    """Re-read runs.json only when the file changes (mtime is the cache key)."""
    return build_tables(load_runs())


def pages_badges(pages: list[int], gold: set[int] | None = None) -> str:
    """Pages as coloured badges: green = a correct page, grey = other (when the gold pages are known)."""
    if not pages:
        return "_no pages_"
    # Streamlit markdown badge syntax: :green-badge[text]
    return " ".join(f":{'green' if gold and p in gold else 'gray'}-badge[p. {p}]" for p in pages)


# ---------- sidebar: choose systems + show the setup ----------
with st.sidebar:
    st.title("📄 RAG face-off")
    st.caption("“Attention Is All You Need” · 15 pages")
    chosen = st.multiselect("Systems", list(SYSTEMS), default=list(SYSTEMS))
    st.divider()
    st.markdown(
        f"""
- **Embeddings:** `{config.EMBED_MODEL}`
- **Answers:** Claude `{config.CHAT_MODEL}` (subscription)
- **Vector DB:** Chroma · {config.CHUNK_SIZE}-word chunks · top {config.TOP_K}
- **Vectorless:** PageIndex local tree
"""
    )

ask_tab, results_tab, how_tab = st.tabs(["💬 Ask", "📊 Comparison results", "🧠 How each system works"])

# ---------- tab 1: Ask ----------
with ask_tab:
    # The 10 evaluation questions are offered as examples, because their correct pages are known
    examples = {q["question"]: q for q in QUESTIONS}
    pick = st.selectbox("Pick an example question (gold pages known) or type your own below",
                        ["—"] + list(examples))
    question = st.text_input("Question", value="" if pick == "—" else pick,
                             placeholder="e.g. How are positional encodings computed?")
    gold = set(examples[question]["gold"]) if question in examples else None
    if gold:
        st.caption(f"Correct pages for this question: {pages_badges(sorted(gold), gold)}")

    if st.button("Ask", type="primary", disabled=not (question and chosen)):
        cols = st.columns(len(chosen))            # one column per chosen system
        # Ask all systems at the same time (each mostly waits for Claude), and catch errors per system
        # so one failing system still shows the others' answers
        with st.spinner("Asking all systems in parallel… (each Claude call takes a few seconds)"):
            with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
                futures = {s: pool.submit(ask, s, question) for s in chosen}
                results = {}
                for s, f in futures.items():
                    try:
                        results[s] = f.result()
                    except Exception as e:
                        if "429" in str(e) and "free-models-per-day" in str(e):
                            # OpenRouter free tier: 50 requests/day (embeddings included)
                            results[s] = {"error": "OpenRouter's free daily limit (50 requests) is used up, so this "
                                                   "question can't be embedded. It resets at 00:00 UTC (05:30 IST). "
                                                   "Questions asked before still work: their vectors are saved."}
                        else:
                            results[s] = {"error": f"{type(e).__name__}: {e}"}

        # Show each system's result card: metrics, pages read, answer, and details in expanders
        for col, s in zip(cols, chosen):
            r = results[s]
            with col.container(border=True):
                st.subheader(s)
                if "error" in r:
                    st.error(r["error"])
                    continue
                m1, m2, m3 = st.columns(3)
                m1.metric("Seconds", r["latency_s"])
                m2.metric("LLM calls", r["llm_calls"])
                m3.metric("Chars read", f"{r['chars_read']:,}")
                st.markdown("**Pages read** " + pages_badges(r["pages"], gold))
                st.markdown(r["answer"])
                if r.get("queries_tried") and len(r["queries_tried"]) > 1:
                    with st.expander("Rewritten search queries"):
                        for q in r["queries_tried"]:
                            st.write("•", q)
                if r.get("chunks"):
                    with st.expander("Retrieved chunks"):
                        for c in r["chunks"]:
                            st.caption(f"{c['id']} · page {c['page']} · distance {c['distance']}")
                            st.write(c["text"])
                if r.get("steps"):
                    with st.expander("PageIndex tool calls"):
                        for step in r["steps"]:
                            st.code(step, language=None)

# ---------- tab 2: Comparison results ----------
with results_tab:
    runs_file = config.COMPARE_DIR / "runs.json"
    if not runs_file.exists():
        st.info("No results yet. Run `uv run vectorless-vs-vectordb-rag compare` or compare.ipynb first.")
    else:
        # passing the file's modified time means the cache refreshes only when runs.json changes
        df, summary = comparison_tables(runs_file.stat().st_mtime)
        st.markdown(f"**{df['id'].nunique()} questions × {df['system'].nunique()} systems**, same answering model. "
                    "Answers judged 1-5 by Claude against the correct pages (shuffled, unlabeled).")

        # headline numbers
        cols = st.columns(len(summary))
        for col, (name, row) in zip(cols, summary.iterrows()):
            with col.container(border=True):
                st.markdown(f"**{name}**")
                a, b, c = st.columns(3)
                a.metric("Recall", f"{row['recall']:.0%}")
                b.metric("Score", f"{row['score']:.2f}")
                c.metric("Sec/q", f"{row['latency_s']:.1f}")

        # one chart per metric (never two scales on one axis), colour fixed per system
        metrics = {"recall": "Recall of correct pages", "precision": "Precision (correct / read)",
                   "score": "Answer score (1-5)", "latency_s": "Seconds per question",
                   "chars_read": "Characters read by the LLM", "llm_calls": "LLM calls per question"}
        # melt = reshape to one row per (system, metric, value), the shape Altair charts expect
        long = summary.reset_index().melt(id_vars="system", value_vars=list(metrics))
        # system names sit under the bars, so no legend box is needed
        color = alt.Color("system:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None)
        chart_cols = st.columns(3)
        for i, (key, title) in enumerate(metrics.items()):
            data = long[long["variable"] == key]
            chart = (alt.Chart(data, title=title)
                     .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                     .encode(x=alt.X("system:N", sort=list(SYSTEMS), title=None,
                                     axis=alt.Axis(labelAngle=0, labelLimit=140)),
                             y=alt.Y("value:Q", title=None),
                             color=color, tooltip=["system", alt.Tooltip("value:Q", format=",.2f")])
                     .properties(height=220))
            chart_cols[i % 3].altair_chart(chart, width="stretch")

        st.subheader("Summary table")
        st.dataframe(summary, width="stretch")

        # Per-question scores, then every answer in an expander with its pages (green = correct)
        st.subheader("Per question")
        view = df.pivot(index=["id", "question"], columns="system", values="score")[list(SYSTEMS)]
        st.caption("Judged answer score per question")
        st.dataframe(view, width="stretch")
        for q in QUESTIONS:
            sub = df[df["id"] == q["id"]]
            if sub.empty:
                continue
            with st.expander(f"{q['id']} · {q['question']}"):
                st.markdown("Correct pages: " + pages_badges(q["gold"], set(q["gold"])))
                for _, r in sub.iterrows():
                    st.markdown(f"**{r['system']}** · score {r['score']} · {r['latency_s']} s · pages "
                                + pages_badges(r["pages"], set(q["gold"])))
                    st.markdown(r["answer"])
                    st.divider()

# ---------- tab 3: How it works (static explanation) ----------
with how_tab:
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.subheader("Vector RAG")
        st.code("question → embed\n→ top-5 similar chunks\n→ Claude answers", language=None)
        st.write("One similarity search in Chroma. Fast and cheap, but it trusts whatever the vectors return.")
    with b.container(border=True):
        st.subheader("Agentic vector RAG")
        st.code("retrieve → grade ─ok──→ generate\n   ↑         │\n   └─ retry ─┘ (better query)", language=None)
        st.write("LangGraph loop: Claude checks the chunks and rewrites the search query when they miss.")
    with c.container(border=True):
        st.subheader("PageIndex (vectorless)")
        st.code("question → Claude reads the tree\n→ opens chosen pages\n→ Claude answers", language=None)
        st.write("No embeddings. The LLM reasons over a table-of-contents tree and decides which pages to read.")
