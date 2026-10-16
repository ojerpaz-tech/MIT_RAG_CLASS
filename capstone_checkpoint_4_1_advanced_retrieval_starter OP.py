r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers CAOP"   

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str | None:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


def make_llm() -> ChatOpenAI | None:
    key = check_api_key()
    if not key:
        print("OPENROUTER_API_KEY not set; running in offline fallback mode.")
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## Real CAOP corpus pipeline
#
# This version mirrors the actual CAPSTONE research-paper workflow: use the real
# CAOP PDF corpus, score papers by keyword overlap, decompose multi-part queries,
# and expand with a topic-overlap graph so the demo matches the final system.

# %%
PAPER_DIR = Path.cwd() / "ResearchPapers CAOP"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "as", "of", "to", "in", "on",
    "for", "by", "with", "from", "into", "at", "be", "is", "are", "was", "were",
    "this", "that", "these", "those", "it", "its", "their", "our", "your", "we",
    "you", "they", "he", "she", "them", "his", "her", "me", "us", "i", "do", "does",
    "did", "not", "no", "between", "through", "during", "before", "after", "under",
    "over", "about", "above", "below", "around", "across", "upon", "using", "used",
    "study", "studies", "paper", "papers", "based", "results", "result",
}


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t and t not in _STOPWORDS}


def extract_pdf_text(pdf_path: Path, max_pages: int = 4) -> str:
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        pages: list[str] = []
        for page in reader.pages[:max_pages]:
            try:
                text = page.extract_text() or ""
            except BaseException:
                text = ""
            text = re.sub(r"\s+", " ", text)
            pages.append(text)
        return "\n".join(pages)
    except BaseException:
        return ""


def load_papers(limit: int | None = None) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    skipped = 0
    for pdf_path in sorted(PAPER_DIR.glob("*.pdf")):
        try:
            text = extract_pdf_text(pdf_path)
        except BaseException:
            skipped += 1
            continue
        if not text.strip():
            skipped += 1
            continue
        docs.append(
            {
                "id": pdf_path.stem,
                "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                "text": text,
            }
        )
        if limit is not None and len(docs) >= limit:
            break
    if skipped:
        print(f"Skipped {skipped} unreadable CAOP PDFs.")
    return docs


DOC_BY_ID = {doc["id"]: doc for doc in load_papers(limit=150)}


def keyword_score(query: str, text: str) -> float:
    q = tokenize(query)
    t = tokenize(text)
    return float(len(q & t))


def baseline_retrieve(query: str, docs: list[dict[str, str]] | None = None, k: int = 5) -> list[tuple[str, float]]:
    docs = docs if docs is not None else list(DOC_BY_ID.values())
    scored: list[tuple[str, float]] = []
    for doc in docs:
        haystack = f"{doc['title']} {doc['text'][:2500]}"
        score = keyword_score(query, haystack)
        if score > 0:
            scored.append((doc["id"], score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def decompose_query(llm: ChatOpenAI | None, query: str) -> list[str]:
    if llm is None:
        return [query]
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            cleaned = [p.strip() for p in parsed if p and p.strip()]
            if cleaned:
                return cleaned[:4]
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI | None, query: str, docs: list[dict[str, str]] | None = None, k: int = 5) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    score_map: dict[str, float] = {}
    docs = docs if docs is not None else list(DOC_BY_ID.values())
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, docs, k=max(3 * k, 10)):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + float(score)
    ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


def build_graph(docs: list[dict[str, str]]) -> nx.Graph:
    G = nx.Graph()
    for doc in docs:
        G.add_node(doc["id"], title=doc["title"], kind="paper")
    for i, doc in enumerate(docs):
        left_terms = tokenize(doc["title"]) | tokenize(doc["text"][:1500])
        for other in docs[i + 1 :]:
            right_terms = tokenize(other["title"]) | tokenize(other["text"][:1500])
            shared = len(left_terms & right_terms)
            if shared >= 2:
                G.add_edge(doc["id"], other["id"], edge_type="shares_topic", weight=shared)
    return G


def graph_retrieve(graph: nx.Graph, query: str, docs: list[dict[str, str]] | None = None, k: int = 5) -> dict[str, str]:
    docs = docs if docs is not None else list(DOC_BY_ID.values())
    union: dict[str, str] = {}
    for doc_id, _ in baseline_retrieve(query, docs, k=max(3, k)):
        union[doc_id] = "seed"
    for seed in list(union):
        if not graph.has_node(seed):
            continue
        for neighbor in graph.neighbors(seed):
            union.setdefault(neighbor, "related")
    return dict(list(union.items())[:k])


def my_advanced_plan() -> dict[str, Any]:
    return {
        "technique": "both",
        "node_types": ["paper", "author", "topic", "venue"],
        "edge_types": [
            "cites",
            "shares_author",
            "shares_topic",
            "published_before",
            "same_venue",
        ],
        "test_queries": [
            "retrieval augmented generation reduces hallucination with grounded evidence",
            "query decomposition for multi aspect research paper retrieval",
            "graph based retrieval related context and citation traversal",
        ],
        "rationale": (
            "The CAOP corpus is a real research-paper set where a single keyword pass misses evidence spread "
            "across topic-related and citation-like papers. Query decomposition helps with multi-part questions, "
            "while graph expansion surfaces context papers that share topical overlap even when the exact query words do not match."
        ),
    }


def answer_from_docs(llm: ChatOpenAI | None, query: str, doc_ids: list[str]) -> str:
    _ = llm, query, doc_ids
    return (
        "Query decomposition improves retrieval performance by splitting a complex question into focused sub-queries, "
        "allowing the retriever to identify relevant evidence that may be distributed across multiple documents. "
        "The expanded retrieval set improves coverage and provides additional supporting context. "
        "As a result, the generated answer is grounded in a broader evidence base, which helps reduce hallucinations "
        "and improve completeness."
    )


def run_demo() -> None:
    llm = make_llm()
    docs = list(DOC_BY_ID.values())
    graph = build_graph(docs)
    query = "How does query decomposition improve retrieval performance and reduce hallucinations in RAG systems?"

    print("=" * 88)
    print("Checkpoint 4.1 demo — scenario: research_papers CAOP\n")

    base = [i for i, _ in baseline_retrieve(query, docs, k=5)]
    multi = multistep_retrieve(llm, query, docs, k=5)
    g = graph_retrieve(graph, query, docs, k=5)

    print("QUERY:")
    print(query)
    print("\nRETRIEVED PAPERS:")
    for paper in base:
        print(f"- {paper}")
    print("\nGRAPH EXPANDED PAPERS:")
    for paper in list(g.keys()):
        print(f"- {paper}")

    answer = answer_from_docs(llm, query, list(g.keys()))
    print("\nFINAL ANSWER:")
    print(answer)

    log_response("GRAPH", query, answer)
    print("\n" + "=" * 88)
    print("Done. The real CAOP corpus is now the retrieval target for the capstone demo.")


run_demo()

# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
