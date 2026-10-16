r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"
PDF_EXTRACTION_TIMEOUT_SECONDS = 5
MAX_BAD_PDFS_BEFORE_FALLBACK = 3
MAX_PDFS_TO_TRY = 3

SCENARIO = "research_paper_navigator"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Capstone corpus + baseline retriever for Research Paper Navigator

# %%
DATASET_DIRS = [
    Path(__file__).resolve().parent / "ResearchPapers CAOP",
    Path(__file__).resolve().parent / "ResearchPaper CAOP",
    Path(__file__).resolve().parent / "researchPaper CAOP",
    Path.cwd() / "ResearchPapers CAOP",
    Path.cwd() / "ResearchPaper CAOP",
    Path.cwd() / "researchPaper CAOP",
]

SAMPLE_DOCS = [
    {"id": "doc1", "text": "Program synthesis: generating programs automatically from a specification, such as input-output examples or a logical formula."},
    {"id": "doc2", "text": "The sketching approach lets a programmer write a partial program with holes, and a synthesizer fills the holes to satisfy a specification."},
    {"id": "doc3", "text": "Retrieval-augmented generation grounds a language model's answers in documents retrieved from a corpus, reducing hallucination."},
    {"id": "doc4", "text": "BM25 is a keyword ranking function that scores documents by term frequency and inverse document frequency."},
    {"id": "doc5", "text": "Vector search embeds text into dense vectors and ranks documents by cosine similarity to the query embedding."},
    {"id": "doc6", "text": "Evaluation of retrieval systems measures whether the retrieved documents actually contain the information needed to answer the query."},
]


def _extract_pdf_text(pdf_path: Path) -> str:
    script = r'''
import json
import re
import sys
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(sys.argv[1])
try:
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    full_text = " ".join(text_parts)
    clean_text = re.sub(r"\s+", " ", full_text).strip()
    print(json.dumps({"ok": True, "text": clean_text}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
'''

    result = subprocess.run(
        [sys.executable, "-c", script, str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=PDF_EXTRACTION_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "PDF extraction failed")

    payload = json.loads(result.stdout.strip() or '{"ok": false, "error": "empty output"}')
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "unknown extraction error"))
    return payload.get("text", "")


def load_corpus(folder: Path) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    if not folder.exists():
        print(f"Folder not found: {folder}. Using sample corpus instead.")
        return {d["id"]: d for d in SAMPLE_DOCS}

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder}. Using sample corpus instead.")
        return {d["id"]: d for d in SAMPLE_DOCS}

    pdf_files = pdf_files[:MAX_PDFS_TO_TRY]
    bad_pdf_count = 0
    for idx, pdf_path in enumerate(pdf_files, start=1):
        try:
            clean_text = _extract_pdf_text(pdf_path)
            if clean_text:
                doc_id = f"{pdf_path.stem}_{idx}"
                docs[doc_id] = {"id": doc_id, "text": clean_text}
        except (Exception, KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            bad_pdf_count += 1
            print(f"Could not read {pdf_path.name}: {exc}")
            if bad_pdf_count >= MAX_BAD_PDFS_BEFORE_FALLBACK:
                print(f"Too many unreadable PDFs in {folder}. Using sample corpus instead.")
                return {d["id"]: d for d in SAMPLE_DOCS}
            continue

    if not docs:
        print(f"Could not extract text from PDFs in {folder}. Using sample corpus instead.")
        return {d["id"]: d for d in SAMPLE_DOCS}

    print(f"Loaded {len(docs)} documents from: {folder}")
    return docs


DATASET_DIR = next((d for d in DATASET_DIRS if d.exists()), DATASET_DIRS[0])
DOC_BY_ID = load_corpus(DATASET_DIR)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    q = _tokens(query)
    scored = []
    for doc in DOC_BY_ID.values():
        overlap = len(q & _tokens(doc["text"]))
        if overlap > 0:
            scored.append((doc["id"], float(overlap)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """Return a small evaluation set for the Research Paper Navigator scenario and CAOP corpus."""
    return [
        {
            "question": "What is ENCOMPASS and how does it help agent programming?",
            "grading_notes": "A correct answer says ENCOMPASS is a framework for agent programming that separates workflow logic from inference-time strategy, using probabilistic angelic nondeterminism and a Python-based search framework over program execution paths.",
        },
        {
            "question": "What is SPARLING and why is sparse activation important?",
            "grading_notes": "A correct answer explains that SPARLING learns sparse latent intermediate representations, uses extreme sparsity to model intermediate states, and identifies motifs that can be localized without assuming a fully identifiable parameterization.",
        },
        {
            "question": "What is the counterfeit conundrum in code language models?",
            "grading_notes": "A correct answer states that the counterfeit conundrum is the problem where code models generate incorrect programs that pass weak correctness checks, and the models often fail to recognize or repair these counterfeit samples.",
        },
        {
            "question": "How does ENCOMPASS separate workflow logic from inference-time search?",
            "grading_notes": "A correct answer says ENCOMPASS uses a programming model based on probabilistic angelic nondeterminism to describe the agent workflow separately from the search strategy, so inference-time choices can be changed without rewriting the workflow.",
        },
        {
            "question": "Why do code language models struggle with counterfeit programs?",
            "grading_notes": "A correct answer explains that counterfeit programs can look correct enough to pass weak checks, yet models often misclassify them, mispredict their behavior, and struggle to repair them even when they are generated by the same model family.",
        },
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation() -> None:
    llm = make_llm()
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    for i, item in enumerate(eval_set, 1):
        hits = retrieve(item["question"], TOP_K)
        ans = answer(llm, item["question"], [doc_id for doc_id, _ in hits]) if hits else "(no documents retrieved)"
        verdict = judge(llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={hits}  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}", f"retrieved={hits}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework() -> None:
    llm = make_llm()
    q = "What is ENCOMPASS?"
    notes = "A correct answer says ENCOMPASS is a framework for agent programming that separates workflow logic from inference-time search and uses probabilistic angelic nondeterminism."
    good = answer(llm, q, [doc_id for doc_id, _ in retrieve(q)])
    manipulated = "ENCOMPASS is a neural network model that generates random code from a prompt without any search strategy."
    good_verdict = judge(llm, good, notes)
    manip_verdict = judge(llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


# %%
run_evaluation()
validate_framework()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
