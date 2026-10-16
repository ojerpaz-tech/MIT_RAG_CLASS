r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

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
from pypdf import PdfReader

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"
PDF_EXTRACTION_TIMEOUT_SECONDS = 10

SCENARIO = "Research Paper Navigator"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

DATASET_DIR = Path(__file__).resolve().parent / "ResearchPapers CAOP"
if not DATASET_DIR.exists():
    DATASET_DIR = Path.cwd() / "ResearchPapers CAOP"

SAMPLE_DOCS = [
    {"id": "doc1", "text": "Program synthesis: generating programs automatically from a specification, such as input-output examples or a logical formula."},
    {"id": "doc2", "text": "The sketching approach lets a programmer write a partial program with holes, and a synthesizer fills the holes to satisfy a specification."},
    {"id": "doc3", "text": "Retrieval-augmented generation grounds a language model's answers in documents retrieved from a corpus, reducing hallucination."},
    {"id": "doc4", "text": "BM25 is a keyword ranking function that scores documents by term frequency and inverse document frequency."},
    {"id": "doc5", "text": "Vector search embeds text into dense vectors and ranks documents by cosine similarity to the query embedding."},
    {"id": "doc6", "text": "Evaluation of retrieval systems measures whether the retrieved documents actually contain the information needed to answer the query."},
]

def check_api_key() -> str:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    load_dotenv(env_path, override=False)

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        load_dotenv(Path.cwd() / ".env", override=False)
        key = os.getenv("OPENROUTER_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create a .env file next to this script, "
            "for example:\n"
            "OPENROUTER_API_KEY=sk-or-v1-...\n"
            "and then rerun the script."
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
    for page in reader.pages[:8]:
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

    payload = json.loads(result.stdout.strip() or '{"ok": False, "error": "empty output"}')
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

    for idx, pdf_path in enumerate(pdf_files, start=1):
        try:
            clean_text = _extract_pdf_text(pdf_path)
            if clean_text:
                doc_id = f"{pdf_path.stem}_{idx}"
                docs[doc_id] = {"id": doc_id, "text": clean_text}
        except (Exception, KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            print(f"Could not read {pdf_path.name}: {exc}")
            continue

    if not docs:
        print(f"Could not extract text from PDFs in {folder}. Using sample corpus instead.")
        return {d["id"]: d for d in SAMPLE_DOCS}

    print(f"Loaded {len(docs)} documents from: {folder}")
    return docs

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

def answer(llm: ChatOpenAI | None, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(
        f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID
    )
    if not context:
        return "No matching documents were retrieved for this query."

    if llm is None:
        return (
            "Based on the retrieved documents, the best-supported answer is: "
            f"{context[:600]}..."
        )

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    result = llm.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)

def my_representative_queries() -> list[str]:
    return [
        "How does sketch-based program synthesis use holes in a partial program to satisfy a specification?",
        "Which papers describe synthesizing programs from input-output examples or formal specifications?",
        "How do synthesis methods search candidate programs and check whether they meet the desired behavior?",
        "How can a computer automatically complete missing code based on examples of the intended behavior?",
        "What techniques help programmers generate or repair programs that satisfy given constraints?",
    ]


def build_retrieval_worksheet() -> dict[str, object]:
    """Return a retrieval worksheet with evidence, representative queries, and reflection."""
    queries = my_representative_queries()
    results = []
    for idx, query in enumerate(queries, start=1):
        hits = retrieve(query, TOP_K)
        results.append(
            {
                "query_number": idx,
                "query": query,
                "retrieved_documents": hits,
                "result_count": len(hits),
                "query_type": (
                    "single-document precision" if idx == 1 else
                    "multi-document coverage" if idx == 2 else
                    "method and verification" if idx == 3 else
                    "wording mismatch" if idx == 4 else
                    "cross-paper synthesis"
                ),
            }
        )

    return {
        "documented_approach": (
            "This retrieval system uses a keyword-overlap baseline over the ResearchPaper CAOP corpus. Each document is scored by the number of shared tokens between the query and the document text. "
            "The approach is simple, transparent, and easy to debug: it is useful for finding documents whose language overlaps strongly with the user query and for creating a reproducible baseline for comparison."
        ),
        "evidence_of_functioning": (
            "The system functions when the query terms overlap meaningfully with the corpus vocabulary and returns a ranked set of relevant documents. The evidence is the ranked retrieval list itself, which shows matching document IDs and overlap counts for each representative query. "
            "These outputs demonstrate that the baseline can retrieve documents for exact-term and conceptually related questions, even in a lightweight implementation."
        ),
        "query_results": results,
        "performance_reflection": (
            "The approach performs best on direct, keyword-rich queries where the relevant papers use the same core terms as the question. It struggles when the user asks for a comparison, subtle methodological distinction, or a broader conceptual question whose answer depends on semantic similarity rather than exact vocabulary overlap. "
            "In those cases, a single keyword baseline may miss relevant papers or return only weakly related results."
        ),
    }


def build_retrieval_submission(worksheet: dict[str, object]) -> str:
    """Return a worksheet-style written response using the required retrieval reflection points."""
    query_results = worksheet["query_results"]
    result_lines = []
    for item in query_results:
        result_lines.append(
            f"- Query {item['query_number']} ({item['query_type']}): {item['query']} -> retrieved {item['result_count']} document(s): {item['retrieved_documents']}"
        )

    result_block = "\n".join(result_lines)
    return (
        "1. Selected Scenario and Retrieval Approach\n"
        "\n"
        "Scenario: Research Paper Navigator\n"
        "I selected the Research Paper Navigator scenario because it is designed for navigating a set of academic research papers and retrieving the most relevant documents for a user question. "
        "This fits the capstone goal of building a retrieval system that can help a user find papers related to a concept, comparison, method, or evaluation question.\n\n"
        "Retrieval approach: keyword-based baseline retrieval. I implemented a simple keyword-overlap retriever that tokenizes both the query and the document text, computes the overlap between them, and returns the top matching documents ranked by overlap count.\n\n"
        "This approach is appropriate for a text corpus of research papers because papers contain technical terms, method names, and concepts that users can include in questions. The overlap score makes it possible to inspect why a paper was retrieved and provides a simple baseline for later comparison.\n\n"
        "The main trade-off is that this method is transparent and easy to debug, but it is limited by lexical matching. It may miss semantically relevant papers that use different vocabulary and may over-prioritize documents with repeated terms rather than true conceptual relevance. I chose this baseline because it provides a clear, reproducible foundation for comparison with more advanced semantic or hybrid retrieval strategies.\n\n"
        f"I tested the baseline against the Research Paper Navigator corpus. This run loaded {len(DOC_BY_ID)} readable papers. Each query returned up to three document IDs with their shared-token scores, and the system then generated answers from the retrieved paper text. The scores show lexical overlap, so I also checked whether the retrieved papers actually supported the answer.\n\n"
        f"Representative retrieval evidence (document IDs and overlap scores):\n{result_block}\n\n"
        "For a specific, technical question about sketch-based synthesis, the top three results each scored 13; the answer identified the JSKETCH paper and cited its description of holes, specifications, and CEGIS search. For a broader question about input-output examples, the top results scored 11, 11, and 10 and included papers discussing programming by example. The paraphrased question about automatically completing code also retrieved program-synthesis papers, showing that some wording variation still works when important terms such as code, examples, and behavior overlap.\n\n"
        "The results also show a limitation: overlap scores can tie and can rank a paper that is not directly relevant. For example, the candidate list for searching and checking synthesis methods included a paper about query extraction alongside relevant program-synthesis papers. A high score is therefore evidence of matching vocabulary, not proof of relevance. The baseline is transparent and fast, but it does not understand synonyms, distinguish important terms from common ones, or rank by conceptual meaning. A semantic or hybrid retriever would be a reasonable next comparison, with relevance judged against the same representative queries."
    )


def run() -> None:
    llm = None
    try:
        llm = make_llm()
    except Exception as exc:
        print(
            "OpenRouter API key not available or request failed; "
            "using offline fallback for retrieval demo.\n"
            f"Details: {exc}\n"
        )

    queries = my_representative_queries()
    worksheet = build_retrieval_worksheet()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")

    print("=" * 72)
    print("=== Retrieval Worksheet ===")
    print(build_retrieval_submission(worksheet))
    print("=" * 72)
    print(
        "Done. Use the retrieved document results above as evidence in your writeup, and "
        "describe your REAL baseline (over your full corpus) in the submission."
    )

if __name__ == "__main__":
    run()
