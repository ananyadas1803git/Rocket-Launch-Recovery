"""Small, local TF-IDF retrieval layer for the project Q&A assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "docs" / "rag_knowledge_base.md"


@dataclass(frozen=True)
class KnowledgeChunk:
    title: str
    text: str
    sources: tuple[str, ...]


@lru_cache(maxsize=1)
def _index() -> tuple[list[KnowledgeChunk], TfidfVectorizer, object]:
    sections: list[KnowledgeChunk] = []
    title = "Project overview"
    body: list[str] = []

    def save_section() -> None:
        content = "\n".join(body).strip()
        if not content:
            return
        source_match = re.search(r"(?m)^Sources:\s*(.+)$", content)
        sources = tuple(
            item.strip() for item in source_match.group(1).split(";")
        ) if source_match else ("Project knowledge base",)
        text = re.sub(r"(?m)^Sources:.*$", "", content).strip()
        sections.append(KnowledgeChunk(title=title, text=text, sources=sources))

    for line in KNOWLEDGE_BASE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            save_section()
            title = line[3:].strip()
            body = []
        elif not line.startswith("# "):
            body.append(line)
    save_section()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([f"{chunk.title}. {chunk.text}" for chunk in sections])
    return sections, vectorizer, matrix


def retrieve(question: str, limit: int = 3) -> list[KnowledgeChunk]:
    """Retrieve the most relevant project notes for a user question."""
    sections, vectorizer, matrix = _index()
    query = vectorizer.transform([question.strip()])
    if query.nnz == 0:
        return []
    scores = cosine_similarity(query, matrix).ravel()
    ranked = scores.argsort()[::-1]
    return [sections[index] for index in ranked[:limit] if scores[index] >= 0.04]


def generate_answer(
    question: str,
    chunks: list[KnowledgeChunk],
    *,
    api_key: str,
    model: str = "gpt-5-mini",
    prediction_context: str | None = None,
) -> str:
    """Generate an answer using only retrieved project notes and app context."""
    from google import genai

    source_context = "\n\n".join(
        f"[{index}] {chunk.title}\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    if prediction_context:
        source_context += f"\n\n[P] Current prediction from the app (user input and classifier output):\n{prediction_context}"

    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=model,
        system_instruction=(
            "You answer questions about the Rocket Launch Classifier project. "
            "Use only the supplied project notes and current prediction context. "
            "If the notes do not contain the answer, say you cannot establish it from "
            "the available project sources. Do not invent technical or safety claims. "
            "Keep the answer concise. Cite factual statements with the exact source "
            "markers [1], [2], etc. For facts from the current app prediction, cite [P]. "
            "Never present model behavior as physical causation or a launch recommendation."
        ),
        input=f"Project notes:\n{source_context or '(No relevant project notes were retrieved.)'}\n\nQuestion: {question}",
        store=False,
    )
    return (interaction.output_text or "").strip()
