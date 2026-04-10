"""
RAG (Retrieval-Augmented Generation): retrieve relevant knowledge chunks for more accurate, grounded replies.
"""

import json
import os
import re
from pathlib import Path

_KB: list[dict] | None = None


def _load_knowledge_base() -> list[dict]:
    global _KB
    if _KB is not None:
        return _KB
    base = Path(__file__).resolve().parent
    path = base / "data" / "rag_knowledge.json"
    if not path.exists():
        _KB = []
        return _KB
    with open(path, encoding="utf-8") as f:
        _KB = json.load(f)
    return _KB


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def retrieve_rag_context(user_message: str, top_k: int = 3) -> str:
    """
    Score knowledge chunks by keyword overlap + word overlap with content.
    Returns formatted string for injection into system prompt, or empty if disabled/empty.
    """
    if os.environ.get("RAG_ENABLED", "true").lower() in ("0", "false", "no"):
        return ""

    kb = _load_knowledge_base()
    if not kb:
        return ""

    try:
        top_k = int(os.environ.get("RAG_TOP_K", str(top_k)))
    except ValueError:
        top_k = 3

    query = user_message.lower()
    query_tokens = _tokenize(user_message)
    scored: list[tuple[float, dict]] = []

    for chunk in kb:
        score = 0.0
        for kw in chunk.get("keywords", []):
            if kw.lower() in query:
                score += 3.0
        content_lower = chunk.get("content", "").lower()
        for t in query_tokens:
            if len(t) > 2 and t in content_lower:
                score += 0.3
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])
    picked = [c for _, c in scored[:top_k]]

    if not picked:
        return ""

    parts = []
    for i, chunk in enumerate(picked, 1):
        parts.append(f"[{chunk.get('id', i)}] {chunk['content']}")
    return "\n\n".join(parts)


def augment_system_prompt(base_prompt: str, user_message: str) -> str:
    """Append retrieved knowledge to system prompt for this turn."""
    rag = retrieve_rag_context(user_message)
    if not rag:
        return base_prompt
    return (
        base_prompt
        + "\n\n---\nRetrieved reference knowledge (ground your answer in this when it matches the user's topic; "
        "still be conversational and never dump lists unless helpful):\n"
        + rag
    )
