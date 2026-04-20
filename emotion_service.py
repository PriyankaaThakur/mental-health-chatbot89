"""
Emotion-aware signals for the chatbot.

- Text: DistilRoBERTa-based classifier (BERT family) via HuggingFace when `transformers` + `torch` are installed.
  Set USE_TRANSFORMERS_EMOTION=false to force lightweight heuristic only.
- Face: ViT image classifier (FER-style labels) when ML stack is available; optional image upload.

This does not diagnose medical conditions—it only supplies soft cues for tone and coping ideas.
"""

from __future__ import annotations

import os
import re
from typing import Any, BinaryIO

_TEXT_PIPE = None
_FACE_PIPE = None


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def _heuristic_text_emotion(text: str) -> dict[str, Any]:
    """Baseline keyword / pattern scorer (no ML)—useful for demos without torch."""
    t = text.lower()
    scores: dict[str, float] = {
        "sadness": 0.0,
        "joy": 0.0,
        "fear": 0.0,
        "anger": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
        "neutral": 0.15,
    }
    sad = r"\b(sad|depress|hopeless|empty|tired of|exhausted|cry|crying|hurt|worthless|lonely|blue)\b"
    fear = r"\b(scared|afraid|terrified|panic|worried|anxious|nervous|dread)\b"
    anger = r"\b(angry|furious|hate|rage|annoyed|pissed|irritated)\b"
    joy = r"\b(happy|glad|joy|excited|grateful|better|good day)\b"
    disgust = r"\b(disgust|gross|revolt|ashamed|hate myself)\b"

    if re.search(sad, t):
        scores["sadness"] += 0.55
    if re.search(fear, t):
        scores["fear"] += 0.5
    if re.search(anger, t):
        scores["anger"] += 0.5
    if re.search(joy, t):
        scores["joy"] += 0.45
    if re.search(disgust, t):
        scores["disgust"] += 0.45
    if "can't go on" in t or "cant go on" in t or "cannot go on" in t:
        scores["sadness"] += 0.35
        scores["fear"] += 0.2

    best = max(scores, key=lambda k: scores[k])
    conf = min(0.95, scores[best] + 0.05)
    return {
        "label": best,
        "confidence": float(conf),
        "backend": "heuristic_keyword",
        "all_scores": scores,
    }


def classify_text_emotion(text: str) -> dict[str, Any]:
    """Return {label, confidence, backend, all_scores?} for the given user text."""
    if not _env_bool("USE_TRANSFORMERS_EMOTION", True):
        return _heuristic_text_emotion(text)

    global _TEXT_PIPE
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        return _heuristic_text_emotion(text)

    try:
        if _TEXT_PIPE is None:
            model_id = os.environ.get(
                "EMOTION_TEXT_MODEL",
                "j-hartmann/emotion-english-distilroberta-base",
            )
            device = 0 if torch.cuda.is_available() else -1
            _TEXT_PIPE = pipeline(
                "text-classification",
                model=model_id,
                top_k=7,
                device=device,
            )
        clip = (text or "")[:512]
        if not clip.strip():
            return {"label": "neutral", "confidence": 0.5, "backend": "transformers", "all_scores": {}}
        raw = _TEXT_PIPE(clip)
        if not raw:
            return _heuristic_text_emotion(text)
        # Single string → list of {label, score}; batched → list of such lists
        ranked = raw[0] if raw and isinstance(raw[0], list) else raw
        if not ranked:
            return _heuristic_text_emotion(text)
        best = ranked[0]
        all_scores = {r["label"]: float(r["score"]) for r in ranked}
        return {
            "label": best["label"],
            "confidence": float(best["score"]),
            "backend": "transformers_distilroberta",
            "all_scores": all_scores,
        }
    except Exception:
        return _heuristic_text_emotion(text)


def classify_face_image(file_obj: BinaryIO) -> dict[str, Any] | None:
    """
    FER-style face emotion from an uploaded image (RGB).
    Returns None if ML stack missing or image invalid.
    """
    if not _env_bool("USE_FACE_EMOTION", True):
        return None

    global _FACE_PIPE
    try:
        import torch
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        return None

    try:
        img = Image.open(file_obj).convert("RGB")
    except Exception:
        return None

    try:
        if _FACE_PIPE is None:
            model_id = os.environ.get("EMOTION_FACE_MODEL", "trpakov/vit-face-expression")
            device = 0 if torch.cuda.is_available() else -1
            _FACE_PIPE = pipeline(
                "image-classification",
                model=model_id,
                device=device,
            )
        out = _FACE_PIPE(img)
        if isinstance(out, list) and out:
            best = out[0]
        elif isinstance(out, dict):
            best = out
        else:
            return None
        return {
            "label": str(best.get("label", "neutral")),
            "confidence": float(best.get("score", 0.0)),
            "backend": "transformers_vit_fer_style",
        }
    except Exception:
        return None


def build_emotion_instruction(
    text_em: dict[str, Any],
    face_em: dict[str, Any] | None,
    pattern_note: str | None,
    coping_lines: list[str],
) -> str:
    """Soft instructions appended to system prompt (not shown to user as diagnosis)."""
    parts: list[str] = []
    parts.append(
        f"- Text-derived emotional tone (soft cue only): **{text_em.get('label', 'neutral')}** "
        f"(~{text_em.get('confidence', 0):.0%} confidence, source: {text_em.get('backend', 'unknown')})."
    )
    if face_em:
        parts.append(
            f"- Optional **face image** cue (FER-style / ViT): **{face_em.get('label')}** "
            f"(~{face_em.get('confidence', 0):.0%}). If this conflicts with the user's words, trust the words."
        )
    if pattern_note:
        parts.append(f"- **Mood pattern over recent messages:** {pattern_note}")
    if coping_lines:
        parts.append(
            "Weave in **one or two** of these coping ideas naturally if they fit (do not dump a list):\n"
            + "\n".join(f"  • {c}" for c in coping_lines[:4])
        )
    parts.append(
        "Do not say you 'diagnosed' the user. Stay supportive, specific, and ethically cautious."
    )
    return "\n".join(parts)
