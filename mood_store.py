"""
Mood / emotion event storage for context memory and analytics.

Default: JSON file under data/mood_events.json (local dev / dissertation demo).
Optional: MOOD_BACKEND=dynamodb + AWS_REGION + table name (see ETHICS_AND_PRIVACY.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_DATA_DIR = Path(__file__).resolve().parent / "data"
_JSON_PATH = _DATA_DIR / "mood_events.json"

NEGATIVE_LABELS = frozenset({"sadness", "fear", "anger", "disgust", "anxious", "anxiety", "stress", "stressed", "worried"})


def _anonymize_session(session_id: str) -> str:
    if os.environ.get("MOOD_ANONYMIZE", "").lower() in ("1", "true", "yes"):
        salt = os.environ.get("MOOD_HASH_SALT", "change-me-in-production")
        h = hashlib.sha256(f"{session_id}:{salt}".encode()).hexdigest()
        return h[:20]
    return session_id


def _load_json_events() -> list[dict]:
    if not _JSON_PATH.exists():
        return []
    try:
        with open(_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_json_events(events: list[dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _JSON_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=0)
    tmp.replace(_JSON_PATH)


def _dynamo_table():
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    table_name = os.environ.get("MOOD_DYNAMODB_TABLE", "mental-health-mood-events")
    if not region:
        return None, None
    db = boto3.resource("dynamodb", region_name=region)
    return db.Table(table_name), table_name


def record_event(
    session_id: str,
    *,
    text_emotion_label: str,
    text_confidence: float,
    text_backend: str,
    face_emotion_label: str | None,
    face_confidence: float | None,
    message_preview: str,
    is_crisis: bool,
) -> None:
    """Append one anonymised (optional) mood event."""
    sid = _anonymize_session(session_id)
    preview = (message_preview or "")[:200]
    evt = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_key": sid,
        "text_emotion": text_emotion_label,
        "text_confidence": text_confidence,
        "text_backend": text_backend,
        "face_emotion": face_emotion_label,
        "face_confidence": face_confidence,
        "message_preview": preview,
        "is_crisis": is_crisis,
    }

    backend = (os.environ.get("MOOD_BACKEND") or "json").lower()
    if backend == "dynamodb":
        try:
            table, _ = _dynamo_table()
            if table:
                table.put_item(Item={k: v for k, v in evt.items() if v is not None})
                return
        except Exception:
            pass

    with _lock:
        events = _load_json_events()
        events.append(evt)
        # cap file size for local demos
        if len(events) > 5000:
            events = events[-5000:]
        _save_json_events(events)


def get_session_events(session_id: str, limit: int = 12) -> list[dict]:
    sid = _anonymize_session(session_id)
    with _lock:
        events = _load_json_events()
    mine = [e for e in events if e.get("session_key") == sid]
    return mine[-limit:]


def detect_mood_pattern(session_id: str) -> str | None:
    """
    If recent messages skew negative, suggest deeper support (not diagnosis).
    """
    recent = get_session_events(session_id, limit=8)
    if len(recent) < 3:
        return None
    last3 = recent[-3:]
    negative_core = frozenset({"sadness", "fear", "anger", "disgust"})
    neg = sum(
        1
        for e in last3
        if (e.get("text_emotion") or "").lower() in negative_core
        or (e.get("text_emotion") or "").lower() in NEGATIVE_LABELS
    )
    if neg >= 3:
        return (
            "Several recent messages lean toward difficult emotions (sadness, fear, anger, or disgust). "
            "Acknowledge that pattern gently and suggest NHS Talking Therapies, GP, or Samaritans if they might want extra support—not as a command, as an caring option."
        )
    if neg == 2 and len(recent) >= 4:
        return (
            "The user may be having a sustained hard patch. Offer warmth and optionally mention professional support if it feels appropriate."
        )
    return None


def load_all_events() -> list[dict]:
    with _lock:
        return list(_load_json_events())


def analytics_summary() -> dict:
    """For /api/analytics/summary and dashboard."""
    events = load_all_events()
    emotion_counts: dict[str, int] = {}
    crisis_count = 0
    by_day: dict[str, int] = {}
    for e in events:
        lab = e.get("text_emotion") or "unknown"
        emotion_counts[lab] = emotion_counts.get(lab, 0) + 1
        if e.get("is_crisis"):
            crisis_count += 1
        ts = e.get("ts", "")[:10]
        if ts:
            by_day[ts] = by_day.get(ts, 0) + 1
    return {
        "total_events": len(events),
        "unique_sessions_estimate": len({e.get("session_key") for e in events}),
        "emotion_counts": emotion_counts,
        "crisis_flags_recorded": crisis_count,
        "messages_per_day": dict(sorted(by_day.items())[-14:]),
    }
