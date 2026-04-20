"""Personalised coping suggestions mapped to normalised emotion labels (for dissertation / UX)."""

COPING_BY_EMOTION: dict[str, list[str]] = {
    "sadness": [
        "Try naming one feeling in one word, then one tiny action for the next 10 minutes (e.g. drink water, open a window).",
        "Gentle movement: a 5-minute walk or stretching—no pressure to 'fix' the mood.",
        "Text or call one person you trust, even with a short ‘having a rough day’ message.",
        "Journaling prompt: ‘What would I tell a friend who felt exactly like this?’",
    ],
    "fear": [
        "Grounding: 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste.",
        "Box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s—repeat a few rounds.",
        "Write the worry in one sentence, then one sentence of what you can control today.",
    ],
    "anger": [
        "Pause: step away from the screen for 2 minutes; shake out your hands and shoulders.",
        "Write an unsent letter—get it out without sending—then tear it up or delete it.",
        "Physical outlet: brisk walk, cold water on wrists, or a safe burst of movement.",
    ],
    "disgust": [
        "Self-compassion break: ‘This is a moment of suffering; suffering is part of life; may I be kind to myself.’",
        "One small act of care: wash your face, change clothes, or tidy one small area.",
        "If body image is involved, limit mirror-checking for today and reach for Beat/Mind resources when ready.",
    ],
    "joy": [
        "Savour it: notice where you feel it in your body and one thing that contributed.",
        "Channel it: do one small thing that reinforces what went well.",
    ],
    "surprise": [
        "Take three slow breaths, then jot what surprised you and what you need next.",
    ],
    "neutral": [
        "Check in: hunger, thirst, sleep, and one small task you could finish in 5 minutes.",
        "Brief mindfulness: 1 minute of noticing breath without changing it.",
    ],
    "anxiety": [
        "Label it: ‘I’m having anxious thoughts’—separate thought from fact.",
        "5-4-3-2-1 grounding, then one small ‘next right step’ only.",
        "Limit doom-scrolling; set a 10-minute timer for worry, then switch activity.",
    ],
}

# Map model labels (e.g. surprise → anxious coping sometimes)
LABEL_ALIASES = {
    "happiness": "joy",
    "happy": "joy",
    "love": "joy",
    "anxious": "anxiety",
    "worry": "anxiety",
    "worried": "anxiety",
    "stress": "anxiety",
    "stressed": "anxiety",
}


def normalise_emotion_label(label: str) -> str:
    l = (label or "neutral").lower().strip()
    return LABEL_ALIASES.get(l, l)


def suggestions_for_emotion(label: str, limit: int = 4) -> list[str]:
    key = normalise_emotion_label(label)
    if key not in COPING_BY_EMOTION and key in LABEL_ALIASES:
        key = LABEL_ALIASES.get(key, key)
    base = COPING_BY_EMOTION.get(key) or COPING_BY_EMOTION["neutral"]
    return base[:limit]


def merge_face_and_text_suggestions(text_label: str, face_label: str | None, limit: int = 4) -> list[str]:
    """Blend suggestions when both modalities present; de-duplicate."""
    seen: set[str] = set()
    out: list[str] = []
    for s in suggestions_for_emotion(text_label, limit=limit) + (
        suggestions_for_emotion(face_label, limit=limit) if face_label else []
    ):
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out
