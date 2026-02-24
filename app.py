"""
Mental Health Support Chatbot - AI-Powered Flask Backend
Uses OpenAI GPT for intelligent, contextual responses. Crisis detection remains rule-based.
No persistent data storage - conversation history kept in memory per session only.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

import uuid
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

# In-memory conversation history (session_id -> list of messages)
# Cleared on server restart. No persistent storage.
conversations: dict[str, list[dict]] = {}

# Crisis keywords - trigger emergency response (bypasses AI)
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self-harm", "self harm", "cutting", "hurt myself"
]

SYSTEM_PROMPT = """You are a warm, caring mental health support assistant—like a supportive friend who truly listens.

Your approach:
- Validate feelings first: "I hear you", "That sounds really hard", "Your feelings make sense"
- Show genuine care and concern in every response
- Offer practical, gentle coping suggestions
- Keep responses conversational (2-3 short paragraphs)—not robotic or clinical
- Use "you" and speak directly to the person
- When they share something difficult, acknowledge it before offering advice

Example tone: "I'm really sorry you're going through this. Feeling [X] can be exhausting. Have you tried [gentle suggestion]? And remember, it's okay to reach out to a therapist if things feel too heavy—they're there to help."

Never: diagnose, prescribe, or sound cold. Always: be warm, human, and supportive."""


def is_crisis_message(text: str) -> bool:
    """Check if message contains crisis keywords."""
    msg_lower = text.lower().strip()
    return any(kw in msg_lower for kw in CRISIS_KEYWORDS)


def get_crisis_response() -> tuple[str, bool]:
    """Return emergency resources. Second value indicates is_crisis."""
    return (
        "I'm really concerned about what you're sharing. "
        "Please reach out for immediate support:\n\n"
        "• National Suicide Prevention Lifeline: 988 (US)\n"
        "• Crisis Text Line: Text HOME to 741741\n"
        "• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "You don't have to face this alone. These services are free and confidential.",
        True,
    )


def get_fallback_response(user_message: str) -> str:
    """Empathetic fallback when AI fails—never show errors to the user."""
    msg = user_message.lower()
    if any(w in msg for w in ["sad", "down", "depressed", "hopeless", "lonely"]):
        return (
            "I'm really sorry you're feeling this way. What you're going through sounds hard, "
            "and it takes courage to reach out. Remember: you don't have to face this alone. "
            "Talking to a friend, family member, or a therapist can make a real difference. "
            "Would you like to share a bit more about what's on your mind? I'm here to listen."
        )
    if any(w in msg for w in ["anxious", "anxiety", "worried", "nervous", "panic"]):
        return (
            "Anxiety can feel overwhelming—I hear you. Try taking a few slow breaths: "
            "breathe in for 4 counts, hold for 4, breathe out for 6. "
            "Grounding can help too: name 5 things you can see, 4 you can hear, 3 you can touch. "
            "If anxiety is affecting your daily life, a therapist can offer tools that really help. "
            "What's been weighing on you lately?"
        )
    if any(w in msg for w in ["stress", "stressed", "overwhelmed", "pressure"]):
        return (
            "Feeling overwhelmed is exhausting, and it's okay to admit that. "
            "Try breaking things into smaller steps—even one small thing at a time helps. "
            "Short breaks, a walk, or talking to someone you trust can lighten the load. "
            "You're doing your best, and that matters. What would feel most helpful right now?"
        )
    if any(w in msg for w in ["hi", "hello", "hey"]):
        return (
            "Hi there. I'm here to listen and support you. "
            "You can share how you're feeling—stress, anxiety, sadness, or anything else. "
            "How are you doing today?"
        )
    if any(w in msg for w in ["thank", "thanks"]):
        return (
            "You're welcome. I'm glad I could be here for you. "
            "Remember, it's okay to reach out whenever you need support. Take care of yourself."
        )
    return (
        "Thank you for sharing. I'm here to listen. "
        "It can help to talk about what's on your mind—whether it's stress, anxiety, sadness, or something else. "
        "What would you like to talk about?"
    )


def get_ai_response(user_message: str, session_id: str) -> tuple[str, bool]:
    """Get AI response. Tries Gemini → Groq → OpenAI. Falls back to empathetic response if all fail."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_key and not groq_key and not openai_key:
        return (
            "AI is not configured yet. Add GEMINI_API_KEY (free at aistudio.google.com/apikey) "
            "or GROQ_API_KEY (free at console.groq.com) in Render → Environment, then redeploy.",
            False,
        )

    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": "Hello. I'm here to listen and support you. You can share how you're feeling—whether it's stress, anxiety, sadness, or anything else. How are you doing today?",
            },
        ]

    history = conversations[session_id]
    history.append({"role": "user", "content": user_message})

    result = (None, "")
    if gemini_key:
        result = _get_gemini_response(history, gemini_key)
    if result[0] is None and groq_key:
        result = _get_groq_response(history, groq_key)
    if result[0] is None and openai_key:
        result = _get_openai_response(history, openai_key)

    if result and result[0] is not None:
        assistant_message = result[0]
        history.append({"role": "assistant", "content": assistant_message})
        if len(history) > 21:
            conversations[session_id] = [history[0]] + history[-20:]
        return (assistant_message, False)

    # All AI failed—use caring fallback so user never sees an error
    return (get_fallback_response(user_message), False)


def _get_gemini_response(history: list, api_key: str) -> tuple[str | None, str]:
    """Use Google Gemini (free). Returns (response, error_msg)."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            system_instruction=SYSTEM_PROMPT,
        )

        # Build chat with history (skip system message)
        gemini_history = []
        for msg in history[1:-1]:  # skip system, skip last (current user)
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(history[-1]["content"])
        return (response.text, "")
    except Exception as e:
        err = str(e).lower()
        if "api_key" in err or "invalid" in err:
            return (None, "Invalid Gemini API key. Check GEMINI_API_KEY.")
        if "quota" in err or "rate" in err:
            return (None, "Gemini limit reached. Try again in a moment.")
        return (None, "Sorry, I couldn't process that. Please try again.")


def _get_groq_response(history: list, api_key: str) -> tuple[str | None, str]:
    """Use Groq (free tier). Returns (response, error_msg)."""
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=history,
            max_tokens=500,
            temperature=0.7,
        )
        return (response.choices[0].message.content, "")
    except Exception as e:
        err = str(e).lower()
        if "api_key" in err or "invalid" in err or "auth" in err:
            return (None, "Invalid Groq API key.")
        if "rate" in err or "quota" in err:
            return (None, "")
        return (None, "")


def _get_openai_response(history: list, api_key: str) -> tuple[str | None, str]:
    """Use OpenAI. Returns (response, error_msg)."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=history,
            max_tokens=500,
            temperature=0.7,
        )
        return (response.choices[0].message.content, "")
    except Exception as e:
        err = str(e).lower()
        if "api_key" in err or "authentication" in err:
            return (None, "Invalid OpenAI API key. Check OPENAI_API_KEY.")
        if "rate" in err or "quota" in err:
            return (None, "OpenAI limit reached. Try again in a moment.")
        return (None, "Sorry, I couldn't process that. Please try again.")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({
            "response": "Please type a message.",
            "is_crisis": False,
            "session_id": session_id,
        })

    # Crisis detection - always highest priority, bypasses AI
    if is_crisis_message(message):
        response, is_crisis = get_crisis_response()
        return jsonify({
            "response": response,
            "is_crisis": is_crisis,
            "session_id": session_id,
        })

    response, is_crisis = get_ai_response(message, session_id)
    return jsonify({
        "response": response,
        "is_crisis": is_crisis,
        "session_id": session_id,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Mental Health Chatbot running at: http://127.0.0.1:{port}")
    print(f"  Open this link in your browser.\n")
    app.run(debug=True, host="127.0.0.1", port=port)
