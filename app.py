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

SYSTEM_PROMPT = """You are a warm, empathetic mental health support assistant. Your role is to:

- Listen actively and validate feelings without judgment
- Offer emotional support and evidence-based coping strategies
- Encourage professional help when appropriate (therapist, counselor, doctor)
- Use a calm, reassuring tone
- Keep responses concise but supportive (2-4 short paragraphs max)
- Never diagnose conditions or prescribe treatments
- Never replace professional mental health care

Important boundaries:
- You are NOT a licensed therapist, psychiatrist, or medical professional
- Always recommend speaking with a professional for ongoing or severe concerns
- If someone mentions crisis, self-harm, or suicide, you will be interrupted—emergency resources will be shown separately
- Be supportive but never make promises you cannot keep

Respond naturally, like a caring friend who wants to help."""


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


def get_ai_response(user_message: str, session_id: str) -> tuple[str, bool]:
    """Get AI response from Gemini (free) or OpenAI. Returns (response_text, is_crisis)."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_key and not openai_key:
        return (
            "AI is not configured yet. Add GEMINI_API_KEY (free at aistudio.google.com/apikey) "
            "or OPENAI_API_KEY in Render → Environment, then redeploy.",
            False,
        )

    # Get or create conversation history
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

    # Try Gemini first (free), then OpenAI
    if gemini_key:
        result = _get_gemini_response(history, gemini_key)
    else:
        result = _get_openai_response(history, openai_key)

    if result[0] is None:
        return (result[1], False)  # error message

    assistant_message = result[0]
    history.append({"role": "assistant", "content": assistant_message})

    if len(history) > 21:
        conversations[session_id] = [history[0]] + history[-20:]

    return (assistant_message, False)


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
