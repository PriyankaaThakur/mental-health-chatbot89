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
    """Get AI response from OpenAI. Returns (response_text, is_crisis)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "AI support is not configured. Please set OPENAI_API_KEY in your environment. "
            "You can get an API key from https://platform.openai.com/api-keys",
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

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=history,
            max_tokens=500,
            temperature=0.7,
        )
        assistant_message = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_message})

        # Limit history to last 20 messages (10 exchanges) to avoid token limits
        if len(history) > 21:
            conversations[session_id] = [history[0]] + history[-20:]

        return (assistant_message, False)

    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "authentication" in error_msg:
            return ("Invalid or missing API key. Please check your OPENAI_API_KEY.", False)
        if "rate" in error_msg or "quota" in error_msg:
            return ("The service is temporarily busy. Please try again in a moment.", False)
        return (
            "I'm sorry, I couldn't process that right now. Please try again in a moment.",
            False,
        )


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
