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
    "suicide", "suicidal", "sucide", "sucid",  # include common typos
    "kill myself", "end my life", "want to die", "end myself", "end it all",
    "self-harm", "self harm", "cutting", "hurt myself",
    "overdose", "take pills", "eat pills", "eat medicine", "100 medicine",
    "take all my medicine", "swallow pills",
]

SYSTEM_PROMPT = """You are a warm, motivating mental health support assistant—like a caring friend who helps people feel better and move forward.

CRITICAL: Be SPECIFIC to what they shared. Never give the same generic response twice. Use the conversation context.

CORE RULES:
- Validate first, then motivate: "I hear you" → "You can get through this" → gentle next step
- Reference what they said: "You mentioned your dress broke..." or "You said you don't trust anyone..."
- Offer hope + small actionable steps
- Use "you" directly. Warm, human, encouraging

SCENARIOS (respond specifically):
- Dress/clothes broke: "I'm sorry your dress broke—that's frustrating. Can you fix it or find an alternative? Remember, a dress doesn't define you."
- Don't trust anyone: "Not being able to trust can feel lonely. It's understandable if you've been hurt. Trust can be rebuilt slowly. A therapist can help."
- Don't like anyone / what's wrong with me: "Feeling disconnected doesn't mean something is wrong with you. Sometimes we need time to heal. Be gentle with yourself."
- "What should I do?": Use the conversation! If they said they don't trust people, give advice about trust. If they said dress broke, give practical + emotional support.
- Stress + specific cause: Address the cause (dress, work, etc.) then offer coping steps.

When they ask about YOU: "I'm here for you. How are you really doing? I'm listening."

Never: generic "Thank you for sharing" without addressing their specific situation. Always: reference what they said, give relevant advice."""


def is_crisis_message(text: str) -> bool:
    """Check if message contains crisis keywords."""
    msg_lower = text.lower().strip()
    if any(kw in msg_lower for kw in CRISIS_KEYWORDS):
        return True
    # Overdose: "100" or "all" + medicine/pills
    if ("medicine" in msg_lower or "pills" in msg_lower) and ("100" in msg_lower or "all" in msg_lower or "many" in msg_lower):
        return True
    return False


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


def _get_recent_user_messages(history: list) -> list[str]:
    """Extract recent user messages from conversation for context."""
    return [m["content"].lower() for m in history if m.get("role") == "user"][-5:]


def get_fallback_response(user_message: str, recent_context: list[str] | None = None) -> str:
    """Context-aware fallback when AI fails—specific, logical responses."""
    msg = user_message.lower()
    context = " ".join(recent_context or [])
    full_context = f"{context} {msg}"

    # Follow-up questions - use conversation context for relevant advice
    if any(w in msg for w in ["what should i do", "what can i do", "what do you think", "what would you do", "any advice", "help me"]):
        if any(w in full_context for w in ["trust", "don't trust", "trust anyone"]):
            return (
                "Based on what you've shared about trust—it's understandable if you've been hurt before. "
                "Trust can be rebuilt slowly. Start with one small step: maybe share something small with someone you feel safest with. "
                "You don't have to trust everyone. A therapist can also help you work through this in a safe space. "
                "You're not alone in feeling this way. What feels like the hardest part right now?"
            )
        if any(w in full_context for w in ["don't like", "don't like anyone", "whats wrong", "wrong with me"]):
            return (
                "Feeling disconnected from people doesn't mean something is wrong with you. "
                "Sometimes we go through phases, or we need time to heal from past hurts. "
                "A few things that might help: be gentle with yourself, try one small social step (even a short chat), "
                "or talk to a therapist to explore what's going on. You don't have to figure it all out today. "
                "What would feel manageable right now?"
            )
        if any(w in full_context for w in ["dress", "broke", "broken", "stress"]):
            return (
                "When small things like a broken dress add to our stress, it can feel overwhelming. "
                "First: it's okay to feel frustrated. Second: can you fix it, borrow something, or get a replacement? "
                "Third: remember—a dress is just a thing. You matter more. Take a breath. "
                "What's one small step you can take right now to feel a bit better?"
            )
        return (
            "Here's what might help: take one small step at a time. Be gentle with yourself. "
            "Talking to someone you trust—a friend, family member, or therapist—can make a big difference. "
            "You don't have to have all the answers. What feels most doable for you right now?"
        )

    # Trust issues
    if any(w in msg for w in ["don't trust", "dont trust", "can't trust", "cant trust", "trust anyone", "trust nobody", "dont trust anyone", "don't trust anyone"]):
        return (
            "Not being able to trust people can feel really lonely and isolating. "
            "It's understandable if you've been hurt before—that can make it hard to open up. "
            "Trust can be built slowly, one small step at a time. You don't have to trust everyone. "
            "Would you like to talk about what happened? I'm here, and I'm not going anywhere. "
            "A therapist can also help you work through this in a safe way."
        )

    # Don't like anyone / disconnected / what's wrong with me
    if any(w in msg for w in ["don't like anyone", "dont like anyone", "dont like", "whats wrong w me", "wrong with me", "idk whats wrong", "something wrong with me"]):
        return (
            "Feeling disconnected from people can be confusing and lonely. "
            "It doesn't mean something is wrong with you—sometimes we go through phases, or we need time to heal. "
            "It could be depression, past hurt, or just needing a break. You don't have to have it all figured out. "
            "Talking to a therapist can help you understand what's going on. "
            "For now: be kind to yourself. You're not broken. Would you like to talk more about what you're feeling?"
        )

    # Things broke / practical problems (dress, etc.)
    if any(w in msg for w in ["broke", "broken", "tore", "ripped", "ruined"]):
        if "dress" in msg or "clothes" in msg or "outfit" in msg:
            return (
                "I'm sorry your dress broke—that's frustrating, especially if it mattered to you or you had plans. "
                "It's okay to feel upset. These things happen. Can you fix it, borrow something, or find an alternative? "
                "And remember—a dress doesn't define you. You're more than what you wear. "
                "How are you holding up? Is there something else adding to the stress?"
            )
        return (
            "I'm sorry that happened—it's frustrating when things break. "
            "It's okay to feel upset. Can you fix it or find a workaround? "
            "Sometimes small things feel big when we're already stressed. How are you doing?"
        )

    # Bad person / guilt / shame
    if any(w in msg for w in ["bad person", "am i bad", "am i evil", "guilty", "guilt", "shame", "did something wrong", "terrible person"]):
        return (
            "Feeling guilty doesn't make you a bad person—it means you care. Everyone makes mistakes. "
            "What matters is that you're trying to be better. You're not defined by one moment or one choice. "
            "I believe in your ability to grow and move forward. Would you like to talk about what's weighing on you? "
            "I'm here, no judgment."
        )

    # Upset (general)
    if any(w in msg for w in ["upset", "unhappy", "not okay", "not ok", "feeling bad", "feel bad"]):
        return (
            "I'm sorry you're feeling upset. It's okay to feel this way—your feelings are valid. "
            "Sometimes it helps to take a breath, or to talk about what's going on. "
            "You don't have to figure it all out right now. What would feel most helpful—talking, or a small step to feel a bit better? "
            "I'm here for you."
        )

    # Meta questions about the chatbot
    if any(w in msg for w in ["are you ", "you a bot", "you a chatbot", "happy to be", "who are you", "what are you"]):
        return (
            "I'm here for you—that's what matters most. I'm a support chatbot, and I care about how you're doing. "
            "How are you really feeling today? I'm listening, and I want to help. "
            "You can share anything—I'm not here to judge, just to support."
        )

    # Body image / self-worth / feeling ugly
    if any(w in msg for w in ["ugly", "unattractive", "unpretty", "hideous", "look bad", "hate how i look", "feel ugly"]):
        return (
            "I want you to know that beauty is so much more than what we see in the mirror. "
            "It's your kindness, your strength, the way you care about others, the things that make you *you*. "
            "Everyone has their own unique beauty—including you. You have value that no one can take away. "
            "I see you, and you matter. Would you like to talk more about what's been on your mind?"
        )

    # Emotional breakdown / overwhelmed
    if any(w in msg for w in ["breakdown", "breaking down", "falling apart", "can't take it", "can't cope", "losing it"]):
        return (
            "It's okay to fall apart sometimes. You're human—and what you're feeling is valid. "
            "You don't have to hold it all together right now. Take a breath. I'm here with you. "
            "When things feel too heavy, it helps to talk. Would you like to share what's going on? "
            "No judgment—just support."
        )

    # Lost something (phone, keys, etc.)
    if any(w in msg for w in ["lost my", "lost the", "can't find", "misplaced", "stolen"]):
        return (
            "Losing something important is really frustrating—I get it. It's okay to feel upset or stressed. "
            "These things happen to everyone. Is there someone who can help you look, or help you report it if needed? "
            "And remember—things can be replaced. You matter more. How are you holding up?"
        )

    # Sadness / depression
    if any(w in msg for w in ["sad", "down", "depressed", "hopeless", "miserable", "empty"]):
        return (
            "I'm really sorry you're feeling this way. What you're going through sounds hard, "
            "and it takes courage to reach out. Remember: you don't have to face this alone, and this feeling won't last forever. "
            "Talking to someone—a friend, family member, or therapist—can make a real difference. "
            "You've already taken a step by being here. Would you like to share more? I'm here to listen and help you move forward."
        )

    # Anxiety
    if any(w in msg for w in ["anxious", "anxiety", "worried", "nervous", "panic", "scared"]):
        return (
            "Anxiety can feel overwhelming—I hear you. Try taking a few slow breaths: "
            "breathe in for 4 counts, hold for 4, breathe out for 6. "
            "Grounding can help too: name 5 things you can see, 4 you can hear, 3 you can touch. "
            "If anxiety is affecting your daily life, a therapist can offer tools that really help. "
            "What's been weighing on you lately?"
        )

    # Stress / overwhelm (check for specific cause like dress)
    if any(w in msg for w in ["stress", "stressed", "overwhelmed", "pressure", "burnout", "too much"]):
        if "dress" in msg or "broke" in msg or "broken" in msg:
            return (
                "I'm sorry—when something like a broken dress adds to your stress, it can feel like a lot. "
                "It's okay to feel overwhelmed. First: take a breath. Second: can you fix the dress, borrow something, or find another option? "
                "Third: remember that small setbacks don't define your day. You're doing your best. "
                "What would feel most helpful right now—practical fix, or taking a moment to decompress?"
            )
        return (
            "Feeling overwhelmed is exhausting, and it's okay to admit that. "
            "Try breaking things into smaller steps—even one small thing at a time helps. "
            "Short breaks, a walk, or talking to someone you trust can lighten the load. "
            "You're doing your best, and that matters. What would feel most helpful right now?"
        )

    # Self-esteem / not good enough / stupid
    if any(w in msg for w in ["not good enough", "worthless", "useless", "failure", "stupid", "dumb", "can't do anything", "am i stupid"]):
        return (
            "You are enough. You don't have to be perfect to deserve kindness—including from yourself. "
            "We all have moments of doubt. That doesn't define you. I believe in you. "
            "What's one small thing you're proud of, even if it feels tiny? Sometimes that helps us see our strength. "
            "You can move past this. I'm here for you."
        )

    # Loneliness
    if any(w in msg for w in ["lonely", "alone", "no friends", "isolated", "left out"]):
        return (
            "Feeling alone is really painful—I'm sorry you're going through that. "
            "You're not broken for feeling this way. Reaching out—even here—takes courage, and I'm glad you did. "
            "Connection can start small: a text to someone, a walk in a busy place, or even just being here. "
            "You matter. Would you like to talk more?"
        )

    # Relationship / breakup / fight
    if any(w in msg for w in ["breakup", "broke up", "relationship", "fight", "argument", "ex", "boyfriend", "girlfriend"]):
        return (
            "Relationships can be really hard—whether it's a breakup, a fight, or something else. "
            "Your feelings are valid. It's okay to hurt. Would you like to talk about what happened? "
            "I'm here to listen, no judgment. Sometimes just sharing can help."
        )

    # Sleep issues
    if any(w in msg for w in ["can't sleep", "insomnia", "tired", "exhausted", "no sleep"]):
        return (
            "Not being able to sleep is exhausting and frustrating. I hear you. "
            "Try a wind-down routine: dim lights, no screens 30 min before bed, maybe some gentle music. "
            "If sleep problems persist, a doctor or therapist can help. "
            "How long has this been going on? I'm here to listen."
        )

    # Anger / frustration
    if any(w in msg for w in ["angry", "mad", "frustrated", "annoyed", "pissed"]):
        return (
            "It's okay to feel angry or frustrated—those feelings are valid. "
            "Sometimes we need to let it out. Try taking a few deep breaths, or stepping away for a moment. "
            "What's been bothering you? I'm here to listen."
        )

    # Greetings
    if any(w in msg for w in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        return (
            "Hi there! I'm here to listen and support you. "
            "You can share how you're feeling—stress, anxiety, sadness, or anything else. "
            "How are you doing today?"
        )

    # Thanks
    if any(w in msg for w in ["thank", "thanks", "appreciate"]):
        return (
            "You're welcome. I'm really glad I could be here for you. "
            "Remember, it's okay to reach out whenever you need support. Take care of yourself. 💙"
        )

    # Default - warm, motivating, inviting
    return (
        "Thank you for sharing. I'm here to listen and support you. "
        "Whatever you're going through—stress, anxiety, sadness, or how you feel about yourself—"
        "you don't have to face it alone. What would you like to talk about? I'm here to help you move forward."
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

    # All AI failed—use context-aware fallback
    recent = _get_recent_user_messages(history)
    return (get_fallback_response(user_message, recent), False)


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
