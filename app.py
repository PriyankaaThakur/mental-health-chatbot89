"""
Mental Health Support Chatbot - AI-Powered Flask Backend
RAG (Retrieval-Augmented Generation): each reply augments the system prompt with matching
snippets from data/rag_knowledge.json for more accurate, grounded answers (see rag.py).
Crisis detection remains rule-based. No persistent data storage.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

import copy
import re
import uuid
from flask import Flask, render_template, request, jsonify

from rag import augment_system_prompt

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
    "sud=cide", "sudcide",  # obfuscated / typo variants
]

SYSTEM_PROMPT = """You are a warm, motivating mental health support assistant—like a caring friend who helps people feel better and move forward.

CRITICAL: Be SPECIFIC to what they shared. Never give the same generic response twice. Use the conversation context.

CORE RULES:
- Validate first, then motivate: "I hear you" → "You can get through this" → gentle next step
- Reference what they said: "You mentioned your dress broke..." or "You said you don't trust anyone..."
- Offer hope + small actionable steps
- Use "you" directly. Warm, human, encouraging

SCENARIOS (respond specifically, give CONCRETE STEPS when they ask for solutions):
- Cheating/infidelity: Validate betrayal, affirm their worth. "You deserve someone who chooses you."
- Breakup / "how do I get over": Give 4-5 concrete steps: allow feelings, limit contact, lean on people, focus on self, time. Relate.org.uk.
- "What should I do?" / "give me solution" / "how to move on": ALWAYS use conversation context. If breakup/cheating → breakup recovery steps. If sad → coping steps. Never generic.
- "Broke my head" (idiom): Stressing over relationship. Offer steps + mention physical safety if abuse.
- Stress + specific cause: Address the cause then offer coping steps.

When they ask about YOU: "I'm here for you. How are you really doing? I'm listening."

When suggesting helplines or services, use UK resources: Samaritans 116 123, Shout 85258, Mind 0300 123 3393, Beat (eating disorders) 0808 801 0677, Cruse (bereavement) 0808 808 1677.

When "Retrieved reference knowledge" appears in your instructions, use it to ground facts (helplines, coping ideas, when to seek professional help)—blend it naturally into your reply, do not read it as a rigid script.

Never: generic "Thank you for sharing" without addressing their specific situation. Always: reference what they said, give relevant advice."""


def is_crisis_message(text: str) -> bool:
    """Check if message contains crisis keywords (including typos / obfuscated spellings)."""
    msg_lower = text.lower().strip()
    if any(kw in msg_lower for kw in CRISIS_KEYWORDS):
        return True
    # Letters-only string catches "sud=cide", "su!cide", spacing tricks
    letters_only = re.sub(r"[^a-z]", "", msg_lower)
    if any(
        s in letters_only
        for s in ("suicide", "suicid", "sucide", "sucid", "sudcide", "killmyself", "endmylife")
    ):
        return True
    # Overdose: "100" or "all" + medicine/pills
    if ("medicine" in msg_lower or "medicines" in msg_lower or "pills" in msg_lower) and ("100" in msg_lower or "all" in msg_lower or "many" in msg_lower):
        return True
    return False


def get_crisis_response() -> tuple[str, bool]:
    """Return emergency resources (UK). Second value indicates is_crisis."""
    return (
        "I'm really concerned about what you're sharing. "
        "Please reach out for immediate support:\n\n"
        "• Samaritans: 116 123 (24/7, free)\n"
        "• Shout: Text SHOUT to 85258 (24/7, free)\n"
        "• Papyrus HopelineUK: 0800 068 4141 (under 35, 9am–midnight)\n"
        "• CALM: 0800 58 58 58 (5pm–midnight)\n"
        "• Mind: 0300 123 3393 (Mon–Fri 9am–6pm)\n"
        "• Emergency: 999 or NHS 111\n\n"
        "You don't have to face this alone. These services are free and confidential.",
        True,
    )


def _history_with_rag(history: list) -> list:
    """Copy history with RAG-augmented system message for this turn."""
    h = copy.deepcopy(history)
    if h and h[0].get("role") == "system":
        last_user = h[-1]["content"] if h[-1].get("role") == "user" else ""
        h[0] = {"role": "system", "content": augment_system_prompt(SYSTEM_PROMPT, last_user)}
    return h


def _get_recent_user_messages(history: list) -> list[str]:
    """Extract recent user messages from conversation for context."""
    return [m["content"].lower() for m in history if m.get("role") == "user"][-5:]


def _is_greeting_only(user_message: str) -> bool:
    """True only for short standalone greetings—not 'Hi I am sad'."""
    s = user_message.strip().lower()
    if len(s) > 80:
        return False
    # Must not look like a full sentence about feelings
    if any(w in s for w in ["sad", "hurt", "hate", "feel", "anxious", "depress", "suic", "kill", "die", "ugly", "alone", "stress"]):
        return False
    return bool(
        re.match(r"^(hi|hello|hey)\s*[!?.]*$", s)
        or re.match(r"^(good\s+(morning|afternoon|evening))[\s!?.]*$", s)
        or re.match(r"^(hi|hello|hey)\s+(there|again)\s*[!?.]*$", s)
    )


def get_fallback_response(user_message: str, recent_context: list[str] | None = None) -> str:
    """Context-aware fallback when AI fails—specific, logical responses."""
    msg = user_message.lower()
    context = " ".join(recent_context or [])
    full_context = f"{context} {msg}"

    # Self-hatred / disgust / "I hate myself" (high priority—avoid generic default)
    if any(w in msg for w in ["hate myself", "hate me", "hating myself", "loathe myself", "wish i was dead", "want to disappear"]):
        return (
            "I'm really sorry you're hurting this much. What you're feeling is heavy—and it doesn't mean you're a bad person. "
            "Many people have moments of intense self-criticism; it can be linked to depression or past hurt. "
            "You deserve support: talking to Samaritans (116 123), your GP, or a therapist can help. "
            "Right now: try naming one small thing that isn't terrible about today, even if it feels tiny. "
            "I'm here with you. You matter."
        )

    if any(w in msg for w in ["disgusting", "revolting", "gross", "feel disgusting", "feeling disgusting", "i'm disgusting", "im disgusting"]):
        return (
            "Feeling disgusting or repelled by yourself is painful—and it's a feeling, not the truth about who you are. "
            "Your worth isn't defined by how you feel in your worst moments. "
            "If body image is part of this, remember many people struggle; Beat and Mind have resources. "
            "Would you like to say more about what's behind this feeling? I'm listening."
        )

    # Hate everyone / misanthropy / anger at people
    if any(w in msg for w in ["hate everyone", "hate people", "hate all people", "hate everybody", "everyone is awful", "i hate humans"]):
        return (
            "Feeling like you hate everyone can come from hurt, burnout, or feeling let down again and again. "
            "Your anger makes sense if you've been wounded. It doesn't make you a bad person. "
            "Sometimes talking to a therapist helps unpack where this is coming from. "
            "Is there one situation or person that started this feeling? I'm here to listen."
        )

    # Follow-up questions - use conversation context for relevant advice and SOLUTIONS
    if any(w in msg for w in [
        "what should i do", "what can i do", "what do you think", "what would you do",
        "any advice", "help me", "how do i get over", "how can i get over", "get over from",
        "give me solution", "give me advice", "how to move on", "how can i move on",
        "tell me how", "can you tell me how", "solution", "what to do"
    ]):
        # Breakup / cheating context - give concrete steps to move on
        if any(w in full_context for w in ["boyfriend", "girlfriend", "cheating", "cheat", "breakup", "broke up", "into another", "another girl", "another guy", "worthless", "ugly"]):
            return (
                "I hear you—getting over a breakup, especially when there's been cheating, is really hard. Here are some steps that help:\n\n"
                "1. **Allow yourself to feel** – Anger, sadness, and hurt are valid. Don't rush to 'get over it'.\n"
                "2. **Limit contact** – Unfollow, mute, or block if it helps. Out of sight can help with healing.\n"
                "3. **Lean on people** – Talk to a friend or family member. You don't have to do this alone.\n"
                "4. **Focus on you** – One small thing you enjoy: a walk, a film, a hobby you've neglected.\n"
                "5. **Time** – Healing isn't linear. Some days will be harder. That's okay.\n\n"
                "You deserve someone who chooses you. His actions say nothing about your worth. "
                "Relate (relate.org.uk) offers relationship counselling in the UK. Would you like to talk more?"
            )
        if any(w in full_context for w in ["trust", "don't trust", "trust anyone"]):
            return (
                "Based on what you've shared about trust—it's understandable if you've been hurt before. "
                "**Steps that help:** Start with one small step—share something small with someone you feel safest with. "
                "You don't have to trust everyone. A therapist can help you work through this in a safe space. "
                "Mind (mind.org.uk) has resources: 0300 123 3393. What feels like the hardest part right now?"
            )
        if any(w in full_context for w in ["don't like", "don't like anyone", "whats wrong", "wrong with me"]):
            return (
                "Feeling disconnected doesn't mean something is wrong with you. "
                "**What might help:** Be gentle with yourself. Try one small social step—even a short chat with someone. "
                "Talking to a therapist can help you understand what's going on. "
                "You don't have to figure it all out today. What would feel manageable right now?"
            )
        if any(w in full_context for w in ["sad", "depressed", "down", "lonely", "alone"]):
            return (
                "When you're feeling low, small steps matter:\n\n"
                "1. **Reach out** – Text one person. You don't have to explain everything.\n"
                "2. **Get outside** – Even 10 minutes of fresh air can help.\n"
                "3. **Routine** – One small thing at the same time each day (e.g. morning stretch).\n"
                "4. **Talk to a GP** – If this has lasted weeks, they can help or refer you.\n\n"
                "Samaritans (116 123) are there 24/7 if you need to talk. You're not alone."
            )
        if any(w in full_context for w in ["dress", "broke", "broken", "stress"]):
            return (
                "When small things add to our stress: (1) Take a breath. (2) Can you fix it, borrow something, or find an alternative? "
                "(3) Remember—a dress is just a thing. You matter more. What's one small step you can take right now?"
            )
        return (
            "Here's what might help: take one small step at a time. Be gentle with yourself. "
            "Talking to someone you trust—a friend, family member, or therapist—can make a big difference. "
            "If you'd like to share what's going on, I can give more specific advice. What feels most doable for you right now?"
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

    # Things broke / practical problems (dress, etc.) - exclude "broke my head"
    if any(w in msg for w in ["broke", "broken", "tore", "ripped", "ruined"]) and "head" not in msg:
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

    # OCD / intrusive thoughts
    if any(w in msg for w in ["ocd", "intrusive thoughts", "cant stop thinking", "can't stop thinking", "repetitive thoughts", "obsessive", "compulsive", "unwanted thoughts"]):
        return (
            "Intrusive or repetitive thoughts can be really distressing—I hear you. "
            "You're not alone in experiencing this. OCD and similar struggles are treatable. "
            "A therapist, especially one trained in CBT, can help you learn to manage these thoughts. "
            "In the moment: try grounding (5 things you see, 4 you hear, 3 you touch). "
            "Be gentle with yourself. Would you like to talk more about what you're experiencing?"
        )

    # PTSD / trauma
    if any(w in msg for w in ["ptsd", "trauma", "traumatised", "traumatized", "flashback", "flashbacks", "nightmares", "triggered", "past abuse", "abused"]):
        return (
            "Living with trauma can be exhausting and overwhelming. What you've been through matters, "
            "and your reactions make sense. Healing takes time. A trauma-informed therapist can help you "
            "work through this in a safe way. For now: you're safe in this moment. Grounding can help—"
            "name 5 things you can see, 4 you can hear, 3 you can touch. "
            "You don't have to face this alone. Would you like to share more?"
        )

    # Eating disorders
    if any(w in msg for w in ["eating disorder", "anorexia", "bulimia", "binge eating", "not eating", "starving myself", "purge", "body image", "weight obsession"]):
        return (
            "Struggling with food and body image is really hard—and you deserve support. "
            "Eating disorders are serious but treatable. Beat (beateatingdisorders.org.uk) offers UK support: 0808 801 0677. "
            "A GP or eating disorder specialist can help. You're not alone in this. "
            "Would you like to talk about what you're going through?"
        )

    # Grief / bereavement
    if any(w in msg for w in ["grief", "grieving", "bereavement", "lost someone", "someone died", "death of", "passed away", "mourning"]):
        return (
            "Grief is one of the hardest things we go through. There's no right way to feel—"
            "sadness, anger, numbness, confusion—all of it is valid. "
            "Cruse Bereavement Care (cruse.org.uk) offers UK support: 0808 808 1677. "
            "Be gentle with yourself. Healing doesn't happen overnight. "
            "Would you like to talk about the person you've lost?"
        )

    # Social anxiety
    if any(w in msg for w in ["social anxiety", "socially anxious", "awkward around people", "fear of people", "scared of people", "cant talk to people", "can't talk to people"]):
        return (
            "Social anxiety can feel really isolating—I hear you. It's more common than people think. "
            "Small steps help: maybe start with one short conversation, or a small group. "
            "A therapist can teach you CBT techniques that really work for social anxiety. "
            "You're not weird or broken. Would you like to talk about what situations feel hardest?"
        )

    # Phobias / specific fears
    if any(w in msg for w in ["phobia", "phobic", "terrified of", "scared of", "fear of", "afraid of"]):
        return (
            "Phobias can feel overwhelming—your fear is real and valid. "
            "The good news: they're very treatable. Exposure therapy and CBT can help. "
            "A therapist can work with you gradually. You don't have to face this alone. "
            "What's been triggering you?"
        )

    # Burnout / exhaustion
    if any(w in msg for w in ["burnout", "burned out", "burnt out", "exhausted", "drained", "no energy", "no motivation"]):
        return (
            "Burnout can leave you feeling empty and exhausted—I hear you. "
            "Rest isn't selfish. Your body and mind are telling you they need a break. "
            "Even small steps help: a short walk, saying no to one thing, talking to someone. "
            "If work or study is the cause, consider speaking to a GP or occupational health. "
            "You deserve to feel better. What would feel most manageable right now?"
        )

    # Imposter syndrome / fear of failure
    if any(w in msg for w in ["imposter", "impostor", "fraud", "don't deserve", "fear of failure", "failing", "will fail"]):
        return (
            "Imposter syndrome is so common—even people who seem confident feel it. "
            "You're not a fraud. You've gotten where you are for a reason. "
            "Try writing down one thing you've done well recently. Sometimes we're our own harshest critic. "
            "A therapist can help you challenge these thoughts. You're capable. I believe in you."
        )

    # Perfectionism
    if any(w in msg for w in ["perfectionist", "perfectionism", "must be perfect", "anything less than perfect"]):
        return (
            "Perfectionism can be exhausting—it's hard to feel like nothing is ever good enough. "
            "Done is often better than perfect. Try allowing yourself one 'good enough' today. "
            "Your worth isn't tied to your output. A therapist can help you ease these standards. "
            "Be kind to yourself. You're doing your best."
        )

    # Family issues
    if any(w in msg for w in ["family", "parents", "mother", "father", "mum", "dad", "sibling", "brother", "sister", "family conflict"]):
        return (
            "Family issues can be really painful—they're the people we're supposed to feel closest to. "
            "Your feelings are valid. Family conflict doesn't mean you're a bad person. "
            "Relate (relate.org.uk) offers UK relationship and family counselling. "
            "Talking to a therapist can help you understand and cope. Would you like to share more?"
        )

    # Bullying
    if any(w in msg for w in ["bullied", "bullying", "bully", "picked on", "harassed"]):
        return (
            "Being bullied is never your fault. It's painful and isolating—I'm sorry you're going through that. "
            "You deserve to feel safe. If it's at school or work, consider telling someone in authority. "
            "Bullying UK and Mind have resources. You're not alone. "
            "Would you like to talk about what's been happening?"
        )

    # Health anxiety
    if any(w in msg for w in ["health anxiety", "hypochondria", "hypochondriac", "worried about my health", "think i'm sick", "convinced i have"]):
        return (
            "Health anxiety can be really distressing—constantly worrying about illness is exhausting. "
            "Your feelings are valid. A GP can help rule out health concerns and refer you if needed. "
            "CBT is often effective for health anxiety. Try limiting how often you search symptoms online. "
            "You're not alone in this. Would you like to talk more?"
        )

    # Numbness / detachment
    if any(w in msg for w in ["numb", "numbness", "detached", "disconnected", "feel nothing", "empty inside", "going through motions"]):
        return (
            "Feeling numb or disconnected can be scary—like you're watching life from the outside. "
            "It's often a way our mind protects us when things feel too much. "
            "It's not permanent. Talking to a therapist can help you understand and reconnect. "
            "Be gentle with yourself. Would you like to talk about what might have led to this?"
        )

    # Sadness / depression
    if any(w in msg for w in ["sad", "down", "depressed", "hopeless", "miserable", "empty", "low mood", "feeling low"]):
        return (
            "I'm really sorry you're feeling this way. What you're going through sounds hard, "
            "and it takes courage to reach out. Remember: you don't have to face this alone, and this feeling won't last forever. "
            "Talking to someone—a friend, family member, or therapist—can make a real difference. "
            "You've already taken a step by being here. Would you like to share more? I'm here to listen and help you move forward."
        )

    # Panic attacks (more specific)
    if any(w in msg for w in ["panic attack", "panic attacks", "had a panic", "having a panic"]):
        return (
            "Panic attacks can feel terrifying—like you're losing control. You're not. "
            "Try: breathe in slowly for 4, hold for 4, breathe out for 6. Repeat. "
            "Grounding: name 5 things you see, 4 you hear, 3 you can touch. "
            "It will pass. You're safe. If panic attacks are frequent, a GP or therapist can help. "
            "How are you feeling now?"
        )

    # Work / school stress
    if any(w in msg for w in ["work stress", "job stress", "boss", "colleague", "exam", "exams", "deadline", "deadlines", "assignment", "university", "college"]):
        return (
            "Work or study stress can be really draining. It's okay to feel overwhelmed. "
            "Break things into smaller steps. Talk to a tutor, manager, or HR if things feel unmanageable. "
            "Your wellbeing matters more than any grade or project. "
            "What's feeling most urgent right now?"
        )

    # ADHD
    if any(w in msg for w in ["adhd", "add", "focus", "concentration", "can't focus", "cant concentrate", "distracted", "easily distracted"]):
        return (
            "Struggling with focus or concentration can be really frustrating. "
            "It doesn't mean you're lazy—many people have ADHD or similar challenges. "
            "A GP can refer you for assessment. In the meantime: break tasks into tiny steps, use timers, "
            "and be kind to yourself. You're not broken. Would you like to talk more?"
        )

    # Jealousy / comparison
    if any(w in msg for w in ["jealous", "jealousy", "comparing myself", "compare myself", "everyone else is", "others have it better"]):
        return (
            "Comparing yourself to others can be really painful—it's human, but it's not helpful. "
            "Everyone's journey is different. What you see of others is often a highlight reel. "
            "Try focusing on one small thing you're grateful for or proud of. "
            "You're on your own path. Be kind to yourself."
        )

    # Rejection / insecurity
    if any(w in msg for w in ["rejected", "rejection", "insecure", "insecurity", "not good enough for"]):
        return (
            "Rejection and insecurity hurt—I hear you. Your worth isn't defined by one person's opinion. "
            "It's okay to feel upset. Be gentle with yourself. "
            "Talking to someone you trust, or a therapist, can help. "
            "You matter. Would you like to talk more about what happened?"
        )

    # Low motivation
    if any(w in msg for w in ["no motivation", "unmotivated", "cant get out of bed", "can't get out of bed", "procrastinating", "procrastination"]):
        return (
            "Low motivation can be a sign of depression or burnout—it's not laziness. "
            "Be gentle with yourself. Even one small step counts: get out of bed, take a shower, go outside for 5 minutes. "
            "If this has lasted a while, a GP can help. You're not alone in feeling this way. "
            "What's one tiny thing you could do today?"
        )

    # Anxiety
    if any(w in msg for w in ["anxious", "anxiety", "worried", "nervous", "panic", "scared", "frightened", "afraid"]):
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

    # Cheating / infidelity
    if any(w in msg for w in ["cheating", "cheat", "cheated", "into another girl", "into another guy", "another woman", "another man", "he is into", "she is into"]):
        return (
            "I'm so sorry—being cheated on is a huge betrayal. Your pain is valid. You deserve someone who chooses you. "
            "It's okay to feel angry, hurt, or confused. You're not overreacting. "
            "**What might help:** Talk to someone you trust. Consider taking space from him to think clearly. "
            "You don't have to decide anything today. Relate (relate.org.uk) offers relationship counselling. "
            "Would you like to talk about how you're feeling? I'm here."
        )

    # "Broke my head" - idiom for stressing/worrying over someone (or literal head injury)
    if ("broke" in msg and "head" in msg) or ("breaking" in msg and "head" in msg):
        if any(w in full_context for w in ["boyfriend", "girlfriend", "relationship", "cheat"]):
            return (
                "It sounds like you've been stressing yourself sick over this relationship—that's exhausting. "
                "When someone hurts us, it can take over our thoughts. You deserve peace. "
                "**Steps that help:** Limit how much you think about him. Focus on one thing you can control today. "
                "Talk to a friend. Consider therapy if the pain feels too heavy. "
                "If you meant a physical head injury from him, please reach out to someone safe—you deserve to be protected. "
                "How are you holding up?"
            )
        return (
            "Stressing or worrying can feel overwhelming—like your head is full. "
            "Try: take a breath, step away from screens, or talk to someone. "
            "If you're in physical pain—please see a doctor or call 111. "
            "How can I help?"
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

    # Irritability / anger
    if any(w in msg for w in ["angry", "mad", "frustrated", "annoyed", "pissed", "irritable", "irritated", "short temper", "losing my temper"]):
        return (
            "It's okay to feel angry or frustrated—those feelings are valid. "
            "Sometimes we need to let it out. Try taking a few deep breaths, or stepping away for a moment. "
            "What's been bothering you? I'm here to listen."
        )

    # Standalone greetings only (not "Hi I am sad")
    if _is_greeting_only(user_message):
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
    """Get AI response. Tries: Local LLM → Gemini → Groq → OpenAI. Falls back to empathetic response if all fail."""
    local_url = os.environ.get("LOCAL_LLM_URL", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not local_url and not gemini_key and not groq_key and not openai_key:
        return (
            "AI is not configured. Add one of: LOCAL_LLM_URL (Ollama/LM Studio), GEMINI_API_KEY (free at aistudio.google.com/apikey), "
            "or GROQ_API_KEY (free at console.groq.com) in your environment.",
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

    # RAG: augment system prompt with retrieved knowledge for this turn (more accurate, grounded replies)
    history_for_llm = _history_with_rag(history)

    result = (None, "")
    # 1. Local LLM (Ollama, LM Studio, or any OpenAI-compatible API)
    if local_url:
        result = _get_local_llm_response(history_for_llm, local_url)
    # 2. Gemini (free)
    if result[0] is None and gemini_key:
        result = _get_gemini_response(history_for_llm, gemini_key)
    # 3. Groq (free)
    if result[0] is None and groq_key:
        result = _get_groq_response(history_for_llm, groq_key)
    # 4. OpenAI (paid)
    if result[0] is None and openai_key:
        result = _get_openai_response(history_for_llm, openai_key)

    if result and result[0] is not None:
        assistant_message = result[0]
        history.append({"role": "assistant", "content": assistant_message})
        if len(history) > 21:
            conversations[session_id] = [history[0]] + history[-20:]
        return (assistant_message, False)

    # All AI failed—use context-aware fallback
    recent = _get_recent_user_messages(history)
    return (get_fallback_response(user_message, recent), False)


def _get_local_llm_response(history: list, base_url: str) -> tuple[str | None, str]:
    """Use locally deployed LLM (Ollama, LM Studio, LocalAI, etc.) via OpenAI-compatible API."""
    try:
        from openai import OpenAI

        # Normalize URL: ensure /v1 for chat completions
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
        client = OpenAI(base_url=url, api_key=os.environ.get("LOCAL_LLM_API_KEY", "ollama"))
        response = client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=500,
            temperature=0.7,
        )
        return (response.choices[0].message.content, "")
    except Exception as e:
        err = str(e).lower()
        if "connection" in err or "refused" in err or "timeout" in err:
            return (None, "Local LLM is not reachable. Is Ollama/LM Studio running?")
        if "model" in err or "not found" in err:
            return (None, "Model not found. Check LOCAL_LLM_MODEL.")
        return (None, str(e)[:100])


def _get_gemini_response(history: list, api_key: str) -> tuple[str | None, str]:
    """Use Google Gemini (free). Returns (response, error_msg)."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        # Use same RAG-augmented system as OpenAI path (history[0] already augmented)
        system_text = history[0]["content"] if history and history[0].get("role") == "system" else SYSTEM_PROMPT
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            system_instruction=system_text,
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
