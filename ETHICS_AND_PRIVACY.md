# Ethics & privacy

## Not a replacement for professional help

This software provides **general emotional support** and **signposting**. It does **not** diagnose, treat, or triage mental health conditions. **Crisis detection is rule-based and incomplete**—it can miss harmful content or over-trigger. Always encourage users in distress to contact **emergency services (999)**, **NHS 111**, or **Samaritans (116 123)**.

## What data is processed

- **Chat text** is sent to whichever AI backend you configure (local LLM, Google, Groq, OpenAI). Those providers have their own terms and retention policies.
- **Optional face images** are processed on your server when `USE_FACE_EMOTION=true` and ML dependencies are installed. Images are **not** stored by default unless you add that behaviour.
- **Mood analytics** (emotion label, confidence, backend name, short message preview, session key) may be written to `data/mood_events.json` when `MOOD_TRACKING_ENABLED=true`.

## Anonymization

Set `MOOD_ANONYMIZE=true` and a strong `MOOD_HASH_SALT` to store a **hashed session identifier** instead of the raw browser `session_id`. This reduces linkability if the analytics file leaks—but is not formal k-anonymity.

## Retention & security

- Delete `data/mood_events.json` when you no longer need evaluation data.
- Do not commit that file to Git (it is in `.gitignore`).
- For production, use HTTPS, restrict `/api/analytics/summary` (e.g. auth or `ANALYTICS_API_ENABLED=false` publicly), and complete a **DPIA** if you handle personal data at scale.

## DynamoDB (optional)

If `MOOD_BACKEND=dynamodb`, events are stored in AWS. Use server-side encryption, least-privilege IAM, and appropriate retention/TTL. Float values may require conversion to `Decimal` for DynamoDB—test your deployment.

## Research & dissertations

Document: purpose, limitations, consent (if human subjects), data flow diagram, and your evaluation methodology—including **false positive/negative rates** for crisis and emotion classifiers where possible.
