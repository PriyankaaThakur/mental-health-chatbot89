# Model comparison (methodology & evaluation)

Use this structure in your **methodology** and **evaluation** chapters. Replace placeholder metrics with your own experiments.

## Text emotion (user utterance → label)

| Approach | Typical role | Pros | Cons |
|----------|--------------|------|------|
| **Lexicon / keyword baseline** | Fast, interpretable | No install heavy ML; works offline | Misses sarcasm, context, negation |
| **Logistic regression (bag-of-words)** | Classical baseline | Fast to train, easy to explain | Weak on long-range context |
| **LSTM / GRU on token sequences** | Sequential model | Captures order better than bag-of-words | More data & tuning; slower than transformers today |
| **BERT-family (e.g. DistilRoBERTa)** | Strong default | Contextual embeddings; good transfer | Heavier compute; API cost if remote |

**This repo (optional):** `j-hartmann/emotion-english-distilroberta-base` via HuggingFace—**DistilRoBERTa** is a distilled transformer in the same ecosystem as BERT-style pretraining. If `transformers` is not installed, **`emotion_service.py` falls back to a keyword heuristic** so you can contrast **baseline vs neural** in write-ups.

### Suggested evaluation

- **Dataset:** Emotion-labelled tweets or similar (e.g. public English emotion corpora); hold-out split.
- **Metrics:** Accuracy, macro-F1, per-class recall (important for rare classes like crisis-adjacent sadness).
- **Error analysis:** Confusion between *fear* vs *sadness*, *anger* vs *disgust*.

## Face emotion (FER-style)

| Approach | Notes |
|----------|--------|
| **CNN on FER2013** | Classic baseline for coursework linking to FER2013 |
| **ViT fine-tuned on expressions** | Often stronger with enough data |

**This repo (optional):** `trpakov/vit-face-expression` (ViT). Document **pose, lighting, occlusion**, and **demographic bias** limitations.

## Crisis detection

- **Current system:** Hybrid **substring / regex** + **letters-only** normalization for obfuscated suicide spellings.
- **Evaluation:** Curate a **small labelled set** of crisis vs non-crisis paraphrases; report precision/recall and describe **false positives** (e.g. figurative language).

## Chat quality

- **Human rating** (empathy, safety, relevance) on a Likert scale.
- **LLM-as-judge** only as a supplement, with caveats.

## Fine-tuning dialogue models

Training **DialoGPT / Llama** etc. on mental-health-style dialogues is **advanced** and needs **ethical review**, **de-identified data**, and **safety evaluation**. This project uses **API / local LLM + RAG + prompts** instead; you can still discuss fine-tuning as **future work**.
