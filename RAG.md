# RAG (Retrieval-Augmented Generation)

The chatbot uses **RAG** (not “GAG”) to improve accuracy: before each AI reply, it **retrieves** short passages from `data/rag_knowledge.json` that match the user’s message and **injects** them into the system prompt so the model stays closer to vetted facts (UK signposting, coping ideas, disclaimers).

## How to extend

1. Edit `data/rag_knowledge.json`.
2. Add objects with:
   - `id`: unique string
   - `keywords`: words that should match user messages
   - `content`: the text the model may use (keep it factual and short)

3. Redeploy or restart the app.

## Settings

| Variable       | Default | Meaning                          |
|----------------|---------|----------------------------------|
| `RAG_ENABLED`  | `true`  | Set `false` to turn RAG off      |
| `RAG_TOP_K`    | `3`     | Max number of chunks to retrieve |

## Retrieval

Matching is **keyword + word overlap** (no extra API). For heavier semantic search later, you could add embeddings.
