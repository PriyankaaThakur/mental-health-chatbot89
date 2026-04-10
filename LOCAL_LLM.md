# Local LLM Setup (Ollama / LM Studio)

Use a locally deployed LLM for privacy, no API costs, and offline use.

---

## Option 1: Ollama (Recommended)

### 1. Install Ollama

- **Windows:** Download from [ollama.com](https://ollama.com)
- **Mac/Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

### 2. Pull a model

```bash
ollama pull llama3.2
```

Other models: `ollama pull mistral`, `ollama pull phi3`, `ollama pull gemma2`

### 3. Add to `.env`

```
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2
```

### 4. Run

Start the Flask app. Ollama runs in the background. The chatbot will use it first, then fall back to Gemini/Groq/OpenAI if local fails.

---

## Option 2: LM Studio

### 1. Install LM Studio

Download from [lmstudio.ai](https://lmstudio.ai)

### 2. Load a model

- Open LM Studio → Search → Download a model (e.g. Llama 3, Mistral)
- Go to **Local Server** → Start the server (default port 1234)

### 3. Add to `.env`

```
LOCAL_LLM_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=your-model-name
```

(Model name is shown in LM Studio when the server is running.)

---

## AI Provider Order

The chatbot tries providers in this order:

1. **Local LLM** (if `LOCAL_LLM_URL` is set)
2. **Gemini** (if `GEMINI_API_KEY` is set)
3. **Groq** (if `GROQ_API_KEY` is set)
4. **OpenAI** (if `OPENAI_API_KEY` is set)
5. **Fallback** (rule-based responses if all fail)

---

## Deployment (Render, Railway, etc.)

`LOCAL_LLM_URL` only works when the LLM runs on the same machine as the app. For cloud hosting:

- Use **Gemini**, **Groq**, or **OpenAI**
- Or host the LLM and app on the same server and set `LOCAL_LLM_URL` accordingly
