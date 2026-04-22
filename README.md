# AI Chat Interface

A full-featured AI chatbot built with Flask and the OpenAI Python SDK. Supports real-time streaming responses, a Retrieval-Augmented Generation (RAG) pipeline, decision-tree agent routing with live web search, multilingual image translation, and configurable model settings — all in a single-page web UI with no frontend framework.

---

## Features

- **Streaming Chat** — responses stream token-by-token via Server-Sent Events (SSE)
- **Chat History** — full conversation memory per session, restored on page refresh, summarised automatically when long
- **RAG Pipeline** — upload documents (PDF, TXT, MD) and the assistant grounds its answers in your content
- **Decision-Tree Agent Routing** — every query is classified and routed through the optimal tool path (RAG only, web only, or RAG + web fallback)
- **Live Web Search** — DuckDuckGo fallback (no API key required) fires automatically when internal knowledge is insufficient
- **Image Translation** — upload a photo containing text in any language; GPT-4o extracts and translates all text to English including small print
- **Model Selector** — switch between GPT-4.1, GPT-4.1 Mini, GPT-4o, GPT-4o Mini, GPT-3.5 Turbo at runtime
- **Temperature & Max Tokens** — tune model creativity and response length from the sidebar
- **Token Usage Display** — prompt + completion token counts shown after every response
- **Sentence-Aware Chunking** — NLTK sentence tokenizer ensures RAG chunks never split mid-sentence
- **Token-Efficient System Prompts** — translation instructions only injected when relevant; RAG chunk count scales with query complexity

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask |
| AI | OpenAI Python SDK (`gpt-4.1-mini`, `gpt-4o`, `text-embedding-3-small`) |
| Agent Routing | Custom decision-tree classifier (`agent.py`) |
| Web Search | DuckDuckGo Search (`duckduckgo-search`) |
| RAG | Cosine similarity over unit-normalised embeddings (NumPy) |
| NLP Chunking | NLTK sentence tokenizer (`punkt_tab`) |
| PDF Parsing | pypdf |
| Frontend | Vanilla HTML / CSS / JavaScript — no framework |

---

## Project Structure

```
AI_Chat_Interface/
├── app.py               # Flask server — all routes and request handling
├── agent.py             # Decision-tree routing: classify, RAG sufficiency, web search
├── rag.py               # RAG pipeline: extract, chunk, embed, store, retrieve
├── templates/
│   └── index.html       # Single-page chat UI
├── requirements.txt     # Python dependencies
├── .env.example         # API key template (safe to commit)
├── .gitignore           # Excludes .env and __pycache__
└── README.md
```

---

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/Notty-cloud/AI-Chat-Interface.git
cd AI-Chat-Interface
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the NLTK sentence tokenizer

Run this once — it downloads a small model file used for sentence-aware chunking:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Configure your API key

Copy the example env file and fill in your values:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-actual-key-here
FLASK_SECRET_KEY=any-random-string-you-choose
```

> **Important:** Never commit your `.env` file. It is already excluded by `.gitignore`.

### 6. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## How to Use

### Chat
Type a message and press **Enter** (or **Shift+Enter** for a new line). Responses stream in real-time. Token usage and routing details appear below each response.

### Upload a Document (RAG)
Drag and drop a `.pdf`, `.txt`, or `.md` file onto the sidebar upload area (or click to browse). The document is chunked, embedded, and indexed. The assistant will automatically reference it when relevant.

### Translate an Image
Click the **🖼** button next to the chat input. Select a `.jpg`, `.jpeg`, `.png`, `.webp`, or `.gif` image containing text in any language. The app will:
1. Send the image to GPT-4o vision
2. Extract all visible text (including small print)
3. Detect the source language
4. Return a full English translation
5. Save the result to chat history for follow-up questions

### Adjust Settings
Use the **Settings** panel in the sidebar to change the model, temperature, and max tokens before sending a message.

---

## Agent Decision Tree

Every message is classified and routed through the optimal tool path:

```
User Message
     │
     ▼
Classify Query  (gpt-4.1-mini, max_tokens=10, temperature=0)
     │
     ├── external_research  →  Web Search only
     ├── policy_lookup      →  RAG only (no web fallback)
     └── all others         →  RAG first → Web Search if RAG score < 0.30
     │
     ▼
Build system prompt (RAG context + web results injected as needed)
     │
     ▼
Stream response  →  Save to session history  →  Summarise if history > 20 messages
```

Query types: `course_career_advice`, `platform_feature`, `external_research`, `policy_lookup`, `general_inquiry`

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Render the chat UI |
| `POST` | `/chat` | Stream a chat completion (SSE) |
| `POST` | `/upload` | Index a document into the RAG pipeline |
| `POST` | `/upload-image` | Extract and translate text from an image |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/history` | Return session chat history |
| `POST` | `/clear` | Reset session history and document store |
| `GET` | `/models` | List available models |

---

## RAG Pipeline

1. **Extract** — text from PDF (pypdf), TXT, or MD files; images via GPT-4o vision
2. **Chunk** — sentence-aware splitting (NLTK) into ~500-character windows with 1-sentence overlap
3. **Embed** — `text-embedding-3-small`, all chunks in a single batched API call, unit-normalised
4. **Store** — in-memory list of `{filename, chunk_index, text, embedding}`
5. **Retrieve** — cosine similarity (dot product), top-k chunks scaled by query complexity (1–3)

---

## Supported File Types

| Extension | Upload Via | Purpose |
|---|---|---|
| `.pdf` | Sidebar | RAG indexing |
| `.txt` | Sidebar | RAG indexing |
| `.md` | Sidebar | RAG indexing |
| `.jpg` / `.jpeg` | Sidebar or 🖼 button | RAG indexing or image translation |
| `.png` | Sidebar or 🖼 button | RAG indexing or image translation |
| `.webp` | Sidebar or 🖼 button | RAG indexing or image translation |
| `.gif` | 🖼 button | Image translation only |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `FLASK_SECRET_KEY` | Recommended | Signs session cookies — set any random string in production |

---

## Troubleshooting

**"The api_key client option must be set" error**
Your `.env` file is missing or `OPENAI_API_KEY` is not set. Check that `.env` exists and contains a valid key.

**"Unsupported file type" on upload**
Hard refresh the browser with `Ctrl+Shift+R` to clear the cached old version of the page.

**Image translation only shows the detected language, not the translation**
Upgrade to the latest version of the app — an earlier parser bug that caused this has been fixed.

**Web search not triggering**
The query classifier may not have detected it as current-events research. Be more explicit: *"Search the web for the latest news on..."*

---

## License

This project is provided for educational purposes.
