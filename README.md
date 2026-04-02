# AI Chat Interface

A full-featured AI chatbot built with Flask and the OpenAI Python SDK. Supports real-time streaming, Retrieval-Augmented Generation (RAG), multilingual image translation, and configurable model settings.

---

## Features

- **Streaming Chat** — responses stream token-by-token via Server-Sent Events (SSE)
- **Chat History** — full conversation memory stored in session, restored on page refresh
- **RAG Pipeline** — upload documents (PDF, TXT, MD) to ground responses in your content
- **Image Translation** — upload a photo containing text in any language; GPT-4o extracts and translates it to English
- **Model Selector** — switch between GPT-4.1, GPT-4.1 Mini, GPT-4o, GPT-4o Mini, GPT-3.5 Turbo
- **Temperature & Max Tokens** — tune model creativity and response length from the sidebar
- **Token Usage Display** — see prompt + completion token counts after each response
- **Sentence-Aware Chunking** — NLTK-based chunking ensures RAG chunks never split mid-sentence
- **Multilingual Translation** — the assistant auto-detects and translates any language

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask |
| AI | OpenAI Python SDK (`gpt-4.1-mini`, `gpt-4o`, embeddings) |
| RAG | `text-embedding-3-small` + cosine similarity (NumPy) |
| NLP Chunking | NLTK sentence tokenizer |
| PDF Parsing | pypdf |
| Frontend | Vanilla HTML/CSS/JavaScript (no framework) |

---

## Project Structure

```
AI_Chat_Interface/
├── app.py               # Flask server — all routes
├── rag.py               # RAG pipeline (extract, chunk, embed, retrieve)
├── templates/
│   └── index.html       # Single-page chat UI
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore           # Excludes .env and __pycache__
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Notty-cloud/AI-Chat-Interface.git
cd AI-Chat-Interface
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-actual-key-here
FLASK_SECRET_KEY=any-random-string
```

> **Never commit your `.env` file.** It is already in `.gitignore`.

### 4. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Render the chat UI |
| `POST` | `/chat` | Stream a chat completion (SSE) |
| `POST` | `/upload` | Index a document into the RAG pipeline |
| `POST` | `/upload-image` | Translate text in an uploaded image |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/history` | Return session chat history |
| `POST` | `/clear` | Reset session + document store |
| `GET` | `/models` | List available models |

---

## RAG Pipeline

1. **Extract** — text from PDF, TXT, or MD files
2. **Chunk** — sentence-aware splitting (NLTK) into ~500-character windows with 1-sentence overlap
3. **Embed** — `text-embedding-3-small` via OpenAI, all chunks in one batched API call
4. **Store** — in-memory list of `{filename, chunk_index, text, embedding}`
5. **Retrieve** — cosine similarity (dot product on unit-normalized vectors), top-3 chunks injected into the system prompt

---

## Image Translation

Upload any image (JPG, JPEG, PNG, WEBP, GIF) containing text via the 🖼 button in the chat input area. The app will:

1. Encode the image as base64
2. Send it to GPT-4o vision
3. Extract all visible text
4. Detect the source language
5. Translate everything to English
6. Display the result as a formatted card in the chat
7. Save the translation to session history so you can ask follow-up questions

---

## Supported File Types

| Type | Upload Method | Purpose |
|---|---|---|
| `.pdf` | Sidebar | RAG indexing |
| `.txt` | Sidebar | RAG indexing |
| `.md` | Sidebar | RAG indexing |
| `.jpg` / `.jpeg` | Sidebar or 🖼 button | RAG indexing or translation |
| `.png` | Sidebar or 🖼 button | RAG indexing or translation |
| `.webp` | Sidebar or 🖼 button | RAG indexing or translation |
| `.gif` | 🖼 button | Translation only |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `FLASK_SECRET_KEY` | Recommended | Secret key for signing session cookies |

---

## License

This project is provided for educational purposes.
