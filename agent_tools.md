# Agent Tool Definitions

This document defines the 5 function-call tools available to the AI Chat Interface agent.
Each tool maps to a real capability already in the system and follows the OpenAI function-calling schema.

---

## Tool 1 — `search_documents`

### Purpose
Search the user's uploaded documents using semantic similarity (RAG pipeline).
This is always the first tool attempted for any query that could be answered from internal content.

### When the agent uses it
- User asks a question about an uploaded PDF, policy document, or knowledge base file
- Query classification returns `policy_lookup`, `course_career_advice`, `platform_feature`, or `general_inquiry`
- Used *before* `search_web` — internal documents are cheaper and more authoritative than live web results

### Schema
```json
{
  "name": "search_documents",
  "description": "Search the indexed document store using semantic similarity to find chunks of text relevant to the user's query. Returns the top matching passages with their source filenames.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The user's question or search query"
      },
      "top_k": {
        "type": "integer",
        "description": "Number of document chunks to return (1–3, scaled by query complexity)",
        "default": 2
      }
    },
    "required": ["query"]
  }
}
```

### Expected Output
```json
{
  "chunks": [
    {
      "filename": "student_handbook.pdf",
      "chunk_index": 4,
      "text": "Students must submit assignments by 11:59 PM on the due date...",
      "score": 0.87
    }
  ],
  "sufficient": true
}
```

---

## Tool 2 — `search_web`

### Purpose
Perform a live web search using DuckDuckGo to retrieve current, real-world information
that the language model does not have in its training data.

### When the agent uses it
- Query classification returns `external_research` (news, elections, prices, current events)
- `search_documents` returns results with a similarity score below 0.30 (RAG insufficient)
- User explicitly asks for current or up-to-date information

### Schema
```json
{
  "name": "search_web",
  "description": "Search the web using DuckDuckGo and return a list of relevant results. Use this when the query requires current information, live data, or topics not covered by uploaded documents.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query to send to the web"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return (default 2, max 4)",
        "default": 2
      }
    },
    "required": ["query"]
  }
}
```

### Expected Output
```json
{
  "results": [
    {
      "title": "Barbados General Election 2025 — Results",
      "url": "https://example.com/barbados-election",
      "snippet": "The Barbados Labour Party won 19 of 30 seats in the..."
    }
  ]
}
```

---

## Tool 3 — `translate_and_detect`

### Purpose
Detect the language of a given text string and translate it to English.
Used when the user pastes or types foreign-language text directly into the chat
(as opposed to uploading an image).

### When the agent uses it
- User pastes text and asks "what does this say?" or "translate this"
- User types a message that contains non-English characters
- Query classification detects translation keywords in the message

### Schema
```json
{
  "name": "translate_and_detect",
  "description": "Detect the language of the provided text and translate it to English. Use this when the user submits foreign-language text directly in the chat input.",
  "parameters": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The foreign-language text to detect and translate"
      },
      "target_language": {
        "type": "string",
        "description": "The language to translate into (default: English)",
        "default": "English"
      }
    },
    "required": ["text"]
  }
}
```

### Expected Output
```json
{
  "detected_language": "Spanish",
  "original_text": "¿Cómo estás hoy?",
  "translation": "How are you today?",
  "confidence": "high"
}
```

---

## Tool 4 — `extract_image_text`

### Purpose
Send an uploaded image to GPT-4o vision to extract all visible text —
including headlines, body text, captions, small print, labels, and watermarks —
then translate everything to English.

### When the agent uses it
- User uploads an image via the 🖼 button in the chat input
- Image contains text in any language (newspaper, menu, sign, screenshot, document scan)
- Sidebar image upload triggers RAG indexing via the same tool

### Schema
```json
{
  "name": "extract_image_text",
  "description": "Extract all visible text from an uploaded image using GPT-4o vision, detect the language, and translate it to English. Handles all font sizes including small print, captions, and footnotes.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_base64": {
        "type": "string",
        "description": "Base64-encoded image data"
      },
      "mime_type": {
        "type": "string",
        "description": "MIME type of the image (image/jpeg, image/png, image/webp, image/gif)",
        "enum": ["image/jpeg", "image/png", "image/webp", "image/gif"]
      },
      "preserve_layout": {
        "type": "boolean",
        "description": "Whether to preserve the original text structure and layout in the output",
        "default": true
      }
    },
    "required": ["image_base64", "mime_type"]
  }
}
```

### Expected Output
```json
{
  "detected_language": "Arabic",
  "original_text": "عناوين الأخبار اليوم...",
  "translation": "Today's news headlines...",
  "sections_found": ["headline", "body text", "caption", "date"]
}
```

---

## Tool 5 — `classify_query`

### Purpose
Classify the user's query into one of 5 categories to determine which tools
should be invoked and in what order. This is always the first step in the
decision tree — it acts as the router for all other tools.

### When the agent uses it
- Automatically on every incoming message before any other tool is called
- Uses `temperature=0` and `max_tokens=10` to keep latency and cost minimal
- Result drives the entire tool selection logic

### Schema
```json
{
  "name": "classify_query",
  "description": "Classify the user's message into one of 5 categories to determine the optimal tool routing path. Always called first before any other tool.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The user's message to classify"
      }
    },
    "required": ["query"]
  }
}
```

### Expected Output
```json
{
  "classification": "external_research",
  "routing": {
    "primary_tool": "search_web",
    "fallback_tool": null,
    "skip_rag": true
  }
}
```

### Classification Categories

| Category | Primary Tool | Fallback Tool | Example Query |
|---|---|---|---|
| `course_career_advice` | `search_documents` | `search_web` | "What skills do I need for data science?" |
| `platform_feature` | `search_documents` | `search_web` | "How do I upload a PDF?" |
| `external_research` | `search_web` | none | "What happened in the Barbados election?" |
| `policy_lookup` | `search_documents` | none | "What is the late submission policy?" |
| `general_inquiry` | `search_documents` | `search_web` | "Explain how RAG works" |

---

## Tool Interaction Flow

```
Incoming message
       │
       ▼
[classify_query]  ──────────────────────────────────────────────────┐
       │                                                             │
       ├── external_research ──► [search_web] ──► synthesise        │
       │                                                             │
       ├── policy_lookup ──────► [search_documents] ──► synthesise  │
       │                                                             │
       └── all others ─────────► [search_documents]                 │
                                        │                            │
                                 score >= 0.30?                      │
                                   │         │                       │
                                  Yes        No                      │
                                   │         │                       │
                              synthesise  [search_web] ──► synthesise│
                                                                     │
       ┌─────────────────────────────────────────────────────────────┘
       │  image uploaded?
       ▼
[extract_image_text]  (runs independently of the text routing above)

       │  foreign text pasted?
       ▼
[translate_and_detect]  (triggered by translation keywords in query)
```

---

## Cost Per Tool Call (Approximate)

| Tool | Model Used | Estimated Cost |
|---|---|---|
| `classify_query` | gpt-4.1-mini (10 tokens) | ~$0.000004 |
| `search_documents` | text-embedding-3-small | ~$0.00002 |
| `search_web` | DuckDuckGo (free) | $0.00 |
| `translate_and_detect` | gpt-4.1-mini | ~$0.001–0.003 |
| `extract_image_text` | gpt-4o (vision) | ~$0.003–0.01 |
