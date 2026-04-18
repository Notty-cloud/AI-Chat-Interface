"""
app.py — Flask server for the AI Chat Interface

Routes:
  GET  /           Render the chat UI
  POST /chat       Stream a chat completion (SSE)
  POST /upload     Index an uploaded document into the RAG pipeline
  GET  /documents  Return list of indexed documents (for sidebar)
  GET  /history    Return session chat history (for page-refresh restore)
  POST /clear      Reset session history + document store

Architecture notes:
  - Chat history lives in a signed Flask session cookie (no database needed)
  - System messages are rebuilt fresh on every request so RAG context is always current
  - Responses are streamed token-by-token via Server-Sent Events (SSE)
  - The RAG pipeline is in rag.py and is completely independent of Flask
"""

import base64
import json
import logging
import os
import pickle
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, Response, jsonify, render_template,
                   request, session, stream_with_context)
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

import agent
from agent import build_web_context
import rag

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


_CACHE_PATH = Path(__file__).parent / "data" / ".rag_cache.pkl"


def preload_business_context():
    """
    Auto-index all .txt files in the data/ folder into the RAG pipeline.

    Embeddings are cached to data/.rag_cache.pkl keyed by filename + mtime.
    On subsequent restarts, unchanged files are restored from disk in
    milliseconds with zero API calls. Only files that have changed (or are
    new) trigger an embedding call.
    """
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return

    # Load existing cache from disk (ignore corrupt/missing cache silently)
    cache: dict = {}
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    cache_updated = False

    for path in sorted(data_dir.glob("*.txt")):
        mtime = path.stat().st_mtime
        cached = cache.get(path.name)

        if cached and cached.get("mtime") == mtime:
            # File unchanged — restore directly from cache (no API call)
            for entry in cached["entries"]:
                rag.document_store.append(entry)
            logger.info(
                "Preloaded '%s' from cache — %d chunks", path.name, len(cached["entries"])
            )
        else:
            # File is new or modified — embed and update cache
            try:
                file_bytes = path.read_bytes()
                result = rag.add_document(file_bytes, path.name, client)
                logger.info("Preloaded '%s' — %d chunks", path.name, result["chunks_added"])
                # Capture the entries just added so we can persist them
                entries = [e for e in rag.document_store if e["filename"] == path.name]
                cache[path.name] = {"mtime": mtime, "entries": entries}
                cache_updated = True
            except Exception as e:
                logger.warning("Failed to preload '%s': %s", path.name, e)

    if cache_updated:
        try:
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            logger.info("Embedding cache saved to %s", _CACHE_PATH)
        except Exception as e:
            logger.warning("Failed to save embedding cache: %s", e)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    {"id": "gpt-4.1",       "label": "GPT-4.1 — Most capable"},
    {"id": "gpt-4.1-mini",  "label": "GPT-4.1 Mini — Fast & cost-effective"},
    {"id": "gpt-4o",        "label": "GPT-4o — Strong reasoning"},
    {"id": "gpt-4o-mini",   "label": "GPT-4o Mini — Balanced"},
    {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo — Cheapest"},
]
MODEL_IDS            = {m["id"] for m in AVAILABLE_MODELS}
DEFAULT_MODEL        = "gpt-4.1-mini"
DEFAULT_TEMPERATURE  = 1.0
MAX_HISTORY_MESSAGES = 20           # max user+assistant messages kept in session
MAX_INPUT_LENGTH     = 4000         # max characters per user message
ALLOWED_EXTENSIONS       = {"pdf", "txt", "md", "jpg", "jpeg", "png", "webp"}
IMAGE_EXTENSIONS         = {"jpg", "jpeg", "png", "webp", "gif"}
IMAGE_MIME_TYPES         = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",  "webp": "image/webp",
    "gif": "image/gif",
}
IMAGE_MAX_BYTES          = 20 * 1024 * 1024   # 20 MB for vision uploads
VISION_MODEL             = "gpt-4o"

TRANSLATION_KEYWORDS = {
    "translat", "language", "english", "spanish", "french", "arabic",
    "japanese", "chinese", "portuguese", "german", "hindi", "korean",
    "what does", "what is this", "what does this say", "what does this mean",
}

def get_system_prompt_base(user_message: str = "") -> str:
    """
    Build the system prompt, injecting translation instructions only when
    the query appears to be translation-related. Saves ~40 tokens per
    non-translation message.
    """
    today = date.today().strftime("%B %d, %Y")
    base = (
        f"You are a helpful AI assistant. Today's date is {today}. "
        "Answer the user's questions clearly and concisely. "
        "When relevant document context is provided between <context> tags, "
        "use it to inform your answer and cite the document filename when you reference it. "
        "If the context doesn't help with the question, answer from your own knowledge."
    )
    # Only add translation instructions when the query needs them
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in TRANSLATION_KEYWORDS):
        base += (
            " You are also a multilingual translation assistant. "
            "When the user asks for a translation: "
            "automatically detect the source language if not specified, "
            "preserve the original tone and formality level, "
            "and return only the translated text unless the user asks for explanation."
        )
    return base


# ---------------------------------------------------------------------------
# Helper — Build the messages list for each API call
# ---------------------------------------------------------------------------
def build_messages(user_message: str, routing_ctx: dict) -> list[dict]:
    """
    Assemble the full messages list to send to the chat completions API.

    Structure:
      [system_message, ...trimmed_history..., new_user_message]

    The system message is built fresh each request and may include:
      - RAG context (<context> block) when internal docs are relevant
      - Web search results (<web_search> block) when web fallback was triggered
    """
    base = get_system_prompt_base(user_message)

    # 1. Inject RAG context if available
    rag_chunks    = routing_ctx.get("rag_chunks", [])
    context_block = rag.build_context_block(rag_chunks)

    # 2. Inject web search context — cap at 2 results to limit tokens
    web_results = routing_ctx.get("web_results", [])[:2]
    web_context = build_web_context(web_results)

    system_content = base
    if context_block:
        system_content += f"\n\n<context>\n{context_block}\n</context>"
    if web_context:
        system_content += (
            f"\n\n<web_search>\nThe following results were retrieved from a live web search. "
            f"Use them to supplement your answer and cite the source URLs where relevant.\n"
            f"{web_context}\n</web_search>"
        )

    system_message = {"role": "system", "content": system_content}

    # 3. Get stored history, trim to cap
    history = session.get("history", [])
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    # 4. Assemble: system + history + new user message
    return [system_message] + history + [{"role": "user", "content": user_message}]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the chat UI. Initialize an empty history if this is a new session."""
    if "history" not in session:
        session["history"] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Stream a chat completion response using Server-Sent Events (SSE).

    Request body (JSON): {"message": str}

    SSE event format:
      data: {"delta": "<token>"}   — one per streamed token
      data: {"done": true}         — signals end of stream
      data: {"error": "<msg>"}     — on API failure

    Why SSE instead of a regular JSON response?
      SSE lets tokens appear in the browser in real-time as they arrive from
      the OpenAI API, giving a much better user experience than waiting for
      the full response before displaying anything.
    """
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    user_message = data["message"].strip()

    if len(user_message) > MAX_INPUT_LENGTH:
        return jsonify({"error": f"Message too long (max {MAX_INPUT_LENGTH} chars)"}), 400

    # Optional parameters with validation
    model = data.get("model", DEFAULT_MODEL)
    if model not in MODEL_IDS:
        model = DEFAULT_MODEL

    try:
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        temperature = max(0.0, min(2.0, temperature))
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE

    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = None
        except (TypeError, ValueError):
            max_tokens = None

    # Scale RAG top_k with query complexity
    word_count = len(user_message.split())
    top_k = 1 if word_count <= 8 else (2 if word_count <= 20 else 3)

    # Classify first (fast, ~10 tokens, temperature=0).
    # external_research queries need live web data — skip the RAG embedding
    # call entirely (saves one embeddings API call on ~31% of queries).
    classification = agent.classify_query(user_message, client)
    if classification != "external_research":
        rag_chunks = rag.retrieve_relevant_chunks(user_message, client, top_k=top_k)
    else:
        rag_chunks = []

    routing_ctx = agent.route_request(
        user_message, rag_chunks, client, classification=classification
    )
    messages = build_messages(user_message, routing_ctx)

    def generate():
        """Generator that yields SSE-formatted chunks as they stream from OpenAI."""
        full_response = []
        try:
            api_kwargs = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "stream_options": {"include_usage": True},
            }
            # Default to 1024 to cap runaway responses; user can override via settings
            api_kwargs["max_tokens"] = max_tokens or 1024

            stream = client.chat.completions.create(**api_kwargs)
            usage_data = None
            for chunk in stream:
                # Final chunk with stream_options usage has empty choices
                if chunk.usage:
                    usage_data = {
                        "prompt_tokens":     chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens":      chunk.usage.total_tokens,
                    }
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response.append(delta.content)
                    payload = json.dumps({"delta": delta.content})
                    yield f"data: {payload}\n\n"

            # Signal that streaming is complete
            done_payload = {
                "done":           True,
                "classification": routing_ctx.get("classification", "general_inquiry"),
                "tools_used":     routing_ctx.get("tools_used", []),
            }
            if usage_data:
                done_payload["usage"] = usage_data
            yield f"data: {json.dumps(done_payload)}\n\n"

        except AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Check your OPENAI_API_KEY.'})}\n\n"
            return
        except RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit exceeded. Please wait and try again.'})}\n\n"
            return
        except APIConnectionError:
            yield f"data: {json.dumps({'error': 'Connection error. Check your internet connection.'})}\n\n"
            return
        except APIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Save the completed exchange to session history
        full_text = "".join(full_response)
        history   = session.get("history", [])
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": full_text})

        # Fix 4: when history exceeds the cap, summarise the oldest half
        # into a single compact assistant message instead of just dropping it.
        # This preserves context while dramatically reducing token count.
        if len(history) > MAX_HISTORY_MESSAGES:
            half       = len(history) // 2
            to_summarise = history[:half]
            recent       = history[half:]
            try:
                summary_resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarise the following conversation in 3-5 sentences, "
                                "preserving all key facts, decisions, and topics discussed. "
                                "Be concise — this summary replaces the full history to save tokens."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "\n".join(
                                f"{m['role'].upper()}: {m['content']}"
                                for m in to_summarise
                            ),
                        },
                    ],
                    max_tokens=200,
                    temperature=0,
                )
                summary_text = summary_resp.choices[0].message.content.strip()
                history = [
                    {"role": "assistant", "content": f"[Earlier conversation summary]: {summary_text}"}
                ] + recent
            except Exception:
                # If summarisation fails, fall back to simple trim
                history = history[-MAX_HISTORY_MESSAGES:]

        session["history"] = history
        session.modified = True

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx buffering if behind a proxy
        },
    )


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a file upload and index it into the RAG pipeline.

    Validates extension, reads bytes, calls rag.add_document() which:
      1. Extracts text
      2. Chunks it
      3. Embeds all chunks in one batched API call
      4. Stores chunks + embeddings in memory

    Returns JSON summary of what was indexed.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "File has no name"}), 400

    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"File type '.{ext}' not supported. Use .pdf, .txt, .md, .jpg, .jpeg, .png, or .webp"
        }), 400

    try:
        file_bytes = file.read()
        result = rag.add_document(file_bytes, file.filename, client)
        return jsonify({
            "success": True,
            "filename": result["filename"],
            "chunks_added": result["chunks_added"],
            "total_chunks": result["total_chunks_in_store"],
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except AuthenticationError:
        return jsonify({"success": False, "error": "Invalid API key."}), 401
    except RateLimitError:
        return jsonify({"success": False, "error": "Rate limit exceeded. Please wait and try again."}), 429
    except APIConnectionError:
        return jsonify({"success": False, "error": "Connection error. Check your internet connection."}), 502
    except APIError as e:
        return jsonify({"success": False, "error": f"OpenAI API error: {e}"}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500


@app.route("/upload-image", methods=["POST"])
def upload_image():
    """
    Accept an image, send it to GPT-4o vision, extract + translate all text.

    Returns JSON:
      {
        "success": true,
        "detected_language": str,
        "original_text":     str,
        "translation":       str
      }

    The result is also appended to session history so the user can ask
    follow-up questions about it in the normal chat input.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "File has no name"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in IMAGE_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported type '.{ext}'. Use jpg, jpeg, png, webp, or gif."
        }), 400

    file_bytes = file.read()
    if len(file_bytes) > IMAGE_MAX_BYTES:
        return jsonify({"success": False, "error": "Image exceeds 20 MB limit."}), 400

    mime_type = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": (
                            "Examine this image and do the following:\n"
                            "1. Extract EVERY piece of text visible in the image — including "
                            "headlines, body text, captions, footnotes, labels, watermarks, "
                            "small print, dates, numbers, and any text regardless of font size or position.\n"
                            "Do NOT skip text because it is small, faint, or in a corner.\n"
                            "2. Detect what language the text is written in.\n"
                            "3. Translate ALL extracted text to English, preserving the original structure.\n\n"
                            "Respond in exactly this format (keep the labels):\n"
                            "Detected language: <language name>\n"
                            "Original text: <all extracted text, preserving layout>\n"
                            "English translation: <full translation of all extracted text>\n\n"
                            "If there is no text in the image, say so clearly in each field."
                        ),
                    },
                ],
            }],
            max_tokens=4000,
        )
    except AuthenticationError:
        return jsonify({"success": False, "error": "Invalid API key."}), 401
    except RateLimitError:
        return jsonify({"success": False, "error": "Rate limit exceeded. Please wait and try again."}), 429
    except APIConnectionError:
        return jsonify({"success": False, "error": "Connection error. Check your internet connection."}), 502
    except APIError as e:
        return jsonify({"success": False, "error": str(e)}), 502

    raw = response.choices[0].message.content.strip()

    # Parse the structured response — each field may span multiple lines.
    # We find where each label starts and capture everything up to the next label.
    def extract_field(label, text, next_labels=None):
        lines = text.splitlines()
        result_lines = []
        inside = False
        for line in lines:
            if line.lower().startswith(label.lower() + ":"):
                inside = True
                # Capture any text on the same line as the label
                after_colon = line.split(":", 1)[-1].strip()
                if after_colon:
                    result_lines.append(after_colon)
                continue
            if inside:
                # Stop when we hit the next label
                if next_labels and any(line.lower().startswith(nl.lower() + ":") for nl in next_labels):
                    break
                result_lines.append(line)
        return "\n".join(result_lines).strip()

    detected_language = extract_field("Detected language", raw,
                                      next_labels=["Original text", "English translation"])
    original_text     = extract_field("Original text", raw,
                                      next_labels=["English translation"])
    translation       = extract_field("English translation", raw)

    # Save to session history so follow-up chat questions have context
    history = session.get("history", [])
    assistant_summary = (
        f"[Image Translation Result]\n"
        f"Detected language: {detected_language}\n"
        f"Original text: {original_text}\n"
        f"English translation: {translation}"
    )
    history.append({"role": "user",      "content": f"[Uploaded image: {file.filename}]"})
    history.append({"role": "assistant", "content": assistant_summary})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    session["history"] = history
    session.modified = True

    return jsonify({
        "success":            True,
        "detected_language":  detected_language,
        "original_text":      original_text,
        "translation":        translation,
    })


@app.route("/models")
def models_list():
    """Return available models for the UI model selector."""
    return jsonify(AVAILABLE_MODELS)


@app.route("/documents")
def documents():
    """Return the list of indexed documents for the sidebar."""
    return jsonify(rag.get_document_list())


@app.route("/history")
def history():
    """
    Return the current session's chat history.
    Called by the frontend on page load to restore the chat thread
    after a browser refresh.
    """
    return jsonify(session.get("history", []))


@app.route("/clear", methods=["POST"])
def clear():
    """Reset both the session history and the in-memory document store."""
    session["history"] = []
    session.modified = True
    rag.clear_documents()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Only preload once — Werkzeug reloader spawns a child process and sets
    # WERKZEUG_RUN_MAIN="true" in it. We want preload to run only in that
    # child (the one that actually serves requests), not in the monitor process.
    preload_business_context()
    # use_reloader=False prevents Werkzeug from spawning a second process that
    # would run preload_business_context() again and make 4 duplicate API calls.
    # Restart the server manually after code changes during development.
    # Never use debug=True in production.
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
