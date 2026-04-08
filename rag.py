"""
rag.py — Retrieval-Augmented Generation pipeline

Responsibilities:
  1. Extract text from uploaded files (.pdf, .txt, .md)
  2. Chunk text into overlapping windows
  3. Generate embeddings via OpenAI (one batched API call per document)
  4. Store chunks + embeddings in memory
  5. Retrieve the most relevant chunks for a given query

This module is completely independent of Flask so it can be tested on its own.
"""

import base64
import io
import nltk
import numpy as np
from openai import OpenAI

# Download the sentence tokenizer model on first run (no-op if already present)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500       # target characters per chunk (chunks won't split mid-sentence)
CHUNK_OVERLAP = 1      # number of sentences to carry over into the next chunk
TOP_K = 3              # number of chunks to inject per query
EMBED_MODEL = "text-embedding-3-small"  # cheapest OpenAI embedding model, 1536 dims
VISION_MODEL = "gpt-4o"                 # used for image text extraction + translation

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGE_MIME_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
}

# ---------------------------------------------------------------------------
# In-memory document store
# Each entry is a dict:
#   {
#     "filename":    str,          # original filename shown in the UI
#     "chunk_index": int,          # position of this chunk within its document
#     "text":        str,          # the raw chunk text sent to the model
#     "embedding":   np.ndarray,   # unit-normalized vector, shape (1536,)
#   }
# ---------------------------------------------------------------------------
document_store: list[dict] = []


# ---------------------------------------------------------------------------
# Step 1 — Text extraction
# ---------------------------------------------------------------------------
def extract_text_from_image(file_bytes: bytes, ext: str, client: OpenAI) -> str:
    """
    Use GPT-4o vision to extract text from an image and translate it to English.

    Sends the image as a base64 data URL. The model detects the source language
    automatically and returns only the translated English text — no extra commentary.
    """
    mime_type = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {
                    "type": "text",
                    "text": (
                        "Extract EVERY piece of text visible in this image — including "
                        "headlines, body text, captions, footnotes, labels, watermarks, "
                        "small print, dates, numbers, and any text regardless of font size or position. "
                        "Do NOT skip text because it is small, faint, or in a corner. "
                        "If the text is not in English, translate it all to English, preserving structure. "
                        "Return only the extracted (and translated) text — no commentary, "
                        "no labels, no explanations."
                    ),
                },
            ],
        }],
        max_tokens=4000,
    )
    return response.choices[0].message.content.strip()


def extract_text_from_file(file_bytes: bytes, filename: str, client: OpenAI = None) -> str:
    """
    Convert raw file bytes to a plain text string.

    Supported formats:
      .pdf        — parsed with pypdf (reads each page, joins with newline)
      .txt / .md  — decoded as UTF-8
      .jpg / .jpeg / .png / .webp — text extracted + translated via GPT-4o vision

    Raises ValueError for unsupported extensions.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Run: pip install pypdf")

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)

    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="replace")

    elif ext in IMAGE_EXTENSIONS:
        if client is None:
            raise ValueError("OpenAI client required for image text extraction.")
        return extract_text_from_image(file_bytes, ext, client)

    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            "Supported: .pdf, .txt, .md, .jpg, .jpeg, .png, .webp"
        )


# ---------------------------------------------------------------------------
# Step 2 — Chunking (sentence-aware)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks that always end on a sentence boundary.

    Algorithm:
      1. Tokenize into sentences with nltk (no mid-sentence cuts).
      2. Greedily accumulate sentences until adding the next would exceed
         chunk_size characters.
      3. When the limit is reached, save the chunk and carry the last
         `overlap` sentences into the next chunk for context continuity.

    Why this beats character-window chunking:
      - No partial sentences wasted at chunk edges.
      - Cleaner text = better embeddings = more relevant retrieval.
      - Overlap is sentence-aligned, so context carries over coherently.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_len = len(sentence)

        # If a single sentence exceeds chunk_size, keep it as its own chunk
        if not current and s_len >= chunk_size:
            chunks.append(sentence)
            continue

        if current_len + s_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Carry the last `overlap` sentences into the next chunk
            current = current[-overlap:] if overlap else []
            current_len = sum(len(s) for s in current) + len(current)

        current.append(sentence)
        current_len += s_len + 1  # +1 for the space separator

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embeddings
# ---------------------------------------------------------------------------
def get_embeddings(texts: list[str], client: OpenAI) -> list[np.ndarray]:
    """
    Generate normalized embeddings for a list of texts.

    Efficiency note: The OpenAI embeddings API accepts a list in `input`,
    so ALL chunks from one document are embedded in a SINGLE API call —
    not one call per chunk.

    Normalization: dividing each vector by its L2 norm means cosine similarity
    simplifies to a dot product, which is faster to compute.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings = []
    for item in response.data:
        vec = np.array(item.embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm  # normalize to unit length
        embeddings.append(vec)
    return embeddings


# ---------------------------------------------------------------------------
# Step 4 — Store a document
# ---------------------------------------------------------------------------
def add_document(file_bytes: bytes, filename: str, client: OpenAI) -> dict:
    """
    Full upload pipeline: extract → chunk → embed → store.

    Returns a summary dict so the Flask route can report back to the UI:
      {"filename": str, "chunks_added": int, "total_chunks_in_store": int}
    """
    # Extract (client needed for image files)
    text = extract_text_from_file(file_bytes, filename, client)
    if not text.strip():
        raise ValueError("Document produced no readable text. "
                         "If this is a scanned PDF, it may lack a text layer.")

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no chunks after processing.")

    # Embed — one batched API call for all chunks
    embeddings = get_embeddings(chunks, client)

    # Store
    for idx, (chunk_text_val, embedding) in enumerate(zip(chunks, embeddings)):
        document_store.append({
            "filename":    filename,
            "chunk_index": idx,
            "text":        chunk_text_val,
            "embedding":   embedding,
        })

    return {
        "filename":             filename,
        "chunks_added":         len(chunks),
        "total_chunks_in_store": len(document_store),
    }


# ---------------------------------------------------------------------------
# Step 5 — Retrieve relevant chunks
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two unit-normalized vectors.

    Because we pre-normalize in get_embeddings(), this reduces to a dot product.
    Scores range from -1 (opposite) to 1 (identical).
    """
    return float(np.dot(vec_a, vec_b))


def retrieve_relevant_chunks(query: str, client: OpenAI, top_k: int = TOP_K) -> list[dict]:
    """
    Find the top-k chunks most relevant to the query.

    Efficiency note: if no documents have been uploaded, we return immediately
    with zero API calls — there is nothing to search.

    Returns a list of dicts with keys: filename, chunk_index, text, score
    """
    if not document_store:
        return []  # no documents — skip API call entirely

    # Embed the query (1 API call)
    query_vec = get_embeddings([query], client)[0]

    # Score every stored chunk
    scored = []
    for entry in document_store:
        score = cosine_similarity(query_vec, entry["embedding"])
        scored.append({
            "filename":    entry["filename"],
            "chunk_index": entry["chunk_index"],
            "text":        entry["text"],
            "score":       score,
        })

    # Return the top-k most relevant
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Helper — Format retrieved chunks for the system prompt
# ---------------------------------------------------------------------------
def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labeled text block for injection into
    the system prompt between <context> tags.

    Returns an empty string if no chunks were retrieved (no documents uploaded,
    or query had no relevant matches).
    """
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        header = f"[Document: {chunk['filename']}, Chunk {chunk['chunk_index'] + 1}]"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helper — Document list for the UI sidebar
# ---------------------------------------------------------------------------
def get_document_list() -> list[dict]:
    """
    Return a deduplicated list of indexed documents with chunk counts.
    Used by the sidebar to show which files have been indexed.
    """
    counts: dict[str, int] = {}
    for entry in document_store:
        counts[entry["filename"]] = counts.get(entry["filename"], 0) + 1

    return [{"filename": fname, "chunk_count": count}
            for fname, count in counts.items()]


# ---------------------------------------------------------------------------
# Helper — Clear all stored documents (called on session reset)
# ---------------------------------------------------------------------------
def clear_documents() -> None:
    """Empty the in-memory document store."""
    document_store.clear()
