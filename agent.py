"""
agent.py — Decision-tree routing (SkillSprint-style architecture)

Flow for every query:
  1. classify_query()   — fast LLM call to determine query type
  2. route_request()    — select tools based on type
  3. is_rag_sufficient()— decide whether to trigger web search fallback
  4. web_search()       — DuckDuckGo (no API key, free)
  5. Return context dict used by build_messages() in app.py

Query types:
  course_career_advice  → RAG primary, web fallback if RAG insufficient
  platform_feature      → RAG primary, web fallback if RAG insufficient
  external_research     → Web primary (skip RAG — inherently needs live data)
  policy_lookup         → RAG only (never fall back to web for policy)
  general_inquiry       → RAG primary, web fallback if RAG insufficient
"""

import logging

from duckduckgo_search import DDGS
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUERY_TYPES = {
    "course_career_advice",
    "platform_feature",
    "external_research",
    "policy_lookup",
    "general_inquiry",
}

CLASSIFIER_MODEL        = "gpt-4.1-mini"   # fast + cheap for classification
RAG_SUFFICIENCY_THRESHOLD = 0.30           # min cosine similarity to consider RAG sufficient
WEB_SEARCH_MAX_RESULTS  = 4               # keep context concise


# ---------------------------------------------------------------------------
# Step 1 — Query Classification
# ---------------------------------------------------------------------------
def classify_query(query: str, client: OpenAI) -> str:
    """
    Classify the query into one of 5 types with a single fast LLM call.
    Uses temperature=0 and max_tokens=10 to keep latency minimal.
    Falls back to 'general_inquiry' if the model returns anything unexpected.
    """
    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user query into exactly one of these categories.\n"
                        "Reply with ONLY the category name — no punctuation, no explanation.\n\n"
                        "Categories:\n"
                        "- course_career_advice : courses, skills, career paths, job market\n"
                        "- platform_feature     : app features, compatibility, how-to guides\n"
                        "- external_research    : market analysis, industry trends, current news\n"
                        "- policy_lookup        : rules, policies, procedures, terms of service\n"
                        "- general_inquiry      : anything else"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower().replace(" ", "_")
        return result if result in QUERY_TYPES else "general_inquiry"
    except Exception as e:
        logger.warning("classify_query failed: %s — defaulting to general_inquiry", e)
        return "general_inquiry"


# ---------------------------------------------------------------------------
# Step 2 — RAG Sufficiency Check
# ---------------------------------------------------------------------------
def is_rag_sufficient(chunks: list[dict]) -> bool:
    """
    Returns True if the top RAG chunk has a similarity score above the
    threshold, meaning the internal knowledge base has a relevant answer.

    Returns False if:
      - No documents have been uploaded (empty store)
      - Top chunk similarity is below RAG_SUFFICIENCY_THRESHOLD
    """
    if not chunks:
        return False
    return chunks[0].get("score", 0) >= RAG_SUFFICIENCY_THRESHOLD


# ---------------------------------------------------------------------------
# Step 3 — Web Search (DuckDuckGo, no API key)
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    Returns a list of {title, url, snippet} dicts.
    Returns [] silently on any failure so the chat still works.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("web_search failed: %s", e)
        return []


def build_web_context(results: list[dict]) -> str:
    """
    Format web results into a labeled block for injection into the system prompt.
    Each result is numbered and includes title, source URL, and snippet.
    """
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r['title']}\n"
            f"Source: {r['url']}\n"
            f"{r['snippet']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 4 — Route Request (full decision tree)
# ---------------------------------------------------------------------------
def route_request(query: str, rag_chunks: list[dict], client: OpenAI) -> dict:
    """
    Run the full decision tree for a query.

    Decision logic:
      external_research → web only (skip RAG)
      policy_lookup     → RAG only (no web fallback)
      all others        → RAG first; web if RAG insufficient

    Returns a context dict:
      {
        classification : str,
        rag_chunks     : list[dict],
        web_results    : list[dict],
        web_context    : str,
        tools_used     : list[str],
      }
    """
    classification = classify_query(query, client)
    web_results: list[dict] = []
    tools_used:  list[str]  = []

    if classification == "external_research":
        # PRIMARY: web — this query inherently requires live data
        web_results = web_search(query)
        if web_results:
            tools_used.append("web_search")

    elif classification == "policy_lookup":
        # RAG only — policy answers must come from internal documents
        if rag_chunks:
            tools_used.append("rag")

    else:
        # course_career_advice / platform_feature / general_inquiry
        # PRIMARY: RAG. SECONDARY: web if RAG is insufficient.
        if rag_chunks:
            tools_used.append("rag")

        if not is_rag_sufficient(rag_chunks):
            web_results = web_search(query)
            if web_results:
                tools_used.append("web_search")

    logger.info(
        "route_request | classification=%s | tools=%s | rag_chunks=%d | web_results=%d",
        classification, tools_used, len(rag_chunks), len(web_results),
    )

    return {
        "classification": classification,
        "rag_chunks":     rag_chunks,
        "web_results":    web_results,
        "web_context":    build_web_context(web_results),
        "tools_used":     tools_used,
    }
