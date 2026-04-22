"""
agent.py — Decision-tree routing + agentic tool-calling loop

Module responsibilities
  1. classify_query()    — fast LLM call to determine query type
  2. route_request()     — select context sources (RAG / web) based on type
  3. is_rag_sufficient() — decide whether to trigger web search fallback
  4. web_search()        — DuckDuckGo (no API key, free)
  5. Four file tools     — list_files, read_file, search_files, write_file
  6. run_agent_loop()    — lets the model call tools autonomously before answering

Tool-calling flow
  Every chat request goes through run_agent_loop(), which streams responses
  normally.  When the model decides to call a tool the loop:
    a) Yields a {"tool_call": ...} SSE event so the UI can show activity
    b) Executes the Python function whose docstring was used as the tool description
    c) Appends the result as a "tool" message and loops
    d) Streams the final answer when the model is satisfied

The key link between Python and OpenAI:
  _tool_schema(func, ...) extracts inspect.getdoc(func) and uses it as the
  "description" field in the JSON schema sent to the model.  The model reads
  that docstring to decide *when* to call each tool.
"""

import fnmatch  # noqa: F401  (used indirectly via rglob in search_files)
import inspect
import json
import logging
from pathlib import Path
from typing import Generator

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
                        "- external_research    : market analysis, industry trends, current news, "
                        "elections, politics, sports results, weather, prices, recent events, "
                        "anything that changes over time or requires up-to-date information\n"
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
def route_request(
    query: str,
    rag_chunks: list[dict],
    client: OpenAI,
    classification: str = None,
) -> dict:
    """
    Run the full decision tree for a query.

    Decision logic:
      external_research → web only (skip RAG)
      policy_lookup     → RAG only (no web fallback)
      all others        → RAG first; web if RAG insufficient

    `classification` may be passed in pre-computed (from app.py) so the
    classify API call is not duplicated when the caller already classified
    the query in order to decide whether to skip RAG retrieval.

    Returns a context dict:
      {
        classification : str,
        rag_chunks     : list[dict],
        web_results    : list[dict],
        web_context    : str,
        tools_used     : list[str],
      }
    """
    if classification is None:
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


# ===========================================================================
# FILE TOOLS — list, read, search, write
#
# Each function carries a detailed docstring.  _tool_schema() reads that
# docstring with inspect.getdoc() and places it in the OpenAI "description"
# field, so the model's tool-call decisions are driven by what the docstrings
# actually say.
# ===========================================================================

# Sandboxed workspace — all file operations are restricted to this directory.
WORKSPACE_DIR = Path(__file__).parent / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

MAX_FILE_READ_BYTES = 100_000   # 100 KB ceiling per read_file call
MAX_SEARCH_MATCHES  = 5         # max files returned by search_files
CONTEXT_LINES       = 2         # lines of context around each search hit
MAX_TOOL_TURNS      = 5         # max tool-calling rounds before forcing a final answer


def _resolve(relative: str) -> Path:
    """
    Resolve a relative path to an absolute path inside WORKSPACE_DIR.
    Raises ValueError on path-traversal attempts (e.g. '../../etc/passwd').
    """
    resolved = (WORKSPACE_DIR / relative).resolve()
    workspace_root = WORKSPACE_DIR.resolve()
    # str comparison works cross-platform; ensure the path stays inside
    if not str(resolved).startswith(str(workspace_root)):
        raise ValueError(f"Path {relative!r} escapes the workspace sandbox.")
    return resolved


# ---------------------------------------------------------------------------
# Tool 1 — list_files
# ---------------------------------------------------------------------------
def list_files(directory: str = ".") -> str:
    """
    List all files and subdirectories in the specified workspace directory.

    Returns a formatted table showing each entry's name, type (FILE / DIR),
    and size in bytes.  Use this tool to discover what files are available
    before deciding which ones to read or search — for example when the user
    asks 'what files do you have?' or 'what's in the workspace?'.

    Args:
        directory: Relative path inside the workspace to list.
                   Defaults to '.' (the workspace root).
    """
    target = _resolve(directory)
    if not target.exists():
        return f"Directory not found: {directory!r}"
    if not target.is_dir():
        return f"{directory!r} is a file, not a directory — use read_file to inspect it."

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not entries:
        return f"Directory '{directory}' is empty."

    lines = [f"Contents of '{directory}':", ""]
    for entry in entries:
        kind = "FILE" if entry.is_file() else "DIR "
        size = f"{entry.stat().st_size:>8} B" if entry.is_file() else "        —"
        lines.append(f"  [{kind}]  {size}   {entry.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — read_file
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    """
    Read and return the full text content of a file in the workspace.

    Use this when you need to examine what a specific file contains — to answer
    a question about it, summarise it, extract facts, or quote from it.
    Files larger than 100 KB are returned truncated with a notice at the top.

    Args:
        path: Relative path to the file within the workspace
              (e.g. 'notes.txt' or 'reports/q1.md').
    """
    target = _resolve(path)
    if not target.exists():
        return f"File not found: {path!r}"
    if not target.is_dir():
        pass  # good — it's a file
    if target.is_dir():
        return f"{path!r} is a directory — use list_files to see its contents."

    raw      = target.read_bytes()
    size     = len(raw)
    truncated = size > MAX_FILE_READ_BYTES
    text     = raw[:MAX_FILE_READ_BYTES].decode("utf-8", errors="replace")
    header   = (
        f"--- {path} ({size} bytes)"
        + (" [TRUNCATED — showing first 100 KB]" if truncated else "")
        + " ---\n"
    )
    return header + text


# ---------------------------------------------------------------------------
# Tool 3 — search_files
# ---------------------------------------------------------------------------
def search_files(query: str, directory: str = ".", file_pattern: str = "*") -> str:
    """
    Search workspace files for those whose content contains a specific text query.

    Performs a case-insensitive full-text scan and returns up to five matching
    files, each with the matching line numbers and a few lines of surrounding
    context.  Use this when you need to find which document discusses a topic
    but you do not know the filename — for example 'find files about pricing'
    or 'which file mentions the deadline?'.

    Args:
        query:        Text string to search for (case-insensitive).
        directory:    Relative path within the workspace to scan.
                      Defaults to '.' (all workspace files).
        file_pattern: Glob pattern to filter filenames, e.g. '*.txt' or
                      'report_*.md'.  Defaults to '*' (all files).
    """
    target = _resolve(directory)
    if not target.exists():
        return f"Directory not found: {directory!r}"

    query_lower = query.lower()
    matches: list[str] = []

    for file_path in sorted(target.rglob(file_pattern)):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines    = text.splitlines()
        hit_rows = [(i + 1, ln) for i, ln in enumerate(lines) if query_lower in ln.lower()]
        if not hit_rows:
            continue

        rel      = file_path.relative_to(WORKSPACE_DIR)
        snippets = []
        for lineno, _ in hit_rows[:3]:          # cap at 3 hits per file
            start = max(0, lineno - 1 - CONTEXT_LINES)
            end   = min(len(lines), lineno + CONTEXT_LINES)
            block = "\n".join(
                f"  {'>' if i + 1 == lineno else ' '} {i + 1:4}: {lines[i]}"
                for i in range(start, end)
            )
            snippets.append(block)
        matches.append(f"[{rel}]\n" + "\n---\n".join(snippets))

        if len(matches) >= MAX_SEARCH_MATCHES:
            break

    if not matches:
        return (
            f"No files containing {query!r} found in '{directory}' "
            f"matching pattern '{file_pattern}'."
        )
    return (
        f"Found {len(matches)} file(s) containing {query!r}:\n\n"
        + "\n\n".join(matches)
    )


# ---------------------------------------------------------------------------
# Tool 4 — write_file
# ---------------------------------------------------------------------------
def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a file in the workspace with the given text content.

    Use this when the user explicitly asks you to save, create, or write
    something to a file — for example 'save this summary as summary.txt',
    'create a report called report.md', or 'write my notes to notes.txt'.
    Parent directories within the workspace are created automatically.

    Args:
        path:    Relative path for the file within the workspace
                 (e.g. 'summary.txt' or 'reports/q1_summary.md').
        content: Full text content to write to the file.
    """
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    size = target.stat().st_size
    return f"File written: '{path}' ({size} bytes)"


# ===========================================================================
# TOOL SCHEMA BUILDER
#
# _tool_schema() is the bridge between Python documentation and the OpenAI
# API.  It reads the function's docstring with inspect.getdoc() and places
# it in the 'description' field of the JSON schema.  The model reads that
# description to decide *when* and *whether* to call each tool.
# ===========================================================================

def _tool_schema(func, properties: dict, required: list[str] | None = None) -> dict:
    """
    Build an OpenAI function-tool schema dict from a Python function.

    The function's docstring (retrieved with inspect.getdoc) becomes the
    'description' field the model reads when deciding whether to call the tool.
    Parameter descriptions live in the 'properties' dict passed by the caller.

    Args:
        func:       Python function whose __name__ and docstring are used.
        properties: Dict of {param_name: {"type": ..., "description": ...}}.
        required:   List of parameter names the model must always supply.
    """
    return {
        "type": "function",
        "function": {
            "name":        func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": {
                "type":       "object",
                "properties": properties,
                "required":   required or [],
            },
        },
    }


# OpenAI tool-schema list — passed as tools= in chat.completions.create()
AGENT_TOOLS: list[dict] = [
    _tool_schema(
        list_files,
        properties={
            "directory": {
                "type":        "string",
                "description": "Relative path inside the workspace to list. Defaults to '.' (root).",
            },
        },
    ),
    _tool_schema(
        read_file,
        properties={
            "path": {
                "type":        "string",
                "description": "Relative file path within the workspace (e.g. 'notes.txt').",
            },
        },
        required=["path"],
    ),
    _tool_schema(
        search_files,
        properties={
            "query": {
                "type":        "string",
                "description": "Case-insensitive text string to search for.",
            },
            "directory": {
                "type":        "string",
                "description": "Relative workspace path to scan. Defaults to '.'.",
            },
            "file_pattern": {
                "type":        "string",
                "description": "Glob filter for filenames, e.g. '*.txt'. Defaults to '*'.",
            },
        },
        required=["query"],
    ),
    _tool_schema(
        write_file,
        properties={
            "path": {
                "type":        "string",
                "description": "Relative file path for the new file (e.g. 'summary.txt').",
            },
            "content": {
                "type":        "string",
                "description": "Full text content to write to the file.",
            },
        },
        required=["path", "content"],
    ),
]

# Dispatch table used by _execute_tool()
_TOOL_FUNCTIONS: dict[str, callable] = {
    "list_files":   list_files,
    "read_file":    read_file,
    "search_files": search_files,
    "write_file":   write_file,
}


def _execute_tool(name: str, args: dict) -> str:
    """
    Dispatch a parsed tool call to the matching Python function.

    Always returns a string — the model needs a text result to continue.
    Runtime exceptions are caught and returned as error messages so the
    loop can continue rather than crashing the entire request.

    Args:
        name: Tool name (must be a key in _TOOL_FUNCTIONS).
        args: Keyword arguments parsed from the model's JSON arguments string.
    """
    func = _TOOL_FUNCTIONS.get(name)
    if func is None:
        return (
            f"Unknown tool: {name!r}. "
            f"Available tools: {', '.join(_TOOL_FUNCTIONS)}"
        )
    try:
        return func(**args)
    except ValueError as e:
        return f"Tool error ({name}): {e}"
    except Exception as e:
        logger.exception("Unexpected error executing tool %r", name)
        return f"Unexpected error in {name!r}: {type(e).__name__}: {e}"


# ===========================================================================
# AGENT LOOP
#
# run_agent_loop() replaces the direct client.chat.completions.create() call
# in app.py.  It streams normally for regular replies, but when the model
# emits tool_calls it buffers, executes, appends results, and loops — all
# while yielding SSE events the browser can display in real-time.
# ===========================================================================

def run_agent_loop(
    messages:      list[dict],
    client:        OpenAI,
    model:         str,
    temperature:   float,
    max_tokens:    int,
    result_holder: list,
) -> Generator[str, None, None]:
    """
    Run the model in an agentic loop, yielding SSE events throughout.

    On each turn the model either:
      a) Calls one or more tools  → execute them, append results, loop again.
      b) Produces a final answer  → stream it token-by-token and stop.

    The model autonomously decides whether to call tools based on the tool
    descriptions (docstrings) provided via AGENT_TOOLS.  Most conversational
    queries will produce a direct answer on turn 1 with no tools called.

    Mode detection per turn:
      The first substantive streaming chunk reveals the turn mode:
        delta.tool_calls present  → "buffer"  (collect tool calls, don't stream text)
        delta.content present     → "stream"  (yield delta events in real-time)
      This means regular answers are truly token-by-token streaming while
      tool-call turns wait for the full response before executing.

    SSE events yielded:
      data: {"tool_call":   {"id", "name", "args"}}        model invoked a tool
      data: {"tool_result": {"id", "name", "output"}}      tool result (truncated)
      data: {"delta":       "<token>"}                     final answer token
      data: {"done":        true, "tools_used", "usage"}   stream complete

    result_holder[0] is set to a dict with full_response, tools_used, usage
    before the final "done" event so the caller can persist the exchange.

    Args:
        messages:      Complete messages list [system, ...history, user].
        client:        Authenticated OpenAI client instance.
        model:         Chat model ID (e.g. 'gpt-4.1-mini').
        temperature:   Sampling temperature (0.0 – 2.0).
        max_tokens:    Upper bound on tokens in the final answer.
        result_holder: Single-element list; populated with result dict on exit.
    """
    local_messages = list(messages)
    tools_used:  list[str] = []
    usage_data:  dict      = {}

    for turn in range(MAX_TOOL_TURNS + 1):
        is_final_turn = (turn == MAX_TOOL_TURNS)

        # Build API call params — remove tools on the forced final turn so the
        # model must produce a text answer rather than calling more tools.
        api_params: dict = {
            "model":          model,
            "messages":       local_messages,
            "temperature":    temperature,
            "max_tokens":     max_tokens,
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        if not is_final_turn:
            api_params["tools"]       = AGENT_TOOLS
            api_params["tool_choice"] = "auto"

        stream = client.chat.completions.create(**api_params)

        # ------------------------------------------------------------------
        # Stream-reading state machine
        # mode = None      → undecided (first substantive chunk not seen yet)
        # mode = "stream"  → regular text, yielding delta events live
        # mode = "buffer"  → tool calls arriving, buffering arguments
        # ------------------------------------------------------------------
        mode:          str | None          = None
        content_parts: list[str]           = []
        finish_reason: str | None          = None
        # tool_calls_raw: index → {id, name, args_parts}
        tool_calls_raw: dict[int, dict]    = {}

        for chunk in stream:
            if chunk.usage:
                usage_data = {
                    "prompt_tokens":     chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens":      chunk.usage.total_tokens,
                }
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta

            # Detect mode from the first substantive chunk
            if mode is None:
                if delta.tool_calls:
                    mode = "buffer"
                elif delta.content:
                    mode = "stream"

            if mode == "stream" and delta.content:
                # Regular text — yield to the browser immediately (true streaming)
                content_parts.append(delta.content)
                yield f"data: {json.dumps({'delta': delta.content})}\n\n"

            elif mode == "buffer" and delta.tool_calls:
                # Tool call arguments arriving — accumulate them
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_raw:
                        tool_calls_raw[idx] = {
                            "id":        tc_delta.id or "",
                            "name":      (tc_delta.function.name or "") if tc_delta.function else "",
                            "args_parts": [],
                        }
                    else:
                        if tc_delta.id:
                            tool_calls_raw[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_raw[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_raw[idx]["args_parts"].append(
                                    tc_delta.function.arguments
                                )

        # ------------------------------------------------------------------
        # End of stream for this turn
        # ------------------------------------------------------------------
        if finish_reason == "tool_calls" and mode == "buffer":
            # Execute each tool call and collect results
            assistant_tool_calls: list[dict] = []
            tool_result_messages: list[dict] = []

            for idx in sorted(tool_calls_raw):
                tc      = tool_calls_raw[idx]
                args_str = "".join(tc["args_parts"])
                tc_id   = tc["id"] or f"call_{idx}"
                tc_name = tc["name"]

                try:
                    args_obj = json.loads(args_str) if args_str.strip() else {}
                except json.JSONDecodeError:
                    args_obj = {}

                # Notify the browser that a tool is being called
                yield f"data: {json.dumps({'tool_call': {'id': tc_id, 'name': tc_name, 'args': args_obj}})}\n\n"

                # Execute the tool
                output = _execute_tool(tc_name, args_obj)
                tools_used.append(tc_name)

                # Send a preview of the result to the browser (truncated for display)
                yield f"data: {json.dumps({'tool_result': {'id': tc_id, 'name': tc_name, 'output': output[:400]}})}\n\n"

                assistant_tool_calls.append({
                    "id":       tc_id,
                    "type":     "function",
                    "function": {"name": tc_name, "arguments": args_str},
                })
                tool_result_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc_id,
                    "content":      output,
                })

            # Append assistant message (with tool_calls) + tool results to history
            local_messages.append({
                "role":       "assistant",
                "content":    None,
                "tool_calls": assistant_tool_calls,
            })
            local_messages.extend(tool_result_messages)
            # Loop to next turn — the model will now formulate its answer

        else:
            # Model produced a final text answer (finish_reason == "stop")
            full_response = "".join(content_parts)
            result_holder[0] = {
                "full_response": full_response,
                "tools_used":    tools_used,
                "usage":         usage_data,
            }
            done_payload = {
                "done":        True,
                "tools_used":  tools_used,
                "usage":       usage_data,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            return

    # Safety net: MAX_TOOL_TURNS exhausted — shouldn't reach here because
    # the final turn runs without tools, forcing a stop.
    result_holder[0] = {"full_response": "", "tools_used": tools_used, "usage": usage_data}
    yield f"data: {json.dumps({'done': True, 'tools_used': tools_used, 'usage': usage_data})}\n\n"
