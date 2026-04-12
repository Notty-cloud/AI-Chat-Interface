# Business Intelligence Context

This document describes the business context data the AI Chat Interface agent
relies on, where it comes from, and how the agent uses it.

---

## Overview

The agent's RAG pipeline is pre-loaded at startup with four business context
documents stored in the `data/` folder. These files give the agent permanent,
session-independent knowledge about its own domain — so it can answer questions
about its capabilities, policies, and performance without requiring the user to
upload anything.

---

## Data Files

### 1. `data/translation_style_guide.txt`

| Property | Detail |
|---|---|
| **What it is** | Internal style guide for translation quality and consistency |
| **Source** | Synthetic — modelled on real translation bureau style guides |
| **Format** | Plain text, structured with sections |
| **Size** | ~4,200 characters |

**Contents:**
- General translation principles (accuracy, tone, register preservation)
- Language-specific rules for Arabic, Japanese, Spanish, French, and Chinese
- Image text extraction standards (reading order, low-quality images, mixed-language docs)
- Output format standards (required response structure)
- What the agent will not translate (hate speech, confidential content, personal data)

**How the agent uses it:**
When a user asks *"How should you handle ambiguous translations?"* or *"Will you translate handwritten text?"*, the RAG pipeline retrieves the relevant section of this guide and the agent answers from it — consistently, every time.

---

### 2. `data/language_support_matrix.txt`

| Property | Detail |
|---|---|
| **What it is** | Reference table of all supported languages and their capability levels |
| **Source** | Synthetic — based on OpenAI GPT-4o and text-embedding-3-small documented capabilities |
| **Format** | Plain text with structured tables |
| **Size** | ~3,800 characters |

**Contents:**
- Tier 1 (High Confidence): 15 languages with full chat, image, and RAG support
- Tier 2 (Medium Confidence): 16 languages with partial support
- Tier 3 (Experimental): 13 languages with chat and image support only
- Known limitations: handwriting, archaic languages, technical jargon, dialects
- Roadmap: upcoming languages planned for 2026

**How the agent uses it:**
When a user asks *"Can you translate Swahili?"* or *"What languages do you support?"*, the agent retrieves the relevant tier from this matrix and gives a specific, accurate answer rather than a generic response.

---

### 3. `data/usage_policy.txt`

| Property | Detail |
|---|---|
| **What it is** | Acceptable use policy for the translation service |
| **Source** | Synthetic — modelled on standard SaaS acceptable use policies |
| **Format** | Plain text, structured with numbered sections |
| **Size** | ~3,100 characters |

**Contents:**
- Permitted uses: personal, academic, business, RAG indexing, public content
- Prohibited uses: hate speech, confidential third-party content, personal data, illegal content
- Data handling: session-only storage, OpenAI API processing, no training use
- Accuracy disclaimer
- Enforcement

**How the agent uses it:**
The agent's query classifier routes policy questions to RAG only (no web fallback). When a user asks *"Is it safe to upload a contract?"* or *"What data do you store?"*, the agent retrieves the exact relevant policy section. This is also the document that triggers refusal responses for prohibited content.

---

### 4. `data/performance_metrics.txt`

| Property | Detail |
|---|---|
| **What it is** | Q1 2026 operational metrics report |
| **Source** | Synthetic — realistic figures generated for demonstration purposes |
| **Format** | Plain text report with tables and statistics |
| **Size** | ~4,600 characters |

**Contents:**
- Usage statistics: 4,847 sessions, 38,214 messages, 1,876 image translations
- Language distribution: top 9 languages by volume
- Translation accuracy by language tier and image quality
- Agent routing performance: 96.8% classification accuracy
- Token usage and cost analysis
- Error rates and most common user complaints
- Recommendations for next development phase

**How the agent uses it:**
When a user or department head asks *"How accurate is the translation?"* or *"What languages are used most?"*, the agent retrieves the relevant metric from this report. This is particularly useful for the department presentation — the agent can answer performance questions on the fly.

---

## How the Data Loads

All four files are indexed automatically when the server starts via the
`preload_business_context()` function in `app.py`:

```
Server starts
     │
     ▼
preload_business_context()
     │
     ├── reads data/translation_style_guide.txt
     ├── reads data/language_support_matrix.txt
     ├── reads data/usage_policy.txt
     └── reads data/performance_metrics.txt
           │
           ▼ (for each file)
     rag.add_document()
           │
           ├── extract text
           ├── sentence-aware chunk (~500 chars, 1-sentence overlap)
           ├── embed all chunks in one batched API call
           └── store in memory with filename tag
```

The data persists for the full server lifetime and is available to every
session without any manual upload required.

---

## Example Queries This Data Enables

| User asks | Data source | How agent answers |
|---|---|---|
| "Can you translate Haitian Creole?" | language_support_matrix.txt | "Yes — Tier 2 support, medium confidence" |
| "What data do you store about me?" | usage_policy.txt | Cites the Data Handling section |
| "How accurate is Arabic translation?" | performance_metrics.txt | "92.6% accuracy — dialect variation is a known factor" |
| "Will you translate a blurry photo?" | translation_style_guide.txt | "Yes, with [unclear] tags for uncertain sections" |
| "What languages are most commonly used?" | performance_metrics.txt | "Spanish 28.4%, French 17.2%, Arabic 14.8%..." |
| "Why won't you translate this content?" | usage_policy.txt | Cites the Prohibited Uses section |

---

## Why This Matters (vs. No Business Context)

| Without pre-loaded context | With pre-loaded context |
|---|---|
| "I support many languages" (vague) | Cites exact tiers, accuracy %, limitations |
| Generic policy refusals | Quotes specific policy section |
| Cannot answer performance questions | Retrieves real Q1 2026 metrics |
| Every session starts with zero knowledge | Agent always has domain authority |

---

## Data Location

```
AI_Chat_Interface/
└── data/
    ├── translation_style_guide.txt   # Style and quality standards
    ├── language_support_matrix.txt   # Supported languages and tiers
    ├── usage_policy.txt              # Acceptable use policy
    └── performance_metrics.txt       # Q1 2026 operational report
```

All files are plain text, version-controlled in the repository, and
loaded via the RAG pipeline — no database or external service required.
