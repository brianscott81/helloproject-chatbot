# Hello! Project Wiki Chatbot

A question-answering chatbot over the [Hello! Project Fandom Wiki](https://helloproject.fandom.com/).
Given the wiki's XML dump, it builds a local SQLite index and a vector store, then answers
natural-language questions using a hybrid structured + semantic search pipeline.

## What it answers

- **Precise** questions like *"What was the second track of Minimoni's second album?"* —
  answered deterministically from the parsed CD Infobox + tracklist data.
- **Discography** questions like *"List all albums by Morning Musume"* — sorted chronologically.
- **Fuzzy** questions like *"Who produced CRAZY ABOUT YOU?"* or *"Tell me about Minimoni"* —
  answered via semantic search over the wiki text, enriched with infobox facts.

## Architecture

```
MediaWiki XML dump
        │
        ▼
   build_index.py       ──► helloproject.db  (SQLite, 11.7k pages, ~100 MB)
        │                  pages, infoboxes, tracklists, redirects, aliases, links
        ▼
   build_embeddings.py  ──► chroma/         (ChromaDB, ~55k chunks, ~150 MB)
        │                  BAAI/bge-small-en-v1.5 embeddings
        ▼
   chat.py              ──► answer to user question
        │
        ├─► classify (regex router)
        ├─► structured tool  (lookup_track / list_releases / get_song_info)
        ├─► semantic tool    (vector search + infobox enrichment)
        └─► LLM synthesis    (Anthropic, OpenAI, Ollama, or MiniMax via Anthropic-compat)
                            falls back to deterministic template formatter
```

## Quick start

### 1. Build the indexes (one-time, ~25 min total)

```bash
# 1a. SQLite index (~75 seconds)
python build_index.py path/to/wiki_dump.xml ./helloproject.db

# 1b. Vector index (~25 minutes for ~50k chunks on CPU)
python build_embeddings.py ./helloproject.db ./chroma
```

### 2. Ask questions

```bash
# Single question
python chat.py -q "What was the second track of Minimoni's second album?"

# Interactive REPL
python chat.py -i

# Web interface (browser-based chat UI)
python web.py                  # serves on http://127.0.0.1:8000
python web.py --port 9000      # custom port
python web.py --no-llm         # template formatters only
python web.py --host 0.0.0.0   # expose on a network (use with caution)

# Run the test suite (9/9 should pass)
python e2e_test.py
python e2e_test.py --structured-only   # skip semantic tests
python e2e_test.py --use-llm           # enable LLM synthesis (needs API keys)
python e2e_test.py --llm-status        # just show which provider is detected
```

## LLM configuration

The chatbot can synthesize natural-prose answers using any of these LLM providers.
It auto-detects which one to use based on environment variables.

### MiniMax (Anthropic-compatible) — recommended setup

MiniMax exposes an Anthropic-compatible endpoint. Configure it the same way as Claude Code:

```bash
# PowerShell
$env:ANTHROPIC_AUTH_TOKEN = "***"
$env:ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
# Optional: pick a specific MiniMax model
$env:HELLO_PROJECT_ANTHROPIC_MODEL = "MiniMax-M3"

python chat.py -q "..."
```

### Real Anthropic

```bash
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python chat.py -q "..."
```

### OpenAI

```bash
$env:OPENAI_API_KEY = "sk-..."
python chat.py -q "..."
```

### Local Ollama

```bash
# Default model: llama3.2, host: localhost:11434
ollama pull llama3.2
python chat.py -q "..."
```

### No LLM (deterministic template answers)

If no API keys are set, the chatbot falls back to a deterministic template formatter.
Answers still contain the right facts, just in a less-polished format.

## Example queries

| Query | Answer |
|---|---|
| `What was the second track of Minimoni's second album?` | **CRAZY ABOUT YOU** |
| `2nd track on Morning Musume 5th album` | Summer Night Town |
| `What was the third track of Berryz Koubou's first album?` | Nicchoku |
| `List all albums by Morning Musume` | 34 albums, chronologically |
| `Who produced CRAZY ABOUT YOU?` | Producer: Tsunku |
| `Tell me about Minimoni` | Synthesized from semantic chunks |

## Files

- `build_index.py` — streaming XML → SQLite parser (uses mwxml + mwparserfromhell)
- `query.py` — high-level query API (resolve_title, find_album_for_artist, etc.)
- `build_embeddings.py` — wikitext → ChromaDB with BAAI/bge-small-en-v1.5
- `chat.py` — CLI + classifier + tool dispatch + template formatters
- `conversation.py` — per-session conversation state (turns, prior-turn context for LLM)
- `llm.py` — LLM synthesis layer (4 providers, auto-detect, graceful fallback) + tool-use routing
- `web.py` — stdlib HTTP server wrapping chat for the browser
- `web/index.html`, `web/chat.js`, `web/chat.css` — chat UI (Markdown via marked.js CDN)
- `demo.py` — quick interactive sanity checks
- `e2e_test.py` — 9-question test battery

## What's *not* in git

The following are listed in `.gitignore` and rebuilt from the source XML dump:

- `helloproject.db` (~106 MB) — SQLite index
- `chroma/` (~150 MB) — vector store
- `__pycache__/`
- `build.log`, `build_emb.log`

To regenerate them: see **Quick start** above.