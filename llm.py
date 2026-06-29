"""
LLM synthesis layer for the Hello! Project wiki chatbot.

Takes the structured `tool_result` produced by chat.py and turns it into
natural-language prose via an LLM. Falls back to deterministic template
formatters if no LLM is configured.

The design choice: the LLM only does *synthesis*. Routing (which tool to
call for which question) and data lookup stay in deterministic code.
This avoids a class of LLM failure modes — the model can't mis-route
a question to the wrong tool or hallucinate a track number, because
those decisions happen in code that just runs.

Provider detection (in order):
  1. ANTHROPIC_API_KEY → Anthropic (Claude)
  2. OPENAI_API_KEY    → OpenAI (GPT-4o-mini or similar)
  3. OLLAMA_HOST       → Ollama (local)
  4. None of the above → template fallback (no LLM call)

Override with env var HELLO_PROJECT_LLM_PROVIDER:
  anthropic / openai / ollama / none / auto

The LLM is given:
  - The system prompt (defines the persona + ground-truth rules)
  - The user's question
  - The tool result formatted as a compact JSON-ish text block

It returns a 1-3 paragraph natural-language answer.

If the LLM call fails (network error, rate limit, etc.) we log a warning
and fall back to the template formatter so the CLI never breaks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# Local imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from query import (
    AlbumMatch,
    Page,
    answer_track_position,
    connect,
    find_album_for_artist,
    find_artist_page,
    get_song_info,
    get_tracklist,
    resolve_title,
)

# Make sure our log output goes to stderr and isn't suppressed by default.
log = logging.getLogger("llm")
if not log.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("llm: %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider() -> str:
    """Return one of: 'anthropic', 'openai', 'ollama', 'minimax', 'none'.

    Honors HELLO_PROJECT_LLM_PROVIDER if set. Otherwise auto-detects.

    Auth signal priority for the Anthropic-compatible path:
      1. ANTHROPIC_API_KEY
      2. ANTHROPIC_AUTH_TOKEN  (Claude Code / MiniMax-style config)

    If ANTHROPIC_BASE_URL is set to a non-Anthropic endpoint (like
    api.minimax.io/anthropic), the Anthropic SDK will route there
    automatically. We don't need a separate 'minimax' branch in the
    detection layer — the SDK does the right thing.
    """
    forced = os.environ.get("HELLO_PROJECT_LLM_PROVIDER", "").lower().strip()
    if forced == "none":
        return "none"
    if forced in ("anthropic", "openai", "ollama", "minimax"):
        # Even with forced provider, verify the credentials are present.
        if forced == "anthropic" and not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            log.warning("HELLO_PROJECT_LLM_PROVIDER=anthropic but neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set; falling back.")
            return "none"
        if forced == "openai" and not os.environ.get("OPENAI_API_KEY"):
            log.warning("HELLO_PROJECT_LLM_PROVIDER=openai but OPENAI_API_KEY is not set; falling back.")
            return "none"
        if forced == "minimax" and not os.environ.get("MINIMAX_API_KEY"):
            log.warning("HELLO_PROJECT_LLM_PROVIDER=minimax but MINIMAX_API_KEY is not set; falling back.")
            return "none"
        return forced

    # Auto-detect
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if has_anthropic and not os.environ.get("MINIMAX_API_KEY"):
        return "anthropic"
    if os.environ.get("MINIMAX_API_KEY"):
        return "minimax"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    # Ollama: probe the local server. If we can't reach it, fall through.
    if _ollama_available():
        return "ollama"
    return "none"


def _ollama_available() -> bool:
    """Quick TCP probe to see if Ollama is listening."""
    import socket
    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    if ":" in host:
        h, p = host.rsplit(":", 1)
        port = int(p)
    else:
        h, port = host, 11434
    try:
        with socket.create_connection((h, port), timeout=1.0):
            return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a knowledgeable, friendly assistant for the Hello! Project Wiki, \
a Fandom wiki about the Japanese idol collective Hello! Project.

GROUND TRUTH RULES (strict):
- Use ONLY the data provided in the tool result. Do not invent facts, \
release dates, track titles, members, or any other details.
- If the tool returned an error, explain the limitation honestly and \
suggest how the user might reformulate.
- Preserve specific numbers (track positions, release dates, sales \
figures, etc.) exactly as given.

VOICE & FORMAT:
- Be concise. Lead with the answer.
- For track-position questions, name the track prominently and offer \
to provide more info about the song if the wiki has it.
- For discography listings, give a numbered list with release dates.
- For fuzzy/semantic questions, synthesize a 2-4 sentence answer that \
answers the question directly, then cite the source page(s) you used.
- Use the artists' preferred romanizations: 'Morning Musume', \
'Minimoni', '℃-ute', 'Berryz Koubou', etc.

WHEN THE TOOL RETURNS MULTIPLE OPTIONS:
- If the user asks about an ordinal ("2nd album") and the artist has \
both the 2nd numbered album AND a compilation at the same position, \
pick the most likely interpretation (numbered studio albums in \
chronological order). Mention the ambiguity briefly if both are relevant.
"""


# Block appended to the system prompt when prior conversation turns
# are available. Tells the model how to use the history without
# hallucinating references.
PRIOR_TURNS_PROMPT_BLOCK = """\

CONTINUING A CONVERSATION:
- The PRIOR TURNS section below shows recent exchanges with the user.
- You may reference them when the user uses pronouns ("it", "they")
  or follow-up phrases ("how about...", "what about...") that depend
  on context you already established.
- ONLY reference turns that are explicitly shown. Do not invent prior
  exchanges ("as you mentioned earlier..." when nothing was mentioned).
- If the prior context is irrelevant to the current question, ignore it.
"""


def build_system_prompt(prior_turns_str: str | None = None) -> str:
    """Return the system prompt, optionally extended with prior-turn
    instructions and content.

    prior_turns_str: a block already formatted by
        conversation.format_prior_turns_for_llm(). If None or empty,
        the system prompt is returned unchanged.
    """
    if not prior_turns_str:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + PRIOR_TURNS_PROMPT_BLOCK + "\n\nPRIOR TURNS:\n" + prior_turns_str


# ---------------------------------------------------------------------------
# Tool-result serialization for the LLM
# ---------------------------------------------------------------------------

def serialize_for_llm(tool_name: str, tool_result: dict) -> str:
    """Turn a tool_result dict into a compact, LLM-friendly text block.

    The goal is to give the model exactly the facts it needs, no more, no
    less. Wikilinks are stripped, infobox fields are normalized, and the
    format is consistent across tool types so the model can learn the
    structure.
    """
    if "error" in tool_result:
        return f"ERROR: {tool_result['error']}\n" + _serialize_extras(tool_result)

    if tool_name == "lookup_track" and "album" in tool_result:
        return _serialize_track(tool_result)
    if tool_name == "list_releases" and "releases" in tool_result:
        return _serialize_releases(tool_result)
    if tool_name == "get_song_info" and "artist" in tool_result:
        return _serialize_song(tool_result)
    if tool_name == "semantic_search" and "chunks" in tool_result:
        return _serialize_semantic(tool_result)
    return json.dumps(tool_result, ensure_ascii=False, indent=2)[:4000]


def _strip_wikilink(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"<br\s*/?>", "; ", s)
    return s.strip()


def _serialize_extras(d: dict) -> str:
    """Pull 'available_releases' / 'tracks' lists into the serialized form."""
    out = []
    if "available_releases" in d:
        out.append("Available releases:")
        for r in d["available_releases"]:
            title = _strip_wikilink(r.get("title", ""))
            date = r.get("release_date") or "?"
            out.append(f"  - {title} ({date})")
    if "tracks" in d:
        out.append("Tracks on this album:")
        for t in d["tracks"][:30]:
            pos = t.get("position", "?")
            raw = _strip_wikilink(t.get("raw", ""))
            linked = _strip_wikilink(t.get("linked_title") or "")
            line = f"  {pos}. {raw}"
            if linked and linked != raw:
                line += f"  [{linked}]"
            out.append(line)
        if len(d["tracks"]) > 30:
            out.append(f"  ... and {len(d['tracks']) - 30} more")
    return "\n".join(out)


def _serialize_track(r: dict) -> str:
    album = r["album"]
    track = r["track"]
    song = r.get("song_info") or {}
    artist = r.get("artist", {}).get("title", "")

    lines = [
        f"TOOL RESULT: track lookup",
        f"Artist: {artist}",
        f"Album: {album.get('title')} (album #{album.get('album_number')}, "
        f"released {album.get('release_date') or 'unknown'})",
        f"Track #{track.get('position')}: {_strip_wikilink(track.get('raw', ''))}"
        + (f"  →  Song page: {track.get('linked_title')}" if track.get("linked_title") else ""),
    ]
    if song:
        lines.append("")
        lines.append(f"Song info for {song.get('title', track.get('linked_title'))}:")
        for key in ("artist", "released", "type", "album", "genre", "format",
                    "length", "label", "producer"):
            v = song.get(key)
            if v:
                lines.append(f"  {key.title()}: {_strip_wikilink(v)}")
        if song.get("intro"):
            intro = _strip_wikilink(song["intro"])[:600]
            lines.append(f"  Intro: {intro}")
    return "\n".join(lines)


def _serialize_releases(r: dict) -> str:
    lines = [
        f"TOOL RESULT: list of {r.get('kind', 'releases')}",
        f"Artist: {r.get('artist')}",
        f"Count: {r.get('count')}",
        "",
    ]
    for rel in r.get("releases", []):
        title = rel.get("title", "?")
        date = rel.get("release_date") or "?"
        n = rel.get("n", "?")
        tc = rel.get("track_count")
        tc_str = f", {tc} tracks" if tc else ""
        lines.append(f"  #{n}  {title} ({date}{tc_str})")
    return "\n".join(lines)


def _serialize_song(r: dict) -> str:
    lines = [
        f"TOOL RESULT: song info",
        f"Title: {r.get('title')}",
    ]
    for key in ("artist", "released", "type", "album", "genre", "format",
                "length", "label", "producer"):
        v = r.get(key)
        if v:
            lines.append(f"  {key.title()}: {_strip_wikilink(v)}")
    if r.get("intro"):
        lines.append(f"  Intro: {_strip_wikilink(r['intro'])[:800]}")
    return "\n".join(lines)


def _serialize_semantic(r: dict) -> str:
    lines = [
        f"TOOL RESULT: semantic search for '{r.get('query')}'",
        f"Top {len(r.get('chunks', []))} matches:",
        "",
    ]
    for i, chunk in enumerate(r.get("chunks", [])[:5], start=1):
        score = chunk.get("score")
        score_s = f" (score {score:.2f})" if score is not None else ""
        lines.append(f"Match {i}: {chunk.get('page_title')} — {chunk.get('section')}{score_s}")
        # Strip the chunk text
        text = re.sub(r"\{\{[^}]*\}\}", "", chunk.get("text", ""))
        text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
        text = text.strip()[:600]
        lines.append(f"  Text: {text}")
        if chunk.get("infobox_facts"):
            lines.append("  Key facts:")
            for fact in chunk["infobox_facts"][:10]:
                lines.append(f"    - {fact}")
        lines.append("")
    return "\n".join(lines)

def call_anthropic(
    question: str,
    tool_name: str,
    tool_result: dict,
    model: str | None = None,
    prior_turns_str: str | None = None,
) -> str | None:
    """Call Anthropic's API (or any Anthropic-compatible endpoint).

    The Anthropic SDK supports two auth styles — `api_key` (sk-ant-...)
    and `auth_token` (the Claude Code / MiniMax convention). We pick
    whichever the user has set. The SDK also honors `ANTHROPIC_BASE_URL`
    for routing to compatible endpoints like `https://api.minimax.io/anthropic`.

    Environment variables:
      ANTHROPIC_API_KEY              (preferred for real Anthropic)
      ANTHROPIC_AUTH_TOKEN           (Claude Code / MiniMax-style auth)
      ANTHROPIC_BASE_URL             (override endpoint)
      HELLO_PROJECT_ANTHROPIC_MODEL  (default: claude-3-5-haiku-latest)

    prior_turns_str: optional block of recent conversation history
        formatted by conversation.format_prior_turns_for_llm(). When
        provided, the system prompt is extended with it.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        log.warning("anthropic package not installed")
        return None

    # Decide which credential to use.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key and not auth_token:
        log.warning("Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set")
        return None

    # Default model: use MiniMax-M3 when routing through MiniMax, or
    # claude-3-5-haiku-latest otherwise. Users can always override with
    # HELLO_PROJECT_ANTHROPIC_MODEL.
    default_model = "claude-3-5-haiku-latest"
    if base_url and "minimax" in base_url.lower():
        default_model = "MiniMax-M3"
    model = model or os.environ.get("HELLO_PROJECT_ANTHROPIC_MODEL", default_model)

    # Build the client. Pass auth_token explicitly when we don't have a
    # real Anthropic api_key — that way the SDK sends x-api-key only if
    # we set it, and the proxy (MiniMax) gets its expected auth header.
    if api_key:
        client_kwargs = {"api_key": api_key}
    else:
        client_kwargs = {"auth_token": auth_token}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = Anthropic(**client_kwargs)

    serialized = serialize_for_llm(tool_name, tool_result)
    user_msg = f"Question: {question}\n\n{serialized}"

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            system=build_system_prompt(prior_turns_str),
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.warning(f"Anthropic call failed: {e}")
        return None


def call_openai(
    question: str,
    tool_name: str,
    tool_result: dict,
    model: str | None = None,
    prior_turns_str: str | None = None,
) -> str | None:
    """Call OpenAI's API. Returns None on failure (caller falls back).

    prior_turns_str: optional block of recent conversation history
        formatted by conversation.format_prior_turns_for_llm(). When
        provided, the system prompt is extended with it.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed")
        return None

    model = model or os.environ.get("HELLO_PROJECT_OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()
    serialized = serialize_for_llm(tool_name, tool_result)
    user_msg = f"Question: {question}\n\n{serialized}"

    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": build_system_prompt(prior_turns_str)},
                {"role": "user", "content": user_msg},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"OpenAI call failed: {e}")
        return None


def call_minimax(
    question: str,
    tool_name: str,
    tool_result: dict,
    model: str | None = None,
    prior_turns_str: str | None = None,
) -> str | None:
    """Call MiniMax's Anthropic-compatible API. Returns None on failure.

    MiniMax exposes an Anthropic-compatible chat completions endpoint
    at https://api.minimax.io/anthropic. We use the Anthropic SDK with
    a custom base_url, which routes to MiniMax under the hood.

    This matches the configuration documented at:
      https://platform.minimax.io/docs/token-plan/claude-code

    Environment variables:
      MINIMAX_API_KEY                       (required) Auth token (Bearer)
      MINIMAX_HOST                          (optional) Override base URL;
                                             default https://api.minimax.io/anthropic
      HELLO_PROJECT_MINIMAX_MODEL           (optional) Model name override;
                                             default MiniMax-M3

    IMPORTANT: Clear any real Anthropic env vars (ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL) before running so the
    SDK doesn't try to talk to api.anthropic.com. The detect_provider()
    function explicitly checks for ANTHROPIC_API_KEY first — if you have
    both set, Anthropic wins unless you force HELLO_PROJECT_LLM_PROVIDER=minimax.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        log.warning("anthropic package not installed (needed for MiniMax)")
        return None

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        log.warning("MINIMAX_API_KEY not set")
        return None

    base_url = os.environ.get("MINIMAX_HOST", "https://api.minimax.io/anthropic")
    model = model or os.environ.get("HELLO_PROJECT_MINIMAX_MODEL", "MiniMax-M3")

    # The Anthropic SDK accepts base_url and api_key directly. The auth
    # header will be sent as `x-api-key: <api_key>` (Anthropic's expected
    # format) which is what MiniMax expects per their docs.
    client = Anthropic(base_url=base_url, api_key=api_key)

    serialized = serialize_for_llm(tool_name, tool_result)
    user_msg = f"Question: {question}\n\n{serialized}"

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            system=build_system_prompt(prior_turns_str),
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.warning(f"MiniMax call failed: {e}")
        return None


def call_ollama(
    question: str,
    tool_name: str,
    tool_result: dict,
    model: str | None = None,
    prior_turns_str: str | None = None,
) -> str | None:
    """Call a local Ollama server. Returns None on failure (caller falls back).

    prior_turns_str: optional block of recent conversation history
        formatted by conversation.format_prior_turns_for_llm(). When
        provided, the system prompt is extended with it.
    """
    import urllib.request
    import urllib.error

    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    model = model or os.environ.get("HELLO_PROJECT_OLLAMA_MODEL", "llama3.2")
    serialized = serialize_for_llm(tool_name, tool_result)

    payload = json.dumps({
        "model": model,
        "prompt": f"{build_system_prompt(prior_turns_str)}\n\n---\n\nQuestion: {question}\n\n{serialized}\n\nAnswer:",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 600},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
            return (data.get("response") or "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
        log.warning(f"Ollama call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tool-use (function-calling) for routing fallback
# ---------------------------------------------------------------------------

# JSON schemas for each structured tool. These are passed to the LLM
# as a `tools=` parameter; the model picks one and supplies args. This
# replaces our regex-based classifier as a FALLBACK for ambiguous
# questions that the regex doesn't handle (Phase 1 of the LLM-tool
# integration plan).
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "lookup_track",
        "description": (
            "Look up the Nth track of an artist's Mth album. Use for "
            "questions like 'what is track 5 of X's 3rd album' or "
            "'what is the second track of Minimoni's 2nd album'. "
            "Returns the exact track title plus song infobox data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name as it appears on the wiki, e.g. 'Minimoni', 'Morning Musume', 'C-ute'.",
                },
                "album_position": {
                    "type": "integer",
                    "description": "Which album in chronological release order (1-indexed). 1st album, 2nd album, etc.",
                },
                "track_position": {
                    "type": "integer",
                    "description": "Track number on the album (1-indexed).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["album", "single"],
                    "description": "Release type. Default 'album'.",
                },
            },
            "required": ["artist", "album_position", "track_position"],
        },
    },
    {
        "name": "list_releases",
        "description": (
            "List albums or singles by an artist, optionally filtered to a "
            "specific year. Use for 'list all albums by X', 'what singles "
            "did X release', 'show me X's discography', or 'X's 2001 "
            "singles'. Returns a chronological list with release dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name as it appears on the wiki.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["album", "single"],
                    "description": "Which type of release to list. Default 'album'.",
                },
                "year": {
                    "type": "integer",
                    "description": (
                        "Optional. If given, only releases from that year "
                        "are returned. Use this when the user asks about "
                        "a specific year ('X's 2003 albums', 'singles in 2001')."
                    ),
                },
            },
            "required": ["artist"],
        },
    },
    {
        "name": "get_tracklist",
        "description": (
            "Get the full tracklist of a release (album or single). Use "
            "when the user asks 'what are the tracks on X', 'what's the "
            "tracklist of X', or asks for individual track titles on a "
            "release. Returns each track's position and title in order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "release_title": {
                    "type": "string",
                    "description": (
                        "Title of the release (album or single) to fetch "
                        "the tracklist for, e.g. 'Minimoni Songs 2' or "
                        "'Last Kiss'."
                    ),
                },
            },
            "required": ["release_title"],
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Fuzzy / open-ended search across the wiki. Use when the "
            "question is about history, biography, general facts, or "
            "when you don't know which specific page to look at. Returns "
            "top-matching wiki chunks with their infobox facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Should be a short keyword phrase or question, not a full sentence.",
                },
            },
            "required": ["query"],
        },
    },
]


def select_tool_with_llm(
    question: str,
    prior_turns_str: str | None = None,
) -> dict | None:
    """Ask the LLM to pick a structured tool for this question.

    Returns a dict like {"tool": "lookup_track", "args": {...}} or
    {"tool": "semantic_search", "args": {"query": "..."}}, or None if
    the LLM call fails / provider doesn't support tools / model didn't
    pick a tool.

    Used as a fallback when the regex classifier in chat.py returns
    the default semantic_search fallback (i.e. when no structured
    pattern matched). The model has prior-turn context via the same
    `prior_turns_str` mechanism as the synthesis path.
    """
    # We route through the Anthropic SDK regardless of provider name,
    # because the MiniMax-compat endpoint accepts the Anthropic SDK's
    # tool-use API. OpenAI and Ollama don't (their API is different).
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        log.warning("select_tool_with_llm: no Anthropic-style credentials")
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        log.warning("select_tool_with_llm: anthropic package not installed")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if os.environ.get("ANTHROPIC_API_KEY"):
        client_kwargs = {"api_key": api_key}
    else:
        client_kwargs = {"auth_token": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = Anthropic(**client_kwargs)

    # Default to MiniMax-M3 when routing through MiniMax, else Haiku.
    default_model = "claude-3-5-haiku-latest"
    if base_url and "minimax" in base_url.lower():
        default_model = "MiniMax-M3"
    model = os.environ.get("HELLO_PROJECT_ANTHROPIC_MODEL", default_model)

    # Build system prompt. Tool-use doesn't need the full synthesis
    # system prompt; just enough to constrain the model to picking a tool.
    system_prompt = (
        "You are a routing assistant. Pick the right tool for the "
        "user's question and supply its arguments. Use prior turns to "
        "resolve pronouns ('it', 'they') and relative references "
        "('next year', 'previous album'). Only call a tool — never "
        "answer in prose."
    )

    user_msg = question
    if prior_turns_str:
        user_msg = f"{prior_turns_str}\n\nCurrent question: {question}"

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "any"},  # force tool use, no plain text
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        log.warning(f"select_tool_with_llm: API call failed: {e}")
        return None

    # Find the tool_use block in the response.
    for block in resp.content:
        # ToolUseBlock has a .type attr of 'tool_use' and .name / .input
        if getattr(block, "type", None) == "tool_use":
            return {
                "tool": block.name,
                "args": dict(block.input) if block.input else {},
            }

    # No tool_use block. Model returned plain text instead of calling
    # a tool. This shouldn't happen with tool_choice=any, but we handle
    # gracefully.
    log.warning("select_tool_with_llm: model did not call a tool")
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class LLMSynthesizer:
    """Wraps provider detection + call dispatch with a graceful fallback.

    Usage:
        synth = LLMSynthesizer()
        if synth.available:
            answer = synth.synthesize(question, tool_name, tool_result)
        else:
            # fall back to template formatters
            ...
    """

    def __init__(self, provider: str | None = None):
        self.provider = provider or detect_provider()
        self.available = self.provider != "none"

    def synthesize(
        self,
        question: str,
        tool_name: str,
        tool_result: dict,
        *,
        max_retries: int = 1,
        prior_turns_str: str | None = None,
    ) -> str | None:
        """Run the LLM. Returns the synthesized answer, or None on failure.

        prior_turns_str: optional pre-formatted block of recent
            conversation turns (use conversation.format_prior_turns_for_llm()
            to build it). When provided, the system prompt is extended
            with it so the LLM can produce context-aware answers.
        """
        if not self.available:
            return None

        for attempt in range(max_retries + 1):
            if self.provider == "anthropic":
                result = call_anthropic(
                    question, tool_name, tool_result,
                    prior_turns_str=prior_turns_str,
                )
            elif self.provider == "openai":
                result = call_openai(
                    question, tool_name, tool_result,
                    prior_turns_str=prior_turns_str,
                )
            elif self.provider == "minimax":
                result = call_minimax(
                    question, tool_name, tool_result,
                    prior_turns_str=prior_turns_str,
                )
            elif self.provider == "ollama":
                result = call_ollama(
                    question, tool_name, tool_result,
                    prior_turns_str=prior_turns_str,
                )
            else:
                return None

            if result:
                return result
            if attempt < max_retries:
                log.info(f"LLM call failed (attempt {attempt + 1}), retrying...")
        return None

    def describe(self) -> str:
        if not self.available:
            return "LLM: disabled (no API keys, no Ollama; using template fallback)"
        return f"LLM: enabled, provider={self.provider}"


if __name__ == "__main__":
    # Quick CLI: detect provider and print status.
    logging.basicConfig(level=logging.INFO)
    s = LLMSynthesizer()
    print(s.describe())
    if s.available:
        # Smoke test
        fake_result = {
            "artist": {"title": "Minimoni"},
            "album": {"title": "Minimoni Songs 2", "album_number": 2, "release_date": "2004-02-11"},
            "track": {"position": 2, "raw": "[[CRAZY ABOUT YOU]]", "linked_title": "CRAZY ABOUT YOU"},
            "song_info": {
                "title": "CRAZY ABOUT YOU", "artist": "Minimoni",
                "released": "October 16, 2003", "producer": "Tsunku",
                "intro": "CRAZY ABOUT YOU is the tenth single by the Morning Musume subgroup Minimoni.",
            },
        }
        print("\nSmoke test:")
        out = s.synthesize(
            "What was the second track of Minimoni's second album?",
            "lookup_track",
            fake_result,
        )
        print(out if out else "(LLM returned nothing)")