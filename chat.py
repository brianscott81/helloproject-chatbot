"""
Chat layer for the Hello! Project wiki.

This is the user-facing interface that ties together:
  - The structured query path (deterministic, precise questions)
  - The semantic search path (fuzzy questions, "tell me about...")
  - An LLM that routes between them and synthesizes the final answer

The chat layer exposes two "tools" that the LLM can call:

  1. lookup_track(artist, album_position, track_position)
     → returns {album_title, track_title, song_info_excerpt}
     → handles: "What's the Nth track of X's Mth album?"

  2. semantic_search(query, k)
     → returns top-k chunks of wiki text matching the query semantically
     → handles: "Tell me about...", "What happened when...", etc.

There's also an implicit "lookup_song" path for when the question
resolves to a single song page (e.g. "Who produced CRAZY ABOUT YOU?").
The LLM can synthesize this from a semantic_search call, or we can
provide it as a third tool.

Usage as a CLI:
    python chat.py --question "What was the 2nd track of Minimoni's 2nd album?"

Or interactive:
    python chat.py --interactive
"""
from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# Tools (deterministic, callable from the LLM)
# ---------------------------------------------------------------------------

def tool_lookup_track(
    conn,
    artist: str,
    album_position: int,
    track_position: int,
    kind: str = "album",
) -> dict:
    """Find the Nth track of the Mth release by an artist.

    Parameters
    ----------
    artist : str
        Name of the artist (e.g. "Minimoni", "Morning Musume").
        Aliases and redirects are followed automatically.
    album_position : int
        1-based release number (sorted by release date).
    track_position : int
        1-based track number within the release.
    kind : str
        "album" (default) or "single".
    """
    artist_page = find_artist_page(conn, artist)
    if not artist_page:
        return {"error": f"Artist '{artist}' not found"}

    releases = find_album_for_artist(conn, artist_page, kind=kind)
    if not releases:
        return {"error": f"No {kind}s found for '{artist}'"}

    if not (1 <= album_position <= len(releases)):
        return {
            "error": f"'{artist}' has only {len(releases)} {kind}s",
            "available_releases": [
                {"n": i + 1, "title": r.page.title, "release_date": r.release_date}
                for i, r in enumerate(releases)
            ],
        }

    target = releases[album_position - 1]

    # Find the Nth track
    tracks = get_tracklist(conn, target.page.id)
    real_tracks = [t for t in tracks if not t.is_karaoke]
    if not (1 <= track_position <= len(real_tracks)):
        return {
            "error": f"'{target.page.title}' has {len(real_tracks)} non-karaoke tracks",
            "tracks": [
                {"position": t.position, "raw": t.raw, "linked_title": t.linked_title}
                for t in real_tracks
            ],
        }

    track = real_tracks[track_position - 1]

    # Get further info if the track links to a song page
    song_info = None
    if track.linked_title:
        song_info = get_song_info(conn, track.linked_title)
        if song_info:
            song_info = _summarize_song_info(song_info)

    return {
        "artist": {
            "title": artist_page.title,
            "id": artist_page.id,
        },
        "album": {
            "title": target.page.title,
            "id": target.page.id,
            "album_number": target.album_number,
            "release_date": target.release_date,
        },
        "track": {
            "position": track.position,
            "raw": track.raw,
            "linked_title": track.linked_title,
        },
        "song_info": song_info,
    }


def tool_get_song_info(conn, song_title: str) -> dict:
    """Get detailed info about a song page."""
    info = get_song_info(conn, song_title)
    if not info:
        return {"error": f"Song '{song_title}' not found"}
    return _summarize_song_info(info)


def tool_list_releases(conn, artist: str, kind: str = "album") -> dict:
    """List all releases (albums or singles) by an artist."""
    artist_page = find_artist_page(conn, artist)
    if not artist_page:
        return {"error": f"Artist '{artist}' not found"}
    releases = find_album_for_artist(conn, artist_page, kind=kind)
    return {
        "artist": artist_page.title,
        "kind": kind,
        "count": len(releases),
        "releases": [
            {
                "n": i + 1,
                "title": r.page.title,
                "release_date": r.release_date,
                "track_count": r.track_count,
            }
            for i, r in enumerate(releases)
        ],
    }


def tool_semantic_search(
    chroma_dir: Path,
    query: str,
    k: int = 5,
    db_path: Path | None = None,
) -> dict:
    """Semantic search over the wiki. Returns top-k matching chunks.

    If db_path is provided, also enriches each chunk with the page's
    infobox fields. This is critical for factoid questions like
    "who produced X" — the producer info lives in the CD Infobox
    template, which we strip from the embedding chunks to keep them
    clean. Augmenting with infobox data gives the LLM everything it
    needs without re-embedding.
    """
    if not chroma_dir.exists():
        return {"error": "Vector index not built. Run build_embeddings.py first."}

    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection("helloproject")

    # Embed the query. We re-use the same model — for v1 we just reload.
    # In production we'd cache this model in a singleton.
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    q_emb = model.encode([query]).tolist()

    # Fetch extra results so we can deduplicate by page and surface more
    # diverse pages.
    res = collection.query(
        query_embeddings=q_emb,
        n_results=min(k * 3, 25),
    )

    # Deduplicate: prefer highest-scoring chunk per page_id, but keep
    # at most one chunk per page so we don't flood with one-page hits.
    seen_pages: set[int] = set()
    chunks: list[dict] = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        distance = res["distances"][0][i] if "distances" in res else None
        page_id = meta.get("page_id")
        if page_id in seen_pages:
            continue
        seen_pages.add(page_id)
        chunks.append({
            "page_id": page_id,
            "page_title": meta.get("page_title"),
            "section": meta.get("section"),
            "score": (1.0 - distance) if distance is not None else None,
            "text": doc[:1500],
        })
        if len(chunks) >= k:
            break

    # Enrich with infobox data from the SQLite DB. We pick a curated set
    # of fields that are most likely to answer factoid questions. We sort
    # the keys in a stable order so important fields (name, artist,
    # producer, label, released) always surface first.
    infobox_keys = (
        "name", "japanese", "artist", "producer", "released", "type",
        "album", "label", "genre", "format", "length", "origin", "years",
        "associated", "members", "caption",
    )
    if db_path is not None and db_path.exists():
        try:
            with connect(db_path) as conn:
                for chunk in chunks:
                    rows = conn.execute(
                        "SELECT key, value FROM infoboxes WHERE page_id=? AND key IN ({})".format(
                            ",".join("?" * len(infobox_keys))
                        ),
                        (chunk["page_id"], *infobox_keys),
                    ).fetchall()
                    if rows:
                        # Order by the canonical key order so important
                        # fields like Producer surface first.
                        key_priority = {k: i for i, k in enumerate(infobox_keys)}
                        rows = sorted(rows, key=lambda r: key_priority.get(r["key"].lower(), 99))
                        facts = []
                        for r in rows:
                            v = r["value"]
                            # Strip wikilink noise for readability
                            v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", v)
                            v = re.sub(r"<[^>]+>", "", v)
                            v = re.sub(r"<br\s*/?>", "; ", v)
                            v = v.strip()
                            facts.append(f"{r['key'].title()}: {v}")
                        chunk["infobox_facts"] = facts
        except Exception:
            pass  # enrichment is best-effort

    return {"query": query, "k": k, "chunks": chunks}


def _summarize_song_info(info: dict) -> dict:
    """Reduce a full song_info dict to a compact form for the LLM."""
    page = info["page"]
    ibx = info.get("infobox", {})

    def gv(key):
        v = ibx.get(key.lower())
        if not v:
            return None
        # Strip wikilink noise: [[Foo|bar]] -> bar, [[Foo]] -> Foo
        v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", v)
        # Strip <br>
        v = re.sub(r"<br\s*/?>", "; ", v)
        # Strip <small> tags
        v = re.sub(r"<[^>]+>", "", v)
        return v.strip()

    return {
        "title": page.title,
        "id": page.id,
        "artist": gv("artist"),
        "released": gv("released"),
        "type": gv("type"),
        "album": gv("album"),
        "genre": gv("genre"),
        "format": gv("format"),
        "length": gv("length"),
        "label": gv("label"),
        "producer": gv("producer"),
        "intro": (info.get("intro") or "")[:800],
    }


# ---------------------------------------------------------------------------
# Router: pick the right tool for the question
# ---------------------------------------------------------------------------

# Patterns for the structured (track-position) lookup.
# Each entry is a tuple:
#   (pattern, track_digit_idx, track_word_idx, artist_idx,
#    album_digit_idx, album_word_idx, kind_idx)
# Indices are absolute capture-group positions (0-based). They were
# determined empirically because ORDINAL expands to two capture groups
# (digit, word) and the artist group can sit between ordinals, shifting
# the indices in ways that are hard to compute without running the engine.
#
# Layouts (verified by inspection):
#   Pattern 1 "What was the Nth track of X's Mth album?"
#     Empirical groups: (td1, tw1, artist, td2, tw2, kind) = 6 groups
#   Pattern 2 "Nth track of X's Mth album"
#     Same as Pattern 1: 6 groups.
#   Pattern 3 "track N of X"
#     Empirical groups: (td, tw, artist) = 3 groups.
ORDINAL = r"(?:(\d+)(?:st|nd|rd|th)?|(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))"

_STRUCTURED_PATTERNS = [
    # "What was the Nth track of X's Mth album?"
    (
        rf"\bwhat(?:\s+is|\s+was)?\s+the\s+{ORDINAL}\s+(?:track|song)\b.*?\bof\s+(.+?)(?:'s|s|’s)?\s+{ORDINAL}\s+(album|mini[\s-]?album|ep|single)\b",
        0, 1, 2, 3, 4, 5,
    ),
    # "Nth track of X's Mth album"
    (
        rf"\b{ORDINAL}\s+(?:track|song)\s+(?:of|on|from)\b\s+(.+?)(?:'s|s)?\s+{ORDINAL}\s+(album|mini[\s-]?album|ep|single)\b",
        0, 1, 2, 3, 4, 5,
    ),
    # "track N of X" (only track ordinal + artist, no album ordinal)
    (
        rf"\b(?:track|song)\s+(?:#|number|num\.?|no\.?)?\s*{ORDINAL}\b.*?\bof\s+(.+?)\b",
        0, 1, 2, None, None, None,
    ),
]


def classify_question(question: str) -> dict:
    """Return a routing decision: which tool to call and with what args.

    For v1 this is regex-based. The next iteration would use the LLM
    itself to classify (with structured output / function calling).
    """
    q = question.strip()

    # Try the structured-track patterns first.
    for spec in _STRUCTURED_PATTERNS:
        pattern = spec[0]
        track_digit_idx = spec[1]
        track_word_idx = spec[2]
        artist_idx = spec[3]
        album_digit_idx = spec[4]
        album_word_idx = spec[5]
        kind_idx = spec[6]
        m = re.search(pattern, q, re.IGNORECASE)
        if not m:
            continue
        groups = m.groups()

        # Track ordinal
        track_n = _pick_ordinal(
            groups[track_digit_idx] if track_digit_idx is not None and track_digit_idx < len(groups) else None,
            groups[track_word_idx] if track_word_idx is not None and track_word_idx < len(groups) else None,
        )

        # Album ordinal (may be None)
        album_n = None
        if album_digit_idx is not None and album_digit_idx < len(groups):
            album_n = _pick_ordinal(
                groups[album_digit_idx],
                groups[album_word_idx] if album_word_idx < len(groups) else None,
            )

        # Artist
        if artist_idx is not None and artist_idx < len(groups) and groups[artist_idx]:
            artist = groups[artist_idx]
        else:
            artist = _extract_artist_after_of(q)

        # Kind
        kind = "album"
        if kind_idx is not None and kind_idx < len(groups) and groups[kind_idx]:
            kind = groups[kind_idx].lower()

        # Clean up artist.
        artist = artist.strip().rstrip("?").strip()
        if artist.lower().endswith(("'s", "s'", "’s")):
            artist = artist[:-2]

        if "single" in kind:
            kind = "single"
        else:
            kind = "album"

        # Default album_n to 1 if missing (pattern 3 case).
        if album_n is None or album_n == 0:
            album_n = 1

        return {
            "tool": "lookup_track",
            "args": {
                "artist": artist,
                "album_position": album_n,
                "track_position": track_n,
                "kind": kind,
            },
        }

    # Other structured lookups
    # "list all albums by X" / "what albums did X release"
    if re.search(r"\b(?:list|all)\b.*\b(?:albums?|singles?)\b.*\bby\b", q, re.IGNORECASE):
        m = re.search(r"\bby\s+([A-Z][\w\s'-]+?)(?:\?|$)", q)
        if m:
            kind = "single" if "single" in q.lower() else "album"
            return {"tool": "list_releases", "args": {"artist": m.group(1).strip(), "kind": kind}}

    # Default: semantic search
    return {"tool": "semantic_search", "args": {"query": q, "k": 5}}


def _pick_ordinal(digit_group: str | None, word_group: str | None) -> int:
    """Pick the integer from an ORDINAL match (digit preferred, else word)."""
    if digit_group:
        try:
            return int(digit_group)
        except (TypeError, ValueError):
            pass
    if word_group:
        return _word_to_int(word_group)
    return 0


def _extract_artist_after_of(q: str) -> str:
    """Pull the artist string out of '... of <artist> 's ...'."""
    m = re.search(r"\bof\s+(.+?)(?:'s|s|’s)\s+\w", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: anything after "of " up to a question mark
    m = re.search(r"\bof\s+(.+?)(?:\?|$)", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _word_to_int(s: str) -> int:
    """Parse '2', '2nd', 'second', 'two' → 2."""
    if s is None:
        return 0
    s = s.strip().lower()
    word_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8,
        "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
    }
    if s in word_map:
        return word_map[s]
    m = re.match(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Answer synthesis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable assistant for the Hello! Project Wiki, \
a Fandom wiki about the Japanese idol collective Hello! Project.

When answering questions:
- Be concise but informative.
- Use the data provided by tools exactly as given. Don't invent facts.
- If the tools return an error, explain what went wrong and suggest \
how the user might reformulate their question.
- For track-list questions, give the track title prominently and offer \
to provide more info about the song.
- For fuzzy/semantic questions, synthesize the answer from the \
relevant wiki chunks and cite the page title.
- Use the artist's preferred romanization (e.g. 'Morning Musume', \
'Minimoni', '℃-ute') when you mention them.
"""


def execute_tool_call(
    db_path: Path,
    chroma_dir: Path,
    call: dict,
) -> dict:
    """Dispatch a tool call and return its result."""
    tool = call.get("tool")
    args = call.get("args", {})

    if tool == "lookup_track":
        with connect(db_path) as conn:
            return tool_lookup_track(conn, **args)
    elif tool == "list_releases":
        with connect(db_path) as conn:
            return tool_list_releases(conn, **args)
    elif tool == "get_song_info":
        with connect(db_path) as conn:
            return tool_get_song_info(conn, **args)
    elif tool == "semantic_search":
        return tool_semantic_search(chroma_dir, db_path=db_path, **args)
    else:
        return {"error": f"Unknown tool: {tool}"}


def synthesize_answer(
    question: str,
    tool_result: dict,
    tool_name: str | None = None,
    llm: "LLMSynthesizer | None" = None,
    verbose: bool = False,
) -> str:
    """Format a tool result into a human-readable answer.

    v1: deterministic template formatters.
    v2: when an LLM is available, pass the tool result + question to it
        and let it write natural-language prose. Falls back to the
        template formatters if no LLM is configured or the call fails.

    This is the integration point — the LLM layer is opt-in. To enable,
    pass an LLMSynthesizer instance or set ANTHROPIC_API_KEY / OPENAI_API_KEY
    in the environment.
    """
    # Try the LLM first if configured.
    if llm is not None and llm.available:
        # Determine the tool name from the result shape if not provided.
        if tool_name is None:
            tool_name = _guess_tool_name(tool_result)
        llm_answer = llm.synthesize(question, tool_name, tool_result)
        if llm_answer:
            if verbose:
                print(f"[llm] {llm.describe()}", file=sys.stderr)
            return llm_answer
        # LLM failed; fall through to template.
        if verbose:
            print(f"[llm] synthesis failed; using template fallback", file=sys.stderr)

    # Template fallback (original behavior).
    if "error" in tool_result:
        return _format_error(tool_result)

    # Track lookup result
    if "album" in tool_result and "track" in tool_result:
        return _format_track_answer(tool_result)

    # List releases result
    if "releases" in tool_result:
        return _format_releases_list(tool_result)

    # Song info result
    if "artist" in tool_result and "released" in tool_result:
        return _format_song_info(tool_result)

    # Semantic search result
    if "chunks" in tool_result:
        return _format_semantic_search(tool_result)

    return f"(no formatter for result: {json.dumps(tool_result, indent=2)[:1000]})"


def _guess_tool_name(tool_result: dict) -> str:
    """Infer which tool produced a result by its shape."""
    if "album" in tool_result and "track" in tool_result:
        return "lookup_track"
    if "releases" in tool_result:
        return "list_releases"
    if "chunks" in tool_result:
        return "semantic_search"
    if "artist" in tool_result and "released" in tool_result:
        return "get_song_info"
    return "unknown"


def _format_error(result: dict) -> str:
    err = result.get("error", "Unknown error")
    msg = f"Sorry, I couldn't answer that: {err}"
    if "available_releases" in result:
        msg += "\n\nAvailable releases:\n"
        for r in result["available_releases"]:
            d = r.get("release_date") or "?"
            msg += f"  • {r['title']} ({d})\n"
    if "tracks" in result:
        msg += "\n\nTracks on that album:\n"
        for t in result["tracks"][:20]:
            msg += f"  {t['position']:2}. {t['raw'][:80]}\n"
        if len(result["tracks"]) > 20:
            msg += f"  ... and {len(result['tracks']) - 20} more\n"
    return msg


def _format_track_answer(r: dict) -> str:
    album = r["album"]
    track = r["track"]
    song = r.get("song_info")

    title = track.get("linked_title") or _strip_wikilink(track["raw"])
    lines = [
        f"Track #{track['position']} of {album['title']} "
        f"({album.get('release_date', 'unknown date')}) is: **{title}**",
    ]

    if song:
        if song.get("released"):
            lines.append(f"Released: {song['released']}")
        if song.get("artist"):
            lines.append(f"By: {song['artist']}")
        if song.get("length"):
            lines.append(f"Length: {song['length']}")
        if song.get("producer"):
            lines.append(f"Producer: {song['producer']}")
        if song.get("intro"):
            intro = song["intro"][:400].strip()
            lines.append(f"\nFrom the wiki:\n{intro}...")

    return "\n".join(lines)


def _format_releases_list(r: dict) -> str:
    artist = r["artist"]
    kind = r["kind"]
    releases = r["releases"]
    lines = [f"{artist} has {r['count']} {kind}s:"]
    for rel in releases:
        d = rel.get("release_date") or "?"
        t = rel.get("track_count")
        t_str = f", {t} tracks" if t else ""
        lines.append(f"  {rel['n']:2}. {rel['title']} ({d}{t_str})")
    return "\n".join(lines)


def _format_song_info(r: dict) -> str:
    lines = [f"**{r.get('title')}**"]
    for k in ("artist", "released", "type", "album", "genre", "format", "length", "label", "producer"):
        v = r.get(k)
        if v:
            lines.append(f"  {k.title()}: {v}")
    if r.get("intro"):
        lines.append(f"\n{r['intro'][:600]}")
    return "\n".join(lines)


def _format_semantic_search(r: dict) -> str:
    if not r.get("chunks"):
        return "No relevant content found in the wiki for that question."

    lines = [f"Top matches from the wiki for '{r['query']}':", ""]
    for i, chunk in enumerate(r["chunks"][:5], start=1):
        score = f"score={chunk['score']:.2f}" if chunk.get("score") is not None else ""
        lines.append(f"--- Match {i}: {chunk['page_title']} ({chunk['section']}) {score}")
        # Trim wikitext-ish artifacts for readability
        text = chunk["text"]
        text = re.sub(r"\{\{[^}]*\}\}", "", text)  # remove templates
        text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
        lines.append(text[:500] + ("..." if len(text) > 500 else ""))
        # Surface infobox facts — this is critical for factoid questions
        # ("who produced X") where the answer lives in the stripped template.
        if chunk.get("infobox_facts"):
            lines.append("  Key facts:")
            for fact in chunk["infobox_facts"]:
                lines.append(f"    • {fact}")
        lines.append("")
    return "\n".join(lines)


def _strip_wikilink(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    db_path: Path,
    chroma_dir: Path,
    verbose: bool = False,
    llm: "LLMSynthesizer | None" = None,
    context: "Context | None" = None,
    on_tool_complete: "callable | None" = None,
    prior_year: int | None = None,
) -> str:
    """Answer a single question end-to-end.

    If `context` is provided (a `conversation.Context` instance), the
    question will be rewritten using remembered entities before the
    classifier runs.

    If `prior_year` is provided, temporal references like 'the next
    year' will be rewritten to specific years (e.g., 'in 2011') using
    prior_year as the anchor.

    If `on_tool_complete` is provided, it will be called as
    `on_tool_complete(tool_call, tool_result, answer)` after the tool
    executes but before synthesis. Used by the REPL to record turns
    into a Conversation without re-running classify/execute.
    """
    notes: list[str] = []
    if context is not None:
        try:
            from conversation import prepare_question
            question, notes = prepare_question(
                question, context, verbose=verbose, prior_year=prior_year,
            )
        except Exception as e:
            if verbose:
                print(f"[context] prepare failed: {e}", file=sys.stderr)

    if verbose:
        print(f"[classify] question={question!r}", file=sys.stderr)

    call = classify_question(question)
    if verbose:
        print(f"[classify] -> {call}", file=sys.stderr)

    result = execute_tool_call(db_path, chroma_dir, call)
    if verbose:
        print(f"[tool_result] {json.dumps(result, indent=2, ensure_ascii=False)[:1500]}", file=sys.stderr)

    if on_tool_complete is not None:
        try:
            on_tool_complete(call, result)
        except Exception as e:
            if verbose:
                print(f"[callback] failed: {e}", file=sys.stderr)

    return synthesize_answer(question, result, tool_name=call.get("tool"), llm=llm, verbose=verbose)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--question", "-q", help="Single question to answer")
    p.add_argument("--interactive", "-i", action="store_true", help="REPL mode")
    p.add_argument("--db", default=str(HERE / "helloproject.db"))
    p.add_argument("--chroma", default=str(HERE / "chroma"))
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable the LLM synthesis layer (use template formatters)")
    args = p.parse_args()

    db_path = Path(args.db)
    chroma_dir = Path(args.chroma)
    if not db_path.exists():
        print(f"No database at {db_path}. Run build_index.py first.", file=sys.stderr)
        return 1

    # Set up the LLM synthesizer. We import lazily so the chat CLI still
    # works even if llm.py has issues.
    llm = None
    if not args.no_llm:
        try:
            from llm import LLMSynthesizer
            llm = LLMSynthesizer()
            if args.verbose:
                print(f"[{llm.describe()}]", file=sys.stderr)
        except Exception as e:
            print(f"[llm] init failed: {e}; using template formatters", file=sys.stderr)
            llm = None

    if args.interactive:
        print("Hello! Project Wiki chatbot. Ask questions (Ctrl-D to exit).")
        if llm and llm.available:
            print(f"[LLM: {llm.provider}]")
        print("Slash commands: /new, /history, /last, /help")
        print()
        try:
            from conversation import Conversation, parse_input, prepare_followup
            conv = Conversation()
        except Exception as e:
            print(f"[conv] init failed: {e}; running single-turn mode", file=sys.stderr)
            conv = None

        # State carried between iterations: the most recent tool call,
        # so a continuation like "yes" can re-issue it.
        last_tool_name: str | None = None
        last_tool_args: dict | None = None
        last_tool_result: dict | None = None

        # Tool functions we may invoke directly (avoid name-lookup inside
        # the hot loop and dodge "from chat import ..." import-time guards).
        from chat import tool_get_song_info, execute_tool_call, synthesize_answer

        try:
            while True:
                try:
                    raw = input("> ").strip()
                except EOFError:
                    raise
                if not raw:
                    continue

                parsed = parse_input(raw)
                if parsed.is_command:
                    cmd = parsed.command
                    if cmd in ("exit", "quit"):
                        print("(exiting)")
                        return 0
                    elif cmd == "new":
                        if conv:
                            conv.reset()
                        last_tool_name = None
                        last_tool_args = None
                        last_tool_result = None
                        print("(conversation reset)")
                    elif cmd == "history":
                        if not conv:
                            print("(no conversation state)")
                        else:
                            print(conv.format_history())
                    elif cmd == "last":
                        # alias for /history for backward compat
                        if not conv:
                            print("(no conversation state)")
                        else:
                            print(conv.format_history())
                    elif cmd == "ctx":
                        if not conv:
                            print("(no conversation state)")
                        else:
                            print(conv.format_context())
                    elif cmd == "help":
                        print("Slash commands:")
                        print("  /new      Start a new conversation (clear history and context)")
                        print("  /history  Show the last few turns of conversation")
                        print("  /ctx      Show currently remembered entities")
                        print("  /help     Show this message")
                        print("  /exit     Quit the REPL (Ctrl-D also works)")
                        print()
                        print("Anything else is treated as a question.")
                        print("Special follow-ups: 'yes' (more on the last topic),")
                        print("  'tracklist' / 'members' / 'history' (ask about the last entity).")
                    else:
                        print(f"Unknown command: /{cmd}. Try /help.")
                    continue

                # It's a question (or follow-up).
                question = parsed.raw
                if conv:
                    conv.add_user_turn(question)

                ctx = conv.context if conv else None

                # Decide whether this is a fresh question, a bare-noun
                # expansion, or a continuation of the prior tool call.
                decision = prepare_followup(
                    question, ctx,
                    last_tool_name=last_tool_name,
                    last_tool_args=last_tool_args,
                    last_tool_result=last_tool_result,
                    verbose=args.verbose,
                )

                if args.verbose and decision.note:
                    print(f"[followup] {decision.note}", file=sys.stderr)

                if decision.kind == "continuation":
                    # Re-issue the prior tool call. If the prior tool was
                    # lookup_track and we now know a song title, switch
                    # to get_song_info for a richer answer.
                    if last_tool_name == "lookup_track" and ctx and ctx.song_title:
                        # Get full song info instead of just the track listing.
                        from query import connect
                        with connect(db_path) as _c:
                            result = tool_get_song_info(_c, song_title=ctx.song_title)
                        call = {"tool": "get_song_info", "args": {"song_title": ctx.song_title}}
                        if args.verbose:
                            print(f"[continuation] -> get_song_info({ctx.song_title})",
                                  file=sys.stderr)
                    else:
                        # Re-issue the same tool call verbatim.
                        call = {"tool": last_tool_name, "args": last_tool_args or {}}
                        result = execute_tool_call(db_path, chroma_dir, call)
                        if args.verbose:
                            print(f"[continuation] -> re-call {last_tool_name}",
                                  file=sys.stderr)

                    answer = synthesize_answer(
                        question, result,
                        tool_name=call.get("tool"),
                        llm=llm, verbose=args.verbose,
                    )

                    if conv:
                        conv.add_assistant_turn(
                            answer, tool_name=call.get("tool"), tool_result=result,
                        )
                    last_tool_name = call.get("tool")
                    last_tool_args = call.get("args")
                    last_tool_result = result
                    print(answer)
                    continue

                # new_question or expansion: run the full pipeline.
                question_to_classify = (
                    decision.rewritten_question if decision.kind == "expansion"
                    else question
                )

                # Pull the year (if any) from the prior user turn so
                # 'the next year' / 'the previous year' can be resolved
                # to specific years ('in 2011') rather than abstract
                # phrases ('the year after').
                #
                # Walk back through turns to find the most recent user
                # turn before the current one. We can't just look at
                # `conv.turns[-2]` because each prior turn is an
                # interleaved user+assistant pair.
                prior_year = None
                if conv is not None and len(conv.turns) >= 2:
                    try:
                        from conversation import extract_year
                        for prior_turn in reversed(conv.turns[:-1]):
                            if prior_turn.role == "user":
                                prior_year = extract_year(prior_turn.content)
                                break
                    except Exception:
                        prior_year = None

                def record_turn(call, result):
                    if not conv:
                        return
                    conv.add_assistant_turn(
                        "(pending)",
                        tool_name=call.get("tool"),
                        tool_result=result,
                    )

                answer = answer_question(
                    question_to_classify, db_path, chroma_dir,
                    verbose=args.verbose, llm=llm, context=ctx,
                    on_tool_complete=record_turn,
                    prior_year=prior_year,
                )
                if conv and conv.turns and conv.turns[-1].role == "assistant":
                    conv.turns[-1].content = answer

                # Stash the most recent tool call for the next iteration.
                # We need both the args (to re-issue) and the result (to
                # detect what kind of tool was called). tool_result on
                # the Turn holds the dispatch dict in our convention; the
                # tool_name we can read directly.
                if conv and conv.turns and conv.turns[-1].tool_name:
                    last_tool_name = conv.turns[-1].tool_name
                    last_tool_result = conv.turns[-1].tool_result
                    # For re-issuing lookup_track we need (artist, album,
                    # track, kind). For others we don't re-issue; only
                    # lookup_track triggers the get_song_info upgrade.
                    if last_tool_name == "lookup_track" and conv.turns[-1].tool_result:
                        # The result has album/artist — we can rebuild args.
                        r = conv.turns[-1].tool_result
                        last_tool_args = {
                            "artist": r.get("artist", {}).get("title"),
                            "album_position": r.get("album", {}).get("album_number"),
                            "track_position": r.get("track", {}).get("position"),
                        }
                    else:
                        last_tool_args = None

                print(answer)
        except (EOFError, KeyboardInterrupt):
            print()
        return 0

    if not args.question:
        p.error("--question is required (or use --interactive)")

    print(answer_question(args.question, db_path, chroma_dir, verbose=args.verbose, llm=llm))
    return 0


if __name__ == "__main__":
    sys.exit(main())