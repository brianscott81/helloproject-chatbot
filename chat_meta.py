"""
Extract entities and sources from a chat tool result for the web UI.

Entities are names that appear in the response (artists, albums, songs,
producers, labels, members) that have corresponding wiki pages. The web
client renders them as clickable links that send a "Tell me about X"
prompt when clicked.

Sources are wiki pages that contributed to the answer. The web client
renders them as a "Sources" section at the bottom of the response with
links to the Fandom wiki.

This module is read-only on the SQLite database — it just looks up
page IDs for entity names that already appear in the tool result. It
does not modify state.

This module is deliberately separate from chat.py so the CLI's REPL
behavior is unchanged. The web app is the only caller.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from query import connect, resolve_title


# Maximum number of entities to extract. The web client can render
# any number, but more than this becomes noisy.
_MAX_ENTITIES = 15

# Maximum number of sources (wiki pages) to cite. Semantic search
# can return up to 5 chunks but each chunk is a different page, so
# 5-10 is typical.
_MAX_SOURCES = 10


def _wiki_url(page_title: str) -> str:
    """Build a Fandom wiki URL from a page title."""
    # Fandom URLs use underscores instead of spaces. Special chars
    # are URL-encoded. For Hello! Project Wiki, the URL format is:
    #   https://helloproject.fandom.com/wiki/<Page_Title>
    encoded = page_title.replace(" ", "_")
    # Minimal URL-encoding for common wiki characters
    encoded = encoded.replace("&", "%26")
    encoded = encoded.replace("?", "%3F")
    encoded = encoded.replace("#", "%23")
    return f"https://helloproject.fandom.com/wiki/{encoded}"


def _resolve_entity(conn, name: str, entity_type: str) -> dict | None:
    """Resolve an entity name to {name, type, page_id, url} or None if not found.

    Strips parenthetical text like "(redirected from ...)" before lookup.
    Trims at semicolons (some infoboxes have multiple values).
    """
    if not name or not isinstance(name, str):
        return None
    # Take the first segment if semicolon-separated
    name = name.split(";")[0].strip()
    # Take the first segment if parenthetical
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if not name:
        return None
    try:
        page = resolve_title(conn, name)
    except Exception:
        return None
    if not page:
        return None
    return {
        "name": page.title,
        "type": entity_type,
        "page_id": page.id,
        "url": _wiki_url(page.title),
    }


# Patterns for member-list extraction from chunk text. The wiki uses
# these formats for listing members in section bodies:
#   *Ishiguro Aya        (single-word prefix, capitalized name)
#   * [[Name]]           (wikilink form, but wikilinks are stripped before
#                          embedding, so this appears as "*Name")
#   Ishiguro Aya         (just the name on a line, no bullet)
# Members are typically "Firstname Lastname" — two Title-Case words.
# Chunks can have multiple bullets on a single line, so the asterisk is
# matched as a separator rather than a line-anchor.
_MEMBER_BULLET_PATTERN = re.compile(
    r"\*\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)"
)


# Patterns for scanning the LLM's response text for entity names. We
# look for 1-3 Title-Case words, optionally with internal '!' or '?'
# (e.g., "Hello! Project"). Periods are NOT included because they
# would let the regex bridge across sentence boundaries (e.g.,
# "Morning Musume. The"). The DB lookup filters out non-entities
# like "April", "Tokyo", "Japan", so false positives don't surface.
_PROSE_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:[!?][!?]?)?"
    r"(?: [A-Z][a-zA-Z]+(?:[!?][!?]?)?){0,2}"
    r")\b"
)


def _extract_candidate_names_from_text(text: str, section: str = "") -> list[str]:
    """Pull candidate entity names from chunk text.

    Looks for patterns like '*Name' bullets and 2+ Title-Case words.
    The candidates are then resolved to page IDs via the DB to confirm
    they're real entities.

    Only fires for member-related sections to avoid noise.
    """
    if not text:
        return []
    candidates: list[str] = []

    # Pattern 1: bullet-style member lists (*Name)
    for m in _MEMBER_BULLET_PATTERN.finditer(text):
        name = m.group(1).strip()
        if name and name not in candidates:
            candidates.append(name)

    return candidates


# Common English words that often appear at the start of candidate
# matches but aren't really entities. Skipping these avoids linking
# things like "April 2004" (date), "Tokyo, Japan" (place), "Members:"
# (header), or "All four were..." (pronoun).
# Some entries are common words that the wiki happens to have pages
# for (e.g., "Japan", "HiP", "W") that the resolver would link to.
_PROSE_STOPLIST = frozenset({
    "All", "April", "August", "December", "February", "Friday",
    "HiP", "Hip", "How", "I", "It", "January", "July", "June",
    "March", "May", "Members", "Monday", "November", "October",
    "September", "Saturday", "Some", "Sunday", "Tell", "The", "They",
    "This", "Thursday", "Tokyo", "Tuesday", "W", "Want", "Wednesday",
    "We", "What", "When", "Where", "Why", "Yes", "You",
    "Japan", "Japanese",
})


def _extract_candidate_names_from_prose(text: str) -> list[str]:
    """Pull candidate entity names from free-form prose (LLM response).

    Scans for 1-3 Title-Case words. Single-word candidates must be at
    least 3 characters long to avoid noise like "W", "the", "of" —
    but this still catches single-word entities like "ZYX" and
    "Tanpopo". Each candidate is resolved to a wiki page via the DB,
    and DB misses are filtered out.

    Common English words (months, days, pronouns, etc.) are skipped
    via a stoplist to avoid linking dates, places, and sentence-starts.

    Returns a list of unique candidate strings, longest first (so we
    prefer "Morning Musume" over "Morning" when both match).
    """
    if not text:
        return []
    matches = _PROSE_ENTITY_PATTERN.findall(text)
    # Dedupe case-insensitively, keeping longest
    seen_lower: dict[str, str] = {}
    for m in matches:
        m = m.strip().rstrip(".,;:!?")
        # Filter out very short single-word candidates (likely noise).
        if " " not in m and len(m) < 3:
            continue
        # Skip if the candidate's first word is a common English word.
        first_word = m.split()[0]
        if first_word in _PROSE_STOPLIST:
            continue
        key = m.lower()
        if key not in seen_lower or len(m) > len(seen_lower[key]):
            seen_lower[key] = m
    # Sort by length descending so longer matches are tried first
    return sorted(seen_lower.values(), key=len, reverse=True)


def _add_entity(entities: list[dict], seen: set,
                name: str, entity_type: str, page_id: int | None = None) -> None:
    """Add an entity to the list if not already present (by page_id or name).

    Duplicate detection: same page_id (preferred) or same (name, type) tuple.
    The `seen` set may contain either ints (page_ids) or (name_lower, type)
    tuples — mixed types are fine in a Python set.
    """
    if len(entities) >= _MAX_ENTITIES:
        return
    if not name or not isinstance(name, str):
        return
    name = name.strip()
    if not name:
        return
    if page_id is not None:
        if page_id in seen:
            return
    else:
        name_key = (name.lower(), entity_type)
        if name_key in seen:
            return
    if page_id is not None:
        seen.add(page_id)
    else:
        seen.add((name.lower(), entity_type))
    entry = {
        "name": name,
        "type": entity_type,
        "page_id": page_id,
    }
    if page_id is not None:
        entry["url"] = _wiki_url(name)
    entities.append(entry)


def extract_entities_from_prose(
    text: str,
    db_path: Path | None = None,
    existing_entities: list[dict] | None = None,
) -> list[dict]:
    """Extract additional entities from the LLM's response text.

    Scans for Title Case word patterns (1-3 capitalized words). Each
    candidate is resolved to a wiki page via the DB. Names that don't
    resolve (e.g., "April", "Tokyo", "Japan") are filtered out.

    This is a second pass that runs AFTER synthesis, complementing
    extract_meta which runs on the structured tool result. Together
    they catch both:

      - Names from the structured tool data (albums, songs, producers,
        members) — caught by extract_meta
      - Names mentioned only in the LLM's prose (other groups, places,
        etc.) — caught by this function

    Returns a list of {name, type, page_id, url}. The type is generic
    "page" for these; we don't try to classify them further (the
    linkify doesn't care about type, just name + page_id).
    """
    if not text or db_path is None or not db_path.exists():
        return []

    candidates = _extract_candidate_names_from_prose(text)

    # Build a set of page_ids and names already extracted so we skip them.
    existing_page_ids: set[int] = set()
    existing_names_lower: set[str] = set()
    for e in (existing_entities or []):
        if e.get("page_id") is not None:
            existing_page_ids.add(e["page_id"])
        if e.get("name"):
            existing_names_lower.add(e["name"].lower())

    entities: list[dict] = []
    seen_page_ids: set[int] = set()
    seen_names: set[str] = set()

    try:
        with connect(db_path) as conn:
            for cand in candidates:
                # Skip if already covered
                if cand.lower() in existing_names_lower:
                    continue
                page = resolve_title(conn, cand)
                if not page:
                    continue
                # Skip if already covered by page_id
                if page.id in existing_page_ids or page.id in seen_page_ids:
                    continue
                if page.title.lower() in seen_names:
                    continue
                seen_page_ids.add(page.id)
                seen_names.add(page.title.lower())
                entities.append({
                    "name": page.title,
                    "type": "page",
                    "page_id": page.id,
                    "url": _wiki_url(page.title),
                })
                if len(entities) >= _MAX_ENTITIES:
                    break
    except Exception:
        pass  # best-effort

    return entities


def extract_meta(tool_result: dict, db_path: Path | None = None) -> dict:
    """Extract entities and sources from a tool result.

    Returns {"entities": [...], "sources": [...]}.

    Each entity: {name, type, page_id, url?} — type is "artist", "album",
    "song", "producer", "label", "member", or "other".

    Each source: {title, page_id, url} — wiki pages that contributed to
    the answer.

    This function does best-effort extraction. If `db_path` is None or
    unavailable, entity extraction is degraded (no page_id resolution
    for string-valued fields like song_info.producer).
    """
    entities: list[dict] = []
    sources: list[dict] = []
    seen_entities: set[tuple[str, int] | int] = set()
    seen_sources: set[int] = set()

    if not isinstance(tool_result, dict):
        return {"entities": entities, "sources": sources}

    # Connect to the DB once for entity resolution.
    conn = None
    if db_path is not None and db_path.exists():
        try:
            conn = connect(db_path)
        except Exception:
            conn = None

    try:
        # ----- tool_lookup_track result -----
        if "album" in tool_result and isinstance(tool_result["album"], dict):
            album = tool_result["album"]
            if album.get("id") is not None:
                _add_entity(entities, seen_entities,
                            album.get("title", ""), "album", album["id"])
                if album["id"] not in seen_sources:
                    seen_sources.add(album["id"])
                    sources.append({
                        "title": album.get("title", ""),
                        "page_id": album["id"],
                        "url": _wiki_url(album.get("title", "")),
                    })

        if "artist" in tool_result and isinstance(tool_result["artist"], dict):
            artist = tool_result["artist"]
            if artist.get("id") is not None:
                _add_entity(entities, seen_entities,
                            artist.get("title", ""), "artist", artist["id"])

        if "song_info" in tool_result and isinstance(tool_result["song_info"], dict):
            si = tool_result["song_info"]
            if si.get("id") is not None:
                if si["id"] not in seen_sources:
                    seen_sources.add(si["id"])
                    sources.append({
                        "title": si.get("title", ""),
                        "page_id": si["id"],
                        "url": _wiki_url(si.get("title", "")),
                    })
            # The song itself is an entity
            _add_entity(entities, seen_entities,
                        si.get("title", ""), "song", si.get("id"))
            # String-valued fields — try to resolve to entities
            if conn is not None:
                for key, etype in [("artist", "artist"),
                                   ("album", "album"),
                                   ("producer", "producer"),
                                   ("label", "label")]:
                    resolved = _resolve_entity(conn, si.get(key, ""), etype)
                    if resolved:
                        _add_entity(entities, seen_entities,
                                    resolved["name"], resolved["type"],
                                    resolved["page_id"])

        # ----- tool_list_releases result -----
        if "releases" in tool_result and isinstance(tool_result["releases"], list):
            artist_name = tool_result.get("artist", "")
            # tool_list_releases returns artist as a string, no ID
            if artist_name and conn is not None:
                resolved = _resolve_entity(conn, artist_name, "artist")
                if resolved:
                    _add_entity(entities, seen_entities,
                                resolved["name"], resolved["type"],
                                resolved["page_id"])
            else:
                _add_entity(entities, seen_entities, artist_name, "artist", None)
            for r in tool_result["releases"]:
                if not isinstance(r, dict):
                    continue
                title = r.get("title", "")
                if conn is not None and title:
                    resolved = _resolve_entity(conn, title, "release")
                    if resolved:
                        _add_entity(entities, seen_entities,
                                    resolved["name"], resolved["type"],
                                    resolved["page_id"])
                        # Also add to sources
                        if resolved["page_id"] not in seen_sources:
                            seen_sources.add(resolved["page_id"])
                            sources.append({
                                "title": resolved["name"],
                                "page_id": resolved["page_id"],
                                "url": _wiki_url(resolved["name"]),
                            })
                else:
                    _add_entity(entities, seen_entities, title, "release", None)

        # ----- tool_get_song_info result -----
        # Shape: {title, id, artist, released, type, album, genre, format,
        #         length, label, producer, intro}
        # The page itself is the song. id and title are at top level.
        if ("title" in tool_result and "released" in tool_result
                and "id" in tool_result
                and "album" in tool_result
                and "releases" not in tool_result):
            sid = tool_result.get("id")
            stitle = tool_result.get("title", "")
            if sid is not None and sid not in seen_sources:
                seen_sources.add(sid)
                sources.append({
                    "title": stitle,
                    "page_id": sid,
                    "url": _wiki_url(stitle),
                })
            _add_entity(entities, seen_entities, stitle, "song", sid)
            # String-valued fields — try to resolve to entities
            if conn is not None:
                for key, etype in [("artist", "artist"),
                                   ("album", "album"),
                                   ("producer", "producer"),
                                   ("label", "label")]:
                    resolved = _resolve_entity(conn, tool_result.get(key, ""), etype)
                    if resolved:
                        _add_entity(entities, seen_entities,
                                    resolved["name"], resolved["type"],
                                    resolved["page_id"])

        # ----- tool_get_tracklist result -----
        if "tracks" in tool_result and "track_count" in tool_result:
            album = tool_result.get("album", {})
            if isinstance(album, dict) and album.get("id") is not None:
                if album["id"] not in seen_sources:
                    seen_sources.add(album["id"])
                    sources.append({
                        "title": album.get("title", ""),
                        "page_id": album["id"],
                        "url": _wiki_url(album.get("title", "")),
                    })
                _add_entity(entities, seen_entities,
                            album.get("title", ""), "album", album["id"])

        # ----- tool_semantic_search result -----
        if "chunks" in tool_result and isinstance(tool_result["chunks"], list):
            for chunk in tool_result["chunks"]:
                if not isinstance(chunk, dict):
                    continue
                pid = chunk.get("page_id")
                ptitle = chunk.get("page_title", "")
                if pid is not None and pid not in seen_sources:
                    seen_sources.add(pid)
                    sources.append({
                        "title": ptitle,
                        "page_id": pid,
                        "url": _wiki_url(ptitle),
                    })
                # The page itself is an entity
                _add_entity(entities, seen_entities, ptitle, "page", pid)
                # Extract entity names from infobox_facts strings like
                # "Producer: Tsunku", "Artist: Minimoni".
                for fact in chunk.get("infobox_facts", []):
                    if not isinstance(fact, str) or ":" not in fact:
                        continue
                    key, _, value = fact.partition(":")
                    key = key.strip().lower()
                    value = value.strip()
                    if not value:
                        continue
                    # Map infobox key to entity type
                    etype = {
                        "name": "page",
                        "artist": "artist",
                        "album": "album",
                        "producer": "producer",
                        "label": "label",
                        "members": "member",
                        "associated": "associated",
                    }.get(key)
                    if etype is None:
                        continue
                    if conn is not None:
                        resolved = _resolve_entity(conn, value, etype)
                        if resolved:
                            _add_entity(entities, seen_entities,
                                        resolved["name"], resolved["type"],
                                        resolved["page_id"])
                    else:
                        _add_entity(entities, seen_entities, value, etype, None)

                # For member-related sections, also pull candidate names
                # from the chunk text. The wiki lists members like:
                #   *Ishiguro Aya *Iida Kaori *Yaguchi Mari
                # which appear in chunk text without wikilink markers
                # (the embedding chunks have wikilinks stripped). Each
                # candidate is resolved to a page_id via the DB so we
                # only surface names that actually exist in the wiki.
                section = (chunk.get("section") or "").lower()
                chunk_text = chunk.get("text", "")
                if section and any(
                    kw in section for kw in (
                        "member", "lineup", "line-up", "generation",
                    )
                ):
                    candidates = _extract_candidate_names_from_text(
                        chunk_text, section,
                    )
                    if conn is not None:
                        for cand in candidates:
                            # Members can also be artists, albums, etc.
                            # Try "member" first, fall back to "page".
                            resolved = _resolve_entity(conn, cand, "member")
                            if not resolved:
                                resolved = _resolve_entity(conn, cand, "page")
                            if resolved:
                                _add_entity(entities, seen_entities,
                                            resolved["name"],
                                            resolved["type"],
                                            resolved["page_id"])

        # Cap the lists
        entities = entities[:_MAX_ENTITIES]
        sources = sources[:_MAX_SOURCES]
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return {"entities": entities, "sources": sources}