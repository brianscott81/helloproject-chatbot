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
                        # Without DB, just add the name without page_id.
                        _add_entity(entities, seen_entities, value, etype, None)

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