"""
Query layer over the helloproject.db SQLite index.

Provides high-level functions for the kinds of questions a user might
ask about the wiki. Used both by the CLI and by the chat layer.

This is the "structured path" — deterministic lookup, not fuzzy
semantic search. Embeddings + LLM synthesis are layered on top later.

Key abstractions:

    resolve_title(title_or_alias)  -> page row (with redirect-following)
    find_artist_page(artist_name)  -> page id of an artist's main page
    find_album_for_artist(artist, position, kind='album')
    find_track_at_position(album_page_id, position) -> track entry
    get_song_info(song_title) -> wikitext excerpt + infobox fields

The resolver chain:
    user says "Minimoni"        -> Minimoni (main page)
    user says "Minimoni."       -> Minimoni. (redirect) -> Minimoni
    user says "C-ute"           -> C-ute (redirect) -> ℃-ute (canonical)
    user says "CRAZY ABOUT YOU" -> CRAZY ABOUT YOU (a song)
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Page:
    id: int
    title: str
    namespace: int
    is_redirect: bool
    redirect_to: str | None = None
    wikitext: str | None = None


@dataclass
class Track:
    position: int
    section: str
    raw: str
    linked_title: str | None
    is_karaoke: bool
    page_id: int
    page_title: str


@dataclass
class AlbumMatch:
    page: Page
    album_number: int | None = None
    release_date: str | None = None
    last_album: str | None = None
    next_album: str | None = None
    track_count: int | None = None


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable FK enforcement + some perf pragmas.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Title resolution
# ---------------------------------------------------------------------------

def _row_to_page(row: sqlite3.Row) -> Page:
    return Page(
        id=row["id"],
        title=row["title"],
        namespace=row["namespace"],
        is_redirect=bool(row["is_redirect"]),
        redirect_to=row["redirect_to"],
    )


def resolve_title(conn: sqlite3.Connection, title: str) -> Page | None:
    """Resolve a title (or alias) to its canonical page.

    Follows redirect chains via the aliases table. Returns None if no page
    matches at all.
    """
    title = title.replace("_", " ").strip()
    if not title:
        return None

    # 1. Exact title match.
    row = conn.execute(
        "SELECT * FROM pages WHERE title=? AND namespace=0 LIMIT 1", (title,)
    ).fetchone()
    if row and not row["is_redirect"]:
        return _row_to_page(row)

    # 2. Alias lookup (redirect-resolved).
    alias = conn.execute(
        "SELECT * FROM aliases WHERE alias_title=?", (title,)
    ).fetchone()
    if alias:
        row = conn.execute(
            "SELECT * FROM pages WHERE id=? AND namespace=0 LIMIT 1",
            (alias["canonical_id"],),
        ).fetchone()
        if row and not row["is_redirect"]:
            return _row_to_page(row)

    # 3. Case-insensitive title match.
    row = conn.execute(
        "SELECT * FROM pages WHERE LOWER(title)=LOWER(?) AND namespace=0 LIMIT 1",
        (title,),
    ).fetchone()
    if row and not row["is_redirect"]:
        return _row_to_page(row)

    # 4. Loose match: title starts with the search term.
    row = conn.execute(
        "SELECT * FROM pages WHERE title LIKE ? AND namespace=0 AND is_redirect=0 LIMIT 1",
        (title + "%",),
    ).fetchone()
    if row:
        return _row_to_page(row)

    return None


def resolve_title_with_aliases(conn: sqlite3.Connection, title: str) -> tuple[Page | None, list[str]]:
    """Like resolve_title, but also returns the chain of aliases that were
    followed."""
    title = title.replace("_", " ").strip()
    chain: list[str] = []
    current = title

    # Walk aliases table iteratively
    for _ in range(5):
        alias = conn.execute(
            "SELECT canonical_id, canonical_title FROM aliases WHERE alias_title=?",
            (current,),
        ).fetchone()
        if not alias:
            break
        chain.append(current)
        current = alias["canonical_title"]

    page = resolve_title(conn, current)
    return page, chain


# ---------------------------------------------------------------------------
# Artist lookup
# ---------------------------------------------------------------------------

def find_artist_page(conn: sqlite3.Connection, artist_name: str) -> Page | None:
    """Look up an artist (group, person, etc.).

    Strategy:
      1. Direct title resolution (handles redirects like 'C-ute' → '℃-ute').
      2. The Unit Infobox is checked: if there's an artist with a
         `name = X` field matching artist_name.
    """
    page = resolve_title(conn, artist_name)
    if page:
        return page

    # Search infoboxes for Unit/Person infoboxes with a matching name.
    row = conn.execute(
        """
        SELECT p.* FROM pages p
        JOIN infoboxes i ON i.page_id = p.id
        WHERE i.key IN ('name', 'jpname') AND i.value LIKE ?
          AND p.namespace = 0 AND p.is_redirect = 0
        LIMIT 1
        """,
        (f"%{artist_name}%",),
    ).fetchone()
    if row:
        return _row_to_page(row)
    return None


# ---------------------------------------------------------------------------
# Album / release lookup
# ---------------------------------------------------------------------------

_KIND_MAP = {
    "album": "album",
    "albums": "album",
    "mini album": "album",
    "mini-album": "album",
    "ep": "album",
    "single": "single",
    "singles": "single",
}


def find_album_for_artist(
    conn: sqlite3.Connection,
    artist_page: Page,
    position: int | None = None,
    kind: str = "album",
    year: int | None = None,
) -> list[AlbumMatch]:
    """Find albums/singles by a given artist.

    Ordering: by release_date ascending (chronological). When two albums
    share a release date (e.g. a same-day reissue), Last/Next infobox
    pointers break the tie. As a final fallback, alphabetical title.

    Each match gets an album_number assigned by its 1-indexed position
    in this sorted list. The Last/Next infobox fields are also exposed
    in case the caller wants to verify.

    If year is given, only releases whose release_date starts with
    that year (e.g. "2001-02-21") are kept. Releases whose release_date
    starts with just "2001" (year only) are also kept. Releases with
    no parseable date are dropped when year is given.

    Returns a list (may be empty).
    """
    kind = _KIND_MAP.get(kind.lower(), "album")

    # Find all CD Infobox pages where the artist links to this artist.
    rows = conn.execute(
        """
        SELECT p.* FROM pages p
        JOIN infoboxes i ON i.page_id = p.id
        WHERE i.key = 'type' AND LOWER(i.value) = ?
          AND p.namespace = 0 AND p.is_redirect = 0
        """,
        (kind,),
    ).fetchall()

    matches: list[AlbumMatch] = []
    for r in rows:
        page = _row_to_page(r)
        # Check if this page's infobox has `artist = [[<artist_title>]]`
        artist_val = conn.execute(
            "SELECT value FROM infoboxes WHERE page_id=? AND key='artist' LIMIT 1",
            (page.id,),
        ).fetchone()
        if not artist_val:
            continue
        artist_text = artist_val["value"]
        # The artist value is wikitext like '[[Minimoni]]'. Extract target.
        m = re.search(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)\]\]", artist_text)
        if not m:
            continue
        target_title = m.group(1).replace("_", " ").strip()
        # Resolve to canonical (handles redirects like Minimoni. → Minimoni)
        target_page, _ = resolve_title_with_aliases(conn, target_title)
        if not target_page or target_page.id != artist_page.id:
            continue

        # Build AlbumMatch
        disco = _extract_discography_fields(conn, page.id)
        release_date = disco.get("release_date")

        # Year filter: if specified, only keep releases whose date
        # starts with the year. Release dates look like "2001-02-21"
        # or just "2001" (year only).
        if year is not None:
            if not release_date or not release_date.startswith(str(year)):
                continue

        track_count = conn.execute(
            "SELECT COUNT(*) FROM tracklists WHERE page_id=? AND is_karaoke=0",
            (page.id,),
        ).fetchone()[0]
        matches.append(AlbumMatch(
            page=page,
            album_number=None,  # assigned below
            release_date=release_date,
            last_album=disco.get("last_target"),
            next_album=disco.get("next_target"),
            track_count=track_count,
        ))

    # Sort by release_date (chronological). Then by title for tiebreaking.
    def sort_key(m: AlbumMatch):
        # Missing release_date goes to the end (use a high-value sentinel).
        return (0 if m.release_date else 1, m.release_date or "", m.page.title)
    matches.sort(key=sort_key)

    # Assign album_number = 1-based position.
    for i, m in enumerate(matches, start=1):
        m.album_number = i

    return matches


def get_tracklist_for_title(
    conn: sqlite3.Connection,
    release_title: str,
) -> list[Track]:
    """Resolve a release title to a page and return its tracklist.

    Returns an empty list if the title doesn't resolve or the page
    has no tracklist.
    """
    page, _ = resolve_title_with_aliases(conn, release_title)
    if not page:
        return []
    return get_tracklist(conn, page.id)


def _extract_discography_fields(conn: sqlite3.Connection, page_id: int) -> dict:
    """Extract Last/Next/release_date/artist from the infoboxes table."""
    out: dict = {}
    rows = conn.execute(
        "SELECT key, value FROM infoboxes WHERE page_id=?", (page_id,)
    ).fetchall()
    by_key = {r["key"].lower(): r["value"] for r in rows}

    # Last/Next fields
    for key in ("last", "next"):
        v = by_key.get(key.lower())
        if not v:
            continue
        m = re.search(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)\]\]", v)
        if m:
            out[f"{key}_target"] = m.group(1).replace("_", " ").strip()
        # Extract the number
        n = re.search(r"(\d+)(?:st|nd|rd|th)", v, re.IGNORECASE)
        if n:
            out[f"{key}_n"] = int(n.group(1))
        # Album/Single kind
        if "album" in v.lower():
            out[f"{key}_kind"] = "album"
        elif "single" in v.lower():
            out[f"{key}_kind"] = "single"

    # Album number inference
    if out.get("last_kind") == "album" and out.get("last_n") is not None:
        out["album_number"] = out["last_n"] + 1
    elif out.get("next_kind") == "album" and out.get("next_n") is not None:
        out["album_number"] = max(1, out["next_n"] - 1)

    if out.get("last_kind") == "single" and out.get("last_n") is not None:
        out["single_number"] = out["last_n"] + 1
    elif out.get("next_kind") == "single" and out.get("next_n") is not None:
        out["single_number"] = max(1, out["next_n"] - 1)

    # Release date
    rel = by_key.get("released")
    if rel:
        import datetime
        # "February 11, 2004" → "2004-02-11"
        m = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", rel)
        if m:
            try:
                dt = datetime.datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
                )
                out["release_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        if "release_date" not in out:
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", rel)
            if m:
                out["release_date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if "release_date" not in out:
            m = re.search(r"(\d{4})", rel)
            if m:
                out["release_year"] = int(m.group(1))

    return out


# ---------------------------------------------------------------------------
# Tracklist lookup
# ---------------------------------------------------------------------------

def get_tracklist(conn: sqlite3.Connection, album_page_id: int) -> list[Track]:
    """Get the tracklist for an album page, excluding karaoke entries
    by default (they're duplicates of the actual songs).
    """
    rows = conn.execute(
        """
        SELECT t.*, p.title AS page_title FROM tracklists t
        JOIN pages p ON p.id = t.page_id
        WHERE t.page_id = ?
        ORDER BY t.section, t.position
        """,
        (album_page_id,),
    ).fetchall()
    return [
        Track(
            position=r["position"],
            section=r["section"] or "Tracklist",
            raw=r["raw"],
            linked_title=r["linked_title"],
            is_karaoke=bool(r["is_karaoke"]),
            page_id=r["page_id"],
            page_title=r["page_title"],
        )
        for r in rows
    ]


def find_track_at_position(
    conn: sqlite3.Connection,
    album_page_id: int,
    position: int,
    *,
    include_karaoke: bool = False,
    prefer_section: str | None = None,
) -> Track | None:
    """Find the Nth track on an album page.

    If prefer_section is given (e.g. 'CD'), restrict to that section.
    By default excludes karaoke entries (instrumentals of the same song).
    """
    sql = """
        SELECT t.*, p.title AS page_title FROM tracklists t
        JOIN pages p ON p.id = t.page_id
        WHERE t.page_id = ?
    """
    params: list = [album_page_id]
    if not include_karaoke:
        sql += " AND t.is_karaoke = 0"
    if prefer_section:
        sql += " AND t.section = ?"
        params.append(prefer_section)
    sql += " ORDER BY t.position LIMIT 1 OFFSET ?"
    params.append(position - 1)
    row = conn.execute(sql, params).fetchone()
    if not row:
        return None
    return Track(
        position=row["position"],
        section=row["section"] or "Tracklist",
        raw=row["raw"],
        linked_title=row["linked_title"],
        is_karaoke=bool(row["is_karaoke"]),
        page_id=row["page_id"],
        page_title=row["page_title"],
    )


# ---------------------------------------------------------------------------
# Song info lookup
# ---------------------------------------------------------------------------

def get_song_info(conn: sqlite3.Connection, song_title: str) -> dict | None:
    """Look up a song page and return its infobox + a wikitext excerpt.

    The wikitext excerpt is the first paragraph (a useful summary for the
    chatbot to use as 'further information').
    """
    page, chain = resolve_title_with_aliases(conn, song_title)
    if not page:
        return None

    # Infobox fields
    rows = conn.execute(
        "SELECT key, value FROM infoboxes WHERE page_id=?", (page.id,)
    ).fetchall()
    info = {r["key"].lower(): r["value"] for r in rows}

    # First paragraph: find the first non-empty line after the infobox template.
    wt_row = conn.execute(
        "SELECT wikitext FROM pages WHERE id=?", (page.id,)
    ).fetchone()
    wikitext = wt_row["wikitext"] if wt_row else ""
    intro = _first_paragraph(wikitext)

    return {
        "page": page,
        "alias_chain": chain,
        "infobox": info,
        "intro": intro,
        "wikitext": wikitext,
    }


def _first_paragraph(wikitext: str) -> str:
    """Extract the first non-template paragraph from a page.

    Strips templates, file refs, wikilinks, and HTML tags so the
    returned text reads as natural prose. Stops at the first heading.

    Templates are detected as a balanced region of {{ ... }} that may
    span multiple lines.
    """
    if not wikitext:
        return ""

    # First pass: find the end of the first template (if any) at the
    # top of the page. Templates can span multiple lines, so we walk
    # the text counting {{ and }} occurrences.
    text = wikitext
    if text.lstrip().startswith("{{"):
        # Skip past the first balanced template.
        depth = 0
        i = 0
        while i < len(text):
            if text[i:i+2] == "{{":
                depth += 1
                i += 2
                continue
            if text[i:i+2] == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    break
                continue
            i += 1
        # Skip any whitespace/newlines after the template.
        text = text[i:].lstrip("\n\r ")

    # Now extract the first paragraph (text up to first blank line or heading).
    lines = text.splitlines()
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        # Skip file refs and headings
        if stripped.startswith(("[[File:", "[[Image:", "[[Category:")):
            continue
        if re.match(r"^={2,6}\s", stripped):
            break
        paragraph.append(stripped)

    out = " ".join(paragraph)
    # Clean wikilinks, HTML for natural reading.
    out = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", out)
    out = re.sub(r"<[^>]+>", "", out)
    out = re.sub(r"'''+", "", out)  # bold/italic markers
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Convenience: high-level answer
# ---------------------------------------------------------------------------

def answer_track_position(
    conn: sqlite3.Connection,
    artist: str,
    album_position: int,
    track_position: int,
) -> dict | None:
    """Answer: 'What was track N of <artist>'s Mth album?'

    Returns a dict like:
        {
            'artist': Page,
            'album': AlbumMatch,
            'track': Track,
            'song_info': dict (if track links to a real song page),
            'alternatives': [AlbumMatch] (other plausible matches),
        }
    """
    artist_page = find_artist_page(conn, artist)
    if not artist_page:
        return None

    albums = find_album_for_artist(conn, artist_page, kind="album")
    if not albums:
        return None

    target_album = None
    for a in albums:
        if a.album_number == album_position:
            target_album = a
            break
    if target_album is None:
        # If no exact positional match, use position-th in the list
        if 1 <= album_position <= len(albums):
            target_album = albums[album_position - 1]

    if target_album is None:
        return {"artist": artist_page, "albums": albums, "track": None}

    track = find_track_at_position(conn, target_album.page.id, track_position)
    song_info = None
    if track and track.linked_title:
        song_info = get_song_info(conn, track.linked_title)

    return {
        "artist": artist_page,
        "album": target_album,
        "track": track,
        "song_info": song_info,
        "all_albums": albums,
    }