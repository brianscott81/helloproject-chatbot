"""
Build a SQLite index from a MediaWiki XML dump (Fandom wiki).

Designed for the Hello! Project Wiki but general enough to be reused.

Strategy:
  - Stream-parse the XML using mwxml (no DOM in memory).
  - For each main-namespace page, store wikitext + metadata.
  - Detect CD/Album infoboxes and extract structured fields
    (type, artist, released, Last/Next discography pointers,
     album number, single number, etc.).
  - Extract tracklists from ==Tracklist== sections, parsing
    `# [[Song]]` numbered lists. Skip "Original Karaoke" entries.
  - Store all wikilinks (resolved to canonical titles via redirect
    map if available; otherwise left as raw).

Usage:
    python build_index.py <xml_dump> <sqlite_out>

Schema overview:
    pages           id, namespace, title, is_redirect, redirect_to,
                    wikitext, wikitext_len
    infoboxes      page_id, template_name, key, value
    tracklists     page_id, position, section, raw, linked_title
    links          from_page_id, target_title
    redirects      from_title, to_title
    aliases        alias_title, canonical_title  (other-name pointers
                                                  we discover in
                                                  infoboxes)

All extraction is best-effort. Missing data just leaves a NULL.
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path

import mwparserfromhell
import mwxml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

DROP TABLE IF EXISTS pages;
DROP TABLE IF EXISTS infoboxes;
DROP TABLE IF EXISTS tracklists;
DROP TABLE IF EXISTS links;
DROP TABLE IF EXISTS redirects;
DROP TABLE IF EXISTS aliases;

CREATE TABLE pages (
    id           INTEGER PRIMARY KEY,
    namespace    INTEGER NOT NULL,
    title        TEXT NOT NULL,
    is_redirect  INTEGER NOT NULL,
    redirect_to  TEXT,
    wikitext     TEXT,
    wikitext_len INTEGER
);
CREATE INDEX idx_pages_title ON pages(title);
CREATE INDEX idx_pages_ns_title ON pages(namespace, title);

CREATE TABLE redirects (
    from_title TEXT PRIMARY KEY,
    to_title   TEXT NOT NULL
);

CREATE TABLE aliases (
    alias_title    TEXT NOT NULL,
    canonical_id   INTEGER NOT NULL,
    canonical_title TEXT NOT NULL,
    source         TEXT,  -- 'redirect', 'infobox', etc.
    PRIMARY KEY (alias_title)
);
CREATE INDEX idx_aliases_canonical ON aliases(canonical_id);

CREATE TABLE infoboxes (
    page_id       INTEGER NOT NULL,
    template_name TEXT NOT NULL,
    key           TEXT NOT NULL,
    value         TEXT
);
CREATE INDEX idx_infobox_page ON infoboxes(page_id);
CREATE INDEX idx_infobox_key ON infoboxes(key);

CREATE TABLE tracklists (
    page_id     INTEGER NOT NULL,
    section     TEXT,
    position    INTEGER NOT NULL,
    raw         TEXT NOT NULL,        -- the raw line, e.g. "[[CRAZY ABOUT YOU]]"
    linked_title TEXT,                -- resolved title if it was a wikilink
    is_karaoke  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_tracklists_page ON tracklists(page_id);

CREATE TABLE links (
    from_page_id INTEGER NOT NULL,
    target_title TEXT NOT NULL
);
CREATE INDEX idx_links_target ON links(target_title);
CREATE INDEX idx_links_from ON links(from_page_id);
"""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Strip the leading wikilink noise and pull out the canonical target title.
WIKILINK_RE = re.compile(r"^\[\[(?:[^|\]]*\|)?([^\]|#]+)")


def clean_title(raw: str) -> str:
    """Normalize a wiki title (strip whitespace, underscores)."""
    if raw is None:
        return ""
    return raw.replace("_", " ").strip()


def first_wikilink_target(text: str) -> str | None:
    """Return the canonical target title of the first wikilink in text,
    or None if no wikilink. Ignores File:, Category:, etc."""
    if not text:
        return None
    m = WIKILINK_RE.search(text)
    if not m:
        return None
    title = clean_title(m.group(1))
    if title.startswith(("File:", "Image:", "Category:", "Help:", "Side:")):
        return None
    return title


def template_name(name: str) -> str:
    """Normalize a template name (strip 'Template:' prefix and namespace)."""
    name = name.strip()
    if name.lower().startswith("template:"):
        name = name[len("Template:"):]
    # Some pages use {{Template:CD Infobox|...}} rather than {{CD Infobox|...}}.
    return name


def parse_infobox(wikitext: str) -> tuple[str, dict[str, str]] | None:
    """Find the first {CD Infobox|...} or similar template and return
    (template_name, {key: value}). Returns None if no infobox found."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
    except Exception:
        return None
    for tmpl in parsed.filter_templates():
        tname = template_name(str(tmpl.name))
        # The wiki uses 'CD Infobox' for all releases. Some pages
        # also have 'Unit Infobox', 'Game Infobox', 'Person Infobox'.
        if "Infobox" not in tname and tname not in ("CD Infobox",):
            continue
        params: dict[str, str] = {}
        for p in tmpl.params:
            key = str(p.name).strip()
            try:
                value = str(p.value).strip()
            except Exception:
                value = ""
            params[key] = value
        return tname, params
    return None


def extract_tracklist(
    wikitext: str,
) -> list[tuple[str, int, str, str | None, bool]]:
    """Find all `==Tracklist==` (and ===CD=== subsections) and pull
    numbered list items.

    mwparserfromhell flattens ordered lists: a `#Foo\\n#Bar\\n` block
    becomes [Tag('#'), Text('Foo\\n'), Tag('#'), Text('Bar\\n')]. We
    walk top-level nodes, watch for `Tag('#')` list markers while inside
    a Tracklist section, and grab the item text up to the next marker or
    non-list node.

    Returns list of (section_name, position, raw, linked_title, is_karaoke).
    """
    try:
        parsed = mwparserfromhell.parse(wikitext)
    except Exception:
        return []

    results: list[tuple[str, int, str, str | None, bool]] = []
    current_section = ""
    in_tracklist = False
    current_item: list = []

    def flush_item():
        nonlocal current_item
        if not current_item:
            return
        raw = "".join(str(x) for x in current_item).strip()
        # Skip empty and divider-only items
        if not raw or raw in ("&nbsp;", "----"):
            current_item = []
            return
        linked = first_wikilink_target(raw)
        is_karaoke = bool(re.search(
            r"karaoke|Original\s+Karaoke|オリジナル・カラオケ",
            raw, re.IGNORECASE))
        position = len(results) + 1
        results.append((current_section, position, raw, linked, is_karaoke))
        current_item = []

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Heading):
            # Heading ends any pending list item
            flush_item()
            heading_text = str(node.title).strip()
            # Section names we treat as tracklist sections. Be permissive:
            # "Tracklist", "Track List", "Songs", "Tracklist (CD)", etc.
            # Inside a tracklist section, sub-headings like ===CD=== or
            # ===Disc 1=== are valid sub-sections.
            if re.match(r"^Track\s*List", heading_text, re.IGNORECASE):
                in_tracklist = True
                current_section = heading_text
            elif in_tracklist and node.level <= 2:
                # leaving the tracklist
                in_tracklist = False
                current_section = ""
            elif in_tracklist:
                # sub-section like "CD", "DVD", "Disc 1"
                current_section = heading_text
            continue

        if not in_tracklist:
            continue

        # We're inside a Tracklist section. Look for `li` list markers.
        # `node` will be a Tag with tag == 'li' or a Text/Wikilink/etc.
        if isinstance(node, mwparserfromhell.nodes.Tag) and node.tag == "li":
            # Start of a new item; flush previous one.
            flush_item()
            continue

        # Anything else in the tracklist section: append to current item.
        # Skip pure whitespace text nodes.
        if isinstance(node, mwparserfromhell.nodes.Text) and not str(node).strip():
            continue
        current_item.append(node)

    flush_item()

    # Fallback: regex scan if the parser missed everything.
    # Some wikitext mixes HTML tables and bullet lists in ways mwparserfromhell
    # silently misparses.
    if not results:
        current_section = "Tracklist"
        in_tracklist = False
        position = 0
        for line in wikitext.splitlines():
            stripped = line.strip()
            if re.match(r"^==+\s*Track\s*List", stripped, re.IGNORECASE):
                in_tracklist = True
                continue
            if in_tracklist and re.match(r"^==", stripped):
                in_tracklist = False
                continue
            if in_tracklist:
                m = re.match(r"^#\s*(.*)$", stripped)
                if m:
                    raw = m.group(1).strip()
                    if not raw:
                        continue
                    linked = first_wikilink_target(raw)
                    is_karaoke = bool(re.search(r"karaoke|Original\s+Karaoke|オリジナル・カラオケ", raw, re.IGNORECASE))
                    position += 1
                    results.append((current_section, position, raw, linked, is_karaoke))

    return results


# Parse text like '[[X|Y]] 2nd Album (2004)' -> ('X', 2, 'Album', 2004)
DISCO_HINT_RE = re.compile(
    r"\[\[(?:[^|\]]*\|)?([^\]|#]+)\]\]\s*(\d+)(?:st|nd|rd|th)?\s*(Album|Single|Mini\s*Album|EP)\s*(?:\((\d{4})\))?",
    re.IGNORECASE,
)


def extract_discography_hints(infobox_params: dict[str, str]) -> dict:
    """Parse Last/Next/Album/SingleN fields from a CD Infobox.

    Returns dict like {'last_album': 'Foo', 'next_album': 'Bar',
    'album_number': 2, 'single_number': 5, 'last_album_n': 1,
    'next_album_n': 2, ...}.
    """
    out: dict = {}

    def parse_hint(value: str) -> tuple[str | None, int | None, str | None, int | None]:
        if not value:
            return None, None, None, None
        m = DISCO_HINT_RE.search(value)
        if not m:
            # Just grab the linked target if there is one
            tgt = first_wikilink_target(value)
            return tgt, None, None, None
        target = clean_title(m.group(1))
        n = int(m.group(2))
        kind = m.group(3).lower().replace(" ", " ")
        year = int(m.group(4)) if m.group(4) else None
        return target, n, kind, year

    for key in ("Last", "Next"):
        if key in infobox_params:
            tgt, n, kind, year = parse_hint(infobox_params[key])
            if tgt:
                out[f"{key.lower()}_target"] = tgt
            if n is not None:
                out[f"{key.lower()}_n"] = n
            if kind:
                out[f"{key.lower()}_kind"] = kind
            if year is not None:
                out[f"{key.lower()}_year"] = year

    # Album/single number can be inferred from "Last = [[X]] 1st Album"
    # on the *current* page's infobox. Find which of Last/Next says Album
    # with N = current page's position.
    if "last_kind" in out and out["last_kind"] == "album" and "last_n" in out:
        out["album_number"] = out["last_n"] + 1
    elif "next_kind" in out and out["next_kind"] == "album" and "next_n" in out:
        out["album_number"] = out["next_n"] - 1

    if "last_kind" in out and out["last_kind"] == "single" and "last_n" in out:
        out["single_number"] = out["last_n"] + 1
    elif "next_kind" in out and out["next_kind"] == "single" and "next_n" in out:
        out["single_number"] = out["next_n"] - 1

    # Release date -> ISO 8601 if possible.
    if "released" in infobox_params:
        d = infobox_params["released"]
        # Take the first 'Month Day, Year' pattern
        m = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", d)
        if m:
            import datetime
            try:
                dt = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y")
                out["release_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", d)
        if m and "release_date" not in out:
            out["release_date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # Just the year as fallback
        if "release_date" not in out:
            m = re.search(r"(\d{4})", d)
            if m:
                out["release_year"] = int(m.group(1))

    return out


def extract_links(wikitext: str, max_links: int = 2000) -> list[str]:
    """Extract all wikilink targets from wikitext (excluding File/Category/etc)."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
    except Exception:
        return []
    out = []
    for link in parsed.filter_wikilinks():
        if len(out) >= max_links:
            break
        target = clean_title(str(link.title))
        if not target:
            continue
        if target.startswith(("File:", "Image:", "Category:", "Help:")):
            continue
        out.append(target)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(xml_path: Path, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()

    pages_inserted = 0
    infoboxes_inserted = 0
    tracklists_inserted = 0
    links_inserted = 0
    redirects_inserted = 0

    t0 = time.time()
    print(f"Reading {xml_path} ({xml_path.stat().st_size / 1024 / 1024:.1f} MB)...")

    with xml_path.open("rb") as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump.pages:
            # Skip non-content namespaces. We keep ns=0 (main), ns=4 (project),
            # ns=10 (template), ns=14 (category) lightly. For v1 we focus on
            # ns=0 — that's where albums, songs, and artists live.
            if page.namespace != 0:
                continue

            # Only process the current (first) revision.
            try:
                rev = next(iter(page))
            except StopIteration:
                continue
            wikitext = rev.text or ""

            # Detect redirects: mwxml exposes the redirect target as a plain
# string on `page.redirect`. (Earlier mwxml versions wrapped it in a
# Redirect object with a `.title` attribute, but 0.3.x returns a str.)
            redirect_to = None
            is_redirect = 0
            if page.redirect:
                redirect_to = clean_title(str(page.redirect))
                is_redirect = 1

            page_id = int(page.id)
            title = clean_title(str(page.title))

            conn.execute(
                "INSERT INTO pages (id, namespace, title, is_redirect, redirect_to, wikitext, wikitext_len) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (page_id, 0, title, is_redirect, redirect_to, wikitext, len(wikitext)),
            )
            pages_inserted += 1

            if is_redirect:
                conn.execute(
                    "INSERT OR REPLACE INTO redirects (from_title, to_title) VALUES (?, ?)",
                    (title, redirect_to),
                )
                # Alias resolution is deferred to post-processing — we may
                # not have inserted the target yet at this point.
                redirects_inserted += 1
                continue

            # Infobox extraction
            ibx = parse_infobox(wikitext)
            if ibx:
                tname, params = ibx
                rows = [(page_id, tname, k, v) for k, v in params.items()]
                conn.executemany(
                    "INSERT INTO infoboxes (page_id, template_name, key, value) VALUES (?, ?, ?, ?)",
                    rows,
                )
                infoboxes_inserted += len(rows)

            # Tracklist extraction
            tl = extract_tracklist(wikitext)
            if tl:
                rows = [
                    (page_id, section, position, raw, linked, 1 if karaoke else 0)
                    for section, position, raw, linked, karaoke in tl
                ]
                conn.executemany(
                    "INSERT INTO tracklists (page_id, section, position, raw, linked_title, is_karaoke) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                tracklists_inserted += len(rows)

            # Link extraction (cap to keep the build fast)
            links = extract_links(wikitext, max_links=500)
            if links:
                conn.executemany(
                    "INSERT INTO links (from_page_id, target_title) VALUES (?, ?)",
                    [(page_id, t) for t in links],
                )
                links_inserted += len(links)

            if pages_inserted % 500 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = pages_inserted / elapsed if elapsed > 0 else 0
                print(
                    f"  [{elapsed:6.1f}s] pages={pages_inserted}  "
                    f"infoboxes={infoboxes_inserted}  tracks={tracklists_inserted}  "
                    f"links={links_inserted}  ({rate:.0f} pg/s)",
                    flush=True,
                )

    conn.commit()

    # ----- post-processing -----

    # Resolve redirect chains into the aliases table.
    # Some redirects point to other redirects (chains of length 1-3).
    # We walk each redirect iteratively until we hit a non-redirect page.
    print("Resolving redirect aliases...")
    cur = conn.cursor()
    cur.execute("SELECT from_title, to_title FROM redirects")
    redirect_pairs = cur.fetchall()

    aliases_inserted = 0
    for from_t, to_t in redirect_pairs:
        # Walk the redirect chain.
        seen = set()
        current = to_t
        canonical_title = None
        canonical_id = None
        while current and current not in seen:
            seen.add(current)
            cur.execute(
                "SELECT id, is_redirect, redirect_to FROM pages WHERE title=? LIMIT 1",
                (current,),
            )
            row = cur.fetchone()
            if not row:
                # target page not found in dump — leave canonical as the last known target
                canonical_title = current
                break
            pid, is_red, red_to = row
            if not is_red:
                canonical_id = pid
                canonical_title = current
                break
            current = red_to
        if canonical_title is not None:
            cur.execute(
                "INSERT OR REPLACE INTO aliases (alias_title, canonical_id, canonical_title, source) "
                "VALUES (?, ?, ?, 'redirect')",
                (from_t, canonical_id or 0, canonical_title),
            )
            aliases_inserted += 1
    conn.commit()

    # Build useful indexes for the query layer.
    print("Building indexes...")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_links_combo ON links(from_page_id, target_title);
        CREATE INDEX IF NOT EXISTS idx_tracklists_link ON tracklists(linked_title);
    """)
    conn.commit()

    # Report
    cur = conn.cursor()
    print("\n=== Build summary ===")
    for label, sql in [
        ("pages", "SELECT COUNT(*) FROM pages"),
        ("redirects", "SELECT COUNT(*) FROM redirects"),
        ("infoboxes", "SELECT COUNT(*) FROM infoboxes"),
        ("tracklists", "SELECT COUNT(*) FROM tracklists"),
        ("links", "SELECT COUNT(*) FROM links"),
        ("aliases", "SELECT COUNT(*) FROM aliases"),
    ]:
        cur.execute(sql)
        print(f"  {label}: {cur.fetchone()[0]}")
    print(f"  wall time: {time.time() - t0:.1f}s")
    print(f"  db size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))