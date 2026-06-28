"""
Build a ChromaDB vector index of wiki pages for semantic search.

Strategy:
  - For each main-namespace page, chunk the wikitext into ~500-1000 token
    segments. Chunk boundaries follow ==Section== headings where possible.
  - Strip wikitext markup to plain text (rough pass) so embeddings aren't
    polluted by syntax. We keep section titles so chunks are self-describing.
  - Embed with a sentence-transformers model (BAAI/bge-small-en-v1.5 by
    default — fast and good for retrieval).
  - Store in ChromaDB with metadata {page_id, page_title, section, chunk_idx}.

This is the "semantic path" used for fuzzy questions that the structured
path can't handle ("tell me about Minimoni's history", "what happened to
Morning Musume in 2014", etc.).

Usage:
    python build_embeddings.py <sqlite_db> <chroma_persist_dir>

The first run will download the embedding model (~130MB for bge-small).
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
import hashlib
from pathlib import Path


def plainify_wikitext(wikitext: str) -> str:
    """Rough wikitext → plain text conversion. Not perfect, but good
    enough for embedding quality. Strips templates, tables, file refs,
    emphasis markers, etc.
    """
    if not wikitext:
        return ""

    text = wikitext

    # Drop templates entirely — their parameters are usually not natural
    # language and they confuse the embedder.
    text = re.sub(r"\{\{[^}]*\}\}", " ", text, flags=re.DOTALL)
    # Drop tables
    text = re.sub(r"\{\|[^}]*\|\}", " ", text, flags=re.DOTALL)
    # Drop image / file references
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Convert wikilinks [[X|Y]] or [[X]] → Y or X
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|#]+)(?:\|[^\]]*)?\]\]", r"\1", text)
    # External links [url label] → label
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", " ", text)
    # Headings: keep them but as plain text
    text = re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", r"\2\n", text, flags=re.MULTILINE)
    # Bold / italic
    text = text.replace("'''", "").replace("''", "")
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Ref tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Split text into chunks of ~max_chars, preferring paragraph boundaries.

    Tries to split at double-newline or sentence boundaries; falls back to
    hard splits at max_chars. Overlap is in characters (not tokens) but for
    our purposes it's fine.
    """
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    # First, split by paragraph (which we collapsed away) — split on sentence
    # boundaries instead.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars and current:
            chunks.append(current.strip())
            # Keep overlap from the previous chunk for context continuity.
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:] + " " + s
            else:
                current = s
        else:
            current = (current + " " + s).strip()
    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_by_section(wikitext: str) -> list[tuple[str, str]]:
    """Split a page into (section_title, section_text) tuples.

    Returns at least one chunk per page (the section-less intro).
    """
    if not wikitext:
        return []

    # Find level-2 headings as section boundaries
    sections: list[tuple[str, str]] = []
    current_title = "Intro"
    current_lines: list[str] = []

    for line in wikitext.splitlines():
        m = re.match(r"^(={2,3})\s*(.+?)\s*\1\s*$", line.strip())
        if m:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append((current_title, text))
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_title, text))

    # If the page had no headings, sections is a single "Intro" chunk.
    return sections


def build(db_path: Path, chroma_dir: Path, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model {model_name} (downloads on first run)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    print("Setting up ChromaDB...")
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(
        name="helloproject",
        metadata={"hnsw:space": "cosine"},
    )

    print("Reading pages and chunking...")
    t0 = time.time()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT p.id, p.title, p.wikitext FROM pages p
        WHERE p.namespace = 0 AND p.is_redirect = 0
          AND LENGTH(p.wikitext) > 200
    """).fetchall()

    print(f"  {len(rows)} pages to process")

    # Batch embed for speed
    batch_size = 64
    pending: list[dict] = []
    total_chunks = 0
    pages_done = 0

    for r in rows:
        sections = chunk_by_section(r["wikitext"])
        global_chunk_idx = 0  # counter across all sections of this page
        for section_title, section_text in sections:
            plain = plainify_wikitext(section_text)
            chunks = chunk_text(plain, max_chars=2000)
            for i, chunk in enumerate(chunks):
                # Prepend section + title for self-describing chunks
                heading = f"{r['title']} - {section_title}\n"
                content = heading + chunk
                pending.append({
                    # Globally unique ID: hash of (page_id, global_chunk_idx).
                    # We use a per-page global counter so that duplicate section
                    # names within a page (e.g. multiple "Single V" subsections)
                    # still produce distinct IDs.
                    "id": hashlib.sha1(
                        f"{r['id']}|{global_chunk_idx}".encode()
                    ).hexdigest()[:16],
                    "content": content,
                    "page_id": r["id"],
                    "page_title": r["title"],
                    "section": section_title,
                    "chunk_idx": i,
                })
                global_chunk_idx += 1
        pages_done += 1
        if pages_done % 500 == 0:
            print(f"  [{time.time()-t0:.1f}s] pages={pages_done} chunks={total_chunks + len(pending)}")

        if len(pending) >= batch_size:
            contents = [p["content"] for p in pending]
            embeddings = model.encode(contents, show_progress_bar=False).tolist()
            collection.add(
                ids=[p["id"] for p in pending],
                documents=contents,
                embeddings=embeddings,
                metadatas=[{
                    "page_id": p["page_id"],
                    "page_title": p["page_title"],
                    "section": p["section"],
                    "chunk_idx": p["chunk_idx"],
                } for p in pending],
            )
            total_chunks += len(pending)
            pending = []

    # Flush remainder
    if pending:
        contents = [p["content"] for p in pending]
        embeddings = model.encode(contents, show_progress_bar=False).tolist()
        collection.add(
            ids=[p["id"] for p in pending],
            documents=contents,
            embeddings=embeddings,
            metadatas=[{
                "page_id": p["page_id"],
                "page_title": p["page_title"],
                "section": p["section"],
                "chunk_idx": p["chunk_idx"],
            } for p in pending],
        )
        total_chunks += len(pending)

    print(f"\nDone. {total_chunks} chunks indexed in {time.time()-t0:.1f}s")
    print(f"ChromaDB persisted to: {chroma_dir}")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))