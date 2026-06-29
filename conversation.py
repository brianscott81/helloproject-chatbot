"""
Conversation memory for the interactive REPL.

The single-turn chatbot (chat.py answer_question) treats every question
in isolation. In a multi-turn conversation, follow-up questions often
omit context that was clear from the prior turn:

    > What was the 2nd track of Minimoni's 2nd album?
    CRAZY ABOUT YOU
    > Who produced it?                       <-- 'it' = CRAZY ABOUT YOU
    > What about the 1st album?              <-- keep artist = Minimoni
    > When did they disband?                 <-- 'they' = Minimoni

This module provides a `Conversation` class that:
  - Stores the prior turns (user + assistant messages + tool results).
  - Extracts "remembered entities" from each tool result (artist,
    album, track, song) so follow-up turns can reference them.
  - Performs argument-preservation: when a question is ambiguous (e.g.,
    "what about singles?"), reuse artist from the last artist-bearing
    turn but swap the other args.
  - Handles slash commands: /new, /history, /last, /exit, /help.

It does NOT replace the regex classifier in chat.py. The classifier
still runs on each turn — but with the option to receive a context
dict with remembered entities. If the classifier can extract args from
the question alone, it does so; otherwise it falls back to the
remembered entities.

The LLM synthesis layer (if enabled) is untouched — it still sees the
raw question. We do not feed prior turns into the LLM in v1; that's a
v2 feature.

Design choices:
  - Conversation state lives in memory only. No persistence between
    REPL sessions (yet).
  - Entity extraction is rule-based, not LLM-based. We're looking at
    tool_result shapes that we produced ourselves, so this is reliable.
  - Pronoun resolution ('it', 'they', 'that') is conservative: only the
    most unambiguous cases are substituted.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One user/assistant exchange."""
    role: str  # "user" or "assistant"
    content: str
    tool_name: str | None = None
    tool_result: dict | None = None
    # The route the tool took — useful for debugging.
    entities: dict = field(default_factory=dict)


@dataclass
class Context:
    """Entity memory carried between turns.

    Each field is the most-recently-mentioned entity of that type. Set
    fields are non-None. The classifier can read this to substitute
    missing arguments.
    """
    artist_id: int | None = None
    artist_title: str | None = None
    album_page_id: int | None = None
    album_title: str | None = None
    track_page_id: int | None = None
    track_title: str | None = None
    song_page_id: int | None = None
    song_title: str | None = None
    # For pronoun substitution: which entity does "it" refer to?
    last_singular_entity: str | None = None  # e.g. "song", "album", "track"
    last_singular_title: str | None = None

    def to_dict(self) -> dict:
        return {
            "artist": self.artist_title,
            "album": self.album_title,
            "track": self.track_title,
            "song": self.song_title,
            "last_singular": self.last_singular_entity,
        }


# ---------------------------------------------------------------------------
# Entity extraction from tool results
# ---------------------------------------------------------------------------

def extract_entities(tool_name: str, tool_result: dict) -> dict:
    """Pull entity references out of a tool_result so we can remember them.

    Returns a dict suitable for storing in a Turn's entities field. The
    Conversation.update_context() method then promotes specific fields
    into the Context object.
    """
    entities: dict[str, Any] = {}

    if "error" in tool_result:
        return entities

    if tool_name == "lookup_track":
        artist = tool_result.get("artist") or {}
        if artist.get("id"):
            entities["artist_id"] = artist["id"]
        if artist.get("title"):
            entities["artist_title"] = artist["title"]

        album = tool_result.get("album") or {}
        if album.get("id"):
            entities["album_page_id"] = album["id"]
        if album.get("title"):
            entities["album_title"] = album["title"]

        track = tool_result.get("track") or {}
        if track.get("linked_title"):
            entities["track_title"] = track["linked_title"]

        song = tool_result.get("song_info") or {}
        if song.get("id"):
            entities["song_page_id"] = song["id"]
        if song.get("title"):
            entities["song_title"] = song["title"]

    elif tool_name == "list_releases":
        # The artist is in the result but its id isn't (we only stored title).
        # Use the entity from the question if available.
        title = tool_result.get("artist")
        if title:
            entities["artist_title"] = title
            # We don't have an ID — classify_question will re-resolve it.

    elif tool_name == "get_song_info":
        song = tool_result
        if song.get("id"):
            entities["song_page_id"] = song["id"]
        if song.get("title"):
            entities["song_title"] = song["title"]
        artist = song.get("artist")
        if artist:
            entities["artist_title"] = artist

    elif tool_name == "semantic_search":
        chunks = tool_result.get("chunks") or []
        if chunks:
            top = chunks[0]
            entities["top_page_id"] = top.get("page_id")
            entities["top_page_title"] = top.get("page_title")

    return entities


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

# Slash commands.
COMMANDS = {
    "new", "history", "last", "exit", "quit", "help", "ctx",
}


class Conversation:
    """Multi-turn dialog state."""

    def __init__(self, max_turns: int = 100):
        self.turns: list[Turn] = []
        self.context = Context()
        self.max_turns = max_turns

    # --- mutation ---

    def add_user_turn(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))
        self._trim()

    def add_assistant_turn(
        self,
        content: str,
        tool_name: str | None = None,
        tool_result: dict | None = None,
    ) -> None:
        entities = extract_entities(tool_name or "", tool_result or {})
        self.turns.append(Turn(
            role="assistant",
            content=content,
            tool_name=tool_name,
            tool_result=tool_result,
            entities=entities,
        ))
        self._update_context(entities)
        self._trim()

    def reset(self) -> None:
        """Wipe history and context. Used by /new."""
        self.turns = []
        self.context = Context()

    # --- query ---

    def last_n(self, n: int = 10) -> list[Turn]:
        return self.turns[-n:]

    def last_assistant_entities(self) -> dict:
        """Return the entities from the most recent assistant turn, if any."""
        for t in reversed(self.turns):
            if t.role == "assistant" and t.entities:
                return t.entities
        return {}

    def format_history(self, n: int = 10) -> str:
        recent = self.last_n(n)
        if not recent:
            return "(no history yet)"
        lines = []
        for i, t in enumerate(recent, start=1):
            role = ">" if t.role == "user" else "<"
            content = t.content
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"  {i:3}. {role} {content}")
        return "\n".join(lines)

    def format_context(self) -> str:
        c = self.context.to_dict()
        items = [(k, v) for k, v in c.items() if v]
        if not items:
            return "(no remembered entities — start with a question that names an artist or song)"
        lines = ["Currently remembered entities:"]
        for k, v in items:
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    # --- internal ---

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def _update_context(self, entities: dict) -> None:
        """Promote specific entities into the context memory."""
        if "artist_id" in entities:
            self.context.artist_id = entities["artist_id"]
        if "artist_title" in entities:
            self.context.artist_title = entities["artist_title"]
        if "album_page_id" in entities:
            self.context.album_page_id = entities["album_page_id"]
            self.context.album_title = entities.get("album_title", self.context.album_title)
        if "track_title" in entities:
            self.context.track_title = entities["track_title"]
            self.context.last_singular_entity = "track"
            self.context.last_singular_title = entities["track_title"]
        if "song_page_id" in entities:
            self.context.song_page_id = entities["song_page_id"]
            self.context.song_title = entities.get("song_title", self.context.song_title)
            # A song takes priority over a track for "it" pronoun resolution.
            self.context.last_singular_entity = "song"
            self.context.last_singular_title = entities.get("song_title")
        if "top_page_id" in entities and not self.context.last_singular_entity:
            # Only set this if we don't already have a more specific entity.
            self.context.last_singular_entity = "page"
            self.context.last_singular_title = entities.get("top_page_title")


# ---------------------------------------------------------------------------
# Pronoun substitution
# ---------------------------------------------------------------------------

# Pronouns that map to the last singular entity in context.
SINGULAR_PRONOUNS = {"it", "this", "that"}
PLURAL_PRONOUNS = {"they", "them", "their"}


def substitute_pronouns(question: str, ctx: Context) -> tuple[str, str | None]:
    """Replace ambiguous pronouns in the question with the remembered entity.

    Returns (rewritten_question, note). The note is a short string
    explaining what we substituted (or None if no substitution was
    done). Callers can use the note to inform the user.

    Conservative: only the most unambiguous pronouns are replaced. If
    the context has no remembered entity, we do nothing.
    """
    words = question.split()
    new_words = []
    note = None
    substituted = False

    for w in words:
        # Strip punctuation for matching but preserve it in output.
        bare = w.strip(".,?!;:")
        punc = w[len(bare):]
        lower = bare.lower()

        if lower in SINGULAR_PRONOUNS and ctx.last_singular_title:
            replacement = ctx.last_singular_title
            new_words.append(replacement + punc)
            substituted = True
            note = f"(interpreted '{bare}' as '{replacement}')"
        elif lower in PLURAL_PRONOUNS and ctx.artist_title:
            replacement = ctx.artist_title
            new_words.append(replacement + punc)
            substituted = True
            note = f"(interpreted '{bare}' as '{replacement}')"
        else:
            new_words.append(w)

    if not substituted:
        return question, None
    return " ".join(new_words), note


# ---------------------------------------------------------------------------
# Argument preservation
# ---------------------------------------------------------------------------

def preserve_arguments(
    question: str,
    ctx: Context,
) -> tuple[str, str | None]:
    """If the question is missing the artist/track context, fill it in
    from the conversation context.

    This is a heuristic — we only inject the artist if the question
    clearly doesn't name one. We detect that by trying to extract an
    artist from the question itself (via a simple Capitalized-Phrase
    heuristic). If we can't find one and the context has a remembered
    artist, we prepend it.

    Returns (rewritten_question, note).
    """
    # Cheap heuristic: a Capitalized phrase is likely an artist name.
    # Look for any title-cased multi-word sequence.
    has_capitalized_phrase = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", question))

    # Heuristic: if the question starts with "what about", "and", or
    # "what about the X", the user is clearly following up on the
    # prior context.
    is_followup_start = bool(re.match(
        r"^\s*(what about|how about|and\b|also,?|and how about)",
        question, re.IGNORECASE,
    ))

    # If the question explicitly references a known context entity by
    # name, we don't need to do anything.
    mentions_known = False
    if ctx.artist_title and ctx.artist_title.lower() in question.lower():
        mentions_known = True
    if ctx.album_title and ctx.album_title.lower() in question.lower():
        mentions_known = True
    if ctx.song_title and ctx.song_title.lower() in question.lower():
        mentions_known = True

    if mentions_known or ctx.artist_title is None:
        return question, None

    if is_followup_start or not has_capitalized_phrase:
        # Inject the remembered artist into the question.
        injected = f"In {ctx.artist_title}, {question.lstrip().rstrip('?.')}"
        note = f"(interpreted as a question about {ctx.artist_title})"
        return injected + "?", note

    return question, None


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedInput:
    is_command: bool
    command: str | None
    args: str
    raw: str


def parse_input(raw: str) -> ParsedInput:
    """Classify a REPL line as a command or a question."""
    s = raw.strip()
    if s.startswith("/"):
        parts = s[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return ParsedInput(is_command=True, command=cmd, args=args, raw=s)
    return ParsedInput(is_command=False, command=None, args="", raw=s)


# ---------------------------------------------------------------------------
# High-level: prepare a question for the classifier
# ---------------------------------------------------------------------------

def prepare_question(
    question: str,
    ctx: Context,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    """Rewrite a raw user question using conversation context.

    Returns (rewritten_question, notes). The notes list contains
    human-readable strings describing each substitution, suitable for
    printing to stderr.
    """
    notes: list[str] = []
    q = question

    # Step 1: pronoun substitution (e.g. "it" -> "CRAZY ABOUT YOU")
    q, note = substitute_pronouns(q, ctx)
    if note:
        notes.append(note)

    # Step 2: argument preservation (e.g. "what about singles?" -> "In Morning Musume, what about singles?")
    q, note = preserve_arguments(q, ctx)
    if note:
        notes.append(note)

    if verbose and notes:
        for n in notes:
            print(f"[context] {n}", file=sys.stderr)

    return q, notes