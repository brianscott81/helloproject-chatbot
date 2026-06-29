"""
Web interface for the Hello! Project Wiki chatbot.

A small stdlib HTTP server that wraps the chatbot in a browser-friendly
UI. Per-session conversation state is held in memory and keyed by a
random session ID stored in a cookie.

Endpoints:
  GET  /             -> serves web/index.html
  GET  /static/*     -> serves CSS / JS / etc. from web/
  POST /api/chat     -> JSON {question: "..."}  -> JSON {answer: "..."}
  POST /api/reset    -> clears the session conversation
  GET  /api/help     -> list of available slash commands

Run with:
    python web.py                       # defaults: 127.0.0.1:8000
    python web.py --port 9000
    python web.py --no-llm              # template formatters only

Binds to 127.0.0.1 by default for safety. Use --host 0.0.0.0 to
expose on a network (do this only with auth in place).
"""
from __future__ import annotations

import argparse
import http.cookies
import json
import os
import re
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Make chat.py importable as a library
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chat
from conversation import Conversation, format_prior_turns_for_llm

# Static file directory (next to this file)
WEB_DIR = HERE / "web"

# Per-session state. Keyed by session_id (random hex string).
# Each value is a dict {conversation: Conversation, llm: LLMSynthesizer}.
# Guarded by a lock because BaseHTTPRequestHandler is multi-threaded.
_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _create_session() -> str:
    """Create a new session and return its ID."""
    sid = secrets.token_urlsafe(24)
    with _sessions_lock:
        _sessions[sid] = {
            "conversation": Conversation(),
        }
    return sid


def _get_session(sid: str) -> dict[str, Any] | None:
    """Look up a session by ID. Returns None if unknown."""
    with _sessions_lock:
        return _sessions.get(sid)


def _reset_session(sid: str) -> None:
    """Drop a session's conversation (the /new slash command)."""
    with _sessions_lock:
        if sid in _sessions:
            _sessions[sid]["conversation"] = Conversation()


def _get_or_create_session(cookie_header: str | None) -> tuple[str, bool]:
    """Parse the session cookie or create a new one.

    Returns (session_id, is_new).
    """
    if cookie_header:
        cookies = http.cookies.SimpleCookie(cookie_header)
        if "session" in cookies:
            sid = cookies["session"].value
            if _get_session(sid) is not None:
                return sid, False
    return _create_session(), True


def _format_prior_turns_for_session(sid: str) -> str | None:
    """Format the recent conversation for the LLM's prior-turn context.

    Returns None if there are no prior turns.
    """
    sess = _get_session(sid)
    if sess is None:
        return None
    conv = sess["conversation"]
    if not conv.turns:
        return None
    # Pass everything except the most recent turn (which is the
    # current user turn being processed).
    return format_prior_turns_for_llm(conv.turns[:-1])


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP request handler. Routes by path."""

    # Set by main() before serve_forever() so the handler can find
    # the DB and chroma dir without reparsing.
    server_db_path: Path = None  # type: ignore[assignment]
    server_chroma_dir: Path = None  # type: ignore[assignment]
    server_llm = None  # LLMSynthesizer | None
    server_use_llm_tool_fallback: bool = True

    # Suppress default access-log spam; we print our own when needed.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, status: int, body: dict, set_cookie: str | None = None) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if set_cookie:
            self.send_header("Set-Cookie", set_cookie)
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, path: str) -> None:
        """Serve a file from web/. Returns False if not found."""
        # Strip leading slash, normalize
        rel = path.lstrip("/")
        # Prevent path traversal: reject anything with .. or absolute paths
        if ".." in rel.split("/") or rel.startswith("/"):
            self.send_error(400, "Bad path")
            return
        full = (WEB_DIR / rel).resolve()
        # Ensure the resolved path is still under WEB_DIR
        if not str(full).startswith(str(WEB_DIR.resolve())):
            self.send_error(400, "Bad path")
            return
        if not full.is_file():
            self.send_error(404, "Not found")
            return
        # Guess content type
        ext = full.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
            ".json": "application/json; charset=utf-8",
        }.get(ext, "application/octet-stream")
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return None
        if length > 1024 * 1024:  # 1 MB cap; questions shouldn't be larger
            self.send_error(413, "Request body too large")
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            self._send_static("index.html")
            return
        if path.startswith("/static/"):
            self._send_static(path[len("/static/"):])
            return
        if path == "/api/help":
            self._send_json(200, {
                "commands": [
                    {"name": "new", "description": "Clear the conversation"},
                    {"name": "help", "description": "Show this help"},
                ],
                "notes": [
                    "The web UI hides historical chats by default.",
                    "Use the 'New Chat' button (top-right) to start fresh.",
                ],
            })
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path

        # Get or create session
        sid, is_new = _get_or_create_session(self.headers.get("Cookie"))
        cookie_header = f"session={sid}; Path=/; HttpOnly; SameSite=Lax"

        if path == "/api/chat":
            body = self._read_json_body()
            if body is None:
                return  # already sent 4xx
            question = (body.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "Empty question"}, set_cookie=cookie_header)
                return
            self._handle_chat(sid, question, is_new, cookie_header)
            return

        if path == "/api/reset":
            _reset_session(sid)
            self._send_json(200, {"status": "ok", "message": "Conversation reset"}, set_cookie=cookie_header)
            return

        self.send_error(404, "Not found")

    def _handle_chat(self, sid: str, question: str, is_new: bool, cookie_header: str) -> None:
        """Process a chat question: rewrite, classify, execute, synthesize."""
        sess = _get_session(sid)
        if sess is None:
            self._send_json(500, {"error": "Session lost"}, set_cookie=cookie_header)
            return
        conv = sess["conversation"]

        # Record the user turn
        conv.add_user_turn(question)

        # Format prior turns for LLM context
        prior_turns_str = format_prior_turns_for_llm(conv.turns[:-1])

        try:
            answer = chat.answer_question(
                question,
                self.server_db_path,
                self.server_chroma_dir,
                verbose=False,
                llm=self.server_llm,
                prior_turns_str=prior_turns_str,
                use_llm_tool_fallback=self.server_use_llm_tool_fallback,
            )
        except Exception as e:
            answer = f"Error: {e}"

        # Record the assistant turn
        conv.add_assistant_turn(answer)

        self._send_json(200, {
            "answer": answer,
            "is_new_session": is_new,
        }, set_cookie=cookie_header)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def make_handler_class() -> type:
    """Return a ChatHandler subclass with the server's settings baked in."""
    class _Handler(ChatHandler):
        pass
    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Web interface for the Hello! Project Wiki chatbot")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--db", default=str(HERE / "helloproject.db"), help="Path to SQLite database")
    parser.add_argument("--chroma-dir", default=str(HERE / "chroma"), help="Path to Chroma embeddings directory")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM synthesis (template only)")
    parser.add_argument("--no-tool-llm", action="store_true", help="Disable LLM tool-use routing fallback")
    args = parser.parse_args()

    db_path = Path(args.db)
    chroma_dir = Path(args.chroma_dir)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Build it with: python build_index.py", file=sys.stderr)
        return 1
    if not chroma_dir.exists():
        print(f"Error: chroma dir not found at {chroma_dir}", file=sys.stderr)
        print("Build it with: python build_index.py", file=sys.stderr)
        return 1
    if not WEB_DIR.exists():
        print(f"Error: web/ directory not found at {WEB_DIR}", file=sys.stderr)
        return 1

    # Configure the handler class with server-side settings
    ChatHandler.server_db_path = db_path
    ChatHandler.server_chroma_dir = chroma_dir
    ChatHandler.server_use_llm_tool_fallback = not args.no_tool_llm

    if args.no_llm:
        ChatHandler.server_llm = None
    else:
        try:
            from llm import LLMSynthesizer
            ChatHandler.server_llm = LLMSynthesizer()
        except Exception as e:
            print(f"Warning: could not initialize LLM ({e}); running with templates only",
                  file=sys.stderr)
            ChatHandler.server_llm = None

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Hello! Project Wiki chatbot — web interface")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  DB:   {db_path}")
    print(f"  Chroma: {chroma_dir}")
    print(f"  LLM:  {'disabled' if args.no_llm else ChatHandler.server_llm.describe() if ChatHandler.server_llm else 'unavailable'}")
    print(f"  Tool-use LLM fallback: {'disabled' if args.no_tool_llm else 'enabled'}")
    print(f"  Press Ctrl-C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
