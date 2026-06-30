"""
Web interface for the Hello! Project Wiki chatbot.

A small stdlib HTTP server that wraps the chatbot in a browser-friendly
UI. The server is stateless: each /api/chat request carries its own
prior-turn context, and the server holds no per-user state. This
solves the per-user-isolation problem and avoids the memory leaks
of an in-process session dict.

Endpoints:
  GET  /             -> serves web/index.html
  GET  /static/*     -> serves CSS / JS / etc. from web/
  POST /api/chat     -> JSON {question: "...", prior_turns: [...]}
                       -> JSON {answer: "..."}
  GET  /api/help     -> list of available slash commands

prior_turns format: [{"role": "user"|"assistant", "content": "..."}, ...]
The server formats these into the LLM's prior-turn context block.

Run with:
    python web.py                       # defaults: 127.0.0.1:8000
    python web.py --port 9000
    python web.py --no-llm              # template formatters only

Binds to 127.0.0.1 by default for safety. Use --host 0.0.0.0 to
expose on a network (do this only with auth in place).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Make chat.py importable as a library
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chat
from conversation import format_prior_turns_for_llm

# Static file directory (next to this file)
WEB_DIR = HERE / "web"

# The server is stateless — no per-session storage, no cookies, no
# locks. All conversation context travels with each /api/chat request.


class _PriorTurn:
    """Adapter that gives the prior-turn list a .role / .content
    attribute interface, matching the Conversation.Turn class.

    format_prior_turns_for_llm uses attribute access, so dicts
    don't work directly. This lightweight wrapper keeps the web
    server free of conversation.py's heavier objects.
    """
    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


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

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, path: str) -> None:
        """Serve a file from web/. Returns False if not found."""
        rel = path.lstrip("/")
        if ".." in rel.split("/") or rel.startswith("/"):
            self.send_error(400, "Bad path")
            return
        full = (WEB_DIR / rel).resolve()
        if not str(full).startswith(str(WEB_DIR.resolve())):
            self.send_error(400, "Bad path")
            return
        if not full.is_file():
            self.send_error(404, "Not found")
            return
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

    @staticmethod
    def _validate_prior_turn(turn: Any) -> tuple[str, str] | None:
        """Validate one prior turn from the request body.

        Returns (role, content) on success, or None if invalid.
        Silently rejects malformed turns — the server is permissive
        and just ignores the bad ones.
        """
        if not isinstance(turn, dict):
            return None
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            return None
        if not isinstance(content, str):
            return None
        if len(content) > 20000:  # ~20KB cap per turn
            return None
        return role, content

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
                    {"name": "new", "description": "Start a fresh conversation (clears client-side history)"},
                    {"name": "help", "description": "Show this help"},
                ],
                "notes": [
                    "Conversation history is stored in your browser only.",
                    "Use the 'New Chat' button to start fresh.",
                    "Refreshing the page keeps your conversation (localStorage).",
                ],
            })
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path

        if path == "/api/chat":
            body = self._read_json_body()
            if body is None:
                return  # already sent 4xx
            self._handle_chat(body)
            return

        self.send_error(404, "Not found")

    def _handle_chat(self, body: dict) -> None:
        """Process a chat question with optional prior-turn context.

        Request body:
          {
            "question": "...",
            "prior_turns": [
              {"role": "user"|"assistant", "content": "..."},
              ...
            ]  # optional
          }

        Response:
          {"answer": "..."}
        """
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "Empty question"})
            return

        # Build a minimal Conversation-like object to feed format_prior_turns_for_llm.
        # The web client is stateless, but the LLM still needs the prior-turn
        # context to handle multi-turn correctly. We synthesize the structure
        # from the request body.
        prior_turns_raw = body.get("prior_turns") or []
        if not isinstance(prior_turns_raw, list):
            self._send_json(400, {"error": "prior_turns must be a list"})
            return

        # Cap to last MAX_PRIOR_TURNS to keep prompt size bounded.
        MAX_PRIOR_TURNS = 20
        validated_turns = []
        for t in prior_turns_raw[-MAX_PRIOR_TURNS:]:
            v = self._validate_prior_turn(t)
            if v is not None:
                role, content = v
                validated_turns.append(_PriorTurn(role, content))
        prior_turns_str = (
            format_prior_turns_for_llm(validated_turns) if validated_turns else None
        )

        # Capture the tool result via on_tool_complete so we can extract
        # entities and sources for the web UI to render as clickable links.
        # on_tool_complete is called by answer_question() after the tool
        # executes but before LLM synthesis. This is the documented hook
        # for inspecting tool output without re-running the tool.
        tool_result_holder: dict[str, Any] = {}

        def _on_tool_complete(call: dict, result: dict) -> None:
            tool_result_holder["call"] = call
            tool_result_holder["result"] = result

        try:
            answer = chat.answer_question(
                question,
                self.server_db_path,
                self.server_chroma_dir,
                verbose=False,
                llm=self.server_llm,
                prior_turns_str=prior_turns_str,
                use_llm_tool_fallback=self.server_use_llm_tool_fallback,
                on_tool_complete=_on_tool_complete,
            )
        except Exception as e:
            answer = f"Error: {e}"

        # Extract entities and sources from the captured tool result.
        # If the tool errored (no result), we still return a 200 with empty
        # meta so the client gets a clean response shape.
        meta = {"entities": [], "sources": []}
        if "result" in tool_result_holder:
            try:
                from chat_meta import extract_meta
                meta = extract_meta(
                    tool_result_holder["result"], self.server_db_path,
                )
            except Exception:
                # Meta extraction is best-effort; never fail the response
                # because of meta issues.
                pass

        self._send_json(200, {
            "answer": answer,
            "entities": meta.get("entities", []),
            "sources": meta.get("sources", []),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
