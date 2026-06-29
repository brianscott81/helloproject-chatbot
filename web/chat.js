/* Frontend for the Hello! Project Wiki chatbot.
 *
 * Stateless design: the server holds no per-user state. We keep the
 * conversation history in the browser (in-memory + localStorage)
 * and send the prior turns with each /api/chat request.
 *
 * - Stores conversation history in JS state (and localStorage so
 *   it survives page reloads)
 * - "New Chat" button clears the local history and the visible chat
 * - "?" button shows the help modal
 * - Renders assistant Markdown via marked.js
 *
 * Per the user's spec: the web UI hides historical chats by default.
 * Each browser (or browser tab, after a refresh) starts with an
 * empty chat. The history is local to the browser.
 */

(function () {
  "use strict";

  // ---- Configure marked.js ----
  if (window.marked && marked.setOptions) {
    marked.setOptions({
      gfm: true,            // GitHub-flavored Markdown (lists, tables, etc.)
      breaks: true,         // Treat \n as <br>
      headerIds: false,     // Don't add id="..." to headers
      mangle: false,        // Don't mangle email addresses
    });
  }

  // ---- DOM ----
  const messagesEl = document.getElementById("messages");
  const emptyStateEl = document.getElementById("empty-state");
  const form = document.getElementById("composer");
  const input = document.getElementById("question");
  const sendBtn = document.getElementById("send");
  const newChatBtn = document.getElementById("new-chat");
  const helpBtn = document.getElementById("help");
  const helpModal = document.getElementById("help-modal");
  const helpClose = document.getElementById("help-close");
  const helpList = document.getElementById("help-list");

  // ---- State (in-memory, mirrored to localStorage) ----
  // conversationHistory is a list of {role, content} objects.
  // It is sent with each /api/chat request as prior_turns, so the
  // server sees the full context but stores nothing.
  const STORAGE_KEY = "helloproject-wiki-chat-history-v1";
  let conversationHistory = loadHistory();

  let inFlight = false;

  // ---- History persistence ----

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      // Filter to valid turns only.
      return parsed.filter(
        (t) => t &&
          (t.role === "user" || t.role === "assistant") &&
          typeof t.content === "string"
      );
    } catch (e) {
      // Corrupt localStorage data — start fresh.
      return [];
    }
  }

  function saveHistory() {
    try {
      // Cap to last 50 turns to keep localStorage small.
      const trimmed = conversationHistory.slice(-50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    } catch (e) {
      // localStorage might be full or disabled (private mode).
      // Best-effort only.
    }
  }

  // ---- Helpers ----

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderMarkdown(text) {
    if (window.marked) {
      // marked.parse can return a string. We use it raw — the assistant's
      // response is treated as semi-trusted Markdown (rendered server-side
      // from a closed wiki dataset).
      return marked.parse(text);
    }
    return "<pre>" + escapeHtml(text) + "</pre>";
  }

  function appendMessage(role, content) {
    // Hide the empty-state once a real message is shown.
    if (emptyStateEl && !emptyStateEl.hidden) {
      emptyStateEl.hidden = true;
    }
    const wrap = document.createElement("div");
    wrap.className = "message " + role;
    const bubble = document.createElement("div");
    bubble.className = "bubble " + role;
    if (role === "user") {
      bubble.textContent = content;  // text only for user input
    } else {
      bubble.innerHTML = renderMarkdown(content);
    }
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function appendTyping() {
    if (emptyStateEl && !emptyStateEl.hidden) {
      emptyStateEl.hidden = true;
    }
    const wrap = document.createElement("div");
    wrap.className = "message assistant";
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant typing";
    bubble.textContent = "Thinking";
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function clearMessages() {
    while (messagesEl.firstChild) {
      messagesEl.removeChild(messagesEl.firstChild);
    }
    if (emptyStateEl) {
      emptyStateEl.hidden = false;
      messagesEl.appendChild(emptyStateEl);
    }
  }

  async function postJson(url, body) {
    const opts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
    const resp = await fetch(url, opts);
    let data = {};
    try {
      data = await resp.json();
    } catch (e) {
      throw new Error("Server returned non-JSON response (" + resp.status + ")");
    }
    if (!resp.ok) {
      throw new Error(data.error || ("HTTP " + resp.status));
    }
    return data;
  }

  async function getJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return resp.json();
  }

  // ---- Render history on page load ----
  // When the user reloads the page, the conversation is restored
  // from localStorage and re-rendered. This is the "persistence
  // across reloads" behavior the spec asked for.
  function renderHistory() {
    if (conversationHistory.length === 0) {
      // Show the empty state
      if (emptyStateEl) {
        emptyStateEl.hidden = false;
        if (messagesEl.firstChild !== emptyStateEl) {
          messagesEl.appendChild(emptyStateEl);
        }
      }
      return;
    }
    // Hide the empty state
    if (emptyStateEl) {
      emptyStateEl.hidden = true;
    }
    for (const turn of conversationHistory) {
      appendMessage(turn.role, turn.content);
    }
  }

  // ---- Form submission ----

  async function sendQuestion(ev) {
    if (ev) ev.preventDefault();
    if (inFlight) return;
    const q = input.value.trim();
    if (!q) return;
    inFlight = true;
    sendBtn.disabled = true;

    // Render the user message immediately
    appendMessage("user", q);

    // Add to history (this is the in-memory list that the server
    // sees as prior_turns on the next request)
    conversationHistory.push({ role: "user", content: q });
    saveHistory();

    input.value = "";
    autoSize();

    const typing = appendTyping();
    try {
      // Send question + the conversation history. The server is
      // stateless; the prior_turns is the only context it sees.
      const data = await postJson("/api/chat", {
        question: q,
        prior_turns: conversationHistory.slice(0, -1),  // exclude current turn
      });
      typing.remove();
      appendMessage("assistant", data.answer);

      // Add the assistant turn to history
      conversationHistory.push({ role: "assistant", content: data.answer });
      saveHistory();
    } catch (e) {
      typing.remove();
      const errBubble = appendMessage("assistant", "");
      errBubble.classList.add("error");
      errBubble.textContent = "Error: " + e.message;
    } finally {
      inFlight = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function autoSize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  // ---- New Chat ----
  // Clears the local conversation history and the visible chat.
  // There's no server-side state to clear.
  function newChat() {
    if (inFlight) return;
    if (conversationHistory.length > 0
        && !confirm("Start a new conversation? This clears your current chat.")) {
      return;
    }
    conversationHistory = [];
    saveHistory();
    clearMessages();
    input.focus();
  }

  async function showHelp() {
    try {
      const data = await getJson("/api/help");
      helpList.innerHTML = "";
      for (const cmd of data.commands) {
        const li = document.createElement("li");
        li.innerHTML = "<code>/" + escapeHtml(cmd.name) + "</code> &mdash; " + escapeHtml(cmd.description);
        helpList.appendChild(li);
      }
      helpModal.classList.remove("hidden");
    } catch (e) {
      alert("Failed to load help: " + e.message);
    }
  }

  // ---- Wire up ----

  form.addEventListener("submit", sendQuestion);
  newChatBtn.addEventListener("click", newChat);
  helpBtn.addEventListener("click", showHelp);
  helpClose.addEventListener("click", () => helpModal.classList.add("hidden"));
  helpModal.addEventListener("click", (e) => {
    if (e.target === helpModal) helpModal.classList.add("hidden");
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });
  input.addEventListener("input", autoSize);

  if (emptyStateEl) {
    emptyStateEl.addEventListener("click", (e) => {
      const li = e.target.closest("li");
      if (!li) return;
      input.value = li.textContent;
      autoSize();
      sendQuestion();
    });
  }

  // Render any history from a previous session, then focus the input
  renderHistory();
  input.focus();
})();
