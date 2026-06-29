/* Frontend for the Hello! Project Wiki chatbot.
 *
 * - POSTs user questions to /api/chat, renders Markdown responses
 * - "New Chat" button calls /api/reset (the /new slash command) and
 *   clears the visible chat window
 * - "?" button shows the help modal listing available commands
 * - Configures marked.js to safely render assistant Markdown
 *
 * Note: per the user's spec, the web UI hides historical chats by
 * default. The empty-state at the start is the only "default" view.
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

  // ---- State ----
  let inFlight = false;

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
    // Remove all message children (but keep the empty-state element)
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
      credentials: "same-origin",  // send the session cookie
      body: JSON.stringify(body),
    };
    const resp = await fetch(url, opts);
    let data = {};
    try {
      data = await resp.json();
    } catch (e) {
      // Non-JSON response (e.g., 4xx/5xx HTML). Surface as error.
      throw new Error("Server returned non-JSON response (" + resp.status + ")");
    }
    if (!resp.ok) {
      throw new Error(data.error || ("HTTP " + resp.status));
    }
    return data;
  }

  async function getJson(url) {
    const resp = await fetch(url, { credentials: "same-origin" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return resp.json();
  }

  // ---- Form submission ----

  async function sendQuestion(ev) {
    if (ev) ev.preventDefault();
    if (inFlight) return;
    const q = input.value.trim();
    if (!q) return;
    inFlight = true;
    sendBtn.disabled = true;
    appendMessage("user", q);
    input.value = "";
    autoSize();
    const typing = appendTyping();
    try {
      const data = await postJson("/api/chat", { question: q });
      typing.remove();
      appendMessage("assistant", data.answer);
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
    // Auto-grow the textarea up to its max-height
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  // ---- Slash-command emulation (for parity with REPL) ----
  // "yes" / "no" / "thanks" are sent as plain text — the LLM will see
  // the prior turns in context and respond appropriately. The /new
  // command is bound to the New Chat button.

  async function newChat() {
    if (inFlight) return;
    if (!confirm("Start a new conversation? This clears the current chat.")) {
      return;
    }
    try {
      await postJson("/api/reset", {});
      clearMessages();
      input.focus();
    } catch (e) {
      alert("Failed to reset: " + e.message);
    }
  }

  async function showHelp() {
    try {
      const data = await getJson("/api/help");
      helpList.innerHTML = "";
      for (const cmd of data.commands) {
        const li = document.createElement("li");
        li.innerHTML = "<code>/" + escapeHtml(cmd.name) + "</code> — " + escapeHtml(cmd.description);
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

  // Keyboard: Enter sends, Shift+Enter newlines
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });
  input.addEventListener("input", autoSize);

  // Click on a suggested question in the empty-state
  if (emptyStateEl) {
    emptyStateEl.addEventListener("click", (e) => {
      const li = e.target.closest("li");
      if (!li) return;
      input.value = li.textContent;
      autoSize();
      sendQuestion();
    });
  }

  // Focus the input on load
  input.focus();
})();
