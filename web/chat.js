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
 * - Wraps entity names (artists, albums, songs, etc.) in clickable
 *   links that send a "Tell me about X" prompt
 * - Shows a "Sources" section at the bottom of assistant responses
 *   with links to the Fandom wiki
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
  // conversationHistory is a list of {role, content, entities?, sources?}
  // objects. entities and sources are only on assistant turns.
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
      return marked.parse(text);
    }
    return "<pre>" + escapeHtml(text) + "</pre>";
  }

  // ---- Entity linking ----
  // Walk all text nodes inside the assistant bubble. For each text node,
  // replace plain-text occurrences of entity names with <a class="entity-link">
  // tags. We use TreeWalker so we never touch attribute values or tags.

  function linkifyEntities(rootEl, entities) {
    if (!entities || entities.length === 0) return;

    // Build a regex from entity names. Longer names first so we don't
    // match "Morning" before "Morning Musume". Escape regex specials.
    const sorted = entities
      .filter((e) => e && e.name && e.name.length >= 2)
      .slice()
      .sort((a, b) => b.name.length - a.name.length);
    if (sorted.length === 0) return;

    // Build a map of normalized-name -> entity for quick lookup
    const entityByName = new Map();
    for (const e of sorted) {
      entityByName.set(e.name.toLowerCase(), e);
    }
    // Pattern: any entity name, word-boundary aware (so we don't link
    // "Morning" inside "Mornings"). Case-sensitive matching.
    const pattern = new RegExp(
      "\\b(" + sorted.map((e) => escapeRegex(e.name)).join("|") + ")\\b",
      "g"
    );

    // Collect text nodes to modify. We can't mutate the tree while walking,
    // so we collect targets first, then replace.
    const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        // Skip text inside <a>, <code>, <pre>, <script>, <style>
        let p = node.parentNode;
        while (p && p !== rootEl) {
          const tag = p.tagName && p.tagName.toLowerCase();
          if (tag === "a" || tag === "code" || tag === "pre"
              || tag === "script" || tag === "style") {
            return NodeFilter.FILTER_REJECT;
          }
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const textNodes = [];
    let n;
    while ((n = walker.nextNode())) {
      // Only process nodes that actually contain an entity name
      if (pattern.test(n.nodeValue)) {
        textNodes.push(n);
        pattern.lastIndex = 0;  // reset after test()
      }
    }

    for (const node of textNodes) {
      const text = node.nodeValue;
      const frag = document.createDocumentFragment();
      let lastIndex = 0;
      pattern.lastIndex = 0;
      let m;
      while ((m = pattern.exec(text)) !== null) {
        const before = text.slice(lastIndex, m.index);
        if (before) frag.appendChild(document.createTextNode(before));
        const matchedName = m[1];
        const entity = entityByName.get(matchedName.toLowerCase());
        const a = document.createElement("a");
        a.className = "entity-link entity-" + (entity ? entity.type : "other");
        a.href = "#";
        a.textContent = matchedName;
        a.dataset.entityName = matchedName;
        if (entity) a.dataset.entityType = entity.type;
        frag.appendChild(a);
        lastIndex = m.index + matchedName.length;
      }
      const after = text.slice(lastIndex);
      if (after) frag.appendChild(document.createTextNode(after));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // ---- Sources section ----

  function appendSourcesSection(wrap, sources) {
    if (!sources || sources.length === 0) return;
    const sec = document.createElement("div");
    sec.className = "sources";
    const heading = document.createElement("div");
    heading.className = "sources-heading";
    heading.textContent = "Sources";
    sec.appendChild(heading);
    const list = document.createElement("ul");
    list.className = "sources-list";
    for (const s of sources) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = s.url || "#";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = s.title || s.url || "Source";
      li.appendChild(a);
      list.appendChild(li);
    }
    sec.appendChild(list);
    wrap.appendChild(sec);
  }

  // ---- Render ----

  function appendMessage(role, content, opts) {
    opts = opts || {};
    if (emptyStateEl && !emptyStateEl.hidden) {
      emptyStateEl.hidden = true;
    }
    const wrap = document.createElement("div");
    wrap.className = "message " + role;
    const bubble = document.createElement("div");
    bubble.className = "bubble " + role;
    if (role === "user") {
      bubble.textContent = content;
    } else {
      // Render markdown first, then linkify entities inside the rendered HTML.
      // We set innerHTML (not textContent) here because marked.js output is
      // trusted: it comes from the LLM which we treat as semi-trusted.
      bubble.innerHTML = renderMarkdown(content);
      // Walk text nodes and wrap entity names in <a> tags.
      if (opts.entities && opts.entities.length > 0) {
        linkifyEntities(bubble, opts.entities);
      }
    }
    wrap.appendChild(bubble);
    // Sources go below the bubble but inside the message wrapper, so the
    // styling is per-message.
    if (role === "assistant" && opts.sources && opts.sources.length > 0) {
      appendSourcesSection(wrap, opts.sources);
    }
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

  function renderHistory() {
    if (conversationHistory.length === 0) {
      if (emptyStateEl) {
        emptyStateEl.hidden = false;
        if (messagesEl.firstChild !== emptyStateEl) {
          messagesEl.appendChild(emptyStateEl);
        }
      }
      return;
    }
    if (emptyStateEl) {
      emptyStateEl.hidden = true;
    }
    for (const turn of conversationHistory) {
      if (turn.role === "user") {
        appendMessage("user", turn.content);
      } else {
        appendMessage("assistant", turn.content, {
          entities: turn.entities || [],
          sources: turn.sources || [],
        });
      }
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

    appendMessage("user", q);
    conversationHistory.push({ role: "user", content: q });
    saveHistory();

    input.value = "";
    autoSize();

    const typing = appendTyping();
    try {
      const data = await postJson("/api/chat", {
        question: q,
        prior_turns: conversationHistory.slice(0, -1),
      });
      typing.remove();
      appendMessage("assistant", data.answer || "", {
        entities: data.entities || [],
        sources: data.sources || [],
      });

      conversationHistory.push({
        role: "assistant",
        content: data.answer || "",
        entities: data.entities || [],
        sources: data.sources || [],
      });
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

  // ---- "Tell me about X" prompts ----
  // When the user clicks an entity link, send a pre-filled prompt.
  // We use preventDefault to stop the browser from following the href.
  function askAbout(name) {
    if (inFlight) return;
    input.value = "Tell me about " + name;
    autoSize();
    sendQuestion();
  }

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

  // Delegated click handler on the messages container. If a click lands
  // on an .entity-link, intercept and ask about it.
  messagesEl.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("a.entity-link");
    if (a && a.dataset && a.dataset.entityName) {
      e.preventDefault();
      askAbout(a.dataset.entityName);
    }
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

  renderHistory();
  input.focus();
})();