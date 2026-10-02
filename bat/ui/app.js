"use strict";

// ---------------------------------------------------------------- state
const store = {
  get key() { return localStorage.getItem("bat.key") || ""; },
  set key(v) { v ? localStorage.setItem("bat.key", v) : localStorage.removeItem("bat.key"); },
  get session() { return localStorage.getItem("bat.session") || ""; },
  set session(v) { v ? localStorage.setItem("bat.session", v) : localStorage.removeItem("bat.session"); },
};
let sessions = [];
let current = null;          // active session id, or null for an unsaved new chat
let streaming = null;        // AbortController while a reply is in flight

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// ---------------------------------------------------------------- api
class AuthError extends Error {}

async function api(method, path, body, signal) {
  const res = await fetch(path, {
    method, signal,
    headers: { Authorization: `Bearer ${store.key}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (res.status === 401) throw new AuthError("Your key was rejected.");
  if (!res.ok) throw new Error(await errorText(res));
  return res.status === 204 ? null : res;
}
const json = async (...a) => (await api(...a)).json();

async function errorText(res) {
  try {
    const p = await res.json();
    if (res.status === 429) return "BAT is busy with another reply. Try again in a moment.";
    return p.detail || p.title || res.statusText;
  } catch { return `${res.status} ${res.statusText}`; }
}

// SSE over fetch: EventSource can't POST or send an Authorization header.
async function* readEvents(res) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      let event = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

// ---------------------------------------------------------------- markdown
// Escape first, then add a small allowlist of formatting. Nothing the model
// writes can become live HTML, so a prompt-injected page can't inject script.
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inline(s) {
  return s
    // Small models emit bits of LaTeX; render the common ones as plain text.
    // ponytail: no real math rendering; add KaTeX if answers become formula-heavy.
    .replace(/\\[()[\]]/g, "")
    .replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\approx/g, "≈")
    .replace(/\^\{([^}]+)\}/g, "<sup>$1</sup>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
}

function markdown(src) {
  const out = [];
  const parts = src.split(/^```[^\n]*\n?/m);   // even = prose, odd = code
  parts.forEach((part, i) => {
    if (i % 2) {
      out.push(`<pre><button class="copy" type="button">Copy</button><code>${esc(part.replace(/\n$/, ""))}</code></pre>`);
      return;
    }
    let list = null;
    const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
    let para = [];
    const flush = () => { if (para.length) { out.push(`<p>${inline(esc(para.join("\n"))).replace(/\n/g, "<br>")}</p>`); para = []; } };
    for (const line of part.split("\n")) {
      let m;
      if ((m = line.match(/^(#{1,3})\s+(.*)/))) { flush(); close(); out.push(`<h${m[1].length}>${inline(esc(m[2]))}</h${m[1].length}>`); }
      else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { flush(); if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(esc(m[1]))}</li>`); }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { flush(); if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(esc(m[1]))}</li>`); }
      else if (!line.trim()) { flush(); close(); }
      else { close(); para.push(line); }
    }
    flush(); close();
  });
  return out.join("");
}

// ---------------------------------------------------------------- ui helpers
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 2600);
}

function setStatus(kind, text) {
  $("status").className = `status ${kind}`;
  $("statusText").textContent = text;
}

function scrollDown(force) {
  const t = $("thread");
  if (force || t.scrollHeight - t.scrollTop - t.clientHeight < 140) t.scrollTop = t.scrollHeight;
}

function autosize() {
  const i = $("input");
  i.style.height = "auto";
  i.style.height = Math.min(i.scrollHeight, 200) + "px";
  $("send").disabled = !i.value.trim() || !!streaming;
}

function showEmpty(on) { $("empty").hidden = !on; }

// ---------------------------------------------------------------- messages
function userMessage(text) {
  const m = el("div", "msg user");
  m.appendChild(el("div", "body", text));
  $("messages").appendChild(m);
  return m;
}

function assistantMessage() {
  const m = el("div", "msg assistant");
  const mark = el("span", "logo sm who-mark", "B");
  const body = el("div", "body");
  const tools = el("div", "tools");
  const content = el("div", "content");
  content.innerHTML = '<span class="thinking"><i></i><i></i><i></i></span>';
  body.append(tools, content);
  m.append(mark, body);
  $("messages").appendChild(m);

  let text = "", frame = 0;
  const paint = (final) => {
    content.innerHTML = markdown(text) || "";
    content.classList.toggle("caret", !final);
  };
  return {
    node: m,
    append(delta) {
      text += delta;
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; paint(false); scrollDown(); });
    },
    finish(finalText) {
      if (!text && finalText) text = finalText;
      cancelAnimationFrame(frame);
      paint(true);
      if (!text) content.innerHTML = '<p class="muted">No reply.</p>';
      const meta = el("div", "meta-line");
      const copy = el("button", null, "Copy");
      copy.type = "button";
      copy.onclick = () => navigator.clipboard.writeText(text).then(() => toast("Copied"));
      meta.appendChild(copy);
      body.appendChild(meta);
    },
    fail(message) {
      cancelAnimationFrame(frame);
      m.classList.add("error");
      content.classList.remove("caret");
      content.innerHTML = (text ? markdown(text) : "") + `<p>⚠ ${esc(message)}</p>`;
    },
    tool: () => tools,
  };
}

const TOOL_LABELS = {
  web_search: (a) => `Searched the web for “${a.query ?? ""}”`,
  memory_search: (a) => `Searched memory for “${a.query ?? ""}”`,
  calculator: (a) => `Calculated ${a.expression ?? ""}`,
  python_exec: () => "Ran Python",
};

function toolCard(container, name, args) {
  const d = el("details", "tool");
  const s = el("summary");
  const spin = el("span", "spin");
  const label = el("span", "label", (TOOL_LABELS[name] || (() => `Used ${name}`))(args || {}));
  s.append(spin, label, el("span", "chev", "›"));
  d.appendChild(s);
  container.appendChild(d);
  return {
    done(content, isError) {
      spin.replaceWith(el("span", null, isError ? "⚠" : "✓"));
      if (isError) d.classList.add("err");
      d.appendChild(el("pre", "out", content));
    },
  };
}

// ---------------------------------------------------------------- sessions
function groupLabel(iso) {
  const day = 864e5, d = new Date(iso), now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (d >= start) return "Today";
  if (d >= start - day) return "Yesterday";
  if (d >= start - 7 * day) return "Previous 7 days";
  return "Older";
}

function renderSessions() {
  const nav = $("sessions");
  nav.replaceChildren();
  if (!sessions.length) { nav.appendChild(el("div", "none", "No chats yet.")); return; }
  let last = "";
  for (const s of sessions) {
    const g = groupLabel(s.updated_at);
    if (g !== last) { nav.appendChild(el("h4", null, g)); last = g; }
    const row = el("button", "session" + (s.id === current ? " active" : ""));
    row.type = "button";
    row.appendChild(el("span", "name", s.title || "Untitled"));
    const del = el("span", "icon-btn del", "🗑");
    del.title = "Delete chat";
    del.onclick = (e) => { e.stopPropagation(); removeSession(s); };
    row.appendChild(del);
    row.onclick = () => openSession(s.id);
    nav.appendChild(row);
  }
}

async function loadSessions() {
  const page = await json("GET", "/v1/sessions?limit=100");
  sessions = page.items.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  renderSessions();
}

function newChat() {
  if (streaming) return;
  current = null;
  store.session = "";
  $("messages").replaceChildren();
  $("chatTitle").textContent = "New chat";
  showEmpty(true);
  renderSessions();
  closeMenu();
  $("input").focus();
}

async function openSession(id) {
  if (streaming) return;
  current = id;
  store.session = id;
  renderSessions();
  closeMenu();
  const s = sessions.find((x) => x.id === id);
  $("chatTitle").textContent = s?.title || "Chat";
  $("messages").replaceChildren();
  showEmpty(false);
  try {
    const history = await json("GET", `/v1/sessions/${id}/messages?limit=200`);
    if (!history.items.length) showEmpty(true);
    for (const m of history.items) {
      if (m.role === "user") userMessage(m.content);
      else if (m.role === "assistant") {
        const a = assistantMessage();
        for (const t of m.tools || []) toolCard(a.tool(), t.name, t.arguments).done(t.result ?? "", t.is_error);
        a.finish(m.content);
      }
    }
    scrollDown(true);
  } catch (e) { handle(e); }
}

async function removeSession(s) {
  if (!confirm(`Delete “${s.title || "Untitled"}”? This can't be undone.`)) return;
  try {
    await api("DELETE", `/v1/sessions/${s.id}`);
    sessions = sessions.filter((x) => x.id !== s.id);
    if (s.id === current) newChat(); else renderSessions();
    toast("Chat deleted");
  } catch (e) { handle(e); }
}

// ---------------------------------------------------------------- send
async function send(text) {
  if (streaming || !text.trim()) return;
  showEmpty(false);
  streaming = new AbortController();
  $("send").hidden = true;
  $("stop").hidden = false;
  autosize();
  setStatus("busy", "thinking");

  userMessage(text);
  const reply = assistantMessage();
  scrollDown(true);

  try {
    if (!current) {
      // Created on first message, not on page load, so empty chats never pile up.
      const title = text.replace(/\s+/g, " ").trim().slice(0, 60);
      const s = await json("POST", "/v1/sessions", { title });
      current = s.id;
      store.session = s.id;
      sessions.unshift(s);
      $("chatTitle").textContent = s.title;
      renderSessions();
    }

    const res = await api("POST", `/v1/sessions/${current}/messages/stream`, { content: text }, streaming.signal);
    let card = null, finished = false;
    for await (const { event, data } of readEvents(res)) {
      if (event === "token") reply.append(data.text);
      else if (event === "tool_call") { setStatus("busy", "using tools"); card = toolCard(reply.tool(), data.name, data.arguments); }
      else if (event === "tool_result") { card?.done(data.content, data.is_error); setStatus("busy", "writing"); }
      else if (event === "final") { reply.finish(data.content); finished = true; }
      else if (event === "error") { reply.fail(data.message); finished = true; }
      scrollDown();
    }
    if (!finished) reply.finish("");
    setStatus("ok", "ready");
  } catch (e) {
    if (e.name === "AbortError") { reply.finish(""); setStatus("ok", "stopped"); }
    else { reply.fail(e.message); setStatus("bad", "error"); if (e instanceof AuthError) signOut(e.message); }
  } finally {
    streaming = null;
    $("send").hidden = false;
    $("stop").hidden = true;
    autosize();
    loadSessions().catch(() => {});
  }
}

// ---------------------------------------------------------------- memory
async function addMemory(text, source) {
  const res = await json("POST", "/v1/memory/documents", { text, kind: "document", source: source || null });
  return res.chunks;
}

function memStatus(text, kind) {
  const s = $("memStatus");
  s.textContent = text;
  s.className = `mem-status ${kind || ""}`;
  s.hidden = !text;
}

async function addFiles(files) {
  let total = 0, names = [];
  for (const f of files) {
    if (f.size > 500_000) { memStatus(`${f.name} is too large (max ~500 KB of text).`, "bad"); return; }
    memStatus(`Indexing ${f.name}…`);
    const text = await f.text();
    if (!text.trim()) continue;
    total += await addMemory(text, f.name);
    names.push(f.name);
  }
  memStatus(names.length ? `Added ${names.join(", ")} (${total} chunk${total === 1 ? "" : "s"}).` : "Nothing to add.", names.length ? "ok" : "bad");
}

// ---------------------------------------------------------------- auth
async function signIn(key) {
  store.key = key;
  const me = await json("GET", "/v1/whoami");
  $("who").textContent = me.display_name || `${me.principal_id} · ${me.tenant_id}`;
  $("avatar").textContent = (me.principal_id[0] || "?").toUpperCase();
  $("login").hidden = true;
  $("app").hidden = false;
  setStatus("ok", "ready");
  await loadSessions();
  if (store.session && sessions.some((s) => s.id === store.session)) await openSession(store.session);
  else newChat();
}

function signOut(reason) {
  store.key = "";
  store.session = "";
  sessions = [];
  current = null;
  $("app").hidden = true;
  $("login").hidden = false;
  $("keyInput").value = "";
  const err = $("loginError");
  err.textContent = reason || "";
  err.hidden = !reason;
  $("keyInput").focus();
}

function handle(e) {
  if (e instanceof AuthError) signOut(e.message);
  else toast(e.message);
}

// ---------------------------------------------------------------- menu (mobile)
const closeMenu = () => $("app").classList.remove("menu");

// ---------------------------------------------------------------- wire up
$("loginForm").onsubmit = async (e) => {
  e.preventDefault();
  const btn = $("loginBtn");
  btn.disabled = true;
  btn.textContent = "Checking…";
  try { await signIn($("keyInput").value.trim()); }
  catch (err) {
    store.key = "";
    $("loginError").textContent = err instanceof AuthError ? "That key isn't valid." : `Can't reach BAT: ${err.message}`;
    $("loginError").hidden = false;
  } finally { btn.disabled = false; btn.textContent = "Continue"; }
};

$("composer").onsubmit = (e) => {
  e.preventDefault();
  const text = $("input").value;
  $("input").value = "";
  send(text);
};
$("input").addEventListener("input", autosize);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("stop").onclick = () => streaming?.abort();
$("newChat").onclick = newChat;
$("signOut").onclick = () => signOut();
$("openSidebar").onclick = () => $("app").classList.add("menu");
$("closeSidebar").onclick = closeMenu;
$("scrim").onclick = closeMenu;

document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => send(c.dataset.prompt)));

$("messages").addEventListener("click", (e) => {
  if (!e.target.matches("pre .copy")) return;
  navigator.clipboard.writeText(e.target.nextElementSibling.textContent).then(() => toast("Code copied"));
});

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
  if (e.key === "Escape" && streaming) streaming.abort();
});

// memory dialog
$("openMemory").onclick = () => { memStatus(""); $("memory").showModal(); closeMenu(); };
$("file").onchange = (e) => addFiles([...e.target.files]).catch((err) => memStatus(err.message, "bad")).finally(() => (e.target.value = ""));
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  addFiles([...e.dataTransfer.files]).catch((err) => memStatus(err.message, "bad"));
};
$("saveMemory").onclick = async () => {
  const text = $("memText").value.trim();
  if (!text) return memStatus("Paste some text first.", "bad");
  memStatus("Indexing…");
  try {
    const n = await addMemory(text, $("memSource").value.trim());
    $("memText").value = "";
    $("memSource").value = "";
    memStatus(`Added to memory (${n} chunk${n === 1 ? "" : "s"}).`, "ok");
  } catch (e) { memStatus(e.message, "bad"); }
};
$("forget").onclick = async () => {
  if (!confirm("Erase everything in your memory? Chats are kept.")) return;
  try { await api("DELETE", "/v1/memory"); memStatus("Memory erased.", "ok"); }
  catch (e) { memStatus(e.message, "bad"); }
};

// boot
(async () => {
  if (!store.key) return signOut();
  try { await signIn(store.key); }
  catch (e) { signOut(e instanceof AuthError ? "Your saved key no longer works." : `Can't reach BAT: ${e.message}`); }
})();
