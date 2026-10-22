/* ═══════════════════════════════════════════════════════════════
   Cortex — front-end
   ═══════════════════════════════════════════════════════════════ */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  health: null,
  settings: { org: "your organisation", model: "claude-sonnet-5", strict: true, show_sources: true, min_relevance: 0.18 },
  docs: [],
  conversations: [],
  convId: null,
  messages: [],
  streaming: false,
  abort: null,
  debug: false,
  lastDiag: null,
  lastQuery: "",
  theme: "system",
  activeTab: "docs",
  popSource: null,
  docScope: null,          // null = all documents, else array of doc ids
  drafts: {},              // scope selection while the menu is open
};

const el = {
  messages: $("#messages"),
  empty: $("#empty"),
  starterGrid: $("#starterGrid"),
  scroll: $("#scrollArea"),
  input: $("#input"),
  composer: $("#composer"),
  sendBtn: $("#sendBtn"),
  stopBtn: $("#stopBtn"),
  chatList: $("#chatList"),
  chatTitle: $("#chatTitle"),
  statusLine: $("#statusLine"),
  banner: $("#banner"),
  bannerText: $("#bannerText"),
  bannerAction: $("#bannerAction"),
  kbDocs: $("#kbDocs"),
  kbMeta: $("#kbMeta"),
  docCount: $("#docCount"),
  modeBadge: $("#modeBadge"),
  drawer: $("#drawer"),
  scrim: $("#scrim"),
  docList: $("#docList"),
  docListMeta: $("#docListMeta"),
  dropzone: $("#dropzone"),
  fileInput: $("#fileInput"),
  queue: $("#uploadQueue"),
  settingsModal: $("#settingsModal"),
  popover: $("#popover"),
  toasts: $("#toasts"),
  strictChip: $("#strictChip"),
  strictLabel: $("#strictLabel"),
};

/* ───────────────────────────── helpers ─────────────────────────── */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (n) => n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : n > 1024 ? Math.round(n / 1024) + " KB" : n + " B";
const fmtNum = (n) => (n ?? 0).toLocaleString("en-US");
function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  if (s < 604800) return Math.floor(s / 86400) + "d ago";
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
const clock = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  el.toasts.appendChild(t);
  setTimeout(() => { t.classList.add("leaving"); setTimeout(() => t.remove(), 220); }, kind === "err" ? 5200 : 3000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error((data && (data.detail || data.message)) || `Request failed (${res.status})`);
  return data;
}

/* ─────────────────────────── markdown ──────────────────────────── */
function inlineFmt(t) {
  const codes = [];
  t = t.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0001${codes.length - 1}\u0001`; });
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
  t = t.replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  t = t.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[+i]}</code>`);
  return t;
}

function md(src) {
  const lines = esc(src || "").split("\n");
  const out = [];
  let i = 0;
  const isListItem = (l) => /^\s{0,3}([-*+]|\d+[.)])\s+/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    if (/^```/.test(line.trim())) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) buf.push(lines[i++]);
      i++;
      out.push(`<pre><code>${buf.join("\n")}</code></pre>`);
      continue;
    }
    if (/^#{1,6}\s+/.test(line)) {
      const m = line.match(/^(#{1,6})\s+(.*)$/);
      const tag = m[1].length <= 2 ? "h2" : m[1].length === 3 ? "h3" : "h4";
      out.push(`<${tag}>${inlineFmt(m[2])}</${tag}>`);
      i++; continue;
    }
    if (/^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { out.push("<hr>"); i++; continue; }

    if (/^\s{0,3}&gt;\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s{0,3}&gt;\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s{0,3}&gt;\s?/, "")); i++;
      }
      out.push(`<blockquote>${inlineFmt(buf.join("<br>"))}</blockquote>`);
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      const cells = (l) => l.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
      const head = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      out.push(
        "<table><thead><tr>" + head.map((c) => `<th>${inlineFmt(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${inlineFmt(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>"
      );
      continue;
    }

    if (isListItem(line)) {
      const ordered = /^\s{0,3}\d+[.)]\s+/.test(line);
      const items = [];
      while (i < lines.length && (isListItem(lines[i]) || (lines[i].startsWith("  ") && lines[i].trim()))) {
        const l = lines[i];
        if (isListItem(l)) items.push(l.replace(/^\s{0,3}([-*+]|\d+[.)])\s+/, ""));
        else if (items.length) items[items.length - 1] += " " + l.trim();
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((it) => `<li>${inlineFmt(it)}</li>`).join("")}</${tag}>`);
      continue;
    }

    const buf = [];
    while (i < lines.length && lines[i].trim() && !/^#{1,6}\s/.test(lines[i]) &&
           !isListItem(lines[i]) && !/^```/.test(lines[i].trim()) && !/^\s{0,3}&gt;/.test(lines[i])) {
      buf.push(lines[i]); i++;
    }
    out.push(`<p>${inlineFmt(buf.join("<br>"))}</p>`);
  }
  let html = out.join("");
  html = html.replace(/\[(\d{1,2})\]/g, (m, n) => `<button class="cite" data-n="${n}" title="View source ${n}">${n}</button>`);
  return html;
}

/* ─────────────────────────── messaging ─────────────────────────── */
function nearBottom() {
  const s = el.scroll;
  return s.scrollHeight - s.scrollTop - s.clientHeight < 140;
}
function scrollToBottom(force = false) {
  if (force || nearBottom()) {
    requestAnimationFrame(() => { el.scroll.scrollTop = el.scroll.scrollHeight; });
  }
}
function toggleEmpty(show) { el.empty.classList.toggle("hidden", !show); }

function msgShell(role) {
  const wrap = document.createElement("article");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${role === "assistant"
      ? `<svg viewBox="0 0 64 64"><path d="M32 13a19 19 0 1 0 13.4 5.6" fill="none" stroke="currentColor" stroke-width="6"/><circle cx="32" cy="32" r="5" fill="currentColor" stroke="none"/></svg>`
      : "You"}</div>
    <div class="msg-body">
      <div class="msg-head">
        <span class="msg-who">${role === "assistant" ? "Cortex" : "You"}</span>
        <span class="msg-time">${clock()}</span>
        <div class="msg-tools"></div>
      </div>
      <div class="msg-content"></div>
    </div>`;
  return wrap;
}

function addUserMessage(text) {
  const wrap = msgShell("user");
  wrap.querySelector(".msg-content").textContent = text;
  el.messages.appendChild(wrap);
  state.messages.push({ role: "user", content: text });
  toggleEmpty(false);
  scrollToBottom(true);
  return wrap;
}

function addAssistantMessage() {
  const wrap = msgShell("assistant");
  const content = wrap.querySelector(".msg-content");
  content.innerHTML = `<div class="thinking"><span class="dots"><i></i><i></i><i></i></span><span class="think-label">Thinking…</span></div>`;
  wrap.classList.add("streaming");
  wrap._buffer = "";
  wrap._sources = [];
  wrap._meta = {};
  el.messages.appendChild(wrap);
  scrollToBottom(true);
  return wrap;
}

function setThinking(wrap, label) {
  const t = wrap.querySelector(".think-label");
  if (t) t.textContent = label;
}

let renderTimer = null;
function streamRender(wrap) {
  if (renderTimer) return;
  renderTimer = setTimeout(() => {
    renderTimer = null;
    const html = md(wrap._buffer);
    const content = wrap.querySelector(".msg-content");
    content.innerHTML = html + `<span class="caret-slot"></span>`;
    scrollToBottom();
  }, 55);
}

function finalizeAssistant(wrap, { stopped = false } = {}) {
  if (wrap._finalized) return;          // `done` and the stream close both fire
  wrap._finalized = true;
  const content = wrap.querySelector(".msg-content");
  content.innerHTML = md(wrap._buffer) || (stopped ? "<p><em>Stopped.</em></p>" : "");
  wrap.classList.remove("streaming");
  finishActivity(wrap);
  if (stopped && wrap._buffer) content.insertAdjacentHTML("beforeend", `<p class="muted-sm"><em>Stopped by you.</em></p>`);
  renderSources(wrap);
  renderMeta(wrap);
  renderGrounding(wrap);
  renderWarning(wrap);
  addTools(wrap);
  state.messages.push({ role: "assistant", content: wrap._buffer, sources: wrap._sources, meta: wrap._meta });
  scrollToBottom();
}

function addTools(wrap) {
  const tools = wrap.querySelector(".msg-tools");
  tools.innerHTML = `
    <button class="icon-btn" data-act="up" title="Good answer" aria-label="Good answer">
      <svg viewBox="0 0 24 24"><path d="M7 11v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3z"/><path d="M7 11l4-7a2 2 0 0 1 3.6 1.7L14 9h5.2a2 2 0 0 1 2 2.4l-1.3 6.6a2 2 0 0 1-2 1.6H7"/></svg>
    </button>
    <button class="icon-btn" data-act="down" title="Bad answer" aria-label="Bad answer">
      <svg viewBox="0 0 24 24"><path d="M17 13V4h3a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-3z"/><path d="M17 13l-4 7a2 2 0 0 1-3.6-1.7L10 15H4.8a2 2 0 0 1-2-2.4l1.3-6.6a2 2 0 0 1 2-1.6H17"/></svg>
    </button>
    <button class="icon-btn" data-act="copy" title="Copy answer" aria-label="Copy answer">
      <svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
    </button>
    <button class="icon-btn" data-act="regen" title="Regenerate" aria-label="Regenerate">
      <svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"/></svg>
    </button>`;
  tools.onclick = (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const act = btn.dataset.act;
    if (act === "copy") {
      navigator.clipboard.writeText(wrap._buffer).then(() => toast("Answer copied"), () => toast("Copy failed", "err"));
    } else if (act === "regen") {
      regenerate();
    } else if (act === "up" || act === "down") {
      const id = wrap._meta?.message_id;
      if (!id) return toast("Save finished answers before rating", "err");
      const rating = act === "up" ? 1 : -1;
      api(`/api/messages/${id}/feedback`, { method: "POST", body: JSON.stringify({ rating }) })
        .then(() => {
          wrap.querySelector('[data-act="up"]').classList.toggle("active-up", rating === 1);
          wrap.querySelector('[data-act="down"]').classList.toggle("active-down", rating === -1);
          toast(rating === 1 ? "Thanks — noted as helpful" : "Thanks — we'll use this to improve");
        })
        .catch((err) => toast(err.message, "err"));
    }
  };
}

function renderSources(wrap) {
  const existing = wrap.querySelector(".sources");
  if (existing) existing.remove();
  const sources = wrap._sources || [];
  if (!sources.length || !state.settings.show_sources) return;
  // an answer built only from live tools cites no passages - don't imply it did
  const answeredText = wrap.querySelector(".msg-content")?.textContent || "";
  if (wrap._toolCallsUsed && !/\[\d+\]/.test(answeredText)) return;
  const box = document.createElement("div");
  box.className = "sources";
  const chips = sources.map((s, i) => {
    const pg = (s.ext === ".pdf" && s.page) ? `p.${s.page}` : "";
    const web = s.kind === "web" || s.url;
    return `<button class="src-chip${web ? " web" : ""}" data-idx="${i}">
      <span class="n">${i + 1}</span>
      ${web ? `<svg class="web-ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 3 2.5 15 0 18"/></svg>` : ""}
      <span class="nm">${esc(s.doc_name)}</span>
      ${pg ? `<span class="pg">${pg}</span>` : ""}
    </button>`;
  }).join("");
  const openByDefault = sources.length <= 3;
  box.innerHTML = `
    <button class="src-head${openByDefault ? " open" : ""}" data-toggle="sources">
      <svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>
      ${sources.length} source${sources.length > 1 ? "s" : ""}
    </button>
    <div class="src-list"${openByDefault ? "" : ' style="display:none"'}></div>`;
  box.querySelector(".src-list").innerHTML = chips;
  box.querySelector("[data-toggle]").onclick = (e) => {
    const btn = e.currentTarget;
    btn.classList.toggle("open");
    box.querySelector(".src-list").style.display = btn.classList.contains("open") ? "" : "none";
  };
  box.addEventListener("click", (e) => {
    const chip = e.target.closest(".src-chip");
    if (!chip) return;
    const idx = +chip.dataset.idx;
    openSource(wrap, idx, chip.getBoundingClientRect());
  });
  wrap.querySelector(".msg-body").appendChild(box);
}

const TOOL_ICONS = {
  web_search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  weather: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  calculator: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 7h8M8 12h2M14 12h2M8 16h2M14 16h2"/>',
  current_datetime: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  wikipedia: '<path d="M4 5h7l4 9 4-9h3"/><path d="M4 19h16"/>',
  fetch_url: '<path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/>',
};

const ACTIVITY_VERBS = {
  web_search: "Searched the web",
  deep_research: "Researched the web",
  fetch_url: "Read a page",
  wikipedia: "Checked Wikipedia",
  weather: "Checked the weather",
  calculator: "Did the maths",
  current_datetime: "Checked the time",
  write_file: "Wrote a file",
  read_file: "Read a file",
  list_files: "Listed files",
  run_code: "Ran code",
  generate_image: "Made an image",
  text_to_speech: "Made audio",
  generate_video: "Made a video",
};

function activityBox(wrap) {
  let box = wrap.querySelector(".activity");
  if (box) return box;
  box = document.createElement("div");
  box.className = "activity";
  box.innerHTML = `
    <button type="button" class="activity-head" aria-expanded="false">
      <span class="a-ico"><span class="a-spin"></span></span>
      <span class="a-label">Working…</span>
      <svg class="a-chev" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>
    </button>
    <div class="activity-body"></div>`;
  box.querySelector(".activity-head").onclick = () => {
    const open = box.classList.toggle("open");
    box.querySelector(".activity-head").setAttribute("aria-expanded", String(open));
  };
  wrap._verbs = new Set();
  const content = wrap.querySelector(".msg-content");
  wrap.querySelector(".msg-body").insertBefore(box, content);
  return box;
}

function refreshActivityHeader(wrap) {
  const box = wrap.querySelector(".activity");
  if (!box) return;
  const verbs = [...(wrap._verbs || [])];
  const label = box.querySelector(".a-label");
  if (verbs.length === 0) label.textContent = "Working…";
  else if (verbs.length === 1) label.textContent = verbs[0];
  else label.textContent = `${verbs[0]} · ${wrap._steps} step${wrap._steps > 1 ? "s" : ""}`;
}

function toolCard(wrap, name, label, argsText) {
  const box = activityBox(wrap);
  wrap._steps = (wrap._steps || 0) + 1;
  const verb = ACTIVITY_VERBS[name] || `Used ${label || name}`;
  wrap._verbs.add(verb);
  const step = document.createElement("div");
  step.className = "step running";
  step.dataset.tool = name;
  step.innerHTML = `
    <button class="s-head" type="button">
      <span class="s-dot"></span>
      <span class="s-name">${esc(label || name)}</span>
      <span class="s-arg">${esc(argsText || "")}</span>
    </button>
    <div class="s-body"></div>`;
  step.querySelector(".s-head").onclick = () => step.classList.toggle("open");
  box.querySelector(".activity-body").appendChild(step);
  refreshActivityHeader(wrap);
  return step;
}

function describeArgs(props) {
  if (!props || typeof props !== "object") return "";
  return Object.entries(props).map(([k, v]) =>
    typeof v === "string" ? v : JSON.stringify(v)).join(" · ").slice(0, 90);
}

function completeToolCard(card, data) {
  if (!card) return;
  const wrap = card.closest(".msg");
  card.classList.remove("running");
  const r = data.result || {};
  const ok = data.ok && !r.error;
  card.classList.add(ok ? "ok" : "err");
  let html = "";
  if (r.error) {
    html = `<span style="color:var(--danger)">${esc(r.error)}</span>`;
  } else if (r.results) {
    const rows = r.results.slice(0, 4);
    html = `<div>${rows.length} result${rows.length === 1 ? "" : "s"}${r.summary ? " · has summary" : ""}</div><ul>` +
      rows.map((x) => typeof x === "string"
        ? `<li>${esc(x)}</li>`
        : `<li>${esc(x.title || "")}${x.url ? ` <code>${esc(String(x.url).replace(/^https?:\/\//, "").slice(0, 44))}</code>` : ""}</li>`).join("") +
      `</ul>`;
  } else if (r.current) {
    const c = r.current;
    html = `<strong>${esc(String(c.temp_c))} °C</strong> · ${esc(c.conditions || "")} — ${esc(r.place || "")}` +
      (r.next_days && r.next_days[0] ? `<div>today ${esc(String(r.next_days[0].max_c))}° / ${esc(String(r.next_days[0].min_c))}°</div>` : "");
  } else if (r.temp_c !== undefined) {
    html = `${esc(String(r.temp_c))}°C · ${esc(r.conditions || "")} — ${esc(r.place || "")}`;
  } else if (r.expression !== undefined) {
    html = `<code>${esc(String(r.expression))}</code> = <strong>${esc(String(r.result))}</strong>`;
  } else if (r.readable) {
    html = esc(r.readable);
  } else if (r.extract) {
    html = `<div><strong>${esc(r.title || "")}</strong></div><div>${esc(r.extract)}</div>`;
  } else if (r.chars !== undefined) {
    html = `${esc(String(r.status || ""))} · ${esc(String(r.chars))} characters from <code>${esc((r.url || "").slice(0, 60))}</code>`;
  } else if (r.written) {
    html = `Saved <code>${esc(r.path || "")}</code>`;
  } else if (r.exit !== undefined) {
    html = `<code>${esc(String(r.command || ""))}</code> → exit ${esc(String(r.exit))}`;
  } else {
    html = esc(JSON.stringify(r).slice(0, 240));
  }
  card.querySelector(".s-body").innerHTML = html;
  if (wrap) refreshActivityHeader(wrap);
}

function finishActivity(wrap) {
  const box = wrap && wrap.querySelector(".activity");
  if (!box) return;
  box.querySelector(".a-spin")?.remove();
  const ico = box.querySelector(".a-ico");
  if (ico && !ico.querySelector("svg")) {
    ico.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>`;
  }
  refreshActivityHeader(wrap);
}

function renderMedia(wrap, item) {
  const body = wrap.querySelector(".msg-body");
  let box = body.querySelector(".msg-media");
  if (!box) {
    box = document.createElement("div");
    box.className = "msg-media";
    const content = body.querySelector(".msg-content");
    body.insertBefore(box, content ? content.nextSibling : null);
  }
  const el = document.createElement(item.kind === "image" ? "img"
    : item.kind === "video" ? "video" : "audio");
  el.src = item.url;
  el.loading = "lazy";
  if (item.kind === "video") { el.controls = true; el.playsInline = true; }
  if (item.kind === "audio") { el.controls = true; el.className = "media-audio"; }
  if (item.kind === "image") {
    el.alt = item.name || "generated image";
    el.onclick = () => window.open(item.url, "_blank", "noopener");
  }
  box.appendChild(el);
  wrap._media = wrap._media || [];
  wrap._media.push(item.url);
}

function renderGrounding(wrap) {
  const old = wrap.querySelector(".grounding");
  if (old) old.remove();
  const g = wrap._meta?.grounding;
  if (!g || g.score === null || g.score === undefined || !g.claims) return;
  const pct = Math.round(g.score * 100);
  const cls = g.score >= 0.8 ? "ok" : g.score >= 0.5 ? "partial" : "low";
  const label = `${g.supported}/${g.claims} claims grounded`;
  const el = document.createElement("span");
  el.className = `grounding ${cls}`;
  el.title = "Each cited sentence was checked against the passage it cites.";
  el.innerHTML = `<svg viewBox="0 0 24 24">${g.score >= 0.5
      ? '<path d="M20 6L9 17l-5-5"/>'
      : '<path d="M12 9v5M12 17h.01M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'}</svg>${label}`;
  el.onclick = (ev) => { ev.stopPropagation(); showGroundingReport(wrap, ev); };
  const meta = wrap.querySelector(".answer-meta");
  if (meta) meta.prepend(el);
  else {
    const div = document.createElement("div");
    div.className = "answer-meta";
    div.appendChild(el);
    wrap.querySelector(".msg-body").appendChild(div);
  }
}

function showGroundingReport(wrap, ev) {
  const g = wrap._meta?.grounding;
  if (!g) return;
  const rect = ev.currentTarget.getBoundingClientRect();
  $("#popTitle").textContent = "Grounding check";
  $("#popMeta").innerHTML = `<span>${g.supported} of ${g.claims} cited claims supported</span>`;
  $("#popBody").innerHTML = `<div class="ground-list">${(g.details || []).map((d) => `
      <div class="ground-row">
        <span class="gdot ${d.supported ? "g" : "b"}"></span>
        <span class="gtxt">${esc(d.claim)}${d.supported ? "" : " — low overlap with the cited passage"}</span>
      </div>`).join("")}</div>`;
  $("#popFootScore").style.width = Math.round((g.score || 0) * 100) + "%";
  $("#popFootVal").textContent = `${Math.round((g.score || 0) * 100)}% supported`;
  const pop = el.popover;
  pop.hidden = false;
  const pr = pop.getBoundingClientRect();
  let left = Math.max(12, Math.min(window.innerWidth - pr.width - 12, rect.left));
  let top = rect.bottom + 8;
  if (top + pr.height > window.innerHeight - 12) top = Math.max(12, rect.top - pr.height - 8);
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}

function renderFollowups(wrap, questions) {
  wrap.querySelector(".followups")?.remove();
  if (!questions || !questions.length) return;
  const box = document.createElement("div");
  box.className = "followups";
  box.innerHTML = `<span class="followups-label">Ask next</span>` +
    questions.map((q) => `<button class="followup">${esc(q)}</button>`).join("");
  box.onclick = (e) => {
    const b = e.target.closest(".followup");
    if (b) send(b.textContent.trim());
  };
  wrap.querySelector(".msg-body").appendChild(box);
}

function renderWarning(wrap) {
  wrap.querySelector(".answer-warning")?.remove();
  if (!wrap._warning) return;
  const box = document.createElement("div");
  box.className = "answer-warning";
  box.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12 9v5M12 17h.01M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>
    <span>${esc(wrap._warning)}</span>`;
  wrap.querySelector(".msg-body").appendChild(box);
}

function renderMeta(wrap) {
  wrap.querySelector(".answer-meta")?.remove();
  const m = wrap._meta || {};
  if (!m.model && !m.latency_ms) return;
  const bits = [];
  if (m.demo) bits.push(`<span class="warnish">demo mode</span>`);
  else if (m.model) bits.push(esc(modelLabel(m.model)));
  if (m.latency_ms) bits.push(`${(m.latency_ms / 1000).toFixed(1)}s`);
  if (m.cost_usd) bits.push(`$${(+m.cost_usd).toFixed(4)}`);
  else if (m.tier === "free" || state.health?.llm?.free_tier) bits.push(`<span class="okish">free</span>`);
  if (!bits.length) return;
  const div = document.createElement("div");
  div.className = "answer-meta";
  div.innerHTML = bits.join(`<span class="sep"></span>`);
  wrap.querySelector(".msg-body").appendChild(div);
}

/* "google/gemini-3.6-flash" -> "gemini-3.6-flash", but prefer the human name the
   server already publishes for that id, and never print a bare vendor segment. */
function modelLabel(id) {
  const catalog = (state.modelCatalog && state.modelCatalog.models) || state.models || state.health?.models || [];
  const hit = catalog.find((x) => x.id === id);
  if (hit && hit.label) return hit.label;
  const parts = String(id).split("/").filter(Boolean);
  const tail = parts[parts.length - 1] || String(id);
  // "demo/model" or "x/default" would read as the word "model"; fall back to the
  // full id so the footer always says something identifiable.
  const generic = ["model", "default", "chat", "completions", "v1"];
  if (parts.length > 1 && generic.includes(tail.toLowerCase())) return parts.join("/");
  return tail;
}

function openSource(wrap, idx, rect) {
  const s = (wrap._sources || [])[idx];
  if (!s) return;
  state.popSource = s;
  $("#popTitle").innerHTML = s.url
    ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.doc_name)}</a>`
    : esc(s.doc_name);
  $("#popMeta").innerHTML = [
    s.url ? `<span style="color:var(--accent)">${esc(s.url.replace(/^https?:\/\//, "").slice(0, 60))}</span>` : "",
    (s.ext === ".pdf" && s.page) ? `page ${s.page}` : "",
    (s.heading_short || s.heading) ? esc(s.heading_short || s.heading) : "",
    s.method ? `match: ${s.method}` : "",
  ].filter(Boolean).map((x) => `<span>${x}</span>`).join("<span>·</span>");
  let body = esc(String(s.text || "").replace(/\*\*/g, "").replace(/^\s*\|.*\|\s*$/gm, (row) => row.replace(/\s*\|\s*/g, " · ").replace(/^ · | · $/g, "")));
  const q = (state.lastQuery || "").toLowerCase().match(/[a-z0-9]{4,}/g) || [];
  if (q.length) {
    const uniq = [...new Set(q)].slice(0, 8).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    try { body = body.replace(new RegExp(`\\b(${uniq.join("|")})\\b`, "gi"), "<mark>$1</mark>"); } catch {}
  }
  $("#popBody").innerHTML = body;
  $("#popFootScore").style.width = Math.max(6, Math.min(100, (s.score || 0) * 100)) + "%";
  $("#popFootVal").textContent = (s.score || 0).toFixed(2) + " relevance";
  const pop = el.popover;
  pop.hidden = false;
  const pr = pop.getBoundingClientRect();
  let left = rect.left + rect.width / 2 - pr.width / 2;
  let top = rect.bottom + 8;
  if (top + pr.height > window.innerHeight - 12) top = Math.max(12, rect.top - pr.height - 8);
  left = Math.max(12, Math.min(window.innerWidth - pr.width - 12, left));
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}

/* ─────────────────────────── chat list ─────────────────────────── */
function renderChatList() {
  const convs = state.conversations;
  if (!convs.length) {
    el.chatList.innerHTML = `<div class="chat-empty">No conversations yet. Ask something to get started.</div>`;
    return;
  }
  el.chatList.innerHTML = convs.map((c) => `
    <button class="chat-item ${c.id === state.convId ? "active" : ""}" data-id="${c.id}">
      <svg viewBox="0 0 24 24" class="i" style="width:14px;height:14px;opacity:.6"><path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z"/></svg>
      <span class="ci-title">${esc(c.title)}</span>
      <span class="ci-del" data-del="${c.id}" title="Delete">
        <svg viewBox="0 0 24 24" style="width:13px;height:13px"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>
      </span>
    </button>`).join("");
  el.chatList.querySelectorAll(".chat-item").forEach((b) => {
    b.onclick = (e) => {
      if (e.target.closest("[data-del]")) {
        e.stopPropagation();
        const id = e.target.closest("[data-del]").dataset.del;
        deleteConversation(id);
        return;
      }
      openConversation(b.dataset.id);
    };
  });
}

async function refreshConversations() {
  try {
    const data = await api("/api/conversations");
    state.conversations = data.conversations || [];
    renderChatList();
  } catch {}
}

async function openConversation(id) {
  if (state.streaming) return toast("Wait for the current answer to finish", "err");
  try {
    const conv = await api(`/api/conversations/${id}`);
    state.convId = id;
    state.messages = [];
    el.messages.innerHTML = "";
    el.chatTitle.textContent = conv.title || "Chat";
    toggleEmpty(false);
    (conv.messages || []).forEach((m) => {
      if (m.role === "user") {
        const w = msgShell("user");
        w.querySelector(".msg-content").textContent = m.content;
        el.messages.appendChild(w);
        state.messages.push({ role: "user", content: m.content });
      } else {
        const w = msgShell("assistant");
        w._buffer = m.content;
        w._sources = m.sources || [];
        w._meta = m.meta || {};
        w.querySelector(".msg-content").innerHTML = md(m.content);
        renderSources(w); renderMeta(w); renderGrounding(w); addTools(w);
        el.messages.appendChild(w);
        state.messages.push({ role: "assistant", content: m.content, sources: w._sources, meta: w._meta });
      }
    });
    renderChatList();
    closeSidebar();
    scrollToBottom(true);
  } catch (e) { toast(e.message, "err"); }
}

async function deleteConversation(id) {
  if (!confirm("Delete this conversation?")) return;
  try {
    await api(`/api/conversations/${id}`, { method: "DELETE" });
    if (state.convId === id) newChat();
    await refreshConversations();
    toast("Conversation deleted");
  } catch (e) { toast(e.message, "err"); }
}

function newChat() {
  if (state.streaming) { state.abort?.abort(); }
  state.convId = null;
  state.messages = [];
  el.messages.innerHTML = "";
  el.chatTitle.textContent = "New chat";
  toggleEmpty(true);
  renderChatList();
  closeSidebar();
  el.input.focus();
}

/* ─────────────────────────── send / stream ─────────────────────── */
function setBusy(busy) {
  state.streaming = busy;
  el.sendBtn.disabled = busy;
  el.stopBtn.hidden = !busy;
}

async function send(text, { regenerate = false } = {}) {
  text = (text || "").trim();
  if (!text || state.streaming) return;
  if (regenerate) {
    const last = el.messages.querySelector(".msg.assistant:last-of-type");
    if (last) last.remove();
    if (state.messages.length && state.messages[state.messages.length - 1].role === "assistant") state.messages.pop();
  } else {
    addUserMessage(text);
    el.input.value = "";
    autoGrow();
    if (!state.convId) el.chatTitle.textContent = text.slice(0, 60);
  }

  const wrap = addAssistantMessage();
  setBusy(true);
  state.lastQuery = text;
  const ctrl = new AbortController();
  state.abort = ctrl;

  const payload = {
    message: regenerate ? "" : text,
    conversation_id: state.convId,
    model: state.settings.model || "auto",
    strict: !!state.settings.strict,
    knowledge: state.settings.knowledge || "web+docs",
    regenerate,
    doc_ids: state.docScope && state.docScope.length ? state.docScope : null,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const t = await res.text();
      let msg = `Request failed (${res.status})`;
      try { msg = JSON.parse(t).detail || msg; } catch {}
      throw new Error(msg);
    }
    await readSSE(res.body, (ev, data) => {
      if (ev === "start") {
        if (data.conversation_id) { state.convId = data.conversation_id; renderChatList(); }
        if (data.demo) el.modeBadge.hidden = false;
      } else if (ev === "sources") {
        wrap._sources = data.sources || [];
        state.lastDiag = data.diagnostics || null;
        state.lastQuery = data.query_used || text;
        const n = wrap._sources.length;
        setThinking(wrap, n ? `Reading ${n} passage${n > 1 ? "s" : ""}…` : "No matching passages…");
        if (n) renderSourcesPreview(wrap);
      } else if (ev === "delta") {
        const th = wrap.querySelector(".thinking");
        if (th) th.remove();
        wrap._buffer += data.text || "";
        streamRender(wrap);
      } else if (ev === "text_replace") {
        wrap._buffer = data.text || "";
        streamRender(wrap);
      } else if (ev === "route") {
        wrap._meta = wrap._meta || {};
        // `data.provider` is the provider's display label; the model id arrives as `model`.
        // The old assignment read `.model` off the route payload but stored it under
        // `provider`, so the footer printed the literal word "model".
        wrap._meta.model = data.model;
        wrap._meta.task = data.task;
        wrap._meta.why = data.why;
        wrap._meta.provider = "";
        wrap._meta.providerLabel = data.provider;
        const pill = wrap.querySelector(".msg-who, .msg-name");
        if (pill && data.model) pill.dataset.model = data.model;
      } else if (ev === "media") {
        renderMedia(wrap, data);
      } else if (ev === "notice") {
        toast(data.message || "Retrying…");
      } else if (ev === "tools") {
        wrap._tools = data.available || [];
      } else if (ev === "tool_start") {
        const th = wrap.querySelector(".thinking");
        if (th) th.remove();
        wrap._toolCards = wrap._toolCards || {};
        wrap._toolCards[data.id] = toolCard(wrap, data.name, data.label, describeArgs(data.arguments));
        wrap._toolCallsUsed = (wrap._toolCallsUsed || 0) + 1;
        setThinking(wrap, `${data.label}…`);
      } else if (ev === "tool_result") {
        completeToolCard((wrap._toolCards || {})[data.id], data);
      } else if (ev === "grounding") {
        wrap._meta = { ...wrap._meta, grounding: data };
      } else if (ev === "warning") {
        wrap._warning = data.message || "";
      } else if (ev === "followups") {
        wrap._pendingFollowups = data.questions || [];
      } else if (ev === "title") {
        el.chatTitle.textContent = data.title;
        refreshConversations();
      } else if (ev === "usage") {
        wrap._meta = { ...wrap._meta, ...data };
      } else if (ev === "error") {
        wrap._error = data.message;
      } else if (ev === "done") {
        // merge the server's final numbers first, then render immediately instead of
        // waiting for the socket to close — nothing after `done` can make it look unfinished
        if (data.conversation_id) state.convId = data.conversation_id;
        if (data.meta) wrap._meta = { ...wrap._meta, ...data.meta };
        if (data.message_id) wrap._meta = { ...wrap._meta, message_id: data.message_id };
        if (!wrap._finalized) finalizeAssistant(wrap);
      }
    });
  } catch (e) {
    if (e.name === "AbortError") {
      wrap._stopped = true;
    } else {
      wrap._error = e.message;
    }
  } finally {
    setBusy(false);
    state.abort = null;
    if (wrap._error) {
      wrap.querySelector(".msg-content").innerHTML = "";
      wrap.classList.remove("streaming");
      wrap.classList.add("msg-error");
      const box = document.createElement("div");
      box.className = "msg-body";
      box.innerHTML = `<div class="msg-who" style="color:var(--danger)">Couldn't complete that</div>
        <div style="margin-top:4px">${esc(wrap._error)}</div>`;
      wrap.innerHTML = "";
      wrap.appendChild(box);
      toast(wrap._error, "err");
    } else {
      finalizeAssistant(wrap, { stopped: !!wrap._stopped });
      if (wrap._toolCards) loadWorkspace();
      if (wrap._pendingFollowups?.length && !wrap._stopped) renderFollowups(wrap, wrap._pendingFollowups);
      if (state.convId) {
        el.chatTitle.textContent = (state.messages.find((m) => m.role === "user")?.content || "Chat").slice(0, 60);
      }
      refreshConversations();
    }
    renderDiag();
  }
}

function renderSourcesPreview(wrap) {
  if (wrap.querySelector(".sources")) return;
  wrap.querySelector(".msg-body").insertAdjacentHTML("beforeend",
    `<div class="sources"><div class="src-list" style="margin-top:2px"></div></div>`);
  const list = wrap.querySelector(".sources .src-list");
  list.innerHTML = wrap._sources.slice(0, 5).map((s, i) =>
    `<span class="src-chip" style="opacity:.6"><span class="n">${i + 1}</span><span class="nm">${esc(s.doc_name)}</span></span>`).join("");
}

async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let ev = "message"; let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) { try { onEvent(ev, JSON.parse(data)); } catch {} }
    }
  }
}

function regenerate() {
  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  if (!lastUser) return toast("Nothing to regenerate", "err");
  if (!state.convId) return;
  const count = state.messages.filter((m) => m.role === "assistant").length;
  send(lastUser.content, { regenerate: true });
}

/* ─────────────────────────── docs / drawer ─────────────────────── */
function openDrawer(tab = "docs") {
  el.drawer.hidden = false;
  el.scrim.hidden = false;
  switchTab(tab);
  refreshDocs();
}
function closeDrawer() { el.drawer.hidden = true; el.scrim.hidden = true; }
function switchTab(tab) {
  state.activeTab = tab;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
  $("#panel-docs").hidden = tab !== "docs";
  $("#panel-retrieval").hidden = tab !== "retrieval";
  if (tab === "retrieval") renderDiag();
}

function docIcon(ext) {
  const label = (ext || "").replace(".", "").toUpperCase().slice(0, 4) || "DOC";
  const cls = (ext || "").replace(".", "");
  return `<div class="doc-ico ${cls}">${label}</div>`;
}

function renderDocs() {
  const docs = state.docs;
  const totalChunks = docs.reduce((a, d) => a + (d.n_chunks || 0), 0);
  if (!docs.length) {
    el.docList.innerHTML = `<div class="empty-mini">No documents yet.<br>Drop a file above or load the demo pack.</div>`;
  } else {
    el.docList.innerHTML = docs.map((d) => `
      <div class="doc" data-doc="${d.id}">
        <div class="doc-main">
          ${docIcon(d.ext)}
          <div style="flex:1;min-width:0">
            <div class="doc-name" data-view="${d.id}">${esc(d.name)}</div>
            <div class="doc-sub">${d.n_chunks} passages · ${fmtBytes(d.size)}${d.ext === ".pdf" && d.pages ? ` · ${d.pages} pages` : ""}</div>
          </div>
          <div class="doc-actions">
            <button class="icon-btn" data-view="${d.id}" title="Preview passages">
              <svg viewBox="0 0 24 24"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>
            </button>
            <button class="icon-btn" data-del="${d.id}" title="Remove document">
              <svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>
            </button>
          </div>
        </div>
      </div>`).join("");
  }
  el.docListMeta.textContent = docs.length ? `${docs.length} document${docs.length > 1 ? "s" : ""} · ${totalChunks} passages` : "";
  el.kbDocs.textContent = docs.length;
  setCount(el.docCount, docs.length);
  el.kbMeta.textContent = docs.length ? `${docs.length} doc${docs.length > 1 ? "s" : ""} · ${fmtNum(totalChunks)} passages` : "No documents yet";
  // drop scoped ids that no longer exist
  if (state.docScope) {
    const live = new Set(docs.map((d) => d.id));
    state.docScope = state.docScope.filter((id) => live.has(id));
    if (!state.docScope.length) state.docScope = null;
  }
  refreshScopeChip();
  $("#demoBtn").hidden = docs.length > 0;
  $("#reindexBtn").hidden = !(state.health && !state.health.embeddings.semantic && docs.length);
}

async function refreshDocs() {
  try {
    const data = await api("/api/documents");
    state.docs = data.documents || [];
    renderDocs();
  } catch (e) { toast(e.message, "err"); }
}

async function viewChunks(docId) {
  const existing = $(`.doc[data-doc="${docId}"] .doc-chunks`);
  if (existing) { existing.remove(); return; }
  try {
    const data = await api(`/api/documents/${docId}/chunks?limit=4`);
    const box = document.createElement("div");
    box.className = "doc-chunks";
    box.innerHTML = (data.chunks || []).map((c) => `
      <p class="chunk-prev"><b>#${c.ordinal + 1}${c.heading ? " · " + esc(c.heading.split("›").pop().trim()) : ""}</b><br>${esc((c.text || "").slice(0, 260))}…</p>`).join("")
      || `<p class="chunk-prev">No passages.</p>`;
    $(`.doc[data-doc="${docId}"]`).appendChild(box);
  } catch (e) { toast(e.message, "err"); }
}

async function deleteDoc(id) {
  if (!confirm("Remove this document and its embeddings?")) return;
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    await refreshDocs();
    await refreshHealth();
    toast("Document removed");
  } catch (e) { toast(e.message, "err"); }
}

async function uploadFiles(files) {
  const list = [...files];
  if (!list.length) return;
  if (el.drawer.hidden) openDrawer("docs");
  el.queue.innerHTML = "";
  list.forEach((f) => el.queue.insertAdjacentHTML("beforeend",
    `<div class="queue-item"><span class="spin"></span> Uploading ${esc(f.name)}…</div>`));
  const fd = new FormData();
  list.forEach((f) => fd.append("files", f));
  try {
    const res = await fetch("/api/documents", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    (data.added || []).forEach((d) => toast(`Added ${d.name} — ${d.n_chunks} passages${d.embedded ? "" : " (keyword only)"}`));
    if ((data.added || []).length) el.queue.innerHTML = "";
    (data.errors || []).forEach((e) => toast(`${e.name}: ${e.error}`, "err"));
    await refreshDocs();
    await refreshHealth();
    loadSuggestions(true);
  } catch (e) {
    toast(e.message, "err");
  } finally {
    el.queue.innerHTML = "";
    el.fileInput.value = "";
  }
}

async function loadDemo() {
  try {
    const btn = $("#heroDemo"); if (btn) btn.disabled = true;
    const data = await api("/api/demo/load", { method: "POST" });
    toast(`Demo knowledge base loaded — ${data.added.length} documents`);
    await refreshDocs(); await refreshHealth(); loadSuggestions(true);
    closeDrawer();
  } catch (e) { toast(e.message, "err"); }
  finally { const btn = $("#heroDemo"); if (btn) btn.disabled = false; }
}

/* ─────────────────────── document scope menu ───────────────────── */
function scopeLabel() {
  if (!state.docScope || !state.docScope.length) return "All documents";
  if (state.docScope.length === 1) {
    const d = state.docs.find((x) => x.id === state.docScope[0]);
    if (!d) return "1 document";
    const dot = d.name.lastIndexOf(".");
    return (dot > 0 ? d.name.slice(0, dot) : d.name).slice(0, 24);
  }
  return `${state.docScope.length} documents`;
}

function refreshScopeChip() {
  $("#scopeChip").hidden = !state.docs.length;
  $("#scopeLabel").textContent = scopeLabel();
  const limited = !!(state.docScope && state.docScope.length);
  $("#scopeChip").classList.toggle("on", limited);
  if (!limited) $("#scopeChip").classList.remove("on");
}

function openScopeMenu() {
  const menu = $("#scopeMenu");
  state.drafts = new Set(state.docScope || []);
  const docs = state.docs;
  $("#scopeList").innerHTML = docs.length
    ? docs.map((d) => `<label class="scope-item">
        <input type="checkbox" value="${d.id}" ${state.drafts.has(d.id) ? "checked" : ""}>
        <span>${esc(d.name)}</span>
      </label>`).join("")
    : `<div class="empty-mini">No documents yet.</div>`;
  $("#scopeCount").textContent = state.drafts.size
    ? `${state.drafts.size} selected` : "All documents";
  $("#scopeList").onchange = (e) => {
    const cb = e.target.closest("input[type=checkbox]");
    if (!cb) return;
    if (cb.checked) state.drafts.add(cb.value); else state.drafts.delete(cb.value);
    $("#scopeCount").textContent = state.drafts.size ? `${state.drafts.size} selected` : "All documents";
  };
  const chip = $("#scopeChip").getBoundingClientRect();
  menu.hidden = false;
  const m = menu.getBoundingClientRect();
  menu.style.left = Math.max(12, Math.min(window.innerWidth - m.width - 12, chip.left)) + "px";
  menu.style.top = Math.max(12, chip.top - m.height - 8) + "px";
  state.scopeOpen = true;
}

function closeScopeMenu() { $("#scopeMenu").hidden = true; state.scopeOpen = false; }

function applyScope() {
  state.docScope = state.drafts.size ? [...state.drafts] : null;
  localStorage.setItem("cortex-scope", JSON.stringify(state.docScope));
  refreshScopeChip();
  closeScopeMenu();
  if (state.docScope) toast(`Searching ${state.docScope.length} document${state.docScope.length > 1 ? "s" : ""}`);
  else toast("Searching all documents");
}

/* ─────────────────────────── diagnostics ───────────────────────── */
function renderDiag() {
  const d = state.lastDiag;
  const box = $("#lastDiag");
  if (!box) return;
  if (!d) { box.innerHTML = `<div class="empty-mini">Ask a question to see live retrieval diagnostics.</div>`; return; }
  box.innerHTML = `
    <div class="diag-cell"><b>${esc(d.mode || "—")}</b><span>retrieval mode</span></div>
    <div class="diag-cell"><b>${d.ms ?? "—"} ms</b><span>latency</span></div>
    <div class="diag-cell"><b>${d.lexical_hits ?? 0} / ${d.semantic_hits ?? 0}</b><span>lexical / semantic hits</span></div>
    <div class="diag-cell"><b>${(d.confidence ?? 0).toFixed(2)}</b><span>top relevance</span></div>
    <div class="diag-cell" style="grid-column:1/-1"><b style="font-size:12px;font-weight:560;line-height:1.4">${esc(d.query_used || "—")}</b><span>search query used</span></div>`;
}

async function probe() {
  const q = $("#probeInput").value.trim();
  if (!q) return;
  const box = $("#probeResults");
  box.innerHTML = `<div class="queue-item"><span class="spin"></span> Searching…</div>`;
  try {
    const data = await api(`/api/search?q=${encodeURIComponent(q)}&k=6`);
    state.lastQuery = q;
    if (!data.sources.length) { box.innerHTML = `<div class="empty-mini">No passages matched.</div>`; return; }
    box.innerHTML = data.sources.map((s, i) => `
      <div class="probe-item">
        <div class="probe-top">
          <span class="src-chip" style="pointer-events:none;padding:2px 6px"><span class="n">${i + 1}</span></span>
          <strong style="font-size:12px">${esc(s.doc_name)}</strong>
          <span class="pg muted-sm" style="margin:0">${s.ext === ".pdf" ? "p." + s.page : ""}</span>
          <span class="probe-score">${(s.score || 0).toFixed(2)}</span>
        </div>
        <div class="probe-text">${esc((s.text || "").slice(0, 220))}</div>
      </div>`).join("");
    state.lastDiag = { ...data.diagnostics, query_used: q };
    renderDiag();
  } catch (e) { box.innerHTML = ""; toast(e.message, "err"); }
}

/* ─────────────────────────── models ───────────────────────────── */
async function loadModels(force = false) {
  try {
    const d = await api(`/api/models${force ? "?force=true" : ""}`);
    state.modelCatalog = d;
    renderModelMenu();
    const label = $("#modelBtnLabel");
    const dot = document.querySelector("#modelBtn .rb-dot, .model-btn .rb-dot");
    const chosen = currentModelLabel();
    if (label) label.textContent = chosen;
    if (dot) dot.classList.toggle("off", !d.provider?.configured);
    const line = $("#catalogueLine");
    if (line) {
      const b = d.by_type || {};
      line.textContent = d.provider?.configured
        ? `${d.count} model${d.count === 1 ? "" : "s"} available · ${b.language || 0} language, ${b.image || 0} image, ${b.video || 0} video, ${b.speech || 0} speech`
        : `${d.count} models in the catalogue — connect a key to use them`;
    }
    renderRouteTable();
  } catch { /* provider may be offline */ }
}

function currentModelLabel() {
  const id = state.settings.model || "auto";
  if (id === "auto") return "Auto";
  if (id.startsWith("task:")) {
    const opt = (state.modelCatalog?.options || []).find((o) => o.id === id);
    return opt ? opt.label : id.slice(5);
  }
  const opt = (state.modelCatalog?.options || []).find((o) => o.id === id);
  return opt ? opt.label : String(id).split("/").pop();
}

/* A fixed-position panel needs coordinates. Anchor it under its button, right
   aligned, and flip it above when there isn't room below. */
function placePopmenu(menu, anchor) {
  if (!menu || !anchor) return;
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth || 320;
  const h = menu.offsetHeight || 420;
  let left = r.right - w;
  left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
  let top = r.bottom + 6;
  if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 6);
  menu.style.left = `${Math.round(left)}px`;
  menu.style.top = `${Math.round(top)}px`;
  menu.style.right = "auto";
}

function renderModelMenu(filter = "") {
  const box = $("#modelList");
  if (!box) return;
  const opts = state.modelCatalog?.options || [];
  const f = filter.trim().toLowerCase();
  const groups = new Map();
  for (const o of opts) {
    if (f && !`${o.label} ${o.id} ${o.group}`.toLowerCase().includes(f)) continue;
    if (!groups.has(o.group)) groups.set(o.group, []);
    groups.get(o.group).push(o);
  }
  const active = state.settings.model || "auto";
  let html = "";
  for (const [group, items] of groups) {
    html += `<div class="popmenu-group">${esc(group)}</div>`;
    html += items.slice(0, f ? 60 : 14).map((o) => `
      <button class="popmenu-item ${o.id === active ? "active" : ""}" data-model="${esc(o.id)}">
        <span class="pm-name">${esc(o.label)}</span>
        <span class="pm-hint">${esc(o.hint || "")}</span>
      </button>`).join("");
  }
  let empty = "";
  if (!(state.models && state.models.length) && !f) {
    empty = `<div class="popmenu-empty">
         <strong>No provider connected</strong>
         <span>Add an AI Gateway or OpenRouter key in Settings and Cortex will route each task to
         the best model behind it. Without one it answers from your documents and the live tools.</span>
         <button class="popmenu-cta" data-open-settings>Open settings</button>
       </div>`;
  } else if (!html) {
    empty = `<div class="popmenu-group">No models match</div>`;
  }
  box.innerHTML = html || empty;
  const prov = state.modelCatalog?.provider || {};
  if ((prov.free || prov.free_tier || prov.provider === "free") && html) {
    const note = document.createElement("div");
    note.className = "popmenu-note";
    note.innerHTML = (prov.free
      ? '<strong>Free mode.</strong> Every task routes to a free model — no key, ' +
        'no card, no bill. Add a provider key any time to unlock more models.'
      : '<strong>Free tier.</strong> This key runs on a no-card free tier, so answers cost ' +
        '$0. Add a different provider key any time to unlock more models.');
    const link = document.createElement("button");
    link.className = "popmenu-cta";
    link.textContent = "Open settings";
    link.onclick = () => { $("#modelMenu").hidden = true; openSettings(); };
    note.appendChild(link);
    box.appendChild(note);
  }
  const cta = box.querySelector("[data-open-settings]");
  if (cta) cta.onclick = () => { $("#modelMenu").hidden = true; openSettings(); };
  box.querySelectorAll("[data-model]").forEach((b) => {
    b.onclick = () => {
      saveSetting({ model: b.dataset.model });
      $("#modelMenu").hidden = true;
      $("#modelBtnLabel").textContent = currentModelLabel();
      renderModelMenu($("#modelFilter").value);
    };
  });
}

function renderFreePanel() {
  const box = $("#freePanel");
  if (!box) return;
  const h = state.health || {};
  const llm = h.llm || {};
  const pool = (llm.pool || []).map((p) => p.label).filter(Boolean);
  const media = h.media || {};
  const ft = !!(llm.free_tier ?? llm.free);
  const rows = [
    ["Text + tools", llm.free ? "free (keyless)" : ft ? "free tier" : "provider key", ft],
    ["Images", "free (flux)", true],
    ["Speech", media.free_speech ? "free (edge voices)" : "unavailable", media.free_speech],
    ["Video", media.free_video ? "free (local render)" : "unavailable", media.free_video],
    ["Web research", h.web?.provider ? h.web.provider : "—", !!h.web?.provider],
    ["Code workspace", "free (local sandbox)", true],
  ];
  box.innerHTML =
    `<p class="muted-sm" style="margin:0 0 10px">` +
    (llm.free
      ? "You are in <strong>free mode</strong> — nothing here is billed. Routes are tried in order " +
        "and the assistant keeps going if one is busy."
      : ft
        ? "You are on the <strong>free tier</strong> — no card on file, so these turns cost $0. " +
          "If the tier runs out of quota the assistant drops to the free routes below."
        : "A provider key is active. The free routes below still work as failovers.") +
    `</p>` +
    `<div class="free-rows">` + rows.map(([k, v, ok]) =>
      `<div class="free-row"><span>${esc(k)}</span>` +
      `<span class="${ok ? "ok" : "off"}">${esc(v)}</span></div>`).join("") + `</div>` +
    (pool.length ? `<p class="muted-sm" style="margin:10px 0 0"><strong>Route order:</strong> ` +
      pool.map(esc).join(" → ") + `</p>` : "") +
    `<p class="muted-sm" style="margin:8px 0 0">` +
    (llm.local
      ? "Local model detected — it answers even with no internet."
      : "Want a truly offline fallback? Run <code>scripts/install_local_model.sh</code> once, then " +
        "start it with <code>~/.local/share/cortex/local/start.sh</code>.") +
    `</p>`;
}

function renderRouteTable() {
  const box = $("#routeTable");
  if (!box) return;
  const caps = state.modelCatalog?.routes;
  if (!caps) {
    box.innerHTML = `<div class="muted-sm">Connect a provider to see routing.</div>`;
    return;
  }
  const labels = { reasoning: "Reasoning", code: "Coding", general: "Everyday", creative: "Creative",
                   fast: "Fast", vision: "Vision", image: "Images", speech: "Speech",
                   video: "Video", transcribe: "Transcription" };
  box.innerHTML = Object.entries(caps).map(([task, ids]) => `
    <div class="route-row">
      <span class="route-task">${esc(labels[task] || task)}</span>
      <span class="route-model">${ids.length ? esc(ids[0]) : "—"}</span>
      <span class="route-rank">${ids.length > 1 ? "+" + (ids.length - 1) + " fallback" : ""}</span>
    </div>`).join("");
}

const KNOWLEDGE_CYCLE = [
  ["web+docs", "Web + Docs"],
  ["web", "Web only"],
  ["docs", "Docs only"],
];

function setKnowledge(mode) {
  const hit = KNOWLEDGE_CYCLE.find(([m]) => m === mode) || KNOWLEDGE_CYCLE[0];
  const label = $("#knowledgeLabel");
  if (label) label.textContent = hit[1];
  const chip = $("#knowledgeChip");
  if (chip) chip.classList.toggle("on", hit[0] !== "web+docs");
  saveSetting({ knowledge: hit[0] });
}

function cycleKnowledge() {
  const cur = state.settings.knowledge || "web+docs";
  const i = KNOWLEDGE_CYCLE.findIndex(([m]) => m === cur);
  setKnowledge(KNOWLEDGE_CYCLE[(i + 1) % KNOWLEDGE_CYCLE.length][0]);
  toast(`Knowledge: ${$("#knowledgeLabel").textContent}`);
}

async function loadWorkspace() {
  if (!state.convId) { $("#wsBody").innerHTML = `<p class="muted-sm">Start a chat — files the assistant writes appear here.</p>`; return; }
  try {
    const d = await api(`/api/workspace/${state.convId}/files`);
    const files = d.files || [];
    const badge = $("#wsCount");
    if (badge) { badge.hidden = !files.length; badge.textContent = files.length; }
    $("#wsBody").innerHTML = files.length ? files.map((f) => `
      <div class="ws-file" data-path="${esc(f.path)}">
        <span class="wf-kind">${esc(f.kind)}</span>
        <span class="wf-path">${esc(f.path)}</span>
        <span class="wf-size">${fmtBytes(f.bytes)}</span>
      </div>`).join("") + `<p class="muted-sm" style="margin-top:12px">Files persist with this conversation. Use Download .zip to take the project with you.</p>`
      : `<p class="muted-sm">No files yet. Ask the assistant to build something.</p>`;
    $("#wsBody").querySelectorAll(".ws-file").forEach((row) => {
      row.onclick = async () => {
        try {
          const r = await api(`/api/workspace/${state.convId}/file?path=${encodeURIComponent(row.dataset.path)}`);
          showPop(row.dataset.path, `<pre style="white-space:pre-wrap;font-size:11.5px">${esc(r.content.slice(0, 6000))}</pre>`);
        } catch (e) { toast(e.message, "err"); }
      };
    });
  } catch { /* ignore */ }
}

/* ─────────────────────────── settings ──────────────────────────── */
function openSettings() {
  const h = state.health || {};
  const keyOn = h.llm?.configured;
  const prov = h.llm?.provider;
  $("#keyState").innerHTML = `<span class="sdot ${keyOn ? "" : "off"}"></span>
    ${keyOn
      ? `Connected to <b>${esc(prov || "provider")}</b> <span style="color:var(--faint)">— ${esc(h.llm?.base_url || "")}</span>`
      : h.llm?.key_kind === "vercel-token (catalogue only)"
        ? "Catalogue-only token: models listed, inference needs a gateway key"
        : "No model provider — demo mode"}`;
  // keep the picker in step with the provider that is actually answering, so the
  // form cannot say "Vercel" while the status line above says "google"
  const psel = $("#providerSelect");
  if (psel) {
    const active = String(h.llm?.provider || "").toLowerCase().trim();
    // the health endpoint reports short names ("google"); accept its label form too
    const known = ["gateway", "openrouter", "openai", "groq", "deepseek", "google",
                   "together", "ollama", "custom"];
    const hit = known.find((k) => active === k || active.startsWith(k));
    if (hit) psel.value = hit;
  }
  $("#apiKey").value = "";
  $("#orgInput").value = state.settings.org || "";
  const ks = $("#knowledgeSelect");
  if (ks) ks.value = state.settings.knowledge || "web+docs";
  $("#strictToggle").checked = !!state.settings.strict;
  $("#sourcesToggle").checked = state.settings.show_sources !== false;
  const mr = $("#minRel");
  mr.value = state.settings.min_relevance;
  $("#minRelVal").textContent = `(${(+state.settings.min_relevance).toFixed(2)})`;
  updateRangeFill();
  const st = h.embeddings || {};
  const stats = h.stats || {};
  api("/api/quality").then((q) => {
    const row = $("#qualityKv");
    if (!row) return;
    row.innerHTML = `
      <dt>Answers</dt><dd>${fmtNum(q.answers)}</dd>
      <dt>Well grounded</dt><dd><span class="ok">${fmtNum(q.well_grounded)}</span></dd>
      <dt>Low grounding</dt><dd>${q.low_grounding ? `<span class="warn">${fmtNum(q.low_grounding)}</span>` : "0"}</dd>
      <dt>Rated helpful</dt><dd>${q.rated ? `${q.positive}/${q.rated}` : "—"}</dd>
      <dt>Total model spend</dt><dd>$${(q.cost_usd || 0).toFixed(3)}</dd>`;
  }).catch(() => {});
  $("#statusKv").innerHTML = `
    <dt>Documents</dt><dd>${fmtNum(stats.documents)}</dd>
    <dt>Passages</dt><dd>${fmtNum(stats.chunks)}</dd>
    <dt>Embeddings</dt><dd>${fmtNum(stats.embedded)} / ${fmtNum(stats.chunks)}</dd>
    <dt>Retrieval</dt><dd>${st.semantic ? `<span class="ok">${h.reranker && h.reranker.status === "ready" ? "hybrid + rerank" : "hybrid (vectors + BM25)"}</span>` : `<span class="warn">keyword only</span>`}</dd>
    <dt>Reranker</dt><dd>${h.reranker && h.reranker.status === "ready" ? `<span class="ok">ready</span>` : `<span class="warn">${esc(h.reranker?.status || "off")}</span>`}</dd>
    <dt>Embedding model</dt><dd style="font-family:var(--mono);font-size:11px">${esc(st.model || "—")}</dd>`;
  el.settingsModal.hidden = false;
  loadTools();
}
function closeSettings() { el.settingsModal.hidden = true; }

function updateRangeFill() {
  const r = $("#minRel");
  const pct = ((r.value - r.min) / (r.max - r.min)) * 100;
  r.style.setProperty("--fill", pct + "%");
}

async function saveKey() {
  const key = $("#apiKey").value.trim();
  const provider = $("#providerSelect").value || "gateway";
  const base = ($("#customBase").value || "").trim();
  if (!key) return toast("Paste an API key first", "err");
  const btn = $("#saveKey");
  btn.disabled = true; btn.textContent = "Testing…";
  try {
    const res = await api("/api/key", { method: "POST",
      body: JSON.stringify({ api_key: key, provider, base_url: base }) });
    $("#apiKey").value = "";
    toast(`Key verified — ${res.provider?.provider || provider} is live`);
    await refreshHealth();
    await loadModels(true);
    openSettings();
    loadSuggestions(true);
  } catch (e) {
    toast(e.message, "err");
  } finally {
    btn.disabled = false; btn.textContent = "Save & test";
  }
}

async function saveSetting(patch, opts = {}) {
  state.settings = { ...state.settings, ...patch };
  applySettingsToUi();
  try {
    state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(patch) });
  } catch (e) { if (!opts.silent) toast(e.message, "err"); }
}

function applySettingsToUi() {
  const on = !!state.settings.strict;
  el.strictChip.classList.toggle("on", on);
  el.strictLabel.textContent = on ? "Strict mode" : "Helpful mode";
  el.strictChip.title = on
    ? "Strict mode: answers only from your documents. Click for helpful mode."
    : "Helpful mode: may answer from general knowledge (clearly marked). Click for strict mode.";
  updateStatusLine();
}

/* ─────────────────────────── tools panel ───────────────────────── */
const TOOL_TEST_ARGS = {
  calculator: { expression: "(649-199)/199*100" },
  current_datetime: { timezone_name: "Africa/Johannesburg" },
  weather: { place: "Johannesburg" },
  wikipedia: { query: "Retrieval-augmented generation" },
  web_search: { query: "latest Anthropic Claude model", max_results: 3 },
  fetch_url: { url: "https://api.open-meteo.com/v1/forecast?latitude=-26.2&longitude=28.0&current=temperature_2m", max_chars: 300 },
};

let toolsState = null;

async function loadTools() {
  try {
    toolsState = await api("/api/tools");
  } catch { return; }
  renderTools();
}

function renderTools() {
  if (!toolsState) return;
  const list = $("#toolsList");
  if (!list) return;
  list.innerHTML = toolsState.tools.map((t) => `
    <div class="tool-row" data-tool="${t.name}">
      <div class="tool-ico"><svg viewBox="0 0 24 24">${TOOL_ICONS[t.name] || '<circle cx="12" cy="12" r="9"/>'}</svg></div>
      <div class="tool-meta">
        <div class="tool-name">${esc(t.label)}${t.usable ? "" : `<span class="tool-need" title="${esc(t.needs || "")}">needs API key</span>`}</div>
        <div class="tool-desc">${esc(t.description)}</div>
      </div>
      <button class="tool-test" data-test="${t.name}" ${t.enabled && t.usable ? "" : "disabled"}>Test</button>
      <label class="switch"><input type="checkbox" data-toggle="${t.name}" ${t.enabled ? "checked" : ""} ${t.usable ? "" : "disabled"}><span></span></label>
    </div>`).join("");

  list.querySelectorAll("[data-toggle]").forEach((cb) => {
    cb.onchange = async () => {
      const enabled = new Set(toolsState.tools.filter((t) => t.enabled).map((t) => t.name));
      if (cb.checked) enabled.add(cb.dataset.toggle); else enabled.delete(cb.dataset.toggle);
      try {
        const patch = await api("/api/tools", { method: "PUT", body: JSON.stringify({ enabled: [...enabled] }) });
        toolsState = { ...toolsState, ...patch };
        renderTools();
        toast(cb.checked ? "Tool enabled" : "Tool disabled");
      } catch (e) { toast(e.message, "err"); cb.checked = !cb.checked; }
    };
  });

  list.querySelectorAll("[data-test]").forEach((btn) => {
    btn.onclick = async () => {
      const name = btn.dataset.test;
      btn.disabled = true; btn.textContent = "…";
      try {
        const r = await api(`/api/tools/${name}/test`, {
          method: "POST", body: JSON.stringify({ arguments: TOOL_TEST_ARGS[name] || {} }),
        });
        const summary = r.ok ? JSON.stringify(r.result).slice(0, 120) : r.result.error;
        toast(`${name}: ${summary} (${r.ms} ms)`, r.ok ? "" : "err");
      } catch (e) { toast(e.message, "err"); }
      finally { btn.disabled = false; btn.textContent = "Test"; }
    };
  });

  const allow = $("#allowlist");
  if (allow && document.activeElement !== allow) {
    allow.value = (toolsState.config.http_allowlist || []).join(", ");
  }
  renderSecrets();
}

function renderSecrets() {
  const box = $("#secretRows");
  if (!box || !toolsState) return;
  const rows = Object.entries(toolsState.config.secrets || {});
  box.innerHTML = rows.length
    ? rows.map(([host, cred], i) => `
        <div class="secret-row" data-i="${i}">
          <input class="s-host" placeholder="api.example.com" value="${esc(host)}" />
          <input class="s-header" placeholder="Authorization" value="${esc(cred.header || "")}" />
          <input class="s-value" type="password" placeholder="Bearer …" value="${esc(cred.value || "")}" />
          <button class="icon-btn" data-remove="${i}" title="Remove">
            <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>
          </button>
        </div>`).join("")
    : `<div class="muted-sm">No credentials configured.</div>`;

  box.querySelectorAll("[data-remove]").forEach((b) => {
    b.onclick = () => saveSecrets(rows.filter((_, i) => i !== +b.dataset.remove));
  });
  box.querySelectorAll(".secret-row input").forEach((inp) => {
    inp.onchange = () => {
      const next = [...box.querySelectorAll(".secret-row")].map((row) => [
        row.querySelector(".s-host").value.trim(),
        { header: row.querySelector(".s-header").value.trim(), value: row.querySelector(".s-value").value },
      ]).filter(([h, c]) => h && c.header);
      saveSecrets(next);
    };
  });
}

async function saveSecrets(entries) {
  const secrets = {};
  (entries || []).forEach(([host, cred]) => { if (host && cred.header && cred.value) secrets[host] = cred; });
  try {
    const patch = await api("/api/tools", { method: "PUT", body: JSON.stringify({ secrets }) });
    toolsState = { ...toolsState, ...patch };
    renderTools();
    toast("Credentials saved (server-side only)");
  } catch (e) { toast(e.message, "err"); }
}

/* ─────────────────────────── status / banner ───────────────────── */
function setCount(node, n) {
  if (!node) return;
  node.textContent = n > 0 ? String(n) : "";
  node.hidden = !n;
}

function updateStatusLine() {
  const h = state.health;
  if (!h) return;
  // The header shows the model and, only when something is actually wrong, a
  // warning. Everything else (tool count, retrieval mode, documents) lives in
  // Settings — a chat window should not read like a dashboard.
  if (!h.llm?.configured) {
    el.statusLine.textContent = "demo mode — add a key in Settings";
    return;
  }
  const model = (h.models || []).find((m) => m.id === state.settings.model);
  const bits = [];
  bits.push(model && state.settings.model !== "auto" ? model.label : "auto");
  if (h.llm?.free || h.llm?.free_tier) bits.push("free");
  el.statusLine.textContent = bits.join(" · ");
  el.modeBadge.hidden = true;
}

function updateBanner() {
  const h = state.health;
  if (!h) return;
  const hide = () => { el.banner.hidden = true; };
  if (h.llm?.configured && (h.llm?.free || h.llm?.free_tier)) {
    // Informative, not promotional: it explains why nothing is billed.
    el.banner.hidden = false;
    el.bannerText.innerHTML = h.llm?.free
      ? "Free mode — no account, no card, nothing billed."
      : "Free tier — no card, nothing billed on this plan.";
    el.bannerAction.hidden = true;
    return;
  }
  if (!h.llm?.configured) {
    el.banner.hidden = false;
    el.bannerText.innerHTML = "Demo mode — retrieval and tools work; answers need a model key.";
    el.bannerAction.hidden = false;
    el.bannerAction.textContent = "Add key";
    el.bannerAction.onclick = openSettings;
    return;
  }
  if (!h.embeddings?.semantic && h.stats?.documents > 0) {
    el.banner.hidden = false;
    el.bannerText.innerHTML = "Keyword-only search — the embedding model isn't available.";
    el.bannerAction.hidden = false;
    el.bannerAction.textContent = "Details";
    el.bannerAction.onclick = openSettings;
    return;
  }
  hide();
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    state.health = h;
    state.settings = { ...state.settings, ...(h.settings || {}) };
    state.docs = state.docs.length ? state.docs : [];
    updateStatusLine();
    updateBanner();
    applySettingsToUi();
    renderFreePanel();
    return h;
  } catch (e) {
    el.statusLine.innerHTML = `<span class="sdot warn"></span>Server unreachable`;
    throw e;
  }
}

/* ─────────────────────────── suggestions ───────────────────────── */
async function loadSuggestions(force = false) {
  try {
    const data = await api(`/api/suggestions${force ? "?force=true" : ""}`);
    const qs = data.questions || [];
    if (qs.length && state.docs.length) {
      const toolish = /weather|temperature|^what is [\d\s\.,%*+\-\/]+\?*$|time is it|calculate/i;
      el.starterGrid.innerHTML = qs.slice(0, 4).map((q) =>
        `<button class="starter-card"><span class="q">${esc(q)}</span>` +
        `<span class="tag">${toolish.test(q) ? "live tool" : "grounded &amp; cited"}</span></button>`).join("");
      $("#starter").hidden = false;
      el.starterGrid.onclick = (e) => {
        const card = e.target.closest(".starter-card");
        if (!card) return;
        const q = card.querySelector(".q")?.textContent.trim() || card.firstChild.textContent.trim();
        send(q);
      };
    } else {
      // no documents yet: offer the four things this assistant is actually good at
      const ideas = [
        ["Search the web for today's load-shedding schedule in Rustenburg", "live web"],
        ["Write and run a Python script that charts my CSV", "code workspace"],
        ["Make a 1024px poster for a coffee pop-up called First Light", "image"],
        ["Explain compound interest like I'm 15", "explanation"],
      ];
      el.starterGrid.innerHTML = ideas.map(([q, tag]) =>
        `<button class="starter-card"><span class="q">${esc(q)}</span>` +
        `<span class="tag">${esc(tag)}</span></button>`).join("");
      $("#starter").hidden = false;
      el.starterGrid.onclick = (e) => {
        const card = e.target.closest(".starter-card");
        if (card) send(card.querySelector(".q").textContent.trim());
      };
    }
    $("#heroDemo").hidden = false;
    const noDocs = state.docs.length === 0;
    $("#heroDemo").hidden = !noDocs;
    $("#heroUpload").innerHTML = noDocs
      ? `<svg viewBox="0 0 24 24" class="i"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></svg> Upload documents`
      : `<svg viewBox="0 0 24 24" class="i"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></svg> Add documents`;
    $("#emptySub").textContent = noDocs
      ? "Ask anything: it searches the live web, reads your files, writes and runs code, and makes images, audio and video."
      : `The live web plus ${state.docs.length} document${state.docs.length > 1 ? "s" : ""} ` +
        `(${fmtNum(state.docs.reduce((a, d) => a + (d.n_chunks || 0), 0))} passages) are both in play. Ask anything.`;
  } catch {}
}

/* ─────────────────────────── theme / ui ────────────────────────── */
function applyTheme() {
  const t = state.theme === "system"
    ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
    : state.theme;
  document.documentElement.dataset.theme = t;
}
function cycleTheme() {
  const effective = document.documentElement.dataset.theme;
  state.theme = effective === "dark" ? "system" : "dark";
  localStorage.setItem("cortex-theme", state.theme);
  applyTheme();
  toast(state.theme === "system" ? "Theme: follow system" : `Theme: ${state.theme}`);
}

function autoGrow() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 200) + "px";
}

function openSidebar() { $("#sidebar").classList.add("open"); el.scrim.hidden = false; }
function closeSidebar() {
  const sb = $("#sidebar");
  if (sb.classList.contains("open")) { sb.classList.remove("open"); if (el.drawer.hidden) el.scrim.hidden = true; }
}

/* ─────────────────────────── wiring ────────────────────────────── */
function wire() {
  el.composer.onsubmit = (e) => { e.preventDefault(); send(el.input.value); };
  el.input.addEventListener("input", autoGrow);
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send(el.input.value);
    }
  });
  el.stopBtn.onclick = () => { state.abort?.abort(); };

  $("#newChat").onclick = newChat;
  $("#menuBtn").onclick = openSidebar;
  $("#sidebarClose").onclick = closeSidebar;
  $("#themeBtn").onclick = cycleTheme;

  $("#openKb").onclick = () => openDrawer("docs");
  $("#kbBtn").onclick = () => openDrawer("docs");
  $("#drawerClose").onclick = closeDrawer;
  el.scrim.onclick = () => { closeDrawer(); closeSidebar(); };

  $$(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
  $("#browseBtn").onclick = () => el.fileInput.click();
  el.fileInput.onchange = () => uploadFiles(el.fileInput.files);

  el.dropzone.addEventListener("dragover", (e) => { e.preventDefault(); el.dropzone.classList.add("drag"); });
  el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("drag"));
  el.dropzone.addEventListener("drop", (e) => {
    e.preventDefault(); el.dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length && !e.target.closest("#dropzone")) {
      e.preventDefault(); uploadFiles(e.dataTransfer.files);
    }
  });

  el.docList.onclick = (e) => {
    const del = e.target.closest("[data-del]");
    if (del) return deleteDoc(del.dataset.del);
    const view = e.target.closest("[data-view]");
    if (view) return viewChunks(view.dataset.view);
  };

  $("#demoBtn").onclick = loadDemo;
  $("#heroDemo").onclick = loadDemo;
  $("#heroUpload").onclick = () => { openDrawer("docs"); setTimeout(() => el.fileInput.click(), 220); };
  $("#reindexBtn").onclick = async () => {
    $("#reindexBtn").disabled = true;
    try {
      const r = await api("/api/reindex", { method: "POST" });
      toast(r.embedded ? `Embedded ${r.embedded} passages` : "Embedding model still unavailable", r.embedded ? "" : "err");
      await refreshHealth(); await refreshDocs();
    } catch (e) { toast(e.message, "err"); }
    finally { $("#reindexBtn").disabled = false; }
  };

  $("#probeBtn").onclick = probe;
  $("#probeInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); probe(); } });
  $("#debugBtn").onclick = () => openDrawer("retrieval");

  $("#addSecret").onclick = () => {
    if (!toolsState) return;
    const existing = Object.entries(toolsState.config.secrets || {}).map(([h, c]) => [h, c]);
    saveSecrets([...existing, ["", { header: "Authorization", value: "" }]]);
  };
  $("#allowlist").onchange = async (e) => {
    const hosts = e.target.value.split(",").map((x) => x.trim()).filter(Boolean);
    try {
      const patch = await api("/api/tools", { method: "PUT", body: JSON.stringify({ http_allowlist: hosts }) });
      toolsState = { ...toolsState, ...patch };
      toast(hosts.length ? `Restricted to ${hosts.length} host(s)` : "Fetching from any public host");
      e.target.value = hosts.join(", ");
    } catch (err) { toast(err.message, "err"); }
  };
  $("#openSettings").onclick = openSettings;
  $("#settingsClose").onclick = closeSettings;
  el.settingsModal.addEventListener("click", (e) => { if (e.target === el.settingsModal) closeSettings(); });
  $("#saveKey").onclick = saveKey;
  $("#knowledgeSelect").onchange = (e) => setKnowledge(e.target.value);
  $("#knowledgeChip").onclick = cycleKnowledge;
  $("#modelBtn").onclick = (e) => {
    e.stopPropagation();
    const menu = $("#modelMenu");
    menu.hidden = !menu.hidden;
    if (!menu.hidden) {
      renderModelMenu($("#modelFilter").value);
      placePopmenu(menu, $("#modelBtn"));
      $("#modelFilter").focus();
    }
  };
  $("#modelFilter").oninput = (e) => renderModelMenu(e.target.value);
  document.addEventListener("click", (e) => {
    const menu = $("#modelMenu");
    if (menu && !menu.hidden && !menu.contains(e.target) && e.target.id !== "modelBtn") menu.hidden = true;
  });
  $("#wsBtn").onclick = () => { $("#wsDrawer").hidden = false; loadWorkspace(); };
  $("#wsClose").onclick = () => { $("#wsDrawer").hidden = true; };
  $("#wsDownload").onclick = () => {
    if (!state.convId) return toast("No conversation yet", "err");
    window.open(`/api/workspace/${state.convId}/download`, "_blank");
  };
  $("#orgInput").onchange = (e) => saveSetting({ org: e.target.value.trim() || "your organisation" });
  $("#strictToggle").onchange = (e) => saveSetting({ strict: e.target.checked });
  $("#sourcesToggle").onchange = (e) => {
    saveSetting({ show_sources: e.target.checked });
    state.messages.forEach(() => {});
    $$(".msg.assistant").forEach((w) => { if (e.target.checked) renderSources(w); else w.querySelector(".sources")?.remove(); });
  };
  $("#minRel").oninput = (e) => { $("#minRelVal").textContent = `(${(+e.target.value).toFixed(2)})`; updateRangeFill(); };
  $("#minRel").onchange = (e) => saveSetting({ min_relevance: +e.target.value });
  $("#wipeDocs").onclick = async () => {
    if (!confirm("Delete ALL documents and embeddings? This cannot be undone.")) return;
    try {
      for (const d of state.docs) await api(`/api/documents/${d.id}`, { method: "DELETE" });
      await refreshDocs(); await refreshHealth(); loadSuggestions(true);
      toast("Knowledge base cleared");
    } catch (e) { toast(e.message, "err"); }
  };

  el.strictChip.onclick = () => saveSetting({ strict: !state.settings.strict });
  $("#scopeChip").onclick = (e) => {
    e.stopPropagation();
    if (state.scopeOpen) closeScopeMenu(); else openScopeMenu();
  };
  $("#scopeAll").onclick = () => {
    state.drafts = new Set(state.docs.map((d) => d.id));
    $("#scopeList").querySelectorAll("input").forEach((i) => (i.checked = true));
    $("#scopeCount").textContent = `${state.drafts.size} selected`;
  };
  $("#scopeApply").onclick = applyScope;
  document.addEventListener("click", (e) => {
    if (state.scopeOpen && !e.target.closest("#scopeMenu") && !e.target.closest("#scopeChip")) closeScopeMenu();
  });

  $("#exportBtn").onclick = () => {
    if (!state.convId) return toast("Nothing to export yet", "err");
    window.open(`/api/conversations/${state.convId}/export`, "_blank");
  };

  $("#bannerClose").onclick = () => { el.banner.hidden = true; };
  $("#popClose").onclick = () => { el.popover.hidden = true; };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      el.popover.hidden = true;
      if (!el.settingsModal.hidden) closeSettings();
      else if (!el.drawer.hidden) closeDrawer();
      else closeSidebar();
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); newChat();
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "/") {
      e.preventDefault(); el.input.focus();
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".cite")) {
      if (!e.target.closest("#popover") && !e.target.closest(".src-chip")) el.popover.hidden = true;
    }
  });

  el.messages.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const wrap = cite.closest(".msg");
    const idx = +cite.dataset.n - 1;
    if (!wrap || !wrap._sources || !wrap._sources[idx]) return toast("That citation isn't in this answer's sources", "err");
    openSource(wrap, idx, cite.getBoundingClientRect());
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.theme === "system") applyTheme();
  });
}

/* ─────────────────────────── boot ──────────────────────────────── */
/* ─────────────────────── install on phone / laptop ──────────────────── */
let deferredInstall = null;

function standalone() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

function showInstall(show) {
  const b = $("#installBtn");
  if (b) b.hidden = !show;
}

async function installApp() {
  if (standalone()) return;
  if (deferredInstall) {
    deferredInstall.prompt();
    try {
      const { outcome } = await deferredInstall.userChoice;
      if (outcome === "accepted") { showInstall(false); toast("Installing Cortex…"); }
    } catch { /* dismissed */ }
    deferredInstall = null;
    return;
  }
  const ua = navigator.userAgent;
  const ios = /iphone|ipad|ipod/i.test(ua);
  const macSafari = /macintosh/i.test(ua) && /safari/i.test(ua) && !/chrome|chromium|edg/i.test(ua);
  toast(ios || macSafari
    ? "Tap Share, then “Add to Home Screen” (or “Add to Dock”)."
    : "Open your browser menu and choose “Install app” / “Add to Home screen”.");
}

function wireInstall() {
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredInstall = e;
    showInstall(true);
  });
  window.addEventListener("appinstalled", () => { showInstall(false); deferredInstall = null; });
  const btn = $("#installBtn");
  if (btn) btn.onclick = installApp;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  if (!standalone() && ios) showInstall(true);        // iOS fires no install event
  if (standalone()) showInstall(false);
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
}

async function boot() {
  state.theme = localStorage.getItem("cortex-theme") || "system";
  try { state.docScope = JSON.parse(localStorage.getItem("cortex-scope") || "null"); } catch { state.docScope = null; }
  applyTheme();
  wire();
  wireInstall();
  try {
    await refreshHealth();
    await refreshDocs();
    await refreshConversations();
    await loadSuggestions();
    await loadModels();
    setKnowledge(state.settings.knowledge || "web+docs");
    const latest = state.conversations[0];
    if (latest) await openConversation(latest.id);
    updateStatusLine();
    updateBanner();
    renderDiag();
  } catch (e) {
    console.error(e);
    el.statusLine.innerHTML = `<span class="sdot warn"></span>Backend unreachable — is the server running?`;
  }
  setInterval(refreshHealth, 60000);
  el.input.focus();
}

boot();
