/* ============================================================
   Theme Toggle
   ============================================================ */

const THEME_KEY = "poc_theme";

function getSystemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function setTheme(theme) {
  const effective = theme === "system" ? getSystemTheme() : theme;
  document.documentElement.classList.add("theme-transitioning");
  document.documentElement.setAttribute("data-theme", effective);
  setTimeout(() => document.documentElement.classList.remove("theme-transitioning"), 300);

  if (theme === "system") localStorage.removeItem(THEME_KEY);
  else localStorage.setItem(THEME_KEY, theme);
}

function initTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  setTheme(stored || "system");
}

initTheme();

document.getElementById("theme-toggle")?.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  setTheme(current === "dark" ? "light" : "dark");
});

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!localStorage.getItem(THEME_KEY)) setTheme("system");
});

/* ============================================================
   Utility Functions
   ============================================================ */

function formatCurrency(num) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(num);
}

function formatDate(isoString) {
  if (!isoString) return "";
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoString);
  let d;
  if (match) {
    const year = Number(match[1]);
    const month = Number(match[2]) - 1;
    const day = Number(match[3]);
    d = new Date(year, month, day);
  } else {
    d = new Date(isoString);
  }
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatPct(num) {
  const sign = num >= 0 ? "+" : "";
  return `${sign}${num.toFixed(2)}%`;
}

function pctClass(num) {
  return num >= 0 ? "positive" : "negative";
}

/* ============================================================
   DOM References
   ============================================================ */

const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const welcomeState = document.getElementById("welcome-state");
const typingIndicator = document.getElementById("typing-indicator");
const routingButton = document.getElementById("routing-button");
const routingModal = document.getElementById("routing-modal");
const routingDetailButton = document.getElementById("routing-detail-button");
const routingPanel = document.querySelector(".routing-panel");
const routingClose = document.getElementById("routing-close");
const routingExplain = document.getElementById("routing-explain");
const routingTeachingText = document.getElementById("routing-teaching-text");
const routingFields = {
  utterance: document.getElementById("routing-utterance"),
  intent: document.getElementById("routing-intent"),
  mode: document.getElementById("routing-mode"),
  confidence: document.getElementById("routing-confidence"),
  router: document.getElementById("routing-router"),
  routerDiagnostics: document.getElementById("routing-router-diagnostics"),
  extracted: document.getElementById("routing-extracted"),
  missing: document.getElementById("routing-missing"),
  candidates: document.getElementById("routing-candidates"),
  tools: document.getElementById("routing-tools"),
  toolParams: document.getElementById("routing-tool-params"),
  context: document.getElementById("routing-context"),
  latency: document.getElementById("routing-latency"),
  latencyBreakdown: document.getElementById("routing-latency-breakdown"),
  grounding: document.getElementById("routing-grounding"),
  groundingValid: document.getElementById("routing-grounding-valid"),
  formatter: document.getElementById("routing-formatter"),
  session: document.getElementById("routing-session"),
  policy: document.getElementById("routing-policy"),
  model: document.getElementById("routing-model"),
  prompt: document.getElementById("routing-prompt"),
  tokens: document.getElementById("routing-tokens"),
  retry: document.getElementById("routing-retry"),
  traceLink: document.getElementById("routing-trace-link"),
  summary: document.getElementById("routing-summary"),
};
let lastTraceId = null;
const STREAMING_ENABLED = false;
const sessionKey = "poc_session_id";
let sessionId = localStorage.getItem(sessionKey);
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem(sessionKey, sessionId);
}

/* ============================================================
   Send Button State
   ============================================================ */

function updateSendBtn() {
  if (sendBtn) {
    sendBtn.disabled = !input.value.trim();
  }
}

input.addEventListener("input", updateSendBtn);
updateSendBtn();

/* ============================================================
   Welcome State
   ============================================================ */

function removeWelcome() {
  if (welcomeState && welcomeState.parentNode) {
    welcomeState.remove();
  }
}

/* Suggestion chip click handlers */
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const query = chip.dataset.query;
    if (query) {
      input.value = query;
      updateSendBtn();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
});

/* ============================================================
   Typing Indicator
   ============================================================ */

function showTyping() {
  if (typingIndicator) {
    typingIndicator.classList.add("visible");
    chat.scrollTop = chat.scrollHeight;
  }
}

function hideTyping() {
  if (typingIndicator) {
    typingIndicator.classList.remove("visible");
  }
}

/* ============================================================
   Routing Modal & Panel
   ============================================================ */

function toggleRoutingModal(show) {
  if (!routingModal) return;
  routingModal.classList.toggle("hidden", !show);
  document.body.style.overflow = show ? "hidden" : "";
}

function toggleRoutingPanel(show) {
  if (!routingPanel) return;
  document.body.classList.toggle("routing-open", show);
}

if (routingButton && routingModal) {
  routingButton.addEventListener("click", () => toggleRoutingModal(true));
  routingModal.addEventListener("click", (event) => {
    const target = event.target;
    if (target?.dataset?.close) {
      toggleRoutingModal(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      toggleRoutingModal(false);
    }
  });
}

if (routingDetailButton && routingPanel) {
  routingDetailButton.addEventListener("click", () => toggleRoutingPanel(true));
}

if (routingClose && routingPanel) {
  routingClose.addEventListener("click", () => toggleRoutingPanel(false));
}

/* ============================================================
   Chat Messages (iMessage-style bubbles)
   ============================================================ */

function addMessage(role, content, citations = []) {
  removeWelcome();

  const isUser = role.toLowerCase() === "you";
  const wrapper = document.createElement("div");
  wrapper.className = `message ${isUser ? "user" : "assistant"}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(roleEl);
  wrapper.appendChild(bubble);

  if (citations.length) {
    const citeEl = document.createElement("div");
    citeEl.className = "citations";
    citeEl.textContent = `Sources: ${citations.join(", ")}`;
    wrapper.appendChild(citeEl);
  }

  // Insert before typing indicator
  if (typingIndicator && typingIndicator.parentNode === chat) {
    chat.insertBefore(wrapper, typingIndicator);
  } else {
    chat.appendChild(wrapper);
  }
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

/* ============================================================
   Streaming Chat
   ============================================================ */

async function streamChat(utterance) {
  const apiBase = window.__API_BASE__ || "";
  const response = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance, stream: true, session_id: sessionId }),
  });

  const isSse = response.headers.get("content-type")?.includes("text/event-stream");
  if (!response.ok || !response.body || !isSse) {
    return fallbackChat(utterance);
  }

  // Create assistant bubble on first content
  let contentEl = null;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let data = "";

  function dispatchEvent(type, payload) {
    if (type === "delta") {
      if (!contentEl) {
        contentEl = addMessage("Assistant", "");
      }
      contentEl.textContent += payload.replace(/\\n/g, "\n");
      chat.scrollTop = chat.scrollHeight;
      return;
    }

    if (type === "final") {
      try {
        const parsed = JSON.parse(payload.trim());
        if (!contentEl) {
          contentEl = addMessage("Assistant", parsed.answer_markdown || "");
        } else if (!contentEl.textContent && parsed.answer_markdown) {
          contentEl.textContent = parsed.answer_markdown;
        }
        if (parsed.citations && parsed.citations.length) {
          const citationsEl = document.createElement("div");
          citationsEl.className = "citations";
          citationsEl.textContent = `Sources: ${parsed.citations.join(", ")}`;
          contentEl.parentElement.appendChild(citationsEl);
        }
        if (parsed.needs_clarification && parsed.clarifying_question) {
          const clarifyEl = document.createElement("div");
          clarifyEl.className = "meta";
          clarifyEl.textContent = `Clarifying question: ${parsed.clarifying_question}`;
          contentEl.parentElement.appendChild(clarifyEl);
        }
        if (parsed.trace_id) {
          lastTraceId = parsed.trace_id;
          const traceEl = document.createElement("div");
          traceEl.className = "meta";
          const link = document.createElement("a");
          link.href = `/debug/trace/${parsed.trace_id}`;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = `Trace: ${parsed.trace_id}`;
          traceEl.appendChild(link);
          contentEl.parentElement.appendChild(traceEl);
        }
      } catch (err) {
        console.error("Failed to parse final payload", err);
      }
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let lineEnd = buffer.indexOf("\n");
    while (lineEnd !== -1) {
      const line = buffer.slice(0, lineEnd);
      buffer = buffer.slice(lineEnd + 1);

      if (line === "") {
        if (data) {
          dispatchEvent(eventType || "message", data.trim());
        }
        eventType = "";
        data = "";
      } else if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        let chunk = line.slice(5);
        if (chunk.startsWith(" ")) chunk = chunk.slice(1);
        data += chunk + "\n";
      }

      lineEnd = buffer.indexOf("\n");
    }
  }

  if (data.trim().length) {
    dispatchEvent(eventType || "message", data.trim());
  }
}

/* ============================================================
   Fallback (non-streaming) Chat
   ============================================================ */

async function fallbackChat(utterance) {
  const apiBase = window.__API_BASE__ || "";
  const response = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance, stream: false, session_id: sessionId }),
  });

  if (!response.ok) {
    addMessage("Assistant", "Error contacting server.");
    return;
  }

  const payload = await response.json();
  const contentEl = addMessage("Assistant", payload.answer_markdown || "");

  if (payload.citations && payload.citations.length) {
    const citationsEl = document.createElement("div");
    citationsEl.className = "citations";
    citationsEl.textContent = `Sources: ${payload.citations.join(", ")}`;
    contentEl.parentElement.appendChild(citationsEl);
  }
  if (payload.trace_id) {
    lastTraceId = payload.trace_id;
    const traceEl = document.createElement("div");
    traceEl.className = "meta";
    const link = document.createElement("a");
    link.href = `/debug/trace/${payload.trace_id}`;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `Trace: ${payload.trace_id}`;
    traceEl.appendChild(link);
    contentEl.parentElement.appendChild(traceEl);
    try {
      const apiBase = window.__API_BASE__ || "";
      const trace = await fetch(`${apiBase}/debug/trace/${payload.trace_id}`).then((res) => res.json());
      updateRoutingDrawer(trace);
    } catch (err) {
      console.warn("Failed to load trace for routing drawer", err);
    }
  }
}

/* ============================================================
   Explain Button
   ============================================================ */

if (routingExplain) {
  routingExplain.addEventListener("click", async () => {
    if (!lastTraceId) {
      if (routingTeachingText) {
        routingTeachingText.textContent = "Run a query first to generate an explanation.";
      }
      return;
    }
    if (routingTeachingText) {
      routingTeachingText.textContent = "Generating explanation...";
    }
    try {
      const apiBase = window.__API_BASE__ || "";
      const response = await fetch(`${apiBase}/debug/trace/${lastTraceId}/explain`, {
        method: "POST",
      });
      const payload = await response.json();
      if (routingTeachingText) {
        routingTeachingText.textContent = payload.explanation || "No explanation returned.";
      }
    } catch (err) {
      if (routingTeachingText) {
        routingTeachingText.textContent = "Failed to generate explanation.";
      }
    }
  });
}

/* ============================================================
   Routing Drawer Update
   ============================================================ */

function updateRoutingDrawer(trace) {
  if (!trace || !routingPanel) return;
  lastTraceId = trace.trace_id || lastTraceId;
  const emptyListItem = () => {
    const li = document.createElement("li");
    li.textContent = "\u2014";
    return li;
  };

  const fillList = (el, items) => {
    if (!el) return;
    el.innerHTML = "";
    if (!items || !items.length) {
      el.appendChild(emptyListItem());
      return;
    }
    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      el.appendChild(li);
    });
  };

  const keyValueList = (obj) => {
    if (!obj || typeof obj !== "object") return [];
    return Object.entries(obj).map(([key, value]) => `${key}: ${JSON.stringify(value)}`);
  };

  const toggleSection = (key, isEmpty) => {
    const section = document.querySelector(`[data-routing-section="${key}"]`);
    if (!section) return;
    section.style.display = isEmpty ? "none" : "block";
  };

  routingFields.utterance.textContent = trace.utterance || "\u2014";
  const mode = trace.routing_mode || "";
  const confidence = typeof trace.routing_confidence === "number" ? trace.routing_confidence : null;
  const modeLabel = mode === "torch_classifier"
    ? "PyTorch intent router"
    : mode === "llm_reroute"
      ? "LLM reroute"
      : mode === "rules"
        ? "Rules-based router"
        : mode || "\u2014";

  routingFields.intent.textContent = trace.intent || "\u2014";
  routingFields.mode.textContent = modeLabel;
  routingFields.confidence.textContent = confidence !== null ? `Confidence ${confidence.toFixed(2)}` : "\u2014";
  if (routingFields.router) {
    routingFields.router.textContent =
      mode === "torch_classifier"
        ? "Router model: torch intent classifier (bag-of-words)"
        : mode === "llm_reroute"
          ? "Router model: LLM fallback"
          : mode === "rules"
            ? "Router model: deterministic rules"
            : "\u2014";
  }

  const extractedItems = keyValueList(trace.routing_extracted);
  fillList(routingFields.extracted, extractedItems);
  toggleSection("extracted", extractedItems.length === 0);

  const missingItems = trace.routing_missing_params || [];
  routingFields.missing.textContent = missingItems.length ? missingItems.join(", ") : "\u2014";
  toggleSection("missing", missingItems.length === 0);

  const candidateItems = (trace.routing_candidates || []).map(
    (candidate) => `${candidate.intent}: ${candidate.score}`
  );
  fillList(routingFields.candidates, candidateItems);
  toggleSection("candidates", candidateItems.length === 0);

  const toolItems = (trace.tool_calls || []).map((tool, index) => {
    const latency = trace.tool_latency_ms?.[index];
    return `${tool.name} (${tool.source_id})${latency !== undefined ? ` \u2014 ${latency}ms` : ""}`;
  });
  fillList(routingFields.tools, toolItems);
  toggleSection("tools", toolItems.length === 0);

  const toolParamItems = (trace.tool_params || []).map(
    (tool) => `${tool.name}: ${JSON.stringify(tool.params)}`
  );
  fillList(routingFields.toolParams, toolParamItems);
  toggleSection("tool-params", toolParamItems.length === 0);

  const contextItems = keyValueList(trace.context_summary);
  fillList(routingFields.context, contextItems);
  toggleSection("context", contextItems.length === 0);

  const routerDiagItems = keyValueList(trace.router_diagnostics);
  fillList(routingFields.routerDiagnostics, routerDiagItems);
  toggleSection("router-diagnostics", routerDiagItems.length === 0);

  if (trace.tool_latency_ms && trace.tool_latency_ms.length) {
    const total = trace.tool_latency_ms.reduce((acc, v) => acc + v, 0);
    routingFields.latency.textContent = `${total}ms total`;
  } else {
    routingFields.latency.textContent = "\u2014";
  }

  const latencyItems = trace.latency_ms
    ? [
        `routing: ${trace.latency_ms.routing_ms ?? 0}ms`,
        `tools: ${trace.latency_ms.tool_ms ?? 0}ms`,
        `llm: ${trace.latency_ms.llm_ms ?? 0}ms`,
        `postprocess: ${trace.latency_ms.postprocess_ms ?? 0}ms`,
        `total: ${trace.latency_ms.total_ms ?? 0}ms`,
      ]
    : [];
  fillList(routingFields.latencyBreakdown, latencyItems);
  toggleSection("latency-breakdown", latencyItems.length === 0);

  routingFields.grounding.textContent = trace.grounded_sources?.length
    ? `${trace.grounded_sources.join(", ")} (rate ${trace.grounding_rate})`
    : "\u2014";
  toggleSection("grounding", !(trace.grounded_sources || []).length);
  if (routingFields.groundingValid) {
    if (trace.grounding_valid === true) {
      routingFields.groundingValid.textContent = "\u2705 citations match sources";
      toggleSection("grounding-valid", false);
    } else if (trace.grounding_valid === false) {
      routingFields.groundingValid.textContent = "\u274c citations missing or mismatched";
      toggleSection("grounding-valid", false);
    } else {
      routingFields.groundingValid.textContent = "\u2014";
      toggleSection("grounding-valid", true);
    }
  }

  if (routingFields.formatter) {
    routingFields.formatter.textContent = trace.formatter_used || "\u2014";
    toggleSection("formatter", !trace.formatter_used || trace.formatter_used === "none");
  }

  const sessionItems = keyValueList(trace.session_state);
  fillList(routingFields.session, sessionItems);
  toggleSection("session", sessionItems.length === 0);

  if (routingFields.policy) {
    if (trace.policy_gate && Object.keys(trace.policy_gate).length) {
      routingFields.policy.textContent = `allowed=${trace.policy_gate.allowed} \u00b7 reason=${trace.policy_gate.reason} \u00b7 tools=${(trace.policy_gate.allowed_tools || []).join(", ")}`;
      toggleSection("policy", false);
    } else {
      routingFields.policy.textContent = "\u2014";
      toggleSection("policy", true);
    }
  }

  routingFields.model.textContent = `${trace.model_version || "\u2014"} / ${trace.prompt_version || "\u2014"}`;
  if (routingFields.prompt) {
    routingFields.prompt.textContent = trace.prompt_name || "\u2014";
    toggleSection("prompt", !trace.prompt_name);
  }
  routingFields.tokens.textContent = `Prompt ${trace.prompt_tokens ?? 0} \u00b7 Completion ${trace.completion_tokens ?? 0} \u00b7 $${trace.estimated_cost_usd ?? 0}`;
  if (routingFields.retry) {
    routingFields.retry.textContent = `${trace.retry_count ?? 0}`;
    toggleSection("retry", !trace.retry_count || trace.retry_count === 0);
  }

  if (routingFields.summary) {
    const toolNames = (trace.tool_calls || []).map((tool) => tool.name).join(", ") || "no tools";
    const formatter = trace.formatter_used && trace.formatter_used !== "none" ? trace.formatter_used : "no formatter";
    const grounding = trace.grounding_valid === true ? "grounding \u2705" : trace.grounding_valid === false ? "grounding \u274c" : "grounding \u2014";
    const routerSummary = mode ? `${modeLabel}${confidence !== null ? ` (${confidence.toFixed(2)})` : ""}` : "router \u2014";
    routingFields.summary.textContent = `${routerSummary} \u2192 ${trace.intent || "intent \u2014"} \u2192 ${toolNames} \u2192 ${formatter} \u2192 ${grounding}`;
  }

  if (routingFields.traceLink) {
    routingFields.traceLink.href = `/debug/trace/${trace.trace_id}`;
    routingFields.traceLink.textContent = `Open trace ${trace.trace_id}`;
  }
}

/* ============================================================
   Chat Submit Handler
   ============================================================ */

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const utterance = input.value.trim();
  if (!utterance) return;

  addMessage("You", utterance);

  input.value = "";
  updateSendBtn();
  input.focus();

  showTyping();

  try {
    if (STREAMING_ENABLED) {
      await streamChat(utterance);
    } else {
      await fallbackChat(utterance);
    }
  } catch (err) {
    addMessage("Assistant", "Something went wrong.");
  }

  hideTyping();
});

/* ============================================================
   Sidebar Data Loaders
   ============================================================ */

// SVG icon helpers
const arrowUpSvg = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z"/></svg>';
const arrowDownSvg = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M20 12l-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8z"/></svg>';

async function loadAccountSummary() {
  try {
    const apiBase = window.__API_BASE__ || "";
    const res = await fetch(`${apiBase}/api/account_summary`);
    const data = await res.json();
    const payload = data.data ?? data;
    const acct = payload.accounts?.[0];
    if (!acct) return;

    const balanceEl = document.getElementById("account-balance");
    const dateEl = document.getElementById("account-date");
    const investedEl = document.getElementById("account-invested");
    const cashEl = document.getElementById("account-cash");
    const settledEl = document.getElementById("account-settled");

    if (balanceEl) balanceEl.textContent = formatCurrency(acct.total_value);
    if (dateEl) dateEl.textContent = `As of ${formatDate(payload.as_of)}`;
    if (investedEl) investedEl.textContent = formatCurrency(acct.total_value - acct.total_cash);
    if (cashEl) cashEl.textContent = formatCurrency(acct.total_cash);
    if (settledEl) settledEl.textContent = formatCurrency(acct.settled_cash);
  } catch (err) {
    console.warn("Failed to load account summary", err);
  }
}

async function loadPerformanceBadge() {
  try {
    const apiBase = window.__API_BASE__ || "";
    const res = await fetch(`${apiBase}/api/performance?timeframe=YTD`);
    const data = await res.json();
    const payload = data.data ?? data;
    const badge = document.getElementById("ytd-badge");
    if (!badge) return;

    badge.className = `ytd-badge ${pctClass(payload.return_pct)}`;
    badge.textContent = `YTD ${formatPct(payload.return_pct)}`;
  } catch (err) {
    console.warn("Failed to load performance badge", err);
  }
}

async function loadPositions() {
  try {
    const apiBase = window.__API_BASE__ || "";
    const posRes = await fetch(`${apiBase}/api/positions_list`);
    const posData = await posRes.json();

    const posPayload = posData.data ?? posData;
    const positions = posPayload.positions || [];

    // Fetch quotes for each symbol in parallel
    const quoteResults = await Promise.all(
      positions.map((pos) =>
        fetch(`${apiBase}/api/quotes?symbol=${encodeURIComponent(pos.symbol)}`)
          .then((r) => r.json())
          .catch(() => null)
      )
    );

    const quoteLookup = {};
    quoteResults.forEach((data) => {
      const payload = data?.data ?? data;
      if (payload?.quotes) {
        payload.quotes.forEach((q) => { quoteLookup[q.symbol] = q; });
      }
    });

    const container = document.getElementById("positions-list");
    const countEl = document.getElementById("positions-count");
    if (!container) return;

    if (countEl) countEl.textContent = `${positions.length} holding${positions.length !== 1 ? "s" : ""}`;

    container.innerHTML = "";
    positions.forEach((pos) => {
      const quote = quoteLookup[pos.symbol];
      const currentPrice = quote?.price ?? pos.cost_basis;
      const gainPct = ((currentPrice - pos.cost_basis) / pos.cost_basis) * 100;

      const row = document.createElement("div");
      row.className = "position-row";
      row.innerHTML = `
        <div>
          <span class="position-symbol">${pos.symbol}</span>
          <span class="position-shares">${pos.quantity} shares</span>
        </div>
        <div class="position-right">
          <div class="position-price">${formatCurrency(currentPrice)}</div>
          <div class="position-change ${pctClass(gainPct)}">${formatPct(gainPct)}</div>
        </div>
      `;
      container.appendChild(row);
    });
  } catch (err) {
    console.warn("Failed to load positions", err);
  }
}

async function loadActivity() {
  try {
    const apiBase = window.__API_BASE__ || "";
    const res = await fetch(`${apiBase}/api/activity`);
    const data = await res.json();
    const payload = data.data ?? data;
    const container = document.getElementById("activity-list");
    if (!container) return;

    const trades = payload.trades || [];
    container.innerHTML = "";

    if (!trades.length) {
      container.innerHTML = '<div class="loading-placeholder">No recent activity</div>';
      return;
    }

    trades.forEach((trade) => {
      const isBuy = trade.side === "BUY";
      const row = document.createElement("div");
      row.className = "activity-row";
      row.innerHTML = `
        <div class="activity-icon ${isBuy ? "buy" : "sell"}">
          ${isBuy ? arrowUpSvg : arrowDownSvg}
        </div>
        <div class="activity-info">
          <div class="activity-action">${trade.side} ${trade.symbol}</div>
          <div class="activity-date">${formatDate(trade.timestamp)}</div>
        </div>
        <div class="activity-amounts">
          <div class="activity-qty">${trade.quantity} shares</div>
          <div class="activity-price">@ ${formatCurrency(trade.price)}</div>
        </div>
      `;
      container.appendChild(row);
    });
  } catch (err) {
    console.warn("Failed to load activity", err);
  }
}

async function loadTransfers() {
  try {
    const apiBase = window.__API_BASE__ || "";
    const res = await fetch(`${apiBase}/api/transfers`);
    const data = await res.json();
    const payload = data.data ?? data;
    const container = document.getElementById("transfers-list");
    if (!container) return;

    const transfers = payload.transfers || [];
    container.innerHTML = "";

    if (!transfers.length) {
      container.innerHTML = '<div class="loading-placeholder">No transfers</div>';
      return;
    }

    let netContributions = 0;

    transfers.forEach((t) => {
      const isDeposit = t.type === "deposit";
      if (t.status === "completed") {
        netContributions += isDeposit ? t.amount : -t.amount;
      }

      const row = document.createElement("div");
      row.className = "transfer-row";
      row.innerHTML = `
        <div class="transfer-icon ${isDeposit ? "deposit" : "withdrawal"}">
          ${isDeposit ? arrowDownSvg : arrowUpSvg}
        </div>
        <div class="transfer-info">
          <div class="transfer-type">${t.type}</div>
          <div class="transfer-date">${formatDate(t.timestamp)}</div>
        </div>
        <div class="transfer-right">
          <div class="transfer-amount">${formatCurrency(t.amount)}</div>
          <span class="status-pill ${t.status}">${t.status}</span>
        </div>
      `;
      container.appendChild(row);
    });

    // Net contributions row
    const netRow = document.createElement("div");
    netRow.className = "transfer-net";
    netRow.innerHTML = `
      <span class="transfer-net-label">Net Contributions (YTD)</span>
      <span class="transfer-net-value">${formatCurrency(netContributions)}</span>
    `;
    container.appendChild(netRow);
  } catch (err) {
    console.warn("Failed to load transfers", err);
  }
}

/* ============================================================
   Init: Load sidebar data on page load
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  loadAccountSummary();
  loadPerformanceBadge();
  loadPositions();
  loadActivity();
  loadTransfers();
});
