# Frontend Spec

> Single-page chat interface for the Personal Finance Assistant POC. iMessage-style bubbles, sidebar with account data, and a live routing trace drawer for debugging.

---

## Overview

The frontend is a vanilla JavaScript SPA (no build step, no framework) that provides:

1. **Chat interface** - iMessage-style conversation with the assistant
2. **Sidebar dashboard** - Account summary, positions, activity, transfers
3. **Routing trace drawer** - DevTools-style panel showing backend pipeline details
4. **Session persistence** - UUID stored in localStorage for multi-turn context

All data is fetched from the FastAPI backend via REST endpoints. The frontend is served statically at `/ui/*` by the backend.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `frontend/index.html` | 294 | HTML structure, semantic layout |
| `frontend/app.js` | 799 | Chat logic, sidebar loaders, routing drawer |
| `frontend/styles.css` | 1268 | Design system, iMessage styling, dark devtools panel |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Shell (1200px max)                    │
├───────────────────┬─────────────────────┬───────────────────────┤
│     Sidebar       │    Chat Panel       │   Routing Panel       │
│     (320px)       │    (flexible)       │   (360px, optional)   │
│                   │                     │                       │
│  - Account Card   │  - Welcome State    │  - Teaching Explain   │
│  - Positions      │  - Message Bubbles  │  - Intent/Routing     │
│  - Activity       │  - Typing Indicator │  - Tool Calls         │
│  - Transfers      │  - Input Form       │  - Latency/Tokens     │
└───────────────────┴─────────────────────┴───────────────────────┘
```

---

## Key Types and Models

### Session Management

```javascript
// frontend/app.js:72-77
const sessionKey = "poc_session_id";
let sessionId = localStorage.getItem(sessionKey);
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem(sessionKey, sessionId);
}
```

The session ID is sent with every `/chat` request to enable multi-turn context on the backend.

### Chat Request (sent to backend)

```javascript
// frontend/app.js:216-217, 313-314
body: JSON.stringify({ utterance, stream: true, session_id: sessionId })
body: JSON.stringify({ utterance, stream: false, session_id: sessionId })
```

Maps to `ChatRequest` in `backend/schemas.py:220-226`:

```python
class ChatRequest(BaseModel):
    utterance: str
    account: Optional[str] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
```

### Chat Response (from backend)

```python
# backend/schemas.py:229-237
class ChatResponse(BaseModel):
    answer_markdown: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    clarifying_question: Optional[str]
    trace_id: str
```

### DOM References Object

```javascript
// frontend/app.js:43-69
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
```

---

## Public API / Functions

### Utility Functions

```javascript
// frontend/app.js:5-24
function formatCurrency(num)  // Returns "$1,234.56" format
function formatDate(isoString)  // Returns "Jan 15, 2024" format
function formatPct(num)  // Returns "+12.34%" or "-5.67%" format
function pctClass(num)  // Returns "positive" or "negative" CSS class
```

### Chat Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `addMessage` | `(role, content, citations=[])` | Adds iMessage-style bubble to chat |
| `streamChat` | `async (utterance, contentEl)` | SSE streaming chat (disabled by default) |
| `fallbackChat` | `async (utterance, contentEl)` | Non-streaming JSON chat (default mode) |
| `showTyping` | `()` | Shows animated typing indicator |
| `hideTyping` | `()` | Hides typing indicator |
| `removeWelcome` | `()` | Removes welcome state on first message |
| `updateSendBtn` | `()` | Enables/disables send button based on input |

### Sidebar Data Loaders

| Function | Endpoint | Updates |
|----------|----------|---------|
| `loadAccountSummary` | `GET /api/account_summary` | Account card (balance, invested, cash) |
| `loadPerformanceBadge` | `GET /api/performance?timeframe=YTD` | YTD return badge |
| `loadPositions` | `GET /api/positions_list` + `GET /api/quotes?symbol=X` | Positions list with live prices |
| `loadActivity` | `GET /api/activity` | Recent trades (buy/sell) |
| `loadTransfers` | `GET /api/transfers` | Deposits/withdrawals + net contributions |

### Routing Drawer Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `updateRoutingDrawer` | `(trace)` | Populates all routing panel fields from trace JSON |
| `toggleRoutingModal` | `(show)` | Shows/hides the scale-view modal |
| `toggleRoutingPanel` | `(show)` | Opens/closes the routing drawer sidebar |

---

## Internal Patterns

### Message Rendering

```javascript
// frontend/app.js:173-206
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
```

### Streaming vs Fallback Chat

The frontend supports both SSE streaming and non-streaming modes, controlled by a constant:

```javascript
// frontend/app.js:71
const STREAMING_ENABLED = false;
```

**Streaming mode** (`streamChat`):
- Sends `stream: true` in request
- Parses SSE events: `delta` (incremental text), `final` (complete response)
- Updates bubble text incrementally

**Fallback mode** (`fallbackChat`):
- Sends `stream: false` in request
- Waits for complete JSON response
- Sets bubble text all at once
- Fetches trace data for routing drawer

```javascript
// frontend/app.js:310-348
async function fallbackChat(utterance, contentEl) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance, stream: false, session_id: sessionId }),
  });

  if (!response.ok) {
    contentEl.textContent = "Error contacting server.";
    return;
  }

  const payload = await response.json();
  contentEl.textContent = payload.answer_markdown || "";

  if (payload.citations && payload.citations.length) {
    const citationsEl = document.createElement("div");
    citationsEl.className = "citations";
    citationsEl.textContent = `Sources: ${payload.citations.join(", ")}`;
    contentEl.parentElement.appendChild(citationsEl);
  }

  if (payload.trace_id) {
    lastTraceId = payload.trace_id;
    // ... add trace link, fetch trace for drawer
    const trace = await fetch(`/debug/trace/${payload.trace_id}`).then((res) => res.json());
    updateRoutingDrawer(trace);
  }
}
```

### Routing Drawer Population

The `updateRoutingDrawer` function maps trace fields to DOM elements:

```javascript
// frontend/app.js:385-557
function updateRoutingDrawer(trace) {
  // Helper to fill <ul> lists
  const fillList = (el, items) => {
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

  // Helper to toggle section visibility
  const toggleSection = (key, isEmpty) => {
    const section = document.querySelector(`[data-routing-section="${key}"]`);
    if (!section) return;
    section.style.display = isEmpty ? "none" : "block";
  };

  // Map routing mode to human-readable label
  const modeLabel = mode === "torch_classifier"
    ? "PyTorch intent router"
    : mode === "llm_reroute"
      ? "LLM reroute"
      : mode === "rules"
        ? "Rules-based router"
        : mode || "—";

  // ... populate all fields
  routingFields.summary.textContent =
    `${routerSummary} → ${trace.intent || "intent —"} → ${toolNames} → ${formatter} → ${grounding}`;
}
```

### Sidebar Data Loading Pattern

Each loader follows the same pattern:

```javascript
// frontend/app.js:598-620 (example: loadAccountSummary)
async function loadAccountSummary() {
  try {
    const res = await fetch("/api/account_summary");
    const data = await res.json();
    const payload = data.data ?? data;  // Handle wrapped or unwrapped response
    const acct = payload.accounts?.[0];
    if (!acct) return;

    // Update DOM elements
    if (balanceEl) balanceEl.textContent = formatCurrency(acct.total_value);
    if (dateEl) dateEl.textContent = `As of ${formatDate(payload.as_of)}`;
    if (investedEl) investedEl.textContent = formatCurrency(acct.total_value - acct.total_cash);
    if (cashEl) cashEl.textContent = formatCurrency(acct.total_cash);
    if (settledEl) settledEl.textContent = formatCurrency(acct.settled_cash);
  } catch (err) {
    console.warn("Failed to load account summary", err);
  }
}
```

### Positions with Live Quotes

Positions loader fetches quotes in parallel for all symbols:

```javascript
// frontend/app.js:637-691
async function loadPositions() {
  const posRes = await fetch("/api/positions_list");
  const posPayload = posData.data ?? posData;
  const positions = posPayload.positions || [];

  // Fetch quotes for each symbol in parallel
  const quoteResults = await Promise.all(
    positions.map((pos) =>
      fetch(`/api/quotes?symbol=${encodeURIComponent(pos.symbol)}`)
        .then((r) => r.json())
        .catch(() => null)
    )
  );

  // Build lookup map
  const quoteLookup = {};
  quoteResults.forEach((data) => {
    const payload = data?.data ?? data;
    if (payload?.quotes) {
      payload.quotes.forEach((q) => { quoteLookup[q.symbol] = q; });
    }
  });

  // Render with computed gain %
  positions.forEach((pos) => {
    const quote = quoteLookup[pos.symbol];
    const currentPrice = quote?.price ?? pos.cost_basis;
    const gainPct = ((currentPrice - pos.cost_basis) / pos.cost_basis) * 100;
    // ... render row
  });
}
```

---

## Integration Points

### Backend Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Main conversation endpoint |
| `/api/account_summary` | GET | Account balances |
| `/api/performance` | GET | YTD/1M/3M/1Y returns |
| `/api/positions_list` | GET | Holdings list |
| `/api/quotes` | GET | Real-time prices |
| `/api/activity` | GET | Trade history |
| `/api/transfers` | GET | Deposit/withdrawal history |
| `/debug/trace/{id}` | GET | Trace JSON for routing drawer |
| `/debug/trace/{id}/explain` | POST | LLM-generated trace explanation |

### Trace Data Structure

The routing drawer expects this structure from `/debug/trace/{id}`:

```javascript
{
  trace_id: string,
  utterance: string,
  intent: string,
  routing_mode: "rules" | "torch_classifier" | "llm_reroute",
  routing_confidence: number,
  routing_extracted: { symbol?: string, account?: string, ... },
  routing_missing_params: string[],
  routing_candidates: [{ intent: string, score: number }],
  router_diagnostics: object,
  tool_calls: [{ name: string, source_id: string }],
  tool_params: [{ name: string, params: object }],
  tool_latency_ms: number[],
  context_summary: object,
  latency_ms: {
    routing_ms: number,
    tool_ms: number,
    llm_ms: number,
    postprocess_ms: number,
    total_ms: number
  },
  grounded_sources: string[],
  grounding_rate: number,
  grounding_valid: boolean,
  formatter_used: string,
  session_state: object,
  policy_gate: { allowed: boolean, reason: string, allowed_tools: string[] },
  model_version: string,
  prompt_version: string,
  prompt_name: string,
  prompt_tokens: number,
  completion_tokens: number,
  estimated_cost_usd: number,
  retry_count: number
}
```

---

## State Management

### Global State

```javascript
// frontend/app.js:70-77
let lastTraceId = null;                    // Most recent trace ID for explain button
const STREAMING_ENABLED = false;           // SSE toggle
const sessionKey = "poc_session_id";
let sessionId = localStorage.getItem(sessionKey) || crypto.randomUUID();
```

### UI State

- **Welcome state**: Removed on first message via `removeWelcome()`
- **Send button**: Disabled when input empty via `updateSendBtn()`
- **Typing indicator**: Shown/hidden via CSS class `.visible`
- **Routing panel**: Toggled via `body.routing-open` class

### CSS Class State Toggles

```javascript
// Routing panel visibility
document.body.classList.toggle("routing-open", show);

// Typing indicator
typingIndicator.classList.add("visible");
typingIndicator.classList.remove("visible");

// Modal visibility
routingModal.classList.toggle("hidden", !show);
```

---

## Error Handling

### Chat Errors

```javascript
// frontend/app.js:583-585
try {
  // ... chat logic
} catch (err) {
  assistantBubble.textContent = "Something went wrong.";
}
```

### Sidebar Errors

Each loader catches and logs errors without crashing:

```javascript
// frontend/app.js:617-619
} catch (err) {
  console.warn("Failed to load account summary", err);
}
```

### Network Fallback

Streaming mode falls back to non-streaming on error:

```javascript
// frontend/app.js:220-221
if (!response.ok || !response.body || !isSse) {
  return fallbackChat(utterance, contentEl);
}
```

---

## Design System (CSS Variables)

```css
/* frontend/styles.css:5-93 */
:root {
  /* Colors */
  --bg: #f5f5f7;
  --surface: #ffffff;
  --text-primary: #1d1d1f;
  --accent: #0a84ff;
  --green: #30d158;
  --red: #ff453a;

  /* Chat bubbles */
  --bubble-user: #0a84ff;
  --bubble-user-text: #ffffff;
  --bubble-assistant: #e9e9eb;
  --bubble-assistant-text: #1d1d1f;

  /* Routing panel (dark devtools) */
  --devtools-bg: #1e1e2e;
  --devtools-text: #cdd6f4;
  --devtools-accent: #89b4fa;

  /* Spacing (4px base) */
  --space-1: 4px;
  --space-4: 16px;
  --space-8: 32px;

  /* Typography */
  --text-sm: 12px;
  --text-base: 14px;
  --font-semibold: 600;

  /* Radius */
  --radius-lg: 16px;
  --radius-pill: 999px;

  /* Animations */
  --ease-out: cubic-bezier(0.25, 0.46, 0.45, 0.94);
  --duration-fast: 150ms;
}
```

---

## Responsive Breakpoints

```css
/* frontend/styles.css:1224-1267 */
@media (min-width: 1400px) {
  .shell { max-width: 1360px; }
}

@media (max-width: 1100px) and (min-width: 901px) {
  /* Hide routing panel but keep sidebar */
  body.routing-open .routing-panel { display: none; }
}

@media (max-width: 900px) {
  /* Stack layout: chat on top, sidebar below */
  .grid { grid-template-columns: 1fr; }
  .routing-panel { display: none !important; }
  .sidebar { order: 2; }
  .chat-panel { order: 1; min-height: 420px; }
}
```

---

## Common Issues / Notes

### Streaming Disabled by Default

```javascript
// frontend/app.js:71
const STREAMING_ENABLED = false;
```

SSE streaming is implemented but disabled. The backend must support `stream=true` to enable it.

### Quote Fetching for Each Position

`loadPositions()` makes N+1 API calls (1 for positions, N for quotes). Consider adding a batch quotes endpoint for performance.

### No Build Step

The frontend is vanilla JS with no bundler, transpiler, or framework. This is intentional for POC simplicity but limits:
- No TypeScript
- No component reuse
- No hot reloading

### Modal Not Currently Used

The routing modal (`#routing-modal`) shows a "scale view" explanation but has no trigger button in the current UI. The routing panel drawer is used instead.

### Version Cache Busting

Static assets use query string versioning:
```html
<link rel="stylesheet" href="/ui/styles.css?v=11" />
<script src="/ui/app.js?v=21"></script>
```

Increment manually on changes.

---

## HTML Structure Reference

```html
<!-- frontend/index.html - key sections -->

<div class="shell">
  <header class="topbar">
    <!-- Brand + Live Trace button + Status chip -->
  </header>

  <main class="grid">
    <aside class="sidebar">
      <section class="card account-card" id="account-card">...</section>
      <section class="card" id="positions-card">...</section>
      <section class="card" id="activity-card">...</section>
      <section class="card" id="transfers-card">...</section>
    </aside>

    <section class="chat-panel">
      <div class="chat-header">...</div>
      <div id="chat" class="chat">
        <div id="welcome-state">...</div>
        <div id="typing-indicator">...</div>
      </div>
      <form id="chat-form" class="chat-form">...</form>
    </section>

    <aside class="routing-panel">
      <div class="routing-teaching"><!-- Explain button --></div>
      <div class="routing-summary"><!-- Pipeline summary --></div>
      <div class="routing-body">
        <!-- ~20 routing sections -->
      </div>
    </aside>
  </main>
</div>

<div id="routing-modal" class="modal hidden">
  <!-- Scale view explanation modal -->
</div>
```

---

## Suggestion Chips

Quick-start queries for empty state:

```html
<!-- frontend/index.html:104-109 -->
<div class="suggestion-chips">
  <button class="suggestion-chip" data-query="What are my current positions?">My positions</button>
  <button class="suggestion-chip" data-query="How is AAPL performing?">AAPL performance</button>
  <button class="suggestion-chip" data-query="Show me recent activity">Recent activity</button>
  <button class="suggestion-chip" data-query="What is my account value?">Account value</button>
</div>
```

Click handler:

```javascript
// frontend/app.js:103-112
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
```

---

## Initialization

```javascript
// frontend/app.js:792-798
document.addEventListener("DOMContentLoaded", () => {
  loadAccountSummary();
  loadPerformanceBadge();
  loadPositions();
  loadActivity();
  loadTransfers();
});
```

All sidebar data loads in parallel on page load.
