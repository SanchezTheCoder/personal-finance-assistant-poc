# Tool Layer Specification

> Simulated brokerage data layer providing mock financial data for grounding LLM responses.

---

## Overview

The Tool Layer is a data abstraction that provides simulated brokerage account data to the rest of the system. It acts as the "source of truth" for all financial information: positions, quotes, activity, transfers, performance metrics, and educational facts.

**Why it exists:**
1. Provides deterministic, controlled data for demo/testing scenarios
2. Decouples business logic from data sources (could swap to real API later)
3. Returns `ToolResult` objects with `source_id` for citation/grounding validation
4. Enables the LLM to cite specific data sources in responses

**Key design decisions:**
- Single unified JSON file (`user_master.json`) consolidates all mock data
- File-mtime caching prevents redundant disk reads
- Each tool function returns a `ToolResult` with traceable `source_id`
- `ToolRegistry` provides intent-based dispatch for the chat pipeline

---

## Source Files

| File | Purpose |
|------|---------|
| `backend/tools.py` | Tool functions and `ToolRegistry` dispatcher |
| `backend/data/user_master.json` | Unified mock data (positions, quotes, transfers, etc.) |
| `backend/data/*.json` | Legacy per-domain JSON files (unused by tools.py) |
| `backend/data/facts/*.md` | Static educational snippets |
| `backend/schemas.py:36-41` | `ToolResult` model definition |

---

## Key Models and Types

### ToolResult

The core return type for all tool functions. Every piece of data fetched includes provenance metadata.

```python
# backend/schemas.py:36-41
class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str      # e.g., "tool:positions:v1"
    data: dict[str, Any]  # payload specific to the tool
    as_of: str          # timestamp, e.g., "2026-01-15"
```

**source_id format:** `tool:<domain>:v1`
- `tool:positions:v1`
- `tool:quotes:v1`
- `tool:activity:v1`
- `tool:transfers:v1`
- `tool:performance:v1`
- `tool:account_summary:v1`
- `tool:facts:v1`
- `tool:positions_list:v1`

The `source_id` is critical for **grounding validation**: the responder verifies that LLM-generated citations match actual tool `source_id` values.

---

## Data Loading Architecture

### Master Data Cache

```python
# backend/tools.py:9-23
DATA_DIR = Path(__file__).parent / "data"
MASTER_PATH = DATA_DIR / "user_master.json"
_MASTER_CACHE: Optional[dict[str, Any]] = None
_MASTER_MTIME: Optional[float] = None


def _load_master() -> dict[str, Any]:
    global _MASTER_CACHE, _MASTER_MTIME
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master data file: {MASTER_PATH}")
    mtime = MASTER_PATH.stat().st_mtime
    if _MASTER_CACHE is None or _MASTER_MTIME != mtime:
        _MASTER_CACHE = json.loads(MASTER_PATH.read_text())
        _MASTER_MTIME = mtime
    return _MASTER_CACHE
```

**Cache invalidation:** Checks file modification time on every call. If the file changed, reloads. This allows hot-reloading data during development.

### Section Extraction

```python
# backend/tools.py:26-33
def _get_section(section: str) -> dict[str, Any]:
    payload = _load_master()
    data = payload.get(section)
    if not data:
        raise KeyError(f"Missing section '{section}' in {MASTER_PATH.name}")
    if "as_of" not in data and payload.get("as_of"):
        data = {**data, "as_of": payload["as_of"]}
    return data
```

Falls back to the top-level `as_of` if a section doesn't have its own timestamp.

---

## Public API: Tool Functions

### get_activity

Retrieves recent trading activity (buys/sells).

```python
# backend/tools.py:36-38
def get_activity(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("activity")
    return ToolResult(source_id="tool:activity:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "account": "Brokerage",
  "trades": [
    {
      "timestamp": "2026-01-14T15:42:00Z",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 10,
      "price": 192.33
    }
  ]
}
```

### get_positions

Retrieves position data. Returns full positions list (filtering happens in context builder).

```python
# backend/tools.py:41-43
def get_positions(symbol: str, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    return ToolResult(source_id="tool:positions:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "account": "Brokerage",
  "positions": [
    {"symbol": "AAPL", "quantity": 42, "cost_basis": 150.25, "asset_class": "stocks"},
    {"symbol": "MSFT", "quantity": 12, "cost_basis": 280.10, "asset_class": "stocks"},
    {"symbol": "VOO", "quantity": 25, "cost_basis": 390.55, "asset_class": "etf"}
  ]
}
```

### get_positions_list

Retrieves all positions with optional asset class filtering.

```python
# backend/tools.py:46-53
def get_positions_list(asset_class: Optional[str] = None, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    if asset_class:
        filtered = [
            p for p in payload["positions"] if p.get("asset_class") == asset_class
        ]
        payload = {**payload, "positions": filtered, "asset_class_filter": asset_class}
    return ToolResult(source_id="tool:positions_list:v1", data=payload, as_of=payload["as_of"])
```

### get_performance

Retrieves portfolio performance metrics.

```python
# backend/tools.py:55-57
def get_performance(timeframe: str, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("performance")
    return ToolResult(source_id="tool:performance:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "account": "Brokerage",
  "timeframe": "YTD",
  "return_pct": 6.2,
  "contributions": 1200.00
}
```

### get_quotes

Retrieves market quotes for symbols.

```python
# backend/tools.py:60-62
def get_quotes(symbol: str) -> ToolResult:
    payload = _get_section("quotes")
    return ToolResult(source_id="tool:quotes:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "quotes": [
    {"symbol": "AAPL", "price": 193.12, "change_pct": 1.1},
    {"symbol": "MSFT", "price": 420.55, "change_pct": -0.6},
    {"symbol": "VOO", "price": 412.34, "change_pct": 0.8},
    {"symbol": "TSLA", "price": 238.22, "change_pct": 2.4}
  ]
}
```

### get_facts

Retrieves educational content based on topic keywords.

```python
# backend/tools.py:65-82
def get_facts(topic: str) -> ToolResult:
    # naive mapping based on keywords
    topic_lower = topic.lower()
    if "roth" in topic_lower:
        file_name = "roth_ira.md"
    elif "etf" in topic_lower:
        file_name = "etf_basics.md"
    else:
        file_name = "rebalancing.md"

    content = (DATA_DIR / "facts" / file_name).read_text().strip()
    payload = {
        "topic": topic,
        "snippet": content.split("\n", 2)[-1].strip(),
        "source": f"facts/{file_name}",
        "as_of": "2026-01-15",
    }
    return ToolResult(source_id="tool:facts:v1", data=payload, as_of=payload["as_of"])
```

**Available fact files:**
| File | Trigger keywords |
|------|------------------|
| `facts/roth_ira.md` | "roth" |
| `facts/etf_basics.md` | "etf" |
| `facts/rebalancing.md` | (default fallback) |

**Fact file format:**
```markdown
# Roth IRA (Simulated)

A Roth IRA is an individual retirement account funded with after-tax dollars...
```

The first line (heading) is stripped; only the content after line 2 is returned as `snippet`.

### get_transfers

Retrieves deposit/withdrawal history.

```python
# backend/tools.py:85-87
def get_transfers(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("transfers")
    return ToolResult(source_id="tool:transfers:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "account": "Brokerage",
  "transfers": [
    {
      "timestamp": "2026-01-12T10:15:00Z",
      "type": "deposit",
      "method": "ACH",
      "amount": 1500.00,
      "status": "completed"
    }
  ]
}
```

### get_account_summary

Retrieves account value and cash balances.

```python
# backend/tools.py:90-92
def get_account_summary(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("account_summary")
    return ToolResult(source_id="tool:account_summary:v1", data=payload, as_of=payload["as_of"])
```

**Data shape:**
```json
{
  "as_of": "2026-01-15",
  "accounts": [
    {
      "account": "Brokerage",
      "total_value": 128450.22,
      "total_cash": 12340.10,
      "settled_cash": 11990.10
    }
  ]
}
```

---

## ToolRegistry: Intent-Based Dispatch

The `ToolRegistry` class provides a single entry point for the chat pipeline to call tools based on intent.

```python
# backend/tools.py:93-113
class ToolRegistry:
    def call_tool(self, intent: str, **params: Any) -> ToolResult:
        if intent == "activity":
            return get_activity(account=params.get("account"))
        if intent == "positions":
            return get_positions(symbol=params["symbol"], account=params.get("account"))
        if intent == "positions_list":
            return get_positions_list(
                asset_class=params.get("asset_class"), account=params.get("account")
            )
        if intent == "performance":
            return get_performance(timeframe=params["timeframe"], account=params.get("account"))
        if intent == "quotes":
            return get_quotes(symbol=params["symbol"])
        if intent == "facts":
            return get_facts(topic=params["topic"])
        if intent == "transfers":
            return get_transfers(account=params.get("account"))
        if intent in {"account_value", "cash_balance"}:
            return get_account_summary(account=params.get("account"))
        raise ValueError(f"Unknown intent: {intent}")
```

**Intent to tool mapping:**

| Intent | Tool Function | Required Params |
|--------|---------------|-----------------|
| `activity` | `get_activity` | - |
| `positions` | `get_positions` | `symbol` |
| `positions_list` | `get_positions_list` | - |
| `performance` | `get_performance` | `timeframe` |
| `quotes` | `get_quotes` | `symbol` |
| `facts` | `get_facts` | `topic` |
| `transfers` | `get_transfers` | - |
| `account_value` | `get_account_summary` | - |
| `cash_balance` | `get_account_summary` | - |

**Note:** `symbol_performance` and `portfolio_ranking` intents are NOT handled by `ToolRegistry.call_tool()`. They require multiple tool calls orchestrated by `main.py`.

---

## Integration Points

### 1. Chat Pipeline (main.py)

The chat endpoint orchestrates tool calls based on routed intent.

**Single-tool intents:**
```python
# backend/main.py:286-289
else:
    params = _merge_account_params(route.extracted, request.account)
    trace.log_tool_params(route.intent.value, params)
    tool_result = registry.call_tool(route.intent.value, **params)
```

**Composite intents (multiple tools):**

```python
# backend/main.py:258-264 (symbol_performance)
if route.intent is Intent.symbol_performance:
    params = _merge_account_params(route.extracted, request.account)
    trace.log_tool_params("positions", params)
    positions_result = registry.call_tool("positions", **params)
    trace.log_tool_params("quotes", {"symbol": params.get("symbol")})
    quotes_result = registry.call_tool("quotes", **params)
    tool_result = positions_result
```

```python
# backend/main.py:265-278 (portfolio_ranking)
elif route.intent is Intent.portfolio_ranking:
    params = _merge_account_params(route.extracted, request.account)
    trace.log_tool_params("positions_list", params)
    positions_result = registry.call_tool("positions_list", **params)
    first_symbol = (
        positions_result.data["positions"][0]["symbol"]
        if positions_result.data.get("positions")
        else None
    )
    if not first_symbol:
        raise ValueError("No positions available for ranking")
    trace.log_tool_params("quotes", {"symbol": first_symbol})
    quotes_result = registry.call_tool("quotes", symbol=first_symbol)
```

```python
# backend/main.py:279-285 (performance with transfers)
elif route.intent is Intent.performance:
    params = _merge_account_params(route.extracted, request.account)
    trace.log_tool_params("performance", params)
    performance_result = registry.call_tool("performance", **params)
    trace.log_tool_params("transfers", {"account": params.get("account")})
    transfers_result = registry.call_tool("transfers", **params)
```

### 2. Context Builder

The `context_builder.py` transforms raw `ToolResult` data into typed context objects for the LLM.

```python
# backend/context_builder.py:55
def build_context(intent: Intent, tool_result: ToolResult, params: dict[str, Any]) -> ContextBundle:
```

For composite contexts:
```python
# backend/context_builder.py:161-196
def build_symbol_performance_context(
    positions_result: ToolResult,
    quotes_result: ToolResult,
    symbol: str,
) -> ContextBundle:
    # Combines position and quote data to compute unrealized P/L
```

```python
# backend/context_builder.py:199-255
def build_portfolio_ranking_context(
    positions_result: ToolResult,
    quotes_result: ToolResult,
    direction: str = "best",
    basis: str = "unrealized_pl",
) -> ContextBundle:
    # Ranks all positions by performance
```

### 3. REST API Endpoints

Direct tool access via REST for frontend/debugging.

```python
# backend/main.py:568-605
@app.get("/api/activity")
async def api_activity(account: Optional[str] = None):
    return get_activity(account=account).model_dump()

@app.get("/api/positions")
async def api_positions(symbol: str, account: Optional[str] = None):
    return get_positions(symbol=symbol, account=account).model_dump()

@app.get("/api/positions_list")
async def api_positions_list(asset_class: Optional[str] = None, account: Optional[str] = None):
    return get_positions_list(asset_class=asset_class, account=account).model_dump()

@app.get("/api/performance")
async def api_performance(timeframe: str, account: Optional[str] = None):
    return get_performance(timeframe=timeframe, account=account).model_dump()

@app.get("/api/quotes")
async def api_quotes(symbol: str):
    return get_quotes(symbol=symbol).model_dump()

@app.get("/api/facts")
async def api_facts(topic: str):
    return get_facts(topic=topic).model_dump()

@app.get("/api/transfers")
async def api_transfers(account: Optional[str] = None):
    return get_transfers(account=account).model_dump()

@app.get("/api/account_summary")
async def api_account_summary(account: Optional[str] = None):
    return get_account_summary(account=account).model_dump()
```

### 4. Tracing

Every tool call is logged to the trace:
```python
# backend/main.py:315-326
tool_latency = int((time.perf_counter() - tool_start) * 1000)
if route.intent is Intent.symbol_performance:
    trace.log_tool("positions", tool_latency, positions_result.source_id)
    trace.log_tool("quotes", tool_latency, quotes_result.source_id)
elif route.intent is Intent.portfolio_ranking:
    trace.log_tool("positions_list", tool_latency, positions_result.source_id)
    trace.log_tool("quotes", tool_latency, quotes_result.source_id)
elif route.intent is Intent.performance:
    trace.log_tool("performance", tool_latency, performance_result.source_id)
    trace.log_tool("transfers", tool_latency, transfers_result.source_id)
else:
    trace.log_tool(route.intent.value, tool_latency, tool_result.source_id)
```

---

## Error Handling

### Missing Master File
```python
if not MASTER_PATH.exists():
    raise FileNotFoundError(f"Missing master data file: {MASTER_PATH}")
```

### Missing Section
```python
if not data:
    raise KeyError(f"Missing section '{section}' in {MASTER_PATH.name}")
```

### Unknown Intent
```python
raise ValueError(f"Unknown intent: {intent}")
```

### Chat Pipeline Error Handling

When tool calls fail, the chat pipeline returns a graceful degradation:

```python
# backend/main.py:290-313
except Exception as exc:
    llm = LLMResponse(
        answer_markdown="I could not retrieve the data needed to answer.",
        citations=[],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question="Can you double-check the request or try a different symbol?",
    )
```

---

## Data File Structure

### user_master.json (Primary)

```json
{
  "as_of": "2026-01-15",
  "account_summary": { ... },
  "activity": { ... },
  "positions": { ... },
  "quotes": { ... },
  "performance": { ... },
  "transfers": { ... }
}
```

### Legacy Files (Not Used by tools.py)

These files exist but are NOT loaded by `tools.py`. They may be used for direct REST API testing or historical reference:

- `backend/data/account.json`
- `backend/data/activity.json`
- `backend/data/positions.json`
- `backend/data/quotes.json`
- `backend/data/performance.json`
- `backend/data/transfers.json`

---

## Common Patterns

### 1. Parameter Pass-Through

Most tool functions accept an `account` parameter but don't filter by it (the mock data has a single account):

```python
def get_activity(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("activity")  # account param ignored
```

### 2. Filtering in Tool vs Context Builder

- **Tool-level filtering:** `get_positions_list` filters by `asset_class`
- **Context-level filtering:** `build_context` for `positions` extracts a single symbol

### 3. Timestamp Propagation

Every `ToolResult` includes `as_of` for data freshness. The context builder preserves this:

```python
context = PositionsContext(
    ...
    as_of=data["as_of"],
)
```

---

## Known Limitations / TODOs

1. **Single account:** The `account` parameter is accepted but largely ignored (only one account in mock data)

2. **No real-time quotes:** Quotes are static; `get_quotes` returns the same data regardless of symbol requested (filtering happens in context builder)

3. **Facts keyword matching is naive:** Simple string contains check, no fuzzy matching

4. **No validation on symbol parameter:** `get_positions(symbol="INVALID")` succeeds; the error surfaces in context builder

5. **Legacy files unused:** The per-domain JSON files in `backend/data/` are not used by `tools.py` but still exist

---

## Testing Considerations

1. **Hot reload:** Modify `user_master.json` and tools will pick up changes (mtime check)

2. **REST endpoints:** Test individual tools via `/api/*` endpoints

3. **Trace inspection:** Check tool `source_id` values in trace JSON for grounding debugging

4. **Error scenarios:** Test with missing symbols, invalid timeframes to verify error handling
