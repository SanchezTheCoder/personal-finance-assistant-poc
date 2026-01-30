# Eval System

> Automated quality checks against golden utterances. Measures intent accuracy, tool minimality, grounding correctness, and latency budget.

## Overview

The Eval System validates the entire NLU pipeline end-to-end by running a predefined set of "golden" test utterances through the system and measuring multiple quality dimensions. It serves as the primary regression test and quality gate for the Personal Finance Assistant.

**Why it exists:**
- Ensures routing accuracy stays high across code changes
- Validates tool selection minimality (no unnecessary tool calls)
- Confirms LLM citations match tool sources (grounding)
- Enforces latency budgets for acceptable UX
- Provides a single pass/fail metric for quality

## Files

| File | Purpose |
|------|---------|
| `backend/eval.py` | `EvalRunner` class, golden test set definition |
| `backend/main.py:562-565` | `/eval` POST endpoint |

## Key Constants

```python
# backend/eval.py:14
LATENCY_BUDGET_MS = 5000  # 5 second max per request

# backend/eval.py:16
DATA_DIR = Path(__file__).parent / "data"
```

## Golden Test Set

The golden set is a list of utterance-intent-tools tuples. Each entry defines:
- `utterance`: The user input to test
- `intent`: Expected `Intent` enum value
- `tools`: Expected tool source_ids to be called

```python
# backend/eval.py:18-45
GOLDEN_SET = [
    {"utterance": "What was my most recent trade?", "intent": Intent.activity, "tools": ["tool:activity:v1"]},
    {"utterance": "positions?", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "what do i own", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "my holdings", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "How many shares of AAPL do I own?", "intent": Intent.positions, "tools": ["tool:positions:v1"]},
    {"utterance": "account balance", "intent": Intent.account_value, "tools": ["tool:account_summary:v1"]},
    {"utterance": "account value", "intent": Intent.account_value, "tools": ["tool:account_summary:v1"]},
    {"utterance": "cash balance", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "cash value", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "settled cash", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "perfomance ytd", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "portfolio return", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "how did I do this year", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "AAPL performance and price", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "AAPL performance?", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "voo performance", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "best performing position", "intent": Intent.portfolio_ranking, "tools": ["tool:positions_list:v1", "tool:quotes:v1"]},
    {"utterance": "biggest unrealized loss", "intent": Intent.portfolio_ranking, "tools": ["tool:positions_list:v1", "tool:quotes:v1"]},
    {"utterance": "apple quote", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "AAPL price", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "What's MSFT price and today's change?", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "What's my most recent transfer?", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "Did my last deposit go through?", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "Show my recent withdrawals", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "ACH pending", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "What is a Roth IRA?", "intent": Intent.facts, "tools": ["tool:facts:v1"]},
]
```

**Coverage notes:**
- 25 test cases across 11 intents
- Includes typos ("perfomance") to test normalization
- Includes aliases ("apple", "voo") to test symbol extraction
- Tests composite intents (symbol_performance, portfolio_ranking, performance) that require multiple tools

## Core Types

### ModelConfig

```python
# backend/responder.py:27-31
@dataclass
class ModelConfig:
    api_key: str
    model: str
    base_url: Optional[str]
```

### EvalRunner

```python
# backend/eval.py:61-65
class EvalRunner:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.registry = ToolRegistry()
```

### EvalResult Structure (returned by `run()`)

```python
{
    "pass_rate": float,           # 0.0-1.0, % of tests fully passing
    "tool_minimality_score": float,  # 0.0-1.0, % matching exact tool count
    "grounding_rate": float,      # 0.0-1.0, % with valid citations
    "results": [
        {
            "utterance": str,
            "pass": bool,
            "intent_ok": bool,      # Did routing match expected intent?
            "tool_ok": bool,        # Did tool calls match expected tools?
            "grounding_ok": bool,   # Did LLM citations match sources?
            "latency_ms": int,      # Time for full pipeline
        },
        # ... one per golden item
    ]
}
```

## Public API

### `EvalRunner.run() -> dict[str, Any]`

Runs all golden tests and returns aggregate metrics.

```python
# backend/eval.py:66-187
def run(self) -> dict[str, Any]:
    results = []
    total = len(GOLDEN_SET)
    pass_count = 0
    grounding_hits = 0
    tool_minimality_hits = 0

    # Pre-flight check: validate quote coverage
    missing_quotes = _validate_quote_coverage()
    if missing_quotes:
        return {
            "pass_rate": 0.0,
            "tool_minimality_score": 0.0,
            "grounding_rate": 0.0,
            "results": [
                {
                    "utterance": "quote_coverage",
                    "pass": False,
                    "reason": f"Missing quotes for positions: {', '.join(missing_quotes)}",
                }
            ],
        }

    for item in GOLDEN_SET:
        # ... run each test
```

### `/eval` Endpoint

```python
# backend/main.py:562-565
@app.post("/eval")
async def run_eval():
    runner = EvalRunner(_model_config())
    return JSONResponse(content=runner.run())
```

**Usage:**
```bash
curl -X POST http://localhost:8000/eval
```

## Internal Patterns

### 1. Pre-flight Quote Coverage Validation

Before running any tests, the eval system validates that all position symbols have corresponding quotes. This prevents false failures due to missing market data.

```python
# backend/eval.py:52-58
def _validate_quote_coverage() -> list[str]:
    positions = _load_json("positions.json")["positions"]
    quotes = _load_json("quotes.json")["quotes"]
    position_symbols = {p["symbol"] for p in positions}
    quote_symbols = {q["symbol"] for q in quotes}
    missing = sorted(position_symbols - quote_symbols)
    return missing
```

If any symbols are missing, eval fails immediately with a diagnostic result.

### 2. LLM-Free Routing

Eval uses `use_llm=False` to ensure deterministic, fast routing:

```python
# backend/eval.py:90
route = route_intent(item["utterance"], use_llm=False)
```

This means eval only tests the rule-based router, not the LLM fallback or torch router.

### 3. Intent-Specific Tool Orchestration

Different intents require different tool call patterns:

**Single-tool intents:**
```python
# backend/eval.py:138-141
else:
    tool_result = self.registry.call_tool(route.intent.value, **route.extracted)
    tool_results = [tool_result]
    context = build_context(route.intent, tool_result, route.extracted)
```

**Symbol performance (positions + quotes):**
```python
# backend/eval.py:103-112
if route.intent is Intent.symbol_performance:
    positions_result = self.registry.call_tool("positions", **route.extracted)
    quotes_result = self.registry.call_tool("quotes", **route.extracted)
    tool_results = [positions_result, quotes_result]
    try:
        context = build_symbol_performance_context(
            positions_result, quotes_result, route.extracted["symbol"]
        )
    except Exception:
        context = build_context(Intent.positions, positions_result, route.extracted)
```

**Portfolio ranking (positions_list + quotes):**
```python
# backend/eval.py:113-127
elif route.intent is Intent.portfolio_ranking:
    positions_result = self.registry.call_tool("positions_list", **route.extracted)
    first_symbol = (
        positions_result.data["positions"][0]["symbol"]
        if positions_result.data.get("positions")
        else "AAPL"
    )
    quotes_result = self.registry.call_tool("quotes", symbol=first_symbol)
    tool_results = [positions_result, quotes_result]
    context = build_portfolio_ranking_context(
        positions_result,
        quotes_result,
        direction=route.extracted.get("direction", "best"),
        basis=route.extracted.get("basis", "unrealized_pl"),
    )
```

**Performance (performance + transfers for YTD contributions):**
```python
# backend/eval.py:128-137
elif route.intent is Intent.performance:
    performance_result = self.registry.call_tool("performance", **route.extracted)
    transfers_result = self.registry.call_tool("transfers")
    tool_results = [performance_result, transfers_result]
    context = build_context(
        Intent.performance,
        performance_result,
        {**route.extracted, "transfers": transfers_result.data.get("transfers", [])},
    )
    context.sources = [performance_result.source_id, transfers_result.source_id]
```

### 4. Pass Criteria

A test passes when ALL of these conditions are met:

```python
# backend/eval.py:167
passed = intent_ok and tool_ok and grounding_ok and elapsed_ms <= LATENCY_BUDGET_MS
```

Where:
- `intent_ok`: Routed intent matches expected intent
- `tool_ok`: Called tools match expected tools (order-independent)
- `grounding_ok`: All LLM citations exist in context.sources
- `elapsed_ms <= LATENCY_BUDGET_MS`: Under 5 second budget

### 5. Grounding Validation

```python
# backend/eval.py:160
grounding_ok = all(c in context.sources for c in llm_response.citations)
```

This ensures the LLM only cites tools that were actually called.

### 6. Tool Minimality Scoring

```python
# backend/eval.py:164-165
if tool_minimality == len(item["tools"]):
    tool_minimality_hits += 1
```

Tracks whether the system called exactly the right number of tools (no extras, no missing).

## Integration Points

### Dependencies

| System | Usage |
|--------|-------|
| Intent Router | `route_intent(utterance, use_llm=False)` for deterministic routing |
| Tool Registry | `ToolRegistry.call_tool(intent, **params)` for tool execution |
| Context Builder | `build_context()`, `build_symbol_performance_context()`, `build_portfolio_ranking_context()` |
| Responder | `generate_response()` for LLM calls |
| Schemas | `Intent`, `IntentRoute`, `ToolResult`, `LLMResponse`, `ContextBundle` |

### Data Dependencies

| File | Purpose |
|------|---------|
| `backend/data/positions.json` | Position data for coverage validation |
| `backend/data/quotes.json` | Quote data for coverage validation |
| `backend/data/user_master.json` | Unified data source for tool calls |

## State Management

The EvalRunner is stateless. Each `run()` call:
1. Creates a fresh `ToolRegistry` instance
2. Loads data files fresh (via `_load_json`)
3. Returns results without side effects

No cross-test state is maintained. Each golden test is independent.

## Error Handling

### Clarify Intent
If routing returns `Intent.clarify`, the test fails immediately:

```python
# backend/eval.py:92-100
if route.intent is Intent.clarify:
    results.append(
        {
            "utterance": item["utterance"],
            "pass": False,
            "reason": f"clarify: missing {route.missing_params}",
        }
    )
    continue
```

### LLM Errors
If `generate_response()` throws, the test fails with the error:

```python
# backend/eval.py:143-153
try:
    llm_response, _ = generate_response(item["utterance"], context, self.model_config)
except Exception as exc:
    results.append(
        {
            "utterance": item["utterance"],
            "pass": False,
            "reason": f"llm_error: {exc}",
        }
    )
    continue
```

### Missing API Key
If `OPENAI_API_KEY` is not set, `_model_config()` raises `HTTPException(500)`:

```python
# backend/main.py:59-67
def _model_config() -> ModelConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return ModelConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", MANIFEST["model_version"]),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
```

## Common Issues

### 1. Quote Coverage Failures
If `positions.json` contains symbols not in `quotes.json`, eval fails before running any tests. The fix is to add the missing quotes to the data file.

### 2. Symbol Performance Fallback
If `build_symbol_performance_context()` throws (e.g., symbol not found in positions), it falls back to a basic positions context:

```python
# backend/eval.py:107-112
try:
    context = build_symbol_performance_context(...)
except Exception:
    context = build_context(Intent.positions, positions_result, route.extracted)
```

This can mask data issues and cause unexpected pass/fail results.

### 3. Hardcoded First Symbol for Portfolio Ranking
The portfolio ranking logic uses the first position's symbol for quotes:

```python
# backend/eval.py:115-119
first_symbol = (
    positions_result.data["positions"][0]["symbol"]
    if positions_result.data.get("positions")
    else "AAPL"
)
```

This is a simplification that may not fetch all needed quotes for accurate ranking.

### 4. LLM Non-Determinism
Since eval calls the real LLM, results can vary between runs. The grounding validation helps, but LLM content may still differ.

## Metrics Interpretation

| Metric | Target | Meaning |
|--------|--------|---------|
| `pass_rate` | 1.0 | All tests passing all criteria |
| `tool_minimality_score` | 1.0 | No extra/missing tool calls |
| `grounding_rate` | 1.0 | LLM always cites correct sources |

A healthy system shows all three at 1.0. Lower scores indicate:
- `pass_rate < 1.0`: Some tests failing (check individual results)
- `tool_minimality_score < pass_rate`: Tool selection issues
- `grounding_rate < pass_rate`: LLM hallucinating citations

## Running Eval

```bash
# Start backend
uv run uvicorn backend.main:app --reload --port 8000

# Run eval suite
curl -X POST http://localhost:8000/eval
```

**Example output:**
```json
{
  "pass_rate": 0.96,
  "tool_minimality_score": 1.0,
  "grounding_rate": 1.0,
  "results": [
    {"utterance": "What was my most recent trade?", "pass": true, "intent_ok": true, "tool_ok": true, "grounding_ok": true, "latency_ms": 1234},
    ...
  ]
}
```
