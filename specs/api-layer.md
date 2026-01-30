# API Layer (FastAPI) Specification

> Main entry point for the Personal Finance Assistant. Orchestrates the NLU pipeline from intent routing through tool calls, context building, LLM response generation, and formatting.

## Overview

The API layer is implemented in `backend/main.py` using FastAPI. It serves two primary functions:

1. **Chat endpoint** (`/chat`): Full NLU pipeline that routes user utterances to intents, calls tools, builds context, generates LLM responses, and formats answers
2. **REST endpoints** (`/api/*`): Direct access to raw tool data for frontend/debugging

The architecture follows an intent-first pattern: route the utterance, call minimal tools, build typed context, generate/format response, validate grounding.

## Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app, all endpoints, pipeline orchestration |
| `backend/schemas.py:220-237` | `ChatRequest`, `ChatResponse` models |
| `backend/artifacts/model_bundle_v1/manifest.json` | Model/prompt version config |

## Key Models

### ChatRequest

```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance: str
    account: Optional[str] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
```

- `utterance`: The user's natural language query
- `account`: Optional account filter (e.g., "Brokerage")
- `stream`: Enable SSE streaming (currently disabled by default)
- `session_id`: For multi-turn conversation state

### ChatResponse

```python
class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    clarifying_question: Optional[str]
    trace_id: str
```

- `answer_markdown`: The formatted answer text
- `citations`: List of source IDs used (e.g., `["tool:positions:v1"]`)
- `confidence`: 0.0-1.0 confidence score
- `needs_clarification`: Whether follow-up is needed
- `clarifying_question`: Question to ask user if clarification needed
- `trace_id`: UUID linking to debug trace

### Intent Enum

```python
class Intent(str, Enum):
    activity = "activity"
    positions = "positions"
    positions_list = "positions_list"
    portfolio_ranking = "portfolio_ranking"
    symbol_performance = "symbol_performance"
    performance = "performance"
    quotes = "quotes"
    facts = "facts"
    transfers = "transfers"
    account_value = "account_value"
    cash_balance = "cash_balance"
    clarify = "clarify"
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str
    citations: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    clarifying_question: Optional[str] = None
```

## API Endpoints

### POST /chat

Main NLU endpoint. Processes user utterances through the full pipeline.

**Request:**
```json
{
  "utterance": "How many shares of AAPL do I own?",
  "account": "Brokerage",
  "stream": false,
  "session_id": "abc123"
}
```

**Response:**
```json
{
  "answer_markdown": "AAPL position in Brokerage (as of 2026-01-15): 50 shares @ $150.00/share.",
  "citations": ["tool:positions:v1"],
  "confidence": 0.95,
  "needs_clarification": false,
  "clarifying_question": null,
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### GET /debug/trace/{trace_id}

Retrieve full trace data for a request.

**Response:** JSON object with all pipeline state (routing, tool calls, tokens, latency, grounding validation).

### POST /debug/trace/{trace_id}/explain

Generate a teaching explanation of how a request was handled.

**Response:**
```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "explanation": "The user asked about AAPL shares. The router used rules to classify this as a positions intent..."
}
```

### POST /eval

Run the evaluation suite against golden test cases.

**Response:**
```json
{
  "pass_rate": 0.92,
  "tool_minimality_score": 0.96,
  "grounding_rate": 0.88,
  "results": [...]
}
```

### REST Data Endpoints

| Endpoint | Parameters | Tool |
|----------|------------|------|
| `GET /api/activity` | `account?` | `get_activity()` |
| `GET /api/positions` | `symbol`, `account?` | `get_positions()` |
| `GET /api/positions_list` | `asset_class?`, `account?` | `get_positions_list()` |
| `GET /api/performance` | `timeframe`, `account?` | `get_performance()` |
| `GET /api/quotes` | `symbol` | `get_quotes()` |
| `GET /api/facts` | `topic` | `get_facts()` |
| `GET /api/transfers` | `account?` | `get_transfers()` |
| `GET /api/account_summary` | `account?` | `get_account_summary()` |

### GET /

Serves frontend `index.html` if exists, otherwise returns `{"status": "ok"}`.

### Static Files

Frontend mounted at `/ui` from `frontend/` directory.

## Pipeline Flow

The `/chat` endpoint implements a 7-stage pipeline:

### Stage 1: Routing (backend/router.py)

```python
route = route_intent(request.utterance, use_llm=True)
```

Three-layer routing:
1. **Rules** (fast): Keyword scoring, symbol extraction
2. **Torch** (learned): BoW classifier for ambiguous cases
3. **LLM** (fallback): OpenAI call when confidence < threshold

Returns `IntentRoute` with:
- `intent`: The classified Intent
- `confidence`: 0.0-1.0 score
- `extracted`: Params like `{"symbol": "AAPL"}`
- `missing_params`: Required params not found
- `routing_mode`: "rules", "torch", or "llm"

### Stage 2: Session State Resolution

```python
if route.intent is Intent.clarify:
    prior = _get_session_state(request.session_id)
    if prior.get("intent") in {"positions_list", "positions"}:
        route.intent = Intent.positions_list
        # ... resolve follow-up
```

Handles multi-turn clarification by checking previous intent context.

### Stage 3: Policy Gate

```python
def _policy_gate(intent: Intent, params: dict[str, str]) -> dict[str, object]:
    allowed_tools = {
        "activity": ["activity"],
        "positions": ["positions"],
        "symbol_performance": ["positions", "quotes"],
        "portfolio_ranking": ["positions_list", "quotes"],
        "performance": ["performance", "transfers"],
        # ...
    }
```

Maps intents to allowed tools. Enforces tool minimality.

### Stage 4: Tool Calls (backend/tools.py)

```python
tool_result = registry.call_tool(route.intent.value, **params)
```

Composite intents require multiple tools:
- `symbol_performance`: positions + quotes
- `portfolio_ranking`: positions_list + quotes
- `performance`: performance + transfers

### Stage 5: Context Building (backend/context_builder.py)

```python
context = build_context(route.intent, tool_result, route.extracted)
# or for composite:
context = build_symbol_performance_context(positions_result, quotes_result, symbol)
```

Transforms raw tool data into typed context objects (e.g., `SymbolPerformanceContext`).

### Stage 6: LLM Response Generation (backend/responder.py)

```python
llm_response, meta = generate_response(request.utterance, context, model_config)
```

- Loads system/user prompts from `artifacts/model_bundle_v1/prompts/`
- Calls OpenAI with JSON schema enforcement
- Validates citations match tool sources
- Retries on parse/grounding failure (max 1 retry)

### Stage 7: Formatting (backend/formatter.py)

```python
formatted = format_answer(context)
if formatted:
    llm_response.answer_markdown = formatted
```

Deterministic formatters override LLM output for known intents. Bypasses LLM variability.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | from manifest | Model for response generation |
| `OPENAI_BASE_URL` | none | Custom API endpoint |
| `ROUTER_MODEL` | `gpt-5-mini` | Model for LLM reroute fallback |
| `ROUTER_CONF_THRESHOLD` | `0.75` | Confidence before LLM fallback |

### Model Config

```python
@dataclass
class ModelConfig:
    api_key: str
    model: str
    base_url: Optional[str]

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

### Manifest (artifacts/model_bundle_v1/manifest.json)

```json
{
  "model_version": "gpt-5.2",
  "router_version": "v2",
  "prompt_version": "v4"
}
```

## State Management

### Session State

In-memory dict for multi-turn conversations:

```python
session_state: dict[str, dict[str, str]] = {}

def _get_session_state(session_id: Optional[str]) -> dict[str, str]:
    if not session_id:
        return {}
    return session_state.get(session_id, {})

def _set_session_state(session_id: Optional[str], state: dict[str, str]) -> None:
    if not session_id:
        return
    session_state[session_id] = state
```

Stored fields:
- `intent`: Previous intent
- `asset_class`: Filter from last query
- `account`: Account from last query
- `cash_type`: Cash type from last query

State persisted for intents: `positions_list`, `positions`, `account_value`, `cash_balance`.

### Tool Registry

Singleton instance:

```python
registry = ToolRegistry()
```

## Error Handling

### Missing API Key

```python
if not api_key:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
```

### Tool Call Failure

```python
except Exception as exc:
    llm = LLMResponse(
        answer_markdown="I could not retrieve the data needed to answer.",
        citations=[],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question="Can you double-check the request or try a different symbol?",
    )
```

### Context Building Failure

```python
except Exception:
    llm = LLMResponse(
        answer_markdown="I do not have enough data to answer that.",
        citations=[],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question="Can you clarify the symbol or account?",
    )
```

### LLM Generation Failure

```python
except Exception:
    llm_response = LLMResponse(
        answer_markdown="I could not generate a grounded response at the moment.",
        citations=[],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question="Please try again or rephrase the question.",
    )
```

### Missing Quote for Symbol

```python
def _quote_missing_response(symbol: str, quotes_result) -> tuple[LLMResponse, str]:
    available = _available_quote_symbols(quotes_result)
    message = f"I don't have a quote for {symbol} in this demo dataset (as of {quotes_result.as_of})."
    if available:
        message += f" Available symbols: {', '.join(available)}."
```

### Trace Not Found

```python
if not data:
    raise HTTPException(status_code=404, detail="Trace not found")
```

## Clarification Logic

```python
def _clarify(intent: Intent, missing: list[str]) -> str:
    if intent is Intent.positions:
        return "Which symbol should I look up (e.g., AAPL, MSFT)?"
    if intent is Intent.positions_list:
        return "Do you want all positions, or only stocks/ETFs?"
    if intent is Intent.symbol_performance:
        return "Which symbol's performance should I compute (e.g., AAPL, VOO)?"
    if intent is Intent.performance:
        return "Which timeframe: YTD, 1Y, or all time?"
    if intent is Intent.quotes:
        return "Which symbol should I quote (e.g., AAPL, MSFT)?"
    # ... more cases
```

## Streaming (Disabled)

SSE streaming is implemented but disabled by default:

```python
STREAMING_ENABLED = False

if STREAMING_ENABLED and request.stream:
    def event_stream() -> Generator[str, None, None]:
        for chunk in stream_chunks(llm_response.answer_markdown):
            safe_chunk = chunk.replace("\n", "\\n")
            yield f"event: delta\ndata: {safe_chunk}\n\n"
        payload = _build_chat_response(llm_response, trace_id).model_dump()
        yield f"event: final\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Tracing Integration

Every `/chat` request generates a trace:

```python
trace = TraceLogger()
trace_id = trace.start(request.utterance)

# Throughout pipeline:
trace.log_intent(route.intent.value)
trace.log_routing(route.routing_mode, route.confidence, route.candidates)
trace.log_tool(route.intent.value, tool_latency, tool_result.source_id)
trace.log_tokens(meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0), ...)
trace.log_grounding_validation(grounding_valid)
trace.log_latency_breakdown({
    "routing_ms": routing_ms,
    "tool_ms": tool_latency,
    "llm_ms": llm_latency,
    "postprocess_ms": postprocess_latency,
    "total_ms": total_ms,
})
trace.finalize(model_config.model, "model_bundle_v1", MANIFEST["prompt_version"])
```

Traces stored in `backend/traces/{trace_id}.json`.

## Grounding Validation

Citations must match tool sources:

```python
grounding_valid = (not llm_response.needs_clarification) and set(llm_response.citations) == set(context.sources)
trace.log_grounding_validation(grounding_valid)
```

The responder validates during generation:

```python
def validate_response(llm_response: LLMResponse, valid_sources: list[str]) -> bool:
    if llm_response.needs_clarification:
        return True
    if not llm_response.citations:
        return False
    return set(llm_response.citations) == set(valid_sources)
```

## Integration Points

### Router (backend/router.py)

```python
from .router import route_intent
route = route_intent(request.utterance, use_llm=True)
```

### Tools (backend/tools.py)

```python
from .tools import ToolRegistry
registry = ToolRegistry()
tool_result = registry.call_tool("positions", symbol="AAPL")
```

### Context Builder (backend/context_builder.py)

```python
from .context_builder import build_context, build_symbol_performance_context
context = build_context(Intent.positions, tool_result, {"symbol": "AAPL"})
```

### Responder (backend/responder.py)

```python
from .responder import ModelConfig, generate_response
llm_response, meta = generate_response(utterance, context, model_config)
```

### Formatter (backend/formatter.py)

```python
from .formatter import format_answer, build_context_summary
formatted = format_answer(context)
```

### Tracing (backend/tracing.py)

```python
from .tracing import TraceLogger, load_trace
trace = TraceLogger()
```

### Eval (backend/eval.py)

```python
from .eval import EvalRunner
runner = EvalRunner(model_config)
results = runner.run()
```

### Teaching (backend/teaching.py)

```python
from .teaching import generate_teaching_explanation
explanation = generate_teaching_explanation(trace, api_key, model, base_url)
```

## Common Patterns

### Account Parameter Merging

```python
def _merge_account_params(extracted: dict[str, str], request_account: Optional[str]) -> dict[str, str]:
    merged = dict(extracted)
    if "account" not in merged and request_account:
        merged["account"] = request_account
    return merged
```

### Composite Intent Handling

Symbol performance requires both positions and quotes:

```python
if route.intent is Intent.symbol_performance:
    params = _merge_account_params(route.extracted, request.account)
    positions_result = registry.call_tool("positions", **params)
    quotes_result = registry.call_tool("quotes", **params)
    context = build_symbol_performance_context(
        positions_result, quotes_result, route.extracted["symbol"]
    )
```

### Latency Breakdown Tracking

```python
overall_start = time.perf_counter()
routing_start = time.perf_counter()
route = route_intent(request.utterance, use_llm=True)
routing_ms = int((time.perf_counter() - routing_start) * 1000)

tool_start = time.perf_counter()
# ... tool calls
tool_latency = int((time.perf_counter() - tool_start) * 1000)

llm_start = time.perf_counter()
# ... LLM call
llm_latency = int((time.perf_counter() - llm_start) * 1000)
```

## Startup

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

The app:
1. Loads `.env` via `python-dotenv`
2. Reads manifest from `artifacts/model_bundle_v1/manifest.json`
3. Mounts frontend static files at `/ui`
4. Creates singleton `ToolRegistry`
5. Initializes empty `session_state` dict

## Known Issues / TODOs

1. **Session state is in-memory**: Lost on restart. Consider Redis for production.

2. **Streaming disabled**: `STREAMING_ENABLED = False`. SSE implementation exists but not exposed.

3. **Quote coverage**: Symbol performance and portfolio ranking fail if quotes missing for positions. Eval suite validates this upfront.

4. **Hardcoded clarification messages**: `_clarify()` function has static strings per intent.

5. **No rate limiting**: API endpoints have no throttling.

6. **Single retry on LLM failure**: `max_retries=1` in `generate_response()`.
