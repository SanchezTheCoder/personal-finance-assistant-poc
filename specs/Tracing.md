# Tracing System Specification

## Overview

The tracing system provides per-request observability for the Personal Finance Assistant. Every `/chat` request generates a structured trace that captures the complete pipeline execution, from intent routing through tool calls to LLM response generation. Traces are persisted as JSON files for debugging, teaching explanations, and quality analysis.

**Why it exists:**
- Debug individual requests by examining the full decision path
- Generate human-readable explanations of how the system handled a request
- Track quality metrics (grounding rate, latency breakdown, token usage)
- Support the frontend "routing drawer" that visualizes request flow
- Enable eval analysis and performance optimization

**Location:** `backend/tracing.py`

---

## Key Types

### TraceLogger

The core class that accumulates trace data throughout a request lifecycle.

```python
# backend/tracing.py:16-51
class TraceLogger:
    def __init__(self) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.trace_id = str(uuid.uuid4())
        self.data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "timestamp": _utc_now(),
            "utterance": "",
            "intent": None,
            "routing_mode": None,
            "routing_confidence": None,
            "routing_candidates": [],
            "routing_extracted": {},
            "routing_missing_params": [],
            "tool_calls": [],
            "tool_latency_ms": [],
            "tool_params": [],
            "context_summary": {},
            "clarification": None,
            "grounding_valid": None,
            "formatter_used": None,
            "session_state": {},
            "prompt_name": None,
            "retry_count": 0,
            "latency_ms": {},
            "policy_gate": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model_version": None,
            "artifact_version": None,
            "prompt_version": None,
            "grounded_sources": [],
            "grounding_rate": 0.0,
            "router_diagnostics": {},
        }
```

### Trace Data Schema

Complete trace structure with all fields:

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | UUID v4 identifier |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `utterance` | `str` | Original user input |
| `intent` | `str` | Resolved intent (e.g., "activity", "positions") |
| `routing_mode` | `str` | Router that made the decision: "rules", "torch_classifier", or "llm_reroute" |
| `routing_confidence` | `float` | Confidence score (0.0-1.0) |
| `routing_candidates` | `list[dict]` | Top intent candidates with scores |
| `routing_extracted` | `dict` | Parameters extracted from utterance (symbol, account, timeframe) |
| `routing_missing_params` | `list[str]` | Required parameters not found |
| `tool_calls` | `list[dict]` | Tools invoked with name and source_id |
| `tool_latency_ms` | `list[int]` | Per-tool latency in milliseconds |
| `tool_params` | `list[dict]` | Parameters passed to each tool |
| `context_summary` | `dict` | Condensed context sent to LLM |
| `clarification` | `str | None` | Clarifying question if needed |
| `grounding_valid` | `bool | None` | Whether citations match tool sources |
| `formatter_used` | `str | None` | Deterministic formatter applied (or "none") |
| `session_state` | `dict` | Session context for follow-up handling |
| `prompt_name` | `str` | Prompt template file used |
| `retry_count` | `int` | LLM retry attempts |
| `latency_ms` | `dict` | Breakdown: routing_ms, tool_ms, llm_ms, postprocess_ms, total_ms |
| `policy_gate` | `dict` | Policy evaluation result (allowed, reason, tools) |
| `prompt_tokens` | `int` | LLM prompt token count |
| `completion_tokens` | `int` | LLM completion token count |
| `estimated_cost_usd` | `float` | Estimated API cost |
| `model_version` | `str` | OpenAI model used |
| `artifact_version` | `str` | Model bundle version (e.g., "model_bundle_v1") |
| `prompt_version` | `str` | Prompt template version |
| `grounded_sources` | `list[str]` | Citations in LLM response |
| `grounding_rate` | `float` | Ratio of citations to available sources |
| `router_diagnostics` | `dict` | Torch router state and debug info |

---

## Public API

### TraceLogger Methods

#### `start(utterance: str) -> str`
Initializes a trace with the user utterance. Returns the `trace_id`.

```python
# backend/tracing.py:53-55
def start(self, utterance: str) -> str:
    self.data["utterance"] = utterance
    return self.trace_id
```

#### `log_intent(intent: str) -> None`
Records the resolved intent.

```python
# backend/tracing.py:57-58
def log_intent(self, intent: str) -> None:
    self.data["intent"] = intent
```

#### `log_routing(mode: str, confidence: float, candidates: list[dict]) -> None`
Captures routing decision details.

```python
# backend/tracing.py:60-63
def log_routing(self, mode: str, confidence: float, candidates: list[dict[str, float]]) -> None:
    self.data["routing_mode"] = mode
    self.data["routing_confidence"] = confidence
    self.data["routing_candidates"] = candidates
```

#### `log_routing_detail(extracted: dict, missing: list[str]) -> None`
Records extracted parameters and missing required params.

```python
# backend/tracing.py:65-67
def log_routing_detail(self, extracted: dict[str, Any], missing: list[str]) -> None:
    self.data["routing_extracted"] = extracted
    self.data["routing_missing_params"] = missing
```

#### `log_tool(name: str, latency_ms: int, source_id: str) -> None`
Logs a tool invocation with timing.

```python
# backend/tracing.py:69-71
def log_tool(self, name: str, latency_ms: int, source_id: str) -> None:
    self.data["tool_calls"].append({"name": name, "source_id": source_id})
    self.data["tool_latency_ms"].append(latency_ms)
```

#### `log_tool_params(name: str, params: dict) -> None`
Records parameters passed to a tool.

```python
# backend/tracing.py:73-74
def log_tool_params(self, name: str, params: dict[str, Any]) -> None:
    self.data["tool_params"].append({"name": name, "params": params})
```

#### `log_tokens(prompt_tokens: int, completion_tokens: int, cost: float) -> None`
Captures LLM token usage and cost.

```python
# backend/tracing.py:76-79
def log_tokens(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
    self.data["prompt_tokens"] = prompt_tokens
    self.data["completion_tokens"] = completion_tokens
    self.data["estimated_cost_usd"] = cost
```

#### `log_grounding(citations: list[str], sources: list[str]) -> None`
Records grounding validation with rate calculation.

```python
# backend/tracing.py:81-84
def log_grounding(self, citations: list[str], sources: list[str]) -> None:
    self.data["grounded_sources"] = citations
    denom = max(1, len(sources))
    self.data["grounding_rate"] = round(len(citations) / denom, 2)
```

#### `log_grounding_validation(valid: bool) -> None`
Records whether citations match available sources.

```python
# backend/tracing.py:86-87
def log_grounding_validation(self, valid: bool) -> None:
    self.data["grounding_valid"] = valid
```

#### `log_context_summary(summary: dict) -> None`
Stores a condensed view of the context sent to LLM.

```python
# backend/tracing.py:89-90
def log_context_summary(self, summary: dict[str, Any]) -> None:
    self.data["context_summary"] = summary
```

#### `log_clarification(text: str) -> None`
Records a clarifying question.

```python
# backend/tracing.py:92-93
def log_clarification(self, text: str) -> None:
    self.data["clarification"] = text
```

#### `log_formatter_used(name: Optional[str]) -> None`
Records which deterministic formatter was applied.

```python
# backend/tracing.py:95-96
def log_formatter_used(self, name: Optional[str]) -> None:
    self.data["formatter_used"] = name
```

#### `log_session_state(state: dict) -> None`
Captures session state for follow-up context.

```python
# backend/tracing.py:98-99
def log_session_state(self, state: dict[str, Any]) -> None:
    self.data["session_state"] = state
```

#### `log_prompt_info(prompt_name: str) -> None`
Records the prompt template file name.

```python
# backend/tracing.py:101-102
def log_prompt_info(self, prompt_name: str) -> None:
    self.data["prompt_name"] = prompt_name
```

#### `log_retry_count(count: int) -> None`
Records LLM retry attempts.

```python
# backend/tracing.py:104-105
def log_retry_count(self, count: int) -> None:
    self.data["retry_count"] = count
```

#### `log_latency_breakdown(breakdown: dict[str, int]) -> None`
Stores latency measurements per pipeline stage.

```python
# backend/tracing.py:107-108
def log_latency_breakdown(self, breakdown: dict[str, int]) -> None:
    self.data["latency_ms"] = breakdown
```

#### `log_policy_gate(payload: dict) -> None`
Records policy evaluation results.

```python
# backend/tracing.py:110-111
def log_policy_gate(self, payload: dict[str, Any]) -> None:
    self.data["policy_gate"] = payload
```

#### `log_router_diagnostics(payload: dict) -> None`
Stores torch router debug info.

```python
# backend/tracing.py:113-114
def log_router_diagnostics(self, payload: dict[str, Any]) -> None:
    self.data["router_diagnostics"] = payload
```

#### `finalize(model_version: str, artifact_version: str, prompt_version: str) -> None`
Writes the complete trace to disk as JSON.

```python
# backend/tracing.py:116-120
def finalize(self, model_version: str, artifact_version: str, prompt_version: str) -> None:
    self.data["model_version"] = model_version
    self.data["artifact_version"] = artifact_version
    self.data["prompt_version"] = prompt_version
    (TRACE_DIR / f"{self.trace_id}.json").write_text(json.dumps(self.data, indent=2))
```

### Module Functions

#### `load_trace(trace_id: str) -> Optional[dict]`
Loads a trace from disk by ID.

```python
# backend/tracing.py:123-127
def load_trace(trace_id: str) -> Optional[dict[str, Any]]:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

---

## Internal Patterns

### Trace Directory

Traces are stored in `backend/traces/` as individual JSON files named `{trace_id}.json`. The directory is auto-created on first `TraceLogger` instantiation.

```python
# backend/tracing.py:9
TRACE_DIR = Path(__file__).parent / "traces"
```

### Timestamp Generation

All timestamps use UTC with ISO 8601 format:

```python
# backend/tracing.py:12-13
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
```

### Latency Measurement Pattern

The `/chat` endpoint uses `time.perf_counter()` for precise timing:

```python
# backend/main.py:186-192
overall_start = time.perf_counter()
routing_start = time.perf_counter()
route = route_intent(request.utterance, use_llm=True)
routing_ms = int((time.perf_counter() - routing_start) * 1000)
tool_latency = 0
llm_latency = 0
postprocess_latency = 0
```

Final breakdown logged before `finalize()`:

```python
# backend/main.py:497-505
total_ms = int((time.perf_counter() - overall_start) * 1000)
trace.log_latency_breakdown(
    {
        "routing_ms": routing_ms,
        "tool_ms": tool_latency,
        "llm_ms": llm_latency,
        "postprocess_ms": postprocess_latency,
        "total_ms": total_ms,
    }
)
```

### Grounding Validation Pattern

Citations are validated against available tool sources:

```python
# backend/main.py:493-495
trace.log_grounding(llm_response.citations, context.sources)
grounding_valid = (not llm_response.needs_clarification) and set(llm_response.citations) == set(context.sources)
trace.log_grounding_validation(grounding_valid)
```

---

## Integration Points

### 1. Chat Endpoint (`backend/main.py`)

The `/chat` handler creates a `TraceLogger` at request start and logs throughout:

```python
# backend/main.py:182-184
@app.post("/chat")
async def chat(request: ChatRequest):
    trace = TraceLogger()
    trace_id = trace.start(request.utterance)
```

Trace ID is included in the response:

```python
# backend/main.py:144-152
def _build_chat_response(llm: LLMResponse, trace_id: str) -> ChatResponse:
    return ChatResponse(
        answer_markdown=llm.answer_markdown,
        citations=llm.citations,
        confidence=llm.confidence,
        needs_clarification=llm.needs_clarification,
        clarifying_question=llm.clarifying_question,
        trace_id=trace_id,
    )
```

### 2. Debug Endpoint (`backend/main.py`)

Exposes raw trace JSON:

```python
# backend/main.py:539-544
@app.get("/debug/trace/{trace_id}")
async def get_trace(trace_id: str):
    data = load_trace(trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trace not found")
    return JSONResponse(content=data)
```

### 3. Teaching Explanation (`backend/teaching.py`)

Generates human-readable explanations from traces:

```python
# backend/teaching.py:9-27
def _build_prompt(trace: dict[str, Any]) -> str:
    summary = {
        "utterance": trace.get("utterance"),
        "intent": trace.get("intent"),
        "routing_mode": trace.get("routing_mode"),
        "routing_confidence": trace.get("routing_confidence"),
        "routing_extracted": trace.get("routing_extracted"),
        "routing_missing_params": trace.get("routing_missing_params"),
        "tool_calls": trace.get("tool_calls"),
        "tool_params": trace.get("tool_params"),
        "context_summary": trace.get("context_summary"),
        "grounded_sources": trace.get("grounded_sources"),
        "grounding_valid": trace.get("grounding_valid"),
        "formatter_used": trace.get("formatter_used"),
        "latency_ms": trace.get("latency_ms"),
        "policy_gate": trace.get("policy_gate"),
        "prompt_name": trace.get("prompt_name"),
        "retry_count": trace.get("retry_count"),
    }
    return (
        "You are explaining how a backend assistant handled a user request. "
        "Use the provided trace fields only. Write 4–6 sentences. "
        ...
    )
```

Explain endpoint:

```python
# backend/main.py:547-559
@app.post("/debug/trace/{trace_id}/explain")
async def explain_trace(trace_id: str):
    trace = load_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    model_config = _model_config()
    explanation = generate_teaching_explanation(
        trace,
        api_key=model_config.api_key,
        model=model_config.model,
        base_url=model_config.base_url,
    )
    return JSONResponse(content={"trace_id": trace_id, "explanation": explanation})
```

### 4. Frontend Routing Drawer (`frontend/app.js`)

The frontend fetches traces after each chat response and displays them in a debug panel:

```python
# frontend/app.js:341-347
if (payload.trace_id) {
    lastTraceId = payload.trace_id;
    ...
    const trace = await fetch(`/debug/trace/${payload.trace_id}`).then((res) => res.json());
    updateRoutingDrawer(trace);
}
```

The `updateRoutingDrawer()` function (lines 385-557) renders all trace fields:
- Routing mode and confidence
- Extracted parameters
- Tool calls with latency
- Context summary
- Grounding validation status
- Latency breakdown
- Policy gate results
- Token usage and cost

### 5. Response Schema (`backend/schemas.py`)

`ChatResponse` includes `trace_id` for frontend access:

```python
# backend/schemas.py:229-237
class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    clarifying_question: Optional[str]
    trace_id: str
```

---

## State Management

### Trace Lifecycle

1. **Creation:** `TraceLogger()` called at request start, generates UUID
2. **Accumulation:** Various `log_*` methods called as pipeline executes
3. **Finalization:** `finalize()` writes JSON to disk

Traces are append-only during a request. No trace modification after `finalize()`.

### Storage

- **Directory:** `backend/traces/`
- **Format:** JSON with 2-space indentation
- **Naming:** `{uuid}.json`
- **No retention policy:** Traces accumulate indefinitely

### Stateless Loading

`load_trace()` is stateless, reading directly from disk. No caching.

---

## Error Handling Patterns

### Missing Trace

Both endpoints return 404 for missing traces:

```python
if not data:
    raise HTTPException(status_code=404, detail="Trace not found")
```

### Tool Call Failure

When tool calls fail, the trace still captures the error path:

```python
# backend/main.py:290-313
except Exception as exc:
    llm = LLMResponse(
        answer_markdown="I could not retrieve the data needed to answer.",
        ...
    )
    trace.log_clarification("Can you double-check the request or try a different symbol?")
    trace.log_grounding([], [])
    trace.log_grounding_validation(False)
    trace.log_formatter_used("none")
    ...
    trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
```

### LLM Failure

LLM errors are caught and traced with zero tokens:

```python
# backend/main.py:469-484
except Exception:
    llm_response = LLMResponse(
        answer_markdown="I could not generate a grounded response at the moment.",
        ...
    )
    meta = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
        "retry_count": 0,
        "prompt_name": "response_system.txt",
    }
    trace.log_formatter_used("none")
```

---

## Example Trace

Complete trace from a successful request:

```json
{
  "trace_id": "983e0893-5415-4817-bbe1-50d588099134",
  "timestamp": "2026-01-28T22:12:42.551900+00:00",
  "utterance": "positions return ytd",
  "intent": "performance",
  "routing_mode": "torch_classifier",
  "routing_confidence": 0.9714298844337463,
  "routing_candidates": [
    {"intent": "positions_list", "score": 0.8},
    {"intent": "performance", "score": 0.8}
  ],
  "routing_extracted": {"timeframe": "YTD"},
  "routing_missing_params": [],
  "tool_calls": [
    {"name": "performance", "source_id": "tool:performance:v1"},
    {"name": "transfers", "source_id": "tool:transfers:v1"}
  ],
  "tool_latency_ms": [0, 0],
  "tool_params": [
    {"name": "performance", "params": {"timeframe": "YTD"}},
    {"name": "transfers", "params": {"account": null}}
  ],
  "context_summary": {
    "return_pct": 6.2,
    "contributions": 1750.0,
    "timeframe": "YTD",
    "account": "Brokerage",
    "as_of": "2026-01-15"
  },
  "clarification": null,
  "grounding_valid": true,
  "formatter_used": "performance",
  "session_state": {},
  "prompt_name": "response_system.txt",
  "retry_count": 0,
  "latency_ms": {
    "routing_ms": 1,
    "tool_ms": 0,
    "llm_ms": 1885,
    "postprocess_ms": 0,
    "total_ms": 1887
  },
  "policy_gate": {
    "allowed": true,
    "reason": "ok",
    "intent": "performance",
    "params": {"timeframe": "YTD"},
    "allowed_tools": ["performance", "transfers"]
  },
  "prompt_tokens": 593,
  "completion_tokens": 123,
  "estimated_cost_usd": 0.00962,
  "model_version": "gpt-5.2",
  "artifact_version": "model_bundle_v1",
  "prompt_version": "v4",
  "grounded_sources": ["tool:performance:v1", "tool:transfers:v1"],
  "grounding_rate": 1.0,
  "router_diagnostics": {
    "torch_available": true,
    "artifacts_present": true,
    "enabled": true,
    "confidence_threshold": 0.6,
    "force_enabled": false,
    "ngram_max": 2,
    "token_pattern": "[A-Za-z0-9']+",
    "torch_attempted": true,
    "torch_used": true,
    "torch_label": "performance",
    "torch_confidence": 0.9714
  }
}
```

---

## Common Issues / TODOs

### No Trace Retention Policy

Traces accumulate indefinitely in `backend/traces/`. Consider adding:
- Age-based cleanup
- Size limits
- Archival to cold storage

### Missing Fields in Legacy Traces

Older traces may lack newer fields (e.g., `router_diagnostics`, `policy_gate`). Code accessing traces should handle missing keys:

```javascript
// frontend/app.js:475-477
const routerDiagItems = keyValueList(trace.router_diagnostics);
fillList(routingFields.routerDiagnostics, routerDiagItems);
toggleSection("router-diagnostics", routerDiagItems.length === 0);
```

### Synchronous File Writes

`finalize()` performs synchronous file I/O on the request path. For high-throughput scenarios, consider:
- Async file writes
- Write batching
- In-memory buffer with background flush

### Trace ID in URL

Trace IDs are UUIDs exposed in URLs. While not sensitive, they allow enumeration if the debug endpoint is public. Consider:
- Auth gating `/debug/*` endpoints
- Shorter, time-limited trace IDs
