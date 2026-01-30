# Teaching Explanation System

## Overview

The Teaching Explanation system generates human-readable explanations of how the assistant handled user requests. It transforms raw trace data (JSON debug logs) into clear, educational narratives suitable for demos and debugging.

**Purpose:** Demonstrate to stakeholders how the NLU pipeline works, showing routing decisions, tool calls, grounding validation, and response generation in plain English.

**Location:** `backend/teaching.py`

---

## Key Types

### Input: Trace Dictionary

The teaching system receives a trace dict loaded from `backend/traces/{trace_id}.json`. The full trace schema is defined in `TraceLogger` (`backend/tracing.py:16-51`):

```python
# backend/tracing.py:20-51
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

### Output: Plain Text Explanation

A 4-6 sentence explanation covering:
- Routing decision (how intent was determined)
- Tool calls (which data sources were queried)
- Context building (how data was transformed)
- Grounding validation (citation correctness)
- Formatting (deterministic vs LLM-generated)

---

## Public API

### `generate_teaching_explanation(trace, api_key, model, base_url=None) -> str`

**Location:** `backend/teaching.py:37-60`

```python
def generate_teaching_explanation(
    trace: dict[str, Any],
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = _build_prompt(trace)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return a short teaching explanation."},
            {"role": "user", "content": prompt},
        ],
    )

    text = getattr(response, "output_text", "") or ""
    if not text and getattr(response, "output", None):
        for item in response.output:
            for content in getattr(item, "content", []):
                chunk = getattr(content, "text", "")
                if chunk:
                    text += chunk
    return text.strip()
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `trace` | `dict[str, Any]` | Full trace dictionary from `load_trace()` |
| `api_key` | `str` | OpenAI API key |
| `model` | `str` | Model identifier (e.g., `gpt-5-mini`, `gpt-5.2`) |
| `base_url` | `Optional[str]` | Custom API endpoint (default: `None`) |

**Returns:** Plain text explanation (4-6 sentences)

**Notes:**
- Uses OpenAI's `responses.create` API (not completions)
- Handles both `output_text` direct response and nested `output[].content[].text` structures
- No structured JSON output, just raw text

---

## Internal Functions

### `_build_prompt(trace) -> str`

**Location:** `backend/teaching.py:9-34`

Extracts a subset of trace fields and constructs the LLM prompt.

```python
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
        "Explain routing, tool calls, context building, grounding, and formatting. "
        "Be clear and teaching-oriented, but concise.\n\n"
        f"Trace JSON:\n{json.dumps(summary, indent=2)}"
    )
```

**Fields extracted for explanation:**

| Field | Purpose |
|-------|---------|
| `utterance` | Original user input |
| `intent` | Classified intent (e.g., `activity`, `positions`, `quotes`) |
| `routing_mode` | How intent was determined (`rules`, `torch`, `llm`) |
| `routing_confidence` | Confidence score (0.0-1.0) |
| `routing_extracted` | Extracted params (symbol, account, timeframe) |
| `routing_missing_params` | Params needed but not found |
| `tool_calls` | List of `{name, source_id}` tool invocations |
| `tool_params` | Parameters passed to each tool |
| `context_summary` | Compact view of built context |
| `grounded_sources` | Citations in response |
| `grounding_valid` | Whether citations match sources |
| `formatter_used` | Deterministic formatter name or `none` |
| `latency_ms` | Breakdown: routing, tool, llm, postprocess, total |
| `policy_gate` | Which tools were allowed for this intent |
| `prompt_name` | System prompt file used |
| `retry_count` | Number of LLM retries |

---

## HTTP Endpoint

### `POST /debug/trace/{trace_id}/explain`

**Location:** `backend/main.py:547-559`

```python
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

**Request:** `POST /debug/trace/{trace_id}/explain`

**Response:**
```json
{
  "trace_id": "abc123-...",
  "explanation": "The user asked 'What was my most recent trade?' The router classified this as an 'activity' intent with high confidence using rule-based matching..."
}
```

**Errors:**
- `404 Not Found`: Trace file doesn't exist at `backend/traces/{trace_id}.json`
- `500 Internal Server Error`: Missing `OPENAI_API_KEY` env var

---

## Integration Points

### Dependency: TraceLogger

Traces are created during `/chat` requests via `TraceLogger` (`backend/tracing.py`):

```python
# backend/main.py:183-184
trace = TraceLogger()
trace_id = trace.start(request.utterance)
```

Throughout the chat handler, trace data is logged:

```python
# backend/main.py:218-222
trace.log_intent(route.intent.value)
trace.log_routing(route.routing_mode, route.confidence, route.candidates)
trace.log_routing_detail(route.extracted, route.missing_params)
trace.log_router_diagnostics(route.routing_meta or torch_router_status())
trace.log_policy_gate(_policy_gate(route.intent, route.extracted))
```

And finalized to disk:

```python
# backend/main.py:506
trace.finalize(model_config.model, "model_bundle_v1", MANIFEST["prompt_version"])
```

### Dependency: load_trace

```python
# backend/tracing.py:123-127
def load_trace(trace_id: str) -> Optional[dict[str, Any]]:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

### Dependency: ModelConfig

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

---

## Flow Diagram

```
User hits /debug/trace/{id}/explain
           │
           ▼
    load_trace(trace_id)
           │
           ├── Not found → 404
           │
           ▼
    _model_config() → ModelConfig(api_key, model, base_url)
           │
           ▼
    generate_teaching_explanation(trace, ...)
           │
           ├── _build_prompt(trace) → extracts 16 key fields
           │
           ├── OpenAI.responses.create() → LLM call
           │
           ▼
    Return {"trace_id": ..., "explanation": ...}
```

---

## Example Trace Input

```json
{
  "trace_id": "0712a245-5223-444f-a7e4-accb5787f113",
  "timestamp": "2026-01-27T19:37:01.599391+00:00",
  "utterance": "What was my most recent trade?",
  "intent": "activity",
  "routing_mode": "rules",
  "routing_confidence": 0.95,
  "routing_extracted": {},
  "routing_missing_params": [],
  "tool_calls": [
    {"name": "activity", "source_id": "tool:activity:v1"}
  ],
  "tool_params": [
    {"name": "activity", "params": {}}
  ],
  "context_summary": {"most_recent_trade": {"symbol": "AAPL", "side": "buy"}},
  "grounded_sources": ["tool:activity:v1"],
  "grounding_valid": true,
  "formatter_used": "activity",
  "latency_ms": {"routing_ms": 5, "tool_ms": 1, "llm_ms": 450, "total_ms": 460},
  "policy_gate": {"allowed": true, "intent": "activity", "allowed_tools": ["activity"]},
  "prompt_name": "response_system.txt",
  "retry_count": 0,
  "prompt_tokens": 397,
  "completion_tokens": 106,
  "estimated_cost_usd": 0.00715,
  "model_version": "gpt-5.2",
  "artifact_version": "model_bundle_v1",
  "prompt_version": "v1",
  "grounding_rate": 1.0
}
```

---

## Example Explanation Output

> The user asked "What was my most recent trade?" The rule-based router classified this as an "activity" intent with 95% confidence. A single tool call was made to the activity data source. The context builder extracted the most recent trade (AAPL buy) for the LLM prompt. Grounding validation passed since the response cited the correct source. The deterministic "activity" formatter was applied to ensure consistent output.

---

## Error Handling

### No Explicit Error Handling in `generate_teaching_explanation`

The function does not catch exceptions. If the OpenAI call fails:
- Network errors propagate to the endpoint
- Invalid API key results in OpenAI SDK exception
- The endpoint returns a 500 with the exception details

### Missing Trace Fields

The `_build_prompt` function uses `.get()` with no defaults, so missing fields become `null` in the JSON passed to the LLM. The LLM prompt instructs: "Use the provided trace fields only."

---

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `_model_config()` | Required. API key for explanation generation |
| `OPENAI_MODEL` | `_model_config()` | Model ID (default: from manifest) |
| `OPENAI_BASE_URL` | `_model_config()` | Custom endpoint for self-hosted models |

---

## Common Issues

### 1. Sparse Traces

Some older traces may lack fields like `routing_mode`, `context_summary`, or `policy_gate` if they were created before those logging calls were added. The explanation will have incomplete information.

**Mitigation:** The LLM prompt says "Use the provided trace fields only" so it adapts to available data.

### 2. Rate Limiting

Each explanation triggers an LLM call. Rapid requests to `/explain` could hit rate limits.

**Mitigation:** Consider caching explanations or adding rate limiting at the endpoint level.

### 3. Response Parsing Fragility

The code handles two response formats:
```python
text = getattr(response, "output_text", "") or ""
if not text and getattr(response, "output", None):
    for item in response.output:
        for content in getattr(item, "content", []):
            chunk = getattr(content, "text", "")
```

This is defensive against SDK version differences but adds complexity.

---

## Testing

To manually test:

```bash
# 1. Make a chat request to generate a trace
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"utterance": "What is my AAPL position?"}'

# 2. Note the trace_id from the response

# 3. Get the explanation
curl -X POST http://localhost:8000/debug/trace/{trace_id}/explain
```

---

## Related Systems

- **Tracing** (`backend/tracing.py`): Creates and stores trace data
- **Responder** (`backend/responder.py`): Generates LLM responses (similar OpenAI pattern)
- **Eval** (`backend/eval.py`): Uses traces for quality measurement
