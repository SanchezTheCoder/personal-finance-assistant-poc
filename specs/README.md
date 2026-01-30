# Personal Finance Assistant POC - System Index (PIN)

> Quick lookup table for Ralph Wiggum Loop. Find synonyms, files, and specs.

## Quick Reference

```bash
# Start backend
uv run uvicorn backend.main:app --reload --port 8000

# Run eval suite
curl -X POST http://localhost:8000/eval

# Train intent router (requires ml deps)
uv run python scripts/train_intent_router.py

# Generate intent dataset
uv run python scripts/generate_intent_dataset.py
```

---

## Systems

### Intent Router

**Terms:** router, routing, intent, classification, dispatch, NLU, torch router, rule-based, LLM reroute

**Files:**
- `backend/router.py` - Rule-based intent routing, keyword scoring, symbol/param extraction
- `backend/router_llm.py` - LLM fallback reroute when rules are ambiguous
- `backend/torch_router.py` - PyTorch bag-of-words intent classifier
- `backend/schemas.py:9-34` - `Intent` enum and `IntentRoute` model
- `ml/intent_templates.json` - Training data templates
- `ml/intent_*.jsonl` - Train/val/test splits
- `scripts/train_intent_router.py` - Training script for torch router
- `scripts/generate_intent_dataset.py` - Synthetic dataset generation

**Description:** Three-layer intent routing: (1) rule-based keyword scorer, (2) PyTorch BoW classifier for ambiguous cases, (3) LLM fallback. Extracts symbols, accounts, timeframes from utterances. Returns `IntentRoute` with confidence, extracted params, and missing params for clarification.

**Spec:** [specs/intent-router.md](intent-router.md)

---

### Tool Layer

**Terms:** tools, data, positions, quotes, activity, transfers, performance, account summary, facts, tool registry

**Files:**
- `backend/tools.py` - Tool functions and `ToolRegistry` dispatcher
- `backend/data/user_master.json` - Unified mock data source
- `backend/data/*.json` - Legacy per-domain JSON files
- `backend/data/facts/*.md` - Static fact snippets (Roth IRA, ETF basics, rebalancing)

**Description:** Simulated brokerage data layer. `ToolRegistry.call_tool(intent, **params)` dispatches to `get_*` functions. Returns `ToolResult` with `source_id`, `data`, and `as_of` timestamp. Used for grounding LLM responses.

**Spec:** [specs/tool-layer.md](tool-layer.md)

---

### Context Builder

**Terms:** context, grounding, ContextBundle, symbol performance, portfolio ranking

**Files:**
- `backend/context_builder.py` - Transforms `ToolResult` into typed context objects
- `backend/schemas.py:44-208` - Context models (`ActivityContext`, `PositionsContext`, etc.)

**Description:** Bridges tool results to LLM prompts. Handles single-tool contexts (positions, quotes) and composite contexts (symbol_performance = positions + quotes, portfolio_ranking = positions_list + quotes). Computes derived values like unrealized P/L.

**Spec:** [specs/Systems.md](Systems.md#context-builder)

---

### Responder (LLM Layer)

**Terms:** responder, LLM, OpenAI, generate, response, model config, grounding, validation, retry

**Files:**
- `backend/responder.py` - LLM call wrapper with retry and grounding validation
- `backend/artifacts/model_bundle_v1/prompts/response_system.txt` - System prompt
- `backend/artifacts/model_bundle_v1/prompts/response_user.txt` - User prompt template
- `backend/artifacts/model_bundle_v1/manifest.json` - Model/prompt versions

**Description:** Calls OpenAI API with structured JSON output. Validates citations match tool sources. Retries on parse/grounding failure. Tracks tokens and estimated cost.

**Spec:** [specs/Systems.md](Systems.md#responder-llm-layer)

---

### Formatter

**Terms:** formatter, answer formatting, deterministic, template, output

**Files:**
- `backend/formatter.py` - Deterministic answer templates per intent
- `backend/responder.py:84-113` - `clean_answer_markdown`, `strip_markdown`

**Description:** Post-LLM formatting layer. Generates consistent, readable answers with reasoning sections. Bypasses LLM variability for known intents. Falls back to LLM raw output for complex cases.

**Spec:** [specs/formatter.md](formatter.md)

---

### Tracing

**Terms:** trace, tracing, debug, observability, latency, grounding rate

**Files:**
- `backend/tracing.py` - `TraceLogger` class, trace file persistence
- `backend/traces/` - JSON trace files (runtime generated)
- `backend/main.py:539-544` - `/debug/trace/{trace_id}` endpoint

**Description:** Per-request trace logging. Captures utterance, routing decision, tool calls, LLM tokens, latency breakdown, grounding validation. Stored as JSON for debugging and teaching explanations.

**Spec:** [`specs/Tracing.md`](Tracing.md)

---

### Eval System

**Terms:** eval, evaluation, golden set, pass rate, grounding rate, tool minimality

**Files:**
- `backend/eval.py` - `EvalRunner` class, golden test set
- `backend/main.py:562-565` - `/eval` endpoint

**Description:** Automated quality checks against golden utterances. Measures intent accuracy, tool minimality, grounding correctness, and latency budget. Returns pass rate and detailed results.

**Spec:** [specs/eval.md](eval.md)

---

### Teaching Explanation

**Terms:** teaching, explain, trace explanation, LLM-generated explanation

**Files:**
- `backend/teaching.py` - `generate_teaching_explanation()` function
- `backend/main.py:547-559` - `/debug/trace/{trace_id}/explain` endpoint

**Description:** Uses LLM to generate human-readable explanations of trace data for demo purposes. Shows how routing, tools, and grounding work together.

**Spec:** [specs/teaching.md](teaching.md)

---

### API Layer (FastAPI)

**Terms:** API, FastAPI, endpoints, chat, REST, streaming

**Files:**
- `backend/main.py` - FastAPI app, `/chat` endpoint, REST endpoints
- `backend/schemas.py:220-237` - `ChatRequest`, `ChatResponse`

**Description:** Main entry point. `/chat` POST handles NLU pipeline. REST endpoints (`/api/*`) expose raw tool data. Supports optional SSE streaming (disabled by default). Serves frontend static files.

**Spec:** [specs/api-layer.md](api-layer.md)

---

### Frontend

**Terms:** frontend, UI, chat, sidebar, routing drawer, iMessage

**Files:**
- `frontend/index.html` - HTML structure
- `frontend/app.js` - Chat logic, sidebar data loaders, routing drawer
- `frontend/styles.css` - Styling

**Description:** Single-page chat interface with iMessage-style bubbles. Sidebar shows account summary, positions, activity, transfers. Routing drawer displays trace details for debugging.

**Spec:** [specs/frontend.md](frontend.md)

---

### ML Training Pipeline

**Terms:** ML, training, intent dataset, PyTorch, bag-of-words

**Files:**
- `scripts/train_intent_router.py` - BoW classifier training
- `scripts/generate_intent_dataset.py` - Synthetic data generation
- `ml/intent_templates.json` - Utterance templates per intent
- `ml/intent_*.jsonl` - Generated train/val/test data

**Description:** Generates synthetic utterances from templates. Trains lightweight PyTorch linear classifier on bag-of-words features with optional bigrams.

**Spec:** [specs/ml-pipeline.md](ml-pipeline.md)

---

## Key Patterns

1. **Intent-first architecture:** Route utterance to intent, then call minimal tools, then format/generate response.

2. **Grounding validation:** LLM citations must match tool `source_id` values. Retry on mismatch.

3. **Three-layer routing:** Rules (fast, deterministic) -> Torch (learned, fast) -> LLM (fallback, expensive).

4. **Deterministic formatters:** Bypass LLM variability for known intents with template-based answers.

5. **Trace-driven debugging:** Every request logged with full pipeline state for replay and explanation.

6. **Composite contexts:** Some intents (symbol_performance, portfolio_ranking, performance) require multiple tool calls merged into a single context.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-mini` | Model for response generation |
| `OPENAI_BASE_URL` | (none) | Custom API endpoint |
| `ROUTER_MODEL` | `gpt-5-mini` | Model for LLM reroute fallback |
| `ROUTER_CONF_THRESHOLD` | `0.75` | Confidence threshold before LLM fallback |
| `TORCH_ROUTER_CONF_THRESHOLD` | `0.6` | Confidence threshold for torch router |
| `TORCH_ROUTER_FORCE` | `false` | Force torch router regardless of confidence |
