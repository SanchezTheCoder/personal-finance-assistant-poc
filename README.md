# Personal Finance Assistant POC (Simulated)

This is a **fully simulated** Personal Finance Assistant demo. It does **not** use any real company data or APIs.

## Quick start

```bash
cd /path/to/personal-finance-assistant-poc
uv venv
uv sync

export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-5.2

uv run uvicorn backend.main:app --reload
```

Open the UI at: http://127.0.0.1:8000 (or http://127.0.0.1:8000/ui)

## Environment variables
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default: `gpt-5.2`)
- `OPENAI_BASE_URL` (optional; for compatible endpoints)

## Demo script (2-3 minutes)
1. Ask: "What was my most recent trade?" (activity intent)
2. Ask: "How many shares of AAPL do I own?" (positions intent)
3. Ask: "How did my portfolio do YTD?" (performance intent)
4. Ask: "What's MSFT price and today's change?" (quotes intent)
5. Ask: "What is a Roth IRA?" (facts intent)
6. Open a trace: `GET /debug/trace/<trace_id>` to show grounding and metrics
7. Run eval: `POST /eval` to show pass rate and grounding rate

## API endpoints
- `POST /chat` { utterance, account?, stream? }
- `GET /debug/trace/{trace_id}`
- `POST /eval` (latency budget: 5000ms)
- Tools: `/api/activity`, `/api/positions`, `/api/performance`, `/api/quotes`, `/api/facts`

## Synthetic data
The single source of truth for mock data is:
- `backend/data/user_master.json`

Tools read directly from the master file at runtime. If you want the per-tool JSON files for inspection or demos, run:

```bash
python scripts/generate_tool_json.py
```

This regenerates:
`backend/data/account.json`, `activity.json`, `positions.json`, `quotes.json`, `performance.json`, `transfers.json`

## Optional: PyTorch intent router (synthetic training)
This POC can use a lightweight PyTorch classifier for intent routing before falling back to the LLM.

Generate synthetic training data:
```bash
python3 scripts/generate_intent_dataset.py
```

Train the router (requires PyTorch):
```bash
uv pip install ".[ml]"
python3 scripts/train_intent_router.py
```

Artifacts are saved to:
`backend/artifacts/model_bundle_v1/intent_router/`

Enable the PyTorch router at runtime:
```bash
export TORCH_ROUTER_MODEL_PATH=backend/artifacts/model_bundle_v1/intent_router/intent_router.pt
export TORCH_ROUTER_VOCAB_PATH=backend/artifacts/model_bundle_v1/intent_router/intent_router_vocab.json
export TORCH_ROUTER_LABELS_PATH=backend/artifacts/model_bundle_v1/intent_router/intent_router_labels.json
export TORCH_ROUTER_CONF_THRESHOLD=0.6
```

## Notes
- This POC uses a **rule-based intent router** and **minimal tool calls** (one tool per intent).
- The LLM only sees compact, structured context from tool results and must return a JSON object with citations.
- If citations are missing or invalid, the system retries once with stricter constraints.
- If no API key is available, the system returns a clear fallback message.

## Routing at scale (real-world pattern)
Production systems typically use a multi-stage routing stack to keep latency low and accuracy high:
- **Stage 1: deterministic rules** for high-precision intents (low cost, low latency).
- **Stage 2: lightweight classifier** for ambiguous queries (small model, strict JSON output).
- **Stage 3: policy gate** to enforce entitlements, account scope, and tool allowlists.
- **Stage 4: orchestration** that minimizes tool fan-out and reuses cached results.
- **Stage 5: observability** on routing confidence, fallbacks, and tool usage.

**POC mapping:** this demo mirrors that flow with a rule-first router, optional LLM reroute on ambiguity, strict tool allowlists, and trace logs that capture routing confidence + tool fan-out.
