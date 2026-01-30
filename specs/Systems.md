# Systems Architecture Specification

> Complete technical specification of all systems in the Personal Finance Assistant POC.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Request Pipeline](#request-pipeline)
3. [Intent Router](#intent-router)
4. [Tool Layer](#tool-layer)
5. [Context Builder](#context-builder)
6. [Responder (LLM Layer)](#responder-llm-layer)
7. [Formatter](#formatter)
8. [Tracing](#tracing)
9. [Eval System](#eval-system)
10. [Teaching Explanation](#teaching-explanation)
11. [ML Training Pipeline](#ml-training-pipeline)
12. [Schema Reference](#schema-reference)
13. [Error Handling Patterns](#error-handling-patterns)
14. [Common Issues](#common-issues)

---

## System Overview

The Personal Finance Assistant is an intent-first NLU pipeline that routes user utterances to financial data tools, builds grounded context, and generates validated responses. The architecture prioritizes:

1. **Determinism over randomness** - Rule-based routing with LLM fallback
2. **Grounding validation** - LLM citations must match tool sources
3. **Trace-driven debugging** - Full pipeline state captured per request
4. **Minimal tool calls** - Only fetch data required for the intent

### High-Level Flow

```
Utterance → Router → Tool Registry → Context Builder → Responder → Formatter → Response
              ↓            ↓              ↓               ↓            ↓
          TraceLogger  TraceLogger   TraceLogger     TraceLogger  TraceLogger
```

---

## Request Pipeline

The main entry point is `POST /chat` in `backend/main.py`. Here's the complete flow:

### Entry Point (`backend/main.py:181-536`)

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    trace = TraceLogger()
    trace_id = trace.start(request.utterance)

    overall_start = time.perf_counter()
    routing_start = time.perf_counter()
    route = route_intent(request.utterance, use_llm=True)
    routing_ms = int((time.perf_counter() - routing_start) * 1000)
```

### Request Model (`backend/schemas.py:220-226`)

```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance: str
    account: Optional[str] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
```

### Response Model (`backend/schemas.py:229-237`)

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

### Session State Management

Session state tracks prior intents for multi-turn clarification:

```python
session_state: dict[str, dict[str, str]] = {}

def _get_session_state(session_id: Optional[str]) -> dict[str, str]:
    if not session_id:
        return {}
    return session_state.get(session_id, {})
```

When the router returns `Intent.clarify`, the system checks session state to infer intent from context:

```python
if route.intent is Intent.clarify:
    prior = _get_session_state(request.session_id)
    if prior.get("intent") in {"positions_list", "positions"}:
        route.intent = Intent.positions_list
        # ... extract asset_class from utterance
```

### Policy Gate

Every request passes through a policy gate that validates allowed tools per intent:

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

---

## Intent Router

The router is a three-layer system: rules → PyTorch classifier → LLM fallback.

### Intent Enum (`backend/schemas.py:9-21`)

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

### IntentRoute Model (`backend/schemas.py:24-34`)

```python
class IntentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    missing_params: List[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    candidates: List[dict[str, Any]] = Field(default_factory=list)
    routing_mode: str = "rules"  # "rules", "torch_classifier", or "llm_reroute"
    routing_meta: dict[str, Any] = Field(default_factory=dict)
```

### Layer 1: Rule-Based Router (`backend/router.py:462-497`)

The rule router scores intents via keyword matching:

```python
def rule_route(utterance: str) -> IntentRoute:
    text = _normalize(utterance)
    candidates = _score_candidates(text, utterance)
    if not candidates:
        return IntentRoute(
            intent=Intent.clarify,
            confidence=0.2,
            missing_params=["intent"],
            extracted={},
            candidates=[],
            routing_mode="rules",
        )

    top = candidates[0]
    intent = Intent(top["intent"])
    intent, extracted, missing = _apply_extraction(intent, text, utterance)
```

### Keyword Scoring (`backend/router.py:289-378`)

```python
def _score_candidates(text: str, utterance: str) -> list[dict[str, float]]:
    scores: dict[str, float] = {
        "activity": 0.0,
        "positions": 0.0,
        # ... all intents initialized to 0.0
    }

    def bump(intent: str, amount: float) -> None:
        scores[intent] = min(1.0, scores[intent] + amount)

    # Keyword triggers
    if any(k in text for k in ["recent trade", "last trade", "most recent trade"]):
        bump("activity", 0.9)

    if any(k in text for k in ["how many shares", "do i own"]):
        bump("positions", 0.9)

    # Ranking detection
    ranking_trigger = any(k in text for k in RANK_KEYWORDS)
    portfolio_context = any(k in text for k in ["position", "holdings", "portfolio"])
    if ranking_trigger and portfolio_context:
        bump("portfolio_ranking", 0.92)
```

### Symbol Extraction (`backend/router.py:228-277`)

```python
SYMBOL_RE = re.compile(r"\b[A-Z]{1,5}\b")
KNOWN_SYMBOLS = _load_known_symbols()  # Loaded from user_master.json

def _extract_symbol(text: str, allow_unknown: bool = False) -> str | None:
    # Prefer explicit uppercase tokens (e.g., AAPL)
    direct_matches = SYMBOL_RE.findall(text)
    for m in direct_matches:
        symbol = m.upper()
        if symbol in SYMBOL_STOPWORDS:
            continue
        if symbol in KNOWN_SYMBOLS or allow_unknown:
            return symbol

    # Check for $-prefixed tickers ($AAPL)
    for match in re.finditer(r"\$([A-Za-z]{1,5})\b", text):
        symbol = match.group(1).upper()
        if symbol not in SYMBOL_STOPWORDS:
            return symbol
```

### Layer 2: PyTorch Classifier (`backend/torch_router.py`)

When rule confidence is low or ambiguous, the torch router attempts classification:

```python
def torch_attempt(utterance: str, candidates: list[dict[str, float]]) -> tuple[Optional[IntentRoute], dict[str, object]]:
    router = _load_router(model_path, vocab_path, labels_path)
    if router is None:
        return None, meta

    label, confidence = router.predict(utterance)

    min_conf = float(status["confidence_threshold"])  # Default 0.6
    force = bool(status["force_enabled"])
    if confidence < min_conf and not force:
        return None, meta

    route = IntentRoute(
        intent=intent,
        confidence=confidence,
        routing_mode="torch_classifier",
    )
    return route, meta
```

### Torch Model Architecture (`backend/torch_router.py:27-33`)

```python
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

Bag-of-words encoding with optional bigrams:

```python
def _encode(self, text: str) -> torch.Tensor:
    vec = torch.zeros(len(self.vocab), dtype=torch.float32)
    tokens = [t.lower() for t in self.token_re.findall(text)]
    features = list(tokens)
    if self.ngram_max >= 2:
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")
    for feature in features:
        idx = self.vocab.get(feature)
        if idx is not None:
            vec[idx] += 1.0
    return vec
```

### Layer 3: LLM Fallback (`backend/router_llm.py:17-69`)

```python
def llm_reroute(
    utterance: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    candidates: list[dict[str, float]],
) -> Optional[IntentRoute]:
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        "You are an intent router. Choose the single best intent for the user query.\n"
        "Return ONLY JSON in this schema: {\"intent\": <string>, \"extracted\": {}}.\n"
        "Valid intents: activity, positions, positions_list, portfolio_ranking, "
        "symbol_performance, performance, quotes, facts, transfers, account_value, cash_balance.\n"
        "User query: " + utterance + "\n"
        "Candidates (intent:score): " + ", ".join([f"{c['intent']}:{c['score']}" for c in candidates])
    )
```

### Router Orchestration (`backend/router.py:500-565`)

```python
def route_intent(utterance: str, use_llm: bool = True) -> IntentRoute:
    rule_result = rule_route(utterance)

    threshold = float(os.getenv("ROUTER_CONF_THRESHOLD", "0.75"))
    candidates = rule_result.candidates
    ambiguous = len(candidates) >= 2 and abs(candidates[0]["score"] - candidates[1]["score"]) < 0.1

    if not use_llm:
        return rule_result

    if rule_result.confidence >= threshold and not ambiguous:
        return rule_result

    # Try torch router
    torch_route, torch_meta = torch_attempt(utterance, candidates)
    if torch_route:
        # Apply extraction to torch result
        text = _normalize(utterance)
        intent, extracted, missing = _apply_extraction(torch_route.intent, text, utterance)
        torch_route.intent = intent
        torch_route.extracted.update(extracted)
        return torch_route

    # Fall back to LLM
    reroute = llm_reroute(utterance, api_key, model, base_url, candidates)
    # ...
```

---

## Tool Layer

The tool layer provides simulated brokerage data via a unified JSON file.

### ToolResult Model (`backend/schemas.py:36-41`)

```python
class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    data: dict[str, Any]
    as_of: str
```

### Data Loading (`backend/tools.py:15-33`)

```python
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

### Tool Functions (`backend/tools.py:36-92`)

```python
def get_activity(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("activity")
    return ToolResult(source_id="tool:activity:v1", data=payload, as_of=payload["as_of"])

def get_positions(symbol: str, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    return ToolResult(source_id="tool:positions:v1", data=payload, as_of=payload["as_of"])

def get_positions_list(asset_class: Optional[str] = None, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    if asset_class:
        filtered = [p for p in payload["positions"] if p.get("asset_class") == asset_class]
        payload = {**payload, "positions": filtered, "asset_class_filter": asset_class}
    return ToolResult(source_id="tool:positions_list:v1", data=payload, as_of=payload["as_of"])

def get_quotes(symbol: str) -> ToolResult:
    payload = _get_section("quotes")
    return ToolResult(source_id="tool:quotes:v1", data=payload, as_of=payload["as_of"])

def get_facts(topic: str) -> ToolResult:
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

### ToolRegistry Dispatcher (`backend/tools.py:93-113`)

```python
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
        if intent in {"account_value", "cash_balance"}:
            return get_account_summary(account=params.get("account"))
        raise ValueError(f"Unknown intent: {intent}")
```

---

## Context Builder

Transforms raw tool results into typed context objects for LLM prompts.

### ContextBundle Model (`backend/schemas.py:190-207`)

```python
class ContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    context: Union[
        ActivityContext,
        PositionsContext,
        PositionsListContext,
        PortfolioRankingContext,
        SymbolPerformanceContext,
        TransfersContext,
        AccountValueContext,
        CashBalanceContext,
        PerformanceContext,
        QuoteContext,
        FactsContext,
    ]
    sources: List[str]
```

### Single-Tool Context Building (`backend/context_builder.py:55-158`)

```python
def build_context(intent: Intent, tool_result: ToolResult, params: dict[str, Any]) -> ContextBundle:
    data = tool_result.data
    if intent is Intent.activity:
        trades = data["trades"]
        most_recent = max(trades, key=lambda t: _parse_ts(t["timestamp"]))
        context = ActivityContext(
            most_recent_trade=ActivityTrade(**most_recent),
            account=data.get("account", "Brokerage"),
            as_of=data["as_of"],
        )
    elif intent is Intent.positions:
        symbol = params["symbol"]
        position = next((p for p in data["positions"] if p["symbol"] == symbol), None)
        if not position:
            raise ValueError(f"No position for symbol {symbol}")
        context = PositionsContext(
            symbol=position["symbol"],
            quantity=position["quantity"],
            cost_basis=position["cost_basis"],
            account=data.get("account", "Brokerage"),
            as_of=data["as_of"],
        )
    # ... other intents

    return ContextBundle(intent=intent, context=context, sources=[tool_result.source_id])
```

### Composite Context: Symbol Performance (`backend/context_builder.py:161-196`)

```python
def build_symbol_performance_context(
    positions_result: ToolResult,
    quotes_result: ToolResult,
    symbol: str,
) -> ContextBundle:
    positions = positions_result.data["positions"]
    position = next((p for p in positions if p["symbol"] == symbol), None)
    if not position:
        raise ValueError(f"No position for symbol {symbol}")

    quotes = quotes_result.data["quotes"]
    quote = next((q for q in quotes if q["symbol"] == symbol), None)
    if not quote:
        raise ValueError(f"No quote for symbol {symbol}")

    cost_basis = position["cost_basis"]
    current_price = quote["price"]
    quantity = position["quantity"]
    unrealized_pl = (current_price - cost_basis) * quantity
    unrealized_pl_pct = (current_price - cost_basis) / cost_basis

    context = SymbolPerformanceContext(
        symbol=symbol,
        quantity=quantity,
        cost_basis=cost_basis,
        current_price=current_price,
        unrealized_pl=round(unrealized_pl, 2),
        unrealized_pl_pct=round(unrealized_pl_pct, 4),
        as_of=quotes_result.as_of,
    )

    return ContextBundle(
        intent=Intent.symbol_performance,
        context=context,
        sources=[positions_result.source_id, quotes_result.source_id],
    )
```

### Composite Context: Portfolio Ranking (`backend/context_builder.py:199-255`)

```python
def build_portfolio_ranking_context(
    positions_result: ToolResult,
    quotes_result: ToolResult,
    direction: str = "best",
    basis: str = "unrealized_pl",
) -> ContextBundle:
    positions = positions_result.data["positions"]
    quotes = quotes_result.data["quotes"]
    quotes_map = {q["symbol"]: q["price"] for q in quotes}
    items: list[PositionPerformanceItem] = []
    missing: list[str] = []

    for position in positions:
        symbol = position["symbol"]
        if symbol not in quotes_map:
            missing.append(symbol)
            continue
        cost_basis = float(position["cost_basis"])
        current_price = float(quotes_map[symbol])
        quantity = int(position["quantity"])
        unrealized_pl = (current_price - cost_basis) * quantity
        unrealized_pl_pct = (current_price - cost_basis) / cost_basis if cost_basis else 0.0
        items.append(PositionPerformanceItem(...))

    reverse = direction != "worst"
    key_fn = (lambda x: x.unrealized_pl_pct) if basis == "unrealized_pl_pct" else (lambda x: x.unrealized_pl)
    ranked = sorted(items, key=key_fn, reverse=reverse)
```

---

## Responder (LLM Layer)

Calls OpenAI API with structured JSON output and validates grounding.

### ModelConfig (`backend/responder.py:27-31`)

```python
@dataclass
class ModelConfig:
    api_key: str
    model: str
    base_url: Optional[str]
```

### JSON Schema for Response (`backend/responder.py:56-77`)

```python
def _json_schema() -> dict[str, Any]:
    return {
        "name": "LLMResponse",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_markdown": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
            },
            "required": [
                "answer_markdown",
                "citations",
                "confidence",
                "needs_clarification",
                "clarifying_question",
            ],
        },
    }
```

### Prompt Building (`backend/responder.py:38-42`)

```python
def _build_user_prompt(question: str, context: ContextBundle, citations: Iterable[str]) -> str:
    template = _load_prompt("response_user.txt")
    return template.replace("{{question}}", question).replace(
        "{{context_json}}", json.dumps(context.model_dump(), indent=2)
    ).replace("{{valid_citations}}", ", ".join(citations))
```

### System Prompt (`backend/artifacts/model_bundle_v1/prompts/response_system.txt`)

```
You are a financial assistant for a simulated demo. You must follow these rules:
- Only answer using the provided structured context.
- If the context does not contain the answer, say you do not have enough data and ask a clarifying question.
- Do not guess or use external knowledge.
- Output MUST be a single JSON object that matches the required schema.
- You MUST include citations that match the provided tool source_ids when you provide an answer.
- If multiple tool sources are provided, your citations MUST include all of them.
...
```

### Grounding Validation (`backend/responder.py:160-165`)

```python
def validate_response(llm_response: LLMResponse, valid_sources: list[str]) -> bool:
    if llm_response.needs_clarification:
        return True
    if not llm_response.citations:
        return False
    return set(llm_response.citations) == set(valid_sources)
```

### Response Generation with Retry (`backend/responder.py:168-222`)

```python
def generate_response(
    utterance: str,
    context: ContextBundle,
    model_config: ModelConfig,
    max_retries: int = 1,
) -> Tuple[LLMResponse, dict[str, Any]]:
    client = OpenAI(api_key=model_config.api_key, base_url=model_config.base_url)
    system_prompt = _load_prompt(PROMPT_NAME)
    user_prompt = _build_user_prompt(utterance, context, context.sources)

    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            llm_response, meta = _call_openai(client, model_config.model, system_prompt, user_prompt)
            if validate_response(llm_response, context.sources):
                meta["retry_count"] = attempt - 1
                return llm_response, meta
            last_error = "Missing or invalid citations"
        except ResponseError as exc:
            last_error = str(exc)

        # Append retry instruction to prompt
        system_prompt = (
            system_prompt
            + "\n\nIMPORTANT: Your previous output failed validation."
            + "\nReturn ONLY valid JSON that matches the schema, and include citations that match the valid list."
        )
        time.sleep(0.05)

    raise ResponseError(f"Failed to get valid response: {last_error}")
```

### Cost Estimation (`backend/responder.py:19-24`, `45-53`)

```python
PRICE_TABLE = {
    "gpt-5.2": {"input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-5": {"input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-5-mini": {"input_per_1m": 3.0, "output_per_1m": 9.0},
    "gpt-5-nano": {"input_per_1m": 1.0, "output_per_1m": 3.0},
}

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICE_TABLE.get(model)
    if not pricing:
        return 0.0
    return round(
        (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
        + (completion_tokens / 1_000_000) * pricing["output_per_1m"],
        6,
    )
```

---

## Formatter

Deterministic answer templates that bypass LLM variability.

### Main Dispatcher (`backend/formatter.py:242-265`)

```python
def format_answer(context: ContextBundle) -> Optional[str]:
    if context.intent is Intent.positions_list:
        return format_positions_list(context.context)
    if context.intent is Intent.positions:
        return format_positions(context.context)
    if context.intent is Intent.activity:
        return format_activity(context.context)
    if context.intent is Intent.performance:
        return format_performance(context.context)
    if context.intent is Intent.symbol_performance:
        return format_symbol_performance(context.context)
    if context.intent is Intent.portfolio_ranking:
        return format_portfolio_ranking(context.context)
    # ... other intents
    return None
```

### Format Helpers (`backend/formatter.py:23-50`)

```python
def _format_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def _format_percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"

def _format_percent_precise(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"

def _reasoning(points: list[str]) -> str:
    sentence = " ".join(p.strip().rstrip(".") + "." for p in points if p and p.strip())
    return f"Reasoning: {sentence}"
```

### Example Formatter: Symbol Performance (`backend/formatter.py:116-131`)

```python
def format_symbol_performance(context: SymbolPerformanceContext) -> str:
    pl = _format_money(context.unrealized_pl)
    pl_pct = _format_percent_precise(context.unrealized_pl_pct)
    answer = (
        f"{context.symbol} performance (as of {context.as_of}): "
        f"{context.quantity} shares, cost basis {_format_money(context.cost_basis)}/share, "
        f"current price {_format_money(context.current_price)}/share, "
        f"unrealized P/L {pl} ({pl_pct})."
    )
    reasoning = _reasoning(
        [
            "I combined your position with the latest quote for the symbol.",
            "Unrealized P/L is (current price − cost basis) × shares.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### Context Summary Builder (`backend/formatter.py:268-353`)

Used for trace logging:

```python
def build_context_summary(context: ContextBundle) -> dict[str, Any]:
    intent = context.intent
    data = context.context
    if intent is Intent.positions_list:
        return {
            "positions_count": len(data.items),
            "account": data.account,
            "as_of": data.as_of,
            "asset_class_filter": data.asset_class_filter,
        }
    # ... other intents
```

### Markdown Cleaning (`backend/responder.py:84-113`)

```python
def clean_answer_markdown(text: str) -> str:
    cleaned = text.strip().replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•]\s*", "", line)  # Remove bullets
        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    # Fix common spacing issues
    cleaned = re.sub(r":(\d)", r": \1", cleaned)
    cleaned = re.sub(r"\byou(hold|own|have)\b", r"you \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcostbasis\b", "cost basis", cleaned, flags=re.IGNORECASE)
    # ...
    return cleaned.strip()

def strip_markdown(text: str) -> str:
    cleaned = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    cleaned = re.sub(r"^#{1,6}\\s+", "", cleaned, flags=re.MULTILINE)
    return cleaned
```

---

## Tracing

Per-request trace logging for debugging and replay.

### TraceLogger Class (`backend/tracing.py:16-120`)

```python
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

### Logging Methods (`backend/tracing.py:53-120`)

```python
def start(self, utterance: str) -> str:
    self.data["utterance"] = utterance
    return self.trace_id

def log_intent(self, intent: str) -> None:
    self.data["intent"] = intent

def log_routing(self, mode: str, confidence: float, candidates: list[dict[str, float]]) -> None:
    self.data["routing_mode"] = mode
    self.data["routing_confidence"] = confidence
    self.data["routing_candidates"] = candidates

def log_tool(self, name: str, latency_ms: int, source_id: str) -> None:
    self.data["tool_calls"].append({"name": name, "source_id": source_id})
    self.data["tool_latency_ms"].append(latency_ms)

def log_grounding(self, citations: list[str], sources: list[str]) -> None:
    self.data["grounded_sources"] = citations
    denom = max(1, len(sources))
    self.data["grounding_rate"] = round(len(citations) / denom, 2)

def log_latency_breakdown(self, breakdown: dict[str, int]) -> None:
    self.data["latency_ms"] = breakdown

def finalize(self, model_version: str, artifact_version: str, prompt_version: str) -> None:
    self.data["model_version"] = model_version
    self.data["artifact_version"] = artifact_version
    self.data["prompt_version"] = prompt_version
    (TRACE_DIR / f"{self.trace_id}.json").write_text(json.dumps(self.data, indent=2))
```

### Trace Loading (`backend/tracing.py:123-127`)

```python
def load_trace(trace_id: str) -> Optional[dict[str, Any]]:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

### Trace API Endpoints (`backend/main.py:539-559`)

```python
@app.get("/debug/trace/{trace_id}")
async def get_trace(trace_id: str):
    data = load_trace(trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trace not found")
    return JSONResponse(content=data)

@app.post("/debug/trace/{trace_id}/explain")
async def explain_trace(trace_id: str):
    trace = load_trace(trace_id)
    explanation = generate_teaching_explanation(trace, ...)
    return JSONResponse(content={"trace_id": trace_id, "explanation": explanation})
```

---

## Eval System

Automated quality checks against golden utterances.

### Golden Set (`backend/eval.py:18-45`)

```python
LATENCY_BUDGET_MS = 5000

GOLDEN_SET = [
    {"utterance": "What was my most recent trade?", "intent": Intent.activity, "tools": ["tool:activity:v1"]},
    {"utterance": "positions?", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "what do i own", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "How many shares of AAPL do I own?", "intent": Intent.positions, "tools": ["tool:positions:v1"]},
    {"utterance": "AAPL performance and price", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "best performing position", "intent": Intent.portfolio_ranking, "tools": ["tool:positions_list:v1", "tool:quotes:v1"]},
    {"utterance": "apple quote", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "What is a Roth IRA?", "intent": Intent.facts, "tools": ["tool:facts:v1"]},
    # ... more utterances
]
```

### EvalRunner Class (`backend/eval.py:61-187`)

```python
class EvalRunner:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.registry = ToolRegistry()

    def run(self) -> dict[str, Any]:
        results = []
        total = len(GOLDEN_SET)
        pass_count = 0
        grounding_hits = 0
        tool_minimality_hits = 0

        # Pre-check: validate quote coverage
        missing_quotes = _validate_quote_coverage()
        if missing_quotes:
            return {
                "pass_rate": 0.0,
                "results": [{"utterance": "quote_coverage", "pass": False,
                             "reason": f"Missing quotes for positions: {', '.join(missing_quotes)}"}],
            }

        for item in GOLDEN_SET:
            start = time.perf_counter()
            route = route_intent(item["utterance"], use_llm=False)

            # Build context based on intent
            if route.intent is Intent.symbol_performance:
                positions_result = self.registry.call_tool("positions", **route.extracted)
                quotes_result = self.registry.call_tool("quotes", **route.extracted)
                context = build_symbol_performance_context(...)
            # ... other intents

            llm_response, _ = generate_response(item["utterance"], context, self.model_config)

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            intent_ok = route.intent == item["intent"]
            tool_ok = sorted(tool_ids) == sorted(item["tools"])
            grounding_ok = all(c in context.sources for c in llm_response.citations)

            passed = intent_ok and tool_ok and grounding_ok and elapsed_ms <= LATENCY_BUDGET_MS

        return {
            "pass_rate": round(pass_count / max(1, total), 2),
            "tool_minimality_score": round(tool_minimality_hits / max(1, total), 2),
            "grounding_rate": round(grounding_hits / max(1, total), 2),
            "results": results,
        }
```

### Eval Metrics

- **pass_rate**: Percentage of golden utterances that pass all checks
- **tool_minimality_score**: Percentage using exactly the expected tools
- **grounding_rate**: Percentage with valid citation grounding
- **latency_budget**: 5000ms per request

---

## Teaching Explanation

LLM-generated explanations of trace data for demos.

### Prompt Building (`backend/teaching.py:9-34`)

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

### Explanation Generation (`backend/teaching.py:37-60`)

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
    # Extract text from response...
    return text.strip()
```

---

## ML Training Pipeline

Generates synthetic data and trains the PyTorch intent classifier.

### Template Format (`ml/intent_templates.json`)

```json
{
  "intents": {
    "activity": [
      "What was my most recent trade?",
      "Show my recent activity",
      "Latest trade in my account"
    ],
    "positions": [
      "How many shares of {SYMBOL} do I own?",
      "Do I hold any {SYMBOL}?",
      "What's my {SYMBOL} position?"
    ]
  },
  "slots": {
    "SYMBOL": ["AAPL", "MSFT", "VOO", "TSLA", "NVDA", "AMZN"],
    "TIMEFRAME": ["YTD", "1Y", "3M", "QTD", "this year"],
    "ACCOUNT": ["Brokerage", "IRA", "Roth IRA"]
  },
  "typos": {
    "performance": ["perfomance", "performnce"],
    "positions": ["postions", "posiions"]
  }
}
```

### Dataset Generation (`scripts/generate_intent_dataset.py:98-150`)

```python
def main() -> int:
    random.seed(SEED)
    config = _load_templates()
    intents = config["intents"]
    slots = config["slots"]
    typos = config.get("typos", {})

    rows: list[dict[str, str]] = []
    for intent, templates in intents.items():
        for template in templates:
            rendered = _render_template(template, slots, max_combos=MAX_COMBOS)
            for text in rendered:
                for variant in _variants(text, typos):
                    add_row(variant, intent)

    # Augment with prefixes/suffixes until target size
    if len(rows) < TARGET_SIZE:
        base_rows = list(rows)
        while len(rows) < TARGET_SIZE:
            base = rng.choice(base_rows)
            augmented = _augment_with_affixes(base["text"], prefixes, suffixes, slots, rng)
            add_row(augmented, base["intent"])

    train, val, test = _split_dataset(rows)  # 80/10/10 split
    _write_jsonl(OUT_DIR / "intent_train.jsonl", train)
    _write_jsonl(OUT_DIR / "intent_val.jsonl", val)
    _write_jsonl(OUT_DIR / "intent_test.jsonl", test)
```

### Training Script (`scripts/train_intent_router.py:117-179`)

```python
def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_jsonl(DATA_DIR / "intent_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "intent_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "intent_test.jsonl")

    labels = sorted({row["intent"] for row in train_rows})
    vocab = build_vocab([row["text"] for row in train_rows], MAX_VOCAB, NGRAM_MAX)

    model = IntentClassifier(len(vocab), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_loader)

    # Save artifacts
    torch.save(model.state_dict(), OUT_DIR / "intent_router.pt")
    (OUT_DIR / "intent_router_vocab.json").write_text(json.dumps(vocab))
    (OUT_DIR / "intent_router_labels.json").write_text(json.dumps(labels))
```

### Training Hyperparameters

```python
SEED = 7
MAX_VOCAB = 6000
NGRAM_MAX = 2
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-2
```

---

## Schema Reference

### Context Models (`backend/schemas.py:44-188`)

| Model | Fields | Used For |
|-------|--------|----------|
| `ActivityContext` | `most_recent_trade`, `account`, `as_of` | Recent trade info |
| `PositionsContext` | `symbol`, `quantity`, `cost_basis`, `account`, `as_of` | Single position |
| `PositionsListContext` | `items[]`, `account`, `as_of`, `asset_class_filter` | All positions |
| `SymbolPerformanceContext` | `symbol`, `quantity`, `cost_basis`, `current_price`, `unrealized_pl`, `unrealized_pl_pct`, `as_of` | Symbol P/L |
| `PortfolioRankingContext` | `account`, `basis`, `direction`, `winner`, `rankings[]`, `top_three[]`, `missing_symbols[]`, `as_of` | Best/worst position |
| `TransfersContext` | `account`, `as_of`, `transfers[]` | Transfer history |
| `AccountValueContext` | `account`, `total_value`, `as_of` | Account balance |
| `CashBalanceContext` | `account`, `total_cash`, `settled_cash`, `as_of` | Cash info |
| `PerformanceContext` | `timeframe`, `account`, `return_pct`, `contributions`, `as_of` | Portfolio return |
| `QuoteContext` | `symbol`, `price`, `change_pct`, `as_of`, `position_held` | Price quote |
| `FactsContext` | `topic`, `snippet`, `source`, `as_of` | Educational content |

---

## Error Handling Patterns

### Clarification Flow

When parameters are missing, the router returns `Intent.clarify`:

```python
if route.intent is Intent.clarify:
    candidate = route.extracted.get("candidate_intent")
    clarification = _clarify(intent_hint, route.missing_params)
    llm = LLMResponse(
        answer_markdown=clarification,
        citations=[],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question=clarification,
    )
```

### Tool Call Failures

Tool exceptions are caught and converted to clarification responses:

```python
try:
    tool_result = registry.call_tool(route.intent.value, **params)
except Exception as exc:
    llm = LLMResponse(
        answer_markdown="I could not retrieve the data needed to answer.",
        needs_clarification=True,
        clarifying_question="Can you double-check the request or try a different symbol?",
    )
```

### LLM Response Errors

ResponseError is raised on parse/validation failure:

```python
class ResponseError(RuntimeError):
    pass

def _parse_llm_output(raw_text: str) -> LLMResponse:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResponseError("LLM output was not valid JSON") from exc

    try:
        return LLMResponse.model_validate(payload)
    except ValidationError as exc:
        raise ResponseError("LLM output did not match schema") from exc
```

### Quote Missing Fallback

Special handling when a requested symbol has no quote:

```python
def _quote_missing_response(symbol: str, quotes_result) -> tuple[LLMResponse, str]:
    available = _available_quote_symbols(quotes_result)
    message = f"I don't have a quote for {symbol} in this demo dataset (as of {quotes_result.as_of})."
    if available:
        message += f" Available symbols: {', '.join(available)}."
        clarifying = f"Try one of: {', '.join(available)}."
    else:
        clarifying = f"Can you provide a quotes source for {symbol}?"
    return LLMResponse(
        answer_markdown=message,
        needs_clarification=True,
        clarifying_question=clarifying,
    ), clarifying
```

---

## Common Issues

### From Code Comments and Patterns

1. **Symbol extraction edge cases** - The `SYMBOL_STOPWORDS` set in `router.py` (lines 15-129) prevents common words from being mistaken for symbols. Add new stopwords here if false positives occur.

2. **Quote coverage gaps** - The eval system validates quote coverage before running:
   ```python
   missing_quotes = _validate_quote_coverage()
   if missing_quotes:
       return {"pass_rate": 0.0, "reason": f"Missing quotes for positions: {missing_quotes}"}
   ```

3. **Typo handling** - Common typos are normalized in `router.py:180-187`:
   ```python
   TYPO_MAP = {
       "perfomance": "performance",
       "postions": "positions",
       "posiions": "positions",
       "qoute": "quote",
   }
   ```

4. **Ambiguous routing** - When top two candidates are within 0.1 score, the system considers it ambiguous and escalates to torch/LLM routing.

5. **Session state race conditions** - The `session_state` dict is in-memory and not thread-safe. For production, use a proper session store.

6. **Streaming disabled** - `STREAMING_ENABLED = False` at module level. SSE streaming code exists but is off by default.

7. **Cost estimation gaps** - Only gpt-5 series models have pricing in `PRICE_TABLE`. Other models return 0.0 cost.

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
| `INTENT_DATASET_SIZE` | `25000` | Target size for generated training data |
