# Intent Router Specification

> Deep technical specification for autonomous coding. Last updated: 2026-01-29

## Overview

The Intent Router is a three-layer natural language understanding (NLU) system that classifies user utterances into discrete intents and extracts parameters for downstream tool execution. It exists to bridge free-form user queries with structured tool calls in a personal finance assistant.

**Why it exists:**
- Deterministic routing avoids expensive LLM calls for common patterns
- Layered fallback provides graceful degradation when rules fail
- Parameter extraction grounds LLM responses with real data

**Architecture flow:**
```
User Utterance
     ↓
[Layer 1: Rule-based scorer]  ← Fast, keyword-based
     ↓ (confidence < 0.75 or ambiguous)
[Layer 2: Torch BoW classifier]  ← Learned, still fast
     ↓ (confidence < 0.6)
[Layer 3: LLM reroute]  ← Fallback, expensive
     ↓
IntentRoute with extracted params
```

---

## Key Models and Types

### Intent Enum

Defined in `backend/schemas.py:9-21`:

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

| Intent | Description | Requires Symbol |
|--------|-------------|-----------------|
| `activity` | Recent trade history | No |
| `positions` | Single symbol position details | Yes |
| `positions_list` | All holdings summary | No |
| `portfolio_ranking` | Best/worst performers | No |
| `symbol_performance` | P/L for specific symbol | Yes |
| `performance` | Overall portfolio return (YTD) | No |
| `quotes` | Current price for symbol | Yes |
| `facts` | Educational content | No (topic extracted) |
| `transfers` | Cash movement history | No |
| `account_value` | Total portfolio value | No |
| `cash_balance` | Available/settled cash | No |
| `clarify` | Insufficient info to route | N/A |

### IntentRoute Model

Defined in `backend/schemas.py:24-33`:

```python
class IntentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    missing_params: List[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    candidates: List[dict[str, Any]] = Field(default_factory=list)
    routing_mode: str = "rules"
    routing_meta: dict[str, Any] = Field(default_factory=dict)
```

**Field details:**
- `intent`: The classified intent enum value
- `confidence`: Score from 0.0-1.0 (threshold-dependent)
- `missing_params`: Parameters that couldn't be extracted (e.g., `["symbol"]`)
- `extracted`: Successfully extracted params (e.g., `{"symbol": "AAPL", "account": "Brokerage"}`)
- `candidates`: Scored alternatives from rule layer (for debugging/fallback)
- `routing_mode`: Which layer produced the route (`"rules"`, `"torch_classifier"`, `"llm_reroute"`)
- `routing_meta`: Debug info about torch router status

---

## Public API

### Main Entry Point

```python
# backend/router.py:500-565
def route_intent(utterance: str, use_llm: bool = True) -> IntentRoute:
```

**Parameters:**
- `utterance`: Raw user input string
- `use_llm`: Whether to allow LLM fallback (default `True`)

**Returns:** `IntentRoute` with classified intent and extracted parameters

**Behavior:**
1. Calls `rule_route(utterance)` for initial classification
2. If confidence >= threshold (0.75) and not ambiguous, returns rule result
3. Otherwise, tries `torch_attempt()` if artifacts present
4. If torch fails, calls `llm_reroute()` if API key available
5. All routes pass through `_apply_extraction()` for param extraction

### Rule-based Router

```python
# backend/router.py:462-497
def rule_route(utterance: str) -> IntentRoute:
```

Returns an `IntentRoute` using keyword scoring alone. Always returns a result (never None).

### Torch Router

```python
# backend/torch_router.py:122-177
def torch_attempt(
    utterance: str,
    candidates: list[dict[str, float]]
) -> tuple[Optional[IntentRoute], dict[str, object]]:
```

**Returns:** Tuple of (route or None, metadata dict)

The metadata dict always contains:
```python
{
    "torch_available": bool,      # Is PyTorch importable
    "artifacts_present": bool,    # Do model files exist
    "enabled": bool,              # torch_available AND artifacts_present
    "confidence_threshold": float, # From TORCH_ROUTER_CONF_THRESHOLD
    "force_enabled": bool,        # From TORCH_ROUTER_FORCE
    "ngram_max": int,             # From config
    "torch_attempted": bool,      # Did we try inference
    "torch_used": bool,           # Did we use torch result
    "torch_label": str,           # Predicted label (if attempted)
    "torch_confidence": float,    # Prediction confidence (if attempted)
}
```

### LLM Reroute

```python
# backend/router_llm.py:17-69
def llm_reroute(
    utterance: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    candidates: list[dict[str, float]],
) -> Optional[IntentRoute]:
```

Calls OpenAI API with a structured prompt. Returns `IntentRoute` with `routing_mode="llm_reroute"` and fixed confidence of 0.7, or `None` on parse failure.

---

## Internal Patterns

### Text Normalization

```python
# backend/router.py:280-286
def _normalize(text: str) -> str:
    cleaned = text.lower().strip()
    for typo, correct in TYPO_MAP.items():
        cleaned = cleaned.replace(typo, correct)
    cleaned = re.sub(r"[^\w\s\-\.\/]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
```

**Typo corrections** (`TYPO_MAP`):
```python
TYPO_MAP = {
    "perfomance": "performance",
    "postions": "positions",
    "posiions": "positions",
    "qoute": "quote",
    "balnce": "balance",
    "tranfer": "transfer",
}
```

### Symbol Extraction

```python
# backend/router.py:228-267
def _extract_symbol(text: str, allow_unknown: bool = False) -> str | None:
```

**Priority order:**
1. Explicit uppercase tokens matching `[A-Z]{1,5}` (e.g., `AAPL`)
2. `$`-prefixed tickers (e.g., `$AAPL`)
3. Lowercase known symbols from positions/quotes data
4. If `allow_unknown=True`, single unique candidate token

**Symbol aliases** (`SYMBOL_ALIASES`):
```python
SYMBOL_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "vanguard": "VOO",
}
```

**Stopwords filtering:** 129 common words excluded (pronouns, verbs, finance terms like "STOCK", "SHARE", "PRICE", etc.)

### Keyword Scoring

```python
# backend/router.py:289-378
def _score_candidates(text: str, utterance: str) -> list[dict[str, float]]:
```

Builds a score dictionary for each intent, bumping scores based on keyword matches:

| Pattern | Intent | Bump |
|---------|--------|------|
| "recent trade", "last trade" | activity | 0.9 |
| "how many shares", "do i own" | positions | 0.9 |
| SYNONYMS["positions"] | positions_list | 0.8 |
| RANK_KEYWORDS + portfolio context | portfolio_ranking | 0.92 |
| SYNONYMS["account_value"] | account_value | 0.85 |
| SYNONYMS["cash_balance"] | cash_balance | 0.85 |
| "ytd", "year to date", SYNONYMS["performance"] | performance | 0.8 |
| "performance" + symbol present | symbol_performance | 0.95 |
| SYNONYMS["transfers"] | transfers | 0.8 |
| SYNONYMS["quotes"], "change" | quotes | 0.8 |
| "what is", "explain", "define" | facts | 0.6 |

**RANK_KEYWORDS** for portfolio ranking detection:
```python
RANK_KEYWORDS = [
    "best", "worst", "top", "biggest", "highest", "lowest",
    "most", "least", "winner", "loser", "outperform",
    "underperform", "strongest", "weakest",
]
```

**SYNONYMS dictionary:**
```python
SYNONYMS = {
    "positions": ["positions", "holdings", "portfolio holdings", "what do i own", "owned shares", "equities"],
    "performance": ["performance", "performer", "performing", "return", "gain", "loss", "p/l", "profit", "up", "down"],
    "quotes": ["price", "quote", "trading at", "current price"],
    "transfers": ["transfer", "deposit", "withdrawal", "cash transfer", "bank transfer", "ach"],
    "account_value": ["account value", "total value", "portfolio value", "account balance"],
    "cash_balance": ["cash value", "cash amount", "cash balance", "settled cash", "total cash", "available cash"],
}
```

### Parameter Extraction

```python
# backend/router.py:381-459
def _apply_extraction(
    intent: Intent,
    text: str,
    utterance: str
) -> tuple[Intent, dict[str, Any], list[str]]:
```

**Intent-specific extraction rules:**

| Intent | Extracted Params | Fallback Behavior |
|--------|-----------------|-------------------|
| `positions` | `symbol` | Falls back to `positions_list` if no symbol |
| `quotes` | `symbol` | Adds `"symbol"` to `missing_params` |
| `symbol_performance` | `symbol` | Falls back to `performance` if no symbol |
| `portfolio_ranking` | `direction` ("best"/"worst"), `basis` ("unrealized_pl"/"unrealized_pl_pct") | Switches to `symbol_performance` if symbol present without portfolio terms |
| `cash_balance` | `cash_type` ("settled"/"total"/"both") | Defaults to "both" |
| `performance` | `timeframe` ("YTD") | Always defaults to "YTD" |
| `facts` | `topic` (full utterance) | N/A |
| `account_value`, `cash_balance` | `account` ("Brokerage") | Default "Brokerage" |

**Asset class detection:**
```python
ASSET_CLASS_MAP = {
    "stocks": ["stocks", "stock", "equities", "equity"],
    "etf": ["etf", "etfs"],
    "funds": ["funds", "mutual funds"],
}
```

### Ambiguity Detection

```python
# backend/router.py:506-507
ambiguous = False
if len(candidates) >= 2:
    ambiguous = abs(candidates[0]["score"] - candidates[1]["score"]) < 0.1
```

If top two candidates are within 0.1 points, route is considered ambiguous and triggers fallback layers.

---

## Torch Router Internals

### Model Architecture

```python
# backend/torch_router.py:27-33
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

**Architecture:** Single linear layer (bag-of-words to logits). No hidden layers.

### Feature Encoding

```python
# backend/torch_router.py:46-57
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

**Features:** Unigrams + bigrams (when `ngram_max >= 2`)

### Model Artifacts

Located in `backend/artifacts/model_bundle_v1/intent_router/`:

| File | Content |
|------|---------|
| `intent_router.pt` | PyTorch state dict |
| `intent_router_vocab.json` | Token-to-index mapping (max 6000 tokens) |
| `intent_router_labels.json` | Label list (11 intents, no "clarify") |
| `intent_router_config.json` | Training hyperparameters |

**Config contents:**
```json
{
  "max_vocab": 6000,
  "token_pattern": "[A-Za-z0-9']+",
  "ngram_max": 2,
  "epochs": 12,
  "batch_size": 32,
  "learning_rate": 0.01
}
```

---

## LLM Reroute Internals

### Prompt Template

```python
# backend/router_llm.py:26-34
prompt = (
    "You are an intent router. Choose the single best intent for the user query.\n"
    "Return ONLY JSON in this schema: {\"intent\": <string>, \"extracted\": {}}.\n"
    "Valid intents: activity, positions, positions_list, portfolio_ranking, "
    "symbol_performance, performance, quotes, facts, transfers, account_value, cash_balance.\n"
    "User query: " + utterance + "\n"
    "Candidates (intent:score): " + ", ".join([f"{c['intent']}:{c['score']}" for c in candidates])
)
```

### Response Parsing

```python
# backend/router_llm.py:12-14
class LLMRoute(BaseModel):
    intent: Intent
    extracted: dict[str, Any] = {}
```

Parses JSON from LLM response, validates against `LLMRoute` model, returns `None` on parse failure.

---

## Integration Points

### With Tool Layer

The `IntentRoute.intent` value maps to tool functions in `backend/tools.py`:

| Intent | Tool Function |
|--------|---------------|
| `activity` | `get_recent_activity()` |
| `positions` | `get_positions(symbol=)` |
| `positions_list` | `get_positions_list()` |
| `portfolio_ranking` | Composite: `get_positions_list()` + `get_quotes()` |
| `symbol_performance` | Composite: `get_positions(symbol=)` + `get_quotes(symbol=)` |
| `performance` | `get_performance()` |
| `quotes` | `get_quotes(symbol=)` |
| `facts` | `get_facts(topic=)` |
| `transfers` | `get_transfers()` |
| `account_value` | `get_account_value()` |
| `cash_balance` | `get_cash_balance()` |

### With Main Pipeline

```python
# In backend/main.py (chat endpoint)
route = route_intent(utterance)
if route.missing_params:
    # Return clarifying question
    ...
tool_result = tool_registry.call_tool(route.intent, **route.extracted)
context = build_context(route.intent, tool_result)
response = generate_response(context)
```

### With Tracing

`IntentRoute` is serialized into trace files for debugging:
- `routing_mode` indicates which layer produced the route
- `routing_meta` contains torch router diagnostic info
- `candidates` shows all scored alternatives

---

## State Management

### Global State

```python
# backend/router.py:178
KNOWN_SYMBOLS = _load_known_symbols()  # Loaded once at module import
```

`KNOWN_SYMBOLS` is populated from `backend/data/user_master.json` positions and quotes at startup.

### Cached State

```python
# backend/torch_router.py:69-92
@lru_cache
def _load_router(model_path: str, vocab_path: str, labels_path: str) -> Optional[TorchRouter]:
```

Torch model is loaded once and cached. Cache key is the tuple of file paths.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_CONF_THRESHOLD` | `0.75` | Rule confidence threshold before fallback |
| `TORCH_ROUTER_CONF_THRESHOLD` | `0.6` | Torch confidence threshold |
| `TORCH_ROUTER_FORCE` | `false` | Force torch regardless of confidence |
| `ROUTER_MODEL` | `gpt-5-mini` | Model for LLM reroute |
| `OPENAI_API_KEY` | (required) | For LLM reroute |
| `OPENAI_BASE_URL` | (none) | Custom API endpoint |
| `TORCH_ROUTER_MODEL_PATH` | `artifacts/.../intent_router.pt` | Override model location |
| `TORCH_ROUTER_VOCAB_PATH` | `artifacts/.../intent_router_vocab.json` | Override vocab location |
| `TORCH_ROUTER_LABELS_PATH` | `artifacts/.../intent_router_labels.json` | Override labels location |

---

## Error Handling

### Rule Layer

- **No candidates:** Returns `Intent.clarify` with confidence 0.2 and `missing_params=["intent"]`
- **Missing required param:** Returns `Intent.clarify` with `candidate_intent` in extracted dict

### Torch Layer

- **PyTorch not installed:** Returns `(None, meta)` with `torch_available=False`
- **Artifacts missing:** Returns `(None, meta)` with `artifacts_present=False`
- **Invalid label:** Returns `(None, meta)` (label not in Intent enum)
- **Low confidence:** Returns `(None, meta)` unless `TORCH_ROUTER_FORCE=true`

### LLM Layer

- **No API key:** Falls back to rule result
- **JSON parse error:** Returns `None`
- **Pydantic validation error:** Returns `None`

### Fallback Chain

```python
# backend/router.py:500-565
def route_intent(utterance: str, use_llm: bool = True) -> IntentRoute:
    rule_result = rule_route(utterance)  # Always succeeds

    # High confidence, non-ambiguous → return rules
    if rule_result.confidence >= threshold and not ambiguous:
        return rule_result

    # Try torch
    torch_route, torch_meta = torch_attempt(utterance, candidates)
    if torch_route:
        return torch_route

    # Try LLM
    if api_key:
        reroute = llm_reroute(...)
        if reroute:
            return reroute

    # Final fallback: return rule result
    return rule_result
```

---

## ML Training Pipeline

### Dataset Generation

```bash
uv run python scripts/generate_intent_dataset.py
```

**Source:** `ml/intent_templates.json`

**Process:**
1. Load templates with placeholders (e.g., `"How is {SYMBOL} performing?"`)
2. Expand slot values via Cartesian product (max 200 combos per template)
3. Generate variants: lowercase, typos, stripped punctuation
4. Augment with prefixes/suffixes until TARGET_SIZE (default 25,000)
5. Split 80/10/10 into train/val/test JSONL files

**Output files:**
- `ml/intent_train.jsonl`
- `ml/intent_val.jsonl`
- `ml/intent_test.jsonl`

**JSONL format:**
```json
{"text": "How is AAPL performing?", "intent": "symbol_performance"}
```

### Model Training

```bash
uv run python scripts/train_intent_router.py
```

**Hyperparameters:**
- Vocab size: 6000 max tokens
- Features: Unigrams + bigrams
- Epochs: 12
- Batch size: 32
- Learning rate: 0.01
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Output artifacts:** Written to `backend/artifacts/model_bundle_v1/intent_router/`

---

## Common Issues

### From Code Comments/TODOs

1. **No explicit TODOs found** in router files. Code is production-ready.

### Known Gotchas

1. **Symbol extraction ambiguity:** Single-letter symbols (e.g., `A`, `I`) are filtered as stopwords. Legitimate single-letter tickers won't be recognized.

2. **Typo map is limited:** Only 6 typos are handled. Uncommon misspellings will degrade routing accuracy.

3. **LLM reroute fixed confidence:** Always returns 0.7 confidence regardless of LLM certainty. No way to distinguish high vs. low LLM confidence.

4. **`clarify` intent not in training data:** Torch router cannot predict `clarify`. Only rules can produce this intent.

5. **Torch artifacts must exist at startup:** If model files are missing, torch layer silently skips. No runtime error, but routing quality degrades.

6. **Symbol aliases are hardcoded:** Only "apple", "microsoft", "vanguard" are mapped. No dynamic alias discovery.

7. **Portfolio ranking direction detection:** Relies on exact keywords like "worst", "lowest". Synonyms like "poorest" or "weakest" may not trigger "worst" direction.

8. **Performance timeframe always YTD:** The `_apply_extraction` function defaults `timeframe` to "YTD" even when templates include other options like "1Y", "3M".

### Debug Tips

1. **Check `routing_mode`** in response to see which layer classified the intent
2. **Check `routing_meta`** for torch router diagnostic info
3. **Check `candidates`** to see all scored alternatives
4. **Use `/debug/trace/{trace_id}` endpoint** to inspect full routing state
5. **Set `TORCH_ROUTER_FORCE=true`** to always use torch (for testing)

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `backend/router.py` | 566 | Rule-based routing, symbol extraction, scoring |
| `backend/router_llm.py` | 70 | LLM fallback reroute |
| `backend/torch_router.py` | 183 | PyTorch BoW classifier |
| `backend/schemas.py` | 238 | Intent enum, IntentRoute model |
| `scripts/train_intent_router.py` | 180 | Training script |
| `scripts/generate_intent_dataset.py` | 155 | Dataset generation |
| `ml/intent_templates.json` | 157 | Utterance templates |
