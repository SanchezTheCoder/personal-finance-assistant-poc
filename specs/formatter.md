# Formatter System Specification

> Deterministic answer generation layer that bypasses LLM variability for known intents.

## Overview

The Formatter system provides **template-based answer generation** for financial assistant responses. Instead of relying on LLM output (which can vary between calls), the formatter produces consistent, structured answers from typed context objects.

**Why it exists:**
- LLM responses are non-deterministic, which makes testing and debugging harder
- Deterministic templates ensure consistent formatting of financial data
- Reduces token usage by generating answers locally
- Provides reliable "reasoning" sections that explain how answers were derived

**Key insight:** The LLM is still called first (for edge cases and clarifications), but the formatter **overwrites** its output for known intents when a deterministic template exists.

## Architecture

```
ContextBundle ─────────┬───────────────────────────────────────────────────────────┐
                       ↓                                                           │
┌──────────────────────────────────────────┐    ┌─────────────────────────────────┐
│            format_answer()               │    │     build_context_summary()     │
│ Dispatches to intent-specific formatter  │    │  Creates trace-friendly dict    │
│ Returns formatted answer or None         │    │  for debugging/logging          │
└────────────────────┬─────────────────────┘    └─────────────────────────────────┘
                     ↓
      ┌──────────────────────────────┐
      │    clean_answer_markdown()   │  (in responder.py)
      │    Normalizes whitespace,    │
      │    fixes common LLM glitches │
      └──────────────────────────────┘
                     ↓
      ┌──────────────────────────────┐
      │      strip_markdown()        │  (in responder.py)
      │    Removes **bold**, etc.    │
      └──────────────────────────────┘
                     ↓
             Final answer string
```

## Source Files

| File | Purpose |
|------|---------|
| `backend/formatter.py` | Core formatting logic, intent-specific templates |
| `backend/responder.py:84-113` | Markdown cleaning/stripping utilities |
| `backend/schemas.py:44-208` | Context model definitions |
| `backend/main.py:451-468` | Integration point in chat pipeline |

---

## Key Types

### ContextBundle

The unified input type for all formatting operations:

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

### Context Models

Each intent has a corresponding Pydantic model. Common fields include `account`, `as_of` (timestamp), and intent-specific data.

**ActivityContext:**
```python
class ActivityTrade(BaseModel):
    timestamp: str
    symbol: str
    side: str
    quantity: int
    price: float

class ActivityContext(BaseModel):
    most_recent_trade: ActivityTrade
    account: str
    as_of: str
```

**PositionsContext:**
```python
class PositionsContext(BaseModel):
    symbol: str
    quantity: int
    cost_basis: float
    account: str
    as_of: str
```

**PositionsListContext:**
```python
class PositionsListItem(BaseModel):
    symbol: str
    quantity: int
    cost_basis: float
    asset_class: str

class PositionsListContext(BaseModel):
    items: List[PositionsListItem]
    account: str
    as_of: str
    asset_class_filter: Optional[str] = None
```

**SymbolPerformanceContext:**
```python
class SymbolPerformanceContext(BaseModel):
    symbol: str
    quantity: int
    cost_basis: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    as_of: str
```

**PortfolioRankingContext:**
```python
class PositionPerformanceItem(BaseModel):
    symbol: str
    quantity: int
    cost_basis: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float

class PortfolioRankingContext(BaseModel):
    account: str
    as_of: str
    basis: str              # "unrealized_pl" or "unrealized_pl_pct"
    direction: str          # "best" or "worst"
    winner: PositionPerformanceItem
    rankings: List[PositionPerformanceItem]
    top_three: List[PositionPerformanceItem]
    missing_symbols: List[str] = Field(default_factory=list)
```

**TransfersContext:**
```python
class TransferItem(BaseModel):
    timestamp: str
    type: str
    method: str
    amount: float
    status: str

class TransfersContext(BaseModel):
    account: str
    as_of: str
    transfers: List[TransferItem]
```

**AccountValueContext:**
```python
class AccountValueContext(BaseModel):
    account: str
    total_value: float
    as_of: str
```

**CashBalanceContext:**
```python
class CashBalanceContext(BaseModel):
    account: str
    total_cash: float
    settled_cash: float
    as_of: str
```

**PerformanceContext:**
```python
class PerformanceContext(BaseModel):
    timeframe: str
    account: str
    return_pct: float
    contributions: float
    as_of: str
```

**QuoteContext:**
```python
class QuoteContext(BaseModel):
    symbol: str
    price: float
    change_pct: float
    as_of: str
    position_held: Optional[bool] = None
```

**FactsContext:**
```python
class FactsContext(BaseModel):
    topic: str
    snippet: str
    source: str
    as_of: str
```

---

## Public API

### `format_answer(context: ContextBundle) -> Optional[str]`

Main entry point. Dispatches to intent-specific formatter based on `context.intent`.

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
    if context.intent is Intent.quotes:
        return format_quote(context.context)
    if context.intent is Intent.transfers:
        return format_transfers(context.context)
    if context.intent is Intent.account_value:
        return format_account_value(context.context)
    if context.intent is Intent.cash_balance:
        return format_cash_balance(context.context)
    if context.intent is Intent.facts:
        return format_facts(context.context)
    return None  # No formatter for this intent, use LLM output
```

**Returns:** Formatted string if intent is supported, `None` otherwise.

### `build_context_summary(context: ContextBundle) -> dict[str, Any]`

Creates a trace-friendly summary of the context for debugging/logging.

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
    # ... similar patterns for other intents
```

### `clean_answer_markdown(text: str) -> str`

*Located in `backend/responder.py:84-107`*

Normalizes LLM output text by fixing common formatting issues:

```python
def clean_answer_markdown(text: str) -> str:
    cleaned = text.strip().replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•]\s*", "", line)  # Remove bullet points
        line = re.sub(r"\s+", " ", line)       # Collapse whitespace
        lines.append(line)

    cleaned = "\n".join(lines)
    # Fix common LLM glitches
    cleaned = re.sub(r":(\d)", r": \1", cleaned)                    # ":42" -> ": 42"
    cleaned = re.sub(r"\byou(hold|own|have)\b", r"you \1", cleaned) # "youhold" -> "you hold"
    cleaned = re.sub(r"\bcostbasis\b", "cost basis", cleaned)       # "costbasis" -> "cost basis"
    cleaned = re.sub(r"([A-Za-z])\$", r"\1 $", cleaned)             # "ABC$123" -> "ABC $123"
    cleaned = re.sub(r"\bYTDcontributions\b", "YTD contributions", cleaned)
    cleaned = re.sub(r"([\.!\?])([A-Za-z])", r"\1 \2", cleaned)     # Missing space after punctuation
    cleaned = re.sub(r"([,])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)          # camelCase -> camel Case
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)          # "42shares" -> "42 shares"
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)          # "AAPL42" -> "AAPL 42"
    cleaned = re.sub(r"\b[Nn]etcontributions\b", "Net contributions", cleaned)
    return cleaned.strip()
```

### `strip_markdown(text: str) -> str`

*Located in `backend/responder.py:110-113`*

Removes markdown formatting (bold, italic, headers) for clean display:

```python
def strip_markdown(text: str) -> str:
    cleaned = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)  # Remove **bold**, *italic*, etc.
    cleaned = re.sub(r"^#{1,6}\\s+", "", cleaned, flags=re.MULTILINE)  # Remove headers
    return cleaned
```

---

## Intent-Specific Formatters

Each formatter follows a consistent pattern:
1. Build an **answer** string with formatted data
2. Build a **reasoning** section explaining the answer source
3. Combine using `_join_sentence()`

### Helper Functions

```python
def _format_money(value: float) -> str:
    """Format as currency with sign. -1234.56 -> '-$1,234.56'"""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _format_percent(value: float) -> str:
    """Format percentage with sign. 5.5 -> '+5.5%'"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _format_percent_precise(value: float) -> str:
    """Format decimal as percentage. 0.055 -> '+5.50%'"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _format_date(ts: str) -> str:
    """Extract date from ISO timestamp. '2024-03-15T10:30:00Z' -> '2024-03-15'"""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ts.split("T")[0] if "T" in ts else ts


def _join_sentence(parts: list[str]) -> str:
    """Join non-empty parts with single spaces."""
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


def _reasoning(points: list[str]) -> str:
    """Build a 'Reasoning:' section from bullet points."""
    sentence = " ".join(p.strip().rstrip(".") + "." for p in points if p and p.strip())
    return f"Reasoning: {sentence}"
```

### `format_positions_list(context: PositionsListContext) -> str`

```python
def format_positions_list(context: PositionsListContext) -> str:
    scope = ""
    if context.asset_class_filter:
        scope = f"{context.asset_class_filter}, "
    header = f"Positions in {context.account} ({scope}as of {context.as_of}):".replace("(, ", "(")
    if not context.items:
        return f"{header} none."
    items = [
        f"{item.symbol} {item.quantity} shares @ {_format_money(item.cost_basis)}/share"
        for item in context.items
    ]
    reasoning = _reasoning(
        [
            f"I used your {context.account} positions as of {context.as_of}.",
            "Each line includes symbol, quantity, and cost basis.",
        ]
    )
    return _join_sentence([header, "; ".join(items) + ".", reasoning])
```

**Example output:**
```
Positions in Brokerage (stocks, as of 2024-03-15): AAPL 100 shares @ $150.25/share; MSFT 50 shares @ $380.00/share. Reasoning: I used your Brokerage positions as of 2024-03-15. Each line includes symbol, quantity, and cost basis.
```

### `format_positions(context: PositionsContext) -> str`

```python
def format_positions(context: PositionsContext) -> str:
    answer = (
        f"{context.symbol} position in {context.account} (as of {context.as_of}): "
        f"{context.quantity} shares @ {_format_money(context.cost_basis)}/share."
    )
    reasoning = _reasoning(
        [
            "I matched the symbol in your positions list.",
            f"Quantity and cost basis are taken from the positions tool as of {context.as_of}.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### `format_activity(context: ActivityContext) -> str`

```python
def format_activity(context: ActivityContext) -> str:
    trade = context.most_recent_trade
    date = _format_date(trade.timestamp)
    price = _format_money(trade.price)
    answer = (
        f"Most recent trade in {context.account} (as of {context.as_of}): "
        f"{trade.side.upper()} {trade.quantity} {trade.symbol} @ {price} on {date}."
    )
    reasoning = _reasoning(
        [
            "I selected the latest trade by timestamp from your activity feed.",
            f"The trade details come directly from the activity tool as of {context.as_of}.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### `format_symbol_performance(context: SymbolPerformanceContext) -> str`

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

### `format_portfolio_ranking(context: PortfolioRankingContext) -> str`

```python
def format_portfolio_ranking(context: PortfolioRankingContext) -> str:
    winner = context.winner
    pl = _format_money(winner.unrealized_pl)
    pl_pct = _format_percent_precise(winner.unrealized_pl_pct)
    direction_label = "Best" if context.direction != "worst" else "Worst"
    basis_label = "unrealized P/L" if context.basis == "unrealized_pl" else "unrealized % return"
    answer = (
        f"{direction_label} performing position by {basis_label} (as of {context.as_of}): "
        f"{winner.symbol} — {winner.quantity} shares, cost basis {_format_money(winner.cost_basis)}/share, "
        f"current price {_format_money(winner.current_price)}/share, "
        f"unrealized P/L {pl} ({pl_pct})."
    )
    top_three = ", ".join(
        [
            f"{item.symbol} {_format_money(item.unrealized_pl)} ({_format_percent_precise(item.unrealized_pl_pct)})"
            for item in context.top_three
        ]
    )
    top_three_line = f"Top 3 by {basis_label}: {top_three}."
    reasoning = _reasoning(
        [
            "I calculated unrealized P/L for each holding using positions and quotes.",
            f"Then I ranked positions by {basis_label}.",
        ]
    )
    return _join_sentence([answer, top_three_line, reasoning])
```

### `format_quote(context: QuoteContext) -> str`

```python
def format_quote(context: QuoteContext) -> str:
    answer = (
        f"{context.symbol} price as of {context.as_of}: {_format_money(context.price)} "
        f"(change {_format_percent(context.change_pct)})."
    )
    note = ""
    if context.position_held is False:
        note = "Note: You do not currently hold this symbol in your positions."
    reasoning_points = [
        "The price and change come directly from the quotes tool.",
        f"Quote timestamp is {context.as_of}.",
    ]
    if context.position_held is False:
        reasoning_points.insert(0, "I checked your positions list and did not find this symbol.")
    reasoning = _reasoning(reasoning_points)
    return _join_sentence([answer, note, reasoning])
```

### `format_performance(context: PerformanceContext) -> str`

```python
def format_performance(context: PerformanceContext) -> str:
    timeframe = context.timeframe or "YTD"
    line1 = f"{context.account} performance {timeframe} (as of {context.as_of}): {_format_percent(context.return_pct)}."
    line2 = f"Net contributions {timeframe}: {_format_money(context.contributions)}."
    reasoning = _reasoning(
        [
            "The return percentage comes from the performance tool for the selected timeframe.",
            "Net contributions are derived from transfers in the same period.",
        ]
    )
    return _join_sentence([line1, line2, reasoning])
```

### `format_transfers(context: TransfersContext) -> str`

```python
def format_transfers(context: TransfersContext) -> str:
    if not context.transfers:
        answer = f"No recent transfers in {context.account} (as of {context.as_of})."
        reasoning = _reasoning(
            [
                "I checked your transfers feed and found no recent records.",
            ]
        )
        return _join_sentence([answer, reasoning])
    items = []
    for transfer in context.transfers:
        date = _format_date(transfer.timestamp)
        items.append(
            f"{date} {transfer.type} {_format_money(transfer.amount)} ({transfer.method}, {transfer.status})"
        )
    summary = "; ".join(items)
    answer = f"Recent transfers in {context.account} (as of {context.as_of}): {summary}."
    reasoning = _reasoning(
        [
            "Each transfer is summarized with date, amount, method, and status.",
            f"Source data is the transfers tool as of {context.as_of}.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### `format_account_value(context: AccountValueContext) -> str`

```python
def format_account_value(context: AccountValueContext) -> str:
    answer = f"{context.account} total value as of {context.as_of}: {_format_money(context.total_value)}."
    reasoning = _reasoning(
        [
            "Total value is pulled from the account summary tool.",
            f"The balance reflects the snapshot as of {context.as_of}.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### `format_cash_balance(context: CashBalanceContext) -> str`

```python
def format_cash_balance(context: CashBalanceContext) -> str:
    answer = (
        f"{context.account} cash as of {context.as_of}: "
        f"settled {_format_money(context.settled_cash)}, total {_format_money(context.total_cash)}."
    )
    reasoning = _reasoning(
        [
            "Cash balances come from the account summary tool.",
            f"Settled vs total is based on the {context.as_of} snapshot.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

### `format_facts(context: FactsContext) -> str`

```python
def format_facts(context: FactsContext) -> str:
    answer = f"{context.topic}: {context.snippet} (Source: {context.source})."
    reasoning = _reasoning(
        [
            "I used the local facts source for this definition.",
            "No external knowledge was added.",
        ]
    )
    return _join_sentence([answer, reasoning])
```

---

## Integration Points

### Main Pipeline (`backend/main.py:451-468`)

The formatter is invoked **after** the LLM generates a response:

```python
llm_response, meta = generate_response(request.utterance, context, model_config)

formatter_used = "none"
if not llm_response.needs_clarification:
    formatted = format_answer(context)
    if formatted:
        llm_response.answer_markdown = formatted  # Overwrite LLM output
        formatter_used = context.intent.value
        llm_response.needs_clarification = False
        llm_response.clarifying_question = None
else:
    # Even on clarification, try formatter as fallback
    formatted = format_answer(context)
    if formatted:
        llm_response.answer_markdown = formatted
        formatter_used = context.intent.value
        llm_response.needs_clarification = False
        llm_response.clarifying_question = None

# Always clean up the output
llm_response.answer_markdown = strip_markdown(clean_answer_markdown(llm_response.answer_markdown))
trace.log_formatter_used(formatter_used)
```

### Context Builder

The formatter depends on `build_context()` from `backend/context_builder.py` to create typed context objects from raw tool results:

```
Tool Result (dict) → build_context() → ContextBundle → format_answer() → String
```

### Tracing

The formatter reports which template was used via `trace.log_formatter_used()`:
- `"none"` - LLM output used directly (no formatter available or clarification needed)
- Intent name (e.g., `"positions_list"`) - Deterministic formatter was used

---

## State Management

The formatter is **stateless**. All state is passed in via the `ContextBundle`:
- No session awareness
- No caching
- Pure function from context to string

Session state (for follow-up questions) is managed at the API layer in `main.py`.

---

## Error Handling

### Graceful Degradation

If `format_answer()` returns `None`, the pipeline falls back to LLM output:

```python
formatted = format_answer(context)
if formatted:
    llm_response.answer_markdown = formatted
# Otherwise, keep LLM's answer_markdown
```

### Date Parsing Fallback

`_format_date()` handles malformed timestamps gracefully:

```python
def _format_date(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ts.split("T")[0] if "T" in ts else ts  # Fallback: naive split
```

### Empty Data Handling

Several formatters handle empty data explicitly:

```python
# format_positions_list
if not context.items:
    return f"{header} none."

# format_transfers
if not context.transfers:
    answer = f"No recent transfers in {context.account} (as of {context.as_of})."
```

---

## Common Patterns

### Consistent Answer Structure

All formatter outputs follow this pattern:
```
[Main answer with data] [Optional note] Reasoning: [Bullet points explaining sources]
```

### Type Ignores

The dispatcher uses `# type: ignore[arg-type]` because the union type in `ContextBundle.context` doesn't narrow automatically:

```python
if context.intent is Intent.positions_list:
    return format_positions_list(context.context)  # type: ignore[arg-type]
```

This is safe because `intent` and `context` are always matched at construction time in `build_context()`.

### Reasoning Section

Every formatter includes a `Reasoning:` section for transparency:
- Explains which tools provided the data
- Describes any calculations performed
- Builds user trust in the answer

---

## Known Issues / TODOs

1. **Em dash in output**: `format_portfolio_ranking` uses em dash (`—`) which conflicts with CLAUDE.md style guide. Should use comma or parentheses instead.

2. **Markdown in formatter output**: Some formatters produce text that then gets cleaned by `strip_markdown()`. The formatters could avoid markdown entirely.

3. **Duplicate code**: The `needs_clarification` check in `main.py:452-465` has identical branches. Could be simplified.

4. **Type narrowing**: The `# type: ignore` comments indicate the type system doesn't understand the intent/context coupling. A discriminated union pattern could fix this.

5. **Hardcoded strings**: Labels like "Best", "Worst", "unrealized P/L" are hardcoded. Could be configuration-driven for i18n.

---

## Testing Considerations

When testing formatters:

1. **Test each intent formatter individually** with mock context objects
2. **Verify money formatting** handles negatives, zeros, and large values
3. **Verify percent formatting** handles positive, negative, and zero values
4. **Test edge cases** like empty lists, missing optional fields
5. **Test the full pipeline** by checking `trace.formatter_used` in response traces

Example test structure:
```python
def test_format_positions():
    context = PositionsContext(
        symbol="AAPL",
        quantity=100,
        cost_basis=150.50,
        account="Brokerage",
        as_of="2024-03-15",
    )
    result = format_positions(context)
    assert "AAPL position in Brokerage" in result
    assert "$150.50" in result
    assert "Reasoning:" in result
```
