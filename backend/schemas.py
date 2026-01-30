from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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


class IntentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    missing_params: List[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    candidates: List[dict[str, Any]] = Field(default_factory=list)
    routing_mode: str = "rules"
    routing_meta: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    data: dict[str, Any]
    as_of: str


class ActivityTrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    symbol: str
    side: str
    quantity: int
    price: float


class ActivityContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    most_recent_trade: ActivityTrade
    account: str
    as_of: str


class PositionsContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: int
    cost_basis: float
    account: str
    as_of: str


class PositionsListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: int
    cost_basis: float
    asset_class: str


class PositionsListContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[PositionsListItem]
    account: str
    as_of: str
    asset_class_filter: Optional[str] = None


class SymbolPerformanceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: int
    cost_basis: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    as_of: str


class PositionPerformanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: int
    cost_basis: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float


class PortfolioRankingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    as_of: str
    basis: str
    direction: str
    winner: PositionPerformanceItem
    rankings: List[PositionPerformanceItem]
    top_three: List[PositionPerformanceItem]
    missing_symbols: List[str] = Field(default_factory=list)


class TransferItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    type: str
    method: str
    amount: float
    status: str


class TransfersContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    as_of: str
    transfers: List[TransferItem]


class AccountValueContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    total_value: float
    as_of: str


class CashBalanceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    total_cash: float
    settled_cash: float
    as_of: str


class PerformanceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeframe: str
    account: str
    return_pct: float
    contributions: float
    as_of: str


class QuoteContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    price: float
    change_pct: float
    as_of: str
    position_held: Optional[bool] = None


class FactsContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    snippet: str
    source: str
    as_of: str


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


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str
    citations: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    clarifying_question: Optional[str] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance: str
    account: Optional[str] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_markdown: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    clarifying_question: Optional[str]
    trace_id: str
