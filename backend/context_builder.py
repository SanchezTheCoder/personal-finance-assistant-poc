from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import (
    ActivityContext,
    ActivityTrade,
    AccountValueContext,
    CashBalanceContext,
    ContextBundle,
    FactsContext,
    Intent,
    PerformanceContext,
    PortfolioRankingContext,
    PositionPerformanceItem,
    PositionsContext,
    PositionsListContext,
    PositionsListItem,
    QuoteContext,
    SymbolPerformanceContext,
    TransfersContext,
    TransferItem,
    ToolResult,
)


def _parse_ts(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _parse_date(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _compute_ytd_contributions(transfers: list[dict[str, Any]], as_of: str) -> float:
    as_of_dt = _parse_date(as_of)
    start_of_year = datetime(as_of_dt.year, 1, 1)
    total = 0.0
    for t in transfers:
        ts = _parse_date(t["timestamp"])
        if ts < start_of_year or ts > as_of_dt:
            continue
        amount = float(t["amount"])
        if t["type"] == "deposit":
            total += amount
        elif t["type"] == "withdrawal":
            total -= amount
    return round(total, 2)


def build_context(intent: Intent, tool_result: ToolResult, params: dict[str, Any]) -> ContextBundle:
    data = tool_result.data
    if intent is Intent.activity:
        trades = data["trades"]
        if not trades:
            raise ValueError("No trades available")
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
    elif intent is Intent.positions_list:
        items = [
            PositionsListItem(
                symbol=p["symbol"],
                quantity=p["quantity"],
                cost_basis=p["cost_basis"],
                asset_class=p.get("asset_class", "unknown"),
            )
            for p in data["positions"]
        ]
        context = PositionsListContext(
            items=items,
            account=data.get("account", "Brokerage"),
            as_of=data["as_of"],
            asset_class_filter=data.get("asset_class_filter"),
        )
    elif intent is Intent.performance:
        transfers = params.get("transfers", [])
        contributions = _compute_ytd_contributions(transfers, data["as_of"]) if transfers else data["contributions"]
        context = PerformanceContext(
            timeframe=data["timeframe"],
            account=data.get("account", "Brokerage"),
            return_pct=data["return_pct"],
            contributions=contributions,
            as_of=data["as_of"],
        )
    elif intent is Intent.quotes:
        symbol = params["symbol"]
        quote = next((q for q in data["quotes"] if q["symbol"] == symbol), None)
        if not quote:
            raise ValueError(f"No quote for symbol {symbol}")
        context = QuoteContext(
            symbol=quote["symbol"],
            price=quote["price"],
            change_pct=quote["change_pct"],
            as_of=data["as_of"],
            position_held=params.get("position_held"),
        )
    elif intent is Intent.symbol_performance:
        raise ValueError("Symbol performance requires combined tool data")
    elif intent is Intent.facts:
        context = FactsContext(
            topic=data["topic"],
            snippet=data["snippet"],
            source=data["source"],
            as_of=data["as_of"],
        )
    elif intent is Intent.transfers:
        context = TransfersContext(
            account=data.get("account", "Brokerage"),
            as_of=data["as_of"],
            transfers=[TransferItem(**t) for t in data["transfers"]],
        )
    elif intent is Intent.account_value:
        account = params.get("account")
        account_data = next(
            (a for a in data["accounts"] if a["account"].lower() == (account or "").lower()),
            None,
        ) or data["accounts"][0]
        context = AccountValueContext(
            account=account_data["account"],
            total_value=account_data["total_value"],
            as_of=data["as_of"],
        )
    elif intent is Intent.cash_balance:
        account = params.get("account")
        account_data = next(
            (a for a in data["accounts"] if a["account"].lower() == (account or "").lower()),
            None,
        ) or data["accounts"][0]
        context = CashBalanceContext(
            account=account_data["account"],
            total_cash=account_data["total_cash"],
            settled_cash=account_data["settled_cash"],
            as_of=data["as_of"],
        )
    else:
        raise ValueError(f"Unsupported intent for context: {intent}")

    return ContextBundle(intent=intent, context=context, sources=[tool_result.source_id])


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
        items.append(
            PositionPerformanceItem(
                symbol=symbol,
                quantity=quantity,
                cost_basis=cost_basis,
                current_price=current_price,
                unrealized_pl=round(unrealized_pl, 2),
                unrealized_pl_pct=round(unrealized_pl_pct, 4),
            )
        )

    if not items:
        raise ValueError("No positions with available quotes for ranking")

    reverse = direction != "worst"
    key_fn = (lambda x: x.unrealized_pl_pct) if basis == "unrealized_pl_pct" else (lambda x: x.unrealized_pl)
    ranked = sorted(items, key=key_fn, reverse=reverse)
    top_three = ranked[:3]

    context = PortfolioRankingContext(
        account=positions_result.data.get("account", "Brokerage"),
        as_of=quotes_result.as_of,
        basis=basis,
        direction=direction,
        winner=ranked[0],
        rankings=ranked,
        top_three=top_three,
        missing_symbols=missing,
    )

    return ContextBundle(
        intent=Intent.portfolio_ranking,
        context=context,
        sources=[positions_result.source_id, quotes_result.source_id],
    )
