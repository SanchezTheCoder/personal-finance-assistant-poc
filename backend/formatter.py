from __future__ import annotations

from datetime import datetime
from typing import Optional, Any

from .schemas import (
    Intent,
    ActivityContext,
    AccountValueContext,
    CashBalanceContext,
    ContextBundle,
    FactsContext,
    PerformanceContext,
    PortfolioRankingContext,
    PositionsContext,
    PositionsListContext,
    QuoteContext,
    SymbolPerformanceContext,
    TransfersContext,
)


def _format_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _format_percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"

def _format_percent_precise(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _format_date(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ts.split("T")[0] if "T" in ts else ts


def _join_sentence(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()

def _reasoning(points: list[str]) -> str:
    sentence = " ".join(p.strip().rstrip(".") + "." for p in points if p and p.strip())
    return f"Reasoning: {sentence}"


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


def format_account_value(context: AccountValueContext) -> str:
    answer = f"{context.account} total value as of {context.as_of}: {_format_money(context.total_value)}."
    reasoning = _reasoning(
        [
            "Total value is pulled from the account summary tool.",
            f"The balance reflects the snapshot as of {context.as_of}.",
        ]
    )
    return _join_sentence([answer, reasoning])


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


def format_facts(context: FactsContext) -> str:
    answer = f"{context.topic}: {context.snippet} (Source: {context.source})."
    reasoning = _reasoning(
        [
            "I used the local facts source for this definition.",
            "No external knowledge was added.",
        ]
    )
    return _join_sentence([answer, reasoning])


def format_answer(context: ContextBundle) -> Optional[str]:
    if context.intent is Intent.positions_list:
        return format_positions_list(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.positions:
        return format_positions(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.activity:
        return format_activity(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.performance:
        return format_performance(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.symbol_performance:
        return format_symbol_performance(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.portfolio_ranking:
        return format_portfolio_ranking(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.quotes:
        return format_quote(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.transfers:
        return format_transfers(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.account_value:
        return format_account_value(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.cash_balance:
        return format_cash_balance(context.context)  # type: ignore[arg-type]
    if context.intent is Intent.facts:
        return format_facts(context.context)  # type: ignore[arg-type]
    return None


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
    if intent is Intent.positions:
        return {
            "symbol": data.symbol,
            "quantity": data.quantity,
            "cost_basis": data.cost_basis,
            "account": data.account,
            "as_of": data.as_of,
        }
    if intent is Intent.performance:
        return {
            "return_pct": data.return_pct,
            "contributions": data.contributions,
            "timeframe": data.timeframe,
            "account": data.account,
            "as_of": data.as_of,
        }
    if intent is Intent.symbol_performance:
        return {
            "symbol": data.symbol,
            "unrealized_pl": data.unrealized_pl,
            "unrealized_pl_pct": data.unrealized_pl_pct,
            "current_price": data.current_price,
            "as_of": data.as_of,
        }
    if intent is Intent.portfolio_ranking:
        return {
            "direction": data.direction,
            "basis": data.basis,
            "winner": data.winner.model_dump(),
            "rankings_count": len(data.rankings),
            "missing_symbols": data.missing_symbols,
            "as_of": data.as_of,
        }
    if intent is Intent.quotes:
        return {
            "symbol": data.symbol,
            "price": data.price,
            "change_pct": data.change_pct,
            "as_of": data.as_of,
        }
    if intent is Intent.activity:
        trade = data.most_recent_trade
        return {
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": trade.price,
            "timestamp": trade.timestamp,
        }
    if intent is Intent.transfers:
        most_recent = data.transfers[0] if data.transfers else None
        return {
            "transfer_count": len(data.transfers),
            "most_recent": most_recent.model_dump() if most_recent else None,
            "as_of": data.as_of,
        }
    if intent is Intent.account_value:
        return {
            "account": data.account,
            "total_value": data.total_value,
            "as_of": data.as_of,
        }
    if intent is Intent.cash_balance:
        return {
            "account": data.account,
            "settled_cash": data.settled_cash,
            "total_cash": data.total_cash,
            "as_of": data.as_of,
        }
    if intent is Intent.facts:
        return {
            "topic": data.topic,
            "source": data.source,
            "as_of": data.as_of,
        }
    return {}
