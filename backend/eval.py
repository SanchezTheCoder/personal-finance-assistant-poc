from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .context_builder import build_context, build_symbol_performance_context, build_portfolio_ranking_context
from .responder import ModelConfig, generate_response
from .router import route_intent
from .schemas import Intent
from .tools import ToolRegistry

LATENCY_BUDGET_MS = 5000

DATA_DIR = Path(__file__).parent / "data"

GOLDEN_SET = [
    {"utterance": "What was my most recent trade?", "intent": Intent.activity, "tools": ["tool:activity:v1"]},
    {"utterance": "positions?", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "what do i own", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "my holdings", "intent": Intent.positions_list, "tools": ["tool:positions_list:v1"]},
    {"utterance": "How many shares of AAPL do I own?", "intent": Intent.positions, "tools": ["tool:positions:v1"]},
    {"utterance": "account balance", "intent": Intent.account_value, "tools": ["tool:account_summary:v1"]},
    {"utterance": "account value", "intent": Intent.account_value, "tools": ["tool:account_summary:v1"]},
    {"utterance": "cash balance", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "cash value", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "settled cash", "intent": Intent.cash_balance, "tools": ["tool:account_summary:v1"]},
    {"utterance": "perfomance ytd", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "portfolio return", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "how did I do this year", "intent": Intent.performance, "tools": ["tool:performance:v1", "tool:transfers:v1"]},
    {"utterance": "AAPL performance and price", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "AAPL performance?", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "voo performance", "intent": Intent.symbol_performance, "tools": ["tool:positions:v1", "tool:quotes:v1"]},
    {"utterance": "best performing position", "intent": Intent.portfolio_ranking, "tools": ["tool:positions_list:v1", "tool:quotes:v1"]},
    {"utterance": "biggest unrealized loss", "intent": Intent.portfolio_ranking, "tools": ["tool:positions_list:v1", "tool:quotes:v1"]},
    {"utterance": "apple quote", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "AAPL price", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "What's MSFT price and today's change?", "intent": Intent.quotes, "tools": ["tool:quotes:v1"]},
    {"utterance": "What's my most recent transfer?", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "Did my last deposit go through?", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "Show my recent withdrawals", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "ACH pending", "intent": Intent.transfers, "tools": ["tool:transfers:v1"]},
    {"utterance": "What is a Roth IRA?", "intent": Intent.facts, "tools": ["tool:facts:v1"]},
]


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text())


def _validate_quote_coverage() -> list[str]:
    positions = _load_json("positions.json")["positions"]
    quotes = _load_json("quotes.json")["quotes"]
    position_symbols = {p["symbol"] for p in positions}
    quote_symbols = {q["symbol"] for q in quotes}
    missing = sorted(position_symbols - quote_symbols)
    return missing


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

        missing_quotes = _validate_quote_coverage()
        if missing_quotes:
            return {
                "pass_rate": 0.0,
                "tool_minimality_score": 0.0,
                "grounding_rate": 0.0,
                "results": [
                    {
                        "utterance": "quote_coverage",
                        "pass": False,
                        "reason": f"Missing quotes for positions: {', '.join(missing_quotes)}",
                    }
                ],
            }

        for item in GOLDEN_SET:
            start = time.perf_counter()
            route = route_intent(item["utterance"], use_llm=False)

            if route.intent is Intent.clarify:
                results.append(
                    {
                        "utterance": item["utterance"],
                        "pass": False,
                        "reason": f"clarify: missing {route.missing_params}",
                    }
                )
                continue

            tool_results = []
            if route.intent is Intent.symbol_performance:
                positions_result = self.registry.call_tool("positions", **route.extracted)
                quotes_result = self.registry.call_tool("quotes", **route.extracted)
                tool_results = [positions_result, quotes_result]
                try:
                    context = build_symbol_performance_context(
                        positions_result, quotes_result, route.extracted["symbol"]
                    )
                except Exception:
                    context = build_context(Intent.positions, positions_result, route.extracted)
            elif route.intent is Intent.portfolio_ranking:
                positions_result = self.registry.call_tool("positions_list", **route.extracted)
                first_symbol = (
                    positions_result.data["positions"][0]["symbol"]
                    if positions_result.data.get("positions")
                    else "AAPL"
                )
                quotes_result = self.registry.call_tool("quotes", symbol=first_symbol)
                tool_results = [positions_result, quotes_result]
                context = build_portfolio_ranking_context(
                    positions_result,
                    quotes_result,
                    direction=route.extracted.get("direction", "best"),
                    basis=route.extracted.get("basis", "unrealized_pl"),
                )
            elif route.intent is Intent.performance:
                performance_result = self.registry.call_tool("performance", **route.extracted)
                transfers_result = self.registry.call_tool("transfers")
                tool_results = [performance_result, transfers_result]
                context = build_context(
                    Intent.performance,
                    performance_result,
                    {**route.extracted, "transfers": transfers_result.data.get("transfers", [])},
                )
                context.sources = [performance_result.source_id, transfers_result.source_id]
            else:
                tool_result = self.registry.call_tool(route.intent.value, **route.extracted)
                tool_results = [tool_result]
                context = build_context(route.intent, tool_result, route.extracted)

            try:
                llm_response, _ = generate_response(item["utterance"], context, self.model_config)
            except Exception as exc:
                results.append(
                    {
                        "utterance": item["utterance"],
                        "pass": False,
                        "reason": f"llm_error: {exc}",
                    }
                )
                continue

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            intent_ok = route.intent == item["intent"]
            tool_ids = [t.source_id for t in tool_results]
            tool_ok = sorted(tool_ids) == sorted(item["tools"])
            tool_minimality = len(tool_results)
            grounding_ok = all(c in context.sources for c in llm_response.citations)

            if grounding_ok:
                grounding_hits += 1
            if tool_minimality == len(item["tools"]):
                tool_minimality_hits += 1

            passed = intent_ok and tool_ok and grounding_ok and elapsed_ms <= LATENCY_BUDGET_MS
            if passed:
                pass_count += 1

            results.append(
                {
                    "utterance": item["utterance"],
                    "pass": passed,
                    "intent_ok": intent_ok,
                    "tool_ok": tool_ok,
                    "grounding_ok": grounding_ok,
                    "latency_ms": elapsed_ms,
                }
            )

        return {
            "pass_rate": round(pass_count / max(1, total), 2),
            "tool_minimality_score": round(tool_minimality_hits / max(1, total), 2),
            "grounding_rate": round(grounding_hits / max(1, total), 2),
            "results": results,
        }
