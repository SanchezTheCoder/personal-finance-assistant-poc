from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .context_builder import build_context, build_symbol_performance_context, build_portfolio_ranking_context
from .eval import EvalRunner
from .responder import ModelConfig, generate_response, stream_chunks, clean_answer_markdown, strip_markdown
from .teaching import generate_teaching_explanation
from .formatter import format_answer, build_context_summary
from .router import route_intent
from .torch_router import torch_router_status
from .schemas import ChatRequest, ChatResponse, Intent, LLMResponse
from .tools import (
    ToolRegistry,
    get_activity,
    get_facts,
    get_performance,
    get_positions,
    get_positions_list,
    get_quotes,
    get_transfers,
    get_account_summary,
)
from .tracing import TraceLogger, load_trace

APP_ROOT = Path(__file__).parent
ARTIFACTS_DIR = APP_ROOT / "artifacts" / "model_bundle_v1"
MANIFEST = json.loads((ARTIFACTS_DIR / "manifest.json").read_text())

load_dotenv()

app = FastAPI(title="Personal Finance Assistant POC")
STREAMING_ENABLED = False
registry = ToolRegistry()
session_state: dict[str, dict[str, str]] = {}

frontend_dir = APP_ROOT.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.get("/")
async def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(content={"status": "ok"})


def _model_config() -> ModelConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return ModelConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", MANIFEST["model_version"]),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _clarify(intent: Intent, missing: list[str]) -> str:
    if intent is Intent.positions:
        return "Which symbol should I look up (e.g., AAPL, MSFT)?"
    if intent is Intent.positions_list:
        return "Do you want all positions, or only stocks/ETFs?"
    if intent is Intent.symbol_performance:
        return "Which symbol’s performance should I compute (e.g., AAPL, VOO)?"
    if intent is Intent.performance:
        return "Which timeframe: YTD, 1Y, or all time?"
    if intent is Intent.quotes:
        return "Which symbol should I quote (e.g., AAPL, MSFT)?"
    if intent is Intent.transfers:
        return "Which account should I check for transfers (e.g., Brokerage)?"
    if intent is Intent.account_value:
        return "Which account’s value should I show (e.g., Brokerage)?"
    if intent is Intent.cash_balance:
        return "Which account (e.g., Brokerage) and which cash type (settled or total)?"
    if "account" in missing and "cash_type" in missing:
        return "Which account (e.g., Brokerage) and which cash type (settled or total)?"
    if "account" in missing:
        return "Which account should I check (e.g., Brokerage)?"
    if "cash_type" in missing:
        return "Do you want settled cash or total cash?"
    if "timeframe" in missing:
        return "Which timeframe should I use (e.g., YTD, 1Y)?"
    if "symbol" in missing:
        return "Which symbol should I look up (e.g., AAPL, MSFT)?"
    if "intent" in missing:
        return "What would you like to know: activity, positions, performance, quotes, transfers, or facts?"


def _merge_account_params(extracted: dict[str, str], request_account: Optional[str]) -> dict[str, str]:
    merged = dict(extracted)
    if "account" not in merged and request_account:
        merged["account"] = request_account
    return merged


def _policy_gate(intent: Intent, params: dict[str, str]) -> dict[str, object]:
    allowed_tools = {
        "activity": ["activity"],
        "positions": ["positions"],
        "positions_list": ["positions_list"],
        "portfolio_ranking": ["positions_list", "quotes"],
        "symbol_performance": ["positions", "quotes"],
        "performance": ["performance", "transfers"],
        "quotes": ["quotes"],
        "facts": ["facts"],
        "transfers": ["transfers"],
        "account_value": ["account_summary"],
        "cash_balance": ["account_summary"],
        "clarify": [],
    }
    return {
        "allowed": True,
        "reason": "ok",
        "intent": intent.value,
        "params": params,
        "allowed_tools": allowed_tools.get(intent.value, []),
    }


def _get_session_state(session_id: Optional[str]) -> dict[str, str]:
    if not session_id:
        return {}
    return session_state.get(session_id, {})


def _set_session_state(session_id: Optional[str], state: dict[str, str]) -> None:
    if not session_id:
        return
    session_state[session_id] = state


def _build_chat_response(llm: LLMResponse, trace_id: str) -> ChatResponse:
    return ChatResponse(
        answer_markdown=llm.answer_markdown,
        citations=llm.citations,
        confidence=llm.confidence,
        needs_clarification=llm.needs_clarification,
        clarifying_question=llm.clarifying_question,
        trace_id=trace_id,
    )


def _has_quote_for_symbol(quotes_result, symbol: str) -> bool:
    return any(q.get("symbol") == symbol for q in quotes_result.data.get("quotes", []))


def _available_quote_symbols(quotes_result) -> list[str]:
    return [q.get("symbol") for q in quotes_result.data.get("quotes", []) if q.get("symbol")]


def _quote_missing_response(symbol: str, quotes_result) -> tuple[LLMResponse, str]:
    available = _available_quote_symbols(quotes_result)
    message = f"I don't have a quote for {symbol} in this demo dataset (as of {quotes_result.as_of})."
    if available:
        message += f" Available symbols: {', '.join(available)}."
        clarifying = f"Try one of: {', '.join(available)}."
    else:
        clarifying = f"Can you provide a quotes source for {symbol}?"
    llm = LLMResponse(
        answer_markdown=message,
        citations=[quotes_result.source_id],
        confidence=0.0,
        needs_clarification=True,
        clarifying_question=clarifying,
    )
    return llm, clarifying


@app.post("/chat")
async def chat(request: ChatRequest):
    trace = TraceLogger()
    trace_id = trace.start(request.utterance)

    overall_start = time.perf_counter()
    routing_start = time.perf_counter()
    route = route_intent(request.utterance, use_llm=True)
    routing_ms = int((time.perf_counter() - routing_start) * 1000)
    tool_latency = 0
    llm_latency = 0
    postprocess_latency = 0

    if route.intent is Intent.clarify:
        prior = _get_session_state(request.session_id)
        if prior.get("intent") in {"positions_list", "positions"}:
            route.intent = Intent.positions_list
            route.missing_params = []
            extracted = {}
            if "stocks" in request.utterance.lower():
                extracted["asset_class"] = "stocks"
            elif "etf" in request.utterance.lower():
                extracted["asset_class"] = "etf"
            elif "fund" in request.utterance.lower():
                extracted["asset_class"] = "funds"
            route.extracted = extracted
        if prior.get("intent") in {"account_value", "cash_balance"}:
            route.intent = Intent(prior["intent"])
            route.missing_params = []
            extracted = {}
            if "brokerage" in request.utterance.lower():
                extracted["account"] = "Brokerage"
            if "settled" in request.utterance.lower():
                extracted["cash_type"] = "settled"
            elif "total" in request.utterance.lower():
                extracted["cash_type"] = "total"
            route.extracted = extracted
    trace.log_intent(route.intent.value)
    trace.log_routing(route.routing_mode, route.confidence, route.candidates)
    trace.log_routing_detail(route.extracted, route.missing_params)
    trace.log_router_diagnostics(route.routing_meta or torch_router_status())
    trace.log_policy_gate(_policy_gate(route.intent, route.extracted))

    if route.intent is Intent.clarify:
        candidate = route.extracted.get("candidate_intent")
        try:
            intent_hint = Intent(candidate) if candidate else Intent.clarify
        except ValueError:
            intent_hint = Intent.clarify
        clarification = _clarify(intent_hint, route.missing_params)
        llm = LLMResponse(
            answer_markdown=clarification,
            citations=[],
            confidence=0.0,
            needs_clarification=True,
            clarifying_question=clarification,
        )
        trace.log_clarification(clarification)
        trace.log_grounding([], [])
        trace.log_grounding_validation(False)
        trace.log_formatter_used("none")
        total_ms = int((time.perf_counter() - overall_start) * 1000)
        trace.log_latency_breakdown(
            {
                "routing_ms": routing_ms,
                "tool_ms": 0,
                "llm_ms": 0,
                "postprocess_ms": 0,
                "total_ms": total_ms,
            }
        )
        trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
        return _build_chat_response(llm, trace_id)

    # tool call
    tool_start = time.perf_counter()
    try:
        if route.intent is Intent.symbol_performance:
            params = _merge_account_params(route.extracted, request.account)
            trace.log_tool_params("positions", params)
            positions_result = registry.call_tool("positions", **params)
            trace.log_tool_params("quotes", {"symbol": params.get("symbol")})
            quotes_result = registry.call_tool("quotes", **params)
            tool_result = positions_result
        elif route.intent is Intent.portfolio_ranking:
            params = _merge_account_params(route.extracted, request.account)
            trace.log_tool_params("positions_list", params)
            positions_result = registry.call_tool("positions_list", **params)
            first_symbol = (
                positions_result.data["positions"][0]["symbol"]
                if positions_result.data.get("positions")
                else None
            )
            if not first_symbol:
                raise ValueError("No positions available for ranking")
            trace.log_tool_params("quotes", {"symbol": first_symbol})
            quotes_result = registry.call_tool("quotes", symbol=first_symbol)
            tool_result = positions_result
        elif route.intent is Intent.performance:
            params = _merge_account_params(route.extracted, request.account)
            trace.log_tool_params("performance", params)
            performance_result = registry.call_tool("performance", **params)
            trace.log_tool_params("transfers", {"account": params.get("account")})
            transfers_result = registry.call_tool("transfers", **params)
            tool_result = performance_result
        else:
            params = _merge_account_params(route.extracted, request.account)
            trace.log_tool_params(route.intent.value, params)
            tool_result = registry.call_tool(route.intent.value, **params)
    except Exception as exc:
        llm = LLMResponse(
            answer_markdown="I could not retrieve the data needed to answer.",
            citations=[],
            confidence=0.0,
            needs_clarification=True,
            clarifying_question="Can you double-check the request or try a different symbol?",
        )
        trace.log_clarification("Can you double-check the request or try a different symbol?")
        trace.log_grounding([], [])
        trace.log_grounding_validation(False)
        trace.log_formatter_used("none")
        total_ms = int((time.perf_counter() - overall_start) * 1000)
        trace.log_latency_breakdown(
            {
                "routing_ms": routing_ms,
                "tool_ms": 0,
                "llm_ms": 0,
                "postprocess_ms": 0,
                "total_ms": total_ms,
            }
        )
        trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
        return _build_chat_response(llm, trace_id)

    tool_latency = int((time.perf_counter() - tool_start) * 1000)
    if route.intent is Intent.symbol_performance:
        trace.log_tool("positions", tool_latency, positions_result.source_id)
        trace.log_tool("quotes", tool_latency, quotes_result.source_id)
    elif route.intent is Intent.portfolio_ranking:
        trace.log_tool("positions_list", tool_latency, positions_result.source_id)
        trace.log_tool("quotes", tool_latency, quotes_result.source_id)
    elif route.intent is Intent.performance:
        trace.log_tool("performance", tool_latency, performance_result.source_id)
        trace.log_tool("transfers", tool_latency, transfers_result.source_id)
    else:
        trace.log_tool(route.intent.value, tool_latency, tool_result.source_id)

    if route.intent is Intent.symbol_performance:
        symbol = route.extracted.get("symbol")
        if symbol and not _has_quote_for_symbol(quotes_result, symbol):
            llm_response, clarification = _quote_missing_response(symbol, quotes_result)
            trace.log_context_summary(
                {"symbol": symbol, "quote_available": False, "available_symbols": _available_quote_symbols(quotes_result)}
            )
            trace.log_clarification(clarification)
            trace.log_formatter_used("none")
            trace.log_tokens(0, 0, 0.0)
            trace.log_prompt_info("response_system.txt")
            trace.log_retry_count(0)
            trace.log_grounding(llm_response.citations, llm_response.citations)
            trace.log_grounding_validation(True)
            total_ms = int((time.perf_counter() - overall_start) * 1000)
            trace.log_latency_breakdown(
                {
                    "routing_ms": routing_ms,
                    "tool_ms": tool_latency,
                    "llm_ms": 0,
                    "postprocess_ms": 0,
                    "total_ms": total_ms,
                }
            )
            trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
            return _build_chat_response(llm_response, trace_id)

    if route.intent is Intent.quotes:
        symbol = route.extracted.get("symbol")
        if symbol and not _has_quote_for_symbol(tool_result, symbol):
            llm_response, clarification = _quote_missing_response(symbol, tool_result)
            trace.log_context_summary(
                {"symbol": symbol, "quote_available": False, "available_symbols": _available_quote_symbols(tool_result)}
            )
            trace.log_clarification(clarification)
            trace.log_formatter_used("none")
            trace.log_tokens(0, 0, 0.0)
            trace.log_prompt_info("response_system.txt")
            trace.log_retry_count(0)
            trace.log_grounding(llm_response.citations, llm_response.citations)
            trace.log_grounding_validation(True)
            total_ms = int((time.perf_counter() - overall_start) * 1000)
            trace.log_latency_breakdown(
                {
                    "routing_ms": routing_ms,
                    "tool_ms": tool_latency,
                    "llm_ms": 0,
                    "postprocess_ms": 0,
                    "total_ms": total_ms,
                }
            )
            trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
            return _build_chat_response(llm_response, trace_id)

    try:
        if route.intent is Intent.symbol_performance:
            try:
                context = build_symbol_performance_context(
                    positions_result, quotes_result, route.extracted["symbol"]
                )
            except Exception:
                symbol = route.extracted.get("symbol")
                held = False
                if symbol:
                    held = any(p["symbol"] == symbol for p in positions_result.data.get("positions", []))
                if symbol and not held:
                    quote_fallback = registry.call_tool("quotes", symbol=symbol)
                    context = build_context(
                        Intent.quotes,
                        quote_fallback,
                        {"symbol": symbol, "position_held": False},
                    )
                    context.sources = [quote_fallback.source_id, positions_result.source_id]
                else:
                    context = build_context(Intent.positions, positions_result, route.extracted)
        elif route.intent is Intent.portfolio_ranking:
            context = build_portfolio_ranking_context(
                positions_result,
                quotes_result,
                direction=route.extracted.get("direction", "best"),
                basis=route.extracted.get("basis", "unrealized_pl"),
            )
        elif route.intent is Intent.performance:
            context = build_context(
                Intent.performance,
                performance_result,
                {**route.extracted, "transfers": transfers_result.data.get("transfers", [])},
            )
            context.sources = [performance_result.source_id, transfers_result.source_id]
        else:
            context = build_context(route.intent, tool_result, route.extracted)
        trace.log_context_summary(build_context_summary(context))
    except Exception:
        llm = LLMResponse(
            answer_markdown="I do not have enough data to answer that.",
            citations=[],
            confidence=0.0,
            needs_clarification=True,
            clarifying_question="Can you clarify the symbol or account?",
        )
        trace.log_clarification("Can you clarify the symbol or account?")
        trace.log_grounding([], [])
        trace.log_grounding_validation(False)
        trace.log_formatter_used("none")
        total_ms = int((time.perf_counter() - overall_start) * 1000)
        trace.log_latency_breakdown(
            {
                "routing_ms": routing_ms,
                "tool_ms": tool_latency,
                "llm_ms": 0,
                "postprocess_ms": 0,
                "total_ms": total_ms,
            }
        )
        trace.finalize(MANIFEST["model_version"], "model_bundle_v1", MANIFEST["prompt_version"])
        return _build_chat_response(llm, trace_id)

    model_config = _model_config()
    try:
        llm_start = time.perf_counter()
        llm_response, meta = generate_response(request.utterance, context, model_config)
        llm_latency = int((time.perf_counter() - llm_start) * 1000)
        postprocess_start = time.perf_counter()
        formatter_used = "none"
        if not llm_response.needs_clarification:
            formatted = format_answer(context)
            if formatted:
                llm_response.answer_markdown = formatted
                formatter_used = context.intent.value
                llm_response.needs_clarification = False
                llm_response.clarifying_question = None
        else:
            formatted = format_answer(context)
            if formatted:
                llm_response.answer_markdown = formatted
                formatter_used = context.intent.value
                llm_response.needs_clarification = False
                llm_response.clarifying_question = None
        llm_response.answer_markdown = strip_markdown(clean_answer_markdown(llm_response.answer_markdown))
        postprocess_latency = int((time.perf_counter() - postprocess_start) * 1000)
        trace.log_formatter_used(formatter_used)
    except Exception:
        llm_response = LLMResponse(
            answer_markdown="I could not generate a grounded response at the moment.",
            citations=[],
            confidence=0.0,
            needs_clarification=True,
            clarifying_question="Please try again or rephrase the question.",
        )
        meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "retry_count": 0,
            "prompt_name": "response_system.txt",
        }
        trace.log_formatter_used("none")

    trace.log_tokens(
        meta.get("prompt_tokens", 0),
        meta.get("completion_tokens", 0),
        meta.get("estimated_cost_usd", 0.0),
    )
    trace.log_prompt_info(meta.get("prompt_name", "response_system.txt"))
    trace.log_retry_count(meta.get("retry_count", 0))
    trace.log_grounding(llm_response.citations, context.sources)
    grounding_valid = (not llm_response.needs_clarification) and set(llm_response.citations) == set(context.sources)
    trace.log_grounding_validation(grounding_valid)
    total_ms = int((time.perf_counter() - overall_start) * 1000)
    trace.log_latency_breakdown(
        {
            "routing_ms": routing_ms,
            "tool_ms": tool_latency,
            "llm_ms": llm_latency,
            "postprocess_ms": postprocess_latency,
            "total_ms": total_ms,
        }
    )
    trace.finalize(model_config.model, "model_bundle_v1", MANIFEST["prompt_version"])

    if request.session_id and route.intent in {
        Intent.positions_list,
        Intent.positions,
        Intent.account_value,
        Intent.cash_balance,
    }:
        _set_session_state(
            request.session_id,
            {
                "intent": route.intent.value,
                "asset_class": route.extracted.get("asset_class", ""),
                "account": route.extracted.get("account", ""),
                "cash_type": route.extracted.get("cash_type", ""),
            },
        )
    if request.session_id:
        trace.log_session_state(_get_session_state(request.session_id))

    if STREAMING_ENABLED and request.stream:
        def event_stream() -> Generator[str, None, None]:
            for chunk in stream_chunks(llm_response.answer_markdown):
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"event: delta\ndata: {safe_chunk}\n\n"
            payload = _build_chat_response(llm_response, trace_id).model_dump()
            yield f"event: final\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return _build_chat_response(llm_response, trace_id)


@app.get("/debug/trace/{trace_id}")
async def get_trace(trace_id: str):
    data = load_trace(trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trace not found")
    return JSONResponse(content=data)


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


@app.post("/eval")
async def run_eval():
    runner = EvalRunner(_model_config())
    return JSONResponse(content=runner.run())


@app.get("/api/activity")
async def api_activity(account: Optional[str] = None):
    return get_activity(account=account).model_dump()


@app.get("/api/positions")
async def api_positions(symbol: str, account: Optional[str] = None):
    return get_positions(symbol=symbol, account=account).model_dump()


@app.get("/api/positions_list")
async def api_positions_list(asset_class: Optional[str] = None, account: Optional[str] = None):
    return get_positions_list(asset_class=asset_class, account=account).model_dump()


@app.get("/api/performance")
async def api_performance(timeframe: str, account: Optional[str] = None):
    return get_performance(timeframe=timeframe, account=account).model_dump()


@app.get("/api/quotes")
async def api_quotes(symbol: str):
    return get_quotes(symbol=symbol).model_dump()


@app.get("/api/facts")
async def api_facts(topic: str):
    return get_facts(topic=topic).model_dump()


@app.get("/api/transfers")
async def api_transfers(account: Optional[str] = None):
    return get_transfers(account=account).model_dump()


@app.get("/api/account_summary")
async def api_account_summary(account: Optional[str] = None):
    return get_account_summary(account=account).model_dump()
